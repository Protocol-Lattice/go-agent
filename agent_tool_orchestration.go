package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"strconv"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/alpkeskin/gotoon"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

const (
	defaultToolLoopMaxSteps        = 12
	defaultToolObservationMaxBytes = 4000
)

type ToolChoice struct {
	UseTool     bool           `json:"use_tool"`
	ToolName    string         `json:"tool_name"`
	Arguments   map[string]any `json:"arguments"`
	Reason      string         `json:"reason"`
	Answer      string         `json:"answer"`
	FinalAnswer string         `json:"final_answer"`
}

func configuredToolLoopMaxSteps() int {
	raw := strings.TrimSpace(os.Getenv("utcp_tool_loop_max_steps"))
	if raw == "" {
		return defaultToolLoopMaxSteps
	}
	steps, err := strconv.Atoi(raw)
	if err != nil || steps <= 0 {
		return defaultToolLoopMaxSteps
	}
	return steps
}

func toolChoiceFinalAnswer(tc ToolChoice) string {
	if final := strings.TrimSpace(tc.FinalAnswer); final != "" {
		return final
	}
	return strings.TrimSpace(tc.Answer)
}

func toolSpecExists(specs []tools.Tool, name string) bool {
	for _, spec := range specs {
		if spec.Name == name {
			return true
		}
	}
	return false
}

func appendCodeModeToolSpec(specs []tools.Tool) []tools.Tool {
	if toolSpecExists(specs, "codemode.run_code") {
		return specs
	}
	return append(specs, tools.Tool{
		Name:        "codemode.run_code",
		Description: "Execute Go code with access to UTCP tools via CallTool() and CallToolStream().",
		Inputs: tools.ToolInputOutputSchema{
			Type: "object",
			Properties: map[string]any{
				"code": map[string]any{
					"type":        "string",
					"description": "Go code statements to execute.",
				},
				"timeout": map[string]any{
					"type":        "integer",
					"description": "Timeout in milliseconds.",
				},
			},
			Required: []string{"code"},
		},
	})
}

func formatToolObservation(step int, toolName string, args map[string]any, result any) string {
	return fmt.Sprintf(
		"[step %d] tool=%s args=%s\nresult=%s",
		step,
		toolName,
		compactJSON(args),
		truncate(fmt.Sprint(result), defaultToolObservationMaxBytes),
	)
}

func compactJSON(v any) string {
	if v == nil {
		return "{}"
	}
	b, err := json.Marshal(v)
	if err != nil {
		return fmt.Sprint(v)
	}
	return string(b)
}

func lastToolObservation(observations []string) string {
	if len(observations) == 0 {
		return ""
	}
	return observations[len(observations)-1]
}

func (a *Agent) toolOrchestrator(
	ctx context.Context,
	sessionID string,
	userInput string,
	records []memory.MemoryRecord,
	files ...models.File,
) (bool, string, error) {
	lowerInput := strings.ToLower(strings.TrimSpace(userInput))
	if !a.likelyNeedsToolCall(lowerInput) {
		return false, "", nil
	}

	toolList := a.ToolSpecs()
	if a.CodeMode != nil {
		toolList = appendCodeModeToolSpec(toolList)
	}
	if len(toolList) == 0 {
		return false, "", nil
	}
	if native, ok := a.model.(models.ToolCallingAgent); ok && len(files) == 0 {
		handled, output, err := a.toolOrchestratorNative(ctx, sessionID, userInput, records, toolList, native)
		if !errors.Is(err, models.ErrToolCallingUnsupported) {
			return handled, output, err
		}
	}

	toolDesc := a.cachedToolPrompt(toolList)
	memoryDesc := a.renderMemory(records)
	fileDesc := a.buildAttachmentPrompt("Files available for this turn", files)
	workspaceRules := fileBackedWorkspaceRules(files)
	maxSteps := configuredToolLoopMaxSteps()

	var (
		observations      []string
		lastToolCallKey   string
		lastToolCallValue string
	)
	for step := 1; step <= maxSteps; step++ {
		choicePrompt := fmt.Sprintf(`
You are an agentic UTCP tool execution loop.

USER REQUEST:
%q

CONVERSATION MEMORY:
%s

FILES:
%s

WORKSPACE FILE SELECTION:
%s

AVAILABLE UTCP TOOLS:
%s

PREVIOUS TOOL OBSERVATIONS:
%s

OBJECTIVE:
Continue working until the user request is complete.

RULES:
1. If another tool is needed, set "use_tool": true.
2. If the task is complete, set "use_tool": false and provide "final_answer".
3. Use only exact tool names from AVAILABLE UTCP TOOLS.
4. Do not stop after listing files when the user asked to create, modify, refactor, test, build, or add a feature.
5. For project refactors, inspect relevant files before writing.
6. Use filesystem.write for file changes.
7. Use shell.run only for safe validation commands like gofmt, go test, or go build.
8. For CodeMode, use codemode.run_code only when CodeMode is clearly the best tool.
9. Return ONLY JSON.
10. For file-backed requests, do not create or edit paths that appear only as illustrative examples; prefer attached existing paths.

JSON shape:
{
  "use_tool": true|false,
  "tool_name": "provider.tool or empty",
  "arguments": {},
  "final_answer": "summary when done",
  "reason": "short reason"
}
`,
			userInput,
			memoryDesc,
			fileDesc,
			workspaceRules,
			toolDesc,
			strings.Join(observations, "\n\n"),
		)

		var (
			raw any
			err error
		)
		if len(files) > 0 {
			raw, err = a.model.GenerateWithFiles(ctx, choicePrompt, files)
		} else {
			raw, err = a.model.Generate(ctx, choicePrompt)
		}
		if err != nil {
			return false, "", err
		}

		jsonStr := extractJSON(fmt.Sprint(raw))
		if jsonStr == "" {
			if len(observations) == 0 {
				return false, "", nil
			}
			final := fmt.Sprintf("Stopped because the tool planner did not return valid JSON after %d tool step(s). Last observation:\n%s", len(observations), lastToolObservation(observations))
			a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "tool_loop"})
			return true, final, nil
		}

		var tc ToolChoice
		if err := json.Unmarshal([]byte(jsonStr), &tc); err != nil {
			if len(observations) == 0 {
				return false, "", nil
			}
			final := fmt.Sprintf("Stopped because the tool planner returned invalid JSON after %d tool step(s). Last observation:\n%s", len(observations), lastToolObservation(observations))
			a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "tool_loop"})
			return true, final, nil
		}

		if !tc.UseTool {
			final := toolChoiceFinalAnswer(tc)
			if final == "" {
				if len(observations) == 0 {
					return false, "", nil
				}
				final = fmt.Sprintf("Done. Last observation:\n%s", lastToolObservation(observations))
			}
			a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "tool_loop"})
			return true, final, nil
		}

		toolName := strings.TrimSpace(tc.ToolName)
		if toolName == "" {
			return true, "", fmt.Errorf("tool loop selected empty tool name")
		}
		if !toolSpecExists(toolList, toolName) {
			return true, "", fmt.Errorf("UTCP tool unknown: %s", toolName)
		}
		if tc.Arguments == nil {
			tc.Arguments = map[string]any{}
		}

		// A planner that repeats the exact same tool request has not learned
		// anything from the previous observation. Stop before replaying a
		// potentially non-idempotent side effect.
		toolCallKey := toolName + "\x00" + compactJSON(tc.Arguments)
		if toolCallKey == lastToolCallKey {
			return true, lastToolCallValue, nil
		}

		result, err := a.executeTool(ctx, sessionID, toolName, tc.Arguments)
		if err != nil {
			a.storeMemory(sessionID, "assistant",
				fmt.Sprintf("tool %s error: %v", toolName, err),
				map[string]string{
					"tool":   toolName,
					"source": "tool_loop",
				},
			)
			return true, "", err
		}

		rawOut := fmt.Sprint(result)
		lastToolCallKey = toolCallKey
		lastToolCallValue = rawOut
		observations = append(observations, formatToolObservation(step, toolName, tc.Arguments, rawOut))

		toonBytes, _ := gotoon.Encode(rawOut)
		a.storeMemory(sessionID, "assistant",
			fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes)),
			map[string]string{
				"tool":   toolName,
				"source": "tool_loop",
			},
		)
	}

	final := fmt.Sprintf(
		"Stopped after %d tool step(s) before the planner reported completion. Last observation:\n%s",
		maxSteps,
		lastToolObservation(observations),
	)
	a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "tool_loop"})
	return true, final, nil
}

func (a *Agent) toolOrchestratorNative(
	ctx context.Context,
	sessionID string,
	userInput string,
	records []memory.MemoryRecord,
	toolList []tools.Tool,
	native models.ToolCallingAgent,
) (bool, string, error) {
	definitions := nativeToolDefinitions(toolList)
	if len(definitions) == 0 {
		return false, "", nil
	}

	memoryDesc := a.renderMemory(records)
	maxSteps := configuredToolLoopMaxSteps()
	var (
		observations      []string
		lastToolCallKey   string
		lastToolCallValue string
	)

	for step := 1; step <= maxSteps; step++ {
		prompt := fmt.Sprintf(`
You are an agentic tool execution loop using native tool calls.

USER REQUEST:
%q

CONVERSATION MEMORY:
%s

PREVIOUS TOOL OBSERVATIONS:
%s

OBJECTIVE:
Continue until the user request is complete. Call a tool when needed; when no
more tools are needed, answer the user directly. Use only the provided tools.
`, userInput, memoryDesc, strings.Join(observations, "\n\n"))

		response, err := native.GenerateWithTools(ctx, prompt, definitions)
		if err != nil {
			return false, "", err
		}
		if len(response.ToolCalls) == 0 {
			final := strings.TrimSpace(response.Content)
			if final == "" {
				if len(observations) == 0 {
					return false, "", nil
				}
				final = fmt.Sprintf("Done. Last observation:\n%s", lastToolObservation(observations))
			}
			a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "native_tool_loop"})
			return true, final, nil
		}

		for _, call := range response.ToolCalls {
			toolName := strings.TrimSpace(call.Name)
			if toolName == "" {
				return true, "", fmt.Errorf("native tool loop selected empty tool name")
			}
			if !toolSpecExists(toolList, toolName) {
				return true, "", fmt.Errorf("native tool unknown: %s", toolName)
			}
			if call.Arguments == nil {
				call.Arguments = map[string]any{}
			}

			toolCallKey := toolName + "\x00" + compactJSON(call.Arguments)
			if toolCallKey == lastToolCallKey {
				return true, lastToolCallValue, nil
			}

			result, err := a.executeTool(ctx, sessionID, toolName, call.Arguments)
			if err != nil {
				a.storeMemory(sessionID, "assistant",
					fmt.Sprintf("tool %s error: %v", toolName, err),
					map[string]string{"tool": toolName, "source": "native_tool_loop"},
				)
				return true, "", err
			}

			rawOut := fmt.Sprint(result)
			lastToolCallKey = toolCallKey
			lastToolCallValue = rawOut
			observations = append(observations, formatToolObservation(step, toolName, call.Arguments, rawOut))
			toonBytes, _ := gotoon.Encode(rawOut)
			a.storeMemory(sessionID, "assistant",
				fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes)),
				map[string]string{"tool": toolName, "source": "native_tool_loop"},
			)
		}
	}

	final := fmt.Sprintf(
		"Stopped after %d native tool step(s) before the model reported completion. Last observation:\n%s",
		maxSteps,
		lastToolObservation(observations),
	)
	a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "native_tool_loop"})
	return true, final, nil
}

func nativeToolDefinitions(specs []tools.Tool) []models.ToolDefinition {
	definitions := make([]models.ToolDefinition, 0, len(specs))
	seen := make(map[string]struct{}, len(specs))
	for _, spec := range specs {
		name := strings.TrimSpace(spec.Name)
		if name == "" {
			continue
		}
		key := strings.ToLower(name)
		if _, ok := seen[key]; ok {
			continue
		}
		seen[key] = struct{}{}

		schema := map[string]any{}
		if encoded, err := json.Marshal(spec.Inputs); err == nil {
			_ = json.Unmarshal(encoded, &schema)
		}
		if schemaType, _ := schema["type"].(string); strings.TrimSpace(schemaType) == "" {
			schema["type"] = "object"
		}
		definitions = append(definitions, models.ToolDefinition{
			Name:        name,
			Description: spec.Description,
			InputSchema: schema,
		})
	}
	return definitions
}

// extractJSON attempts to extract valid JSON from a response that may contain
// markdown code fences, extra text, or concatenated content.
func extractJSON(response string) string {
	response = strings.TrimSpace(response)

	// Strip a markdown code fence before looking for the first JSON value.
	// ```json\n{...}\n```
	if strings.Contains(response, "```") {
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimPrefix(response, "```")
		response = strings.TrimSpace(response)

		if idx := strings.Index(response, "```"); idx != -1 {
			response = response[:idx]
		}
		response = strings.TrimSpace(response)
	}

	// Decode from each opening brace. json.Decoder stops after the first
	// complete value, so planner JSON followed by arbitrary prompt text (even
	// more JSON-shaped text) is handled without a hand-rolled brace scanner.
	for start := strings.IndexByte(response, '{'); start >= 0; {
		decoder := json.NewDecoder(strings.NewReader(response[start:]))
		var value json.RawMessage
		if err := decoder.Decode(&value); err == nil && len(value) > 0 {
			return string(value)
		}

		next := strings.IndexByte(response[start+1:], '{')
		if next < 0 {
			break
		}
		start += next + 1
	}

	return ""
}
