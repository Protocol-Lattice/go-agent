package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"regexp"
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
	name = strings.TrimSpace(name)
	if name == "" {
		return false
	}
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
		Name: "codemode.run_code",
		Description: `Execute Go code with access to the canonical UTCP tool registry.
IMPORTANT:
- CallTool() and CallToolStream() may ONLY be called with exact tool names from the provided canonical registry.
- Never invent, rename, abbreviate, pluralize, infer, or compose tool names.
- The runtime validates every CallTool/CallToolStream invocation before execution.`,
		Inputs: tools.ToolInputOutputSchema{Type: "object", Properties: map[string]any{
			"code":    map[string]any{"type": "string", "description": "Go code statements to execute using ONLY canonical UTCP tool names."},
			"timeout": map[string]any{"type": "integer", "description": "Timeout in milliseconds."},
		}, Required: []string{"code"}},
	})
}

func formatToolObservation(step int, toolName string, args map[string]any, result any) string {
	return fmt.Sprintf("[step %d] tool=%s args=%s\nresult=%s", step, toolName, compactJSON(args), truncate(fmt.Sprint(result), defaultToolObservationMaxBytes))
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

func requestRequiresMutation(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	if idx := strings.Index(lower, "user instruction:\n"); idx >= 0 {
		lower = strings.TrimSpace(lower[idx+len("user instruction:\n"):])
	}
	for _, marker := range []string{"for example", "e.g.", "e.g.,"} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			lower = strings.TrimSpace(lower[:idx])
			break
		}
	}
	return regexp.MustCompile(`(?i)\b(refactor|rewrite|modify|edit|update|change|fix|write|create|add|remove|delete|rename|move|implement|patch|replace)\b`).MatchString(lower)
}

func toolMutates(toolName string) bool {
	name := strings.ToLower(strings.TrimSpace(toolName))
	for _, word := range []string{"write", "edit", "patch", "delete", "remove", "create", "rename", "move", "apply", "replace"} {
		if strings.Contains(name, word) {
			return true
		}
	}
	return false
}

func toolInspects(toolName string) bool {
	name := strings.ToLower(strings.TrimSpace(toolName))
	for _, word := range []string{"read", "get", "search", "find", "inspect", "stat", "cat"} {
		if strings.Contains(name, word) {
			return true
		}
	}
	return false
}

func mutationToolNames(toolList []tools.Tool) string {
	var b strings.Builder
	for _, spec := range toolList {
		name := strings.TrimSpace(spec.Name)
		if name == "" || name == "codemode.run_code" || !toolMutates(name) {
			continue
		}
		b.WriteString("- ")
		b.WriteString(name)
		b.WriteByte('\n')
	}
	return b.String()
}

var codeModeToolCallRE = regexp.MustCompile(`\bCallTool(?:Stream)?\s*\(\s*"([^"]+)"`)

func codeModeMutates(code string) bool {
	for _, match := range codeModeToolCallRE.FindAllStringSubmatch(code, -1) {
		if len(match) == 2 && toolMutates(match[1]) {
			return true
		}
	}
	return false
}

func toolLoopCompletionAllowed(userInput string, mutationDone bool) bool {
	return !requestRequiresMutation(userInput) || mutationDone
}

func validateCodeModeCode(code string, toolList []tools.Tool) error {
	code = strings.TrimSpace(code)
	if code == "" {
		return errors.New("codemode.run_code received empty code")
	}
	if !strings.Contains(code, "CallTool(") && !strings.Contains(code, "CallTool (") && !strings.Contains(code, "CallToolStream(") && !strings.Contains(code, "CallToolStream (") {
		return nil
	}
	matches := codeModeToolCallRE.FindAllStringSubmatch(code, -1)
	hasCanonicalTools := false
	for _, spec := range toolList {
		name := strings.TrimSpace(spec.Name)
		if name != "" && name != "codemode.run_code" {
			hasCanonicalTools = true
			break
		}
	}
	if !hasCanonicalTools {
		return nil
	}
	for _, match := range matches {
		if len(match) == 2 && !toolSpecExists(toolList, strings.TrimSpace(match[1])) {
			return fmt.Errorf("codemode unknown_tool: %q is not registered in the canonical UTCP tool registry; use an exact registered tool name", strings.TrimSpace(match[1]))
		}
	}
	if len(matches) == 0 {
		return errors.New("codemode invalid_tool_reference: CallTool/CallToolStream requires an exact string-literal tool name")
	}
	return nil
}

func codeModeToolNames(toolList []tools.Tool) string {
	var b strings.Builder
	for _, spec := range toolList {
		name := strings.TrimSpace(spec.Name)
		if name == "" || name == "codemode.run_code" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(name)
		b.WriteByte('\n')
	}
	return b.String()
}

// isCodeModeCompilationError makes generated-code compilation failures retryable
// planner observations instead of fatal agent errors. CodeMode code is produced
// by the model, so a compiler error is recoverable model feedback.
func isCodeModeCompilationError(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "compilation failed") ||
		strings.Contains(s, "compilation error") ||
		strings.Contains(s, "undefined:")
}

func (a *Agent) toolOrchestrator(ctx context.Context, sessionID, userInput string, records []memory.MemoryRecord, files ...models.File) (bool, string, error) {
	if !a.likelyNeedsToolCall(strings.ToLower(strings.TrimSpace(userInput))) {
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
	canonicalToolNames := codeModeToolNames(toolList)
	requiresMutation := requestRequiresMutation(userInput)
	mutationDone := false
	inspectionDone := false
	mutationTools := mutationToolNames(toolList)
	var observations []string
	var lastToolCallKey, lastToolCallValue string

	for step := 1; step <= maxSteps; step++ {
		choicePrompt := fmt.Sprintf(`
You are an agentic UTCP tool execution loop.

SYSTEM INSTRUCTIONS:
%s

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

CANONICAL TOOL NAMES FOR CODEMODE:
%s

PREVIOUS TOOL OBSERVATIONS:
%s

REQUEST MUTATION STATE:
requires_mutation=%t
mutation_done=%t
inspection_done=%t

MUTATION-CAPABLE TOOLS:
%s

OBJECTIVE:
Continue working until the user request is complete.

RULES:
1. Use only exact tool names from AVAILABLE UTCP TOOLS.
2. Do not stop after listing files when the user asked to create, modify, refactor, test, build, or add a feature.
3. Inspect relevant artifacts before mutating them.
4. After a mutation, verify the resulting artifact when practical.
5. For multi-step dependent work, prefer codemode.run_code.
6. CodeMode MUST use exact canonical tool names and exact schema argument names.
7. A refactor/edit/create/fix task is not complete until the mutation has actually executed successfully.
8. If requires_mutation=true and inspection_done=true and mutation_done=false, NEVER call another read/list/search tool. Select a mutation-capable tool or codemode.run_code containing a mutation.
9. CodeMode MUST execute dependent calls sequentially.
10. CodeMode variables MUST be declared in the same lexical scope where they are consumed. Do not declare r2/r3/etc inside an if/for block and consume them outside that block.
11. For CodeMode, prefer one straight-line sequence of CallTool calls with top-level r1/r2/r3 assignments. Avoid unnecessary nested scopes.
12. CodeMode MUST NOT simulate tool execution or invent tool names.
13. If a previous CodeMode observation says compilation failed, generate a fresh complete snippet; do not repeat the invalid snippet.
14. If the request is a mutation, CodeMode should perform inspect -> mutate -> verify in one run when possible.

JSON shape:
{"use_tool":true|false,"tool_name":"provider.tool or empty","arguments":{},"final_answer":"summary when done","reason":"short reason"}
`, a.systemInstructions(), userInput, memoryDesc, fileDesc, workspaceRules, toolDesc, canonicalToolNames, strings.Join(observations, "\n\n"), requiresMutation, mutationDone, inspectionDone, mutationTools)

		var raw any
		var err error
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
			observations = append(observations, fmt.Sprintf("[planner] invalid_json: return ONLY the required JSON object. Previous response: %s", truncate(fmt.Sprint(raw), 2000)))
			continue
		}
		var tc ToolChoice
		if err := json.Unmarshal([]byte(jsonStr), &tc); err != nil {
			observations = append(observations, fmt.Sprintf("[planner] invalid_json: %v. Previous response: %s", err, truncate(fmt.Sprint(raw), 2000)))
			continue
		}

		if !tc.UseTool {
			if !toolLoopCompletionAllowed(userInput, mutationDone) {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=mutation_required; the request requires a real mutation before completion. Select the exact registered mutation tool.", step))
				continue
			}
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
			observations = append(observations, fmt.Sprintf("[step %d] planner_error=unknown_tool requested=%q; choose ONLY an exact registered tool name", step, toolName))
			continue
		}
		if tc.Arguments == nil {
			tc.Arguments = map[string]any{}
		}

		var plannedMutation bool
		if toolName == "codemode.run_code" {
			code, ok := tc.Arguments["code"].(string)
			if !ok {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=codemode_invalid_code", step))
				continue
			}
			if err := validateCodeModeCode(code, toolList); err != nil {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=%v; CodeMode execution was blocked before execution.", step, err))
				continue
			}
			plannedMutation = codeModeMutates(code)
		} else {
			plannedMutation = toolMutates(toolName)
		}

		if requiresMutation && inspectionDone && !mutationDone && !plannedMutation {
			observations = append(observations, fmt.Sprintf("[step %d] planner_error=mutation_required_next tool=%s; target already inspected; choose an exact mutation-capable tool or codemode.run_code containing a mutation: %s", step, toolName, strings.TrimSpace(mutationTools)))
			continue
		}

		toolCallKey := toolName + "\x00" + compactJSON(tc.Arguments)
		if toolCallKey == lastToolCallKey {
			if requiresMutation && !mutationDone {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=duplicate_call tool=%s; repeating a read-only call cannot satisfy this mutation request", step, toolName))
				continue
			}
			if mutationDone && plannedMutation {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=duplicate_mutation_after_success tool=%s; continue with verification or completion", step, toolName))
				continue
			}
			return true, lastToolCallValue, nil
		}

		result, err := a.executeTool(ctx, sessionID, toolName, tc.Arguments)
		if err != nil {
			if toolName == "codemode.run_code" && isCodeModeCompilationError(err) {
				observations = append(observations, fmt.Sprintf("[step %d] codemode_compilation_error: %v. Generate a fresh valid snippet. Keep all rN variables in one lexical scope and do not repeat the previous code.", step, err))
				lastToolCallKey = ""
				continue
			}
			a.storeMemory(sessionID, "assistant", fmt.Sprintf("tool %s error: %v", toolName, err), map[string]string{"tool": toolName, "source": "tool_loop"})
			return true, "", err
		}
		if plannedMutation {
			mutationDone = true
		} else if toolInspects(toolName) {
			inspectionDone = true
		}
		rawOut := fmt.Sprint(result)
		lastToolCallKey, lastToolCallValue = toolCallKey, rawOut
		observations = append(observations, formatToolObservation(step, toolName, tc.Arguments, rawOut))
		toonBytes, _ := gotoon.Encode(rawOut)
		a.storeMemory(sessionID, "assistant", fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes)), map[string]string{"tool": toolName, "source": "tool_loop"})
	}

	final := fmt.Sprintf("Stopped after %d tool step(s) before the planner reported completion. Last observation:\n%s", maxSteps, lastToolObservation(observations))
	a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "tool_loop"})
	return true, final, nil
}

func (a *Agent) toolOrchestratorNative(ctx context.Context, sessionID, userInput string, records []memory.MemoryRecord, toolList []tools.Tool, native models.ToolCallingAgent) (bool, string, error) {
	definitions := nativeToolDefinitions(toolList)
	if len(definitions) == 0 {
		return false, "", nil
	}
	memoryDesc := a.renderMemory(records)
	maxSteps := configuredToolLoopMaxSteps()
	requiresMutation := requestRequiresMutation(userInput)
	mutationDone := false
	inspectionDone := false
	var observations []string
	var lastToolCallKey, lastToolCallValue string

	for step := 1; step <= maxSteps; step++ {
		prompt := fmt.Sprintf(`
You are an agentic tool execution loop using native tool calls.

SYSTEM INSTRUCTIONS:
%s

USER REQUEST:
%q

CONVERSATION MEMORY:
%s

PREVIOUS TOOL OBSERVATIONS:
%s

OBJECTIVE:
Continue until the user request is complete.

For refactor/edit/write/create/change/fix requests, inspect the target first, then execute a real mutation tool before completion. After mutation, verify when practical. If the target has already been inspected and mutation has not happened, do not issue another read-only call. Use only exact native tool names. If a previous CodeMode-related execution failed compilation, make the next tool/code decision fresh rather than repeating it.
`, a.systemInstructions(), userInput, memoryDesc, strings.Join(observations, "\n\n"))
		response, err := native.GenerateWithTools(ctx, prompt, definitions)
		if err != nil {
			return false, "", err
		}
		if len(response.ToolCalls) == 0 {
			if requiresMutation && !mutationDone {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=mutation_required; discovery/read-only execution is insufficient for this request", step))
				continue
			}
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
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=unknown_tool requested=%q", step, toolName))
				continue
			}
			if call.Arguments == nil {
				call.Arguments = map[string]any{}
			}

			var plannedMutation bool
			if toolName == "codemode.run_code" {
				if code, ok := call.Arguments["code"].(string); ok {
					plannedMutation = codeModeMutates(code)
				}
			} else {
				plannedMutation = toolMutates(toolName)
			}
			if requiresMutation && inspectionDone && !mutationDone && !plannedMutation {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=mutation_required_next tool=%s; target already inspected", step, toolName))
				continue
			}

			toolCallKey := toolName + "\x00" + compactJSON(call.Arguments)
			if toolCallKey == lastToolCallKey {
				if requiresMutation && !mutationDone {
					observations = append(observations, fmt.Sprintf("[step %d] planner_error=duplicate_call tool=%s; choose the mutation tool", step, toolName))
					continue
				}
				return true, lastToolCallValue, nil
			}

			result, err := a.executeTool(ctx, sessionID, toolName, call.Arguments)
			if err != nil {
				if toolName == "codemode.run_code" && isCodeModeCompilationError(err) {
					observations = append(observations, fmt.Sprintf("[step %d] codemode_compilation_error: %v. Generate fresh valid code and keep dependent variables in one lexical scope.", step, err))
					lastToolCallKey = ""
					continue
				}
				a.storeMemory(sessionID, "assistant", fmt.Sprintf("tool %s error: %v", toolName, err), map[string]string{"tool": toolName, "source": "native_tool_loop"})
				return true, "", err
			}
			if plannedMutation {
				mutationDone = true
			} else if toolInspects(toolName) {
				inspectionDone = true
			}
			rawOut := fmt.Sprint(result)
			lastToolCallKey, lastToolCallValue = toolCallKey, rawOut
			observations = append(observations, formatToolObservation(step, toolName, call.Arguments, rawOut))
			toonBytes, _ := gotoon.Encode(rawOut)
			a.storeMemory(sessionID, "assistant", fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes)), map[string]string{"tool": toolName, "source": "native_tool_loop"})
		}
	}

	final := fmt.Sprintf("Stopped after %d native tool step(s) before the model reported completion. Last observation:\n%s", maxSteps, lastToolObservation(observations))
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
		definitions = append(definitions, models.ToolDefinition{Name: name, Description: spec.Description, InputSchema: schema})
	}
	return definitions
}

func extractJSON(response string) string {
	response = strings.TrimSpace(response)
	if strings.Contains(response, "```") {
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimPrefix(response, "```")
		response = strings.TrimSpace(response)
		if idx := strings.Index(response, "```"); idx != -1 {
			response = response[:idx]
		}
		response = strings.TrimSpace(response)
	}
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
