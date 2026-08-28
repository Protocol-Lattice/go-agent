package agent

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/alpkeskin/gotoon"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

const (
	defaultToolLoopMaxSteps        = 12
	defaultToolObservationMaxBytes = 4000
)

func orchestratorLoggingEnabled() bool {
	switch strings.ToLower(strings.TrimSpace(os.Getenv("AGENT_ORCHESTRATOR_LOG"))) {
	case "1", "true", "yes", "on":
		return true
	default:
		return false
	}
}

func orchestratorLogf(format string, args ...any) {
	if !orchestratorLoggingEnabled() {
		return
	}
	log.Printf("[orchestrator] "+format, args...)
}

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

	// Text after an illustrative marker is not an instruction.
	// Example:
	// "Inspect the codebase. For example, read foo.go and refactor it."
	// The second sentence describes an example workflow, not necessarily
	// the requested mutation.
	for _, marker := range []string{
		"for example",
		"e.g.",
		"e.g.,",
	} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			lower = strings.TrimSpace(lower[:idx])
			break
		}
	}

	mutationRE := regexp.MustCompile(
		`(?i)\b(refactor|rewrite|modify|edit|update|change|fix|write|create|add|remove|delete|rename|move|implement|patch|replace)\b`,
	)

	return mutationRE.MatchString(lower)
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

var codeModeToolCallRE = regexp.MustCompile(`\bCallTool(?:Stream)?\s*\(\s*"([^"]+)"`)

func validateCodeModeCode(code string, toolList []tools.Tool) error {
	code = strings.TrimSpace(code)
	if code == "" {
		return errors.New("codemode.run_code received empty code")
	}

	if !strings.Contains(code, "CallTool(") &&
		!strings.Contains(code, "CallTool (") &&
		!strings.Contains(code, "CallToolStream(") &&
		!strings.Contains(code, "CallToolStream (") {
		return nil
	}

	// When the agent has no discoverable canonical tools, let the
	// configured UTCP client be the execution authority. This is important
	// for CodeMode-only clients where tools are intentionally not exposed
	// through SearchTools().
	hasCanonicalTools := false
	for _, spec := range toolList {
		if name := strings.TrimSpace(spec.Name); name != "" &&
			name != "codemode.run_code" {
			hasCanonicalTools = true
			break
		}
	}
	if !hasCanonicalTools {
		return nil
	}

	matches := codeModeToolCallRE.FindAllStringSubmatch(code, -1)
	for _, match := range matches {
		if len(match) != 2 {
			continue
		}

		toolName := strings.TrimSpace(match[1])
		if !toolSpecExists(toolList, toolName) {
			return fmt.Errorf(
				"codemode unknown_tool: %q is not registered in the canonical UTCP tool registry; use an exact registered tool name",
				toolName,
			)
		}
	}

	if len(matches) == 0 {
		return errors.New(
			"codemode invalid_tool_reference: CallTool/CallToolStream requires an exact string-literal tool name",
		)
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

type plannedTool struct {
	toolName  string
	arguments map[string]any
}

type toolPlan struct {
	useTool     bool
	toolCalls   []plannedTool
	finalAnswer string
}

func (a *Agent) toolOrchestrator(ctx context.Context, sessionID, userInput string, records []memory.MemoryRecord, files ...models.File) (bool, string, error) {
	startedAt := time.Now()
	orchestratorLogf("request started input_bytes=%d memory_records=%d files=%d", len(userInput), len(records), len(files))

	lowerInput := strings.ToLower(strings.TrimSpace(userInput))
	if !a.likelyNeedsToolCall(lowerInput) {
		orchestratorLogf("request skipped reason=no_tool_intent duration=%s", time.Since(startedAt))
		return false, "", nil
	}

	toolList := a.ToolSpecs()
	if a.CodeMode != nil {
		toolList = appendCodeModeToolSpec(toolList)
	}
	if len(toolList) == 0 {
		orchestratorLogf("request skipped reason=no_registered_tools duration=%s", time.Since(startedAt))
		return false, "", nil
	}
	orchestratorLogf("tools ready count=%d codemode=%t", len(toolList), a.CodeMode != nil)

	if native, ok := a.model.(models.ToolCallingAgent); ok && len(files) == 0 {
		orchestratorLogf("route selected mode=native")
		handled, output, err := a.toolOrchestratorNative(ctx, sessionID, userInput, records, toolList, native)
		if !errors.Is(err, models.ErrToolCallingUnsupported) {
			orchestratorLogf("request finished mode=native handled=%t output_bytes=%d duration=%s err=%v", handled, len(output), time.Since(startedAt), err)
			return handled, output, err
		}
		orchestratorLogf("native unsupported; falling back to json planner")
	}
	orchestratorLogf("route selected mode=json_planner")

	toolDesc := a.cachedToolPrompt(toolList)
	memoryDesc := a.renderMemory(records)
	fileDesc := a.buildAttachmentPrompt("Files available for this turn", files)
	workspaceRules := fileBackedWorkspaceRules(files)
	canonicalToolNames := codeModeToolNames(toolList)

	planPrompt := func(observations []string) string {
		return fmt.Sprintf(`
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

OBJECTIVE:
Continue working until the user request is complete.

RULES:
1. If another tool is needed, set "use_tool": true.
2. If the task is complete, set "use_tool": false and provide "final_answer".
3. Use only exact tool names from AVAILABLE UTCP TOOLS.
4. Do not stop after listing files when the user asked to create, modify, refactor, test, build, or add a feature.
5. For project refactors, inspect relevant files before writing.
6. Use filesystem.write or the provider's exact mutation tool for file changes.
7. After a mutation, verify the resulting artifact when practical.
8. For CodeMode, use codemode.run_code when it can execute the required multi-tool workflow.
9. CodeMode MUST use exact canonical tool names.
10. Never invent, infer, abbreviate, rename, pluralize, or compose a tool name.
11. If the request is a refactor/edit/write/create/fix/change task, DISCOVERY ALONE IS NOT COMPLETION. A real mutation tool MUST execute before completion.
12. Return ONLY JSON.

JSON shape:
{"use_tool":true|false,"tool_name":"provider.tool or empty","arguments":{},"final_answer":"summary when done","reason":"short reason"}
`, a.systemInstructions(), userInput, memoryDesc, fileDesc, workspaceRules, toolDesc, canonicalToolNames, strings.Join(observations, "\n\n"))
	}

	planFromModel := func(ctx context.Context, step int, observations []string) (toolPlan, error) {
		prompt := planPrompt(observations)
		var raw any
		var err error
		modelStartedAt := time.Now()
		orchestratorLogf("planner request mode=json step=%d observations=%d prompt_bytes=%d", step, len(observations), len(prompt))
		if len(files) > 0 {
			raw, err = a.model.GenerateWithFiles(ctx, prompt, files)
		} else {
			raw, err = a.model.Generate(ctx, prompt)
		}
		if err != nil {
			orchestratorLogf("planner failed mode=json step=%d duration=%s err=%v", step, time.Since(modelStartedAt), err)
			return toolPlan{}, err
		}
		rawText := fmt.Sprint(raw)
		orchestratorLogf("planner response mode=json step=%d response_bytes=%d duration=%s", step, len(rawText), time.Since(modelStartedAt))
		jsonText := extractJSON(rawText)
		if jsonText == "" {
			orchestratorLogf("planner invalid response mode=json step=%d reason=no_json", step)
			return toolPlan{}, fmt.Errorf("planner_invalid_json")
		}
		var tc ToolChoice
		if err := json.Unmarshal([]byte(jsonText), &tc); err != nil {
			orchestratorLogf("planner invalid response mode=json step=%d reason=decode_error err=%v", step, err)
			return toolPlan{}, fmt.Errorf("planner_invalid_json: %w", err)
		}
		if tc.Arguments == nil {
			tc.Arguments = map[string]any{}
		}
		if !tc.UseTool {
			return toolPlan{useTool: false, finalAnswer: toolChoiceFinalAnswer(tc)}, nil
		}
		toolName := strings.TrimSpace(tc.ToolName)
		if toolName == "" {
			return toolPlan{}, fmt.Errorf("tool name must not be empty")
		}
		return toolPlan{useTool: true, toolCalls: []plannedTool{{toolName: toolName, arguments: tc.Arguments}}}, nil
	}

	return a.runToolLoop(ctx, sessionID, userInput, toolList, planFromModel, runOptions{
		stepLimit: configuredToolLoopMaxSteps(),
		sourceTag: "tool_loop",
	})
}

type toolPlanFromModel func(context.Context, int, []string) (toolPlan, error)

func (a *Agent) toolOrchestratorNative(ctx context.Context, sessionID, userInput string, records []memory.MemoryRecord, toolList []tools.Tool, native models.ToolCallingAgent) (bool, string, error) {
	definitions := nativeToolDefinitions(toolList)
	if len(definitions) == 0 {
		return false, "", nil
	}
	memoryDesc := a.renderMemory(records)

	planPrompt := func(step int, observations []string) string {
		return fmt.Sprintf(`
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

For refactor/edit/write/create/change/fix requests, discovery is not completion. A real mutation tool must execute before you answer. After mutation, verify the result when practical. Use ONLY exact names supplied by the native tool definitions. Return tool calls until the requested mutation is actually complete.
`, a.systemInstructions(), userInput, memoryDesc, strings.Join(observations, "\n\n"))
	}

	planFromModel := func(ctx context.Context, step int, observations []string) (toolPlan, error) {
		modelStartedAt := time.Now()
		orchestratorLogf("planner request mode=native step=%d observations=%d", step, len(observations))
		response, err := native.GenerateWithTools(ctx, planPrompt(step, observations), definitions)
		if err != nil {
			orchestratorLogf("planner failed mode=native step=%d duration=%s err=%v", step, time.Since(modelStartedAt), err)
			return toolPlan{}, err
		}
		orchestratorLogf("planner response mode=native step=%d tool_calls=%d content_bytes=%d duration=%s", step, len(response.ToolCalls), len(response.Content), time.Since(modelStartedAt))
		if len(response.ToolCalls) == 0 {
			return toolPlan{
				useTool:     false,
				finalAnswer: strings.TrimSpace(response.Content),
			}, nil
		}
		calls := make([]plannedTool, 0, len(response.ToolCalls))
		for _, call := range response.ToolCalls {
			callName := strings.TrimSpace(call.Name)
			if callName == "" {
				return toolPlan{}, fmt.Errorf("native tool call name must not be empty")
			}
			args := call.Arguments
			if args == nil {
				args = map[string]any{}
			}
			calls = append(calls, plannedTool{
				toolName:  callName,
				arguments: args,
			})
		}
		return toolPlan{useTool: true, toolCalls: calls}, nil
	}

	return a.runToolLoop(ctx, sessionID, userInput, toolList, planFromModel, runOptions{
		stepLimit: configuredToolLoopMaxSteps(),
		sourceTag: "native_tool_loop",
	})
}

type runOptions struct {
	stepLimit int
	sourceTag string
}

func (a *Agent) runToolLoop(
	ctx context.Context,
	sessionID string,
	userInput string,
	toolList []tools.Tool,
	plan toolPlanFromModel,
	opts runOptions,
) (bool, string, error) {
	startedAt := time.Now()
	requiresMutation := requestRequiresMutation(userInput)
	mutationDone := false
	var observations []string
	var lastToolCallKey, lastToolCallValue string

	maxSteps := opts.stepLimit
	if maxSteps <= 0 {
		maxSteps = configuredToolLoopMaxSteps()
	}
	orchestratorLogf("loop started source=%s max_steps=%d tools=%d requires_mutation=%t", opts.sourceTag, maxSteps, len(toolList), requiresMutation)

	for step := 1; step <= maxSteps; step++ {
		stepStartedAt := time.Now()
		orchestratorLogf("step started source=%s step=%d observations=%d mutation_done=%t", opts.sourceTag, step, len(observations), mutationDone)
		decision, err := plan(ctx, step, observations)
		if err != nil {
			switch {
			case strings.HasPrefix(err.Error(), "planner_invalid_json"):
				orchestratorLogf("step retry source=%s step=%d reason=planner_invalid_json duration=%s", opts.sourceTag, step, time.Since(stepStartedAt))
				observations = append(observations, fmt.Sprintf("[planner] invalid_json: planner response was not valid JSON: %v", err))
				continue
			default:
				orchestratorLogf("loop failed source=%s step=%d duration=%s err=%v", opts.sourceTag, step, time.Since(startedAt), err)
				return false, "", err
			}
		}
		if !decision.useTool {
			if !toolLoopCompletionAllowed(userInput, mutationDone) {
				orchestratorLogf("step blocked source=%s step=%d reason=mutation_required", opts.sourceTag, step)
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=mutation_required; the request is a refactor/edit/write task and only discovery/read tools have executed. Continue with the exact registered mutation tool before completion.", step))
				continue
			}
			final := strings.TrimSpace(decision.finalAnswer)
			if final == "" {
				if len(observations) == 0 {
					return false, "", nil
				}
				final = fmt.Sprintf("Done. Last observation:\n%s", lastToolObservation(observations))
			}
			a.storeMemory(sessionID, "assistant", final, map[string]string{"source": opts.sourceTag})
			orchestratorLogf("loop completed source=%s step=%d output_bytes=%d duration=%s", opts.sourceTag, step, len(final), time.Since(startedAt))
			return true, final, nil
		}
		orchestratorLogf("step planned source=%s step=%d tool_calls=%d planning_duration=%s", opts.sourceTag, step, len(decision.toolCalls), time.Since(stepStartedAt))

		for _, call := range decision.toolCalls {
			if !toolSpecExists(toolList, call.toolName) {
				orchestratorLogf("tool rejected source=%s step=%d tool=%q reason=unknown_tool", opts.sourceTag, step, call.toolName)
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=unknown_tool requested=%q; choose ONLY an exact name from AVAILABLE UTCP TOOLS.", step, call.toolName))
				continue
			}

			var plannedMutation bool
			if call.toolName == "codemode.run_code" {
				code, ok := call.arguments["code"].(string)
				if !ok {
					orchestratorLogf("tool rejected source=%s step=%d tool=%q reason=invalid_code", opts.sourceTag, step, call.toolName)
					observations = append(observations, fmt.Sprintf("[step %d] planner_error=codemode_invalid_code", step))
					continue
				}
				if err := validateCodeModeCode(code, toolList); err != nil {
					orchestratorLogf("tool rejected source=%s step=%d tool=%q reason=validation_failed err=%v", opts.sourceTag, step, call.toolName, err)
					observations = append(observations, fmt.Sprintf("[step %d] planner_error=%v; CodeMode execution was blocked before execution.", step, err))
					continue
				}
				plannedMutation = codeModeMutates(code)
			} else {
				plannedMutation = toolMutates(call.toolName)
			}

			toolCallKey := call.toolName + "\x00" + compactJSON(call.arguments)
			if toolCallKey == lastToolCallKey {
				if requiresMutation && !mutationDone {
					orchestratorLogf("tool skipped source=%s step=%d tool=%q reason=duplicate_before_mutation", opts.sourceTag, step, call.toolName)
					observations = append(observations, fmt.Sprintf(
						"[step %d] planner_error=duplicate_call tool=%s args=%s; repeating the same read-only call will not satisfy this request. Call the exact registered mutation tool (write/edit/patch) instead of re-reading.",
						step,
						call.toolName,
						compactJSON(call.arguments),
					))
					continue
				}
				if mutationDone && plannedMutation {
					orchestratorLogf("tool skipped source=%s step=%d tool=%q reason=duplicate_mutation", opts.sourceTag, step, call.toolName)
					observations = append(observations, fmt.Sprintf(
						"[step %d] planner_error=duplicate_mutation_after_success tool=%s args=%s; the mutation already succeeded. Continue with verification or report completion.",
						step,
						call.toolName,
						compactJSON(call.arguments),
					))
					continue
				}
				orchestratorLogf("loop completed source=%s step=%d reason=duplicate_read output_bytes=%d duration=%s", opts.sourceTag, step, len(lastToolCallValue), time.Since(startedAt))
				return true, lastToolCallValue, nil
			}

			toolStartedAt := time.Now()
			orchestratorLogf("tool started source=%s step=%d tool=%q argument_fields=%d mutation=%t", opts.sourceTag, step, call.toolName, len(call.arguments), plannedMutation)
			result, err := a.executeTool(ctx, sessionID, call.toolName, call.arguments)
			if err != nil {
				orchestratorLogf("tool failed source=%s step=%d tool=%q duration=%s err=%v", opts.sourceTag, step, call.toolName, time.Since(toolStartedAt), err)
				a.storeMemory(sessionID, "assistant", fmt.Sprintf("tool %s error: %v", call.toolName, err), map[string]string{"tool": call.toolName, "source": opts.sourceTag})
				return true, "", err
			}
			if plannedMutation {
				mutationDone = true
			}
			rawOut := fmt.Sprint(result)
			orchestratorLogf("tool completed source=%s step=%d tool=%q output_bytes=%d duration=%s", opts.sourceTag, step, call.toolName, len(rawOut), time.Since(toolStartedAt))
			lastToolCallKey, lastToolCallValue = toolCallKey, rawOut
			observations = append(observations, formatToolObservation(step, call.toolName, call.arguments, rawOut))
			toonBytes, _ := gotoon.Encode(rawOut)
			a.storeMemory(sessionID, "assistant", fmt.Sprintf("%s\n\n.toon:\n%s", rawOut, string(toonBytes)), map[string]string{"tool": call.toolName, "source": opts.sourceTag})
		}
	}

	final := fmt.Sprintf("Stopped after %d tool step(s) before the planner reported completion. Last observation:\n%s", maxSteps, lastToolObservation(observations))
	a.storeMemory(sessionID, "assistant", final, map[string]string{"source": opts.sourceTag})
	orchestratorLogf("loop stopped source=%s reason=step_limit max_steps=%d output_bytes=%d duration=%s", opts.sourceTag, maxSteps, len(final), time.Since(startedAt))
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
