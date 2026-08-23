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
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

const (
	defaultToolLoopMaxSteps        = 6
	defaultToolObservationMaxBytes = 4000
	defaultPlannerRepairAttempts   = 2
)

type ToolChoice struct {
	UseTool     bool           `json:"use_tool"`
	ToolName    string         `json:"tool_name"`
	Arguments   map[string]any `json:"arguments"`
	Reason      string         `json:"reason"`
	Answer      string         `json:"answer"`
	FinalAnswer string         `json:"final_answer"`
}

type orchestrationState struct {
	requiresMutation bool
	inspected        bool
	mutated          bool
	verified         bool
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
		if strings.TrimSpace(spec.Name) == name {
			return true
		}
	}
	return false
}

func appendCodeModeToolSpec(specs []tools.Tool) []tools.Tool {
	if toolSpecExists(specs, codemode.CodeModeToolName) || toolSpecExists(specs, "codemode.run_code") {
		return specs
	}
	return append(append([]tools.Tool(nil), specs...), tools.Tool{
		Name:        codemode.CodeModeToolName,
		Description: "Execute Go code against the canonical UTCP tool registry. All repository inspection and mutation must happen inside CodeMode.",
		Inputs: tools.ToolInputOutputSchema{
			Type: "object",
			Properties: map[string]any{
				"code":    map[string]any{"type": "string", "description": "Go source to execute. Invoke only exact canonical UTCP tool names through codemode.CallTool or codemode.CallToolStream."},
				"timeout": map[string]any{"type": "integer", "description": "Execution timeout in milliseconds."},
			},
			Required: []string{"code"},
		},
	})
}

func codeModePlannerTools(specs []tools.Tool) []tools.Tool {
	var result []tools.Tool
	for _, spec := range specs {
		if spec.Name == codemode.CodeModeToolName || spec.Name == "codemode.run_code" {
			result = append(result, spec)
		}
	}
	return result
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

var mutationRequestRE = regexp.MustCompile(`(?i)\b(refactor|rewrite|modify|update|edit|change|fix|patch|write|create|delete|remove|rename|move|replace|apply)\b`)

func requestRequiresMutation(input string) bool {
	lower := strings.ToLower(strings.TrimSpace(input))
	for _, marker := range []string{"for example", "e.g.", "e.g.,"} {
		if idx := strings.Index(lower, marker); idx >= 0 {
			lower = strings.TrimSpace(lower[:idx])
			break
		}
	}
	return mutationRequestRE.MatchString(lower)
}

var codeModeToolCallRE = regexp.MustCompile(`\bCallTool(?:Stream)?\s*\(\s*"((?:\\.|[^"\\])*)"`)

func toolMutates(toolName string) bool {
	name := strings.ToLower(strings.TrimSpace(toolName))
	for _, word := range []string{"write", "edit", "patch", "delete", "remove", "create", "rename", "move", "apply", "replace", "update", "insert"} {
		if strings.Contains(name, word) {
			return true
		}
	}
	return false
}

func toolInspects(toolName string) bool {
	name := strings.ToLower(strings.TrimSpace(toolName))
	for _, word := range []string{"read", "list", "search", "find", "stat", "inspect", "describe", "get"} {
		if strings.Contains(name, word) {
			return true
		}
	}
	return false
}

func codeModeMutates(code string) bool {
	for _, match := range codeModeToolCallRE.FindAllStringSubmatch(code, -1) {
		if len(match) == 2 {
			name, err := strconv.Unquote(`"` + match[1] + `"`)
			if err == nil && toolMutates(name) {
				return true
			}
		}
	}
	return false
}

func codeModeInspects(code string) bool {
	for _, match := range codeModeToolCallRE.FindAllStringSubmatch(code, -1) {
		if len(match) == 2 {
			name, err := strconv.Unquote(`"` + match[1] + `"`)
			if err == nil && toolInspects(name) {
				return true
			}
		}
	}
	return false
}

func validateCodeModeCode(code string, canonicalTools []tools.Tool) error {
	code = strings.TrimSpace(code)
	if code == "" {
		return errors.New("codemode.run_code received empty code")
	}

	// CodeMode has a small, explicit helper API. Reject invented helpers before
	// execution so the planner gets a deterministic repair signal.
	for _, forbidden := range []string{
		"codemode.Return",
		"codemode.Result",
		"codemode.Output",
		"codemode.Emit",
		"codemode.Finish",
		"codemode.Done",
	} {
		if strings.Contains(code, forbidden) {
			return fmt.Errorf("codemode invalid_api: %s does not exist; use codemode.CallTool, codemode.CallToolStream, codemode.SearchTools, codemode.Sprintf, or codemode.Errorf; assign the final result to __out", forbidden)
		}
	}

	if !strings.Contains(code, "CallTool") {
		return nil
	}
	matches := codeModeToolCallRE.FindAllStringSubmatch(code, -1)
	invocations := regexp.MustCompile(`\bCallTool(?:Stream)?\s*\(`).FindAllStringIndex(code, -1)
	if len(matches) != len(invocations) {
		return errors.New("codemode rejected: every CallTool/CallToolStream invocation must use an exact string-literal tool name")
	}
	for _, match := range matches {
		name, err := strconv.Unquote(`"` + match[1] + `"`)
		if err != nil {
			return fmt.Errorf("codemode rejected: invalid tool name literal: %w", err)
		}
		if !toolSpecExists(canonicalTools, name) {
			return fmt.Errorf("codemode unknown_tool: %q is not registered in the canonical UTCP tool registry", name)
		}
	}
	return nil
}

func isCodeModeCompilationError(err error) bool {
	if err == nil {
		return false
	}
	s := strings.ToLower(err.Error())
	return strings.Contains(s, "compilation failed") || strings.Contains(s, "compilation error") || strings.Contains(s, "undefined:")
}

func (s *orchestrationState) observe(plannedMutation, plannedInspection, plannedVerification bool) {
	if plannedInspection {
		s.inspected = true
	}
	if plannedMutation {
		s.mutated = true
	}
	if plannedVerification {
		s.verified = true
	}
}

func (s orchestrationState) completionAllowed() bool {
	if !s.requiresMutation {
		return true
	}
	return s.mutated && s.verified
}

func validatePlannedTool(plannerTools, canonicalTools []tools.Tool, state orchestrationState, toolName string, args map[string]any) (bool, error) {
	if toolName != codemode.CodeModeToolName && toolName != "codemode.run_code" {
		return false, fmt.Errorf("codemode_only: direct tool %q is forbidden; invoke UTCP tools through codemode.run_code", toolName)
	}
	if !toolSpecExists(plannerTools, toolName) {
		return false, fmt.Errorf("unknown_tool: %q is not registered as the CodeMode orchestration tool", toolName)
	}
	code, ok := args["code"].(string)
	if !ok || strings.TrimSpace(code) == "" {
		return false, errors.New("codemode_invalid_code: code must be a non-empty string")
	}
	if err := validateCodeModeCode(code, canonicalTools); err != nil {
		return false, err
	}
	plannedMutation := codeModeMutates(code)
	plannedInspection := codeModeInspects(code)
	if state.requiresMutation && state.inspected && !state.mutated && !plannedMutation {
		return false, errors.New("mutation_required: inspection is complete; the next CodeMode step must perform a real mutation")
	}
	if state.requiresMutation && !state.inspected && !plannedMutation && !plannedInspection {
		return false, errors.New("inspection_required: mutation requests must first inspect the repository inside CodeMode")
	}
	return plannedMutation, nil
}

func validateToolChoice(choice ToolChoice) error {
	if choice.UseTool {
		if strings.TrimSpace(choice.ToolName) == "" {
			return errors.New("invalid_plan: tool_name is required when use_tool=true")
		}
		if choice.Arguments == nil {
			return errors.New("invalid_plan: arguments is required when use_tool=true")
		}
		return nil
	}
	if strings.TrimSpace(choice.ToolName) != "" {
		return errors.New("invalid_plan: tool_name must be empty when use_tool=false")
	}
	return nil
}

func buildMutationRepairPrompt(
	originalPrompt string,
	rawResponse string,
	plannerErr error,
) string {
	return fmt.Sprintf(`%s

MUTATION REPAIR REQUIRED

Your previous CodeMode action was rejected.

Error:
%v

Previous planner response:
<invalid-plan>
%s
</invalid-plan>

The repository has already been inspected.

The NEXT action MUST perform a real mutation.

Generate a NEW CodeMode program that:
- uses codemode.CallTool or codemode.CallToolStream;
- invokes an exact canonical mutation tool;
- prefers filesystem.patch or filesystem.write when available;
- does NOT perform another read/list/search operation;
- does NOT return a completion plan;
- keeps dependent values in one lexical scope.

Return ONLY one JSON object:
{"use_tool":true,"tool_name":"codemode.run_code","arguments":{"code":"..."},"reason":"perform required mutation","final_answer":""}
`, originalPrompt, plannerErr, rawResponse)
}

func parseToolChoice(raw string) (ToolChoice, error) {
	jsonStr := extractJSON(raw)
	if jsonStr == "" {
		return ToolChoice{}, errors.New("invalid_json: planner must return a JSON object")
	}
	var choice ToolChoice
	if err := json.Unmarshal([]byte(jsonStr), &choice); err != nil {
		return ToolChoice{}, fmt.Errorf("invalid_json: %w", err)
	}
	if err := validateToolChoice(choice); err != nil {
		return ToolChoice{}, err
	}
	return choice, nil
}

func buildPlannerRepairPrompt(originalPrompt, rawResponse string, parseErr error) string {
	return fmt.Sprintf(`%s

PLANNER REPAIR REQUIRED
Your previous planner response was rejected.
Error: %v
Previous response:
<invalid-planner-response>
%s
</invalid-planner-response>

Return ONLY one valid JSON object using this exact envelope.
For a tool action:
{"use_tool":true,"tool_name":"codemode.run_code","arguments":{"code":"..."},"reason":"next concrete action","final_answer":""}
For completion:
{"use_tool":false,"tool_name":"","arguments":{},"reason":"complete","final_answer":""}
Do not return markdown, prose, arrays, multiple objects, or unregistered tool names.`, originalPrompt, parseErr, rawResponse)
}

func (a *Agent) toolOrchestrator(ctx context.Context, sessionID, userInput string, records []memory.MemoryRecord, files ...models.File) (bool, string, error) {
	if !a.likelyNeedsToolCall(strings.ToLower(strings.TrimSpace(userInput))) {
		return false, "", nil
	}

	canonicalTools := a.ToolSpecs()
	if len(canonicalTools) == 0 || a.CodeMode == nil {
		return false, "", nil
	}

	orchestrationTools := appendCodeModeToolSpec(nil)
	plannerTools := codeModePlannerTools(orchestrationTools)
	if len(plannerTools) != 1 {
		return false, "", errors.New("codemode_only: CodeMode is not configured as the sole planner tool")
	}

	state := orchestrationState{requiresMutation: requestRequiresMutation(userInput)}
	memoryPrompt := a.renderMemory(records)
	filePrompt := a.buildAttachmentPrompt("Files available for this turn", files)
	workspacePrompt := fileBackedWorkspaceRules(files)
	maxSteps := configuredToolLoopMaxSteps()
	observations := make([]string, 0, maxSteps)
	lastKey := ""
	lastValue := ""

	for step := 1; step <= maxSteps; step++ {
		toolPrompt := a.cachedToolPrompt(plannerTools)
		canonical := codeModeToolNames(canonicalTools)
		prompt := buildToolPlannerPrompt(a.systemInstructions(), userInput, memoryPrompt, filePrompt, workspacePrompt, toolPrompt, canonical, observations, state)
		raw, err := a.model.Generate(ctx, prompt)
		if err != nil {
			return false, "", err
		}

		choice, err := parseToolChoice(fmt.Sprint(raw))
		if err != nil {
			repaired := false
			for attempt := 1; attempt <= defaultPlannerRepairAttempts; attempt++ {
				repairPrompt := buildPlannerRepairPrompt(prompt, fmt.Sprint(raw), err)
				repairedRaw, repairErr := a.model.Generate(ctx, repairPrompt)
				if repairErr != nil {
					err = repairErr
					continue
				}
				repairedChoice, parseErr := parseToolChoice(fmt.Sprint(repairedRaw))
				if parseErr == nil {
					choice = repairedChoice
					repaired = true
					break
				}
				raw = repairedRaw
				err = parseErr
			}
			if !repaired {
				final := fmt.Sprintf("Stopped after %d CodeMode step(s) before completion. planner_error=invalid_json: %v", step-1, err)
				a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "codemode_tool_loop"})
				return true, final, nil
			}
		}

		if !choice.UseTool {
			if !state.completionAllowed() {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=verification_required; mutation occurred but verification is incomplete", step))
				continue
			}
			final := toolChoiceFinalAnswer(choice)
			if final == "" {
				if len(observations) == 0 {
					return false, "", nil
				}
				final = "Done. Last observation:\n" + lastToolObservation(observations)
			}
			a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "codemode_tool_loop"})
			return true, final, nil
		}

		toolName := strings.TrimSpace(choice.ToolName)
		plannedArgs := choice.Arguments
		plannedMutation, err := validatePlannedTool(
			plannerTools,
			canonicalTools,
			state,
			toolName,
			plannedArgs,
		)
		if err != nil {
			observations = append(
				observations,
				fmt.Sprintf("[step %d] planner_error=%v", step, err),
			)

			if strings.Contains(err.Error(), "mutation_required") {
				repairPrompt := buildMutationRepairPrompt(
					prompt,
					fmt.Sprint(raw),
					err,
				)

				repairedRaw, repairErr := a.model.Generate(ctx, repairPrompt)
				if repairErr != nil {
					observations = append(
						observations,
						fmt.Sprintf(
							"[step %d] mutation_repair_error=%v",
							step,
							repairErr,
						),
					)
					continue
				}

				repairedChoice, parseErr := parseToolChoice(
					fmt.Sprint(repairedRaw),
				)
				if parseErr != nil {
					observations = append(
						observations,
						fmt.Sprintf(
							"[step %d] mutation_repair_invalid_json=%v",
							step,
							parseErr,
						),
					)
					continue
				}

				if !repairedChoice.UseTool {
					observations = append(
						observations,
						fmt.Sprintf(
							"[step %d] mutation_repair_invalid: repair returned completion instead of mutation",
							step,
						),
					)
					continue
				}

				repairedToolName := strings.TrimSpace(
					repairedChoice.ToolName,
				)

				repairedArgs := repairedChoice.Arguments

				repairedMutation, repairValidationErr :=
					validatePlannedTool(
						plannerTools,
						canonicalTools,
						state,
						repairedToolName,
						repairedArgs,
					)

				if repairValidationErr != nil {
					observations = append(
						observations,
						fmt.Sprintf(
							"[step %d] mutation_repair_rejected=%v",
							step,
							repairValidationErr,
						),
					)
					continue
				}

				if !repairedMutation {
					observations = append(
						observations,
						fmt.Sprintf(
							"[step %d] mutation_repair_rejected=generated CodeMode program is not mutating",
							step,
						),
					)
					continue
				}

				// Replace the rejected planner decision with the repaired one.
				toolName = repairedToolName
				plannedArgs = repairedArgs
				plannedMutation = repairedMutation
				code, ok := plannedArgs["code"].(string)
				if !ok || strings.TrimSpace(code) == "" {
					observations = append(
						observations,
						fmt.Sprintf(
							"[step %d] mutation_repair_rejected=missing code",
							step,
						),
					)
					continue
				}
			} else {
				continue
			}
		}

		code := plannedArgs["code"].(string)
		plannedInspection := codeModeInspects(code)
		plannedVerification := state.mutated && plannedInspection
		key := toolName + "\x00" + compactJSON(plannedArgs)
		if key == lastKey {
			if state.requiresMutation && state.inspected && !state.mutated {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=duplicate_call; inspection is complete and mutation has not occurred", step))
				continue
			}
			if state.requiresMutation && state.mutated && !state.verified {
				observations = append(observations, fmt.Sprintf("[step %d] planner_error=verification_required; duplicate action cannot satisfy post-mutation verification", step))
				continue
			}
			return true, lastValue, nil
		}

		result, err := a.executeTool(ctx, sessionID, toolName, plannedArgs)
		if err != nil {
			if isCodeModeCompilationError(err) {
				observations = append(observations, fmt.Sprintf("[step %d] codemode_compilation_error=%v; generate fresh code. The runtime API is limited to codemode.CallTool, codemode.CallToolStream, codemode.SearchTools, codemode.Sprintf, and codemode.Errorf. Do not use codemode.Return or other invented helpers. Assign the final result to __out and keep dependent values in one lexical scope.", step, err))
				lastKey = ""
				continue
			}
			a.storeMemory(sessionID, "assistant", fmt.Sprintf("tool %s error: %v", toolName, err), map[string]string{"tool": toolName, "source": "codemode_tool_loop"})
			return true, "", err
		}

		state.observe(plannedMutation, plannedInspection, plannedVerification)
		rawResult := fmt.Sprint(result)
		lastKey, lastValue = key, rawResult
		observations = append(observations, formatToolObservation(step, toolName, plannedArgs, rawResult))
		storeToolObservation(a, sessionID, toolName, rawResult, "codemode_tool_loop")
	}

	final := fmt.Sprintf("Stopped after %d CodeMode step(s) before completion. Last observation:\n%s", maxSteps, lastToolObservation(observations))
	a.storeMemory(sessionID, "assistant", final, map[string]string{"source": "codemode_tool_loop"})
	return true, final, nil
}

func storeToolObservation(a *Agent, sessionID, toolName, rawResult, source string) {
	toonBytes, _ := gotoon.Encode(rawResult)
	a.storeMemory(sessionID, "assistant", fmt.Sprintf("%s\n\n.toon:\n%s", rawResult, string(toonBytes)), map[string]string{"tool": toolName, "source": source})
}

func buildToolPlannerPrompt(systemPrompt, userInput, memoryPrompt, filePrompt, workspacePrompt, toolPrompt, canonicalTools string, observations []string, state orchestrationState) string {
	phase := ""
	if state.requiresMutation && !state.inspected {
		phase = "\nPHASE: INSPECTION\nPerform repository inspection first using CodeMode. Do not attempt to finish the request until you have inspected the relevant files.\n"
	} else if state.requiresMutation && state.inspected && !state.mutated {
		phase = "\nPHASE: MUTATION\nInspection is complete. Your next CodeMode action MUST perform the requested mutation using filesystem.patch or filesystem.write when available.\n"
	} else if state.requiresMutation && state.mutated && !state.verified {
		phase = "\nPHASE: VERIFICATION\nThe mutation has occurred. Verify the resulting file and only then finish.\n"
	}

	return fmt.Sprintf(`
You are the execution controller for a UTCP agent.

MISSION
Execute the user's request. Do not merely describe a plan.

HARD EXECUTION CONTRACT
- CodeMode is the ONLY planner/execution tool available to you.
- NEVER return filesystem.*, shell.*, git.*, or any other canonical UTCP tool as tool_name.
- Repository inspection and mutation MUST happen inside codemode.run_code.
- If the task requires repository work, your next action must be a codemode.run_code call containing Go code that invokes exact canonical tools through codemode.CallTool or codemode.CallToolStream.
- Do NOT emit narration such as "I'll inspect..." instead of a tool call.
- Return the JSON tool call immediately when another action is required.
%s
USER REQUEST
%q

SYSTEM INSTRUCTIONS
%s

ONLY AVAILABLE PLANNER TOOL
%s

CANONICAL UTCP TOOL NAMES AVAILABLE INSIDE CODEMODE
%s

MEMORY
<untrusted-memory>
%s
</untrusted-memory>

FILES
<untrusted-files>
%s
</untrusted-files>

WORKSPACE
<runtime-context>
%s
</runtime-context>

OBSERVATIONS
<runtime-observations>
%s
</runtime-observations>

EXECUTION STATE
requires_mutation=%t
inspection_complete=%t
mutation_complete=%t
verification_complete=%t

CODEMODE RULES
- Use only the real CodeMode API: codemode.CallTool, codemode.CallToolStream, codemode.SearchTools, codemode.Sprintf, codemode.Errorf.
- NEVER invent codemode.Return, codemode.Result, codemode.Output, codemode.Emit, codemode.Finish, or codemode.Done.
- The final CodeMode value must be assigned to __out.
- Keep dependent values in one lexical scope.
- For inspection, call the appropriate canonical read/search/list tool from inside CodeMode.
- For mutation, call filesystem.write or filesystem.patch from inside CodeMode when those exact tools are registered.
- After mutation, use CodeMode again to verify when practical.
- A prose explanation is not an execution step.
- A CodeMode program containing only reads is allowed during the inspection phase.
- Once inspection_complete=true and mutation_complete=false, a read-only CodeMode program is forbidden.
- Once mutation_complete=true and verification_complete=false, use CodeMode to verify the resulting state before returning complete.

OUTPUT
Return ONLY one JSON object:
{"use_tool":true,"tool_name":"codemode.run_code","arguments":{"code":"..."},"reason":"next concrete action","final_answer":""}
OR, only when the request is actually complete:
{"use_tool":false,"tool_name":"","arguments":{},"reason":"complete","final_answer":""}
`, phase, userInput, systemPrompt, toolPrompt, canonicalTools, memoryPrompt, filePrompt, workspacePrompt, strings.Join(observations, "\n\n"), state.requiresMutation, state.inspected, state.mutated, state.verified)
}

func codeModeToolNames(toolList []tools.Tool) string {
	var b strings.Builder
	for _, spec := range toolList {
		name := strings.TrimSpace(spec.Name)
		if name == "" || name == codemode.CodeModeToolName || name == "codemode.run_code" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(name)
		b.WriteByte('\n')
	}
	return b.String()
}

func extractJSON(response string) string {
	response = strings.TrimSpace(response)
	if strings.Contains(response, "```") {
		response = strings.TrimPrefix(response, "```json")
		response = strings.TrimPrefix(response, "```")
		response = strings.TrimSpace(response)
		if idx := strings.Index(response, "```"); idx >= 0 {
			response = response[:idx]
		}
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
