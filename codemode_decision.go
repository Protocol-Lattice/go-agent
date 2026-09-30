package agent

import (
	"context"
	"errors"
	"encoding/json"
	"fmt"
	"os"
	"sort"
	"strconv"
	"strings"
	"unicode"

	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

const defaultCodeModeJevCandidateLimit = 20

type codeModeDecisionPlan struct {
	Tools  []string `json:"tools"`
	Code   string   `json:"code"`
	Stream bool     `json:"stream"`
}

// callCodeMode is the single CodeMode planning entry point used by Agent.
// Without a decision layer it preserves the native go-utcp CodeMode behavior.
// With a decision layer, Jev selects the exact UTCP tool first and the
// generative CodeMode model is restricted to producing arguments/Expr code for
// that selection.
func (a *Agent) callCodeMode(ctx context.Context, userInput string) (bool, any, error) {
	if a.CodeMode == nil {
		return false, "", nil
	}
	if a.codeModeDecisionLayer == nil {
		return a.CodeMode.CallTool(ctx, userInput)
	}

	specs := rankCodeModeDecisionCandidates(
		userInput,
		a.CodeMode.ToolSpecs(),
		configuredCodeModeJevCandidateLimit(),
	)
	if len(specs) == 0 {
		return false, "", nil
	}

	decisionTools := make([]ToolDecisionTool, 0, len(specs))
	for _, spec := range specs {
		decisionTools = append(decisionTools, ToolDecisionTool{
			Name:        spec.Name,
			Description: codeModeDecisionToolDescription(spec),
		})
	}

	decision, err := a.codeModeDecisionLayer.DecideTool(ctx, ToolDecisionInput{
		UserInput:          userInput,
		SystemInstructions: a.systemInstructions(),
		Tools:              decisionTools,
		Step:               1,
	})
	if err != nil {
		if ctx.Err() != nil {
			return false, "", ctx.Err()
		}
		var confidenceErr *ToolDecisionConfidenceError
		if errors.As(err, &confidenceErr) {
			return true, "", confidenceErr
		}
		orchestratorLogf("codemode decision layer failed err=%v; falling back to native codemode planner", err)
		return a.CodeMode.CallTool(ctx, userInput)
	}

	orchestratorLogf(
		"codemode decision layer selected use_tool=%t tool=%q confidence=%.3f provider=%q request_id=%q candidates=%d",
		decision.UseTool,
		decision.ToolName,
		decision.Confidence,
		decision.Provider,
		decision.RequestID,
		len(specs),
	)

	if !decision.UseTool {
		return false, "", nil
	}

	selected, ok := codeModeToolByName(specs, decision.ToolName)
	if !ok {
		return true, "", fmt.Errorf("codemode decision layer selected unavailable tool %q", decision.ToolName)
	}

	plan, err := a.generateSelectedCodeModePlan(ctx, userInput, selected)
	if err != nil {
		return true, "", err
	}

	if err := a.validateSelectedCodeModePlan(plan, selected.Name); err != nil {
		return true, "", err
	}

	result, err := a.CodeMode.Execute(ctx, codemode.CodeModeArgs{
		Code:    plan.Code,
		Timeout: 20000,
	})
	if err != nil {
		return true, "", err
	}
	return true, result, nil
}

func (a *Agent) generateSelectedCodeModePlan(
	ctx context.Context,
	userInput string,
	selected tools.Tool,
) (codeModeDecisionPlan, error) {
	planner := a.codeModePlannerModel
	if planner == nil {
		planner = a.model
	}

	inputSchema, _ := json.Marshal(selected.Inputs)
	outputSchema, _ := json.Marshal(selected.Outputs)

	prompt := fmt.Sprintf(`
You are a UTCP CodeMode Expr generator.

A separate decision model (Jev) already selected the exact next UTCP tool.
You MUST NOT choose, replace, infer, or add another tool.

SELECTED TOOL:
name: %s
description: %s
input_schema: %s
output_schema: %s

USER QUERY:
%q

Generate the arguments and one executable Expr v1.17 program for the selected
tool. Use exactly one of:
- codemode.CallTool("%s", {...})
- codemode.CallToolStream("%s", {...})

You may use codemode.Get(...) only to read a field from the selected tool's
result. Do not reference any other tool. Do not emit Go syntax or markdown.

Return exactly one JSON object:
{"tools":["%s"],"code":"<Expr source>","stream":false}
`,
		selected.Name,
		selected.Description,
		string(inputSchema),
		string(outputSchema),
		userInput,
		selected.Name,
		selected.Name,
		selected.Name,
	)

	raw, err := planner.Generate(ctx, prompt)
	if err != nil {
		return codeModeDecisionPlan{}, fmt.Errorf("codemode selected-tool generation: %w", err)
	}

	jsonText := extractJSON(fmt.Sprint(raw))
	if jsonText == "" {
		return codeModeDecisionPlan{}, fmt.Errorf("codemode selected-tool generation returned no JSON")
	}

	var plan codeModeDecisionPlan
	if err := json.Unmarshal([]byte(jsonText), &plan); err != nil {
		return codeModeDecisionPlan{}, fmt.Errorf("decode codemode selected-tool plan: %w", err)
	}
	plan.Code = strings.TrimSpace(plan.Code)
	return plan, nil
}

func (a *Agent) validateSelectedCodeModePlan(plan codeModeDecisionPlan, selectedTool string) error {
	if len(plan.Tools) != 1 || strings.TrimSpace(plan.Tools[0]) != selectedTool {
		return fmt.Errorf("codemode plan must declare only Jev-selected tool %q", selectedTool)
	}
	if plan.Code == "" {
		return fmt.Errorf("codemode plan for %q returned empty Expr source", selectedTool)
	}

	if err := a.validateCodeModeToolCalls(plan.Code); err != nil {
		return err
	}

	matches := codeModeToolCallPattern.FindAllStringSubmatch(plan.Code, -1)
	if len(matches) == 0 {
		return fmt.Errorf("codemode plan for %q contains no CallTool/CallToolStream invocation", selectedTool)
	}
	for _, match := range matches {
		if len(match) < 2 || match[1] != selectedTool {
			var got string
			if len(match) >= 2 {
				got = match[1]
			}
			return fmt.Errorf("codemode decision selected %q but generated code references %q", selectedTool, got)
		}
	}

	usesStream := strings.Contains(plan.Code, "codemode.CallToolStream(")
	if usesStream != plan.Stream {
		return fmt.Errorf("codemode stream flag does not match generated code")
	}
	return nil
}

func codeModeDecisionToolDescription(spec tools.Tool) string {
	parts := make([]string, 0, 4)
	if description := strings.TrimSpace(spec.Description); description != "" {
		parts = append(parts, description)
	}
	if len(spec.Tags) > 0 {
		tags := append([]string(nil), spec.Tags...)
		sort.Strings(tags)
		parts = append(parts, "tags: "+strings.Join(tags, ", "))
	}
	if len(spec.Inputs.Properties) > 0 {
		fields := make([]string, 0, len(spec.Inputs.Properties))
		for field := range spec.Inputs.Properties {
			fields = append(fields, field)
		}
		sort.Strings(fields)
		parts = append(parts, "input fields: "+strings.Join(fields, ", "))
	}
	if len(spec.Inputs.Required) > 0 {
		required := append([]string(nil), spec.Inputs.Required...)
		sort.Strings(required)
		parts = append(parts, "required: "+strings.Join(required, ", "))
	}
	return strings.Join(parts, "; ")
}

func codeModeToolByName(specs []tools.Tool, name string) (tools.Tool, bool) {
	name = strings.TrimSpace(name)
	for _, spec := range specs {
		if spec.Name == name {
			return spec, true
		}
	}
	return tools.Tool{}, false
}

func configuredCodeModeJevCandidateLimit() int {
	raw := strings.TrimSpace(os.Getenv("UTCP_CODEMODE_JEV_CANDIDATE_LIMIT"))
	if raw == "" {
		return defaultCodeModeJevCandidateLimit
	}
	limit, err := strconv.Atoi(raw)
	if err != nil || limit <= 0 {
		return defaultCodeModeJevCandidateLimit
	}
	return limit
}

type codeModeScoredTool struct {
	spec  tools.Tool
	score int
	index int
}

func rankCodeModeDecisionCandidates(query string, specs []tools.Tool, limit int) []tools.Tool {
	if limit <= 0 {
		limit = defaultCodeModeJevCandidateLimit
	}
	lowerQuery := strings.ToLower(query)
	terms := codeModeDecisionTerms(lowerQuery)

	ranked := make([]codeModeScoredTool, 0, len(specs))
	for index, spec := range specs {
		name := strings.TrimSpace(spec.Name)
		if name == "" || name == codemode.CodeModeToolName || name == "codemode.run_code" {
			continue
		}

		lowerName := strings.ToLower(name)
		score := 0
		if strings.Contains(lowerQuery, lowerName) {
			score += 200
		}
		if provider, _, ok := strings.Cut(lowerName, "."); ok && strings.Contains(lowerQuery, provider) {
			score += 30
		}

		for _, term := range terms {
			if strings.Contains(lowerName, term) {
				score += 20
			}
			if strings.Contains(strings.ToLower(spec.Description), term) {
				score += 4
			}
			for _, tag := range spec.Tags {
				if strings.Contains(strings.ToLower(tag), term) {
					score += 8
					break
				}
			}
			for field := range spec.Inputs.Properties {
				if strings.Contains(strings.ToLower(field), term) {
					score += 6
				}
			}
		}

		ranked = append(ranked, codeModeScoredTool{
			spec:  spec,
			score: score,
			index: index,
		})
	}

	sort.SliceStable(ranked, func(i, j int) bool {
		if ranked[i].score == ranked[j].score {
			return ranked[i].index < ranked[j].index
		}
		return ranked[i].score > ranked[j].score
	})

	if len(ranked) > limit {
		ranked = ranked[:limit]
	}

	out := make([]tools.Tool, len(ranked))
	for i, candidate := range ranked {
		out[i] = candidate.spec
	}
	return out
}

func codeModeDecisionTerms(query string) []string {
	parts := strings.FieldsFunc(query, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r) && r != '_' && r != '-'
	})
	seen := make(map[string]struct{}, len(parts))
	terms := make([]string, 0, len(parts))
	for _, part := range parts {
		part = strings.ToLower(strings.TrimSpace(part))
		if len(part) < 2 || isCodeModeDecisionStopWord(part) {
			continue
		}
		if _, ok := seen[part]; ok {
			continue
		}
		seen[part] = struct{}{}
		terms = append(terms, part)
	}
	return terms
}

func isCodeModeDecisionStopWord(word string) bool {
	switch word {
	case "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "in", "is", "it", "of", "on", "or", "the", "to", "use", "using", "with":
		return true
	default:
		return false
	}
}
