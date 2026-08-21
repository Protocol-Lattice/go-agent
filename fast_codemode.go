package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

// fastCodeModePrompt turns a natural-language request into a single CodeMode
// execution pass. The normal planner remains the fallback for requests that
// CodeMode cannot satisfy.
func fastCodeModePrompt(systemPrompt, userInput string, files []models.File) string {
	var b strings.Builder
	b.Grow(len(systemPrompt) + len(userInput) + 4096)

	b.WriteString("FAST CODEMODE EXECUTION\n")
	b.WriteString("Use exactly one CodeMode generation/execution pass.\n")
	b.WriteString("Select only exact registered UTCP tools.\n")
	b.WriteString("Do not explain a plan instead of executing it.\n")
	b.WriteString("For mutation requests, a real mutation-capable tool MUST execute successfully before returning.\n")
	b.WriteString("Prefer one straight-line program: inspect -> mutate -> verify.\n")
	b.WriteString("Do not use another planner round-trip for intermediate tool selection.\n\n")

	if strings.TrimSpace(systemPrompt) != "" {
		b.WriteString("SYSTEM / SKILL INSTRUCTIONS:\n")
		b.WriteString(systemPrompt)
		b.WriteString("\n\n")
	}

	if len(files) > 0 {
		b.WriteString(fileBackedWorkspaceRules(files))
		b.WriteString("\n")
		b.WriteString((&Agent{}).buildAttachmentPrompt("FILES AVAILABLE FOR THIS TURN", files))
		b.WriteString("\n")
	}

	b.WriteString("USER REQUEST:\n")
	b.WriteString(sanitizeInput(userInput))
	b.WriteString("\n\n")
	b.WriteString("FAST EXECUTION CONTRACT:\n")
	b.WriteString("- Read/list/search only when needed to perform the task.\n")
	b.WriteString("- If the request changes state, execute a mutation in this same CodeMode run.\n")
	b.WriteString("- Never report success for a mutation that was not executed.\n")
	b.WriteString("- If CodeMode cannot safely complete the request, return no tool plan so the caller can use the normal planner fallback.\n")
	return b.String()
}

func (a *Agent) generateFastCodeMode(ctx context.Context, sessionID, userInput string, routing SkillRouting, files []models.File) (any, bool, error) {
	requestAgent, err := a.newSkillScopedAgent(routing)
	if err != nil {
		return nil, false, err
	}
	if requestAgent.CodeMode == nil {
		return nil, false, nil
	}

	prompt := fastCodeModePrompt(requestAgent.systemPrompt, userInput, files)
	handled, output, err := requestAgent.CodeMode.CallTool(ctx, prompt)
	if err != nil {
		// CodeMode compilation/planning failures are recoverable. The caller
		// falls back to the existing bounded planner/tool loop.
		return nil, false, nil
	}
	if !handled {
		return nil, false, nil
	}

	if requestAgent.Guardrails != nil {
		validated, guardrailErr := requestAgent.Guardrails.ValidateAndRepair(ctx, fmt.Sprint(output))
		if guardrailErr != nil {
			return nil, false, guardrailErr
		}
		output = validated
	}

	// CodeMode has executed the tool program, so the fast path is complete.
	// The normal Generate path is intentionally bypassed to avoid another LLM
	// round-trip after a successful CodeMode execution.
	return output, true, nil
}
