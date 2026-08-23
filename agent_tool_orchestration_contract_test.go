package agent

import (
	"strings"
	"testing"

	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

func TestParseToolChoiceRejectsInvalidJSON(t *testing.T) {
	_, err := parseToolChoice(`{"use_tool":true`)
	if err == nil {
		t.Fatal("expected invalid JSON to be rejected")
	}
	if !strings.Contains(err.Error(), "invalid_json") {
		t.Fatalf("expected invalid_json error, got %v", err)
	}
}

func TestParseToolChoiceRejectsEmptyPlannerObject(t *testing.T) {
	_, err := parseToolChoice(`{}`)
	if err == nil {
		t.Fatal("expected empty planner object to be rejected")
	}
	if !strings.Contains(err.Error(), "invalid_plan") {
		t.Fatalf("expected invalid_plan error, got %v", err)
	}
}

func TestParseToolChoiceRejectsToolWithoutName(t *testing.T) {
	_, err := parseToolChoice(`{"use_tool":true,"arguments":{}}`)
	if err == nil {
		t.Fatal("expected tool choice without tool_name to be rejected")
	}
	if !strings.Contains(err.Error(), "invalid_plan") {
		t.Fatalf("expected invalid_plan error, got %v", err)
	}
}

func TestParseToolChoiceAcceptsCompletionEnvelope(t *testing.T) {
	choice, err := parseToolChoice(`{"use_tool":false,"tool_name":"","arguments":{},"reason":"complete","final_answer":"done"}`)
	if err != nil {
		t.Fatalf("expected valid completion envelope, got %v", err)
	}
	if choice.UseTool {
		t.Fatal("expected completion envelope to disable tool use")
	}
}

func TestCompletionRequiresVerificationAfterMutation(t *testing.T) {
	state := orchestrationState{
		requiresMutation: true,
		inspected:        true,
		mutated:          true,
		verified:         false,
	}
	if state.completionAllowed() {
		t.Fatal("mutation without verification must not allow completion")
	}
	state.verified = true
	if !state.completionAllowed() {
		t.Fatal("verified mutation should allow completion")
	}
}

func TestValidatePlannedToolRejectsDirectTool(t *testing.T) {
	plannerTools := []tools.Tool{{Name: codemode.CodeModeToolName}}
	canonicalTools := []tools.Tool{{Name: "filesystem.write"}}
	_, err := validatePlannedTool(plannerTools, canonicalTools, orchestrationState{}, "filesystem.write", map[string]any{
		"path": "README.md",
	})
	if err == nil || !strings.Contains(err.Error(), "codemode_only") {
		t.Fatalf("expected codemode_only error, got %v", err)
	}
}

func TestValidatePlannedToolRejectsUnknownCodeModeTool(t *testing.T) {
	plannerTools := []tools.Tool{{Name: codemode.CodeModeToolName}}
	canonicalTools := []tools.Tool{{Name: "filesystem.read"}}
	_, err := validatePlannedTool(plannerTools, canonicalTools, orchestrationState{}, codemode.CodeModeToolName, map[string]any{
		"code": `CallTool("filesystem.write", map[string]any{"path":"README.md"})`,
	})
	if err == nil || !strings.Contains(err.Error(), "unknown_tool") {
		t.Fatalf("expected unknown_tool error, got %v", err)
	}
}

func TestValidatePlannedToolAllowsMutationAfterInspection(t *testing.T) {
	plannerTools := []tools.Tool{{Name: codemode.CodeModeToolName}}
	canonicalTools := []tools.Tool{{Name: "filesystem.read"}, {Name: "filesystem.write"}}
	state := orchestrationState{requiresMutation: true, inspected: true}

	mutates, err := validatePlannedTool(plannerTools, canonicalTools, state, codemode.CodeModeToolName, map[string]any{
		"code": `CallTool("filesystem.write", map[string]any{"path":"README.md","content":"updated"})`,
	})
	if err != nil {
		t.Fatalf("expected mutation to be allowed, got %v", err)
	}
	if !mutates {
		t.Fatal("expected filesystem.write CodeMode plan to be mutation-capable")
	}
}

func TestValidatePlannedToolRejectsReadAfterInspection(t *testing.T) {
	plannerTools := []tools.Tool{{Name: codemode.CodeModeToolName}}
	canonicalTools := []tools.Tool{{Name: "filesystem.read"}, {Name: "filesystem.write"}}
	state := orchestrationState{requiresMutation: true, inspected: true}

	_, err := validatePlannedTool(plannerTools, canonicalTools, state, codemode.CodeModeToolName, map[string]any{
		"code": `CallTool("filesystem.read", map[string]any{"path":"README.md"})`,
	})
	if err == nil {
		t.Fatal("expected read-only CodeMode plan to be rejected after inspection")
	}
	if !strings.Contains(err.Error(), "mutation_required") {
		t.Fatalf("expected mutation_required error, got %v", err)
	}
}
