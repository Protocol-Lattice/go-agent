package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	utcpTools "github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

type stubCodeModeDecisionLayer struct {
	decision ToolDecision
	err      error
	calls    int
	input    ToolDecisionInput
}

func (s *stubCodeModeDecisionLayer) DecideTool(_ context.Context, input ToolDecisionInput) (ToolDecision, error) {
	s.calls++
	s.input = input
	return s.decision, s.err
}

func TestCodeModeJevDecisionRestrictsGeneratedTool(t *testing.T) {
	model := &dynamicStubModel{responses: map[string]string{
		"A separate decision model (Jev) already selected the exact next UTCP tool": `{"tools":["beta"],"code":"let result = codemode.CallTool(\\"beta\\", {\\"input\\": \\"hello\\"}); result","stream":false}`,
	}}
	client := &stubUTCPClient{searchTools: []utcpTools.Tool{
		{
			Name:        "alpha",
			Description: "Alpha tool",
			Inputs: utcpTools.ToolInputOutputSchema{
				Type:       "object",
				Properties: map[string]any{"input": map[string]any{"type": "string"}},
				Required:   []string{"input"},
			},
		},
		{
			Name:        "beta",
			Description: "Beta tool",
			Inputs: utcpTools.ToolInputOutputSchema{
				Type:       "object",
				Properties: map[string]any{"input": map[string]any{"type": "string"}},
				Required:   []string{"input"},
			},
		},
	}}
	layer := &stubCodeModeDecisionLayer{decision: ToolDecision{
		UseTool:    true,
		ToolName:   "beta",
		Confidence: 0.94,
		Provider:   "TypeSafe",
		RequestID:  "gen-dec-test",
	}}

	a, err := New(Options{
		Model:                 model,
		Memory:                memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		UTCPClient:            client,
		CodeMode:              codemode.NewCodeModeUTCP(client, model),
		CodeModePlannerModel:  model,
		CodeModeDecisionLayer: layer,
		AllowUnsafeTools:      true,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	out, err := a.Generate(context.Background(), "session", "Run code with CodeMode using the beta tool with input hello.")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if layer.calls != 1 {
		t.Fatalf("decision layer calls = %d, want 1", layer.calls)
	}
	if client.callCount != 1 || client.lastToolName != "beta" {
		t.Fatalf("UTCP execution = count %d tool %q, want one beta call", client.callCount, client.lastToolName)
	}
	if _, ok := out.(codemode.CodeModeResult); !ok {
		t.Fatalf("Generate output type = %T, want codemode.CodeModeResult", out)
	}

	foundBeta := false
	for _, tool := range layer.input.Tools {
		if tool.Name == "beta" {
			foundBeta = true
			break
		}
	}
	if !foundBeta {
		t.Fatalf("Jev candidates did not include beta: %#v", layer.input.Tools)
	}
}

func TestCodeModeJevDecisionRejectsDifferentGeneratedTool(t *testing.T) {
	model := &dynamicStubModel{responses: map[string]string{
		"A separate decision model (Jev) already selected the exact next UTCP tool": `{"tools":["beta"],"code":"codemode.CallTool(\\"alpha\\", {\\"input\\": \\"hello\\"})","stream":false}`,
	}}
	client := &stubUTCPClient{searchTools: []utcpTools.Tool{
		{Name: "alpha", Description: "Alpha tool"},
		{Name: "beta", Description: "Beta tool"},
	}}
	layer := &stubCodeModeDecisionLayer{decision: ToolDecision{UseTool: true, ToolName: "beta"}}

	a, err := New(Options{
		Model:                 model,
		Memory:                memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		UTCPClient:            client,
		CodeMode:              codemode.NewCodeModeUTCP(client, model),
		CodeModePlannerModel:  model,
		CodeModeDecisionLayer: layer,
		AllowUnsafeTools:      true,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = a.Generate(context.Background(), "session", "Run code with CodeMode using beta.")
	if err == nil || !strings.Contains(err.Error(), "generated code references") {
		t.Fatalf("Generate error = %v, want selected-tool mismatch", err)
	}
	if client.callCount != 0 {
		t.Fatalf("UTCP call count = %d, want 0 after mismatch", client.callCount)
	}
}

func TestCodeModeDecisionLayerFailureFallsBackToNativePlanner(t *testing.T) {
	model := &dynamicStubModel{responses: map[string]string{
		"You are a strict UTCP CodeMode planner and executor": `{"tools":["alpha"],"code":"codemode.CallTool(\\"alpha\\", {})","stream":false}`,
	}}
	client := &stubUTCPClient{searchTools: []utcpTools.Tool{{Name: "alpha", Description: "Alpha tool"}}}
	layer := &stubCodeModeDecisionLayer{err: errors.New("decision backend unavailable")}

	a, err := New(Options{
		Model:                 model,
		Memory:                memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		UTCPClient:            client,
		CodeMode:              codemode.NewCodeModeUTCP(client, model),
		CodeModePlannerModel:  model,
		CodeModeDecisionLayer: layer,
		AllowUnsafeTools:      true,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = a.Generate(context.Background(), "session", "Run code with CodeMode using alpha.")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if client.callCount != 1 || client.lastToolName != "alpha" {
		t.Fatalf("fallback execution = count %d tool %q, want one alpha call", client.callCount, client.lastToolName)
	}
}
