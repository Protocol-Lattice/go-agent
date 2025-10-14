package agent

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

type stubModel struct {
	response string
	err      error
}

func (m *stubModel) Generate(ctx context.Context, prompt string) (any, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.response + " | " + prompt, nil
}

type stubTool struct {
	spec      ToolSpec
	lastInput ToolRequest
}

func (t *stubTool) Spec() ToolSpec { return t.spec }
func (t *stubTool) Invoke(ctx context.Context, req ToolRequest) (ToolResponse, error) {
	t.lastInput = req
	val := req.Arguments["input"]
	if val == nil {
		return ToolResponse{Content: ""}, nil
	}
	str, _ := val.(string)
	return ToolResponse{Content: str}, nil
}

type stubSubAgent struct {
	name        string
	description string
}

func (s *stubSubAgent) Name() string        { return s.name }
func (s *stubSubAgent) Description() string { return s.description }
func (s *stubSubAgent) Run(ctx context.Context, input string) (string, error) {
	return input, nil
}

func TestNewAppliesDefaults(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	agent, err := New(Options{Model: model, Memory: mem})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if agent.systemPrompt == "" {
		t.Fatalf("expected default system prompt to be applied")
	}
	if agent.contextLimit != 8 {
		t.Fatalf("expected default context limit of 8, got %d", agent.contextLimit)
	}
}

func TestNewRegistersToolsAndSubagents(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)

	tool := &stubTool{spec: ToolSpec{Name: "Echo", Description: "desc"}}
	researcher := &stubSubAgent{name: "Researcher", description: "desc"}

	agent, err := New(Options{
		Model:     model,
		Memory:    mem,
		Tools:     []Tool{tool},
		SubAgents: []SubAgent{researcher},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if _, ok := agent.tools["echo"]; !ok {
		t.Fatalf("expected tool to be registered in lowercase")
	}
	if spec := agent.toolSpecs["echo"]; spec.Name != "Echo" {
		t.Fatalf("expected tool spec to keep original casing")
	}
	if len(agent.toolOrder) != 1 || agent.toolOrder[0] != "echo" {
		t.Fatalf("expected tool order to preserve insertion")
	}

	if _, ok := agent.subagents["researcher"]; !ok {
		t.Fatalf("expected subagent to be registered in lowercase")
	}
	if len(agent.subagentOrder) != 1 || agent.subagentOrder[0] != researcher {
		t.Fatalf("expected subagent order to preserve insertion")
	}
}

func TestNewValidatesRequirements(t *testing.T) {
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	if _, err := New(Options{Memory: mem}); err == nil {
		t.Fatalf("expected error when model is missing")
	}
	model := &stubModel{response: "ok"}
	if _, err := New(Options{Model: model}); err == nil {
		t.Fatalf("expected error when memory is missing")
	}
}

func TestSplitCommand(t *testing.T) {
	name, args := splitCommand("toolName   with extra spacing")
	if name != "toolName" {
		t.Fatalf("unexpected name: %q", name)
	}
	if args != "with extra spacing" {
		t.Fatalf("unexpected args: %q", args)
	}
}

func TestMetadataRole(t *testing.T) {
	payload, _ := json.Marshal(map[string]string{"role": "assistant"})
	role := metadataRole(string(payload))
	if role != "assistant" {
		t.Fatalf("expected role assistant, got %q", role)
	}

	if role := metadataRole("{invalid json}"); role != "unknown" {
		t.Fatalf("expected fallback role unknown, got %q", role)
	}

	if role := metadataRole("{}"); role != "unknown" {
		t.Fatalf("expected missing role to map to unknown, got %q", role)
	}
}

func TestNewHonorsExplicitSettings(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	agent, err := New(Options{Model: model, Memory: mem, SystemPrompt: "custom", ContextLimit: 2})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if agent.systemPrompt != "custom" {
		t.Fatalf("expected custom prompt, got %q", agent.systemPrompt)
	}
	if agent.contextLimit != 2 {
		t.Fatalf("expected custom context limit, got %d", agent.contextLimit)
	}
}

func TestToolsWithEmptyNamesAreSkipped(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	tool := &stubTool{spec: ToolSpec{Name: ""}}
	agent, err := New(Options{Model: model, Memory: mem, Tools: []Tool{tool}})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if len(agent.tools) != 0 {
		t.Fatalf("expected unnamed tool to be ignored")
	}
}

func TestSubagentsWithEmptyNamesAreIgnored(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	sub := &stubSubAgent{name: ""}
	agent, err := New(Options{Model: model, Memory: mem, SubAgents: []SubAgent{sub}})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	if len(agent.subagents) != 0 {
		t.Fatalf("expected unnamed subagent to be ignored")
	}
}
