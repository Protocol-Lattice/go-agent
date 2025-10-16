package agent

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

type stubModel struct {
	response   string
	err        error
	lastPrompt string
}

func (m *stubModel) Generate(ctx context.Context, prompt string) (any, error) {
	if m.err != nil {
		return nil, m.err
	}
	m.lastPrompt = prompt
	return m.response, nil
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

type stubPlanner struct {
	output    PlannerOutput
	err       error
	lastInput PlannerInput
	called    bool
}

func (p *stubPlanner) Plan(ctx context.Context, input PlannerInput) (PlannerOutput, error) {
	p.called = true
	p.lastInput = input
	if p.err != nil {
		return PlannerOutput{}, p.err
	}
	return p.output, nil
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

	specs := agent.ToolSpecs()
	if len(specs) != 1 {
		t.Fatalf("expected 1 tool spec, got %d", len(specs))
	}
	if specs[0].Name != "Echo" {
		t.Fatalf("expected tool spec to retain original casing, got %q", specs[0].Name)
	}

	subagents := agent.SubAgents()
	if len(subagents) != 1 {
		t.Fatalf("expected 1 subagent, got %d", len(subagents))
	}
	if subagents[0] != researcher {
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
	if len(agent.Tools()) != 0 {
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
	if len(agent.SubAgents()) != 0 {
		t.Fatalf("expected unnamed subagent to be ignored")
	}
}

func TestStaticToolCatalogRejectsDuplicate(t *testing.T) {
	catalog := NewStaticToolCatalog(nil)
	if err := catalog.Register(&stubTool{spec: ToolSpec{Name: "Echo"}}); err != nil {
		t.Fatalf("unexpected register error: %v", err)
	}
	if err := catalog.Register(&stubTool{spec: ToolSpec{Name: "echo"}}); err == nil {
		t.Fatalf("expected duplicate registration error")
	}
}

func TestAgentPropagatesCustomCatalogErrors(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	catalog := NewStaticToolCatalog([]Tool{&stubTool{spec: ToolSpec{Name: "Echo"}}})
	_, err := New(Options{
		Model:       model,
		Memory:      mem,
		ToolCatalog: catalog,
		Tools:       []Tool{&stubTool{spec: ToolSpec{Name: "Echo"}}},
	})
	if err == nil {
		t.Fatalf("expected duplicate registration error from custom catalog")
	}
}

func TestAgentPropagatesCustomDirectoryErrors(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	dir := NewStaticSubAgentDirectory([]SubAgent{&stubSubAgent{name: "researcher"}})
	_, err := New(Options{
		Model:             model,
		Memory:            mem,
		SubAgentDirectory: dir,
		SubAgents:         []SubAgent{&stubSubAgent{name: "Researcher"}},
	})
	if err == nil {
		t.Fatalf("expected duplicate registration error from custom directory")
	}
}

func TestAgentRespondInvokesPlanner(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 8)
	mem.WithEmbedder(memory.DummyEmbedder{})

	planner := &stubPlanner{output: PlannerOutput{
		Thoughts: []string{"Evaluate requested task"},
		Steps:    []string{"Gather recent context", "Draft concise answer"},
		Decision: "Answer using context and stay terse",
	}}

	agent, err := New(Options{Model: model, Memory: mem, Planner: planner})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	if _, err := agent.Respond(ctx, "session-1", "Summarise reasoning layer"); err != nil {
		t.Fatalf("Respond returned error: %v", err)
	}

	if !planner.called {
		t.Fatalf("expected planner to be invoked")
	}
	if planner.lastInput.UserInput != "Summarise reasoning layer" {
		t.Fatalf("planner received unexpected input: %q", planner.lastInput.UserInput)
	}
	if len(planner.lastInput.Context) == 0 {
		t.Fatalf("expected planner to receive conversation context")
	}
	if !strings.Contains(model.lastPrompt, "Internal planner notes") {
		t.Fatalf("expected prompt to contain planner notes, got %q", model.lastPrompt)
	}
	if !strings.Contains(model.lastPrompt, "Draft concise answer") {
		t.Fatalf("expected prompt to include planner steps")
	}

	records, err := mem.RetrieveContext(ctx, "session-1", "follow up", 8)
	if err != nil {
		t.Fatalf("RetrieveContext returned error: %v", err)
	}

	foundPlanner := false
	for _, rec := range records {
		if metadataRole(rec.Metadata) == "planner" && strings.Contains(rec.Content, "Draft concise answer") {
			foundPlanner = true
			break
		}
	}
	if !foundPlanner {
		t.Fatalf("expected planner output to be persisted in memory")
	}
}
