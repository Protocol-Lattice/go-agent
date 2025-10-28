package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"strings"
	"testing"

	"github.com/Protocol-Lattice/agent/core/memory"
	"github.com/Protocol-Lattice/agent/core/models"
)

type stubModel struct {
	response string
	err      error
}

func (g *stubModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return nil, nil
}
func (m *stubModel) Generate(ctx context.Context, prompt string) (any, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.response + " | " + prompt, nil
}

type fileEchoModel struct {
	response string
}

func (m *fileEchoModel) Generate(ctx context.Context, prompt string) (any, error) {
	return m.response, nil
}

func (m *fileEchoModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
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

func TestAgentSharesMemoryAcrossSpaces(t *testing.T) {
	ctx := context.Background()
	bank := memory.NewMemoryBankWithStore(memory.NewInMemoryStore())
	mem := memory.NewSessionMemory(bank, 8).WithEmbedder(memory.DummyEmbedder{})

	mem.Spaces.Grant("team:shared", "agent:alpha", memory.SpaceRoleWriter, 0)
	mem.Spaces.Grant("team:shared", "agent:beta", memory.SpaceRoleWriter, 0)

	alphaShared := memory.NewSharedSession(mem, "agent:alpha", "team:shared")
	betaShared := memory.NewSharedSession(mem, "agent:beta", "team:shared")

	alphaAgent, err := New(Options{Model: &stubModel{response: "ok"}, Memory: mem, Shared: alphaShared})
	if err != nil {
		t.Fatalf("alpha agent: %v", err)
	}
	betaAgent, err := New(Options{Model: &stubModel{response: "ok"}, Memory: mem, Shared: betaShared})
	if err != nil {
		t.Fatalf("beta agent: %v", err)
	}

	alphaAgent.storeMemory("agent:alpha", "assistant", "Swarm update ready for review", nil)

	records, err := betaShared.Retrieve(ctx, "swarm update", 5)
	if err != nil {
		t.Fatalf("retrieve shared: %v", err)
	}
	found := false
	for _, rec := range records {
		if strings.Contains(rec.Content, "Swarm update ready") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected shared record to be retrievable")
	}

	prompt, err := betaAgent.buildPrompt(ctx, "agent:beta", "Provide the latest swarm plan")
	if err != nil {
		t.Fatalf("build prompt: %v", err)
	}
	if !strings.Contains(prompt, "Swarm update ready for review") {
		t.Fatalf("expected prompt to include shared memory, got: %s", prompt)
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

func TestGenerateWithFilesStoresTextAttachments(t *testing.T) {
	ctx := context.Background()
	bank := memory.NewMemoryBankWithStore(memory.NewInMemoryStore())
	mem := memory.NewSessionMemory(bank, 8).WithEmbedder(memory.DummyEmbedder{})

	agent, err := New(Options{Model: &fileEchoModel{response: "ok"}, Memory: mem})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	files := []models.File{{Name: "notes.txt", MIME: "text/plain", Data: []byte("alpha beta")}}
	if _, err := agent.GenerateWithFiles(ctx, "session", "summarize the attachment", files); err != nil {
		t.Fatalf("GenerateWithFiles returned error: %v", err)
	}

	records, err := agent.SessionMemory().RetrieveContext(ctx, "session", "", 5)
	if err != nil {
		t.Fatalf("RetrieveContext returned error: %v", err)
	}

	found := false
	for _, rec := range records {
		if metadataRole(rec.Metadata) != "attachment" {
			continue
		}
		if !strings.Contains(rec.Content, "Attachment notes.txt") {
			t.Fatalf("expected attachment name in memory, got %q", rec.Content)
		}
		if !strings.Contains(rec.Content, "alpha beta") {
			t.Fatalf("expected attachment content in memory, got %q", rec.Content)
		}
		payload := map[string]any{}
		if err := json.Unmarshal([]byte(rec.Metadata), &payload); err != nil {
			t.Fatalf("unmarshal metadata: %v", err)
		}
		if got := payload["filename"]; got != "notes.txt" {
			t.Fatalf("expected filename metadata, got %v", got)
		}
		if got := payload["text"]; got != "true" {
			t.Fatalf("expected text flag true, got %v", got)
		}
		wantB64 := base64.StdEncoding.EncodeToString([]byte("alpha beta"))
		if got := payload["data_base64"]; got != wantB64 {
			t.Fatalf("expected base64 payload, got %v", got)
		}
		found = true
	}
	if !found {
		t.Fatalf("expected attachment memory to be stored")
	}
}

func TestGenerateWithFilesStoresNonTextAttachments(t *testing.T) {
	ctx := context.Background()
	bank := memory.NewMemoryBankWithStore(memory.NewInMemoryStore())
	mem := memory.NewSessionMemory(bank, 8).WithEmbedder(memory.DummyEmbedder{})

	agent, err := New(Options{Model: &fileEchoModel{response: "ok"}, Memory: mem})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	files := []models.File{{Name: "diagram.png", MIME: "image/png", Data: []byte{0x89, 0x50, 0x4E, 0x47}}}
	if _, err := agent.GenerateWithFiles(ctx, "session", "summarize the attachment", files); err != nil {
		t.Fatalf("GenerateWithFiles returned error: %v", err)
	}

	records, err := agent.SessionMemory().RetrieveContext(ctx, "session", "", 5)
	if err != nil {
		t.Fatalf("RetrieveContext returned error: %v", err)
	}

	found := false
	for _, rec := range records {
		if metadataRole(rec.Metadata) != "attachment" {
			continue
		}
		if !strings.Contains(rec.Content, "non-text content") {
			t.Fatalf("expected placeholder for non-text attachment, got %q", rec.Content)
		}
		payload := map[string]any{}
		if err := json.Unmarshal([]byte(rec.Metadata), &payload); err != nil {
			t.Fatalf("unmarshal metadata: %v", err)
		}
		if got := payload["text"]; got != "false" {
			t.Fatalf("expected text flag false, got %v", got)
		}
		if got := payload["mime"]; got != "image/png" {
			t.Fatalf("expected mime metadata, got %v", got)
		}
		wantB64 := base64.StdEncoding.EncodeToString([]byte{0x89, 0x50, 0x4E, 0x47})
		if got := payload["data_base64"]; got != wantB64 {
			t.Fatalf("expected base64 metadata, got %v", got)
		}
		found = true
	}
	if !found {
		t.Fatalf("expected attachment memory to be stored for non-text file")
	}
}

func TestRetrieveAttachmentFilesReturnsBinaryData(t *testing.T) {
	ctx := context.Background()
	bank := memory.NewMemoryBankWithStore(memory.NewInMemoryStore())
	mem := memory.NewSessionMemory(bank, 8).WithEmbedder(memory.DummyEmbedder{})

	agent, err := New(Options{Model: &fileEchoModel{response: "ok"}, Memory: mem})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	files := []models.File{
		{Name: "diagram.png", MIME: "image/png", Data: []byte{0x89, 0x50, 0x4E, 0x47}},
		{Name: "clip.mp4", MIME: "video/mp4", Data: []byte{0x00, 0x01, 0x02}},
	}

	if _, err := agent.GenerateWithFiles(ctx, "session", "describe media", files); err != nil {
		t.Fatalf("GenerateWithFiles returned error: %v", err)
	}

	retrieved, err := agent.RetrieveAttachmentFiles(ctx, "session", 10)
	if err != nil {
		t.Fatalf("RetrieveAttachmentFiles returned error: %v", err)
	}

	if len(retrieved) != len(files) {
		t.Fatalf("expected %d attachments, got %d", len(files), len(retrieved))
	}

	for i, file := range retrieved {
		want := files[i]
		if file.Name != want.Name {
			t.Fatalf("attachment %d: expected name %q, got %q", i, want.Name, file.Name)
		}
		if file.MIME != want.MIME {
			t.Fatalf("attachment %d: expected MIME %q, got %q", i, want.MIME, file.MIME)
		}
		if string(file.Data) != string(want.Data) {
			t.Fatalf("attachment %d: expected data %v, got %v", i, want.Data, file.Data)
		}
	}
}
