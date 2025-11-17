package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"strings"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/chain"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/base"
	"github.com/universal-tool-calling-protocol/go-utcp/src/repository"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
	utcpTools "github.com/universal-tool-calling-protocol/go-utcp/src/tools"
	"github.com/universal-tool-calling-protocol/go-utcp/src/transports"
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

type stubUTCPClient struct {
	callCount       int
	lastToolName    string
	searchTools     []utcpTools.Tool
	lastSearchQuery string
	lastSearchLimit int
	fakeStream      *FakeStream
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

func (c *stubUTCPClient) CallTool(ctx context.Context, toolName string, args map[string]any) (any, error) {
	c.callCount++
	c.lastToolName = toolName
	return "utcp says " + toolName, nil
}

func (c *stubUTCPClient) SearchTools(query string, limit int) ([]utcpTools.Tool, error) {
	c.lastSearchQuery = query
	c.lastSearchLimit = limit
	return c.searchTools, nil
}

func (c *stubUTCPClient) DeregisterToolProvider(ctx context.Context, name string) error {
	return nil
}

func (c *stubUTCPClientV2) DeregisterToolProvider(ctx context.Context, name string) error {
	return nil
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

func (u *stubUTCPClient) GetTransports() map[string]repository.ClientTransport {
	return nil
}
func (u *stubUTCPClientV2) GetTransports() map[string]repository.ClientTransport {
	return nil
}

func TestAgentCallsUTCPClientForRemoteTools(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	utcpClient := &stubUTCPClient{}

	agent, err := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcpClient,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	// Execute a tool that does not exist locally.
	_, err = agent.executeTool(ctx, "session1", "remote_tool", nil)
	if err != nil {
		t.Fatalf("executeTool returned an unexpected error: %v", err)
	}

	if utcpClient.callCount != 1 {
		t.Fatalf("expected UTCP client to be called once, got %d", utcpClient.callCount)
	}
	if utcpClient.lastToolName != "remote_tool" {
		t.Fatalf("expected UTCP client to be called with 'remote_tool', got %q", utcpClient.lastToolName)
	}
}

func (u *stubUTCPClient) RegisterToolProvider(ctx context.Context, prov base.Provider) ([]tools.Tool, error) {
	return nil, nil
}

func (u *stubUTCPClientV2) RegisterToolProvider(ctx context.Context, prov base.Provider) ([]tools.Tool, error) {
	return nil, nil
}

func TestAgentMergesUTCPSearchResults(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{
		response: `{
			"use_tool": true,
			"tool_name": "utcp_tool",
			"arguments": {"input": "test"},
			"reason": "testing utcp search"
		}`,
	}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	utcpClient := &stubUTCPClient{
		searchTools: []utcpTools.Tool{
			{Name: "utcp_tool", Description: "A tool from UTCP search"},
		},
	}

	agent, err := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcpClient,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	// This will trigger the tool orchestrator, which calls ToolSpecs, which calls SearchTools.
	_, err = agent.Generate(ctx, "session1", "use the utcp tool")
	if err != nil {
		t.Fatalf("Generate returned an unexpected error: %v", err)
	}

	// Verify SearchTools was called by ToolSpecs()
	if utcpClient.lastSearchQuery != "" || utcpClient.lastSearchLimit != 50 {
		t.Fatalf("expected SearchTools to be called with ('', 50), got (%q, %d)", utcpClient.lastSearchQuery, utcpClient.lastSearchLimit)
	}

	if utcpClient.callCount != 1 {
		t.Fatalf("expected UTCP client's CallTool to be called once, got %d", utcpClient.callCount)
	}
	if utcpClient.lastToolName != "utcp_tool" {
		t.Fatalf("expected UTCP client to be called with 'utcp_tool', got %q", utcpClient.lastToolName)
	}
}

func TestCodeMode_ExecutesCallToolInsideDSL(t *testing.T) {
	ctx := context.Background()

	// stub model instructs to run CodeMode
	model := &stubModel{
		response: `{"use_tool": true, "tool_name": "codemode.run_code", "arguments": { "code": "codemode.CallTool(\"echo\", map[string]any{\"input\": \"hi\"})" }}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)

	utcpClient := &stubUTCPClient{}

	agent, err := New(Options{
		Model:    model,
		Memory:   mem,
		CodeMode: codemode.NewCodeModeUTCP(utcpClient),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	// The tool orchestrator will call `codemode.run_code`, which is not a real UTCP tool,
	// so the first call won't go through the stub. The second call inside the code
	// will. To simulate the orchestrator "calling" the tool, we can just check the end state.
	// The logic in `toolOrchestrator` handles `codemode.run_code` as a special case.

	out, err := agent.Generate(ctx, "session1", "run code")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	log.Println(out)

	// CodeMode should return raw output
	if out == "" {
		t.Fatalf("expected non-empty output")
	}

	if utcpClient.callCount != 1 {
		t.Fatalf("expected 1 UTCP call from inside the DSL, got %d", utcpClient.callCount)
	}

	if utcpClient.lastToolName != "echo" {
		t.Fatalf("expected last tool to be 'echo', got %q", utcpClient.lastToolName)
	}
}

func TestCodeMode_ExecutesCallToolStreamInsideDSL(t *testing.T) {
	ctx := context.Background()

	model := &stubModel{
		response: `{"use_tool": true, "tool_name": "codemode.run_code", "arguments": { "code": "s, _ := codemode.CallToolStream(\"stream.echo\", map[string]any{\"input\": \"x\"}); __out, _ = s.Next()" }}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)

	// stub stream
	stream := &FakeStream{
		chunks: []any{"A", "B", nil},
		index:  0,
	}

	utcpClient := &stubUTCPClient{}
	utcpClient.fakeStream = stream

	agent, err := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcpClient,
		CodeMode:   codemode.NewCodeModeUTCP(utcpClient),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = agent.Generate(ctx, "s1", "run code")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if utcpClient.lastToolName != "stream.echo" {
		t.Fatalf("expected streaming tool to run")
	}
}

type FakeStream struct {
	chunks []any
	index  int
}

func (s *FakeStream) Next() (any, error) {
	if s.index >= len(s.chunks) {
		return nil, io.EOF
	}

	ch := s.chunks[s.index]
	s.index++

	// ❗ IMPORTANT: nil chunk must terminate the stream
	if ch == nil {
		return nil, io.EOF
	}

	return ch, nil
}

func (s *FakeStream) Close() error { return nil }

func (c *stubUTCPClient) CallToolStream(ctx context.Context, name string, args map[string]any) (transports.StreamResult, error) {
	c.callCount++
	c.lastToolName = name
	return c.fakeStream, nil

}
func TestCodeMode_StoresToonMemory(t *testing.T) {
	ctx := context.Background()

	model := &stubModel{
		response: `{"use_tool": true, "tool_name": "codemode.run_code", "arguments": { "code": "1+1" }}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)
	mem = mem.WithEmbedder(memory.DummyEmbedder{})

	utcpClient := &stubUTCPClient{}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcpClient,
		CodeMode:   codemode.NewCodeModeUTCP(utcpClient),
	})

	_, _ = agent.Generate(ctx, "sess", "run code")

	recs, _ := mem.RetrieveContext(ctx, "sess", "", 10)
	found := false
	for _, r := range recs {
		if strings.Contains(r.Content, ".toon:") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected CodeMode output to be stored with TOON")
	}
}

func TestCodeMode_ComplexLogicAndToolChain(t *testing.T) {
	ctx := context.Background()

	// This Go code snippet iterates three times, calling the "echo" tool in each loop.
	codeSnippet := `		
		var out any 
		for i := 0; i < 3; i++ {
			out, _ = codemode.CallTool("echo", map[string]any{"input": "ping"})
		}
		__out = out
	`

	model := &stubModel{
		response: fmt.Sprintf(`{"use_tool": true, "tool_name": "codemode.run_code", "arguments": { "code": %q }}`, codeSnippet),
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)
	utcpClient := &stubUTCPClient{}

	agent, err := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcpClient,
		CodeMode:   codemode.NewCodeModeUTCP(utcpClient),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = agent.Generate(ctx, "session1", "run complex code")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	if utcpClient.callCount != 3 {
		t.Fatalf("expected 3 UTCP calls from inside the DSL loop, got %d", utcpClient.callCount)
	}
}

func TestGenerate_ExecutesUTCPCalledTool(t *testing.T) {
	ctx := context.Background()

	// LLM returns JSON prompting the tool orchestrator to use the UTCP tool.
	model := &stubModel{
		response: `{
			"use_tool": true,
			"tool_name": "echo",
			"arguments": { "input": "hi" }
		}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4).WithEmbedder(memory.DummyEmbedder{})

	utcpClient := &stubUTCPClient{
		searchTools: []utcpTools.Tool{
			{Name: "echo", Description: "echo test"},
		},
	}

	agent, err := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcpClient,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	out, err := agent.Generate(ctx, "s1", "hello")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	// Expect UTCP tool call to happen
	if utcpClient.callCount != 1 {
		t.Fatalf("expected UTCP client CallTool to be called once, got %d", utcpClient.callCount)
	}
	if utcpClient.lastToolName != "echo" {
		t.Fatalf("expected UTCP tool 'echo', got %q", utcpClient.lastToolName)
	}

	// Returned output should be raw non-TOON tool result
	if out != "utcp says echo" {
		t.Fatalf("unexpected output: %q", out)
	}

	// Verify TOON memory stored
	recs, _ := mem.RetrieveContext(ctx, "s1", "", 10)
	foundToon := false
	for _, r := range recs {
		if strings.Contains(r.Content, ".toon:") {
			foundToon = true
			break
		}
	}
	if !foundToon {
		t.Fatalf("expected TOON-encoded assistant memory")
	}
}

func TestDirectJsonToolInvocationCallsUTCP(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{response: "ignored"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	utcp := &stubUTCPClient{}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcp,
	})

	_, err := agent.Generate(ctx, "s1", `{
		"tool": "echo",
		"arguments": { "input": "hi" }
	}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if utcp.callCount != 1 {
		t.Fatalf("expected UTCP CallTool once, got %d", utcp.callCount)
	}
	if utcp.lastToolName != "echo" {
		t.Fatalf("expected UTCP call to echo, got %q", utcp.lastToolName)
	}
}
func TestDSLToolInvocationCallsUTCP(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{response: "ignored"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	utcp := &stubUTCPClient{}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcp,
	})

	_, err := agent.Generate(ctx, "s1", `tool: echo {"input":"hi"}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if utcp.callCount != 1 {
		t.Fatalf("expected UTCP CallTool once")
	}
	if utcp.lastToolName != "echo" {
		t.Fatalf("expected echo, got %q", utcp.lastToolName)
	}
}

func TestShorthandInvocationCallsUTCP(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	utcp := &stubUTCPClient{}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcp,
	})

	_, _ = agent.Generate(ctx, "s1", `echo {"input":"hi"}`)

	if utcp.callCount != 1 {
		t.Fatalf("expected UTCP CallTool once")
	}
	if utcp.lastToolName != "echo" {
		t.Fatalf("expected tool echo, got %q", utcp.lastToolName)
	}
}

func TestDirectJsonStreamInvocationCallsUTCPStream(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	stream := &FakeStream{
		chunks: []any{"chunk1", nil},
	}
	utcp := &stubUTCPClient{fakeStream: stream}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcp,
	})

	_, err := agent.Generate(ctx, "s1", `{
		"tool": "stream.echo",
		"arguments": { "input": "x", "stream": true }
	}`)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if utcp.lastToolName != "stream.echo" {
		t.Fatalf("expected stream.echo, got %q", utcp.lastToolName)
	}
	if utcp.callCount != 1 {
		t.Fatalf("expected one stream call, got %d", utcp.callCount)
	}
}

func TestToolOrchestratorSelectsStreamingUTCPTool(t *testing.T) {
	ctx := context.Background()

	model := &stubModel{
		response: `{
			"use_tool": true,
			"tool_name": "stream.echo",
			"arguments": { "input": "abc", "stream": true },
			"stream": true
		}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)

	stream := &FakeStream{chunks: []any{"A", "B", nil}}
	utcp := &stubUTCPClient{
		fakeStream: stream,
		searchTools: []utcpTools.Tool{
			{Name: "stream.echo", Description: "streaming echo"},
		},
	}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcp,
	})

	_, err := agent.Generate(ctx, "s1", "stream something")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if utcp.lastToolName != "stream.echo" {
		t.Fatalf("expected streaming tool echo, got %q", utcp.lastToolName)
	}
}

func TestToolSpecsMergesAllSources(t *testing.T) {
	model := &stubModel{}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	utcp := &stubUTCPClient{
		searchTools: []utcpTools.Tool{
			{Name: "remote.echo"},
		},
	}

	local := &stubTool{spec: ToolSpec{Name: "local.echo"}}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		Tools:      []Tool{local},
		UTCPClient: utcp,
		CodeMode:   codemode.NewCodeModeUTCP(utcp),
	})

	tools := agent.ToolSpecs()

	var names []string
	for _, t := range tools {
		names = append(names, t.Name)
	}

	wants := []string{"local.echo", "remote.echo", "codemode.run_code"}
	for _, want := range wants {
		found := false
		for _, got := range names {
			if got == want {
				found = true
				break
			}
		}
		if !found {
			t.Fatalf("expected tool %q in ToolSpecs, got %v", want, names)
		}
	}
}
func TestCodeMode_SimpleExpression(t *testing.T) {
	ctx := context.Background()

	model := &stubModel{
		response: `{"use_tool": true, "tool_name": "codemode.run_code",
            "arguments": { "code": "1 + 2" }}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)

	utcp := &stubUTCPClient{}

	agent, _ := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcp,
		CodeMode:   codemode.NewCodeModeUTCP(utcp),
	})

	out, err := agent.Generate(ctx, "sess", "run code")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	// Current CodeMode returns a nil result because `1+2` is not wrapped
	if !strings.Contains(out, "3") {
		t.Fatalf("expected nil Codemode result, got %q", out)
	}
}

func TestCodeMode_ExecutesCallToolInsideDSL2(t *testing.T) {
	ctx := context.Background()

	// The LLM response triggers CodeMode
	model := &stubModel{
		response: `{
			"use_tool": true,
			"tool_name": "codemode.run_code",
			"arguments": {
				"code": "codemode.CallTool(\"echo\", map[string]any{\"input\": \"hi\"})"
			}
		}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)

	// Our UTCP stub: tracks calls to CallTool
	utcpClient := &stubUTCPClient{}

	agent, err := New(Options{
		Model:      model,
		Memory:     mem,
		UTCPClient: utcpClient,
		CodeMode:   codemode.NewCodeModeUTCP(utcpClient),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	out, err := agent.Generate(ctx, "session1", "run code")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	// We don’t care about the output — CodeMode returns a struct string.
	if out == "" {
		t.Fatalf("expected non-empty codemode output")
	}

	// This is the key assertion:
	// CodeMode must trigger UTCP CallTool exactly once.
	if utcpClient.callCount != 1 {
		t.Fatalf("expected 1 UTCP call from DSL, got %d", utcpClient.callCount)
	}

	if utcpClient.lastToolName != "echo" {
		t.Fatalf("expected last UTCP tool to be 'echo', got %q", utcpClient.lastToolName)
	}
}

func TestChainOrchestrator_ParsesToolNameAndInputs(t *testing.T) {
	ctx := context.Background()

	model := &stubModel{
		response: `{
			"use_chain": true,
			"steps": [
				{ 
				  "tool_name": "echo", 
				  "inputs": { "input": "alpha" }
				}
			],
			"timeout": 20000
		}`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)
	utcp := &stubUTCPClient{}

	agent, _ := New(Options{
		Model:     model,
		Memory:    mem,
		CodeChain: chain.NewChainModeUTCP(utcp),
	})

	_, err := agent.Generate(ctx, "sess", "run")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if utcp.lastToolName != "echo" {
		t.Fatalf("expected tool 'echo', got %s", utcp.lastToolName)
	}
}

func TestChainOrchestrator_ParsesStreamFlag(t *testing.T) {
	ctx := context.Background()

	stream := &FakeStream{chunks: []any{"A", "B", nil}}
	utcp := &stubUTCPClientV2{fakeStream: stream}

	model := &stubModel{
		response: `{
            "use_chain": true,
            "steps": [
                {
                    "tool_name": "stream.echo",
                    "inputs": { "input": "xyz" },
                    "stream": true
                }
            ],
            "timeout": 20000
        }`,
	}

	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 4)

	agent, _ := New(Options{
		Model:     model,
		Memory:    mem,
		CodeChain: chain.NewChainModeUTCP(utcp),
	})

	_, err := agent.Generate(ctx, "sess3", "run")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if utcp.lastToolName != "stream.echo" {
		t.Fatalf("expected streaming tool to run, got %q", utcp.lastToolName)
	}
	if utcp.callCount != 1 {
		t.Fatalf("expected at least one tool call")
	}
}

func (c *stubUTCPClient) CallToolChain(
	ctx context.Context,
	steps []chain.ChainStep,
	timeout time.Duration,
) (map[string]any, error) {

	var sb strings.Builder
	var last any

	for _, s := range steps {
		c.callCount++
		c.lastToolName = s.ToolName

		// -------------------------------
		// STREAMING STEP
		// -------------------------------
		if s.Stream {
			stream, err := c.CallToolStream(ctx, s.ToolName, s.Inputs)
			if err != nil {
				return nil, err
			}

			for {
				chunk, err := stream.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					return nil, err
				}

				if chunk != nil {
					sb.WriteString(fmt.Sprint(chunk))
					last = chunk
				}
			}
			continue
		}

		// -------------------------------
		// NORMAL STEP
		// -------------------------------
		out, err := c.CallTool(ctx, s.ToolName, s.Inputs)
		if err != nil {
			return nil, err
		}
		if out != nil {
			sb.WriteString(fmt.Sprint(out))
			last = out
		}
	}

	// THIS is the real UTCP chain return shape.
	return map[string]any{
		"output": sb.String(),
		"last":   last,
	}, nil
}

// stubUTCPClient is a simplified UTCP client for testing chain execution.
type stubUTCPClientV2 struct {
	fakeStream   transports.StreamResult
	lastToolName string
	callCount    int
	searchTools  []utcpTools.Tool
}

// --- Normal tool call (non-stream) ---
func (c *stubUTCPClientV2) CallTool(
	ctx context.Context,
	toolName string,
	inputs map[string]any,
) (any, error) {
	c.callCount++
	c.lastToolName = toolName

	// For tests, return simple string = toolName + input
	if v, ok := inputs["input"].(string); ok {
		return v, nil
	}
	return "ok", nil
}

// --- Streaming tool call ---
func (c *stubUTCPClientV2) CallToolStream(
	ctx context.Context,
	toolName string,
	inputs map[string]any,
) (transports.StreamResult, error) {
	c.callCount++
	c.lastToolName = toolName

	// Return the preset fake stream
	return c.fakeStream, nil
}

// --- CHAIN EXECUTION (stubbed but production-shaped) ---
func (c *stubUTCPClientV2) CallToolChain(
	ctx context.Context,
	steps []chain.ChainStep,
	timeout time.Duration,
) (map[string]any, error) {

	results := make(map[string]any, len(steps))
	var lastOutput string

	for i, step := range steps {
		c.callCount++
		c.lastToolName = step.ToolName

		// --- USE_PREVIOUS BEHAVIOR ---
		if step.UsePrevious {
			for k, v := range results {
				if _, exists := step.Inputs[k]; !exists {
					step.Inputs[k] = v
				}
			}
			if lastOutput != "" {
				step.Inputs["__previous_output"] = lastOutput
			}
		}

		var res any
		var err error

		// --- STREAMING STEP ---
		if step.Stream {
			stream, err := c.CallToolStream(ctx, step.ToolName, step.Inputs)
			if err != nil {
				return results, err
			}

			var sb strings.Builder

			for {
				chunk, err := stream.Next()
				if err == io.EOF {
					break
				}
				if err != nil {
					return results, err
				}

				// For stub: chunk is expected to be string or contains Data
				switch v := chunk.(type) {
				case string:
					sb.WriteString(v)
					lastOutput = v
				case map[string]any:
					if data, ok := v["data"]; ok {
						sb.WriteString(fmt.Sprint(data))
						lastOutput = fmt.Sprint(data)
					}
				default:
					sb.WriteString(fmt.Sprint(v))
					lastOutput = fmt.Sprint(v)
				}
			}

			key := step.ID
			if key == "" {
				key = step.ToolName
			}

			results[key] = lastOutput
			continue
		}

		// --- NORMAL STEP ---
		res, err = c.CallTool(ctx, step.ToolName, step.Inputs)
		if err != nil {
			return results, fmt.Errorf("step %d (%s) failed: %w", i+1, step.ToolName, err)
		}

		key := step.ID
		if key == "" {
			key = step.ToolName
		}

		if s, ok := res.(string); ok {
			lastOutput = s
		} else {
			lastOutput = fmt.Sprint(res)
		}

		results[key] = res
	}

	return results, nil
}

func (c *stubUTCPClientV2) SearchTools(query string, limit int) ([]utcpTools.Tool, error) {

	return c.searchTools, nil
}
