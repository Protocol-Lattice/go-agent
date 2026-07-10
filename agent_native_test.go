package agent

import (
	"context"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	utcpTools "github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

type nativeToolModel struct {
	responses []models.ToolCallResponse
	calls     int
	prompts   []string
	tools     []models.ToolDefinition
}

func (m *nativeToolModel) Generate(context.Context, string) (any, error) {
	return "fallback", nil
}

func (m *nativeToolModel) GenerateWithFiles(context.Context, string, []models.File) (any, error) {
	return "fallback", nil
}

func (m *nativeToolModel) GenerateStream(context.Context, string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk, 1)
	ch <- models.StreamChunk{Delta: "fallback", FullText: "fallback", Done: true}
	close(ch)
	return ch, nil
}

func (m *nativeToolModel) GenerateWithTools(_ context.Context, prompt string, tools []models.ToolDefinition) (models.ToolCallResponse, error) {
	m.calls++
	m.prompts = append(m.prompts, prompt)
	if len(m.tools) == 0 {
		m.tools = append([]models.ToolDefinition(nil), tools...)
	}
	response := m.responses[0]
	m.responses = m.responses[1:]
	return response, nil
}

func TestAgentUsesNativeToolCallsWhenModelSupportsThem(t *testing.T) {
	model := &nativeToolModel{
		responses: []models.ToolCallResponse{
			{ToolCalls: []models.ToolCall{{Name: "echo", Arguments: map[string]any{"input": "hello"}}}},
			{Content: "finished"},
		},
	}
	localTool := &stubTool{spec: ToolSpec{
		Name:        "echo",
		Description: "Echoes input",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{"input": map[string]any{"type": "string"}},
			"required":   []string{"input"},
		},
	}}

	a, err := New(Options{
		Model:  model,
		Memory: memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		Tools:  []Tool{localTool},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	out, err := a.Generate(context.Background(), "session", "run echo")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if out != "finished" {
		t.Fatalf("Generate returned %q, want %q", out, "finished")
	}
	if model.calls != 2 {
		t.Fatalf("native model called %d times, want 2", model.calls)
	}
	if len(model.tools) != 1 || model.tools[0].Name != "echo" {
		t.Fatalf("native model received unexpected tools: %#v", model.tools)
	}
	if model.tools[0].InputSchema["type"] != "object" {
		t.Fatalf("native model received malformed schema: %#v", model.tools[0].InputSchema)
	}
	if got := localTool.lastInput.Arguments["input"]; got != "hello" {
		t.Fatalf("tool received input %v, want hello", got)
	}
}

func TestNativeToolDefinitionsDefaultSchemaType(t *testing.T) {
	definitions := nativeToolDefinitions([]utcpTools.Tool{{Name: "empty"}})
	if len(definitions) != 1 {
		t.Fatalf("got %d definitions, want 1", len(definitions))
	}
	if got := definitions[0].InputSchema["type"]; got != "object" {
		t.Fatalf("schema type = %v, want object", got)
	}
}
