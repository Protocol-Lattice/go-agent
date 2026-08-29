package agent

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

var errToolPlannerTest = errors.New("tool planner unavailable")
var errToolStreamTest = errors.New("tool stream interrupted")

type failingToolStream struct {
	closed bool
}

func (s *failingToolStream) Next() (any, error) {
	return nil, errToolStreamTest
}

func (s *failingToolStream) Close() error {
	s.closed = true
	return nil
}

type toolPlannerErrorModel struct {
	plannerCalls  int
	fallbackCalls int
	streamCalls   int
}

type invalidToolPlannerModel struct {
	plannerCalls  int
	fallbackCalls int
}

type mutationPlannerRecoveryModel struct {
	plannerCalls int
	prompts      []string
}

func (m *mutationPlannerRecoveryModel) Generate(_ context.Context, prompt string) (any, error) {
	m.prompts = append(m.prompts, prompt)
	if !strings.Contains(prompt, "You are an agentic UTCP tool execution loop") {
		return "unexpected fallback", nil
	}
	m.plannerCalls++
	switch m.plannerCalls {
	case 1:
		return "I will write the file now, then verify it.", nil
	case 2:
		return "Corrected plan:\n```json\n" +
			`{"use_tool":true,"tool_name":"filesystem.write","arguments":"{\"input\":\"fixed\"}"}` +
			"\n```", nil
	default:
		return `{"use_tool":false,"final_answer":"fixed"}`, nil
	}
}

func (m *mutationPlannerRecoveryModel) GenerateWithFiles(ctx context.Context, prompt string, _ []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *mutationPlannerRecoveryModel) GenerateStream(context.Context, string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk)
	close(ch)
	return ch, nil
}

func (m *invalidToolPlannerModel) Generate(_ context.Context, prompt string) (any, error) {
	if strings.Contains(prompt, "You are an agentic UTCP tool execution loop") {
		m.plannerCalls++
		return "this is not JSON", nil
	}
	m.fallbackCalls++
	return "unexpected fallback", nil
}

func (m *invalidToolPlannerModel) GenerateWithFiles(ctx context.Context, prompt string, _ []models.File) (any, error) {
	return m.Generate(ctx, prompt)
}

func (m *invalidToolPlannerModel) GenerateStream(context.Context, string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk)
	close(ch)
	return ch, nil
}

func (m *toolPlannerErrorModel) Generate(_ context.Context, prompt string) (any, error) {
	if strings.Contains(prompt, "You are an agentic UTCP tool execution loop") {
		m.plannerCalls++
		return nil, errToolPlannerTest
	}
	m.fallbackCalls++
	return "unexpected fallback", nil
}

func (m *toolPlannerErrorModel) GenerateWithFiles(_ context.Context, prompt string, _ []models.File) (any, error) {
	if strings.Contains(prompt, "You are an agentic UTCP tool execution loop") {
		m.plannerCalls++
		return nil, errToolPlannerTest
	}
	m.fallbackCalls++
	return "unexpected fallback", nil
}

func (m *toolPlannerErrorModel) GenerateStream(context.Context, string) (<-chan models.StreamChunk, error) {
	m.streamCalls++
	ch := make(chan models.StreamChunk, 1)
	ch <- models.StreamChunk{Delta: "unexpected fallback", FullText: "unexpected fallback", Done: true}
	close(ch)
	return ch, nil
}

func newToolPlannerErrorAgent(t *testing.T) (*Agent, *toolPlannerErrorModel) {
	t.Helper()

	model := &toolPlannerErrorModel{}
	a, err := New(Options{
		Model:  model,
		Memory: memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		Tools: []Tool{&stubTool{spec: ToolSpec{
			Name:        "echo",
			Description: "Echoes input",
		}}},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return a, model
}

func TestGeneratePropagatesToolPlannerError(t *testing.T) {
	a, model := newToolPlannerErrorAgent(t)

	_, err := a.Generate(context.Background(), "session", "Run the echo tool with hello.")
	if !errors.Is(err, errToolPlannerTest) {
		t.Fatalf("Generate error = %v, want %v", err, errToolPlannerTest)
	}
	if model.plannerCalls != 1 {
		t.Fatalf("planner calls = %d, want 1", model.plannerCalls)
	}
	if model.fallbackCalls != 0 {
		t.Fatalf("fallback calls = %d, want 0", model.fallbackCalls)
	}
}

func TestGenerateWithFilesPropagatesToolPlannerError(t *testing.T) {
	a, model := newToolPlannerErrorAgent(t)

	_, err := a.GenerateWithFiles(
		context.Background(),
		"session",
		"Inspect this file with the available tools.",
		[]models.File{{Name: "main.go", MIME: "text/plain", Data: []byte("package main")}},
	)
	if !errors.Is(err, errToolPlannerTest) {
		t.Fatalf("GenerateWithFiles error = %v, want %v", err, errToolPlannerTest)
	}
	if model.plannerCalls != 1 {
		t.Fatalf("planner calls = %d, want 1", model.plannerCalls)
	}
	if model.fallbackCalls != 0 {
		t.Fatalf("fallback calls = %d, want 0", model.fallbackCalls)
	}
}

func TestGenerateStreamPropagatesToolPlannerError(t *testing.T) {
	a, model := newToolPlannerErrorAgent(t)

	stream, err := a.GenerateStream(context.Background(), "session", "Run the echo tool with hello.")
	if !errors.Is(err, errToolPlannerTest) {
		t.Fatalf("GenerateStream error = %v, want %v", err, errToolPlannerTest)
	}
	if stream != nil {
		t.Fatal("GenerateStream returned a stream after the tool planner failed")
	}
	if model.plannerCalls != 1 {
		t.Fatalf("planner calls = %d, want 1", model.plannerCalls)
	}
	if model.streamCalls != 0 {
		t.Fatalf("fallback stream calls = %d, want 0", model.streamCalls)
	}
}

func TestGenerateLimitsInvalidToolPlannerResponses(t *testing.T) {
	model := &invalidToolPlannerModel{}
	a, err := New(Options{
		Model:  model,
		Memory: memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		Tools: []Tool{&stubTool{spec: ToolSpec{
			Name:        "echo",
			Description: "Echoes input",
		}}},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = a.Generate(context.Background(), "session", "Run the echo tool with hello.")
	if err == nil || !strings.Contains(err.Error(), "invalid JSON 2 consecutive times") {
		t.Fatalf("Generate error = %v, want invalid JSON retry-limit error", err)
	}
	if model.plannerCalls != defaultPlannerInvalidJSONLimit {
		t.Fatalf("planner calls = %d, want %d", model.plannerCalls, defaultPlannerInvalidJSONLimit)
	}
	if model.fallbackCalls != 0 {
		t.Fatalf("fallback calls = %d, want 0", model.fallbackCalls)
	}
}

func TestMutationPlannerRecoversFromInvalidJSON(t *testing.T) {
	model := &mutationPlannerRecoveryModel{}
	writeTool := &stubTool{spec: ToolSpec{
		Name:        "filesystem.write",
		Description: "Write file content",
	}}
	a, err := New(Options{
		Model:  model,
		Memory: memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		Tools:  []Tool{writeTool},
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	out, err := a.Generate(context.Background(), "session", "Fix the file.")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if got, want := out, any("fixed"); got != want {
		t.Fatalf("Generate output = %v, want %v", got, want)
	}
	if model.plannerCalls != 3 {
		t.Fatalf("planner calls = %d, want 3", model.plannerCalls)
	}
	if len(model.prompts) < 2 {
		t.Fatalf("planner prompts = %d, want at least 2", len(model.prompts))
	}
	retryPrompt := model.prompts[1]
	if !strings.Contains(retryPrompt, "PREVIOUS PLANNER RESPONSE") ||
		!strings.Contains(retryPrompt, "I will write the file now") {
		t.Fatalf("retry prompt does not contain the previous invalid response:\n%s", retryPrompt)
	}
	if got := writeTool.lastInput.Arguments["input"]; got != "fixed" {
		t.Fatalf("write tool input = %v, want fixed", got)
	}
}

func TestExecuteToolPropagatesStreamReadError(t *testing.T) {
	stream := &failingToolStream{}
	client := &stubUTCPClient{fakeStream: stream}
	a, err := New(Options{
		Model:      &stubModel{response: "unused"},
		Memory:     memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		UTCPClient: client,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = a.executeTool(context.Background(), "session", "stream.echo", map[string]any{"stream": true})
	if !errors.Is(err, errToolStreamTest) {
		t.Fatalf("executeTool error = %v, want %v", err, errToolStreamTest)
	}
	if !stream.closed {
		t.Fatal("executeTool did not close the failed stream")
	}
}

func TestObservedUTCPClientRejectsNilStream(t *testing.T) {
	client := NewObservedUTCPClient(&stubUTCPClient{})

	stream, err := client.CallToolStream(context.Background(), "stream.echo", nil)
	if err == nil || !strings.Contains(err.Error(), "nil stream") {
		t.Fatalf("CallToolStream error = %v, want nil stream error", err)
	}
	if stream != nil {
		t.Fatal("CallToolStream returned a non-nil wrapper around a nil stream")
	}
}
