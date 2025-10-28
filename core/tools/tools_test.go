package tools

import (
	"context"
	"testing"
	"time"

	agent "github.com/Protocol-Lattice/agent/core/agentic"
)

func TestEchoTool(t *testing.T) {
	tool := &EchoTool{}
	out, err := tool.Invoke(context.Background(), agent.ToolRequest{Arguments: map[string]any{"input": "  hello world  "}})
	if err != nil {
		t.Fatalf("Invoke returned error: %v", err)
	}
	if out.Content != "hello world" {
		t.Fatalf("unexpected output: %q", out.Content)
	}
}

func TestCalculatorTool(t *testing.T) {
	tool := &CalculatorTool{}
	out, err := tool.Invoke(context.Background(), agent.ToolRequest{Arguments: map[string]any{"expression": "21 / 3"}})
	if err != nil {
		t.Fatalf("Invoke returned error: %v", err)
	}
	if out.Content != "7" {
		t.Fatalf("unexpected calculator result: %q", out.Content)
	}
}

func TestCalculatorToolErrors(t *testing.T) {
	tool := &CalculatorTool{}
	if _, err := tool.Invoke(context.Background(), agent.ToolRequest{Arguments: map[string]any{"expression": "bad input"}}); err == nil {
		t.Fatalf("expected format error")
	}
	if _, err := tool.Invoke(context.Background(), agent.ToolRequest{Arguments: map[string]any{"expression": "1 / 0"}}); err == nil {
		t.Fatalf("expected division by zero error")
	}
}

func TestTimeTool(t *testing.T) {
	tool := &TimeTool{}
	out, err := tool.Invoke(context.Background(), agent.ToolRequest{Arguments: map[string]any{}})
	if err != nil {
		t.Fatalf("Invoke returned error: %v", err)
	}
	if _, err := time.Parse(time.RFC3339, out.Content); err != nil {
		t.Fatalf("expected RFC3339 output, got %q: %v", out.Content, err)
	}
}
