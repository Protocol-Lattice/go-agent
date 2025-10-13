package tools

import (
	"context"
	"testing"
	"time"
)

func TestEchoTool(t *testing.T) {
	tool := &EchoTool{}
	out, err := tool.Run(context.Background(), "  hello world  ")
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if out != "hello world" {
		t.Fatalf("unexpected output: %q", out)
	}
}

func TestCalculatorTool(t *testing.T) {
	tool := &CalculatorTool{}
	out, err := tool.Run(context.Background(), "21 / 3")
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if out != "7" {
		t.Fatalf("unexpected calculator result: %q", out)
	}
}

func TestCalculatorToolErrors(t *testing.T) {
	tool := &CalculatorTool{}
	if _, err := tool.Run(context.Background(), "bad input"); err == nil {
		t.Fatalf("expected format error")
	}
	if _, err := tool.Run(context.Background(), "1 / 0"); err == nil {
		t.Fatalf("expected division by zero error")
	}
}

func TestTimeTool(t *testing.T) {
	tool := &TimeTool{}
	out, err := tool.Run(context.Background(), "")
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if _, err := time.Parse(time.RFC3339, out); err != nil {
		t.Fatalf("expected RFC3339 output, got %q: %v", out, err)
	}
}
