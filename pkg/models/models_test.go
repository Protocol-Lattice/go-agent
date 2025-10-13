package models

import (
	"context"
	"testing"
)

func TestNewDummyLLMDefaultPrefix(t *testing.T) {
	llm := NewDummyLLM("")
	resp, err := llm.Generate(context.Background(), "line1\nline2")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if got := resp.(string); got != "Dummy response: line2" {
		t.Fatalf("unexpected response: %q", got)
	}
}

func TestNewDummyLLMUsesLastNonEmptyLine(t *testing.T) {
	llm := NewDummyLLM("Prefix:")
	resp, err := llm.Generate(context.Background(), "first\n\nsecond\n  \nthird")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if got := resp.(string); got != "Prefix: third" {
		t.Fatalf("unexpected response: %q", got)
	}
}

func TestNewLLMProviderErrorsOnUnknownProvider(t *testing.T) {
	if _, err := NewLLMProvider(context.Background(), "unknown", "model", ""); err == nil {
		t.Fatalf("expected error for unknown provider")
	}
}

func TestDummyLLMHandlesEmptyPrompt(t *testing.T) {
	llm := NewDummyLLM("Prefix")
	resp, err := llm.Generate(context.Background(), "\n\n\n")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if got := resp.(string); got != "Prefix <empty prompt>" {
		t.Fatalf("unexpected response: %q", got)
	}
}
