package models

import (
	"context"
	"strings"
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

// --- New tests for GenerateWithFiles ---

func TestGenerateWithFiles_InlineTextAndNonText(t *testing.T) {
	llm := NewDummyLLM("PFX")
	files := []File{
		{Name: "a.md", MIME: "text/markdown", Data: []byte("# hello\nworld")},
		{Name: "image.png", MIME: "image/png", Data: []byte{0x89, 0x50, 0x4e, 0x47}},
	}

	out, err := llm.GenerateWithFiles(context.Background(), "summarize:", files)
	if err != nil {
		t.Fatalf("GenerateWithFiles returned error: %v", err)
	}

	got := out.(string)

	// Must include the base prompt
	if !strings.Contains(got, "summarize:") {
		t.Fatalf("missing base prompt in output:\n%s", got)
	}

	// Must include attachments section markers
	for _, must := range []string{
		"ATTACHMENTS CONTEXT",
		"<<<FILE a.md",
		"<<<END FILE a.md",
		"image.png", // referenced (non-text)
	} {
		if !strings.Contains(got, must) {
			t.Fatalf("expected output to contain %q; got:\n%s", must, got)
		}
	}
}

func TestGenerateWithFiles_EmptyListFallsBackToPlainPrompt(t *testing.T) {
	llm := NewDummyLLM("PFX")
	out, err := llm.GenerateWithFiles(context.Background(), "just do it", nil)
	if err != nil {
		t.Fatalf("GenerateWithFiles returned error: %v", err)
	}
	got := out.(string)
	if !strings.Contains(got, "just do it") {
		t.Fatalf("missing plain prompt in output: %q", got)
	}
	// Should not inject the attachments banner when there are no files
	if strings.Contains(got, "ATTACHMENTS CONTEXT") {
		t.Fatalf("unexpected attachments banner for empty files: %q", got)
	}
}
