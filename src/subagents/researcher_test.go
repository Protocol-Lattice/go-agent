package subagents

import (
	"context"
	"strings"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

type fakeModel struct {
	response any
	err      error
	prompts  []string
}

func (g *fakeModel) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return nil, nil
}

func (f *fakeModel) Generate(ctx context.Context, prompt string) (any, error) {
	f.prompts = append(f.prompts, prompt)
	if f.err != nil {
		return nil, f.err
	}
	if f.response != nil {
		return f.response, nil
	}
	return "ok", nil
}

func (f *fakeModel) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	ch := make(chan models.StreamChunk, 1)
	ch <- models.StreamChunk{Delta: "ok", FullText: "ok", Done: true}
	close(ch)
	return ch, nil
}

func TestResearcherRunIncludesPersonaAndTask(t *testing.T) {
	fm := &fakeModel{response: "result"}
	researcher := NewResearcher(fm)

	out, err := researcher.Run(context.Background(), "Investigate topic")
	if err != nil {
		t.Fatalf("Run returned error: %v", err)
	}
	if out != "result" {
		t.Fatalf("unexpected result: %q", out)
	}
	if len(fm.prompts) != 1 {
		t.Fatalf("expected one prompt to be generated")
	}
	prompt := fm.prompts[0]
	if !strings.Contains(prompt, researcher.persona) {
		t.Fatalf("prompt missing persona text: %q", prompt)
	}
	if !strings.Contains(prompt, "Investigate topic") {
		t.Fatalf("prompt missing task text: %q", prompt)
	}
}

func TestResearcherRunRequiresModel(t *testing.T) {
	researcher := &Researcher{}
	if _, err := researcher.Run(context.Background(), "anything"); err == nil {
		t.Fatalf("expected error when model is missing")
	}
}
