package models

import (
	"context"
	"fmt"
	"strings"
)

// DummyLLM is a lightweight model implementation useful for local testing without API calls.
type DummyLLM struct {
	Prefix string
}

func NewDummyLLM(prefix string) *DummyLLM {
	if strings.TrimSpace(prefix) == "" {
		prefix = "Dummy response:"
	}
	return &DummyLLM{Prefix: prefix}
}

func (d *DummyLLM) Generate(_ context.Context, prompt string) (any, error) {
	lines := strings.Split(prompt, "\n")
	var last string
	for i := len(lines) - 1; i >= 0; i-- {
		candidate := strings.TrimSpace(lines[i])
		if candidate != "" {
			last = candidate
			break
		}
	}
	if last == "" {
		last = "<empty prompt>"
	}
	return fmt.Sprintf("%s %s", d.Prefix, last), nil
}

func (d *DummyLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	combined := combinePromptWithFiles(prompt, files)
	// For the dummy, we return the composed prompt directly (prefixed),
	// rather than picking the last non-empty line like Generate().
	return fmt.Sprintf("%s %s", d.Prefix, combined), nil
}

var _ Agent = (*DummyLLM)(nil)
