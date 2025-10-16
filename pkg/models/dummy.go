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

func (d *DummyLLM) UploadFiles(_ context.Context, files []UploadFile) ([]UploadedFile, error) {
	uploads := make([]UploadedFile, 0, len(files))
	for _, file := range files {
		resolved, err := file.resolve()
		if err != nil {
			return nil, err
		}
		name := resolved.name
		if name == "" {
			name = "upload"
		}
		uploads = append(uploads, UploadedFile{
			ID:        fmt.Sprintf("dummy-%s", name),
			Name:      name,
			SizeBytes: resolved.size,
			MIMEType:  resolved.mimeType,
			Provider:  "dummy",
			Purpose:   file.Purpose,
		})
		resolved.Close()
	}
	return uploads, nil
}

var _ Agent = (*DummyLLM)(nil)
