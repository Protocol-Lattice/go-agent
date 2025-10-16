package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"strings"
	"time"

	ollama "github.com/ollama/ollama/api" // <- correct import
)

// ---------------------------- Ollama -----------------------------------------

type OllamaLLM struct {
	Client       *ollama.Client
	Model        string
	PromptPrefix string
}

func NewOllamaLLM(model string, promptPrefix string) (*OllamaLLM, error) {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434"
	}

	u, err := url.Parse(host)
	if err != nil {
		return nil, fmt.Errorf("invalid OLLAMA_HOST %q: %w", host, err)
	}

	httpClient := &http.Client{
		Timeout: 60 * time.Second,
		// (Optional) Transport tweaks, TLS settings, proxy, etc.
	}

	c := ollama.NewClient(u, httpClient)
	return &OllamaLLM{Client: c, Model: model}, nil
}

func (o *OllamaLLM) Generate(ctx context.Context, prompt string) (any, error) {
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = fmt.Sprintf("%s\n\n%s", o.PromptPrefix, prompt)
	}

	var (
		text strings.Builder
		last ollama.GenerateResponse
	)

	req := &ollama.GenerateRequest{
		Model:  o.Model,
		Prompt: fullPrompt,
	}

	if err := o.Client.Generate(ctx, req, func(gr ollama.GenerateResponse) error {
		if gr.Response != "" {
			text.WriteString(gr.Response)
		}
		last = gr
		return nil
	}); err != nil {
		return nil, err
	}

	return struct {
		Text       string
		Done       bool
		DoneReason string
	}{
		Text:       text.String(),
		Done:       last.Done,
		DoneReason: last.DoneReason,
	}, nil
}

func (o *OllamaLLM) UploadFiles(_ context.Context, files []UploadFile) ([]UploadedFile, error) {
	uploads := make([]UploadedFile, 0, len(files))
	for _, file := range files {
		resolved, err := file.resolve()
		if err != nil {
			return nil, err
		}
		data, readErr := io.ReadAll(resolved.reader)
		closeErr := resolved.Close()
		if readErr == nil {
			readErr = closeErr
		}
		if readErr != nil {
			return nil, readErr
		}
		sum := sha256.Sum256(data)
		mimeType := resolved.mimeType
		if mimeType == "" {
			if len(data) > 0 {
				mimeType = http.DetectContentType(data)
			} else {
				mimeType = "application/octet-stream"
			}
		}
		uploads = append(uploads, UploadedFile{
			ID:        "ollama-" + hex.EncodeToString(sum[:]),
			Name:      resolved.name,
			SizeBytes: int64(len(data)),
			MIMEType:  mimeType,
			Provider:  "ollama",
			Purpose:   file.Purpose,
			Data:      data,
		})
	}
	return uploads, nil
}
