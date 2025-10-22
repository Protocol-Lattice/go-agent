package models

import (
	"context"
	"fmt"
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
	}

	c := ollama.NewClient(u, httpClient)
	return &OllamaLLM{Client: c, Model: model, PromptPrefix: promptPrefix}, nil
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

func (o *OllamaLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	combined := combinePromptWithFiles(prompt, files)
	return o.Generate(ctx, combined)
}
