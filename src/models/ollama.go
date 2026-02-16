package models

import (
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
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
	httpClient   *http.Client
	host         string
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
	return &OllamaLLM{
		Client:       c,
		Model:        model,
		PromptPrefix: promptPrefix,
		httpClient:   httpClient,
		host:         host,
	}, nil
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
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = fmt.Sprintf("%s\n\n%s", o.PromptPrefix, prompt)
	}

	// Separate text files from images/videos
	var textFiles []File
	var imageData []ollama.ImageData

	for _, f := range files {
		mt := normalizeMIME(f.Name, f.MIME)

		if isImageOrVideoMIME(mt) {
			// Encode image/video as base64 for Ollama API
			encoded := base64.StdEncoding.EncodeToString(f.Data)
			imageData = append(imageData, ollama.ImageData(encoded))
		} else if isTextMIME(mt) {
			textFiles = append(textFiles, f)
		}
	}

	// Combine text files into the prompt
	if len(textFiles) > 0 {
		fullPrompt = combinePromptWithFiles(fullPrompt, textFiles)
	}

	var (
		text strings.Builder
		last ollama.GenerateResponse
	)

	req := &ollama.GenerateRequest{
		Model:  o.Model,
		Prompt: fullPrompt,
		Images: imageData, // Send images/videos to Ollama
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

// GenerateStream leverages Ollama's native callback-based streaming.
func (o *OllamaLLM) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = fmt.Sprintf("%s\n\n%s", o.PromptPrefix, prompt)
	}

	req := &ollama.GenerateRequest{
		Model:  o.Model,
		Prompt: fullPrompt,
	}

	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)
		var sb strings.Builder
		err := o.Client.Generate(ctx, req, func(gr ollama.GenerateResponse) error {
			if gr.Response != "" {
				sb.WriteString(gr.Response)
				ch <- StreamChunk{Delta: gr.Response}
			}
			return nil
		})
		if err != nil {
			ch <- StreamChunk{Done: true, FullText: sb.String(), Err: err}
			return
		}
		ch <- StreamChunk{Done: true, FullText: sb.String()}
	}()

	return ch, nil
}

// WebSearch queries the Ollama Web Search API and returns top results.
func (o *OllamaLLM) WebSearch(ctx context.Context, query string, limit int) ([]map[string]string, error) {
	endpoint := fmt.Sprintf("%s/api/web_search", strings.TrimRight(o.host, "/"))

	reqBody := map[string]any{"query": query}
	if limit > 0 {
		reqBody["limit"] = limit
	}
	buf := new(bytes.Buffer)
	if err := json.NewEncoder(buf).Encode(reqBody); err != nil {
		return nil, fmt.Errorf("encode request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, buf)
	if err != nil {
		return nil, fmt.Errorf("new request: %w", err)
	}
	req.Header.Set("Content-Type", "application/json")
	if key := os.Getenv("OLLAMA_API_KEY"); key != "" {
		req.Header.Set("Authorization", "Bearer "+key)
	}

	resp, err := o.httpClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode >= 300 {
		return nil, fmt.Errorf("web search failed: %s", resp.Status)
	}

	var data struct {
		Results []map[string]string `json:"results"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&data); err != nil {
		return nil, fmt.Errorf("decode response: %w", err)
	}

	return data.Results, nil
}
