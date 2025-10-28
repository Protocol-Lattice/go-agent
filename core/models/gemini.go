package models

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	genai "github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type GeminiLLM struct {
	Client       *genai.Client
	Model        string
	PromptPrefix string
}

func NewGeminiLLM(ctx context.Context, model string, promptPrefix string) (Agent, error) {
	key := os.Getenv("GOOGLE_API_KEY")
	if key == "" {
		key = os.Getenv("GEMINI_API_KEY")
	}
	if key == "" {
		return nil, errors.New("gemini: missing GOOGLE_API_KEY/GEMINI_API_KEY")
	}
	cl, err := genai.NewClient(ctx, option.WithAPIKey(key))
	if err != nil {
		return nil, err
	}
	return &GeminiLLM{Client: cl, Model: model, PromptPrefix: promptPrefix}, nil
}

func (g *GeminiLLM) Generate(ctx context.Context, prompt string) (any, error) {
	model := g.Client.GenerativeModel(g.Model)
	full := prompt
	if g.PromptPrefix != "" {
		full = g.PromptPrefix + "\n\n" + prompt
	}
	resp, err := model.GenerateContent(ctx, genai.Text(full))
	if err != nil {
		return nil, fmt.Errorf("gemini generate: %w", err)
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("gemini: empty response")
	}
	return resp.Candidates[0].Content.Parts[0], nil
}

// NEW: pass images/videos as parts so Gemini can read them.
// Falls back to text-only if there are no binary attachments.
// gemini.go (inside package models)

func (g *GeminiLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	model := g.Client.GenerativeModel(g.Model)

	// Build normalized copies (never pass raw f.MIME to Gemini)
	norm := make([]File, 0, len(files))
	for _, f := range files {
		normalizedMIME := normalizeMIME(f.Name, f.MIME)
		norm = append(norm, File{
			Name: f.Name,
			MIME: normalizedMIME,
			Data: f.Data,
		})
	}

	// Text context always present
	text := combinePromptWithFiles(prompt, norm)

	var parts []genai.Part
	if p := strings.TrimSpace(g.PromptPrefix); p != "" {
		parts = append(parts, genai.Text(p))
	}
	parts = append(parts, genai.Text(text))

	// Attach only if MIME is sanitized for Gemini
	for _, f := range norm {
		if len(f.Data) == 0 {
			continue
		}
		sanitized := sanitizeForGemini(f.MIME)

		if sanitized == "" {
			continue // skip unsupported/unknown
		}

		if strings.HasPrefix(sanitized, "image/") {
			// ImageData expects just the format (e.g., "png") not "image/png"
			// The SDK prepends "image/" automatically
			format := strings.TrimPrefix(sanitized, "image/")
			parts = append(parts, genai.ImageData(format, f.Data))
		} else if strings.HasPrefix(sanitized, "video/") {
			parts = append(parts, genai.Blob{MIMEType: sanitized, Data: f.Data})
		}
	}

	resp, err := model.GenerateContent(ctx, parts...)
	if err != nil {
		return nil, fmt.Errorf("gemini generateWithFiles: %w", err)
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil || len(resp.Candidates[0].Content.Parts) == 0 {
		return nil, errors.New("gemini: empty response")
	}
	return resp.Candidates[0].Content.Parts[0], nil
}
