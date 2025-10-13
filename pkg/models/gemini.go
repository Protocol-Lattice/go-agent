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

// ---------------------------- Google Gemini ----------------------------------

type GeminiLLM struct {
	Client       *genai.Client
	Model        string
	PromptPrefix string
}

func NewGeminiLLM(ctx context.Context, model, promptPrefix string) (*GeminiLLM, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("missing GOOGLE_API_KEY or GEMINI_API_KEY")
	}
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("gemini init: %w", err)
	}
	return &GeminiLLM{Client: client, Model: model, PromptPrefix: promptPrefix}, nil
}

func (g *GeminiLLM) Generate(ctx context.Context, prompt string) (any, error) {
	model := g.Client.GenerativeModel(g.Model)

	// 🧠 Prepend the role or personality if defined.
	fullPrompt := prompt
	if prefix := strings.TrimSpace(g.PromptPrefix); prefix != "" {
		fullPrompt = fmt.Sprintf("%s %s", prefix, prompt)
	}

	resp, err := model.GenerateContent(ctx, genai.Text(fullPrompt))
	if err != nil {
		return nil, fmt.Errorf("gemini generate: %w", err)
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return nil, errors.New("gemini: empty response")
	}

	return resp.Candidates[0].Content.Parts[0], nil
}
