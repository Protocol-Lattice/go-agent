package models

import (
	"context"
	"errors"
	"fmt"
	"os"

	genai "github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// ---------------------------- Google Gemini ----------------------------------

type GeminiLLM struct {
	Client *genai.Client
	Model  string
}

func NewGeminiLLM(ctx context.Context, model string) (*GeminiLLM, error) {
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
	return &GeminiLLM{Client: client, Model: model}, nil
}

func (g *GeminiLLM) Generate(ctx context.Context, prompt string) (any, error) {
	model := g.Client.GenerativeModel(g.Model)
	resp, err := model.GenerateContent(ctx, genai.Text(prompt))
	if err != nil {
		return nil, err
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return nil, errors.New("gemini: empty response")
	}
	return resp.Candidates[0].Content.Parts[0], nil
}
