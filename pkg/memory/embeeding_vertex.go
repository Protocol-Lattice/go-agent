package memory

import (
	"context"
	"errors"
	"os"

	genai "github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

type VertexAIEmbedder struct {
	client *genai.Client
	model  *genai.EmbeddingModel
}

func NewVertexAIEmbedder(model string) (Embedder, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("missing GOOGLE_API_KEY or GEMINI_API_KEY")
	}
	cli, err := genai.NewClient(context.Background(), option.WithAPIKey(apiKey))
	if err != nil {
		return nil, err
	}
	if model == "" {
		model = "text-embedding-004"
	}
	return &VertexAIEmbedder{client: cli, model: cli.EmbeddingModel(model)}, nil
}

func (e *VertexAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := e.model.EmbedContent(ctx, genai.Text(text))
	if err != nil {
		return nil, err
	}
	if resp == nil || resp.Embedding == nil || len(resp.Embedding.Values) == 0 {
		return nil, ErrNotSupported
	}
	return resp.Embedding.Values, nil
}
