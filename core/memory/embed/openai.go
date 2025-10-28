package embed

import (
	"context"
	"os"

	openai "github.com/sashabaranov/go-openai"
)

type OpenAIEmbedder struct {
	client *openai.Client
	model  string
}

func NewOpenAIEmbedder(model string) (Embedder, error) {
	key := os.Getenv("OPENAI_API_KEY")
	if key == "" {
		key = os.Getenv("OPENAI_KEY")
	}
	cfg := openai.DefaultConfig(key)
	cli := openai.NewClientWithConfig(cfg)
	if model == "" {
		// Good default in 2025; user may override via ADK_EMBED_MODEL
		model = "text-embedding-3-small"
	}
	return &OpenAIEmbedder{client: cli, model: model}, nil
}

func (e *OpenAIEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	resp, err := e.client.CreateEmbeddings(ctx, openai.EmbeddingRequest{
		Model: openai.EmbeddingModel(e.model),
		Input: []string{text},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Data) == 0 || len(resp.Data[0].Embedding) == 0 {
		return nil, ErrNotSupported
	}
	// resp.Data[0].Embedding is []float32 in go-openai
	return resp.Data[0].Embedding, nil
}
