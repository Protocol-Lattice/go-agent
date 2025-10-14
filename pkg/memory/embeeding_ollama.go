package memory

import (
	"context"
	"net/http"
	"net/url"
	"os"
	"time"

	ollama "github.com/ollama/ollama/api"
)

type OllamaEmbedder struct {
	client *ollama.Client
	model  string
}

func NewOllamaEmbedder(model string) (Embedder, error) {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434"
	}
	u, err := url.Parse(host)
	if err != nil {
		return nil, err
	}
	httpClient := &http.Client{Timeout: 60 * time.Second}
	cli := ollama.NewClient(u, httpClient)

	if model == "" {
		// Commonly available local embedding model; override as needed.
		model = "nomic-embed-text"
	}
	return &OllamaEmbedder{client: cli, model: model}, nil
}

func (e *OllamaEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	res, err := e.client.Embed(ctx, &ollama.EmbedRequest{
		Model: e.model,
		Input: text, // can also be []string; we use single input
	})
	if err != nil {
		return nil, err
	}
	if res == nil || len(res.Embeddings) == 0 || len(res.Embeddings[0]) == 0 {
		return nil, ErrNotSupported
	}
	return res.Embeddings[0], nil
}
