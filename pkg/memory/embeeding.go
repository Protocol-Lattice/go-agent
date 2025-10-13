package memory

import (
	"context"
	"log"
	"os"
	"sync"

	"github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// VertexAIEmbedder provides a thread-safe embedding generator.
type VertexAIEmbedder struct {
	client *genai.Client
	model  *genai.EmbeddingModel
	once   sync.Once
	err    error
}

// singleton
var embedder VertexAIEmbedder

// initVertex initializes the Vertex AI embedding model once
func initVertex() {
	ctx := context.Background()
	client, err := genai.NewClient(ctx, option.WithAPIKey(os.Getenv("GOOGLE_API_KEY")))
	if err != nil {
		embedder.err = err
		return
	}

	embedder.client = client
	embedder.model = client.EmbeddingModel("text-embedding-004")
}

// VertexAIEmbedding returns the embedding vector for the given text.
func VertexAIEmbedding(text string) []float32 {
	embedder.once.Do(initVertex)
	if embedder.err != nil {
		log.Printf("Vertex init failed, using dummy embedding: %v", embedder.err)
		return DummyEmbedding(text)
	}

	ctx := context.Background()
	resp, err := embedder.model.EmbedContent(ctx, genai.Text(text))
	if err != nil {
		log.Printf("Vertex embedding failed, using dummy: %v", err)
		return DummyEmbedding(text)
	}

	if resp == nil || resp.Embedding == nil {
		log.Printf("Vertex returned empty embedding, using dummy")
		return DummyEmbedding(text)
	}

	return resp.Embedding.Values
}

// fallback for resilience
func DummyEmbedding(text string) []float32 {
	vec := make([]float32, 768)
	for i, ch := range []byte(text) {
		vec[i%768] += float32(ch) / 255.0
	}
	return vec
}
