package memory

import (
	"context"
	"fmt"
	"strings"
)

func DummyEmbedding(text string) []float32 {
	vec := make([]float32, 768)
	for i, ch := range []byte(text) {
		vec[i%768] += float32(ch) / 255.0
	}
	return vec
}

// BuildPrompt retrieves top-k similar memories and composes a context-aware prompt
func (mb *MemoryBank) BuildPrompt(ctx context.Context, query string, limit int) (string, error) {
	// Step 1: Embed query using the same embedding strategy
	queryEmbedding := DummyEmbedding(query) // placeholder — replace with real embedding provider

	// Step 2: Search memory bank for relevant context
	results, err := mb.SearchMemory(ctx, queryEmbedding, limit)
	if err != nil {
		return "", fmt.Errorf("memory search failed: %w", err)
	}

	// Step 3: Build prompt string
	var sb strings.Builder

	sb.WriteString("You are a helpful and knowledgeable AI agent.\n")
	sb.WriteString("Use the context below to answer the question accurately.\n\n")
	sb.WriteString("Context:\n")

	if len(results) == 0 {
		sb.WriteString("(No relevant memories found)\n")
	} else {
		for i, r := range results {
			sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, strings.TrimSpace(r.Content)))
		}
	}

	sb.WriteString("\nUser Query:\n")
	sb.WriteString(query)
	sb.WriteString("\n\nAnswer:")
	return sb.String(), nil
}
