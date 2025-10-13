package memory

import (
	"context"
	"fmt"
	"strings"
)

// BuildPrompt now supports short- and long-term context
func (sm *SessionMemory) BuildPrompt(ctx context.Context, sessionID, query string, limit int) (string, error) {
	results, err := sm.RetrieveContext(ctx, sessionID, query, limit)
	if err != nil {
		return "", fmt.Errorf("context retrieval failed: %w", err)
	}

	var sb strings.Builder
	sb.WriteString("You are a helpful AI agent.\n")
	sb.WriteString("Use the context below to answer accurately.\n\n")
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
