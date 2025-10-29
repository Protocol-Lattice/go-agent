package engine

import (
	"context"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

// Summarizer abstracts cluster summarization backends (LLMs or heuristics).
type Summarizer interface {
	Summarize(ctx context.Context, cluster []model.MemoryRecord) (string, error)
}

// HeuristicSummarizer produces deterministic summaries suitable for tests.
type HeuristicSummarizer struct{}

func (HeuristicSummarizer) Summarize(_ context.Context, cluster []model.MemoryRecord) (string, error) {
	if len(cluster) == 0 {
		return "", nil
	}
	var sentences []string
	for _, rec := range cluster {
		sentences = append(sentences, rec.Content)
	}
	summary := strings.Join(sentences, " ")
	if len(summary) > 280 {
		summary = summary[:280]
	}
	return summary, nil
}
