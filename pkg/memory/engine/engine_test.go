package engine

import (
	"context"
	"testing"
	"time"

	embedpkg "github.com/Raezil/lattice-agent/pkg/memory/embed"
	storepkg "github.com/Raezil/lattice-agent/pkg/memory/store"
)

func TestEngineWeightedRetrievalAndSummaries(t *testing.T) {
	memStore := storepkg.NewInMemoryStore()
	now := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	opts := Options{
		Weights:  ScoreWeights{Similarity: 0.45, Keywords: 0.20, Importance: 0.20, Recency: 0.10, Source: 0.05},
		HalfLife: 24 * time.Hour,
		TTL:      720 * time.Hour,
		SourceBoost: map[string]float64{
			"default":   0.6,
			"pagerduty": 1.0,
		},
		Clock: func() time.Time { return now },
	}
	engine := NewEngine(memStore, opts).WithEmbedder(embedpkg.DummyEmbedder{}).WithSummarizer(HeuristicSummarizer{})
	ctx := context.Background()

	if _, err := engine.Store(ctx, "team", "Critical production outage impacting users", map[string]any{"source": "pagerduty"}); err != nil {
		t.Fatalf("store critical memory: %v", err)
	}
	if _, err := engine.Store(ctx, "team", "Lunch options for tomorrow", map[string]any{"source": "slack"}); err != nil {
		t.Fatalf("store low priority memory: %v", err)
	}

	records, err := engine.Retrieve(ctx, "production issue", 2)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(records))
	}
	if records[0].Importance < records[1].Importance {
		t.Fatalf("expected high importance record first: got %.2f < %.2f", records[0].Importance, records[1].Importance)
	}
	if records[0].Summary == "" {
		t.Fatalf("expected summary to be populated")
	}
	snap := engine.MetricsSnapshot()
	if snap.ClustersSummarized == 0 {
		t.Fatalf("expected summarization metrics to record cluster summaries")
	}
	if snap.RecencySamples == 0 || snap.RecencyDecayAvg <= 0 {
		t.Fatalf("expected recency observations to be tracked, got samples=%d avg=%.4f", snap.RecencySamples, snap.RecencyDecayAvg)
	}
}

func TestEngineDeduplicationAndPrune(t *testing.T) {
	memStore := storepkg.NewInMemoryStore()
	opts := Options{
		TTL:   500 * time.Millisecond,
		Clock: time.Now,
	}
	engine := NewEngine(memStore, opts).WithEmbedder(embedpkg.DummyEmbedder{})
	ctx := context.Background()

	_, err := engine.Store(ctx, "alpha", "System upgrade completed", nil)
	if err != nil {
		t.Fatalf("store initial memory: %v", err)
	}
	beforeCount, _ := memStore.Count(ctx)
	if _, err := engine.Store(ctx, "alpha", "System upgrade completed", nil); err != nil {
		t.Fatalf("store duplicate: %v", err)
	}
	afterCount, _ := memStore.Count(ctx)
	if beforeCount != afterCount {
		t.Fatalf("duplicate should not increase count: before=%d after=%d", beforeCount, afterCount)
	}

	engine.clock = func() time.Time { return time.Now().Add(2 * opts.TTL) }
	if err := engine.Prune(ctx); err != nil {
		t.Fatalf("prune: %v", err)
	}
	remaining, _ := memStore.Count(ctx)
	if remaining != 0 {
		t.Fatalf("expected prune to remove expired record, got %d", remaining)
	}
	snap := engine.MetricsSnapshot()
	if snap.Deduplicated == 0 {
		t.Fatalf("expected deduplication metric to increment")
	}
	if snap.TTLExpired == 0 {
		t.Fatalf("expected TTL expiration metric to increment")
	}
}

func TestEngineHybridKeywordBoost(t *testing.T) {
	memStore := storepkg.NewInMemoryStore()
	opts := Options{
		Weights: ScoreWeights{
			Similarity: 0.2,
			Keywords:   0.6,
			Importance: 0.1,
			Recency:    0.05,
			Source:     0.05,
		},
		HalfLife: 24 * time.Hour,
	}
	engine := NewEngine(memStore, opts).WithEmbedder(embedpkg.DummyEmbedder{})
	ctx := context.Background()

	if _, err := engine.Store(ctx, "team", "Marketing campaign launch checklist", map[string]any{"source": "notion"}); err != nil {
		t.Fatalf("store marketing memory: %v", err)
	}
	if _, err := engine.Store(ctx, "team", "Investigate SAML login failure impacting enterprise customers", map[string]any{"source": "pagerduty"}); err != nil {
		t.Fatalf("store incident memory: %v", err)
	}

	records, err := engine.Retrieve(ctx, "SAML login failure escalation", 2)
	if err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	if len(records) != 2 {
		t.Fatalf("expected 2 records, got %d", len(records))
	}
	if records[0].KeywordScore <= records[1].KeywordScore {
		t.Fatalf("expected incident memory to rank higher on keywords: got %.2f vs %.2f", records[0].KeywordScore, records[1].KeywordScore)
	}
	if records[0].KeywordScore < 0.5 {
		t.Fatalf("expected strong keyword match, got %.2f", records[0].KeywordScore)
	}
}

func TestEngineReembedOnDrift(t *testing.T) {
	memStore := storepkg.NewInMemoryStore()
	opts := Options{HalfLife: time.Second}
	engine := NewEngine(memStore, opts).WithEmbedder(embedpkg.DummyEmbedder{})
	ctx := context.Background()

	rec, err := engine.Store(ctx, "beta", "Investigate latency regression", nil)
	if err != nil {
		t.Fatalf("store memory: %v", err)
	}

	zeros := make([]float32, len(rec.Embedding))
	if err := memStore.UpdateEmbedding(ctx, rec.ID, zeros, time.Now().Add(-2*opts.HalfLife)); err != nil {
		t.Fatalf("update embedding: %v", err)
	}

	if _, err := engine.Retrieve(ctx, "latency", 1); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	snap := engine.MetricsSnapshot()
	if snap.Reembedded == 0 {
		t.Fatalf("expected drift re-embedding metric to increment")
	}
}
