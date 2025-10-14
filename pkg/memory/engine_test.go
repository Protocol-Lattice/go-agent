package memory

import (
	"context"
	"testing"
	"time"
)

func TestEngineWeightedRetrievalAndSummaries(t *testing.T) {
	store := NewInMemoryStore()
	now := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	opts := Options{
		Weights:  ScoreWeights{Similarity: 0.55, Importance: 0.25, Recency: 0.15, Source: 0.05},
		HalfLife: 24 * time.Hour,
		TTL:      720 * time.Hour,
		SourceBoost: map[string]float64{
			"default":   0.6,
			"pagerduty": 1.0,
		},
		Clock: func() time.Time { return now },
	}
	engine := NewEngine(store, opts).WithEmbedder(DummyEmbedder{}).WithSummarizer(HeuristicSummarizer{})
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
}

func TestEngineDeduplicationAndPrune(t *testing.T) {
	store := NewInMemoryStore()
	now := time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
	opts := Options{
		TTL:   time.Hour,
		Clock: func() time.Time { return now },
	}
	engine := NewEngine(store, opts).WithEmbedder(DummyEmbedder{})
	ctx := context.Background()

	rec, err := engine.Store(ctx, "alpha", "System upgrade completed", nil)
	if err != nil {
		t.Fatalf("store initial memory: %v", err)
	}
	beforeCount, _ := store.Count(ctx)
	if _, err := engine.Store(ctx, "alpha", "System upgrade completed", nil); err != nil {
		t.Fatalf("store duplicate: %v", err)
	}
	afterCount, _ := store.Count(ctx)
	if beforeCount != afterCount {
		t.Fatalf("duplicate should not increase count: before=%d after=%d", beforeCount, afterCount)
	}

	store.mu.Lock()
	entry := store.records[rec.ID]
	entry.CreatedAt = now.Add(-2 * opts.TTL)
	store.records[rec.ID] = entry
	store.mu.Unlock()

	if err := engine.Prune(ctx); err != nil {
		t.Fatalf("prune: %v", err)
	}
	remaining, _ := store.Count(ctx)
	if remaining != 0 {
		t.Fatalf("expected prune to remove expired record, got %d", remaining)
	}
}

func TestEngineReembedOnDrift(t *testing.T) {
	store := NewInMemoryStore()
	opts := Options{HalfLife: time.Second}
	engine := NewEngine(store, opts).WithEmbedder(DummyEmbedder{})
	ctx := context.Background()

	rec, err := engine.Store(ctx, "beta", "Investigate latency regression", nil)
	if err != nil {
		t.Fatalf("store memory: %v", err)
	}

	store.mu.Lock()
	entry := store.records[rec.ID]
	for i := range entry.Embedding {
		entry.Embedding[i] = 0
	}
	entry.LastEmbedded = time.Now().Add(-2 * opts.HalfLife)
	store.records[rec.ID] = entry
	store.mu.Unlock()

	if _, err := engine.Retrieve(ctx, "latency", 1); err != nil {
		t.Fatalf("retrieve: %v", err)
	}
	snap := engine.MetricsSnapshot()
	if snap.Reembedded == 0 {
		t.Fatalf("expected drift re-embedding metric to increment")
	}
}
