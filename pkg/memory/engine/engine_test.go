package engine

import (
	"context"
	"sync"
	"testing"
	"time"

	embedpkg "github.com/Raezil/go-agent-development-kit/pkg/memory/embed"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/model"
	storepkg "github.com/Raezil/go-agent-development-kit/pkg/memory/store"
)

func TestEngineWeightedRetrievalAndSummaries(t *testing.T) {
	memStore := storepkg.NewInMemoryStore()
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

func TestEngineMCTSExpandsGraphNeighborhood(t *testing.T) {
	ctx := context.Background()

	baseOpts := Options{
		Weights:                ScoreWeights{Similarity: 0.6, Importance: 0.25, Recency: 0.1, Source: 0.05},
		HalfLife:               24 * time.Hour,
		GraphNeighborhoodLimit: -1,
		Clock: func() time.Time {
			return time.Date(2024, 1, 1, 12, 0, 0, 0, time.UTC)
		},
	}

	withoutMCTSStore := newGraphTestStore()
	withoutEngine := NewEngine(withoutMCTSStore, baseOpts).WithEmbedder(embedpkg.DummyEmbedder{})
	rootA, neighborA, _ := seedGraphMemories(t, withoutEngine, withoutMCTSStore)

	recordsWithout, err := withoutEngine.Retrieve(ctx, "core topic", 2)
	if err != nil {
		t.Fatalf("retrieve without mcts: %v", err)
	}
	rootPresent := false
	for _, rec := range recordsWithout {
		if rec.ID == neighborA.ID {
			t.Fatalf("neighbor should not be returned without MCTS")
		}
		if rec.ID == rootA.ID {
			rootPresent = true
		}
	}
	if !rootPresent {
		t.Fatalf("expected root memory to be present without MCTS")
	}

	withMCTSStore := newGraphTestStore()
	mctsOpts := baseOpts
	mctsOpts.EnableMCTS = true
	mctsOpts.MCTSSimulations = 64
	mctsOpts.MCTSExpansion = 4
	mctsOpts.MCTSMaxDepth = 2
	withEngine := NewEngine(withMCTSStore, mctsOpts).WithEmbedder(embedpkg.DummyEmbedder{})
	rootB, neighborB, _ := seedGraphMemories(t, withEngine, withMCTSStore)

	recordsWith, err := withEngine.Retrieve(ctx, "core topic", 2)
	if err != nil {
		t.Fatalf("retrieve with mcts: %v", err)
	}
	foundNeighbor := false
	foundRoot := false
	for _, rec := range recordsWith {
		if rec.ID == neighborB.ID {
			foundNeighbor = true
		}
		if rec.ID == rootB.ID {
			foundRoot = true
		}
	}
	if !foundNeighbor {
		t.Fatalf("expected neighbor to be surfaced via MCTS traversal")
	}
	if !foundRoot {
		t.Fatalf("expected root memory to remain accessible with MCTS")
	}
}

type graphTestStore struct {
	base       *storepkg.InMemoryStore
	neighbors  map[int64][]int64
	skipSearch map[int64]bool
	cache      map[int64]model.MemoryRecord
	mu         sync.RWMutex
}

func newGraphTestStore() *graphTestStore {
	return &graphTestStore{
		base:       storepkg.NewInMemoryStore(),
		neighbors:  make(map[int64][]int64),
		skipSearch: make(map[int64]bool),
		cache:      make(map[int64]model.MemoryRecord),
	}
}

func (s *graphTestStore) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	if err := s.base.StoreMemory(ctx, sessionID, content, metadata, embedding); err != nil {
		return err
	}
	s.refreshCache(ctx)
	return nil
}

func (s *graphTestStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	if limit <= 0 {
		return nil, nil
	}
	// request additional candidates so filtering still returns enough
	results, err := s.base.SearchMemory(ctx, queryEmbedding, limit*4)
	if err != nil {
		return nil, err
	}
	filtered := make([]model.MemoryRecord, 0, len(results))
	for _, rec := range results {
		s.mu.RLock()
		skip := s.skipSearch[rec.ID]
		s.mu.RUnlock()
		if skip {
			continue
		}
		filtered = append(filtered, rec)
		if len(filtered) >= limit {
			break
		}
	}
	return filtered, nil
}

func (s *graphTestStore) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	return s.base.UpdateEmbedding(ctx, id, embedding, lastEmbedded)
}

func (s *graphTestStore) DeleteMemory(ctx context.Context, ids []int64) error {
	if err := s.base.DeleteMemory(ctx, ids); err != nil {
		return err
	}
	s.refreshCache(ctx)
	return nil
}

func (s *graphTestStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	return s.base.Iterate(ctx, fn)
}

func (s *graphTestStore) Count(ctx context.Context) (int, error) {
	return s.base.Count(ctx)
}

func (s *graphTestStore) UpsertGraph(_ context.Context, record model.MemoryRecord, edges []model.GraphEdge) error {
	ids := make([]int64, 0, len(edges))
	for _, edge := range edges {
		ids = append(ids, edge.Target)
	}
	s.mu.Lock()
	s.neighbors[record.ID] = ids
	s.mu.Unlock()
	return nil
}

func (s *graphTestStore) Neighborhood(_ context.Context, seedIDs []int64, hops, limit int) ([]model.MemoryRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit <= 0 {
		return nil, nil
	}
	type pair struct {
		id    int64
		depth int
	}
	queue := make([]pair, 0, len(seedIDs))
	seen := make(map[int64]struct{}, len(seedIDs))
	for _, id := range seedIDs {
		queue = append(queue, pair{id: id, depth: 0})
		seen[id] = struct{}{}
	}
	results := make([]model.MemoryRecord, 0, limit)
	for len(queue) > 0 && len(results) < limit {
		cur := queue[0]
		queue = queue[1:]
		if cur.depth >= hops {
			continue
		}
		neighbors := s.neighbors[cur.id]
		for _, nb := range neighbors {
			if _, ok := seen[nb]; ok {
				continue
			}
			seen[nb] = struct{}{}
			if rec, ok := s.cache[nb]; ok {
				results = append(results, rec)
				if len(results) >= limit {
					break
				}
				queue = append(queue, pair{id: nb, depth: cur.depth + 1})
			}
		}
	}
	return results, nil
}

func (s *graphTestStore) refreshCache(ctx context.Context) {
	s.mu.Lock()
	defer s.mu.Unlock()
	cache := make(map[int64]model.MemoryRecord)
	_ = s.base.Iterate(ctx, func(rec model.MemoryRecord) bool {
		cache[rec.ID] = rec
		return true
	})
	s.cache = cache
}

func (s *graphTestStore) SetNeighbors(source int64, targets []int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.neighbors[source] = append([]int64(nil), targets...)
}

func (s *graphTestStore) SetSkipSearch(id int64, skip bool) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if skip {
		s.skipSearch[id] = true
	} else {
		delete(s.skipSearch, id)
	}
}

func seedGraphMemories(t *testing.T, engine *Engine, store *graphTestStore) (model.MemoryRecord, model.MemoryRecord, model.MemoryRecord) {
	t.Helper()
	ctx := context.Background()
	root, err := engine.Store(ctx, "graph-session", "Core topic overview", map[string]any{"importance": 0.6})
	if err != nil {
		t.Fatalf("store root: %v", err)
	}
	neighbor, err := engine.Store(ctx, "graph-session", "Detailed mitigation steps for the core topic", map[string]any{"importance": 0.95, "source": "pagerduty"})
	if err != nil {
		t.Fatalf("store neighbor: %v", err)
	}
	distractor, err := engine.Store(ctx, "graph-session", "Random lunch discussion", map[string]any{"importance": 0.2, "source": "slack"})
	if err != nil {
		t.Fatalf("store distractor: %v", err)
	}

	store.SetNeighbors(root.ID, []int64{neighbor.ID})
	store.SetSkipSearch(neighbor.ID, true)

	return root, neighbor, distractor
}
