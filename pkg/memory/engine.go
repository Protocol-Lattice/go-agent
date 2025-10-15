package memory

import (
	"context"
	"errors"
	"fmt"
	"log"
	"math"
	"os"
	"sort"
	"strings"
	"sync"
	"time"
)

// Engine coordinates scoring, clustering, pruning and retrieval of memories.
type Engine struct {
	store      VectorStore
	opts       Options
	embedder   Embedder
	summarizer Summarizer
	metrics    *Metrics
	logger     *log.Logger
	clock      func() time.Time
	mu         sync.Mutex
}

// NewEngine constructs an advanced memory engine on top of a VectorStore implementation.
func NewEngine(store VectorStore, opts Options) *Engine {
	opts = opts.withDefaults()
	engine := &Engine{
		store:      store,
		opts:       opts,
		embedder:   AutoEmbedder(),
		summarizer: HeuristicSummarizer{},
		metrics:    &Metrics{},
		logger:     log.New(os.Stderr, "memory-engine: ", log.LstdFlags),
		clock:      opts.Clock,
	}
	if engine.clock == nil {
		engine.clock = time.Now
	}
	return engine
}

// WithEmbedder overrides the default embedder.
func (e *Engine) WithEmbedder(embedder Embedder) *Engine {
	if embedder != nil {
		e.embedder = embedder
	}
	return e
}

// WithSummarizer overrides the default cluster summarizer.
func (e *Engine) WithSummarizer(s Summarizer) *Engine {
	if s != nil {
		e.summarizer = s
		// ✅ Auto-enable summaries when a summarizer is set
		e.opts.EnableSummaries = true
	}
	return e
}

// WithLogger overrides the default logger.
func (e *Engine) WithLogger(logger *log.Logger) *Engine {
	if logger != nil {
		e.logger = logger
	}
	return e
}

func (e *Engine) logf(format string, args ...any) {
	if e.logger != nil {
		e.logger.Printf(format, args...)
	}
}

// MetricsSnapshot returns a copy of the runtime counters.
func (e *Engine) MetricsSnapshot() MetricsSnapshot {
	return e.metrics.Snapshot()
}

// Store embeds, scores and persists a new memory.
func (e *Engine) Store(ctx context.Context, sessionID, content string, metadata map[string]any) (MemoryRecord, error) {
	if e.store == nil {
		return MemoryRecord{}, errors.New("memory engine has no store")
	}
	embedding, err := e.embed(ctx, content)
	if err != nil {
		return MemoryRecord{}, fmt.Errorf("embed content: %w", err)
	}
	now := e.clock().UTC()
	if metadata == nil {
		metadata = map[string]any{}
	}
	if _, ok := metadata["space"]; !ok {
		metadata["space"] = sessionID
	}
	if _, ok := metadata["source"]; !ok {
		metadata["source"] = "default"
	}
	edges := sanitizeGraphEdges(metadata)
	importance := importanceScore(content, metadata)
	metadata["importance"] = importance
	// Deduplication based on cosine similarity.
	candidates, err := e.store.SearchMemory(ctx, embedding, 5)
	if err != nil {
		return MemoryRecord{}, err
	}
	for _, cand := range candidates {
		sim := cosineSimilarity(embedding, cand.Embedding)
		if sim >= e.opts.DuplicateSimilarity {
			e.metrics.IncDeduplicated()
			return cand, nil
		}
	}
	// Cluster summary for the new record.
	newRecord := MemoryRecord{
		SessionID:    sessionID,
		Content:      content,
		Embedding:    embedding,
		Importance:   importance,
		Source:       stringFromAny(metadata["source"]),
		CreatedAt:    now,
		LastEmbedded: now,
	}
	if e.opts.EnableSummaries {
		summary, sumErr := e.clusterSummary(ctx, append(candidates, newRecord), newRecord)
		if sumErr != nil {
			e.logf("failed to summarize cluster: %v", sumErr)
		} else {
			metadata["summary"] = summary
		}
	}
	metadata["last_embedded"] = now.UTC().Format(time.RFC3339Nano)
	if err := e.store.StoreMemory(ctx, sessionID, content, metadata, embedding); err != nil {
		return MemoryRecord{}, err
	}
	stored := newRecord
	if results, err := e.store.SearchMemory(ctx, embedding, 1); err == nil && len(results) > 0 {
		stored = results[0]
	}
	if stored.Space == "" {
		stored.Space = sessionID
	}
	if len(stored.GraphEdges) == 0 {
		stored.GraphEdges = edges
	}
	e.metrics.IncStored()
	if err := e.Prune(ctx); err != nil {
		e.logf("prune error: %v", err)
	}
	stored.Metadata = stringFromAny(metadata)
	stored.Summary = stringFromAny(metadata["summary"])
	stored.Importance = importance
	if graphStore, ok := e.store.(GraphStore); ok {
		if err := graphStore.UpsertGraph(ctx, stored, stored.GraphEdges); err != nil {
			e.logf("upsert graph: %v", err)
		}
	}
	return stored, nil
}

// Retrieve performs weighted retrieval with MMR diversification and optional re-embedding.
func (e *Engine) Retrieve(ctx context.Context, query string, limit int) ([]MemoryRecord, error) {
	if e.store == nil {
		return nil, errors.New("memory engine has no store")
	}
	if limit <= 0 {
		return nil, nil
	}
	embedding, err := e.embed(ctx, query)
	if err != nil {
		return nil, fmt.Errorf("embed query: %w", err)
	}
	searchLimit := limit * 4
	if searchLimit < limit {
		searchLimit = limit
	}
	candidates, err := e.store.SearchMemory(ctx, embedding, searchLimit)
	if err != nil {
		return nil, err
	}
	if len(candidates) == 0 {
		return nil, nil
	}
	if graphStore, ok := e.store.(GraphStore); ok && e.opts.GraphNeighborhoodLimit > 0 {
		seedIDs := make([]int64, 0, len(candidates))
		for _, cand := range candidates {
			if cand.ID != 0 {
				seedIDs = append(seedIDs, cand.ID)
			}
		}
		if len(seedIDs) > 0 {
			hops := e.opts.GraphNeighborhoodHops
			if hops <= 0 {
				hops = 1
			}
			neighbors, err := graphStore.Neighborhood(ctx, seedIDs, hops, e.opts.GraphNeighborhoodLimit)
			if err != nil {
				e.logf("graph neighborhood: %v", err)
			} else if len(neighbors) > 0 {
				existingByID := make(map[int64]struct{}, len(candidates))
				existingKey := make(map[string]struct{})
				for _, cand := range candidates {
					if cand.ID != 0 {
						existingByID[cand.ID] = struct{}{}
					} else {
						key := cand.SessionID + "\u241F" + strings.TrimSpace(cand.Content)
						existingKey[key] = struct{}{}
					}
				}
				for _, nb := range neighbors {
					if nb.ID != 0 {
						if _, ok := existingByID[nb.ID]; ok {
							continue
						}
						existingByID[nb.ID] = struct{}{}
					} else {
						key := nb.SessionID + "\u241F" + strings.TrimSpace(nb.Content)
						if _, ok := existingKey[key]; ok {
							continue
						}
						existingKey[key] = struct{}{}
					}
					if len(nb.Embedding) > 0 {
						nb.Score = cosineSimilarity(embedding, nb.Embedding)
					}
					candidates = append(candidates, nb)
				}
			}
		}
	}
	weights := e.opts.normalizedWeights()
	now := e.clock().UTC()
	for i := range candidates {
		rec := &candidates[i]
		if rec.Importance == 0 {
			rec.Importance = importanceScore(rec.Content, decodeMetadata(rec.Metadata))
		}
		rec.Score = cosineSimilarity(embedding, rec.Embedding)
		recency := recencyScore(now.Sub(rec.CreatedAt), e.opts.HalfLife)
		sourceScore := e.sourceScore(rec.Source)
		rec.WeightedScore = weights.Similarity*rec.Score + weights.Importance*rec.Importance + weights.Recency*recency + weights.Source*sourceScore
	}
	selected := mmrSelect(candidates, embedding, limit, e.opts.LambdaMMR)
	if e.opts.EnableSummaries {
		if err := e.populateSummaries(ctx, selected); err != nil {
			e.logf("populate summaries: %v", err)
		}
	}
	if err := e.reembedOnDrift(ctx, selected); err != nil {
		e.logf("reembed drift: %v", err)
	}
	e.metrics.IncRetrieved(len(selected))
	sort.Slice(selected, func(i, j int) bool {
		// 1) Highest importance first (hard rule)
		if selected[i].Importance != selected[j].Importance {
			return selected[i].Importance > selected[j].Importance
		}
		// 2) Then by weighted relevance
		if selected[i].WeightedScore != selected[j].WeightedScore {
			return selected[i].WeightedScore > selected[j].WeightedScore
		}
		// 3) Finally by recency (newer first)
		return selected[i].CreatedAt.After(selected[j].CreatedAt)
	})

	return selected, nil
}

// Prune applies TTL, size and deduplication policies.
func (e *Engine) Prune(ctx context.Context) error {
	if e.store == nil {
		return nil
	}
	now := e.clock().UTC()
	var toDelete []int64
	seenHashes := make(map[string]int64)
	err := e.store.Iterate(ctx, func(rec MemoryRecord) bool {
		if !rec.CreatedAt.IsZero() && now.Sub(rec.CreatedAt) > e.opts.TTL {
			toDelete = append(toDelete, rec.ID)
			return true
		}
		hash := strings.ToLower(strings.TrimSpace(rec.Content))
		if existing, ok := seenHashes[hash]; ok {
			toDelete = append(toDelete, rec.ID)
			e.metrics.IncDeduplicated()
			e.logf("prune dedupe: dropping id=%d duplicate of %d", rec.ID, existing)
			return true
		}
		seenHashes[hash] = rec.ID
		return true
	})
	if err != nil {
		return err
	}
	if len(toDelete) > 0 {
		if err := e.store.DeleteMemory(ctx, toDelete); err != nil {
			return err
		}
		e.metrics.IncPruned(len(toDelete))
	}
	count, err := e.store.Count(ctx)
	if err != nil {
		return err
	}
	if count <= e.opts.MaxSize {
		return nil
	}
	overflow := count - e.opts.MaxSize
	var scored []struct {
		id    int64
		score float64
	}
	err = e.store.Iterate(ctx, func(rec MemoryRecord) bool {
		ageHours := now.Sub(rec.CreatedAt).Hours() + 1
		importance := rec.Importance
		if importance == 0 {
			importance = importanceScore(rec.Content, decodeMetadata(rec.Metadata))
		}
		pruneScore := ageHours * (1 - importance)
		scored = append(scored, struct {
			id    int64
			score float64
		}{id: rec.ID, score: pruneScore})
		return true
	})
	if err != nil {
		return err
	}
	sort.Slice(scored, func(i, j int) bool { return scored[i].score > scored[j].score })
	if overflow > len(scored) {
		overflow = len(scored)
	}
	var evict []int64
	for i := 0; i < overflow; i++ {
		evict = append(evict, scored[i].id)
	}
	if len(evict) > 0 {
		if err := e.store.DeleteMemory(ctx, evict); err != nil {
			return err
		}
		e.metrics.IncPruned(len(evict))
	}
	return nil
}

func (e *Engine) embed(ctx context.Context, text string) ([]float32, error) {
	if e.embedder == nil {
		e.embedder = AutoEmbedder()
	}
	vec, err := e.embedder.Embed(ctx, text)
	if err != nil || len(vec) == 0 {
		return DummyEmbedding(text), nil
	}
	return vec, nil
}

func (e *Engine) reembedOnDrift(ctx context.Context, records []MemoryRecord) error {
	for _, rec := range records {
		if rec.ID == 0 {
			continue
		}
		age := e.clock().UTC().Sub(rec.LastEmbedded)
		if age < e.opts.HalfLife && rec.LastEmbedded.After(time.Time{}) {
			continue
		}
		vec, err := e.embed(ctx, rec.Content)
		if err != nil {
			return err
		}
		sim := cosineSimilarity(vec, rec.Embedding)
		if sim >= e.opts.DriftThreshold {
			continue
		}
		if err := e.store.UpdateEmbedding(ctx, rec.ID, vec, e.clock().UTC()); err != nil {
			return err
		}
		e.metrics.IncReembedded()
	}
	return nil
}

func (e *Engine) populateSummaries(ctx context.Context, records []MemoryRecord) error {
	if e.summarizer == nil || !e.opts.EnableSummaries {
		return nil
	}
	clusters := clusterRecords(records, e.opts.ClusterSimilarity)
	for _, cluster := range clusters {
		summary, err := e.summarizer.Summarize(ctx, cluster)
		if err != nil {
			return err
		}
		e.metrics.IncClustersSummarized()
		for i := range cluster {
			for j := range records {
				if records[j].ID == cluster[i].ID {
					records[j].Summary = summary
				}
			}
		}
	}
	return nil
}

func (e *Engine) clusterSummary(ctx context.Context, records []MemoryRecord, target MemoryRecord) (string, error) {
	clusters := clusterRecords(records, e.opts.ClusterSimilarity)
	for _, cluster := range clusters {
		for _, rec := range cluster {
			if rec.ID == target.ID || (rec.ID == 0 && rec.Content == target.Content) {
				if e.summarizer == nil {
					return "", nil
				}
				return e.summarizer.Summarize(ctx, cluster)
			}
		}
	}
	return "", nil
}

func (e *Engine) sourceScore(source string) float64 {
	if source == "" {
		source = "default"
	}
	if e.opts.SourceBoost == nil {
		return 1
	}
	if val, ok := e.opts.SourceBoost[strings.ToLower(source)]; ok {
		return clamp(val, 0, 1)
	}
	if val, ok := e.opts.SourceBoost["default"]; ok {
		return clamp(val, 0, 1)
	}
	return 1
}

func recencyScore(age time.Duration, halfLife time.Duration) float64 {
	if halfLife <= 0 {
		return 1
	}
	decay := math.Pow(0.5, age.Seconds()/(halfLife.Seconds()))
	if decay < 0 {
		return 0
	}
	return decay
}

func importanceScore(content string, metadata map[string]any) float64 {
	if val := floatFromAny(metadata["importance"]); val > 0 {
		return clamp(val, 0, 1)
	}
	tokens := strings.Fields(strings.ToLower(content))
	lengthScore := math.Min(float64(len(tokens))/60.0, 1.0)

	keywordBoost := 0.0
	urgentKeywords := []string{"urgent", "critical", "deadline", "important", "alert", "error", "outage", "failure"}
	for _, kw := range urgentKeywords {
		if strings.Contains(strings.ToLower(content), kw) {
			keywordBoost += 0.25 // was 0.1 — give stronger boost
		}
	}
	if keywordBoost > 0.6 {
		keywordBoost = 0.6
	}
	return clamp(lengthScore+keywordBoost, 0, 1)
}

func mmrSelect(records []MemoryRecord, query []float32, limit int, lambda float64) []MemoryRecord {
	if limit >= len(records) {
		out := make([]MemoryRecord, len(records))
		copy(out, records)
		return out
	}
	if lambda < 0 {
		lambda = 0
	}
	if lambda > 1 {
		lambda = 1
	}
	remaining := make([]MemoryRecord, len(records))
	copy(remaining, records)
	selected := make([]MemoryRecord, 0, limit)
	for len(selected) < limit && len(remaining) > 0 {
		bestIdx := 0
		bestScore := math.Inf(-1)
		for i, cand := range remaining {
			relevance := cand.WeightedScore
			if relevance == 0 {
				relevance = cosineSimilarity(query, cand.Embedding)
			}
			var maxSim float64
			for _, sel := range selected {
				if sim := cosineSimilarity(cand.Embedding, sel.Embedding); sim > maxSim {
					maxSim = sim
				}
			}
			score := lambda*relevance - (1-lambda)*maxSim
			if lambda == 0 {
				score = -maxSim
			}
			if score > bestScore {
				bestScore = score
				bestIdx = i
			}
		}
		selected = append(selected, remaining[bestIdx])
		remaining = append(remaining[:bestIdx], remaining[bestIdx+1:]...)
	}
	return selected
}

func clusterRecords(records []MemoryRecord, threshold float64) [][]MemoryRecord {
	if threshold <= 0 {
		threshold = 0.8
	}
	type cluster struct {
		centroid []float64
		records  []MemoryRecord
	}
	var clusters []cluster
	for _, rec := range records {
		placed := false
		for i := range clusters {
			sim := cosineSimilarity(rec.Embedding, float32Slice(clusters[i].centroid))
			if sim >= threshold {
				clusters[i].records = append(clusters[i].records, rec)
				clusters[i].centroid = updateCentroid(clusters[i].centroid, rec.Embedding)
				placed = true
				break
			}
		}
		if !placed {
			clusters = append(clusters, cluster{
				centroid: float32To64(rec.Embedding),
				records:  []MemoryRecord{rec},
			})
		}
	}
	result := make([][]MemoryRecord, len(clusters))
	for i, cl := range clusters {
		result[i] = cl.records
	}
	return result
}

func updateCentroid(current []float64, vec []float32) []float64 {
	if len(current) == 0 {
		return float32To64(vec)
	}
	if len(vec) == 0 {
		return current
	}
	if len(current) != len(vec) {
		return current
	}
	for i := range current {
		current[i] = (current[i] + float64(vec[i])) / 2
	}
	return current
}

func float32To64(vec []float32) []float64 {
	out := make([]float64, len(vec))
	for i, v := range vec {
		out[i] = float64(v)
	}
	return out
}

func float32Slice(vec []float64) []float32 {
	out := make([]float32, len(vec))
	for i, v := range vec {
		out[i] = float32(v)
	}
	return out
}

func cosineSimilarity(a, b []float32) float64 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}
	var dot, normA, normB float64
	length := len(a)
	if len(b) < length {
		length = len(b)
	}
	for i := 0; i < length; i++ {
		dot += float64(a[i]) * float64(b[i])
		normA += float64(a[i]) * float64(a[i])
		normB += float64(b[i]) * float64(b[i])
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}

func clamp(val, minVal, maxVal float64) float64 {
	if val < minVal {
		return minVal
	}
	if val > maxVal {
		return maxVal
	}
	return val
}
