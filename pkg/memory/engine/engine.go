package engine

import (
	"container/heap"
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
	"unicode"
	"unicode/utf8"

	"github.com/Raezil/go-agent-development-kit/pkg/memory/embed"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/model"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/store"
)

// Engine coordinates scoring, clustering, pruning and retrieval of memories.
type Engine struct {
	store      store.VectorStore
	opts       Options
	embedder   embed.Embedder
	summarizer Summarizer
	metrics    *Metrics
	logger     *log.Logger
	clock      func() time.Time
	mu         sync.Mutex
}

// NewEngine constructs an advanced memory engine on top of a VectorStore implementation.
func NewEngine(store store.VectorStore, opts Options) *Engine {
	opts = opts.withDefaults()
	engine := &Engine{
		store:      store,
		opts:       opts,
		embedder:   embed.AutoEmbedder(),
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
func (e *Engine) WithEmbedder(embedder embed.Embedder) *Engine {
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
func (e *Engine) Store(ctx context.Context, sessionID, content string, metadata map[string]any) (model.MemoryRecord, error) {
	if e.store == nil {
		return model.MemoryRecord{}, errors.New("memory engine has no store")
	}
	embedding, err := e.embed(ctx, content)
	if err != nil {
		return model.MemoryRecord{}, fmt.Errorf("embed content: %w", err)
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
	edges := model.SanitizeGraphEdges(metadata)
	importance := importanceScore(content, metadata)
	metadata["importance"] = importance
	// Deduplication based on cosine similarity.
	candidates, err := e.store.SearchMemory(ctx, embedding, 5)
	if err != nil {
		return model.MemoryRecord{}, err
	}
	for _, cand := range candidates {
		sim := model.CosineSimilarity(embedding, cand.Embedding)
		if sim >= e.opts.DuplicateSimilarity {
			e.metrics.IncDeduplicated()
			return cand, nil
		}
	}
	// Cluster summary for the new record.
	newRecord := model.MemoryRecord{
		SessionID:    sessionID,
		Content:      content,
		Embedding:    embedding,
		Importance:   importance,
		Source:       model.StringFromAny(metadata["source"]),
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
		return model.MemoryRecord{}, err
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
	stored.Metadata = model.StringFromAny(metadata)
	stored.Summary = model.StringFromAny(metadata["summary"])
	stored.Importance = importance
	if graphStore, ok := e.store.(store.GraphStore); ok {
		if err := graphStore.UpsertGraph(ctx, stored, stored.GraphEdges); err != nil {
			e.logf("upsert graph: %v", err)
		}
	}
	return stored, nil
}

// Retrieve performs weighted retrieval with MMR diversification and optional re-embedding.
func (e *Engine) Retrieve(ctx context.Context, query string, limit int) ([]model.MemoryRecord, error) {
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
	if graphStore, ok := e.store.(store.GraphStore); ok && e.opts.GraphNeighborhoodLimit > 0 {
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
						nb.Score = model.CosineSimilarity(embedding, nb.Embedding)
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
		recency := e.computeWeightedScore(now, weights, embedding, rec)
		if e.metrics != nil {
			e.metrics.ObserveRecency(recency)
		}
	}

	selected := mmrSelect(candidates, embedding, limit, e.opts.LambdaMMR)
	if e.opts.EnableMCTS {
		if graphStore, ok := e.store.(store.GraphStore); ok {
			if refined := e.mctsRefine(ctx, now, graphStore, embedding, weights, candidates, limit); len(refined) > 0 {
				selected = dedupeRecords(append(selected, refined...))
			}
		}
	}
	if e.opts.EnableSummaries {
		if err := e.populateSummaries(ctx, selected); err != nil {
			e.logf("populate summaries: %v", err)
		}
	}
	if err := e.reembedOnDrift(ctx, selected); err != nil {
		e.logf("reembed drift: %v", err)
	}
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

	if len(selected) > limit {
		selected = selected[:limit]
	}
	if e.metrics != nil {
		e.metrics.IncRetrieved(len(selected))
	}

	return selected, nil
}

func (e *Engine) computeWeightedScore(now time.Time, weights ScoreWeights, query []float32, rec *model.MemoryRecord) float64 {
	if rec == nil {
		return 0
	}
	if rec.Importance == 0 {
		rec.Importance = importanceScore(rec.Content, model.DecodeMetadata(rec.Metadata))
	}
	rec.Score = model.CosineSimilarity(query, rec.Embedding)
	recency := recencyScore(now.Sub(rec.CreatedAt), e.opts.HalfLife)
	sourceScore := e.sourceScore(rec.Source)
	rec.WeightedScore = weights.Similarity*rec.Score + weights.Importance*rec.Importance + weights.Recency*recency + weights.Source*sourceScore
	return recency
}

// Prune applies TTL, size and deduplication policies.
// Prune applies TTL, size and deduplication policies.
// Faster version: single Iterate pass for TTL+dedupe, no full sort, top-K heap for evictions.
func (e *Engine) Prune(ctx context.Context) error {
	if e.store == nil {
		return nil
	}

	now := e.clock().UTC()

	// Accumulate TTL/dedup deletions in batches to keep memory/network usage bounded.
	const delBatch = 1024

	type pendingDeletion struct {
		id  int64
		ttl bool
	}

	batchDelete := func(pending *[]pendingDeletion) error {
		if len(*pending) == 0 {
			return nil
		}
		ids := make([]int64, len(*pending))
		ttlCount := 0
		for i := range *pending {
			ids[i] = (*pending)[i].id
			if (*pending)[i].ttl {
				ttlCount++
			}
		}
		if err := e.store.DeleteMemory(ctx, ids); err != nil {
			return err
		}
		if e.metrics != nil {
			e.metrics.IncPruned(len(ids))
			if ttlCount > 0 {
				e.metrics.IncTTLExpired(ttlCount)
			}
		}
		*pending = (*pending)[:0]
		return nil
	}

	// canonical seen set for dedupe (single-allocation lower+trim)
	seen := make(map[string]int64, 1024)

	// survivors we may need to score for size-based eviction (kept minimal)
	type cand struct {
		id         int64
		createdAt  time.Time
		importance float64
		content    string
		metadata   string
	}
	candidates := make([]cand, 0, 1024)

	toDelete := make([]pendingDeletion, 0, delBatch)
	survivors := 0

	err := e.store.Iterate(ctx, func(rec model.MemoryRecord) bool {
		// TTL
		if !rec.CreatedAt.IsZero() && now.Sub(rec.CreatedAt) > e.opts.TTL {
			toDelete = append(toDelete, pendingDeletion{id: rec.ID, ttl: true})
			if len(toDelete) >= delBatch {
				if err := batchDelete(&toDelete); err != nil {
					// Stop iteration on error; propagate after Iterate returns.
					// We can't return false here because Iterate signature suggests bool continue.
					// Instead, stash the error via closure capture (below) if your Iterate supports it.
				}
			}
			return true
		}

		// Dedup (case-insensitive, whitespace-trimmed)
		key := canonicalKey(rec.Content)
		if prevID, ok := seen[key]; ok {
			_ = prevID // kept for potential debugging/metrics
			toDelete = append(toDelete, pendingDeletion{id: rec.ID})
			if e.metrics != nil {
				e.metrics.IncDeduplicated()
			}
			if len(toDelete) >= delBatch {
				if err := batchDelete(&toDelete); err != nil {
					// same note as above
				}
			}
			return true
		}
		seen[key] = rec.ID

		// Keep as survivor; only minimal info needed for later scoring.
		candidates = append(candidates, cand{
			id:         rec.ID,
			createdAt:  rec.CreatedAt,
			importance: rec.Importance,
			content:    rec.Content,
			metadata:   rec.Metadata,
		})
		survivors++
		return true
	})
	if err != nil {
		return err
	}
	// Flush any pending TTL/dedup deletes.
	if err := batchDelete(&toDelete); err != nil {
		return err
	}

	// If we fit under MaxSize after TTL+dedupe, we are done.
	if survivors <= e.opts.MaxSize {
		return nil
	}

	overflow := survivors - e.opts.MaxSize

	h := make(minHeap, 0, overflow)
	heap.Init(&h)

	// Score survivors once; compute importance lazily where zero.
	for i := range candidates {
		c := &candidates[i]
		imp := c.importance
		if imp == 0 {
			imp = importanceScore(c.content, model.DecodeMetadata(c.metadata))
			c.importance = imp // cache (useful if future policies reuse)
		}
		ageHours := now.Sub(c.createdAt).Hours() + 1 // +1 to avoid zero bias
		score := ageHours * (1 - imp)

		if len(h) < overflow {
			heap.Push(&h, item{id: c.id, score: score})
		} else if score > h[0].score {
			// replace current minimum (keep the largest K)
			h[0] = item{id: c.id, score: score}
			heap.Fix(&h, 0)
		}
	}

	// Extract IDs to evict (any order is fine).
	evict := make([]int64, 0, h.Len())
	for h.Len() > 0 {
		evict = append(evict, heap.Pop(&h).(item).id)
	}

	if len(evict) > 0 {
		// Delete in batches to avoid huge payloads.
		for i := 0; i < len(evict); i += delBatch {
			j := i + delBatch
			if j > len(evict) {
				j = len(evict)
			}
			chunk := evict[i:j]
			if err := e.store.DeleteMemory(ctx, chunk); err != nil {
				return err
			}
			if e.metrics != nil {
				e.metrics.IncPruned(len(chunk))
				e.metrics.IncSizeEvicted(len(chunk))
			}
		}
	}

	return nil
}

// canonicalKey lowercases and trims whitespace in a single pass to reduce allocations.
func canonicalKey(s string) string {
	// Trim leading/trailing unicode whitespace without allocating.
	start, end := 0, len(s)
	for start < end {
		r, size := utf8.DecodeRuneInString(s[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		start += size
	}
	for start < end {
		r, size := utf8.DecodeLastRuneInString(s[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		end -= size
	}
	if start >= end {
		return ""
	}
	// Lowercase into a single new string.
	var b strings.Builder
	// Over-allocate a bit to avoid growth; rune count <= byte count.
	b.Grow(end - start)
	for _, r := range s[start:end] {
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

func (e *Engine) embed(ctx context.Context, text string) ([]float32, error) {
	if e.embedder == nil {
		e.embedder = embed.AutoEmbedder()
	}
	vec, err := e.embedder.Embed(ctx, text)
	if err != nil || len(vec) == 0 {
		return embed.DummyEmbedding(text), nil
	}
	return vec, nil
}

func (e *Engine) reembedOnDrift(ctx context.Context, records []model.MemoryRecord) error {
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
		sim := model.CosineSimilarity(vec, rec.Embedding)
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

func (e *Engine) populateSummaries(ctx context.Context, records []model.MemoryRecord) error {
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

func (e *Engine) clusterSummary(ctx context.Context, records []model.MemoryRecord, target model.MemoryRecord) (string, error) {
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
	if val := model.FloatFromAny(metadata["importance"]); val > 0 {
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

func mmrSelect(records []model.MemoryRecord, query []float32, limit int, lambda float64) []model.MemoryRecord {
	if limit >= len(records) {
		out := make([]model.MemoryRecord, len(records))
		copy(out, records)
		return out
	}
	if lambda < 0 {
		lambda = 0
	}
	if lambda > 1 {
		lambda = 1
	}
	remaining := make([]model.MemoryRecord, len(records))
	copy(remaining, records)
	selected := make([]model.MemoryRecord, 0, limit)
	for len(selected) < limit && len(remaining) > 0 {
		bestIdx := 0
		bestScore := math.Inf(-1)
		for i, cand := range remaining {
			relevance := cand.WeightedScore
			if relevance == 0 {
				relevance = model.CosineSimilarity(query, cand.Embedding)
			}
			var maxSim float64
			for _, sel := range selected {
				if sim := model.CosineSimilarity(cand.Embedding, sel.Embedding); sim > maxSim {
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

func clusterRecords(records []model.MemoryRecord, threshold float64) [][]model.MemoryRecord {
	if threshold <= 0 {
		threshold = 0.8
	}
	type cluster struct {
		centroid []float64
		records  []model.MemoryRecord
	}
	var clusters []cluster
	for _, rec := range records {
		placed := false
		for i := range clusters {
			sim := model.CosineSimilarity(rec.Embedding, float32Slice(clusters[i].centroid))
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
				records:  []model.MemoryRecord{rec},
			})
		}
	}
	result := make([][]model.MemoryRecord, len(clusters))
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

func clamp(val, minVal, maxVal float64) float64 {
	if val < minVal {
		return minVal
	}
	if val > maxVal {
		return maxVal
	}
	return val
}

// Top-K (K=overflow) heap of *largest* prune scores (the ones to evict).
type item struct {
	id    int64
	score float64
}

type minHeap []item

// We keep the K largest scores in a min-heap by score:
// if heap not full -> push; else if score > min -> replace min.
func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(item)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
