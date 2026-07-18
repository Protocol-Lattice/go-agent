package store

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

// InMemoryStore implements VectorStore for tests and lightweight deployments.
type InMemoryStore struct {
	mu      sync.RWMutex
	nextID  int64
	records map[int64]*inMemoryRecord
}

// inMemoryRecord keeps search-only derived data beside the record so a scan
// needs one map lookup and does not copy the comparatively large MemoryRecord
// value unless that record enters the top-k result heap.
type inMemoryRecord struct {
	record     model.MemoryRecord
	magnitudes recordVectorMagnitudes
}

// recordVectorMagnitudes caches the invariant half of cosine similarity for
// immutable stored embeddings. Matrix magnitudes are only allocated for the
// uncommon records that actually contain a matrix.
type recordVectorMagnitudes struct {
	embedding float64
	matrix    []float64
}

type scoredMemoryRecord struct {
	record model.MemoryRecord
	score  float64
}

// topMemoryRecords is a specialized min-heap that retains only the best k
// records while scanning the in-memory store. Keeping it typed avoids the
// interface allocations used by container/heap on a hot retrieval path.
type topMemoryRecords []scoredMemoryRecord

func (h *topMemoryRecords) push(item scoredMemoryRecord) {
	items := append(*h, item)
	child := len(items) - 1
	for child > 0 {
		parent := (child - 1) / 2
		if items[parent].score <= item.score {
			break
		}
		items[child] = items[parent]
		child = parent
	}
	items[child] = item
	*h = items
}

func (h topMemoryRecords) replaceMin(item scoredMemoryRecord) {
	parent := 0
	for {
		child := parent*2 + 1
		if child >= len(h) {
			break
		}
		if right := child + 1; right < len(h) && h[right].score < h[child].score {
			child = right
		}
		if h[child].score >= item.score {
			break
		}
		h[parent] = h[child]
		parent = child
	}
	h[parent] = item
}

func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{
		records: make(map[int64]*inMemoryRecord),
	}
}

func (s *InMemoryStore) StoreMemory(_ context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.records == nil {
		s.records = make(map[int64]*inMemoryRecord)
	}
	now := time.Now().UTC()
	record := prepareMemoryRecord(sessionID, content, metadata, embedding, now, false)
	s.nextID++
	record.ID = s.nextID
	s.records[record.ID] = &inMemoryRecord{
		record:     record,
		magnitudes: calculateRecordMagnitudes(record),
	}
	return nil
}

func (s *InMemoryStore) SearchMemory(_ context.Context, sessionID string, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit <= 0 {
		return nil, nil
	}
	query := model.NewCosineQuery(queryEmbedding)
	scoredRecords := make(topMemoryRecords, 0, min(limit, len(s.records)))
	for _, stored := range s.records {
		rec := &stored.record
		if sessionID != "" && rec.SessionID != sessionID {
			continue
		}
		score := maxSimilarityWithMagnitudes(query, rec, stored.magnitudes)
		if len(scoredRecords) < limit {
			resultRecord := *rec
			resultRecord.Score = score
			scoredRecords.push(scoredMemoryRecord{record: resultRecord, score: score})
			continue
		}
		if score > scoredRecords[0].score {
			resultRecord := *rec
			resultRecord.Score = score
			scoredRecords.replaceMin(scoredMemoryRecord{record: resultRecord, score: score})
		}
	}
	sort.Slice(scoredRecords, func(i, j int) bool {
		return scoredRecords[i].score > scoredRecords[j].score
	})
	result := make([]model.MemoryRecord, len(scoredRecords))
	for i, sc := range scoredRecords {
		result[i] = sc.record
	}
	return result, nil
}

func (s *InMemoryStore) UpdateEmbedding(_ context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	stored, ok := s.records[id]
	if !ok {
		return errors.New("memory not found")
	}
	stored.record.Embedding = append([]float32(nil), embedding...)
	stored.record.LastEmbedded = lastEmbedded
	stored.magnitudes = calculateRecordMagnitudes(stored.record)
	return nil
}

func (s *InMemoryStore) DeleteMemory(_ context.Context, ids []int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		delete(s.records, id)
	}
	return nil
}

func calculateRecordMagnitudes(rec model.MemoryRecord) recordVectorMagnitudes {
	magnitudes := recordVectorMagnitudes{
		embedding: model.VectorMagnitude(rec.Embedding),
	}
	if len(rec.EmbeddingMatrix) > 0 {
		magnitudes.matrix = make([]float64, len(rec.EmbeddingMatrix))
		for i, vector := range rec.EmbeddingMatrix {
			magnitudes.matrix[i] = model.VectorMagnitude(vector)
		}
	}
	return magnitudes
}

func maxSimilarityWithMagnitudes(
	query model.CosineQuery,
	rec *model.MemoryRecord,
	magnitudes recordVectorMagnitudes,
) float64 {
	var (
		best      float64
		hasVector bool
	)
	if len(rec.Embedding) > 0 {
		best = query.SimilarityWithMagnitude(rec.Embedding, magnitudes.embedding)
		hasVector = true
	}
	for i, vector := range rec.EmbeddingMatrix {
		if len(vector) == 0 {
			continue
		}
		magnitude := 0.0
		if i < len(magnitudes.matrix) {
			magnitude = magnitudes.matrix[i]
		}
		similarity := query.SimilarityWithMagnitude(vector, magnitude)
		if !hasVector || similarity > best {
			best = similarity
			hasVector = true
		}
	}
	return best
}

func (s *InMemoryStore) Iterate(_ context.Context, fn func(model.MemoryRecord) bool) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := make([]int64, 0, len(s.records))
	for id := range s.records {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool {
		return s.records[ids[i]].record.CreatedAt.Before(s.records[ids[j]].record.CreatedAt)
	})
	for _, id := range ids {
		if !fn(s.records[id].record) {
			break
		}
	}
	return nil
}

func (s *InMemoryStore) Count(_ context.Context) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.records), nil
}
