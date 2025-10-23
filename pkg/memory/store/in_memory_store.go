package store

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory/model"
)

// InMemoryStore implements VectorStore for tests and lightweight deployments.
type InMemoryStore struct {
	mu      sync.RWMutex
	nextID  int64
	records map[int64]model.MemoryRecord
}

func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{records: make(map[int64]model.MemoryRecord)}
}

func (s *InMemoryStore) StoreMemory(_ context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.records == nil {
		s.records = make(map[int64]model.MemoryRecord)
	}
	now := time.Now().UTC()
	importance, source, summary, lastEmbedded, metadataJSON := model.NormalizeMetadata(metadata, now)
	s.nextID++
	record := model.MemoryRecord{
		ID:              s.nextID,
		SessionID:       sessionID,
		Content:         content,
		Metadata:        metadataJSON,
		Embedding:       append([]float32(nil), embedding...),
		MultiEmbeddings: model.Float32MatrixFromAny(model.DecodeMetadata(metadataJSON)["multi_embeddings"]),
		Importance:      importance,
		Source:          source,
		Summary:         summary,
		CreatedAt:       now,
		LastEmbedded:    lastEmbedded,
	}
	s.records[record.ID] = record
	return nil
}

// StoreMemoryMulti persists a record with multiple embeddings.
func (s *InMemoryStore) StoreMemoryMulti(ctx context.Context, sessionID, content string, metadata map[string]any, embeddings [][]float32) error {
	primary := []float32(nil)
	if len(embeddings) > 0 {
		primary = embeddings[0]
	}
	if metadata == nil {
		metadata = map[string]any{}
	}
	if len(embeddings) > 1 {
		extras := cloneMatrix(embeddings[1:])
		metadata["multi_embeddings"] = extras
	} else {
		delete(metadata, "multi_embeddings")
	}
	return s.StoreMemory(ctx, sessionID, content, metadata, primary)
}

func (s *InMemoryStore) SearchMemory(_ context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit <= 0 {
		return nil, nil
	}
	type scored struct {
		rec   model.MemoryRecord
		score float64
	}
	scoredRecords := make([]scored, 0, len(s.records))
	for _, rec := range s.records {
		score := model.CosineSimilarity(queryEmbedding, rec.Embedding)
		rec.Score = score
		rec.MultiEmbeddings = cloneMatrix(rec.MultiEmbeddings)
		scoredRecords = append(scoredRecords, scored{rec: rec, score: score})
	}
	sort.Slice(scoredRecords, func(i, j int) bool {
		return scoredRecords[i].score > scoredRecords[j].score
	})
	if len(scoredRecords) > limit {
		scoredRecords = scoredRecords[:limit]
	}
	result := make([]model.MemoryRecord, len(scoredRecords))
	for i, sc := range scoredRecords {
		result[i] = sc.rec
	}
	return result, nil
}

// SearchMemoryMulti scores records against multiple query embeddings.
func (s *InMemoryStore) SearchMemoryMulti(_ context.Context, queryEmbeddings [][]float32, limit int) ([]model.MemoryRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit <= 0 {
		return nil, nil
	}
	if len(queryEmbeddings) == 0 {
		return nil, nil
	}
	type scored struct {
		rec   model.MemoryRecord
		score float64
	}
	scoredRecords := make([]scored, 0, len(s.records))
	for _, rec := range s.records {
		vectors := rec.AllEmbeddings()
		score := model.MaxCosineSimilarity(queryEmbeddings, vectors)
		rec.Score = score
		rec.MultiEmbeddings = cloneMatrix(rec.MultiEmbeddings)
		scoredRecords = append(scoredRecords, scored{rec: rec, score: score})
	}
	sort.Slice(scoredRecords, func(i, j int) bool {
		return scoredRecords[i].score > scoredRecords[j].score
	})
	if len(scoredRecords) > limit {
		scoredRecords = scoredRecords[:limit]
	}
	result := make([]model.MemoryRecord, len(scoredRecords))
	for i, sc := range scoredRecords {
		result[i] = sc.rec
	}
	return result, nil
}

func (s *InMemoryStore) UpdateEmbedding(_ context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	rec, ok := s.records[id]
	if !ok {
		return errors.New("memory not found")
	}
	rec.Embedding = append([]float32(nil), embedding...)
	rec.LastEmbedded = lastEmbedded
	s.records[id] = rec
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

func (s *InMemoryStore) Iterate(_ context.Context, fn func(model.MemoryRecord) bool) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := make([]int64, 0, len(s.records))
	for id := range s.records {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return s.records[ids[i]].CreatedAt.Before(s.records[ids[j]].CreatedAt) })
	for _, id := range ids {
		rec := s.records[id]
		rec.MultiEmbeddings = cloneMatrix(rec.MultiEmbeddings)
		if !fn(rec) {
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

func cloneMatrix(src [][]float32) [][]float32 {
	if len(src) == 0 {
		return nil
	}
	out := make([][]float32, len(src))
	for i, vec := range src {
		if len(vec) == 0 {
			continue
		}
		cp := make([]float32, len(vec))
		copy(cp, vec)
		out[i] = cp
	}
	return out
}
