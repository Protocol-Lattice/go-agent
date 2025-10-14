package memory

import (
	"context"
	"errors"
	"sort"
	"sync"
	"time"
)

// InMemoryStore implements VectorStore for tests and lightweight deployments.
type InMemoryStore struct {
	mu      sync.RWMutex
	nextID  int64
	records map[int64]MemoryRecord
}

func NewInMemoryStore() *InMemoryStore {
	return &InMemoryStore{records: make(map[int64]MemoryRecord)}
}

func (s *InMemoryStore) StoreMemory(_ context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.records == nil {
		s.records = make(map[int64]MemoryRecord)
	}
	now := time.Now().UTC()
	importance, source, summary, lastEmbedded, metadataJSON := normalizeMetadata(metadata, now)
	s.nextID++
	record := MemoryRecord{
		ID:           s.nextID,
		SessionID:    sessionID,
		Content:      content,
		Metadata:     metadataJSON,
		Embedding:    append([]float32(nil), embedding...),
		Importance:   importance,
		Source:       source,
		Summary:      summary,
		CreatedAt:    now,
		LastEmbedded: lastEmbedded,
	}
	s.records[record.ID] = record
	return nil
}

func (s *InMemoryStore) SearchMemory(_ context.Context, queryEmbedding []float32, limit int) ([]MemoryRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit <= 0 {
		return nil, nil
	}
	type scored struct {
		rec   MemoryRecord
		score float64
	}
	scoredRecords := make([]scored, 0, len(s.records))
	for _, rec := range s.records {
		score := cosineSimilarity(queryEmbedding, rec.Embedding)
		rec.Score = score
		scoredRecords = append(scoredRecords, scored{rec: rec, score: score})
	}
	sort.Slice(scoredRecords, func(i, j int) bool {
		return scoredRecords[i].score > scoredRecords[j].score
	})
	if len(scoredRecords) > limit {
		scoredRecords = scoredRecords[:limit]
	}
	result := make([]MemoryRecord, len(scoredRecords))
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

func (s *InMemoryStore) Iterate(_ context.Context, fn func(MemoryRecord) bool) error {
	s.mu.RLock()
	defer s.mu.RUnlock()
	ids := make([]int64, 0, len(s.records))
	for id := range s.records {
		ids = append(ids, id)
	}
	sort.Slice(ids, func(i, j int) bool { return s.records[ids[i]].CreatedAt.Before(s.records[ids[j]].CreatedAt) })
	for _, id := range ids {
		if !fn(s.records[id]) {
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
