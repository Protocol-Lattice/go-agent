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
	records map[int64]model.MemoryRecord
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
	return &InMemoryStore{records: make(map[int64]model.MemoryRecord)}
}

func (s *InMemoryStore) StoreMemory(_ context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.records == nil {
		s.records = make(map[int64]model.MemoryRecord)
	}
	now := time.Now().UTC()
	record := prepareMemoryRecord(sessionID, content, metadata, embedding, now, false)
	s.nextID++
	record.ID = s.nextID
	s.records[record.ID] = record
	return nil
}

func (s *InMemoryStore) SearchMemory(_ context.Context, sessionID string, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if limit <= 0 {
		return nil, nil
	}
	scoredRecords := make(topMemoryRecords, 0, min(limit, len(s.records)))
	for _, rec := range s.records {
		if sessionID != "" && rec.SessionID != sessionID {
			continue
		}
		score := model.MaxCosineSimilarity(queryEmbedding, rec)
		rec.Score = score
		candidate := scoredMemoryRecord{record: rec, score: score}
		if len(scoredRecords) < limit {
			scoredRecords.push(candidate)
			continue
		}
		if candidate.score > scoredRecords[0].score {
			scoredRecords.replaceMin(candidate)
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
