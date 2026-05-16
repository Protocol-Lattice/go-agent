package main

import (
	"context"
	"encoding/json"
	"errors"
	"log"
	"os"
	"sort"
	"sync"
	"time"

	kit "github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

// FileBackedStore is a simple persistent VectorStore that writes to a local JSON file.
// It is ideal for examples and local testing where you want continuity without running a database.
type FileBackedStore struct {
	filePath string
	mu       sync.RWMutex
	nextID   int64
	records  map[int64]model.MemoryRecord
}

func NewFileBackedStore(filePath string) *FileBackedStore {
	store := &FileBackedStore{
		filePath: filePath,
		records:  make(map[int64]model.MemoryRecord),
	}
	store.load()
	return store
}

func (s *FileBackedStore) load() {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.filePath)
	if err != nil {
		if !os.IsNotExist(err) {
			println("Warning: failed to read memory file:", err.Error())
		}
		return
	}

	var records []model.MemoryRecord
	if err := json.Unmarshal(data, &records); err != nil {
		println("Warning: failed to unmarshal memory file:", err.Error())
		return
	}

	for _, rec := range records {
		s.records[rec.ID] = rec
		if rec.ID > s.nextID {
			s.nextID = rec.ID
		}
	}
}

func (s *FileBackedStore) save() error {
	var records []model.MemoryRecord
	for _, rec := range s.records {
		records = append(records, rec)
	}

	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(s.filePath, data, 0644)
}

func (s *FileBackedStore) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.records == nil {
		s.records = make(map[int64]model.MemoryRecord)
	}
	now := time.Now().UTC()
	importance, source, summary, lastEmbedded, metadataJSON := model.NormalizeMetadata(metadata, now)
	meta := model.DecodeMetadata(metadataJSON)
	space := model.StringFromAny(meta["space"])
	if space == "" {
		space = sessionID
	}
	matrix := model.ValidEmbeddingMatrix(meta)
	storedEmbedding := append([]float32(nil), embedding...)
	if len(storedEmbedding) == 0 {
		for _, vec := range matrix {
			if len(vec) == 0 {
				continue
			}
			storedEmbedding = append([]float32(nil), vec...)
			break
		}
	}

	s.nextID++
	record := model.MemoryRecord{
		ID:              s.nextID,
		SessionID:       sessionID,
		Space:           space,
		Content:         content,
		Metadata:        metadataJSON,
		Embedding:       storedEmbedding,
		Importance:      importance,
		Source:          source,
		Summary:         summary,
		CreatedAt:       now,
		LastEmbedded:    lastEmbedded,
		GraphEdges:      model.ValidGraphEdges(meta),
		EmbeddingMatrix: matrix,
	}

	s.records[record.ID] = record
	return s.save()
}

func (s *FileBackedStore) SearchMemory(ctx context.Context, sessionID string, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
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
		if sessionID != "" && rec.SessionID != sessionID {
			continue
		}
		score := model.MaxCosineSimilarity(queryEmbedding, rec)
		rec.Score = score
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

func (s *FileBackedStore) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	rec, ok := s.records[id]
	if !ok {
		return errors.New("memory not found")
	}

	rec.Embedding = append([]float32(nil), embedding...)
	rec.LastEmbedded = lastEmbedded
	s.records[id] = rec

	return s.save()
}

func (s *FileBackedStore) DeleteMemory(ctx context.Context, ids []int64) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for _, id := range ids {
		delete(s.records, id)
	}

	return s.save()
}

func (s *FileBackedStore) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
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

func (s *FileBackedStore) Count(ctx context.Context) (int, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return len(s.records), nil
}

func FileBackedMemoryModule(filePath string, window int, embedder memory.Embedder, opts *memory.Options) *modules.MemoryModule {
	size := window
	if size <= 0 {
		size = 8
	}
	bank := memory.NewMemoryBankWithStore(NewFileBackedStore(filePath))
	mem := memory.NewSessionMemory(bank, size)
	mem.WithEmbedder(embedder)

	engineOpts := memory.DefaultOptions()
	if opts != nil {
		engineOpts = *opts
	}

	memoryEngine := memory.NewEngine(bank.Store, engineOpts)
	mem.WithEngine(memoryEngine)

	engineLogger := log.New(os.Stderr, "memory-engine: ", log.LstdFlags)
	mem.Engine.WithLogger(engineLogger)

	provider := func(context.Context) (kit.MemoryBundle, error) {
		shared := func(local string, spaces ...string) *memory.SharedSession {
			return memory.NewSharedSession(mem, local, spaces...)
		}
		return kit.MemoryBundle{Session: mem, Shared: shared}, nil
	}

	return modules.NewMemoryModule("file_memory", provider)
}
