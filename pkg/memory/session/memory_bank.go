package session

import (
	"context"
	"encoding/json"
	"io"
	"sync"

	"github.com/Raezil/go-agent-development-kit/pkg/memory/embed"
	memengine "github.com/Raezil/go-agent-development-kit/pkg/memory/engine"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/model"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/store"
)

// MemoryBank is a thin wrapper around a VectorStore implementation.
type MemoryBank struct {
	Store store.VectorStore
}

// SessionMemory wraps MemoryBank with short- and long-term layers
// including a configurable short-term buffer and embedding provider.
type SessionMemory struct {
	Bank          *MemoryBank
	shortTerm     map[string][]model.MemoryRecord
	mu            sync.RWMutex
	shortTermSize int
	Embedder      embed.Embedder
	Engine        *memengine.Engine
	Spaces        *SpaceRegistry
}

// NewMemoryBank creates a new Postgres-backed memory bank.
func NewMemoryBank(ctx context.Context, connStr string) (*MemoryBank, error) {
	s, err := store.NewPostgresStore(ctx, connStr)
	if err != nil {
		return nil, err
	}
	return &MemoryBank{Store: s}, nil
}

// NewMemoryBankWithStore creates a memory bank backed by a custom vector store implementation.
func NewMemoryBankWithStore(s store.VectorStore) *MemoryBank {
	return &MemoryBank{Store: s}
}

// NewSessionMemory wraps MemoryBank with short-term cache
func NewSessionMemory(bank *MemoryBank, shortTermSize int) *SessionMemory {
	return &SessionMemory{
		Bank:          bank,
		shortTerm:     make(map[string][]model.MemoryRecord),
		shortTermSize: shortTermSize,
		Embedder:      embed.AutoEmbedder(),
		Spaces:        NewSpaceRegistry(0),
	}
}

// AddShortTerm stores in ephemeral session cache
func (sm *SessionMemory) AddShortTerm(sessionID, content, metadata string, embedding []float32) {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	record := model.MemoryRecord{SessionID: sessionID, Space: sessionID, Content: content, Metadata: metadata, Embedding: embedding}
	sm.shortTerm[sessionID] = append(sm.shortTerm[sessionID], record)

	if len(sm.shortTerm[sessionID]) > sm.shortTermSize {
		sm.shortTerm[sessionID] = sm.shortTerm[sessionID][len(sm.shortTerm[sessionID])-sm.shortTermSize:]
	}
}

// FlushToLongTerm writes the short-term cache to the configured vector store.
func (sm *SessionMemory) FlushToLongTerm(ctx context.Context, sessionID string) error {
	sm.mu.Lock()
	defer sm.mu.Unlock()

	records := sm.shortTerm[sessionID]
	for _, r := range records {
		if sm.Engine != nil {
			meta := model.DecodeMetadata(r.Metadata)
			if _, err := sm.Engine.Store(ctx, sessionID, r.Content, meta); err != nil {
				return err
			}
		} else {
			if err := sm.Bank.StoreMemory(ctx, sessionID, r.Content, r.Metadata, r.Embedding); err != nil {
				return err
			}
		}
	}
	delete(sm.shortTerm, sessionID)
	return nil
}

// Embed ensures an embedder is available and returns the embedding for the text.
func (sm *SessionMemory) Embed(ctx context.Context, text string) ([]float32, error) {
	if sm.Embedder == nil {
		sm.Embedder = embed.AutoEmbedder()
	}
	vec, err := sm.Embedder.Embed(ctx, text)
	if err != nil || len(vec) == 0 {
		return embed.DummyEmbedding(text), nil
	}
	return vec, nil
}

// RetrieveContext returns combined short- and long-term memory
func (sm *SessionMemory) RetrieveContext(ctx context.Context, sessionID, query string, limit int) ([]model.MemoryRecord, error) {
	var longTerm []model.MemoryRecord
	if sm.Engine != nil {
		records, err := sm.Engine.Retrieve(ctx, query, limit)
		if err != nil {
			return nil, err
		}
		longTerm = records
	} else if sm.Bank != nil {
		queryEmbedding, err := sm.Embed(ctx, query)
		if err != nil {
			return nil, err
		}
		records, err := sm.Bank.SearchMemory(ctx, queryEmbedding, limit)
		if err != nil {
			return nil, err
		}
		longTerm = records
	}

	sm.mu.RLock()
	shortTerm := sm.shortTerm[sessionID]
	sm.mu.RUnlock()

	return append(shortTerm, longTerm...), nil
}

// WithEmbedder overrides the embedder used by the session memory.
func (sm *SessionMemory) WithEmbedder(e embed.Embedder) *SessionMemory {
	if e != nil {
		sm.Embedder = e
	}
	return sm
}

// WithEngine attaches an advanced memory engine for long-term storage and retrieval.
func (sm *SessionMemory) WithEngine(e *memengine.Engine) *SessionMemory {
	sm.Engine = e
	return sm
}

// StoreMemory inserts a long-term record
func (mb *MemoryBank) StoreMemory(ctx context.Context, sessionID, content, metadata string, embedding []float32) error {
	if mb == nil || mb.Store == nil {
		return nil
	}
	meta := map[string]any{}
	if metadata != "" {
		_ = json.Unmarshal([]byte(metadata), &meta)
	}
	if _, ok := meta["space"]; !ok {
		meta["space"] = sessionID
	}
	return mb.Store.StoreMemory(ctx, sessionID, content, meta, embedding)
}

// SearchMemory returns top-k similar memories
func (mb *MemoryBank) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	if mb == nil || mb.Store == nil {
		return nil, nil
	}
	return mb.Store.SearchMemory(ctx, queryEmbedding, limit)
}

// CreateSchema initialises the backing store if it supports schema management.
func (mb *MemoryBank) CreateSchema(ctx context.Context, schemaPath string) error {
	if mb == nil || mb.Store == nil {
		return nil
	}
	initializer, ok := mb.Store.(store.SchemaInitializer)
	if !ok {
		return nil
	}
	return initializer.CreateSchema(ctx, schemaPath)
}

// Close releases underlying resources if the store implements io.Closer.
func (mb *MemoryBank) Close() error {
	if mb == nil || mb.Store == nil {
		return nil
	}
	if closer, ok := mb.Store.(io.Closer); ok {
		return closer.Close()
	}
	return nil
}
