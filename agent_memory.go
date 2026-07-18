package agent

import (
	"context"
	"encoding/json"
	"strings"
	"sync"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory"
)

type memoryStoreTask struct {
	once  sync.Once
	ready <-chan preparedMemoryStore
}

type preparedMemoryStore struct {
	agent       *Agent
	shared      *memory.SharedSession
	memory      *memory.SessionMemory
	sessionID   string
	content     string
	metadata    map[string]string
	metadataRaw string
	embedding   []float32
	embedded    bool
}

// Flush persists session memory into the long-term store.
func (a *Agent) Flush(ctx context.Context, sessionID string) error {
	return a.memory.FlushToLongTerm(ctx, sessionID)
}

// Checkpoint serializes the agent's current state (system prompt and short-term memory)
// to a byte slice. This can be saved to disk or a database to pause the agent.
func (a *Agent) Checkpoint() ([]byte, error) {
	a.mu.Lock()
	defer a.mu.Unlock()

	state := AgentState{
		SystemPrompt: a.systemPrompt,
		ShortTerm:    a.memory.ExportShortTerm(),
		Timestamp:    time.Now(),
	}

	if a.Shared != nil {
		state.JoinedSpaces = a.Shared.ExportJoinedSpaces()
	}

	return json.Marshal(state)
}

// Restore rehydrates the agent's state from a checkpoint.
// It restores the system prompt and short-term memory.
func (a *Agent) Restore(data []byte) error {
	a.mu.Lock()
	defer a.mu.Unlock()

	var state AgentState
	if err := json.Unmarshal(data, &state); err != nil {
		return err
	}

	a.systemPrompt = state.SystemPrompt
	a.memory.ImportShortTerm(state.ShortTerm)

	if a.Shared != nil && len(state.JoinedSpaces) > 0 {
		a.Shared.ImportJoinedSpaces(state.JoinedSpaces)
	}

	return nil
}

func (a *Agent) storeMemory(sessionID, role, content string, extra map[string]string) {
	prepared, ok := a.prepareMemoryStore(sessionID, role, content, extra)
	if !ok {
		return
	}
	prepared.embed()
	prepared.commit()
}

// startMemoryStore computes the session embedding in the background. Wait
// commits the prepared record, retaining the synchronous visibility and
// ordering guarantees of storeMemory while allowing callers to overlap the
// expensive embedding request with model work.
func (a *Agent) startMemoryStore(sessionID, role, content string, extra map[string]string) *memoryStoreTask {
	prepared, ok := a.prepareMemoryStore(sessionID, role, content, extra)
	if !ok {
		return nil
	}

	ready := make(chan preparedMemoryStore, 1)
	task := &memoryStoreTask{ready: ready}
	if prepared.memory == nil || prepared.memory.Embedder == nil {
		ready <- prepared
		return task
	}

	go func() {
		prepared.embed()
		ready <- prepared
	}()

	return task
}

func (a *Agent) prepareMemoryStore(sessionID, role, content string, extra map[string]string) (preparedMemoryStore, bool) {
	if a == nil || strings.TrimSpace(content) == "" {
		return preparedMemoryStore{}, false
	}

	// Build metadata safely.
	meta := map[string]string{}
	if rs := strings.TrimSpace(role); rs != "" {
		meta["role"] = rs
	}
	if extra != nil {
		for k, v := range extra {
			ks, vs := strings.TrimSpace(k), strings.TrimSpace(v)
			if ks != "" && vs != "" {
				meta[ks] = vs
			}
		}
	}

	metaBytes, _ := json.Marshal(meta)

	// Snapshot pointers without holding the lock during external calls.
	a.mu.Lock()
	shared := a.Shared
	mem := a.memory
	a.mu.Unlock()

	return preparedMemoryStore{
		agent:       a,
		shared:      shared,
		memory:      mem,
		sessionID:   sessionID,
		content:     content,
		metadata:    meta,
		metadataRaw: string(metaBytes),
	}, true
}

func (p *preparedMemoryStore) embed() {
	if p.memory == nil || p.memory.Embedder == nil {
		return
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	embedding, err := p.memory.Embedder.Embed(ctx, p.content)
	if err == nil {
		p.embedding = embedding
		p.embedded = true
	}
}

// Wait commits a background memory store exactly once. It is safe to call on
// a nil task and safe to call repeatedly (for example, explicitly and via a
// deferred cleanup on error paths).
func (t *memoryStoreTask) Wait() {
	if t == nil {
		return
	}
	t.once.Do(func() {
		prepared := <-t.ready
		prepared.commit()
	})
}

func (p preparedMemoryStore) commit() {
	// Best-effort writes to shared spaces retain the existing behavior. Shared
	// sessions own their embedding policy, so these calls remain synchronous.
	if p.shared != nil {
		p.shared.AddShortLocal(p.content, p.metadata)
		for _, space := range p.shared.Spaces() {
			_ = p.shared.AddShortTo(space, p.content, p.metadata)
		}
	}

	if p.memory == nil || !p.embedded {
		return
	}

	p.agent.mu.Lock()
	defer p.agent.mu.Unlock()
	p.memory.AddShortTerm(p.sessionID, p.content, p.metadataRaw, p.embedding)
}

func (a *Agent) retrieveContext(ctx context.Context, sessionID, query string, limit int) ([]memory.MemoryRecord, error) {
	if a.Shared != nil {
		return a.Shared.Retrieve(ctx, query, limit)
	}
	return a.memory.RetrieveContext(ctx, sessionID, query, limit)
}

func metadataRole(metadata string) string {
	if metadata == "" {
		return "unknown"
	}
	var payload map[string]any
	if err := json.Unmarshal([]byte(metadata), &payload); err != nil {
		return "unknown"
	}
	if role, ok := payload["role"].(string); ok && role != "" {
		return role
	}
	return "unknown"
}

func (a *Agent) SetSharedSpaces(shared *memory.SharedSession) {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.Shared = shared
}

// EnsureSpaceGrants gives the provided sessionID writer access to each space.
// This mirrors how tests set up spaces: mem.Spaces.Grant(space, session, role, ttl).
func (a *Agent) EnsureSpaceGrants(sessionID string, spaces []string) {
	if a == nil || a.memory == nil {
		return
	}
	for _, s := range spaces {
		s = strings.TrimSpace(s)
		if s == "" {
			continue
		}
		a.memory.Spaces.Grant(s, sessionID, memory.SpaceRoleWriter, 0)
	}
}

// SessionMemory exposes the underlying session memory (useful for advanced setup/tests).
func (a *Agent) SessionMemory() *memory.SessionMemory {
	return a.memory
}
