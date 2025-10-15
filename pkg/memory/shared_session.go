package memory

import (
	"context"
	"encoding/json"
	"errors"
	"sort"
	"strings"
	"sync"
)

// SharedSession layers on top of SessionMemory to let multiple agents
// (with distinct local sessionIDs) share BOTH short-term and long-term
// memory via one or more shared space IDs (e.g. "team:alpha").
//
// Design notes:
//   - Short-term sharing: we write/read directly to SessionMemory.shortTerm
//     but across a set of allowed sessionIDs (local + shared spaces).
//   - Long-term sharing: we store/retrieve using the underlying Engine/Bank
//     but filter results to allowed sessionIDs.
//   - No changes are required in existing stores. This is a thin, safe wrapper.
//
// Typical usage:
//   ss := NewSharedSession(sm, "agent-A", "team:project-x")
//   ss.Join("guild:search")
//   ss.AddShortLocal("Parsed sitemap of example.com", map[string]string{"role":"tool"})
//   ctx := context.Background()
//   ctxRecs, _ := ss.Retrieve(ctx, "crawl schedule", 10)
//   _ = ss.FlushLocal(ctx) // promote local short-term to long-term
//
// You can also write long-term memories *to a space* so all agents see it:
//   _, _ = ss.StoreLongTo(ctx, "team:project-x", "PRD v1 approved", nil)

type SharedSession struct {
	base     *SessionMemory
	local    string
	mu       sync.RWMutex
	joined   map[string]struct{}
	registry *SpaceRegistry
}

// NewSharedSession binds a local sessionID and optional initial shared spaces.
func NewSharedSession(base *SessionMemory, local string, spaces ...string) *SharedSession {
	set := make(map[string]struct{}, len(spaces))
	for _, s := range spaces {
		if strings.TrimSpace(s) == "" {
			continue
		}
		set[s] = struct{}{}
	}
	ss := &SharedSession{base: base, local: local, joined: set}
	if base != nil {
		ss.registry = base.Spaces
	}
	return ss
}

// Join adds a shared space (sessionID) to the view.
func (ss *SharedSession) Join(space string) error {
	space = strings.TrimSpace(space)
	if space == "" {
		return errors.New("space is empty")
	}
	if ss.registry != nil {
		if err := ss.registry.Check(space, ss.local, false); err != nil {
			return err
		}
	}
	ss.mu.Lock()
	defer ss.mu.Unlock()
	if ss.joined == nil {
		ss.joined = map[string]struct{}{}
	}
	ss.joined[space] = struct{}{}
	return nil
}

// Leave removes a shared space (sessionID) from the view.
func (ss *SharedSession) Leave(space string) {
	ss.mu.Lock()
	defer ss.mu.Unlock()
	delete(ss.joined, space)
}

// Spaces returns the current list of shared spaces.
func (ss *SharedSession) Spaces() []string {
	ss.mu.RLock()
	snapshot := make([]string, 0, len(ss.joined))
	for s := range ss.joined {
		snapshot = append(snapshot, s)
	}
	ss.mu.RUnlock()

	filtered := make([]string, 0, len(snapshot))
	for _, s := range snapshot {
		if ss.canRead(s) {
			filtered = append(filtered, s)
		}
	}
	sort.Strings(filtered)
	return filtered
}

// AddShortLocal appends a short-term memory to the *local* session buffer.
func (ss *SharedSession) AddShortLocal(content string, metadata map[string]string) {
	if ss == nil || ss.base == nil || strings.TrimSpace(content) == "" {
		return
	}
	metaBytes, _ := json.Marshal(metadata)
	// Compute an embedding using the session embedder.
	emb, err := ss.base.Embed(context.Background(), content)
	if err != nil {
		return
	}
	ss.base.AddShortTerm(ss.local, content, string(metaBytes), emb)
}

// AddShortTo writes a short-term memory directly into a shared space buffer.
func (ss *SharedSession) AddShortTo(space, content string, metadata map[string]string) error {
	if ss == nil || ss.base == nil {
		return errors.New("nil shared session")
	}
	space = strings.TrimSpace(space)
	if space == "" {
		return errors.New("space is empty")
	}
	if strings.TrimSpace(content) == "" {
		return errors.New("content is empty")
	}
	if !ss.canWrite(space) {
		return ErrSpaceForbidden
	}
	metaBytes, _ := json.Marshal(metadata)
	emb, err := ss.base.Embed(context.Background(), content)
	if err != nil {
		return err
	}
	ss.base.AddShortTerm(space, content, string(metaBytes), emb)
	return nil
}

// FlushLocal promotes the local short-term buffer to long-term storage.
func (ss *SharedSession) FlushLocal(ctx context.Context) error {
	if ss == nil || ss.base == nil {
		return nil
	}
	return ss.base.FlushToLongTerm(ctx, ss.local)
}

// FlushSpace promotes a shared space short-term buffer to long-term.
func (ss *SharedSession) FlushSpace(ctx context.Context, space string) error {
	if ss == nil || ss.base == nil {
		return nil
	}
	space = strings.TrimSpace(space)
	if space == "" {
		return errors.New("space is empty")
	}
	if !ss.canWrite(space) {
		return ErrSpaceForbidden
	}
	return ss.base.FlushToLongTerm(ctx, space)
}

// StoreLongTo writes a long-term memory directly to a specific session/space.
// If Engine is configured it will be used; otherwise we fall back to Bank + Embed.
func (ss *SharedSession) StoreLongTo(ctx context.Context, sessionID, content string, metadata map[string]any) (MemoryRecord, error) {
	if ss == nil || ss.base == nil {
		return MemoryRecord{}, errors.New("nil shared session")
	}
	if strings.TrimSpace(content) == "" {
		return MemoryRecord{}, errors.New("content is empty")
	}
	if sessionID != ss.local && !ss.canWrite(sessionID) {
		return MemoryRecord{}, ErrSpaceForbidden
	}
	if ss.base.Engine != nil {
		return ss.base.Engine.Store(ctx, sessionID, content, metadata)
	}
	// Bank-only path: compute embedding and store.
	emb, err := ss.base.Embed(ctx, content)
	if err != nil {
		return MemoryRecord{}, err
	}
	metaBytes, _ := json.Marshal(metadata)
	if err := ss.base.Bank.StoreMemory(ctx, sessionID, content, string(metaBytes), emb); err != nil {
		return MemoryRecord{}, err
	}
	// Best-effort record (ID may be zero if not re-fetched from store).
	return MemoryRecord{SessionID: sessionID, Content: content, Metadata: string(metaBytes), Embedding: emb}, nil
}

// BroadcastLong writes a long-term memory to the local session and all spaces.
func (ss *SharedSession) BroadcastLong(ctx context.Context, content string, metadata map[string]any) ([]MemoryRecord, error) {
	if ss == nil || ss.base == nil {
		return nil, errors.New("nil shared session")
	}
	targets := ss.allowedWriteSessions()
	out := make([]MemoryRecord, 0, len(targets))
	for _, sid := range targets {
		rec, err := ss.StoreLongTo(ctx, sid, content, metadata)
		if err != nil {
			return nil, err
		}
		out = append(out, rec)
	}
	return out, nil
}

// Retrieve merges short-term (local + spaces) and filtered long-term results.
// Long-term retrieval oversamples then filters to the allowed sessionIDs.
func (ss *SharedSession) Retrieve(ctx context.Context, query string, limit int) ([]MemoryRecord, error) {
	if ss == nil || ss.base == nil || limit <= 0 {
		return nil, nil
	}
	allowed := make(map[string]struct{})
	for _, sid := range ss.allowedReadSessions() {
		allowed[sid] = struct{}{}
	}
	// 1) Collect short-term across all allowed sessions.
	ss.base.mu.RLock()
	var short []MemoryRecord
	for sid := range allowed {
		if buf := ss.base.shortTerm[sid]; len(buf) > 0 {
			short = append(short, buf...)
		}
	}
	ss.base.mu.RUnlock()

	// 2) Retrieve long-term (oversample, then filter by SessionID).
	oversample := limit * 6
	if oversample < limit {
		oversample = limit
	}
	var long []MemoryRecord
	if ss.base.Engine != nil {
		recs, err := ss.base.Engine.Retrieve(ctx, query, oversample)
		if err != nil {
			return nil, err
		}
		for _, r := range recs {
			if _, ok := allowed[r.SessionID]; ok {
				long = append(long, r)
			}
		}
	} else if ss.base.Bank != nil {
		emb, err := ss.base.Embed(ctx, query)
		if err != nil {
			return nil, err
		}
		recs, err := ss.base.Bank.SearchMemory(ctx, emb, oversample)
		if err != nil {
			return nil, err
		}
		for _, r := range recs {
			if _, ok := allowed[r.SessionID]; ok {
				long = append(long, r)
			}
		}
	}

	// 3) Deduplicate by ID (when present) and by (session,content) as fallback.
	seen := make(map[int64]struct{})
	seenKey := make(map[string]struct{})
	push := func(dst *[]MemoryRecord, rec MemoryRecord) {
		if rec.ID != 0 {
			if _, ok := seen[rec.ID]; ok {
				return
			}
			seen[rec.ID] = struct{}{}
		} else {
			key := rec.SessionID + "\u241F" + strings.TrimSpace(rec.Content)
			if _, ok := seenKey[key]; ok {
				return
			}
			seenKey[key] = struct{}{}
		}
		*dst = append(*dst, rec)
	}
	var merged []MemoryRecord
	for _, r := range short {
		push(&merged, r)
	}
	for _, r := range long {
		push(&merged, r)
	}

	// 4) If we have more than limit, keep short-term first, then most relevant long-term that remain.
	if len(merged) <= limit {
		return merged, nil
	}
	// Keep all short-term if possible, then top long-term until limit.
	shortCount := len(short)
	if shortCount >= limit {
		return merged[:limit], nil
	}
	// We already appended short first; trim to keep them and the best-scored long that followed.
	return merged[:limit], nil
}

// allowedReadSessions returns local + joined spaces with read access.
func (ss *SharedSession) allowedReadSessions() []string {
	ss.mu.RLock()
	spaces := make([]string, 0, len(ss.joined))
	for s := range ss.joined {
		spaces = append(spaces, s)
	}
	ss.mu.RUnlock()
	out := []string{ss.local}
	for _, space := range spaces {
		if ss.canRead(space) {
			out = append(out, space)
		}
	}
	return out
}

// allowedWriteSessions returns local + joined spaces with write access.
func (ss *SharedSession) allowedWriteSessions() []string {
	ss.mu.RLock()
	spaces := make([]string, 0, len(ss.joined))
	for s := range ss.joined {
		spaces = append(spaces, s)
	}
	ss.mu.RUnlock()
	out := []string{ss.local}
	for _, space := range spaces {
		if ss.canWrite(space) {
			out = append(out, space)
		}
	}
	return out
}

func (ss *SharedSession) canRead(space string) bool {
	if ss.registry == nil {
		return true
	}
	if space == ss.local {
		return true
	}
	return ss.registry.CanRead(space, ss.local)
}

func (ss *SharedSession) canWrite(space string) bool {
	if ss.registry == nil {
		return true
	}
	if space == ss.local {
		return true
	}
	return ss.registry.CanWrite(space, ss.local)
}
