package markdown

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

type Store struct {
	root    string
	mu      sync.RWMutex
	cacheMu sync.Mutex
	cache   map[string]cachedMarkdownFile
}

type cachedMarkdownFile struct {
	size    int64
	modTime time.Time
	records []Record
}

func NewStore(root string) (*Store, error) {
	if root == "" {
		root = ".agent-memory"
	}

	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}

	s := &Store{root: abs, cache: make(map[string]cachedMarkdownFile)}

	for _, dir := range []string{
		filepath.Join(abs, "sessions"),
		filepath.Join(abs, "spaces"),
	} {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}

	return s, nil
}

func (s *Store) Root() string {
	return s.root
}

func (s *Store) Save(ctx context.Context, rec Record) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	rec = rec.normalized()
	if rec.Content == "" {
		return errors.New("memory content is empty")
	}

	path, err := s.pathFor(rec.Scope, rec.SessionID)
	if err != nil {
		return err
	}

	block := renderBlock(rec)

	s.mu.Lock()
	defer s.mu.Unlock()

	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		return err
	}

	needHeader := false
	if _, err := os.Stat(path); errors.Is(err, os.ErrNotExist) {
		needHeader = true
	} else if err != nil {
		return err
	}

	f, err := os.OpenFile(path, os.O_CREATE|os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		return err
	}
	defer f.Close()

	if needHeader {
		if _, err := fmt.Fprintf(f, "# %s: %s\n\n", title(rec.Scope), rec.SessionID); err != nil {
			return err
		}
	}

	_, err = f.WriteString(block)
	if err == nil {
		s.invalidateFile(path)
	}
	return err
}

func (s *Store) List(ctx context.Context, scope, sessionID string) ([]Record, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	path, err := s.pathFor(scope, sessionID)
	if err != nil {
		return nil, err
	}

	s.mu.RLock()
	defer s.mu.RUnlock()

	info, err := os.Stat(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	records, err := s.recordsForFile(path, info)
	if err != nil {
		return nil, err
	}
	out := make([]Record, len(records))
	for i := range records {
		out[i] = records[i].clone()
	}
	return out, nil
}

func (s *Store) Search(ctx context.Context, query string, limit int) ([]Record, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 8
	}

	query = strings.TrimSpace(strings.ToLower(query))
	if query == "" {
		return nil, nil
	}

	all, err := s.all(ctx)
	if err != nil {
		return nil, err
	}

	scored := scoreRecords(all, query, limit)

	out := make([]Record, 0, len(scored))
	for _, item := range scored {
		out = append(out, item.Record.clone())
	}

	return out, nil
}

func (s *Store) Delete(ctx context.Context, id string) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	id = strings.TrimSpace(id)
	if id == "" {
		return errors.New("missing memory id")
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	return filepath.WalkDir(s.root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}

		info, err := entry.Info()
		if err != nil {
			return err
		}
		records, err := s.recordsForFile(path, info)
		if err != nil {
			return err
		}
		var kept []Record
		changed := false

		for _, rec := range records {
			if rec.ID == id {
				changed = true
				continue
			}
			kept = append(kept, rec)
		}

		if !changed {
			return nil
		}

		var builder strings.Builder

		// FIX: Reconstruct header using scope/sessionID from kept records
		// (or use defaults if all records were deleted)
		var scope, sessionID string
		if len(kept) > 0 {
			scope = kept[0].Scope
			sessionID = kept[0].SessionID
		} else {
			scope = "sessions"
			sessionID = "default"
		}
		fmt.Fprintf(&builder, "# %s: %s\n\n", title(scope), sessionID)

		for _, rec := range kept {
			builder.WriteString(renderBlock(rec))
		}

		if err := os.WriteFile(path, []byte(builder.String()), 0o644); err != nil {
			return err
		}
		s.invalidateFile(path)
		return nil
	})
}

func (s *Store) all(ctx context.Context) ([]Record, error) {
	var out []Record
	visited := make(map[string]struct{})

	s.mu.RLock()
	defer s.mu.RUnlock()

	err := filepath.WalkDir(s.root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}

		info, err := entry.Info()
		if err != nil {
			return err
		}
		records, err := s.recordsForFile(path, info)
		if err != nil {
			return err
		}
		visited[path] = struct{}{}
		out = append(out, records...)
		return nil
	})
	if err == nil {
		s.pruneFileCache(visited)
	}

	return out, err
}

func (s *Store) Count(ctx context.Context) (int, error) {
	count := 0
	err := s.walkRecords(ctx, func(Record) bool {
		count++
		return true
	})
	return count, err
}

// VectorStore Interface Implementation

// StoreMemory stores a memory with embedding (satisfies VectorStore interface)
func (s *Store) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	rec := Record{
		SessionID:    sessionID,
		Content:      content,
		Metadata:     metadata,
		Embedding:    embedding,
		LastEmbedded: time.Now().UTC(),
	}
	return s.Save(ctx, rec)
}

// SearchMemory searches by embedding vector (cosine similarity)
// Falls back to keyword search if no embedding provided
func (s *Store) SearchMemory(ctx context.Context, sessionID string, queryEmbedding []float32, limit int) ([]model.MemoryRecord, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	if limit <= 0 {
		limit = 8
	}

	hasQuery := len(queryEmbedding) > 0
	query := model.NewCosineQuery(queryEmbedding)
	scored := make(topVectorRecords, 0, limit)
	ordinal := 0
	err := s.walkRecords(ctx, func(rec Record) bool {
		currentOrdinal := ordinal
		ordinal++
		if sessionID != "" && rec.SessionID != sessionID {
			return true
		}
		score := 0.0
		if hasQuery {
			if len(rec.Embedding) == 0 {
				return true
			}
			score = query.Similarity(rec.Embedding)
			if score <= 0 {
				return true
			}
		}
		candidate := scoredVectorRecord{record: rec, ordinal: currentOrdinal, score: score}
		if len(scored) < limit {
			scored.push(candidate)
			return true
		}
		if vectorRecordBetter(candidate, scored[0]) {
			scored.replaceWorst(candidate)
		}
		return true
	})
	if err != nil {
		return nil, err
	}

	sort.SliceStable(scored, func(i, j int) bool {
		return vectorRecordBetter(scored[i], scored[j])
	})

	result := make([]model.MemoryRecord, len(scored))
	for i, item := range scored {
		result[i] = item.record.toMemoryRecord()
		result[i].Score = item.score
	}
	return result, nil
}

type scoredVectorRecord struct {
	record  Record
	ordinal int
	score   float64
}

type topVectorRecords []scoredVectorRecord

func vectorRecordBetter(a, b scoredVectorRecord) bool {
	if a.score != b.score {
		return a.score > b.score
	}
	return a.ordinal < b.ordinal
}

func vectorRecordWorse(a, b scoredVectorRecord) bool {
	return vectorRecordBetter(b, a)
}

func (h *topVectorRecords) push(item scoredVectorRecord) {
	items := append(*h, item)
	child := len(items) - 1
	for child > 0 {
		parent := (child - 1) / 2
		if !vectorRecordWorse(item, items[parent]) {
			break
		}
		items[child] = items[parent]
		child = parent
	}
	items[child] = item
	*h = items
}

func (h topVectorRecords) replaceWorst(item scoredVectorRecord) {
	parent := 0
	for {
		child := parent*2 + 1
		if child >= len(h) {
			break
		}
		if right := child + 1; right < len(h) && vectorRecordWorse(h[right], h[child]) {
			child = right
		}
		if !vectorRecordWorse(h[child], item) {
			break
		}
		h[parent] = h[child]
		parent = child
	}
	h[parent] = item
}

// UpdateEmbedding updates the embedding vector for a record
func (s *Store) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	return filepath.WalkDir(s.root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}

		info, err := entry.Info()
		if err != nil {
			return err
		}
		records, err := s.recordsForFile(path, info)
		if err != nil {
			return err
		}
		var updated []Record
		changed := false

		for _, rec := range records {
			if rec.NumID == id {
				rec.Embedding = embedding
				rec.LastEmbedded = lastEmbedded
				changed = true
			}
			updated = append(updated, rec)
		}

		if !changed {
			return nil
		}

		var scope, sessionID string
		if len(updated) > 0 {
			scope = updated[0].Scope
			sessionID = updated[0].SessionID
		} else {
			scope = "sessions"
			sessionID = "default"
		}

		var builder strings.Builder
		fmt.Fprintf(&builder, "# %s: %s\n\n", title(scope), sessionID)
		for _, rec := range updated {
			builder.WriteString(renderBlock(rec))
		}

		if err := os.WriteFile(path, []byte(builder.String()), 0o644); err != nil {
			return err
		}
		s.invalidateFile(path)
		return nil
	})
}

// DeleteMemory deletes memories by numeric IDs
func (s *Store) DeleteMemory(ctx context.Context, ids []int64) error {
	if len(ids) == 0 {
		return nil
	}
	idSet := make(map[int64]struct{}, len(ids))
	for _, id := range ids {
		idSet[id] = struct{}{}
	}
	return s.deleteByNumIDs(ctx, idSet)
}

// deleteByNumID is a helper to delete by numeric ID
func (s *Store) deleteByNumID(ctx context.Context, id int64) error {
	return s.deleteByNumIDs(ctx, map[int64]struct{}{id: {}})
}

func (s *Store) deleteByNumIDs(ctx context.Context, ids map[int64]struct{}) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	return filepath.WalkDir(s.root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if entry.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}

		info, err := entry.Info()
		if err != nil {
			return err
		}
		records, err := s.recordsForFile(path, info)
		if err != nil {
			return err
		}
		var kept []Record
		changed := false

		for _, rec := range records {
			if _, remove := ids[rec.NumID]; remove {
				changed = true
				continue
			}
			kept = append(kept, rec)
		}

		if !changed {
			return nil
		}

		var scope, sessionID string
		if len(kept) > 0 {
			scope = kept[0].Scope
			sessionID = kept[0].SessionID
		} else {
			scope = "sessions"
			sessionID = "default"
		}

		var builder strings.Builder
		fmt.Fprintf(&builder, "# %s: %s\n\n", title(scope), sessionID)
		for _, rec := range kept {
			builder.WriteString(renderBlock(rec))
		}

		if err := os.WriteFile(path, []byte(builder.String()), 0o644); err != nil {
			return err
		}
		s.invalidateFile(path)
		return nil
	})
}

func (s *Store) recordsForFile(path string, info os.FileInfo) ([]Record, error) {
	s.cacheMu.Lock()
	defer s.cacheMu.Unlock()

	if cached, ok := s.cache[path]; ok && cached.size == info.Size() && cached.modTime.Equal(info.ModTime()) {
		return cached.records, nil
	}
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	records := s.parseFile(path, b)
	if s.cache == nil {
		s.cache = make(map[string]cachedMarkdownFile)
	}
	s.cache[path] = cachedMarkdownFile{
		size:    info.Size(),
		modTime: info.ModTime(),
		records: records,
	}
	return records, nil
}

func (s *Store) invalidateFile(path string) {
	s.cacheMu.Lock()
	delete(s.cache, path)
	s.cacheMu.Unlock()
}

func (s *Store) pruneFileCache(visited map[string]struct{}) {
	s.cacheMu.Lock()
	for path := range s.cache {
		if _, ok := visited[path]; !ok {
			delete(s.cache, path)
		}
	}
	s.cacheMu.Unlock()
}

var errStopRecordWalk = errors.New("stop markdown record walk")

func (s *Store) walkRecords(ctx context.Context, fn func(Record) bool) error {
	visited := make(map[string]struct{})

	s.mu.RLock()
	defer s.mu.RUnlock()

	err := filepath.WalkDir(s.root, func(path string, entry os.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		if entry.IsDir() || filepath.Ext(path) != ".md" {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			return err
		}
		records, err := s.recordsForFile(path, info)
		if err != nil {
			return err
		}
		visited[path] = struct{}{}
		for _, rec := range records {
			if !fn(rec) {
				return errStopRecordWalk
			}
		}
		return nil
	})
	if errors.Is(err, errStopRecordWalk) {
		return nil
	}
	if err == nil {
		s.pruneFileCache(visited)
	}
	return err
}

// Iterate calls fn for each memory record, stopping if fn returns false
func (s *Store) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	return s.walkRecords(ctx, func(rec Record) bool {
		return fn(rec.toMemoryRecord())
	})
}

func (s *Store) pathFor(scope, sessionID string) (string, error) {
	scope = cleanName(scope)
	sessionID = cleanName(sessionID)

	if scope == "" {
		scope = "sessions"
	}
	if sessionID == "" {
		sessionID = "default"
	}

	switch scope {
	case "sessions", "spaces":
	default:
		return "", fmt.Errorf("invalid memory scope: %s", scope)
	}

	return filepath.Join(s.root, scope, sessionID+".md"), nil
}

func (s *Store) parseFile(path string, b []byte) []Record {
	scope, sessionID := s.identityForPath(path)
	return parseBlocksWithDefaults(string(b), scope, sessionID)
}

func (s *Store) identityForPath(path string) (string, string) {
	rel, err := filepath.Rel(s.root, path)
	if err != nil {
		return "", strings.TrimSuffix(filepath.Base(path), ".md")
	}

	parts := strings.Split(filepath.ToSlash(rel), "/")
	if len(parts) < 2 {
		return "", strings.TrimSuffix(filepath.Base(path), ".md")
	}

	return parts[0], strings.TrimSuffix(parts[len(parts)-1], ".md")
}

func title(scope string) string {
	switch scope {
	case "spaces":
		return "Space"
	default:
		return "Session"
	}
}

func cleanName(v string) string {
	v = strings.TrimSpace(v)
	v = strings.ReplaceAll(v, "\\", "/")
	v = filepath.Base(v)
	v = strings.TrimSuffix(v, ".md")
	v = strings.Map(func(r rune) rune {
		switch {
		case r >= 'a' && r <= 'z':
			return r
		case r >= 'A' && r <= 'Z':
			return r
		case r >= '0' && r <= '9':
			return r
		case r == '-', r == '_', r == '.':
			return r
		default:
			return '-'
		}
	}, v)
	return strings.Trim(v, "-.")
}

func readLines(s string) []string {
	scanner := bufio.NewScanner(strings.NewReader(s))
	var out []string
	for scanner.Scan() {
		out = append(out, scanner.Text())
	}
	return out
}
