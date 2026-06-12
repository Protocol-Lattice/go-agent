package markdown

import (
	"bufio"
	"context"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

type Store struct {
	root string
	mu   sync.RWMutex
}

func NewStore(root string) (*Store, error) {
	if root == "" {
		root = ".agent-memory"
	}

	abs, err := filepath.Abs(root)
	if err != nil {
		return nil, err
	}

	s := &Store{root: abs}

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

	b, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}

	return s.parseFile(path, b), nil
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

	scored := scoreRecords(all, query)
	if len(scored) > limit {
		scored = scored[:limit]
	}

	out := make([]Record, 0, len(scored))
	for _, item := range scored {
		out = append(out, item.Record)
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

		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		records := s.parseFile(path, b)
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

		return os.WriteFile(path, []byte(builder.String()), 0o644)
	})
}

func (s *Store) all(ctx context.Context) ([]Record, error) {
	var out []Record

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

		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		out = append(out, s.parseFile(path, b)...)
		return nil
	})

	return out, err
}

func (s *Store) Count(ctx context.Context) (int, error) {
	records, err := s.all(ctx)
	return len(records), err
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

	all, err := s.all(ctx)
	if err != nil {
		return nil, err
	}

	// Filter by sessionID
	var candidates []Record
	for _, rec := range all {
		if sessionID == "" || rec.SessionID == sessionID {
			candidates = append(candidates, rec)
		}
	}

	// Score by embedding similarity
	type scoredRec struct {
		Record Record
		Score  float32
	}

	var scored []scoredRec

	for _, rec := range candidates {
		score := float32(0)
		if len(queryEmbedding) > 0 && len(rec.Embedding) > 0 {
			score = cosineSimilarity(queryEmbedding, rec.Embedding)
		}
		if score > 0 || len(queryEmbedding) == 0 {
			scored = append(scored, scoredRec{rec, score})
		}
	}

	// Sort by score descending
	for i := 0; i < len(scored); i++ {
		for j := i + 1; j < len(scored); j++ {
			if scored[j].Score > scored[i].Score {
				scored[i], scored[j] = scored[j], scored[i]
			}
		}
	}

	if len(scored) > limit {
		scored = scored[:limit]
	}

	result := make([]model.MemoryRecord, len(scored))
	for i, s := range scored {
		result[i] = s.Record.toMemoryRecord()
	}
	return result, nil
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

		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		records := s.parseFile(path, b)
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

		return os.WriteFile(path, []byte(builder.String()), 0o644)
	})
}

// DeleteMemory deletes memories by numeric IDs
func (s *Store) DeleteMemory(ctx context.Context, ids []int64) error {
	for _, id := range ids {
		// Convert int64 back to string ID (would need reverse mapping in practice)
		// For now, iterate and find matching NumID
		if err := s.deleteByNumID(ctx, id); err != nil {
			return err
		}
	}
	return nil
}

// deleteByNumID is a helper to delete by numeric ID
func (s *Store) deleteByNumID(ctx context.Context, id int64) error {
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

		b, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		records := s.parseFile(path, b)
		var kept []Record
		changed := false

		for _, rec := range records {
			if rec.NumID == id {
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

		return os.WriteFile(path, []byte(builder.String()), 0o644)
	})
}

// Iterate calls fn for each memory record, stopping if fn returns false
func (s *Store) Iterate(ctx context.Context, fn func(model.MemoryRecord) bool) error {
	if err := ctx.Err(); err != nil {
		return err
	}

	all, err := s.all(ctx)
	if err != nil {
		return err
	}

	for _, rec := range all {
		if !fn(rec.toMemoryRecord()) {
			break
		}
	}

	return nil
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

// cosineSimilarity computes cosine similarity between two embedding vectors
func cosineSimilarity(a, b []float32) float32 {
	if len(a) == 0 || len(b) == 0 {
		return 0
	}

	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}

	var dotProduct, normA, normB float32
	for i := 0; i < minLen; i++ {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	// Using simple approximation for sqrt (could use math.Sqrt for precision)
	return dotProduct / (sqrt(normA) * sqrt(normB))
}

// sqrt computes integer square root approximation
func sqrt(x float32) float32 {
	if x < 0 {
		return 0
	}
	if x == 0 {
		return 0
	}

	// Newton's method approximation
	result := x
	for i := 0; i < 10; i++ {
		result = (result + x/result) / 2
	}
	return result
}
