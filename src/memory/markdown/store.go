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

	return parseBlocks(string(b)), nil
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

		records := parseBlocks(string(b))
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
		builder.WriteString("# Memory\n\n")
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

		out = append(out, parseBlocks(string(b))...)
		return nil
	})

	return out, err
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
