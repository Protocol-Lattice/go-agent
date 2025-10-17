package uploads

import (
	"io/fs"
	"os"
	"path/filepath"
	"strings"
)

// CodeChunker walks a repository honoring a subset of .gitignore patterns and emits per-file chunks.
type CodeChunker struct {
	Root           string
	MaxFileTokens  int
	AdditionalMeta map[string]any
}

func (c CodeChunker) Chunk(_ ReaderWithName, src Source) ([]DocumentChunk, error) {
	if c.Root == "" {
		return nil, ErrUnsupported
	}
	max := c.MaxFileTokens
	if max <= 0 {
		max = 800
	}
	matcher := newIgnoreMatcher(filepath.Join(c.Root, ".gitignore"))

	var chunks []DocumentChunk
	err := filepath.WalkDir(c.Root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		rel, relErr := filepath.Rel(c.Root, path)
		if relErr != nil {
			rel = path
		}
		if rel == "." {
			rel = ""
		}
		if matcher != nil && matcher.Match(rel, d.IsDir()) {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if d.IsDir() {
			if d.Name() == ".git" {
				return filepath.SkipDir
			}
			return nil
		}
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		content := string(data)
		if strings.TrimSpace(content) == "" {
			return nil
		}
		chunks = append(chunks, makeCodeChunks(rel, content, src, max, c.AdditionalMeta)...)
		return nil
	})
	return chunks, err
}

func makeCodeChunks(path, content string, src Source, max int, extra map[string]any) []DocumentChunk {
	lines := strings.Split(content, "\n")
	var (
		chunks     []DocumentChunk
		builder    strings.Builder
		tokenCount int
		idx        int
	)
	emit := func() {
		if builder.Len() == 0 {
			return
		}
		meta := map[string]any{
			"chunk_index": idx,
			"path":        path,
		}
		for k, v := range extra {
			meta[k] = v
		}
		chunk := DocumentChunk{
			ID:       chunkID(path, idx),
			Content:  builder.String(),
			Metadata: meta,
		}
		chunk = chunk.WithProvenance(src.Name, src.URI, 0, src.Version)
		chunks = append(chunks, chunk)
		idx++
		builder.Reset()
		tokenCount = 0
	}

	for _, line := range lines {
		estimated := estimateTokens(line)
		if tokenCount+estimated > max && builder.Len() > 0 {
			emit()
		}
		builder.WriteString(line)
		builder.WriteByte('\n')
		tokenCount += estimated
	}
	emit()
	return chunks
}

type ignoreMatcher struct {
	patterns []string
}

func newIgnoreMatcher(path string) *ignoreMatcher {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	lines := strings.Split(string(data), "\n")
	var patterns []string
	for _, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		patterns = append(patterns, line)
	}
	if len(patterns) == 0 {
		return nil
	}
	return &ignoreMatcher{patterns: patterns}
}

func (m *ignoreMatcher) Match(path string, dir bool) bool {
	if m == nil {
		return false
	}
	normalized := filepath.ToSlash(path)
	for _, pattern := range m.patterns {
		pat := pattern
		negate := false
		if strings.HasPrefix(pat, "!") {
			negate = true
			pat = strings.TrimPrefix(pat, "!")
		}
		if dir && !strings.HasSuffix(pat, "/") {
			pat += "/"
		}
		if strings.HasPrefix(pat, "/") {
			pat = strings.TrimPrefix(pat, "/")
		}
		matched, _ := filepath.Match(pat, normalized)
		if !matched && strings.Contains(pat, "/") {
			// allow contains matches for directory patterns
			matched = strings.HasPrefix(normalized, pat)
		}
		if matched {
			if negate {
				return false
			}
			return true
		}
	}
	return false
}
