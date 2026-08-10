package workspace

import (
	"context"
	"os"
	"path/filepath"
	"sort"
)

// Embedder turns source text into a vector. Implementations may use any local
// or remote embedding provider; the workspace package stays provider-agnostic.
type Embedder interface {
	Embed(ctx context.Context, text string) ([]float32, error)
}

type SemanticResult struct {
	Path  string
	Score float32
}

// BuildEmbeddings computes one vector per indexed file. It is intentionally
// explicit so callers control when embedding work and its cost occur.
func (i *Index) BuildEmbeddings(ctx context.Context) error {
	if i.embedder == nil {
		return nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	vectors := make(map[string][]float32, len(i.files))
	for _, f := range i.Files() {
		if err := ctx.Err(); err != nil {
			return err
		}
		b, err := os.ReadFile(filepath.Join(i.root, filepath.FromSlash(f.Path)))
		if err != nil {
			return err
		}
		v, err := i.embedder.Embed(ctx, string(b))
		if err != nil {
			return err
		}
		if len(v) == 0 {
			continue
		}
		vectors[f.Path] = append([]float32(nil), v...)
	}
	i.vectors = vectors
	return nil
}

func (i *Index) SearchSemantic(ctx context.Context, query string, limit int) ([]SemanticResult, error) {
	if i.embedder == nil || query == "" {
		return nil, nil
	}
	if ctx == nil {
		ctx = context.Background()
	}
	if limit <= 0 {
		limit = 20
	}
	q, err := i.embedder.Embed(ctx, query)
	if err != nil {
		return nil, err
	}
	results := make([]SemanticResult, 0, len(i.vectors))
	for path, vector := range i.vectors {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		if len(vector) != len(q) {
			continue
		}
		results = append(results, SemanticResult{Path: path, Score: cosine(q, vector)})
	}
	sort.SliceStable(results, func(a, b int) bool {
		if results[a].Score != results[b].Score {
			return results[a].Score > results[b].Score
		}
		return results[a].Path < results[b].Path
	})
	if len(results) > limit {
		results = results[:limit]
	}
	return results, nil
}

func cosine(a, b []float32) float32 {
	var dot, aa, bb float32
	for n := range a {
		dot += a[n] * b[n]
		aa += a[n] * a[n]
		bb += b[n] * b[n]
	}
	if aa == 0 || bb == 0 {
		return 0
	}
	return dot / float32Sqrt(aa*bb)
}

// Newton iteration avoids adding a numerical dependency for a tiny operation.
func float32Sqrt(x float32) float32 {
	if x <= 0 { return 0 }
	z := x
	for n := 0; n < 8; n++ { z = (z + x/z) / 2 }
	return z
}
