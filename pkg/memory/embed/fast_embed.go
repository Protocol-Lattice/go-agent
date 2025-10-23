package embed

import (
	"context"
	"fmt"
	"runtime"

	fastembed "github.com/anush008/fastembed-go"
)

type Options struct {
	Model     fastembed.EmbeddingModel // e.g. fastembed.BGESmallENV15 (default)
	CacheDir  string                   // e.g. ".fastembed"
	MaxLength int                      // token limit, 0 = default
	BatchSize int                      // defaults to 256 in library; we’ll cap by CPUs below
}

type FastEmbedder struct {
	m   *fastembed.FlagEmbedding
	dim int
	bs  int
}

func NewFastEmbeed(ctx context.Context, opt *Options) (Embedder, error) {
	var init *fastembed.InitOptions
	if opt != nil {
		init = &fastembed.InitOptions{
			Model:     opt.Model,    // zero value picks default (bge-small-en-v1.5)
			CacheDir:  opt.CacheDir, // optional
			MaxLength: opt.MaxLength,
		}
	}
	m, err := fastembed.NewFlagEmbedding(init)
	if err != nil {
		return nil, err
	}
	// Batch heuristic: keep it modest for desktop CPUs
	bs := 64
	if opt != nil && opt.BatchSize > 0 {
		bs = opt.BatchSize
	}
	if bs > 4*runtime.GOMAXPROCS(0) {
		bs = 4 * runtime.GOMAXPROCS(0)
	}
	return &FastEmbedder{m: m, dim: 768, bs: bs}, nil // bge-small-en-v1.5 = 768 dims
}

func (e *FastEmbedder) Close() error {
	if e.m != nil {
		e.m.Destroy()
	}
	return nil
}

func (e *FastEmbedder) Dim() int { return e.dim }

// EmbedPassages embeds a batch of “documents/passages” (adds prefix if missing).
func (e *FastEmbedder) EmbedPassages(ctx context.Context, docs []string) ([][]float32, error) {
	inputs := make([]string, len(docs))
	for i, d := range docs {
		if len(d) >= 8 && d[:8] == "passage:" {
			inputs[i] = d
		} else {
			inputs[i] = "passage: " + d
		}
	}
	out, err := e.m.PassageEmbed(inputs, e.bs)
	if err != nil {
		return nil, fmt.Errorf("passage embed: %w", err)
	}
	return out, nil
}

// EmbedQuery embeds a single query string.
func (e *FastEmbedder) Embed(ctx context.Context, q string) ([]float32, error) {
	return e.m.QueryEmbed(q)
}
