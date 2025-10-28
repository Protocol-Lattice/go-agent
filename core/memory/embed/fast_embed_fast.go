//go:build fastembed

package embed

import (
	"context"
	"fmt"
	"runtime"

	fastembed "github.com/anush008/fastembed-go"
)

type FastEmbedder struct {
	m   *fastembed.FlagEmbedding
	dim int
	bs  int
}

func defaultFastEmbedOptions() *Options {
	return &Options{
		Model:     string(fastembed.BGESmallENV15),
		CacheDir:  ".fastembed",
		BatchSize: 64,
	}
}

func NewFastEmbeed(ctx context.Context, opt *Options) (Embedder, error) {
	var init *fastembed.InitOptions
	if opt != nil {
		init = &fastembed.InitOptions{
			Model:     fastembed.EmbeddingModel(opt.Model),
			CacheDir:  opt.CacheDir,
			MaxLength: opt.MaxLength,
		}
	}
	m, err := fastembed.NewFlagEmbedding(init)
	if err != nil {
		return nil, err
	}
	bs := 64
	if opt != nil && opt.BatchSize > 0 {
		bs = opt.BatchSize
	}
	if bs > 4*runtime.GOMAXPROCS(0) {
		bs = 4 * runtime.GOMAXPROCS(0)
	}
	return &FastEmbedder{m: m, dim: 768, bs: bs}, nil
}

func (e *FastEmbedder) Close() error {
	if e.m != nil {
		e.m.Destroy()
	}
	return nil
}

func (e *FastEmbedder) Dim() int { return e.dim }

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

func (e *FastEmbedder) Embed(ctx context.Context, q string) ([]float32, error) {
	return e.m.QueryEmbed(q)
}
