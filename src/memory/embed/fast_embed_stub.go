//go:build !fastembed

package embed

import (
	"context"
	"fmt"
)

type FastEmbedder struct{}

func defaultFastEmbedOptions() *Options { return nil }

func NewFastEmbeed(ctx context.Context, opt *Options) (Embedder, error) {
	return nil, fmt.Errorf("fastembed support not included; rebuild with -tags fastembed")
}

func (FastEmbedder) Close() error { return nil }

func (FastEmbedder) Dim() int { return 0 }

func (FastEmbedder) EmbedPassages(ctx context.Context, docs []string) ([][]float32, error) {
	return nil, fmt.Errorf("fastembed support not included")
}

func (FastEmbedder) Embed(ctx context.Context, q string) ([]float32, error) {
	return nil, fmt.Errorf("fastembed support not included")
}
