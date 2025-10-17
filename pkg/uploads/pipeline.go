package uploads

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand"
	"sync"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

// EmbeddedChunk couples a chunk with its embedding vector.
type EmbeddedChunk struct {
	Chunk    DocumentChunk
	Vector   []float32
	Attempts int
	Err      error
	Duration time.Duration
}

// Pipeline orchestrates chunk processing, content safety middleware, and embedding.
type Pipeline struct {
	Embedder       memory.Embedder
	Middlewares    []Middleware
	BatchSize      int
	WorkerCount    int
	ExpectedDims   int
	Normalize      bool
	SimilarityMode string
	RetryOptions   RetryOptions
	Metrics        Metrics
}

// Middleware can mutate a chunk before it is embedded.
type Middleware interface {
	Process(ctx context.Context, chunk *DocumentChunk) error
}

// MiddlewareFunc converts a function into Middleware.
type MiddlewareFunc func(ctx context.Context, chunk *DocumentChunk) error

func (f MiddlewareFunc) Process(ctx context.Context, chunk *DocumentChunk) error {
	return f(ctx, chunk)
}

// Metrics is a light-weight interface so callers can instrument the pipeline.
type Metrics interface {
	ObserveQueueDepth(depth int)
	RecordEmbedding(duration time.Duration, ok bool)
}

// RetryOptions control the retry behaviour when embedding fails.
type RetryOptions struct {
	MaxAttempts int
	Jitter      time.Duration
	BaseDelay   time.Duration
}

// Process embeds the provided chunks using a bounded worker pool.
func (p Pipeline) Process(ctx context.Context, chunks []DocumentChunk) ([]EmbeddedChunk, error) {
	if p.Embedder == nil {
		return nil, errors.New("uploads pipeline requires an embedder")
	}
	if p.WorkerCount <= 0 {
		p.WorkerCount = 4
	}
	if p.BatchSize <= 0 {
		p.BatchSize = 32
	}
	if p.RetryOptions.MaxAttempts <= 0 {
		p.RetryOptions.MaxAttempts = 3
	}
	if p.RetryOptions.BaseDelay <= 0 {
		p.RetryOptions.BaseDelay = 200 * time.Millisecond
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	input := make(chan DocumentChunk, p.BatchSize)
	output := make(chan EmbeddedChunk, len(chunks))
	var wg sync.WaitGroup

	for i := 0; i < p.WorkerCount; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for chunk := range input {
				result := p.embedWithRetry(ctx, chunk)
				select {
				case <-ctx.Done():
					return
				case output <- result:
				}
			}
		}()
	}

	go func() {
		defer close(input)
		for _, chunk := range chunks {
			if err := p.applyMiddleware(ctx, &chunk); err != nil {
				output <- EmbeddedChunk{Chunk: chunk, Err: err}
				continue
			}
			select {
			case <-ctx.Done():
				return
			case input <- chunk:
				if p.Metrics != nil {
					p.Metrics.ObserveQueueDepth(len(input))
				}
			}
		}
	}()

	go func() {
		wg.Wait()
		close(output)
	}()

	results := make([]EmbeddedChunk, 0, len(chunks))
	for {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case res, ok := <-output:
			if !ok {
				return results, nil
			}
			if res.Err == nil {
				if p.ExpectedDims > 0 && len(res.Vector) != p.ExpectedDims {
					res.Err = fmt.Errorf("embedding dimension mismatch: got %d expected %d", len(res.Vector), p.ExpectedDims)
				}
				if res.Err == nil && p.Normalize && p.SimilarityMode == "cosine" {
					normalize(res.Vector)
				}
			}
			results = append(results, res)
		}
	}
}

func (p Pipeline) applyMiddleware(ctx context.Context, chunk *DocumentChunk) error {
	for _, mw := range p.Middlewares {
		if mw == nil {
			continue
		}
		if err := mw.Process(ctx, chunk); err != nil {
			return err
		}
	}
	return nil
}

func (p Pipeline) embedWithRetry(ctx context.Context, chunk DocumentChunk) EmbeddedChunk {
	start := time.Now()
	attempts := 0
	for {
		if ctx.Err() != nil {
			return EmbeddedChunk{Chunk: chunk, Attempts: attempts, Err: ctx.Err()}
		}
		attempts++
		vec, err := p.Embedder.Embed(ctx, chunk.Content)
		duration := time.Since(start)
		if err == nil && len(vec) > 0 {
			if p.Metrics != nil {
				p.Metrics.RecordEmbedding(duration, true)
			}
			return EmbeddedChunk{Chunk: chunk, Vector: vec, Attempts: attempts, Duration: duration}
		}
		if attempts >= p.RetryOptions.MaxAttempts {
			if p.Metrics != nil {
				p.Metrics.RecordEmbedding(duration, false)
			}
			if err == nil {
				err = fmt.Errorf("empty embedding for chunk %s", chunk.ID)
			}
			return EmbeddedChunk{Chunk: chunk, Attempts: attempts, Err: err, Duration: duration}
		}
		// jittered backoff
		delay := p.RetryOptions.BaseDelay * time.Duration(attempts)
		if p.RetryOptions.Jitter > 0 {
			delay += time.Duration(rand.Int63n(int64(p.RetryOptions.Jitter)))
		}
		select {
		case <-ctx.Done():
			return EmbeddedChunk{Chunk: chunk, Attempts: attempts, Err: ctx.Err(), Duration: duration}
		case <-time.After(delay):
		}
	}
}

func normalize(vec []float32) {
	var sum float64
	for _, v := range vec {
		sum += float64(v * v)
	}
	magnitude := math.Sqrt(sum)
	if magnitude == 0 {
		return
	}
	inv := 1.0 / magnitude
	for i := range vec {
		vec[i] = float32(float64(vec[i]) * inv)
	}
}
