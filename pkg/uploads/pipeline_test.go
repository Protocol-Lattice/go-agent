package uploads

import (
	"context"
	"errors"
	"testing"
	"time"
)

type fakeEmbedder struct {
	err     error
	vector  []float32
	latency time.Duration
	calls   int
}

func (f *fakeEmbedder) Embed(_ context.Context, _ string) ([]float32, error) {
	f.calls++
	if f.latency > 0 {
		time.Sleep(f.latency)
	}
	if f.err != nil {
		return nil, f.err
	}
	return f.vector, nil
}

type fakeMetrics struct {
	queue []int
	ok    int
	fail  int
}

func (m *fakeMetrics) ObserveQueueDepth(depth int) {
	m.queue = append(m.queue, depth)
}

func (m *fakeMetrics) RecordEmbedding(_ time.Duration, ok bool) {
	if ok {
		m.ok++
	} else {
		m.fail++
	}
}

func TestPipelineProcess(t *testing.T) {
	embedder := &fakeEmbedder{vector: []float32{1, 2, 3}}
	pipeline := Pipeline{
		Embedder:       embedder,
		WorkerCount:    2,
		ExpectedDims:   3,
		Normalize:      true,
		SimilarityMode: "cosine",
		Metrics:        &fakeMetrics{},
	}

	chunks := []DocumentChunk{{ID: "1", Content: "hello"}, {ID: "2", Content: "world"}}
	results, err := pipeline.Process(context.Background(), chunks)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != len(chunks) {
		t.Fatalf("expected %d results got %d", len(chunks), len(results))
	}
	for _, res := range results {
		if res.Err != nil {
			t.Fatalf("unexpected chunk error: %v", res.Err)
		}
		if len(res.Vector) != 3 {
			t.Fatalf("unexpected vector dims: %d", len(res.Vector))
		}
	}
}

func TestPipelineRetry(t *testing.T) {
	embedder := &fakeEmbedder{err: errors.New("boom"), vector: []float32{1}}
	pipeline := Pipeline{Embedder: embedder, RetryOptions: RetryOptions{MaxAttempts: 2, BaseDelay: time.Millisecond}}

	chunks := []DocumentChunk{{ID: "1", Content: "retry"}}
	results, err := pipeline.Process(context.Background(), chunks)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(results) != 1 {
		t.Fatalf("expected 1 result got %d", len(results))
	}
	if results[0].Err == nil {
		t.Fatalf("expected error result")
	}
	if embedder.calls != 2 {
		t.Fatalf("expected 2 calls got %d", embedder.calls)
	}
}
