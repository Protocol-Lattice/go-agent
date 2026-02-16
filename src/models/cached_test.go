package models

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

type MockAgent struct {
	CallCount int32
}

func (m *MockAgent) Generate(ctx context.Context, prompt string) (any, error) {
	atomic.AddInt32(&m.CallCount, 1)
	return "mock response", nil
}

func (m *MockAgent) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	atomic.AddInt32(&m.CallCount, 1)
	return "mock response with files", nil
}

func (m *MockAgent) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	atomic.AddInt32(&m.CallCount, 1)
	ch := make(chan StreamChunk, 1)
	ch <- StreamChunk{Delta: "mock stream response", FullText: "mock stream response", Done: true}
	close(ch)
	return ch, nil
}

func TestCachedLLM_Generate(t *testing.T) {
	mock := &MockAgent{}
	cached := NewCachedLLM(mock, 10, time.Minute, "")

	ctx := context.Background()
	prompt := "hello"

	// First call - should hit the agent
	_, err := cached.Generate(ctx, prompt)
	if err != nil {
		t.Fatalf("first call failed: %v", err)
	}
	if count := atomic.LoadInt32(&mock.CallCount); count != 1 {
		t.Errorf("expected 1 call, got %d", count)
	}

	// Second call - should hit the cache
	_, err = cached.Generate(ctx, prompt)
	if err != nil {
		t.Fatalf("second call failed: %v", err)
	}
	if count := atomic.LoadInt32(&mock.CallCount); count != 1 {
		t.Errorf("expected 1 call (cached), got %d", count)
	}

	// Different prompt - should hit the agent
	_, err = cached.Generate(ctx, "world")
	if err != nil {
		t.Fatalf("third call failed: %v", err)
	}
	if count := atomic.LoadInt32(&mock.CallCount); count != 2 {
		t.Errorf("expected 2 calls, got %d", count)
	}
}

func TestCachedLLM_GenerateWithFiles(t *testing.T) {
	mock := &MockAgent{}
	cached := NewCachedLLM(mock, 10, time.Minute, "")

	ctx := context.Background()
	prompt := "analyze"
	files := []File{{Name: "a.txt", Data: []byte("content")}}

	// First call
	_, err := cached.GenerateWithFiles(ctx, prompt, files)
	if err != nil {
		t.Fatalf("first call failed: %v", err)
	}
	if count := atomic.LoadInt32(&mock.CallCount); count != 1 {
		t.Errorf("expected 1 call, got %d", count)
	}

	// Second call - same files
	_, err = cached.GenerateWithFiles(ctx, prompt, files)
	if err != nil {
		t.Fatalf("second call failed: %v", err)
	}
	if count := atomic.LoadInt32(&mock.CallCount); count != 1 {
		t.Errorf("expected 1 call, got %d", count)
	}

	// Different file content
	files2 := []File{{Name: "a.txt", Data: []byte("different")}}
	_, err = cached.GenerateWithFiles(ctx, prompt, files2)
	if err != nil {
		t.Fatalf("third call failed: %v", err)
	}
	if count := atomic.LoadInt32(&mock.CallCount); count != 2 {
		t.Errorf("expected 2 calls, got %d", count)
	}
}
