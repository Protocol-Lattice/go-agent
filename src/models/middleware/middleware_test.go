package middleware

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

type stubAgent struct {
	generate          func(context.Context, string) (any, error)
	generateWithFiles func(context.Context, string, []models.File) (any, error)
	generateStream    func(context.Context, string) (<-chan models.StreamChunk, error)
}

func (s *stubAgent) Generate(ctx context.Context, prompt string) (any, error) {
	if s.generate != nil {
		return s.generate(ctx, prompt)
	}
	return "ok", nil
}

func (s *stubAgent) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	if s.generateWithFiles != nil {
		return s.generateWithFiles(ctx, prompt, files)
	}
	return s.Generate(ctx, prompt)
}

func (s *stubAgent) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	if s.generateStream != nil {
		return s.generateStream(ctx, prompt)
	}
	out := make(chan models.StreamChunk, 1)
	out <- models.StreamChunk{Delta: "ok", FullText: "ok", Done: true}
	close(out)
	return out, nil
}

type nativeStubAgent struct {
	*stubAgent
	generateWithTools func(context.Context, string, []models.ToolDefinition) (models.ToolCallResponse, error)
}

func (s *nativeStubAgent) GenerateWithTools(ctx context.Context, prompt string, tools []models.ToolDefinition) (models.ToolCallResponse, error) {
	if s.generateWithTools != nil {
		return s.generateWithTools(ctx, prompt, tools)
	}
	return models.ToolCallResponse{Content: "native"}, nil
}

type recordingPolicy struct {
	name   string
	events *[]string
}

func (p recordingPolicy) Wrap(next models.Agent) (models.Agent, error) {
	return &recordingAgent{next: next, name: p.name, events: p.events}, nil
}

type recordingAgent struct {
	next   models.Agent
	name   string
	events *[]string
}

func (a *recordingAgent) Generate(ctx context.Context, prompt string) (any, error) {
	*a.events = append(*a.events, a.name+":before")
	value, err := a.next.Generate(ctx, prompt)
	*a.events = append(*a.events, a.name+":after")
	return value, err
}

func (a *recordingAgent) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return a.next.GenerateWithFiles(ctx, prompt, files)
}

func (a *recordingAgent) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	return a.next.GenerateStream(ctx, prompt)
}

func TestWrapUsesDeclaredOrder(t *testing.T) {
	var events []string
	wrapped, err := Wrap(
		&stubAgent{},
		recordingPolicy{name: "outer", events: &events},
		recordingPolicy{name: "inner", events: &events},
	)
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	if _, err := wrapped.Generate(context.Background(), "hello"); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	want := []string{"outer:before", "inner:before", "inner:after", "outer:after"}
	if fmt.Sprint(events) != fmt.Sprint(want) {
		t.Fatalf("events = %v, want %v", events, want)
	}
}

func TestWrapRejectsInvalidInputs(t *testing.T) {
	if _, err := Wrap(nil); err == nil {
		t.Fatal("Wrap(nil) succeeded")
	}
	if _, err := Wrap(&stubAgent{}, nil); err == nil {
		t.Fatal("Wrap() accepted nil middleware")
	}
	if _, err := Wrap(&stubAgent{}, TimeoutPolicy{}); err == nil {
		t.Fatal("Wrap() accepted zero timeout")
	}
	if _, err := Wrap(&stubAgent{}, RateLimitPolicy{}); err == nil {
		t.Fatal("Wrap() accepted empty rate limit")
	}
}

func TestRetryPolicyRetriesUntilSuccess(t *testing.T) {
	var calls atomic.Int32
	base := &stubAgent{generate: func(context.Context, string) (any, error) {
		if calls.Add(1) < 3 {
			return nil, errors.New("temporary")
		}
		return "done", nil
	}}
	wrapped, err := Wrap(base, RetryPolicy{
		MaxAttempts:    3,
		InitialBackoff: time.Nanosecond,
		MaxBackoff:     time.Nanosecond,
		Multiplier:     1,
	})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	got, err := wrapped.Generate(context.Background(), "hello")
	if err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if got != "done" || calls.Load() != 3 {
		t.Fatalf("Generate() = %v after %d calls", got, calls.Load())
	}
}

func TestRetryPolicyDoesNotRetryCancellation(t *testing.T) {
	var calls atomic.Int32
	base := &stubAgent{generate: func(context.Context, string) (any, error) {
		calls.Add(1)
		return nil, context.Canceled
	}}
	wrapped, err := Wrap(base, RetryPolicy{MaxAttempts: 5})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	_, err = wrapped.Generate(context.Background(), "hello")
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Generate() error = %v, want context.Canceled", err)
	}
	if calls.Load() != 1 {
		t.Fatalf("calls = %d, want 1", calls.Load())
	}
}

func TestRetryPolicyRetriesStreamSetupOnly(t *testing.T) {
	var calls atomic.Int32
	base := &stubAgent{generateStream: func(context.Context, string) (<-chan models.StreamChunk, error) {
		if calls.Add(1) == 1 {
			return nil, errors.New("temporary setup failure")
		}
		out := make(chan models.StreamChunk, 1)
		out <- models.StreamChunk{Delta: "ok", FullText: "ok", Done: true}
		close(out)
		return out, nil
	}}
	wrapped, err := Wrap(base, RetryPolicy{
		MaxAttempts:    2,
		InitialBackoff: time.Nanosecond,
		MaxBackoff:     time.Nanosecond,
		Multiplier:     1,
	})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	stream, err := wrapped.GenerateStream(context.Background(), "hello")
	if err != nil {
		t.Fatalf("GenerateStream() error = %v", err)
	}
	for range stream {
	}
	if calls.Load() != 2 {
		t.Fatalf("calls = %d, want 2", calls.Load())
	}
}

func TestTimeoutPolicyEnforcesNonCooperativeCallDeadline(t *testing.T) {
	release := make(chan struct{})
	base := &stubAgent{generate: func(context.Context, string) (any, error) {
		<-release
		return "late", nil
	}}
	wrapped, err := Wrap(base, TimeoutPolicy{Duration: 10 * time.Millisecond})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	started := time.Now()
	_, err = wrapped.Generate(context.Background(), "hello")
	close(release)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Generate() error = %v, want deadline exceeded", err)
	}
	if elapsed := time.Since(started); elapsed > 100*time.Millisecond {
		t.Fatalf("timeout returned after %s", elapsed)
	}
}

func TestTimeoutPolicyCoversStreamLifetime(t *testing.T) {
	base := &stubAgent{generateStream: func(ctx context.Context, _ string) (<-chan models.StreamChunk, error) {
		out := make(chan models.StreamChunk, 1)
		out <- models.StreamChunk{Delta: "partial"}
		go func() {
			defer close(out)
			<-ctx.Done()
		}()
		return out, nil
	}}
	wrapped, err := Wrap(base, TimeoutPolicy{Duration: 10 * time.Millisecond})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	stream, err := wrapped.GenerateStream(context.Background(), "hello")
	if err != nil {
		t.Fatalf("GenerateStream() error = %v", err)
	}
	var chunks []models.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 || chunks[0].Delta != "partial" {
		t.Fatalf("chunks = %#v", chunks)
	}
	last := chunks[len(chunks)-1]
	if !last.Done || !errors.Is(last.Err, context.DeadlineExceeded) || last.FullText != "partial" {
		t.Fatalf("terminal chunk = %#v", last)
	}
}

func TestRateLimitPolicyRejectsWithoutPermit(t *testing.T) {
	wrapped, err := Wrap(&stubAgent{}, RateLimitPolicy{
		Requests: 1,
		Per:      time.Hour,
		Burst:    1,
		Mode:     RateLimitReject,
	})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	if _, err := wrapped.Generate(context.Background(), "first"); err != nil {
		t.Fatalf("first Generate() error = %v", err)
	}
	if _, err := wrapped.Generate(context.Background(), "second"); !errors.Is(err, ErrRateLimitExceeded) {
		t.Fatalf("second Generate() error = %v", err)
	}
}

func TestRateLimitPolicyWaitRespectsDeadline(t *testing.T) {
	wrapped, err := Wrap(&stubAgent{}, RateLimitPolicy{
		Requests: 1,
		Per:      time.Hour,
		Burst:    1,
		Mode:     RateLimitWait,
	})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	if _, err := wrapped.Generate(context.Background(), "first"); err != nil {
		t.Fatalf("first Generate() error = %v", err)
	}
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Millisecond)
	defer cancel()
	if _, err := wrapped.Generate(ctx, "second"); !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("second Generate() error = %v, want deadline exceeded", err)
	}
}

func exactEstimator(text string) int64 { return int64(len(text)) }

func TestTokenBudgetRejectsInputBeforeModelCall(t *testing.T) {
	budget, err := NewTokenBudget(3, exactEstimator)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	var calls atomic.Int32
	base := &stubAgent{generate: func(context.Context, string) (any, error) {
		calls.Add(1)
		return "", nil
	}}
	wrapped, err := Wrap(base, TokenBudgetPolicy{Budget: budget})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	_, err = wrapped.Generate(context.Background(), "four")
	if !errors.Is(err, ErrTokenBudgetExceeded) {
		t.Fatalf("Generate() error = %v", err)
	}
	if calls.Load() != 0 || budget.Used() != 0 {
		t.Fatalf("calls = %d, used = %d", calls.Load(), budget.Used())
	}
}

func TestTokenBudgetAccountsInputOutputAndFiles(t *testing.T) {
	budget, err := NewTokenBudget(10, exactEstimator)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	base := &stubAgent{generateWithFiles: func(context.Context, string, []models.File) (any, error) {
		return "z", nil
	}}
	wrapped, err := Wrap(base, TokenBudgetPolicy{Budget: budget})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	files := []models.File{
		{Name: "a.txt", MIME: "text/plain", Data: []byte("abc")},
		{Name: "image.png", MIME: "image/png", Data: make([]byte, 100)},
	}
	if _, err := wrapped.GenerateWithFiles(context.Background(), "p", files); err != nil {
		t.Fatalf("GenerateWithFiles() error = %v", err)
	}
	if budget.Used() != 5 || budget.Remaining() != 5 {
		t.Fatalf("used = %d, remaining = %d", budget.Used(), budget.Remaining())
	}
}

func TestContextTokenBudgetOverridesFallback(t *testing.T) {
	fallback, err := NewTokenBudget(100, exactEstimator)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	requestBudget, err := NewTokenBudget(3, exactEstimator)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	base := &stubAgent{generate: func(context.Context, string) (any, error) {
		return "zz", nil
	}}
	wrapped, err := Wrap(base, TokenBudgetPolicy{Budget: fallback})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	ctx := ContextWithTokenBudget(context.Background(), requestBudget)
	_, err = wrapped.Generate(ctx, "ab")
	if !errors.Is(err, ErrTokenBudgetExceeded) {
		t.Fatalf("Generate() error = %v", err)
	}
	if fallback.Used() != 0 || requestBudget.Used() != 4 {
		t.Fatalf("fallback used = %d, request used = %d", fallback.Used(), requestBudget.Used())
	}
}

func TestTokenBudgetStopsStreamBeforeExceededChunk(t *testing.T) {
	budget, err := NewTokenBudget(4, exactEstimator)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	base := &stubAgent{generateStream: func(context.Context, string) (<-chan models.StreamChunk, error) {
		out := make(chan models.StreamChunk, 3)
		out <- models.StreamChunk{Delta: "ab"}
		out <- models.StreamChunk{Delta: "cd"}
		out <- models.StreamChunk{Done: true, FullText: "abcd"}
		close(out)
		return out, nil
	}}
	wrapped, err := Wrap(base, TokenBudgetPolicy{Budget: budget})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	stream, err := wrapped.GenerateStream(context.Background(), "p")
	if err != nil {
		t.Fatalf("GenerateStream() error = %v", err)
	}
	var chunks []models.StreamChunk
	for chunk := range stream {
		chunks = append(chunks, chunk)
	}
	if len(chunks) != 2 || chunks[0].Delta != "ab" {
		t.Fatalf("chunks = %#v", chunks)
	}
	last := chunks[1]
	if !last.Done || !errors.Is(last.Err, ErrTokenBudgetExceeded) || last.FullText != "ab" {
		t.Fatalf("terminal chunk = %#v", last)
	}
	if budget.Used() != 5 {
		t.Fatalf("used = %d, want 5", budget.Used())
	}
}

func TestTokenBudgetIsConcurrencySafe(t *testing.T) {
	budget, err := NewTokenBudget(10, exactEstimator)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	base := &stubAgent{generate: func(context.Context, string) (any, error) {
		return "", nil
	}}
	wrapped, err := Wrap(base, TokenBudgetPolicy{Budget: budget})
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}

	var successes atomic.Int32
	var wg sync.WaitGroup
	for range 20 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			if _, err := wrapped.Generate(context.Background(), "x"); err == nil {
				successes.Add(1)
			} else if !errors.Is(err, ErrTokenBudgetExceeded) {
				t.Errorf("Generate() error = %v", err)
			}
		}()
	}
	wg.Wait()
	if successes.Load() != 10 || budget.Used() != 10 {
		t.Fatalf("successes = %d, used = %d", successes.Load(), budget.Used())
	}
}

func TestPoliciesPreserveNativeToolCalling(t *testing.T) {
	budget, err := NewTokenBudget(1_000, nil)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	var calls atomic.Int32
	base := &nativeStubAgent{
		stubAgent: &stubAgent{},
		generateWithTools: func(context.Context, string, []models.ToolDefinition) (models.ToolCallResponse, error) {
			calls.Add(1)
			return models.ToolCallResponse{Content: "native"}, nil
		},
	}
	wrapped, err := Wrap(
		base,
		TimeoutPolicy{Duration: time.Second},
		RetryPolicy{MaxAttempts: 2},
		RateLimitPolicy{Requests: 10, Per: time.Second, Burst: 1, Mode: RateLimitWait},
		TokenBudgetPolicy{Budget: budget},
	)
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	native, ok := wrapped.(models.ToolCallingAgent)
	if !ok {
		t.Fatal("wrapped model lost ToolCallingAgent")
	}
	result, err := native.GenerateWithTools(context.Background(), "call", []models.ToolDefinition{{Name: "echo"}})
	if err != nil {
		t.Fatalf("GenerateWithTools() error = %v", err)
	}
	if result.Content != "native" || calls.Load() != 1 {
		t.Fatalf("result = %#v, calls = %d", result, calls.Load())
	}
}

func TestPoliciesReportUnsupportedNativeToolCallingWithoutCharging(t *testing.T) {
	budget, err := NewTokenBudget(100, exactEstimator)
	if err != nil {
		t.Fatalf("NewTokenBudget() error = %v", err)
	}
	wrapped, err := Wrap(
		&stubAgent{},
		TimeoutPolicy{Duration: time.Second},
		RetryPolicy{MaxAttempts: 2},
		RateLimitPolicy{Requests: 1, Per: time.Hour, Burst: 1, Mode: RateLimitReject},
		TokenBudgetPolicy{Budget: budget},
	)
	if err != nil {
		t.Fatalf("Wrap() error = %v", err)
	}
	native := wrapped.(models.ToolCallingAgent)
	_, err = native.GenerateWithTools(context.Background(), "call", nil)
	if !errors.Is(err, models.ErrToolCallingUnsupported) {
		t.Fatalf("GenerateWithTools() error = %v", err)
	}
	if budget.Used() != 0 {
		t.Fatalf("used = %d, want 0", budget.Used())
	}
	if _, err := wrapped.Generate(context.Background(), "first real call"); err != nil {
		t.Fatalf("Generate() after unsupported native call error = %v", err)
	}
}
