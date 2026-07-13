package middleware

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

const (
	defaultRetryAttempts       = 3
	defaultRetryInitialBackoff = 100 * time.Millisecond
	defaultRetryMaxBackoff     = 2 * time.Second
	defaultRetryMultiplier     = 2.0
	defaultRetryJitter         = 0.2
)

// RetryPolicy retries model calls that fail before producing a result. Stream
// creation can be retried, but errors emitted after a stream starts cannot be
// retried safely because doing so could duplicate chunks.
type RetryPolicy struct {
	MaxAttempts    int
	InitialBackoff time.Duration
	MaxBackoff     time.Duration
	Multiplier     float64
	Jitter         float64
	DisableJitter  bool

	// ShouldRetry overrides the default decision. The default retries all
	// non-cancellation errors except policy rejections and unsupported native
	// tool calling. A deadline created by an inner timeout policy is retryable
	// while expiry of the caller's context is not.
	ShouldRetry func(context.Context, error) bool
}

type retryConfig struct {
	maxAttempts    int
	initialBackoff time.Duration
	maxBackoff     time.Duration
	multiplier     float64
	jitter         float64
	shouldRetry    func(context.Context, error) bool
}

func (p RetryPolicy) config() (retryConfig, error) {
	cfg := retryConfig{
		maxAttempts:    p.MaxAttempts,
		initialBackoff: p.InitialBackoff,
		maxBackoff:     p.MaxBackoff,
		multiplier:     p.Multiplier,
		jitter:         p.Jitter,
		shouldRetry:    p.ShouldRetry,
	}
	if cfg.maxAttempts == 0 {
		cfg.maxAttempts = defaultRetryAttempts
	}
	if cfg.initialBackoff == 0 {
		cfg.initialBackoff = defaultRetryInitialBackoff
	}
	if cfg.maxBackoff == 0 {
		cfg.maxBackoff = defaultRetryMaxBackoff
	}
	if cfg.multiplier == 0 {
		cfg.multiplier = defaultRetryMultiplier
	}
	if p.DisableJitter {
		cfg.jitter = 0
	} else if p.Jitter == 0 {
		cfg.jitter = defaultRetryJitter
	}
	if cfg.shouldRetry == nil {
		cfg.shouldRetry = defaultShouldRetry
	}

	switch {
	case cfg.maxAttempts < 1:
		return retryConfig{}, errors.New("retry max attempts must be at least 1")
	case cfg.initialBackoff < 0:
		return retryConfig{}, errors.New("retry initial backoff cannot be negative")
	case cfg.maxBackoff < cfg.initialBackoff:
		return retryConfig{}, errors.New("retry max backoff cannot be less than initial backoff")
	case cfg.multiplier < 1 || math.IsNaN(cfg.multiplier) || math.IsInf(cfg.multiplier, 0):
		return retryConfig{}, errors.New("retry multiplier must be finite and at least 1")
	case cfg.jitter < 0 || cfg.jitter > 1 || math.IsNaN(cfg.jitter):
		return retryConfig{}, errors.New("retry jitter must be between 0 and 1")
	}
	return cfg, nil
}

// Wrap applies the retry policy.
func (p RetryPolicy) Wrap(next models.Agent) (models.Agent, error) {
	if next == nil {
		return nil, errNilModel
	}
	cfg, err := p.config()
	if err != nil {
		return nil, err
	}
	return &retryAgent{next: next, config: cfg}, nil
}

type retryAgent struct {
	next   models.Agent
	config retryConfig
}

func (a *retryAgent) wrappedModel() models.Agent { return a.next }

func (a *retryAgent) Generate(ctx context.Context, prompt string) (any, error) {
	return retryCall(ctx, a.config, func(callCtx context.Context) (any, error) {
		return a.next.Generate(callCtx, prompt)
	})
}

func (a *retryAgent) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return retryCall(ctx, a.config, func(callCtx context.Context) (any, error) {
		return a.next.GenerateWithFiles(callCtx, prompt, files)
	})
}

func (a *retryAgent) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	return retryCall(ctx, a.config, func(callCtx context.Context) (<-chan models.StreamChunk, error) {
		return a.next.GenerateStream(callCtx, prompt)
	})
}

func (a *retryAgent) GenerateWithTools(ctx context.Context, prompt string, tools []models.ToolDefinition) (models.ToolCallResponse, error) {
	native, err := nativeModel(a.next)
	if err != nil {
		return models.ToolCallResponse{}, err
	}
	return retryCall(ctx, a.config, func(callCtx context.Context) (models.ToolCallResponse, error) {
		return native.GenerateWithTools(callCtx, prompt, tools)
	})
}

func retryCall[T any](ctx context.Context, cfg retryConfig, call func(context.Context) (T, error)) (T, error) {
	var zero T
	if ctx == nil {
		ctx = context.Background()
	}
	for attempt := 1; attempt <= cfg.maxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return zero, err
		}

		value, err := call(ctx)
		if err == nil {
			return value, nil
		}
		if attempt == cfg.maxAttempts || !cfg.shouldRetry(ctx, err) {
			return zero, err
		}
		if err := waitForRetry(ctx, retryBackoff(cfg, attempt)); err != nil {
			return zero, err
		}
	}
	return zero, fmt.Errorf("retry attempts exhausted")
}

func defaultShouldRetry(ctx context.Context, err error) bool {
	if err == nil {
		return false
	}
	if errors.Is(err, context.Canceled) {
		return false
	}
	if errors.Is(err, context.DeadlineExceeded) && ctx.Err() != nil {
		return false
	}
	return !errors.Is(err, ErrRateLimitExceeded) &&
		!errors.Is(err, ErrTokenBudgetExceeded) &&
		!errors.Is(err, models.ErrToolCallingUnsupported)
}

func retryBackoff(cfg retryConfig, failedAttempt int) time.Duration {
	delay := float64(cfg.initialBackoff) * math.Pow(cfg.multiplier, float64(failedAttempt-1))
	if delay > float64(cfg.maxBackoff) {
		delay = float64(cfg.maxBackoff)
	}
	if cfg.jitter > 0 && delay > 0 {
		factor := 1 + ((rand.Float64()*2 - 1) * cfg.jitter)
		delay *= factor
	}
	if delay < 0 {
		return 0
	}
	if delay > float64(cfg.maxBackoff) {
		delay = float64(cfg.maxBackoff)
	}
	return time.Duration(delay)
}

func waitForRetry(ctx context.Context, delay time.Duration) error {
	if delay <= 0 {
		return nil
	}
	timer := time.NewTimer(delay)
	defer timer.Stop()
	select {
	case <-ctx.Done():
		return ctx.Err()
	case <-timer.C:
		return nil
	}
}
