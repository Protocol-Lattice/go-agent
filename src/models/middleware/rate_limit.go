package middleware

import (
	"context"
	"errors"
	"fmt"
	"math"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/models"
	"golang.org/x/time/rate"
)

// ErrRateLimitExceeded indicates that a reject-mode limiter had no permit.
var ErrRateLimitExceeded = errors.New("model rate limit exceeded")

// RateLimitMode controls what happens when no request permit is immediately
// available.
type RateLimitMode uint8

const (
	// RateLimitWait waits for the next permit and respects context cancellation.
	RateLimitWait RateLimitMode = iota
	// RateLimitReject fails immediately with ErrRateLimitExceeded.
	RateLimitReject
)

// RateLimitPolicy limits model request starts. One permit is consumed for each
// retry attempt when this policy is placed inside RetryPolicy.
type RateLimitPolicy struct {
	Requests int
	Per      time.Duration
	Burst    int
	Mode     RateLimitMode
}

// Wrap applies the rate-limit policy.
func (p RateLimitPolicy) Wrap(next models.Agent) (models.Agent, error) {
	if next == nil {
		return nil, errNilModel
	}
	burst := p.Burst
	if burst == 0 {
		burst = 1
	}
	requestsPerSecond := float64(p.Requests) / p.Per.Seconds()
	switch {
	case p.Requests <= 0:
		return nil, errors.New("rate limit requests must be greater than zero")
	case p.Per <= 0:
		return nil, errors.New("rate limit period must be greater than zero")
	case burst < 1:
		return nil, errors.New("rate limit burst must be at least 1")
	case p.Mode != RateLimitWait && p.Mode != RateLimitReject:
		return nil, errors.New("unknown rate limit mode")
	case math.IsNaN(requestsPerSecond) || math.IsInf(requestsPerSecond, 0) || requestsPerSecond <= 0:
		return nil, errors.New("rate limit must resolve to a finite positive rate")
	}

	limiter := rate.NewLimiter(rate.Limit(requestsPerSecond), burst)
	return &rateLimitAgent{next: next, limiter: limiter, mode: p.Mode}, nil
}

type rateLimitAgent struct {
	next    models.Agent
	limiter *rate.Limiter
	mode    RateLimitMode
}

func (a *rateLimitAgent) wrappedModel() models.Agent { return a.next }

func (a *rateLimitAgent) Generate(ctx context.Context, prompt string) (any, error) {
	if err := a.acquire(ctx); err != nil {
		return nil, err
	}
	return a.next.Generate(ctx, prompt)
}

func (a *rateLimitAgent) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	if err := a.acquire(ctx); err != nil {
		return nil, err
	}
	return a.next.GenerateWithFiles(ctx, prompt, files)
}

func (a *rateLimitAgent) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	if err := a.acquire(ctx); err != nil {
		return nil, err
	}
	return a.next.GenerateStream(ctx, prompt)
}

func (a *rateLimitAgent) GenerateWithTools(ctx context.Context, prompt string, tools []models.ToolDefinition) (models.ToolCallResponse, error) {
	native, err := nativeModel(a.next)
	if err != nil {
		return models.ToolCallResponse{}, err
	}
	if err := a.acquire(ctx); err != nil {
		return models.ToolCallResponse{}, err
	}
	return native.GenerateWithTools(ctx, prompt, tools)
}

func (a *rateLimitAgent) acquire(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	if a.mode == RateLimitReject {
		if !a.limiter.Allow() {
			return ErrRateLimitExceeded
		}
		return nil
	}
	if err := a.limiter.Wait(ctx); err != nil {
		if ctxErr := ctx.Err(); ctxErr != nil {
			return ctxErr
		}
		if _, hasDeadline := ctx.Deadline(); hasDeadline {
			return fmt.Errorf("wait for model rate limit: %w", context.DeadlineExceeded)
		}
		return fmt.Errorf("wait for model rate limit: %w", err)
	}
	return nil
}
