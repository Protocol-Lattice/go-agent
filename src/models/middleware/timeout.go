package middleware

import (
	"context"
	"errors"
	"strings"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

// TimeoutPolicy bounds a complete model operation. For streams, the deadline
// remains active until the stream finishes rather than only covering setup.
// The wrapper can return on time even if a custom provider ignores context,
// though such a provider may continue its work in the background.
type TimeoutPolicy struct {
	Duration time.Duration
}

// Wrap applies the timeout policy.
func (p TimeoutPolicy) Wrap(next models.Agent) (models.Agent, error) {
	if next == nil {
		return nil, errNilModel
	}
	if p.Duration <= 0 {
		return nil, errors.New("model timeout must be greater than zero")
	}
	return &timeoutAgent{next: next, duration: p.Duration}, nil
}

type timeoutAgent struct {
	next     models.Agent
	duration time.Duration
}

func (a *timeoutAgent) wrappedModel() models.Agent { return a.next }

func (a *timeoutAgent) Generate(ctx context.Context, prompt string) (any, error) {
	return timeoutCall(ctx, a.duration, func(callCtx context.Context) (any, error) {
		return a.next.Generate(callCtx, prompt)
	})
}

func (a *timeoutAgent) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	return timeoutCall(ctx, a.duration, func(callCtx context.Context) (any, error) {
		return a.next.GenerateWithFiles(callCtx, prompt, files)
	})
}

func (a *timeoutAgent) GenerateWithTools(ctx context.Context, prompt string, tools []models.ToolDefinition) (models.ToolCallResponse, error) {
	native, err := nativeModel(a.next)
	if err != nil {
		return models.ToolCallResponse{}, err
	}
	return timeoutCall(ctx, a.duration, func(callCtx context.Context) (models.ToolCallResponse, error) {
		return native.GenerateWithTools(callCtx, prompt, tools)
	})
}

func (a *timeoutAgent) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	if ctx == nil {
		ctx = context.Background()
	}
	callCtx, cancel := context.WithTimeout(ctx, a.duration)

	type streamResult struct {
		stream <-chan models.StreamChunk
		err    error
	}
	resultCh := make(chan streamResult, 1)
	go func() {
		stream, err := a.next.GenerateStream(callCtx, prompt)
		resultCh <- streamResult{stream: stream, err: err}
	}()

	var inner <-chan models.StreamChunk
	select {
	case <-callCtx.Done():
		cancel()
		return nil, callCtx.Err()
	case result := <-resultCh:
		if result.err != nil {
			cancel()
			return nil, result.err
		}
		if result.stream == nil {
			cancel()
			return nil, errors.New("model returned a nil stream")
		}
		inner = result.stream
	}

	out := make(chan models.StreamChunk, 16)
	go func() {
		defer close(out)
		defer cancel()
		var full strings.Builder
		for {
			select {
			case <-callCtx.Done():
				out <- models.StreamChunk{
					Done:     true,
					FullText: full.String(),
					Err:      callCtx.Err(),
				}
				return
			case chunk, ok := <-inner:
				if !ok {
					return
				}
				if chunk.Delta != "" {
					full.WriteString(chunk.Delta)
				}
				if !sendStreamChunk(callCtx, out, chunk) {
					return
				}
				if chunk.Done || chunk.Err != nil {
					return
				}
			}
		}
	}()
	return out, nil
}

type timeoutResult[T any] struct {
	value T
	err   error
}

func timeoutCall[T any](ctx context.Context, duration time.Duration, call func(context.Context) (T, error)) (T, error) {
	var zero T
	if ctx == nil {
		ctx = context.Background()
	}
	callCtx, cancel := context.WithTimeout(ctx, duration)
	defer cancel()

	resultCh := make(chan timeoutResult[T], 1)
	go func() {
		value, err := call(callCtx)
		resultCh <- timeoutResult[T]{value: value, err: err}
	}()

	select {
	case <-callCtx.Done():
		return zero, callCtx.Err()
	case result := <-resultCh:
		return result.value, result.err
	}
}

func sendStreamChunk(ctx context.Context, out chan<- models.StreamChunk, chunk models.StreamChunk) bool {
	select {
	case out <- chunk:
		return true
	case <-ctx.Done():
		return false
	}
}
