package workflow

import (
	"context"
	"fmt"
	"sync/atomic"
)

var invocationCounter atomic.Uint64

// Context is passed to workflow nodes during graph execution.
type Context struct {
	context.Context

	SessionID    string
	InvocationID string
	State        map[string]any
}

func newContext(ctx context.Context, sessionID string) Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return Context{
		Context:      ctx,
		SessionID:    sessionID,
		InvocationID: fmt.Sprintf("workflow-%d", invocationCounter.Add(1)),
		State:        map[string]any{},
	}
}
