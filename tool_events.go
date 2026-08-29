package agent

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	utcp "github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/transports"
)

// ToolExecutionEvent describes one real UTCP tool execution. It is emitted
// from the same client used by the agent and CodeMode, so the WebUI can render
// the actual execution order instead of inferring tool calls from model text.
type ToolExecutionEvent struct {
	Type       string         `json:"type"`
	Tool       string         `json:"tool"`
	Arguments  map[string]any `json:"arguments,omitempty"`
	Result     any            `json:"result,omitempty"`
	Error      string         `json:"error,omitempty"`
	DurationMS int64          `json:"duration_ms,omitempty"`
}

type toolExecutionObserverKey struct{}

// WithToolExecutionObserver attaches a per-request observer to the context.
// Observers are request-scoped and therefore safe when the same Agent serves
// multiple concurrent WebUI sessions.
func WithToolExecutionObserver(ctx context.Context, observer func(ToolExecutionEvent)) context.Context {
	if observer == nil {
		return ctx
	}
	return context.WithValue(ctx, toolExecutionObserverKey{}, observer)
}

func toolExecutionObserver(ctx context.Context) func(ToolExecutionEvent) {
	if ctx == nil {
		return nil
	}
	observer, _ := ctx.Value(toolExecutionObserverKey{}).(func(ToolExecutionEvent))
	return observer
}

func emitToolExecutionEvent(ctx context.Context, event ToolExecutionEvent) {
	if observer := toolExecutionObserver(ctx); observer != nil {
		observer(event)
	}
}

// ObservedUTCPClient wraps a UTCP client without changing its behavior. Tool
// calls made directly by the Agent and calls made from CodeMode both pass
// through this wrapper, which gives us one canonical execution event stream.
type ObservedUTCPClient struct {
	utcp.UtcpClientInterface
}

func NewObservedUTCPClient(client utcp.UtcpClientInterface) utcp.UtcpClientInterface {
	if client == nil {
		return nil
	}
	return &ObservedUTCPClient{UtcpClientInterface: client}
}

func (c *ObservedUTCPClient) CallTool(ctx context.Context, toolName string, args map[string]any) (any, error) {
	startedAt := time.Now()
	orchestratorLogf("utcp call started tool=%q stream=false argument_fields=%d", toolName, len(args))
	emitToolExecutionEvent(ctx, ToolExecutionEvent{
		Type: "tool_start", Tool: toolName, Arguments: args,
	})

	result, err := c.UtcpClientInterface.CallTool(ctx, toolName, args)
	duration := time.Since(startedAt)
	if err != nil {
		orchestratorLogf("utcp call failed tool=%q stream=false duration=%s err=%v", toolName, duration, err)
		emitToolExecutionEvent(ctx, ToolExecutionEvent{
			Type: "tool_result", Tool: toolName, Arguments: args, Error: err.Error(), DurationMS: duration.Milliseconds(),
		})
		return nil, err
	}

	orchestratorLogf("utcp call completed tool=%q stream=false duration=%s result_type=%T", toolName, duration, result)
	emitToolExecutionEvent(ctx, ToolExecutionEvent{
		Type: "tool_result", Tool: toolName, Arguments: args, Result: result, DurationMS: duration.Milliseconds(),
	})
	return result, nil
}

func (c *ObservedUTCPClient) CallToolStream(ctx context.Context, toolName string, args map[string]any) (transports.StreamResult, error) {
	startedAt := time.Now()
	orchestratorLogf("utcp call started tool=%q stream=true argument_fields=%d", toolName, len(args))
	emitToolExecutionEvent(ctx, ToolExecutionEvent{
		Type: "tool_start", Tool: toolName, Arguments: args,
	})

	stream, err := c.UtcpClientInterface.CallToolStream(ctx, toolName, args)
	if err != nil {
		duration := time.Since(startedAt)
		orchestratorLogf("utcp call failed tool=%q stream=true phase=open duration=%s err=%v", toolName, duration, err)
		emitToolExecutionEvent(ctx, ToolExecutionEvent{
			Type: "tool_result", Tool: toolName, Arguments: args, Error: err.Error(), DurationMS: duration.Milliseconds(),
		})
		return nil, err
	}
	if stream == nil {
		duration := time.Since(startedAt)
		err = fmt.Errorf("CallToolStream returned nil stream for %s", toolName)
		orchestratorLogf("utcp call failed tool=%q stream=true phase=open duration=%s err=%v", toolName, duration, err)
		emitToolExecutionEvent(ctx, ToolExecutionEvent{
			Type: "tool_result", Tool: toolName, Arguments: args, Error: err.Error(), DurationMS: duration.Milliseconds(),
		})
		return nil, err
	}

	orchestratorLogf("utcp stream opened tool=%q duration=%s", toolName, time.Since(startedAt))
	return &observedStreamResult{inner: stream, ctx: ctx, tool: toolName, args: args, startedAt: startedAt}, nil
}

type observedStreamResult struct {
	inner     transports.StreamResult
	ctx       context.Context
	tool      string
	args      map[string]any
	startedAt time.Time
	chunks    int
	done      bool
}

func (s *observedStreamResult) Next() (any, error) {
	value, err := s.inner.Next()
	if err == nil {
		s.chunks++
	} else {
		s.finish(err)
	}
	return value, err
}

func (s *observedStreamResult) Close() error {
	if s.inner == nil {
		return nil
	}
	err := s.inner.Close()
	s.finish(err)
	return err
}

func (s *observedStreamResult) finish(err error) {
	if s.done {
		return
	}
	s.done = true
	duration := time.Since(s.startedAt)
	event := ToolExecutionEvent{
		Type:       "tool_result",
		Tool:       s.tool,
		Arguments:  s.args,
		DurationMS: duration.Milliseconds(),
	}
	if err != nil && !errors.Is(err, io.EOF) {
		event.Error = err.Error()
		orchestratorLogf("utcp call failed tool=%q stream=true phase=read chunks=%d duration=%s err=%v", s.tool, s.chunks, duration, err)
	} else {
		orchestratorLogf("utcp call completed tool=%q stream=true chunks=%d duration=%s", s.tool, s.chunks, duration)
	}
	emitToolExecutionEvent(s.ctx, event)
}
