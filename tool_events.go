package agent

import (
	"context"

	utcp "github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/transports"
)

// ToolExecutionEvent describes one real UTCP tool execution. It is emitted
// from the same client used by the agent and CodeMode, so the WebUI can render
// the actual execution order instead of inferring tool calls from model text.
type ToolExecutionEvent struct {
	Type      string         `json:"type"`
	Tool      string         `json:"tool"`
	Arguments map[string]any `json:"arguments,omitempty"`
	Result    any            `json:"result,omitempty"`
	Error     string         `json:"error,omitempty"`
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
	emitToolExecutionEvent(ctx, ToolExecutionEvent{
		Type: "tool_start", Tool: toolName, Arguments: args,
	})

	result, err := c.UtcpClientInterface.CallTool(ctx, toolName, args)
	if err != nil {
		emitToolExecutionEvent(ctx, ToolExecutionEvent{
			Type: "tool_result", Tool: toolName, Arguments: args, Error: err.Error(),
		})
		return nil, err
	}

	emitToolExecutionEvent(ctx, ToolExecutionEvent{
		Type: "tool_result", Tool: toolName, Arguments: args, Result: result,
	})
	return result, nil
}

func (c *ObservedUTCPClient) CallToolStream(ctx context.Context, toolName string, args map[string]any) (transports.StreamResult, error) {
	emitToolExecutionEvent(ctx, ToolExecutionEvent{
		Type: "tool_start", Tool: toolName, Arguments: args,
	})

	stream, err := c.UtcpClientInterface.CallToolStream(ctx, toolName, args)
	if err != nil {
		emitToolExecutionEvent(ctx, ToolExecutionEvent{
			Type: "tool_result", Tool: toolName, Arguments: args, Error: err.Error(),
		})
		return nil, err
	}

	return &observedStreamResult{inner: stream, ctx: ctx, tool: toolName, args: args}, nil
}

type observedStreamResult struct {
	inner transports.StreamResult
	ctx   context.Context
	tool  string
	args  map[string]any
	done  bool
}

func (s *observedStreamResult) Next() (any, error) {
	value, err := s.inner.Next()
	if err != nil && !s.done {
		s.done = true
		event := ToolExecutionEvent{Type: "tool_result", Tool: s.tool, Arguments: s.args}
		if err.Error() != "EOF" {
			event.Error = err.Error()
		}
		emitToolExecutionEvent(s.ctx, event)
	}
	return value, err
}

func (s *observedStreamResult) Close() error {
	if s.inner == nil {
		return nil
	}
	return s.inner.Close()
}
