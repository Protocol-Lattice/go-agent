package models

import (
	"context"
	"errors"
)

// ErrToolCallingUnsupported lets wrappers preserve the optional capability
// without forcing callers to abandon the prompt-based fallback.
var ErrToolCallingUnsupported = errors.New("native tool calling unsupported")

// File is a lightweight in-memory attachment.
// Name is used for display; MIME should be best-effort (e.g., "text/markdown").
type File struct {
	Name string
	MIME string
	Data []byte
}

// StreamChunk represents a single piece of a streaming LLM response.
// When Done is true, the stream is complete and FullText holds the aggregated output.
// When Err is non-nil, the stream encountered an error.
type StreamChunk struct {
	Delta    string // incremental text token
	Done     bool   // true on the final chunk
	FullText string // aggregated text (populated only on the final chunk)
	Err      error  // non-nil if the stream encountered a fatal error
}

type Agent interface {
	Generate(context.Context, string) (any, error)
	GenerateWithFiles(context.Context, string, []File) (any, error)

	// GenerateStream returns a channel that yields incremental text chunks.
	// The final chunk has Done=true and FullText set to the complete response.
	// If the provider doesn't support streaming natively, it falls back to
	// a single-chunk response wrapping Generate.
	GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error)
}

// ToolDefinition is the provider-neutral description of a callable tool.
// Providers adapt this shape to their native tool/function-calling API.
type ToolDefinition struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	InputSchema map[string]any `json:"input_schema"`
}

// ToolCall is a provider-neutral tool invocation selected by a model.
type ToolCall struct {
	ID        string         `json:"id,omitempty"`
	Name      string         `json:"name"`
	Arguments map[string]any `json:"arguments"`
}

// ToolCallResponse contains either assistant text, one or more native tool
// calls, or both depending on the provider response.
type ToolCallResponse struct {
	Content   string     `json:"content,omitempty"`
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
}

// ToolCallingAgent is an optional capability. It intentionally sits beside
// Agent so existing custom model implementations remain source-compatible.
// Agents that do not implement it continue through the prompt-based fallback.
type ToolCallingAgent interface {
	GenerateWithTools(ctx context.Context, prompt string, tools []ToolDefinition) (ToolCallResponse, error)
}
