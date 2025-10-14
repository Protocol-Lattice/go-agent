package agent

import "context"

// ToolSpec describes how the agent should present a tool to the model.
type ToolSpec struct {
	Name        string           `json:"name"`
	Description string           `json:"description"`
	InputSchema map[string]any   `json:"input_schema"`
	Examples    []map[string]any `json:"examples,omitempty"`
}

// ToolRequest captures an invocation request for a tool.
type ToolRequest struct {
	SessionID string
	Arguments map[string]any
}

// ToolResponse represents the structured response returned by a tool.
type ToolResponse struct {
	Content  string
	Metadata map[string]string
}

// Tool exposes structured metadata and an invocation handler.
type Tool interface {
	Spec() ToolSpec
	Invoke(ctx context.Context, req ToolRequest) (ToolResponse, error)
}

// SubAgent represents a specialist agent that can be delegated work.
type SubAgent interface {
	Name() string
	Description() string
	Run(ctx context.Context, input string) (string, error)
}
