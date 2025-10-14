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

// ToolCatalog maintains an ordered set of tools and provides lookup by name.
type ToolCatalog interface {
	Register(tool Tool) error
	Lookup(name string) (Tool, ToolSpec, bool)
	Specs() []ToolSpec
	Tools() []Tool
}

// SubAgent represents a specialist agent that can be delegated work.
type SubAgent interface {
	Name() string
	Description() string
	Run(ctx context.Context, input string) (string, error)
}

// SubAgentDirectory stores sub-agents by name while preserving insertion order.
type SubAgentDirectory interface {
	Register(subAgent SubAgent) error
	Lookup(name string) (SubAgent, bool)
	All() []SubAgent
}
