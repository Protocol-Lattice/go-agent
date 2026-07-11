package agent

import (
	"context"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

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

// AgentState represents the serializable state of an agent for checkpointing.
type AgentState struct {
	SystemPrompt string                          `json:"system_prompt"`
	ShortTerm    map[string][]model.MemoryRecord `json:"short_term"`
	JoinedSpaces []string                        `json:"joined_spaces,omitempty"`
	Timestamp    time.Time                       `json:"timestamp"`
}

// SafetyPolicy defines an interface for validating LLM responses.
type SafetyPolicy interface {
	Validate(ctx context.Context, response string) error
}

// FormatEnforcer defines an interface for validating or repairing the format of LLM responses.
type FormatEnforcer interface {
	Enforce(ctx context.Context, response string) (string, error)
}

// OutputGuardrails holds the policy engines and formatting rules.
type OutputGuardrails struct {
	SafetyPolicies  []SafetyPolicy
	FormatEnforcers []FormatEnforcer
}

// ValidateAndRepair applies safety checks and format enforcing to the response.
func (g *OutputGuardrails) ValidateAndRepair(ctx context.Context, response string) (string, error) {
	if g == nil {
		return response, nil
	}
	for _, policy := range g.SafetyPolicies {
		if err := policy.Validate(ctx, response); err != nil {
			return "", err
		}
	}

	final := response
	var err error
	for _, enforcer := range g.FormatEnforcers {
		final, err = enforcer.Enforce(ctx, final)
		if err != nil {
			return "", err
		}
	}
	return final, nil
}

// InputSafetyPolicy defines an interface for validating LLM inputs.
type InputSafetyPolicy interface {
	Validate(ctx context.Context, input string) error
}

// InputTransformer defines an interface for modifying or sanitizing LLM inputs.
type InputTransformer interface {
	Transform(ctx context.Context, input string) (string, error)
}

// InputGuardrails holds the input safety policies and transformers.
type InputGuardrails struct {
	SafetyPolicies []InputSafetyPolicy
	Transformers   []InputTransformer
}

// ValidateAndTransform applies safety checks and transformers to the input.
func (g *InputGuardrails) ValidateAndTransform(ctx context.Context, input string) (string, error) {
	if g == nil {
		return input, nil
	}
	for _, policy := range g.SafetyPolicies {
		if err := policy.Validate(ctx, input); err != nil {
			return "", err
		}
	}

	final := input
	var err error
	for _, transformer := range g.Transformers {
		final, err = transformer.Transform(ctx, final)
		if err != nil {
			return "", err
		}
	}
	return final, nil
}
