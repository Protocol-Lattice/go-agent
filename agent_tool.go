package agent

import (
	"context"
	"fmt"
)

// SubAgentTool adapts a SubAgent to the Tool interface.
type SubAgentTool struct {
	subAgent SubAgent
}

// NewSubAgentTool creates a new tool that wraps a SubAgent.
func NewSubAgentTool(sa SubAgent) Tool {
	return &SubAgentTool{subAgent: sa}
}

func (t *SubAgentTool) Spec() ToolSpec {
	return ToolSpec{
		Name:        t.subAgent.Name(),
		Description: t.subAgent.Description(),
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"instruction": map[string]any{
					"type":        "string",
					"description": "The instruction or query for the sub-agent.",
				},
			},
			"required": []string{"instruction"},
		},
	}
}

func (t *SubAgentTool) Invoke(ctx context.Context, req ToolRequest) (ToolResponse, error) {
	instruction, ok := req.Arguments["instruction"].(string)
	if !ok {
		return ToolResponse{}, fmt.Errorf("missing or invalid 'instruction' argument")
	}

	result, err := t.subAgent.Run(ctx, instruction)
	if err != nil {
		return ToolResponse{}, err
	}

	return ToolResponse{
		Content: result,
	}, nil
}

// AgentToolAdapter adapts an Agent to the Tool interface.
type AgentToolAdapter struct {
	agent       *Agent
	name        string
	description string
}

// NewAgentTool creates a new tool that wraps an Agent.
func NewAgentTool(name, description string, agent *Agent) Tool {
	return &AgentToolAdapter{
		agent:       agent,
		name:        name,
		description: description,
	}
}

func (t *AgentToolAdapter) Spec() ToolSpec {
	return ToolSpec{
		Name:        t.name,
		Description: t.description,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"instruction": map[string]any{
					"type":        "string",
					"description": "The instruction or query for the sub-agent.",
				},
			},
			"required": []string{"instruction"},
		},
	}
}

func (t *AgentToolAdapter) Invoke(ctx context.Context, req ToolRequest) (ToolResponse, error) {
	instruction, ok := req.Arguments["instruction"].(string)
	if !ok {
		return ToolResponse{}, fmt.Errorf("missing or invalid 'instruction' argument")
	}

	// Create a sub-session ID to keep context separate but related
	subSessionID := fmt.Sprintf("%s.sub.%s", req.SessionID, t.name)

	result, err := t.agent.Generate(ctx, subSessionID, instruction)
	if err != nil {
		return ToolResponse{}, err
	}

	return ToolResponse{
		Content: fmt.Sprint(result),
	}, nil
}

// AsTool returns a Tool representation of the Agent.
func (a *Agent) AsTool(name, description string) Tool {
	return NewAgentTool(name, description, a)
}
