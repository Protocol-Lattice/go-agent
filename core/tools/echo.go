package tools

import (
	"context"
	"fmt"
	"strings"

	agent "github.com/Protocol-Lattice/agent/core/agentic"
)

// EchoTool repeats the provided input. Useful for testing tool wiring.
type EchoTool struct{}

func (e *EchoTool) Spec() agent.ToolSpec {
	return agent.ToolSpec{
		Name:        "echo",
		Description: "Echoes the provided text back to the caller.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"input": map[string]any{
					"type":        "string",
					"description": "Text to echo back.",
				},
			},
			"required": []any{"input"},
		},
	}
}

func (e *EchoTool) Invoke(_ context.Context, req agent.ToolRequest) (agent.ToolResponse, error) {
	raw := req.Arguments["input"]
	if raw == nil {
		return agent.ToolResponse{Content: ""}, nil
	}
	return agent.ToolResponse{Content: strings.TrimSpace(fmt.Sprint(raw))}, nil
}
