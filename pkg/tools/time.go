package tools

import (
	"context"
	"time"

	"github.com/Raezil/lattice-agent/pkg/agent"
)

// TimeTool reports the current UTC time in RFC3339 format.
type TimeTool struct{}

func (t *TimeTool) Spec() agent.ToolSpec {
	return agent.ToolSpec{
		Name:        "time",
		Description: "Returns the current UTC time.",
		InputSchema: map[string]any{
			"type":       "object",
			"properties": map[string]any{},
		},
	}
}

func (t *TimeTool) Invoke(_ context.Context, _ agent.ToolRequest) (agent.ToolResponse, error) {
	return agent.ToolResponse{Content: time.Now().UTC().Format(time.RFC3339)}, nil
}
