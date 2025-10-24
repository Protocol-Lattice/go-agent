package tools

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"

	"github.com/Raezil/lattice-agent/pkg/agent"
)

// CalculatorTool evaluates basic arithmetic expressions in the form "a op b".
type CalculatorTool struct{}

func (c *CalculatorTool) Spec() agent.ToolSpec {
	return agent.ToolSpec{
		Name:        "calculator",
		Description: "Evaluates simple math expressions such as '2 + 2' or '5 * 3'.",
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"expression": map[string]any{
					"type":        "string",
					"description": "Expression in the form '<number> <operator> <number>'.",
				},
			},
			"required": []any{"expression"},
		},
	}
}

func (c *CalculatorTool) Invoke(_ context.Context, req agent.ToolRequest) (agent.ToolResponse, error) {
	exprRaw, ok := req.Arguments["expression"]
	if !ok {
		return agent.ToolResponse{}, fmt.Errorf("missing 'expression' argument")
	}
	expression := strings.TrimSpace(fmt.Sprint(exprRaw))
	fields := strings.Fields(expression)
	if len(fields) != 3 {
		return agent.ToolResponse{}, fmt.Errorf("expected format '<number> <op> <number>'")
	}

	left, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return agent.ToolResponse{}, fmt.Errorf("invalid left operand: %w", err)
	}
	right, err := strconv.ParseFloat(fields[2], 64)
	if err != nil {
		return agent.ToolResponse{}, fmt.Errorf("invalid right operand: %w", err)
	}

	var result float64
	switch fields[1] {
	case "+":
		result = left + right
	case "-":
		result = left - right
	case "*", "x", "X":
		result = left * right
	case "/":
		if math.Abs(right) < 1e-12 {
			return agent.ToolResponse{}, fmt.Errorf("division by zero")
		}
		result = left / right
	default:
		return agent.ToolResponse{}, fmt.Errorf("unsupported operator %q", fields[1])
	}

	return agent.ToolResponse{Content: strconv.FormatFloat(result, 'f', -1, 64)}, nil
}
