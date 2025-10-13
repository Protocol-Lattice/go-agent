package tools

import (
	"context"
	"fmt"
	"math"
	"strconv"
	"strings"
)

// CalculatorTool evaluates basic arithmetic expressions in the form "a op b".
type CalculatorTool struct{}

func (c *CalculatorTool) Name() string { return "calculator" }
func (c *CalculatorTool) Description() string {
	return "Evaluates simple math expressions such as '2 + 2' or '5 * 3'."
}

func (c *CalculatorTool) Run(_ context.Context, input string) (string, error) {
	fields := strings.Fields(input)
	if len(fields) != 3 {
		return "", fmt.Errorf("expected format '<number> <op> <number>'")
	}

	left, err := strconv.ParseFloat(fields[0], 64)
	if err != nil {
		return "", fmt.Errorf("invalid left operand: %w", err)
	}
	right, err := strconv.ParseFloat(fields[2], 64)
	if err != nil {
		return "", fmt.Errorf("invalid right operand: %w", err)
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
			return "", fmt.Errorf("division by zero")
		}
		result = left / right
	default:
		return "", fmt.Errorf("unsupported operator %q", fields[1])
	}

	return strconv.FormatFloat(result, 'f', -1, 64), nil
}
