package tools

import (
	"context"
	"strings"
)

// EchoTool repeats the provided input. Useful for testing tool wiring.
type EchoTool struct{}

func (e *EchoTool) Name() string        { return "echo" }
func (e *EchoTool) Description() string { return "Echoes the provided text back to the caller." }

func (e *EchoTool) Run(_ context.Context, input string) (string, error) {
	return strings.TrimSpace(input), nil
}
