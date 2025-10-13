package agent

import "context"

// Tool is an executable capability the primary agent can call into.
type Tool interface {
	Name() string
	Description() string
	Run(ctx context.Context, input string) (string, error)
}

// SubAgent represents a specialist agent that can be delegated work.
type SubAgent interface {
	Name() string
	Description() string
	Run(ctx context.Context, input string) (string, error)
}
