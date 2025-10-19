package kit

import "context"

// Module represents a pluggable unit that can provision capabilities on the
// AgentDevelopmentKit. Modules are executed during bootstrapping and can
// install model providers, memory backends, tool registries, sub-agents, or any
// other integration required by the host application.
type Module interface {
	// Name returns a human-friendly identifier used purely for debugging
	// and log messages. Names do not need to be unique, but providing
	// descriptive names makes it easier to understand bootstrapping
	// failures.
	Name() string

	// Provision attaches functionality to the kit. Implementations may call
	// the various Use* helpers to register providers or mutate defaults.
	Provision(ctx context.Context, kit *AgentDevelopmentKit) error
}
