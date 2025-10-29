package adk

import (
	"context"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

// Option configures the AgentDevelopmentKit during construction.
type Option func(*AgentDevelopmentKit) error

// WithModule registers a single module with the kit.
func WithModule(module Module) Option {
	return func(kit *AgentDevelopmentKit) error {
		return kit.RegisterModule(module)
	}
}

// WithModules registers multiple modules using a single option invocation.
func WithModules(modules ...Module) Option {
	return func(kit *AgentDevelopmentKit) error {
		for _, module := range modules {
			if err := kit.RegisterModule(module); err != nil {
				return err
			}
		}
		return nil
	}
}

// WithDefaultSystemPrompt overrides the default system prompt used by the kit.
func WithDefaultSystemPrompt(prompt string) Option {
	return func(kit *AgentDevelopmentKit) error {
		kit.SetDefaultSystemPrompt(prompt)
		return nil
	}
}

// WithDefaultContextLimit overrides the default context window size used when
// constructing agents.
func WithDefaultContextLimit(limit int) Option {
	return func(kit *AgentDevelopmentKit) error {
		kit.SetDefaultContextLimit(limit)
		return nil
	}
}

// WithAgentOptions registers default agent options that will be applied to
// every agent created via BuildAgent.
func WithAgentOptions(opts ...AgentOption) Option {
	return func(kit *AgentDevelopmentKit) error {
		for _, opt := range opts {
			kit.UseAgentOption(opt)
		}
		return nil
	}
}

// WithSubAgents registers one or more sub-agents directly on the kit. The
// sub-agents are appended to the aggregated set before the coordinator agent is
// constructed. Nil entries are ignored to simplify conditional wiring.
func WithSubAgents(subAgents ...agent.SubAgent) Option {
	return func(kit *AgentDevelopmentKit) error {
		provider := func(context.Context) (SubAgentBundle, error) {
			bundle := SubAgentBundle{}
			for _, sa := range subAgents {
				if sa == nil {
					continue
				}
				bundle.SubAgents = append(bundle.SubAgents, sa)
			}
			return bundle, nil
		}
		kit.UseSubAgentProvider(provider)
		return nil
	}
}

func WithUTCP(client utcp.UtcpClientInterface) Option {
	return func(kit *AgentDevelopmentKit) error {
		kit.UTCP = client
		return nil
	}
}
