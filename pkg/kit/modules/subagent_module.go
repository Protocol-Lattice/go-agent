package modules

import (
	"context"

	"github.com/Raezil/go-agent-development-kit/pkg/kit"
)

// SubAgentModule registers a sub-agent provider with the kit.
type SubAgentModule struct {
	name     string
	provider kit.SubAgentProvider
}

// NewSubAgentModule creates a sub-agent module. If name is empty the module is
// registered as "subagents".
func NewSubAgentModule(name string, provider kit.SubAgentProvider) *SubAgentModule {
	if name == "" {
		name = "subagents"
	}
	return &SubAgentModule{name: name, provider: provider}
}

func (m *SubAgentModule) Name() string { return m.name }

func (m *SubAgentModule) Provision(_ context.Context, kitInstance *kit.AgentDevelopmentKit) error {
	kitInstance.UseSubAgentProvider(m.provider)
	return nil
}
