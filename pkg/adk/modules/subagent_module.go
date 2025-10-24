package modules

import (
	"context"

	"github.com/Raezil/lattice-agent/pkg/adk"
)

// SubAgentModule registers a sub-agent provider with the kit.
type SubAgentModule struct {
	name     string
	provider adk.SubAgentProvider
}

// NewSubAgentModule creates a sub-agent module. If name is empty the module is
// registered as "subagents".
func NewSubAgentModule(name string, provider adk.SubAgentProvider) *SubAgentModule {
	if name == "" {
		name = "subagents"
	}
	return &SubAgentModule{name: name, provider: provider}
}

func (m *SubAgentModule) Name() string { return m.name }

func (m *SubAgentModule) Provision(_ context.Context, kitInstance *adk.AgentDevelopmentKit) error {
	kitInstance.UseSubAgentProvider(m.provider)
	return nil
}
