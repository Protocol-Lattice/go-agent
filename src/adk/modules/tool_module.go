package modules

import (
	"context"

	"github.com/Protocol-Lattice/go-agent/src/adk"
)

// ToolModule registers a tool provider with the kit.
type ToolModule struct {
	name     string
	provider adk.ToolProvider
}

func (m *ToolModule) Name() string { return m.name }

func (m *ToolModule) Provision(_ context.Context, kitInstance *adk.AgentDevelopmentKit) error {
	kitInstance.UseToolProvider(m.provider)
	return nil
}
