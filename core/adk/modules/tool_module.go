package modules

import (
	"context"

	"github.com/Protocol-Lattice/agent/core/adk"
)

// ToolModule registers a tool provider with the kit.
type ToolModule struct {
	name     string
	provider adk.ToolProvider
}

// NewToolModule constructs a tool module. If name is empty it defaults to
// "tools". When registering multiple tool modules provide distinct names to
// preserve ordering in diagnostics.
func NewToolModule(name string, provider adk.ToolProvider) *ToolModule {
	if name == "" {
		name = "tools"
	}
	return &ToolModule{name: name, provider: provider}
}

func (m *ToolModule) Name() string { return m.name }

func (m *ToolModule) Provision(_ context.Context, kitInstance *adk.AgentDevelopmentKit) error {
	kitInstance.UseToolProvider(m.provider)
	return nil
}
