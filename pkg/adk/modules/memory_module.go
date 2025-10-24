package modules

import (
	"context"
	"fmt"

	"github.com/Raezil/lattice-agent/pkg/adk"
)

// MemoryModule registers a session memory provider with the kit.
type MemoryModule struct {
	name     string
	provider adk.MemoryProvider
}

// NewMemoryModule creates a memory module with the supplied provider. If name
// is empty the module is registered as "memory".
func NewMemoryModule(name string, provider adk.MemoryProvider) *MemoryModule {
	if name == "" {
		name = "memory"
	}
	return &MemoryModule{name: name, provider: provider}
}

func (m *MemoryModule) Name() string { return m.name }

func (m *MemoryModule) Provision(_ context.Context, kitInstance *adk.AgentDevelopmentKit) error {
	if m.provider == nil {
		return fmt.Errorf("memory provider is nil")
	}
	kitInstance.UseMemoryProvider(m.provider)
	return nil
}
