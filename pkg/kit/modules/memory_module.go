package modules

import (
	"context"
	"fmt"

	"github.com/Raezil/go-agent-development-kit/pkg/kit"
)

// MemoryModule registers a session memory provider with the kit.
type MemoryModule struct {
	name     string
	provider kit.MemoryProvider
}

// NewMemoryModule creates a memory module with the supplied provider. If name
// is empty the module is registered as "memory".
func NewMemoryModule(name string, provider kit.MemoryProvider) *MemoryModule {
	if name == "" {
		name = "memory"
	}
	return &MemoryModule{name: name, provider: provider}
}

func (m *MemoryModule) Name() string { return m.name }

func (m *MemoryModule) Provision(_ context.Context, kitInstance *kit.AgentDevelopmentKit) error {
	if m.provider == nil {
		return fmt.Errorf("memory provider is nil")
	}
	kitInstance.UseMemoryProvider(m.provider)
	return nil
}
