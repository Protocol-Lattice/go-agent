package modules

import (
	"context"
	"fmt"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
)

// RuntimeModule registers a runtime provider with the kit.
type RuntimeModule struct {
	name     string
	provider adk.RuntimeProvider
}

// NewRuntimeModule creates a runtime module with the provided factory. If name
// is empty the module uses "runtime".
func NewRuntimeModule(name string, provider adk.RuntimeProvider) *RuntimeModule {
	if name == "" {
		name = "runtime"
	}
	return &RuntimeModule{name: name, provider: provider}
}

func (m *RuntimeModule) Name() string { return m.name }

func (m *RuntimeModule) Provision(_ context.Context, kitInstance *adk.AgentDevelopmentKit) error {
	if m.provider == nil {
		return fmt.Errorf("runtime provider is nil")
	}
	kitInstance.UseRuntimeProvider(m.provider)
	return nil
}
