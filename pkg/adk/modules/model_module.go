package modules

import (
	"context"
	"fmt"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
)

// ModelModule wires a model provider into the kit.
type ModelModule struct {
	name     string
	provider adk.ModelProvider
}

// NewModelModule creates a module that registers the supplied model provider.
// If name is empty the module will expose "model".
func NewModelModule(name string, provider adk.ModelProvider) *ModelModule {
	if name == "" {
		name = "model"
	}
	return &ModelModule{name: name, provider: provider}
}

func (m *ModelModule) Name() string { return m.name }

func (m *ModelModule) Provision(_ context.Context, kitInstance *adk.AgentDevelopmentKit) error {
	if m.provider == nil {
		return fmt.Errorf("model provider is nil")
	}
	kitInstance.UseModelProvider(m.provider)
	return nil
}
