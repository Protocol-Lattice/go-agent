package adk

import (
	"context"
	"fmt"
	"strings"
	"sync"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/chain"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"

	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
	"github.com/universal-tool-calling-protocol/go-utcp/src/transports"
)

const defaultCoordinatorPrompt = "You are the primary coordinator for an AI agent team. Provide concise, accurate answers and explain when you call tools or delegate work to specialist sub-agents."

// AgentDevelopmentKit orchestrates modules that provision models, memory,
// tools, sub-agents and runtime glue. It acts as a lightweight dependency
// injection container for agent deployments.
type AgentDevelopmentKit struct {
	mu sync.RWMutex

	modules      []Module
	bootstrapped bool

	modelProvider    ModelProvider
	memoryProvider   MemoryProvider
	memoryInstance   *memory.SessionMemory
	sharedFactory    SharedSessionFactory
	toolProviders    []ToolProvider
	subAgentProvider []SubAgentProvider

	defaultSystemPrompt string
	defaultContextLimit int

	agentOptions []AgentOption
	UTCP         utcp.UtcpClientInterface
	CodeMode     codemode.CodeModeUTCP
	ChainMode    chain.UtcpChainClient
}

// New constructs a kit, applies the provided options and bootstraps registered
// modules. Modules may in turn register providers that are later used to build
// agents.
func New(ctx context.Context, opts ...Option) (*AgentDevelopmentKit, error) {
	kit := &AgentDevelopmentKit{
		defaultSystemPrompt: defaultCoordinatorPrompt,
		defaultContextLimit: 8,
	}

	for _, opt := range opts {
		if opt == nil {
			continue
		}
		if err := opt(kit); err != nil {
			return nil, err
		}
	}

	if err := kit.Bootstrap(ctx); err != nil {
		return nil, err
	}
	return kit, nil
}

// Bootstrap executes all registered modules. The operation is idempotent, so
// calling it multiple times is safe. Modules are executed in the order they were
// registered.
func (k *AgentDevelopmentKit) Bootstrap(ctx context.Context) error {
	k.mu.Lock()
	if k.bootstrapped {
		k.mu.Unlock()
		return nil
	}
	modules := append([]Module(nil), k.modules...)
	k.mu.Unlock()

	for _, module := range modules {
		if module == nil {
			continue
		}
		if err := module.Provision(ctx, k); err != nil {
			name := "<unnamed module>"
			if module.Name() != "" {
				name = module.Name()
			}
			return fmt.Errorf("kit module %s: %w", name, err)
		}
	}

	k.mu.Lock()
	k.bootstrapped = true
	k.mu.Unlock()
	return nil
}

// RegisterModule appends a module to the bootstrapping sequence.
func (k *AgentDevelopmentKit) RegisterModule(module Module) error {
	if module == nil {
		return fmt.Errorf("kit module cannot be nil")
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	k.modules = append(k.modules, module)
	k.bootstrapped = false
	return nil
}

// Modules returns a copy of the registered modules in registration order.
func (k *AgentDevelopmentKit) Modules() []Module {
	k.mu.RLock()
	defer k.mu.RUnlock()
	out := make([]Module, len(k.modules))
	copy(out, k.modules)
	return out
}

// UseModelProvider registers the provider responsible for constructing the
// coordinator language model.
func (k *AgentDevelopmentKit) UseModelProvider(provider ModelProvider) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.modelProvider = provider
}

// ModelProvider returns the currently registered model provider.
func (k *AgentDevelopmentKit) ModelProvider() ModelProvider {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.modelProvider
}

// UseMemoryProvider registers the provider that builds session memory
// instances.
func (k *AgentDevelopmentKit) UseMemoryProvider(provider MemoryProvider) {
	k.mu.Lock()
	defer k.mu.Unlock()
	k.memoryProvider = provider
}

// MemoryProvider returns the registered memory provider.
func (k *AgentDevelopmentKit) MemoryProvider() MemoryProvider {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.memoryProvider
}

// UseToolProvider appends a tool provider to the kit.
func (k *AgentDevelopmentKit) UseToolProvider(provider ToolProvider) {
	if provider == nil {
		return
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	k.toolProviders = append(k.toolProviders, provider)
}

// ToolProviders returns the registered tool providers in order.
func (k *AgentDevelopmentKit) ToolProviders() []ToolProvider {
	k.mu.RLock()
	defer k.mu.RUnlock()
	out := make([]ToolProvider, len(k.toolProviders))
	copy(out, k.toolProviders)
	return out
}

// UseSubAgentProvider appends a sub-agent provider to the kit.
func (k *AgentDevelopmentKit) UseSubAgentProvider(provider SubAgentProvider) {
	if provider == nil {
		return
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	k.subAgentProvider = append(k.subAgentProvider, provider)
}

// SubAgentProviders returns the registered sub-agent providers.
func (k *AgentDevelopmentKit) SubAgentProviders() []SubAgentProvider {
	k.mu.RLock()
	defer k.mu.RUnlock()
	out := make([]SubAgentProvider, len(k.subAgentProvider))
	copy(out, k.subAgentProvider)
	return out
}

// UseAgentOption appends a default agent option applied prior to constructing
// the coordinator agent.
func (k *AgentDevelopmentKit) UseAgentOption(opt AgentOption) {
	if opt == nil {
		return
	}
	k.mu.Lock()
	defer k.mu.Unlock()
	k.agentOptions = append(k.agentOptions, opt)
}

// AgentOptions returns a copy of the default agent options registered on the
// kit.
func (k *AgentDevelopmentKit) AgentOptions() []AgentOption {
	k.mu.RLock()
	defer k.mu.RUnlock()
	out := make([]AgentOption, len(k.agentOptions))
	copy(out, k.agentOptions)
	return out
}

// SetDefaultSystemPrompt overrides the default system prompt used when
// building agents.
func (k *AgentDevelopmentKit) SetDefaultSystemPrompt(prompt string) {
	k.mu.Lock()
	defer k.mu.Unlock()
	trimmed := strings.TrimSpace(prompt)
	if trimmed == "" {
		k.defaultSystemPrompt = defaultCoordinatorPrompt
		return
	}
	k.defaultSystemPrompt = trimmed
}

// DefaultSystemPrompt returns the system prompt applied to agents when no
// explicit prompt is provided.
func (k *AgentDevelopmentKit) DefaultSystemPrompt() string {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.defaultSystemPrompt
}

// SetDefaultContextLimit updates the default retrieval window size.
func (k *AgentDevelopmentKit) SetDefaultContextLimit(limit int) {
	k.mu.Lock()
	defer k.mu.Unlock()
	if limit <= 0 {
		k.defaultContextLimit = 8
		return
	}
	k.defaultContextLimit = limit
}

// DefaultContextLimit returns the configured default context limit.
func (k *AgentDevelopmentKit) DefaultContextLimit() int {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.defaultContextLimit
}

// BuildAgent constructs a coordinator agent using the registered providers and
// optional overrides.
func (k *AgentDevelopmentKit) BuildAgent(ctx context.Context, opts ...AgentOption) (*agent.Agent, error) {
	if err := k.Bootstrap(ctx); err != nil {
		return nil, err
	}

	k.mu.RLock()
	modelProvider := k.modelProvider
	memoryProvider := k.memoryProvider
	toolProviders := append([]ToolProvider(nil), k.toolProviders...)
	subAgentProviders := append([]SubAgentProvider(nil), k.subAgentProvider...)
	defaultPrompt := k.defaultSystemPrompt
	defaultLimit := k.defaultContextLimit
	defaultAgentOptions := append([]AgentOption(nil), k.agentOptions...)
	utcp := k.UTCP
	codeMode := k.CodeMode
	chainMode := k.ChainMode
	k.mu.RUnlock()

	if modelProvider == nil {
		return nil, fmt.Errorf("kit requires a model provider")
	}
	if memoryProvider == nil {
		return nil, fmt.Errorf("kit requires a memory provider")
	}

	model, err := modelProvider(ctx)
	if err != nil {
		return nil, fmt.Errorf("model provider: %w", err)
	}

	bundle, err := memoryProvider(ctx)
	if err != nil {
		return nil, fmt.Errorf("memory provider: %w", err)
	}
	if bundle.Session == nil {
		return nil, fmt.Errorf("memory provider: session memory is nil")
	}

	k.mu.Lock()
	if k.memoryInstance == nil {
		k.memoryInstance = bundle.Session
	}
	if bundle.Shared != nil {
		k.sharedFactory = bundle.Shared
	}
	k.mu.Unlock()

	toolBundles := make([]ToolBundle, 0, len(toolProviders))
	for _, provider := range toolProviders {
		bundle, err := provider(ctx)
		if err != nil {
			return nil, fmt.Errorf("tool provider: %w", err)
		}
		toolBundles = append(toolBundles, bundle)
	}

	subBundles := make([]SubAgentBundle, 0, len(subAgentProviders))
	for _, provider := range subAgentProviders {
		bundle, err := provider(ctx)
		if err != nil {
			return nil, fmt.Errorf("sub-agent provider: %w", err)
		}
		subBundles = append(subBundles, bundle)
	}

	agentOpts := agent.Options{
		Model:        model,
		Memory:       bundle.Session,
		SystemPrompt: defaultPrompt,
		ContextLimit: defaultLimit,
		UTCPClient:   utcp,
		CodeMode:     &codeMode,
		CodeChain:    &chainMode,
	}

	for _, opt := range defaultAgentOptions {
		if opt != nil {
			opt(&agentOpts)
		}
	}
	for _, opt := range opts {
		if opt != nil {
			opt(&agentOpts)
		}
	}

	if strings.TrimSpace(agentOpts.SystemPrompt) == "" {
		agentOpts.SystemPrompt = defaultCoordinatorPrompt
	}
	if agentOpts.ContextLimit <= 0 {
		agentOpts.ContextLimit = defaultLimit
		if agentOpts.ContextLimit <= 0 {
			agentOpts.ContextLimit = 8
		}
	}

	if agentOpts.ToolCatalog == nil {
		for _, bundle := range toolBundles {
			if bundle.Catalog != nil {
				agentOpts.ToolCatalog = bundle.Catalog
				break
			}
		}
	}
	if agentOpts.ToolCatalog == nil {
		agentOpts.ToolCatalog = agent.NewStaticToolCatalog(nil)
	}

	var aggregatedTools []agent.Tool
	for _, bundle := range toolBundles {
		for _, tool := range bundle.Tools {
			if tool == nil {
				continue
			}
			aggregatedTools = append(aggregatedTools, tool)
		}
	}
	agentOpts.Tools = append(agentOpts.Tools, aggregatedTools...)

	if agentOpts.SubAgentDirectory == nil {
		for _, bundle := range subBundles {
			if bundle.Directory != nil {
				agentOpts.SubAgentDirectory = bundle.Directory
				break
			}
		}
	}
	if agentOpts.SubAgentDirectory == nil {
		agentOpts.SubAgentDirectory = agent.NewStaticSubAgentDirectory(nil)
	}

	var aggregatedSubAgents []agent.SubAgent
	for _, bundle := range subBundles {
		for _, sa := range bundle.SubAgents {
			if sa == nil {
				continue
			}
			aggregatedSubAgents = append(aggregatedSubAgents, sa)
		}
	}
	agentOpts.SubAgents = append(agentOpts.SubAgents, aggregatedSubAgents...)

	built, err := agent.New(agentOpts)
	if err != nil {
		return nil, err
	}
	return built, nil
}

// SharedSession returns a collaborative session view backed by the configured
// session memory. Agents created from the same kit can use this to read or
// write memories in shared spaces (for example, team channels). The factory is
// lazily initialised on first use so callers may request shared sessions before
// or after constructing the coordinator agent.
func (k *AgentDevelopmentKit) NewSharedSession(ctx context.Context, local string, spaces ...string) (*memory.SharedSession, error) {
	if err := k.Bootstrap(ctx); err != nil {
		return nil, err
	}

	k.mu.RLock()
	factory := k.sharedFactory
	memoryProvider := k.memoryProvider
	instance := k.memoryInstance
	k.mu.RUnlock()

	if factory != nil {
		return factory(local, spaces...), nil
	}

	if instance != nil {
		return memory.NewSharedSession(instance, local, spaces...), nil
	}

	if memoryProvider == nil {
		return nil, fmt.Errorf("kit requires a memory provider")
	}

	bundle, err := memoryProvider(ctx)
	if err != nil {
		return nil, fmt.Errorf("memory provider: %w", err)
	}
	if bundle.Session == nil {
		return nil, fmt.Errorf("memory provider: session memory is nil")
	}

	factory = bundle.Shared
	if factory == nil {
		factory = func(local string, spaces ...string) *memory.SharedSession {
			return memory.NewSharedSession(bundle.Session, local, spaces...)
		}
	}

	k.mu.Lock()
	if k.memoryInstance == nil {
		k.memoryInstance = bundle.Session
	}
	if k.sharedFactory == nil {
		k.sharedFactory = factory
	}
	k.mu.Unlock()

	return factory(local, spaces...), nil
}

func (adk *AgentDevelopmentKit) CallTool(ctx context.Context, toolName string, args map[string]any) (any, error) {
	return adk.UTCP.CallTool(ctx, toolName, args)
}

func (adk *AgentDevelopmentKit) CallToolStream(ctx context.Context, toolName string, args map[string]any) (transports.StreamResult, error) {
	return adk.UTCP.CallToolStream(ctx, toolName, args)
}

func (adk *AgentDevelopmentKit) SearchTools(query string, limit int) ([]tools.Tool, error) {
	return adk.UTCP.SearchTools(query, limit)
}
