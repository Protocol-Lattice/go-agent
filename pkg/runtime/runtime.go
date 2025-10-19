package runtime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/universal-tool-calling-protocol/go-utcp"
)

// ModelLoader constructs a language model instance for the primary agent.
type ModelLoader func(ctx context.Context) (models.Agent, error)

// MemoryBankFactory creates the long-term memory store used by the runtime.
type MemoryBankFactory func(ctx context.Context, connStr string) (*memory.MemoryBank, error)

// SessionMemoryFactory wraps the memory bank with a short-term cache implementation.
type SessionMemoryFactory func(bank *memory.MemoryBank, window int) *memory.SessionMemory

// Option configures runtime construction.
type Option func(*config)

type config struct {
	dsn                  string
	schemaPath           string
	sessionWindow        int
	contextLimit         int
	systemPrompt         string
	coordinatorModel     ModelLoader
	tools                []agent.Tool
	subAgents            []agent.SubAgent
	memoryFactory        MemoryBankFactory
	sessionMemoryFactory SessionMemoryFactory
	utcpClient           utcp.UtcpClientInterface
}

func defaultConfig() *config {
	return &config{
		sessionWindow:        8,
		contextLimit:         8,
		systemPrompt:         "You are the primary coordinator for an AI agent team.",
		memoryFactory:        defaultMemoryFactory,
		sessionMemoryFactory: memory.NewSessionMemory,
	}
}

func (c *config) validate() error {
	if c.coordinatorModel == nil {
		return errors.New("runtime requires a coordinator model loader")
	}
	if c.memoryFactory == nil {
		return errors.New("runtime requires a memory factory")
	}
	return nil
}

func (c *config) sessionWindowValue() int {
	if c.sessionWindow <= 0 {
		return 8
	}
	return c.sessionWindow
}

func (c *config) contextLimitValue() int {
	if c.contextLimit <= 0 {
		return 8
	}
	return c.contextLimit
}

func (c *config) systemPromptValue() string {
	if strings.TrimSpace(c.systemPrompt) == "" {
		return "You are the primary coordinator for an AI agent team."
	}
	return c.systemPrompt
}

func (c *config) memoryFactoryFunc() MemoryBankFactory {
	if c.memoryFactory != nil {
		return c.memoryFactory
	}
	return defaultMemoryFactory
}

func (c *config) sessionMemoryFactoryFunc() SessionMemoryFactory {
	if c.sessionMemoryFactory != nil {
		return c.sessionMemoryFactory
	}
	return memory.NewSessionMemory
}

func defaultMemoryFactory(ctx context.Context, connStr string) (*memory.MemoryBank, error) {
	if strings.TrimSpace(connStr) == "" {
		return memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), nil
	}
	return memory.NewMemoryBank(ctx, connStr)
}

// WithDSN configures the connection string for the default memory factory.
func WithDSN(dsn string) Option {
	return func(c *config) {
		c.dsn = strings.TrimSpace(dsn)
	}
}

// WithSchemaPath configures an optional schema path applied after the memory bank is created.
func WithSchemaPath(path string) Option {
	return func(c *config) {
		c.schemaPath = strings.TrimSpace(path)
	}
}

// WithSessionWindow overrides the short-term memory window size.
func WithSessionWindow(window int) Option {
	return func(c *config) {
		c.sessionWindow = window
	}
}

// WithContextLimit overrides how many records are retrieved from memory when generating prompts.
func WithContextLimit(limit int) Option {
	return func(c *config) {
		c.contextLimit = limit
	}
}

// WithSystemPrompt replaces the default coordinator system prompt.
func WithSystemPrompt(prompt string) Option {
	return func(c *config) {
		c.systemPrompt = prompt
	}
}

// WithCoordinatorModel sets the loader responsible for constructing the coordinator model.
func WithCoordinatorModel(loader ModelLoader) Option {
	return func(c *config) {
		c.coordinatorModel = loader
	}
}

// WithTools registers one or more tools with the runtime.
func WithTools(tools ...agent.Tool) Option {
	return func(c *config) {
		for _, tool := range tools {
			if tool == nil {
				continue
			}
			c.tools = append(c.tools, tool)
		}
	}
}

// WithSubAgents registers specialist sub-agents with the runtime.
func WithSubAgents(subAgents ...agent.SubAgent) Option {
	return func(c *config) {
		for _, sa := range subAgents {
			if sa == nil {
				continue
			}
			c.subAgents = append(c.subAgents, sa)
		}
	}
}

// WithMemoryFactory supplies a custom memory bank factory.
func WithMemoryFactory(factory MemoryBankFactory) Option {
	return func(c *config) {
		if factory != nil {
			c.memoryFactory = factory
		}
	}
}

// WithSessionMemoryBuilder supplies a custom session memory builder.
func WithSessionMemoryBuilder(builder SessionMemoryFactory) Option {
	return func(c *config) {
		if builder != nil {
			c.sessionMemoryFactory = builder
		}
	}
}

// WithUTCPClient attaches a UTCP client instance to the runtime.
func WithUTCPClient(client utcp.UtcpClientInterface) Option {
	return func(c *config) {
		c.utcpClient = client
	}
}

// Runtime wires together models, tools, memory and sub-agents into a cohesive execution environment.
type Runtime struct {
	agent  *agent.Agent
	memory *memory.SessionMemory
	bank   *memory.MemoryBank

	sessions *sessionManager
}

// New builds a runtime based on the supplied configuration options.
func New(ctx context.Context, opts ...Option) (*Runtime, error) {
	cfg := defaultConfig()
	for _, opt := range opts {
		if opt != nil {
			opt(cfg)
		}
	}

	if err := cfg.validate(); err != nil {
		return nil, err
	}

	bank, err := cfg.memoryFactoryFunc()(ctx, cfg.dsn)
	if err != nil {
		return nil, fmt.Errorf("create memory bank: %w", err)
	}

	if cfg.schemaPath != "" {
		if err := bank.CreateSchema(ctx, cfg.schemaPath); err != nil {
			bank.Close()
			return nil, fmt.Errorf("apply schema: %w", err)
		}
	}

	sessionMemory := cfg.sessionMemoryFactoryFunc()(bank, cfg.sessionWindowValue())

	coordinator, err := cfg.coordinatorModel(ctx)
	if err != nil {
		bank.Close()
		return nil, fmt.Errorf("load coordinator model: %w", err)
	}

	agentInstance, err := agent.New(agent.Options{
		Model:        coordinator,
		Memory:       sessionMemory,
		SystemPrompt: cfg.systemPromptValue(),
		ContextLimit: cfg.contextLimitValue(),
		Tools:        append([]agent.Tool(nil), cfg.tools...),
		SubAgents:    append([]agent.SubAgent(nil), cfg.subAgents...),
		UTCPClient:   cfg.utcpClient,
	})
	if err != nil {
		bank.Close()
		return nil, fmt.Errorf("initialise coordinator agent: %w", err)
	}

	rt := &Runtime{
		agent:  agentInstance,
		memory: sessionMemory,
		bank:   bank,
	}
	rt.sessions = newSessionManager(rt)
	return rt, nil
}

// Agent exposes the underlying primary agent orchestration component.
func (rt *Runtime) Agent() *agent.Agent {
	return rt.agent
}

// Memory returns the shared session memory cache.
func (rt *Runtime) Memory() *memory.SessionMemory {
	return rt.memory
}

// Tools returns the toolset attached to the runtime.
func (rt *Runtime) Tools() []agent.Tool {
	return rt.agent.Tools()
}

// SubAgents returns the delegated specialist components available to the runtime.
func (rt *Runtime) SubAgents() []agent.SubAgent {
	return rt.agent.SubAgents()
}

// Close releases resources associated with the runtime's memory bank.
func (rt *Runtime) Close() error {
	return rt.bank.Close()
}

// NewSession provisions an interactive session. If id is empty a unique identifier is generated.
func (rt *Runtime) NewSession(id string) *Session {
	return rt.sessions.newSession(id)
}

// GetSession retrieves an active session by its ID.
func (rt *Runtime) GetSession(id string) (*Session, error) {
	return rt.sessions.getSession(strings.TrimSpace(id))
}

// RemoveSession removes a session from the active sessions map.
func (rt *Runtime) RemoveSession(id string) {
	rt.sessions.removeSession(strings.TrimSpace(id))
}

// ActiveSessions returns a copy of all active session IDs.
func (rt *Runtime) ActiveSessions() []string {
	return rt.sessions.activeIDs()
}

// Generate forwards a user prompt to the coordinator agent and captures the response.
func (rt *Runtime) Generate(ctx context.Context, sessionID string, userInput string) (string, error) {
	return rt.agent.Respond(ctx, sessionID, userInput)
}

// Session encapsulates the conversational context for a single user.
type Session struct {
	runtime *Runtime
	id      string
}

// ID returns the unique identifier associated with the session.
func (s *Session) ID() string { return s.id }

// Flush persists short-term memory into the configured long-term store.
func (s *Session) Flush(ctx context.Context) error {
	if s.runtime == nil {
		return errors.New("session runtime is nil")
	}
	return s.runtime.agent.Flush(ctx, s.id)
}

// CloseFlush is a helper that flushes the session and logs any failure using the provided logger function.
func (s *Session) CloseFlush(ctx context.Context, logger func(error)) {
	if err := s.Flush(ctx); err != nil && logger != nil {
		logger(err)
	}
}

// Sleep is a small helper to make demos deterministic in tests by injecting delays.
func Sleep(d time.Duration) {
	time.Sleep(d)
}
