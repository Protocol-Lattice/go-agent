package runtime

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
)

// ModelLoader constructs a language model instance for the primary agent.
type ModelLoader func(ctx context.Context) (models.Agent, error)

// MemoryBankFactory creates the long-term memory store backed by Postgres.
type MemoryBankFactory func(ctx context.Context, connStr string) (*memory.MemoryBank, error)

// SessionMemoryFactory wraps the memory bank with a short-term cache implementation.
type SessionMemoryFactory func(bank *memory.MemoryBank, window int) *memory.SessionMemory

// Config controls how a Runtime instance is constructed.
type Config struct {
	DSN                  string
	SchemaPath           string
	SessionWindow        int
	ContextLimit         int
	SystemPrompt         string
	CoordinatorModel     ModelLoader
	Tools                []agent.Tool
	SubAgents            []agent.SubAgent
	MemoryFactory        MemoryBankFactory
	SessionMemoryBuilder SessionMemoryFactory
}

// Validate ensures the configuration has the minimum information required to build a runtime.
func (c Config) Validate() error {
	if c.CoordinatorModel == nil {
		return errors.New("runtime requires a coordinator model loader")
	}
	if c.MemoryFactory == nil && strings.TrimSpace(c.DSN) == "" {
		return errors.New("runtime requires a Postgres DSN when MemoryFactory is not provided")
	}
	return nil
}

func (c Config) sessionWindow() int {
	if c.SessionWindow <= 0 {
		return 8
	}
	return c.SessionWindow
}

func (c Config) contextLimit() int {
	if c.ContextLimit <= 0 {
		return 8
	}
	return c.ContextLimit
}

func (c Config) systemPrompt() string {
	if strings.TrimSpace(c.SystemPrompt) == "" {
		return "You are the primary coordinator for an AI agent team."
	}
	return c.SystemPrompt
}

func (c Config) memoryFactory() MemoryBankFactory {
	if c.MemoryFactory != nil {
		return c.MemoryFactory
	}
	return memory.NewMemoryBank
}

func (c Config) sessionMemoryFactory() SessionMemoryFactory {
	if c.SessionMemoryBuilder != nil {
		return c.SessionMemoryBuilder
	}
	return memory.NewSessionMemory
}

// Runtime wires together models, tools, memory and sub-agents into a cohesive execution environment.
type Runtime struct {
	agent     *agent.Agent
	memory    *memory.SessionMemory
	bank      *memory.MemoryBank
	tools     []agent.Tool
	subagents []agent.SubAgent

	sessionCounter atomic.Uint64
}

// New builds a runtime based on the supplied configuration.
func New(ctx context.Context, cfg Config) (*Runtime, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}

	bank, err := cfg.memoryFactory()(ctx, cfg.DSN)
	if err != nil {
		return nil, fmt.Errorf("create memory bank: %w", err)
	}

	if cfg.SchemaPath != "" {
		if err := bank.CreateSchema(ctx, cfg.SchemaPath); err != nil {
			if bank.DB != nil {
				bank.DB.Close()
			}
			return nil, fmt.Errorf("apply schema: %w", err)
		}
	}

	sessionMemory := cfg.sessionMemoryFactory()(bank, cfg.sessionWindow())

	coordinator, err := cfg.CoordinatorModel(ctx)
	if err != nil {
		if bank.DB != nil {
			bank.DB.Close()
		}
		return nil, fmt.Errorf("load coordinator model: %w", err)
	}

	agentInstance, err := agent.New(agent.Options{
		Model:        coordinator,
		Memory:       sessionMemory,
		SystemPrompt: cfg.systemPrompt(),
		ContextLimit: cfg.contextLimit(),
		Tools:        append([]agent.Tool(nil), cfg.Tools...),
		SubAgents:    append([]agent.SubAgent(nil), cfg.SubAgents...),
	})
	if err != nil {
		if bank.DB != nil {
			bank.DB.Close()
		}
		return nil, fmt.Errorf("initialise coordinator agent: %w", err)
	}

	rt := &Runtime{
		agent:     agentInstance,
		memory:    sessionMemory,
		bank:      bank,
		tools:     append([]agent.Tool(nil), cfg.Tools...),
		subagents: append([]agent.SubAgent(nil), cfg.SubAgents...),
	}
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
	return append([]agent.Tool(nil), rt.tools...)
}

// SubAgents returns the delegated specialist components available to the runtime.
func (rt *Runtime) SubAgents() []agent.SubAgent {
	return append([]agent.SubAgent(nil), rt.subagents...)
}

// Close releases database connections associated with the runtime.
func (rt *Runtime) Close() error {
	if rt.bank != nil && rt.bank.DB != nil {
		rt.bank.DB.Close()
	}
	return nil
}

// NewSession provisions an interactive session. If id is empty a unique identifier is generated.
func (rt *Runtime) NewSession(id string) *Session {
	if strings.TrimSpace(id) == "" {
		counter := rt.sessionCounter.Add(1)
		id = fmt.Sprintf("session-%d", counter)
	}
	return &Session{runtime: rt, id: id}
}

// Session encapsulates the conversational context for a single user.
type Session struct {
	runtime *Runtime
	id      string
}

// ID returns the unique identifier associated with the session.
func (s *Session) ID() string { return s.id }

// Ask forwards a user prompt to the coordinator agent and captures the response.
func (s *Session) Ask(ctx context.Context, userInput string) (string, error) {
	if s.runtime == nil {
		return "", errors.New("session runtime is nil")
	}
	return s.runtime.agent.Respond(ctx, s.id, userInput)
}

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
