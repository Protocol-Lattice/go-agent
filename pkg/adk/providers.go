package adk

import (
	"context"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
)

// ModelProvider constructs a language model used by the coordinator agent.
type ModelProvider func(ctx context.Context) (models.Agent, error)

// SharedSessionFactory builds collaborative session views on top of the base
// session memory. It mirrors memory.NewSharedSession but lets providers expose
// preconfigured factories (for example to reuse ACL registries or custom
// defaults).
type SharedSessionFactory func(local string, spaces ...string) *memory.SharedSession

// MemoryBundle groups the session memory used by the coordinator together with
// an optional shared-session factory so callers can join collaborative spaces.
type MemoryBundle struct {
	Session *memory.SessionMemory
	Shared  SharedSessionFactory
}

// MemoryProvider provisions the conversational memory layer used by agents and
// optionally exposes helper factories to construct shared sessions.
type MemoryProvider func(ctx context.Context) (MemoryBundle, error)

// ToolBundle describes the tool catalog and the concrete tool instances
// contributed by a module. Catalog may be nil meaning the default catalog
// should be used.
type ToolBundle struct {
	Catalog agent.ToolCatalog
	Tools   []agent.Tool
}

// ToolProvider returns a ToolBundle to be merged into the agent configuration.
type ToolProvider func(ctx context.Context) (ToolBundle, error)

// SubAgentBundle exposes delegated agents and an optional directory.
type SubAgentBundle struct {
	Directory agent.SubAgentDirectory
	SubAgents []agent.SubAgent
}

// SubAgentProvider returns the sub-agent bundle to merge into the agent.
type SubAgentProvider func(ctx context.Context) (SubAgentBundle, error)

// AgentOption is applied to the low-level agent options prior to constructing
// the coordinator agent instance.
type AgentOption func(*agent.Options)
