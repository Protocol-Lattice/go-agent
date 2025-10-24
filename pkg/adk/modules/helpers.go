package modules

import (
	"context"
	"log"
	"os"

	kit "github.com/Raezil/lattice-agent/pkg/adk"
	"github.com/Raezil/lattice-agent/pkg/agent"
	"github.com/Raezil/lattice-agent/pkg/memory"
	"github.com/Raezil/lattice-agent/pkg/models"
)

// StaticModelProvider returns a provider that always yields the supplied model.
func StaticModelProvider(model models.Agent) kit.ModelProvider {
	return func(context.Context) (models.Agent, error) {
		return model, nil
	}
}

// StaticMemoryProvider returns a provider that always yields the supplied
// session memory instance.
func StaticMemoryProvider(mem *memory.SessionMemory) kit.MemoryProvider {
	shared := func(local string, spaces ...string) *memory.SharedSession {
		if mem == nil {
			return nil
		}
		return memory.NewSharedSession(mem, local, spaces...)
	}
	return func(context.Context) (kit.MemoryBundle, error) {
		return kit.MemoryBundle{Session: mem, Shared: shared}, nil
	}
}

// InMemoryMemoryModule constructs a memory module backed by the in-memory
// store. The underlying bank is initialised once and reused so agents built
// from the same kit share memories and collaborative spaces.
func InMemoryMemoryModule(window int, embeeder memory.Embedder, opts *memory.Options) *MemoryModule {
	size := window
	if size <= 0 {
		size = 8
	}
	bank := memory.NewMemoryBankWithStore(memory.NewInMemoryStore())
	mem := memory.NewSessionMemory(bank, size)
	mem.WithEmbedder(embeeder)
	engine := memory.NewEngine(bank.Store, *opts)
	mem.WithEngine(engine)
	engineLogger := log.New(os.Stderr, "memory-engine: ", log.LstdFlags)
	mem.Engine.WithLogger(engineLogger)

	provider := func(context.Context) (kit.MemoryBundle, error) {
		shared := func(local string, spaces ...string) *memory.SharedSession {
			return memory.NewSharedSession(mem, local, spaces...)
		}
		return kit.MemoryBundle{Session: mem, Shared: shared}, nil
	}
	return NewMemoryModule("memory", provider)
}

func InQdrantMemory(window int, baseURL string, collection string, embeeder memory.Embedder, opts *memory.Options) *MemoryModule {
	size := window
	if size <= 0 {
		size = 8
	}
	bank := memory.NewMemoryBankWithStore(memory.NewQdrantStore(baseURL, collection, ""))
	mem := memory.NewSessionMemory(bank, size)
	mem.WithEmbedder(embeeder)
	engine := memory.NewEngine(bank.Store, *opts)
	mem.WithEngine(engine)
	engineLogger := log.New(os.Stderr, "memory-engine: ", log.LstdFlags)
	mem.Engine.WithLogger(engineLogger)

	provider := func(context.Context) (kit.MemoryBundle, error) {
		shared := func(local string, spaces ...string) *memory.SharedSession {
			return memory.NewSharedSession(mem, local, spaces...)
		}
		return kit.MemoryBundle{Session: mem, Shared: shared}, nil
	}
	return NewMemoryModule("qdrant", provider)
}

func InPostgresMemory(ctx context.Context, window int, connStr string, embeeder memory.Embedder, opts *memory.Options) *MemoryModule {
	var (
		cached    *memory.SessionMemory
		cachedErr error
	)
	provider := func(context.Context) (kit.MemoryBundle, error) {
		if cached == nil && cachedErr == nil {
			ps, err := memory.NewPostgresStore(ctx, connStr)
			if err != nil {
				cachedErr = err
				return kit.MemoryBundle{}, err
			}
			bank := memory.NewMemoryBankWithStore(ps)
			size := window
			if size <= 0 {
				size = 8
			}
			mem := memory.NewSessionMemory(bank, size)
			mem.WithEmbedder(embeeder)
			engine := memory.NewEngine(bank.Store, *opts)
			mem.WithEngine(engine)
			engineLogger := log.New(os.Stderr, "memory-engine: ", log.LstdFlags)
			mem.Engine.WithLogger(engineLogger)
			cached = mem
		}
		if cached == nil {
			return kit.MemoryBundle{}, cachedErr
		}
		shared := func(local string, spaces ...string) *memory.SharedSession {
			return memory.NewSharedSession(cached, local, spaces...)
		}
		return kit.MemoryBundle{Session: cached, Shared: shared}, nil
	}
	return NewMemoryModule("postgres", provider)
}

// StaticToolProvider wraps a fixed tool slice and optional catalog into a
// provider implementation.
func StaticToolProvider(tools []agent.Tool, catalog agent.ToolCatalog) kit.ToolProvider {
	return func(context.Context) (kit.ToolBundle, error) {
		bundle := kit.ToolBundle{Catalog: catalog}
		for _, tool := range tools {
			if tool == nil {
				continue
			}
			bundle.Tools = append(bundle.Tools, tool)
		}
		return bundle, nil
	}
}

// StaticSubAgentProvider wraps a fixed set of sub-agents and optional
// directory.
func StaticSubAgentProvider(subAgents []agent.SubAgent, directory agent.SubAgentDirectory) kit.SubAgentProvider {
	return func(context.Context) (kit.SubAgentBundle, error) {
		bundle := kit.SubAgentBundle{Directory: directory}
		for _, sa := range subAgents {
			if sa == nil {
				continue
			}
			bundle.SubAgents = append(bundle.SubAgents, sa)
		}
		return bundle, nil
	}
}
