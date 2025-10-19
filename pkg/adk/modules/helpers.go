package modules

import (
	"context"
	"log"
	"os"

	kit "github.com/Raezil/go-agent-development-kit/pkg/adk"
	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/runtime"
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
	return func(context.Context) (*memory.SessionMemory, error) {
		return mem, nil
	}
}

// InMemoryMemoryModule constructs a memory module backed by the in-memory
// store. A new memory bank is created for each request which keeps agents
// isolated by default.
func InMemoryMemoryModule(window int, embeeder memory.Embedder, opts *memory.Options) *MemoryModule {
	provider := func(context.Context) (*memory.SessionMemory, error) {
		bank := memory.NewMemoryBankWithStore(memory.NewInMemoryStore())
		size := window
		if size <= 0 {
			size = 8
		}
		mem := memory.NewSessionMemory(bank, size)
		mem.WithEmbedder(embeeder)
		mem.WithEmbedder(embeeder)
		engine := memory.NewEngine(bank.Store, *opts)
		mem.WithEngine(engine)
		engineLogger := log.New(os.Stderr, "memory-engine: ", log.LstdFlags)
		mem.Engine.WithLogger(engineLogger)
		return mem, nil
	}
	return NewMemoryModule("memory", provider)
}

func InQdrantMemory(window int, baseURL string, collection string, embeeder memory.Embedder, opts *memory.Options) *MemoryModule {
	provider := func(context.Context) (*memory.SessionMemory, error) {
		bank := memory.NewMemoryBankWithStore(memory.NewQdrantStore(baseURL, collection, ""))
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
		return mem, nil
	}
	return NewMemoryModule("qdrant", provider)
}

func InPostgresMemory(ctx context.Context, window int, connStr string, embeeder memory.Embedder, opts *memory.Options) *MemoryModule {
	provider := func(context.Context) (*memory.SessionMemory, error) {
		ps, err := memory.NewPostgresStore(ctx, connStr)
		if err != nil {
			return nil, err
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
		return mem, nil
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

// StaticRuntimeProvider returns a provider that always yields the supplied
// runtime instance.
func StaticRuntimeProvider(rt *runtime.Runtime) kit.RuntimeProvider {
	return func(context.Context) (*runtime.Runtime, error) {
		return rt, nil
	}
}
