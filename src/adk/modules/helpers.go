package modules

import (
	"context"
	"errors"
	"log"
	"os"
	"sync"

	agent "github.com/Protocol-Lattice/go-agent"
	kit "github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	memorystore "github.com/Protocol-Lattice/go-agent/src/memory/store"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

var (
	newMemoryBankWithStore = memory.NewMemoryBankWithStore
	newSessionMemory       = memory.NewSessionMemory
	newEngine              = memory.NewEngine
	newSharedSession       = memory.NewSharedSession
	newPostgresStore       = memory.NewPostgresStore
	newMongoStore          = memorystore.NewMongoStore
)

const defaultMemoryWindow = 8

type memoryStoreFactory func() (memorystore.VectorStore, error)

func memoryOptions(opts *memory.Options) memory.Options {
	if opts == nil {
		return memory.DefaultOptions()
	}
	return *opts
}

func memoryBundle(store memorystore.VectorStore, window int, embedder memory.Embedder, opts *memory.Options) kit.MemoryBundle {
	if window <= 0 {
		window = defaultMemoryWindow
	}

	bank := newMemoryBankWithStore(store)
	mem := newSessionMemory(bank, window)
	mem.WithEmbedder(embedder)
	engine := newEngine(bank.Store, memoryOptions(opts))
	engine.WithLogger(log.New(os.Stderr, "memory-engine: ", log.LstdFlags))
	mem.WithEngine(engine)

	shared := func(local string, spaces ...string) *memory.SharedSession {
		return newSharedSession(mem, local, spaces...)
	}
	return kit.MemoryBundle{Session: mem, Shared: shared}
}

func eagerMemoryModule(name string, store memorystore.VectorStore, window int, embedder memory.Embedder, opts *memory.Options) *MemoryModule {
	bundle := memoryBundle(store, window, embedder, opts)
	provider := func(context.Context) (kit.MemoryBundle, error) {
		return bundle, nil
	}
	return NewMemoryModule(name, provider)
}

func lazyMemoryModule(name string, factory memoryStoreFactory, window int, embedder memory.Embedder, opts *memory.Options) *MemoryModule {
	var (
		once    sync.Once
		bundle  kit.MemoryBundle
		initErr error
	)

	provider := func(context.Context) (kit.MemoryBundle, error) {
		once.Do(func() {
			store, err := factory()
			if err != nil {
				initErr = err
				return
			}
			bundle = memoryBundle(store, window, embedder, opts)
		})
		if initErr != nil {
			return kit.MemoryBundle{}, initErr
		}
		return bundle, nil
	}

	return NewMemoryModule(name, provider)
}

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
func InMemoryMemoryModule(window int, embedder memory.Embedder, opts *memory.Options) *MemoryModule {
	return eagerMemoryModule("memory", memory.NewInMemoryStore(), window, embedder, opts)
}

func InQdrantMemory(window int, baseURL string, collection string, embedder memory.Embedder, opts *memory.Options) *MemoryModule {
	return eagerMemoryModule("qdrant", memory.NewQdrantStore(baseURL, collection, ""), window, embedder, opts)
}

func InPostgresMemory(ctx context.Context, window int, connStr string, embedder memory.Embedder, opts *memory.Options) *MemoryModule {
	factory := func() (memorystore.VectorStore, error) {
		return newPostgresStore(ctx, connStr)
	}
	return lazyMemoryModule("postgres", factory, window, embedder, opts)
}

func InMongoMemory(ctx context.Context, window int, uri, database, collection string, embedder memory.Embedder, opts *memory.Options) *MemoryModule {
	factory := func() (memorystore.VectorStore, error) {
		return newMongoStore(ctx, uri, database, collection)
	}
	return lazyMemoryModule("mongo", factory, window, embedder, opts)
}

func InNeo4jMemory(ctx context.Context, window int, factory func(context.Context) (*memory.Neo4jStore, error), embedder memory.Embedder, opts *memory.Options) *MemoryModule {
	storeFactory := func() (memorystore.VectorStore, error) {
		if factory == nil {
			return nil, errors.New("neo4j store factory is nil")
		}
		store, err := factory(ctx)
		if err != nil {
			return nil, err
		}
		if store == nil {
			return nil, errors.New("neo4j store is nil")
		}
		return store, nil
	}
	return lazyMemoryModule("neo4j", storeFactory, window, embedder, opts)
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
