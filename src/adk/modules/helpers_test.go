package modules

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	agent "github.com/Protocol-Lattice/go-agent"
	kit "github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/model"
	"github.com/Protocol-Lattice/go-agent/src/memory/store"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

type stubAgent struct{ called int32 }

func (s *stubAgent) Generate(context.Context, string) (any, error) {
	atomic.AddInt32(&s.called, 1)
	return "ok", nil
}

func (s *stubAgent) GenerateWithFiles(context.Context, string, []models.File) (any, error) {
	atomic.AddInt32(&s.called, 1)
	return "files", nil
}

type stubTool struct{ name string }

func (s stubTool) Spec() agent.ToolSpec { return agent.ToolSpec{Name: s.name} }
func (stubTool) Invoke(context.Context, agent.ToolRequest) (agent.ToolResponse, error) {
	return agent.ToolResponse{}, nil
}

type stubSubAgent struct{ name string }

func (s stubSubAgent) Name() string                              { return s.name }
func (stubSubAgent) Description() string                         { return "stub" }
func (stubSubAgent) Run(context.Context, string) (string, error) { return "", nil }

type stubVectorStore struct{}

func (stubVectorStore) StoreMemory(context.Context, string, string, map[string]any, []float32) error {
	return nil
}

func (stubVectorStore) SearchMemory(context.Context, []float32, int) ([]model.MemoryRecord, error) {
	return nil, nil
}

func (stubVectorStore) UpdateEmbedding(context.Context, int64, []float32, time.Time) error {
	return nil
}

func (stubVectorStore) DeleteMemory(context.Context, []int64) error {
	return nil
}

func (stubVectorStore) Iterate(context.Context, func(model.MemoryRecord) bool) error {
	return nil
}

func (stubVectorStore) Count(context.Context) (int, error) { return 0, nil }

var _ store.VectorStore = (*stubVectorStore)(nil)

type stubSchemaStore struct {
	stubVectorStore
	schemaCalled int32
}

func (s *stubSchemaStore) CreateSchema(context.Context, string) error {
	atomic.AddInt32(&s.schemaCalled, 1)
	return nil
}

var _ store.SchemaInitializer = (*stubSchemaStore)(nil)

type stubSubAgentDirectory struct{}

func (stubSubAgentDirectory) Register(agent.SubAgent) error        { return nil }
func (stubSubAgentDirectory) Lookup(string) (agent.SubAgent, bool) { return nil, false }
func (stubSubAgentDirectory) All() []agent.SubAgent                { return nil }

func TestStaticModelProvider(t *testing.T) {
	model := &stubAgent{}
	provider := StaticModelProvider(model)
	got, err := provider(context.Background())
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if got != model {
		t.Fatalf("expected same model instance")
	}
}

func TestStaticMemoryProvider(t *testing.T) {
	bank := newMemoryBankWithStore(&stubVectorStore{})
	mem := newSessionMemory(bank, 4)
	bundle, err := StaticMemoryProvider(mem)(context.Background())
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if bundle.Session != mem {
		t.Fatalf("expected same memory instance")
	}
	shared := bundle.Shared("local", "space-a")
	if shared == nil {
		t.Fatalf("expected shared session factory to produce session")
	}

	bundle, err = StaticMemoryProvider(nil)(context.Background())
	if err != nil {
		t.Fatalf("nil memory provider returned error: %v", err)
	}
	if bundle.Shared("any") != nil {
		t.Fatalf("expected shared factory to return nil when session memory is nil")
	}
}

func TestStaticToolProvider(t *testing.T) {
	tools := []agent.Tool{stubTool{name: "alpha"}, nil, stubTool{name: "beta"}}
	bundle, err := StaticToolProvider(tools, nil)(context.Background())
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if len(bundle.Tools) != 2 {
		t.Fatalf("expected two tools, got %d", len(bundle.Tools))
	}
	if bundle.Tools[0].Spec().Name != "alpha" || bundle.Tools[1].Spec().Name != "beta" {
		t.Fatalf("unexpected tool order: %+v", bundle.Tools)
	}
}

func TestStaticSubAgentProvider(t *testing.T) {
	sas := []agent.SubAgent{stubSubAgent{name: "one"}, nil, stubSubAgent{name: "two"}}
	bundle, err := StaticSubAgentProvider(sas, nil)(context.Background())
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if len(bundle.SubAgents) != 2 {
		t.Fatalf("expected 2 sub-agents, got %d", len(bundle.SubAgents))
	}
	if bundle.SubAgents[0].Name() != "one" || bundle.SubAgents[1].Name() != "two" {
		t.Fatalf("unexpected sub-agent order")
	}
}

func TestModelModuleProvision(t *testing.T) {
	module := NewModelModule("", StaticModelProvider(&stubAgent{}))
	if module.Name() != "model" {
		t.Fatalf("expected default name 'model'")
	}
	kitInstance := &kit.AgentDevelopmentKit{}
	if err := module.Provision(context.Background(), kitInstance); err != nil {
		t.Fatalf("provision failed: %v", err)
	}
	if kitInstance.ModelProvider() == nil {
		t.Fatalf("expected model provider to be registered")
	}

	bad := NewModelModule("custom", nil)
	if bad.Name() != "custom" {
		t.Fatalf("expected custom name preserved")
	}
	if err := bad.Provision(context.Background(), kitInstance); err == nil {
		t.Fatalf("expected error when provider is nil")
	}
}

func TestToolModuleProvision(t *testing.T) {
	invoked := false
	provider := func(context.Context) (kit.ToolBundle, error) {
		invoked = true
		return kit.ToolBundle{}, nil
	}
	module := NewToolModule("", provider)
	if module.Name() != "tools" {
		t.Fatalf("expected default name 'tools'")
	}
	kitInstance := &kit.AgentDevelopmentKit{}
	if err := module.Provision(context.Background(), kitInstance); err != nil {
		t.Fatalf("provision failed: %v", err)
	}
	if len(kitInstance.ToolProviders()) != 1 {
		t.Fatalf("expected tool provider to be registered")
	}
	if !invoked {
		if _, err := kitInstance.ToolProviders()[0](context.Background()); err != nil {
			t.Fatalf("invoking provider failed: %v", err)
		}
	}
}

func TestSubAgentModuleProvision(t *testing.T) {
	provider := StaticSubAgentProvider([]agent.SubAgent{stubSubAgent{name: "x"}}, nil)
	module := NewSubAgentModule("", provider)
	if module.Name() != "subagents" {
		t.Fatalf("expected default name 'subagents'")
	}
	kitInstance := &kit.AgentDevelopmentKit{}
	if err := module.Provision(context.Background(), kitInstance); err != nil {
		t.Fatalf("provision failed: %v", err)
	}
	if len(kitInstance.SubAgentProviders()) != 1 {
		t.Fatalf("expected sub-agent provider to be registered")
	}
}

func TestInMemoryMemoryModule(t *testing.T) {
	opts := memory.DefaultOptions()
	module := InMemoryMemoryModule(0, memory.DummyEmbedder{}, &opts)
	bundle, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if bundle.Session == nil || bundle.Shared == nil {
		t.Fatalf("bundle not fully initialised: %+v", bundle)
	}
	if bundle.Shared("local") == nil {
		t.Fatalf("expected shared session for default window")
	}
	if _, err := bundle.Session.Embed(context.Background(), "hello"); err != nil {
		t.Fatalf("embed failed: %v", err)
	}

	module = InMemoryMemoryModule(3, memory.DummyEmbedder{}, &opts)
	bundle, err = module.provider(context.Background())
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if bundle.Session == nil || bundle.Shared == nil {
		t.Fatalf("expected session memory on second invocation")
	}
	if bundle.Shared("local") == nil {
		t.Fatalf("expected shared session for positive window")
	}
}

func TestInQdrantMemory(t *testing.T) {
	opts := memory.DefaultOptions()
	module := InQdrantMemory(0, "http://localhost:6333", "collection", memory.DummyEmbedder{}, &opts)
	bundle, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("provider returned error: %v", err)
	}
	if bundle.Session == nil || bundle.Shared == nil {
		t.Fatalf("expected session and shared factory")
	}
	if bundle.Shared("local") == nil {
		t.Fatalf("expected shared session for default window")
	}

	module = InQdrantMemory(4, "http://localhost:6333", "collection", memory.DummyEmbedder{}, &opts)
	bundle, err = module.provider(context.Background())
	if err != nil {
		t.Fatalf("provider returned error for positive window: %v", err)
	}
	if bundle.Shared("local") == nil {
		t.Fatalf("expected shared session for positive window")
	}
}

func TestInPostgresMemoryCachesError(t *testing.T) {
	original := newPostgresStore
	defer func() { newPostgresStore = original }()

	var calls int32
	newPostgresStore = func(context.Context, string) (*memory.PostgresStore, error) {
		atomic.AddInt32(&calls, 1)
		return nil, errors.New("connect failed")
	}

	opts := memory.DefaultOptions()
	module := InPostgresMemory(context.Background(), 4, "postgres://invalid", memory.DummyEmbedder{}, &opts)
	provider := module.provider

	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected error on first call")
	}
	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected cached error on second call")
	}
	if calls != 1 {
		t.Fatalf("expected single attempt to create store, got %d", calls)
	}
}

func TestInPostgresMemorySuccess(t *testing.T) {
	originalStore := newPostgresStore
	originalShared := newSharedSession
	originalBank := newMemoryBankWithStore
	originalSession := newSessionMemory

	defer func() {
		newPostgresStore = originalStore
		newSharedSession = originalShared
		newMemoryBankWithStore = originalBank
		newSessionMemory = originalSession
	}()

	fakeStore := &stubSchemaStore{}
	newPostgresStore = func(context.Context, string) (*memory.PostgresStore, error) {
		return (*memory.PostgresStore)(nil), nil
	}

	newMemoryBankWithStore = func(store.VectorStore) *memory.MemoryBank {
		return memory.NewMemoryBankWithStore(fakeStore)
	}

	newSessionMemory = func(bank *memory.MemoryBank, size int) *memory.SessionMemory {
		return memory.NewSessionMemory(bank, size)
	}

	opts := memory.DefaultOptions()
	module := InPostgresMemory(context.Background(), 0, "", memory.DummyEmbedder{}, &opts)
	bundle, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bundle.Session == nil {
		t.Fatalf("expected session memory")
	}
	shared := bundle.Shared("local")
	if shared == nil {
		t.Fatalf("expected shared session")
	}

	cached, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("unexpected error on cached provider: %v", err)
	}
	if cached.Session != bundle.Session {
		t.Fatalf("expected cached session to be reused")
	}
}

func TestInMongoMemoryCachesError(t *testing.T) {
	original := newMongoStore
	defer func() { newMongoStore = original }()

	var calls int32
	newMongoStore = func(context.Context, string, string, string) (*store.MongoStore, error) {
		atomic.AddInt32(&calls, 1)
		return nil, errors.New("connect failed")
	}

	opts := memory.DefaultOptions()
	module := InMongoMemory(context.Background(), 4, "mongodb://invalid", "db", "collection", memory.DummyEmbedder{}, &opts)
	provider := module.provider

	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected error on first call")
	}
	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected cached error on second call")
	}
	if calls != 1 {
		t.Fatalf("expected single attempt to create store, got %d", calls)
	}
}

func TestInMongoMemorySuccess(t *testing.T) {
	originalStore := newMongoStore
	originalBank := newMemoryBankWithStore
	originalSession := newSessionMemory

	defer func() {
		newMongoStore = originalStore
		newMemoryBankWithStore = originalBank
		newSessionMemory = originalSession
	}()

	fakeStore := &stubSchemaStore{}
	newMongoStore = func(context.Context, string, string, string) (*store.MongoStore, error) {
		return nil, nil
	}

	newMemoryBankWithStore = func(store.VectorStore) *memory.MemoryBank {
		return memory.NewMemoryBankWithStore(fakeStore)
	}

	newSessionMemory = func(bank *memory.MemoryBank, size int) *memory.SessionMemory {
		return memory.NewSessionMemory(bank, size)
	}

	opts := memory.DefaultOptions()
	module := InMongoMemory(context.Background(), 0, "", "", "", memory.DummyEmbedder{}, &opts)
	bundle, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bundle.Session == nil {
		t.Fatalf("expected session memory")
	}
	shared := bundle.Shared("local")
	if shared == nil {
		t.Fatalf("expected shared session")
	}

	cached, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("unexpected error on cached provider: %v", err)
	}
	if cached.Session != bundle.Session {
		t.Fatalf("expected cached session to be reused")
	}
}

func TestInNeo4jMemoryNilFactory(t *testing.T) {
	opts := memory.DefaultOptions()
	module := InNeo4jMemory(context.Background(), 4, nil, memory.DummyEmbedder{}, &opts)
	provider := module.provider

	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected error when factory is nil")
	}
	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected cached error when factory is nil")
	}
}

func TestInNeo4jMemoryNilStore(t *testing.T) {
	opts := memory.DefaultOptions()
	module := InNeo4jMemory(context.Background(), 4, func(context.Context) (*memory.Neo4jStore, error) {
		return nil, nil
	}, memory.DummyEmbedder{}, &opts)
	provider := module.provider

	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected error when store is nil")
	}
	if _, err := provider(context.Background()); err == nil {
		t.Fatalf("expected cached error when store is nil")
	}
}

func TestInNeo4jMemorySuccess(t *testing.T) {
	opts := memory.DefaultOptions()
	module := InNeo4jMemory(context.Background(), 0, func(context.Context) (*memory.Neo4jStore, error) {
		return &memory.Neo4jStore{}, nil
	}, memory.DummyEmbedder{}, &opts)
	bundle, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if bundle.Session == nil {
		t.Fatalf("expected session memory")
	}
	shared := bundle.Shared("local")
	if shared == nil {
		t.Fatalf("expected shared session")
	}

	cached, err := module.provider(context.Background())
	if err != nil {
		t.Fatalf("unexpected error on cached provider: %v", err)
	}
	if cached.Session != bundle.Session {
		t.Fatalf("expected cached session to be reused")
	}
}

func TestMemoryModuleProvision(t *testing.T) {
	module := NewMemoryModule("", nil)
	if module.Name() != "memory" {
		t.Fatalf("expected default name 'memory'")
	}
	if err := module.Provision(context.Background(), &kit.AgentDevelopmentKit{}); err == nil {
		t.Fatalf("expected error when provider is nil")
	}

	bank := newMemoryBankWithStore(&stubVectorStore{})
	mem := newSessionMemory(bank, 2)
	good := NewMemoryModule("custom", StaticMemoryProvider(mem))
	kitInstance := &kit.AgentDevelopmentKit{}
	if err := good.Provision(context.Background(), kitInstance); err != nil {
		t.Fatalf("expected successful provision, got %v", err)
	}
	if kitInstance.MemoryProvider() == nil {
		t.Fatalf("memory provider not registered")
	}
}
