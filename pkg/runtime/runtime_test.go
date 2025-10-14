package runtime

import (
	"context"
	"strings"
	"testing"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func TestConfigValidate(t *testing.T) {
	cfg := Config{}
	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected error when coordinator model is missing")
	}

	cfg.CoordinatorModel = func(ctx context.Context) (models.Agent, error) {
		return models.NewDummyLLM("demo"), nil
	}
	if err := cfg.Validate(); err == nil {
		t.Fatalf("expected error when DSN and memory factory are missing")
	}

	cfg.DSN = "postgres://user:pass@localhost/db"
	if err := cfg.Validate(); err != nil {
		t.Fatalf("unexpected validate error: %v", err)
	}
}

func TestRuntimeNewSessionAndAsk(t *testing.T) {
	ctx := context.Background()
	cfg := Config{
		CoordinatorModel: func(ctx context.Context) (models.Agent, error) {
			return models.NewDummyLLM("Coordinator:"), nil
		},
		MemoryFactory: func(ctx context.Context, _ string) (*memory.MemoryBank, error) {
			return &memory.MemoryBank{}, nil
		},
		SessionMemoryBuilder: func(bank *memory.MemoryBank, window int) *memory.SessionMemory {
			return memory.NewSessionMemory(bank, window)
		},
		Tools: []agent.Tool{&tools.EchoTool{}},
	}

	rt, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("runtime.New returned error: %v", err)
	}

	session := rt.NewSession("test")
	if session.ID() != "test" {
		t.Fatalf("expected session id to be 'test', got %q", session.ID())
	}

	reply, err := rt.Generate(ctx, session.ID(), "hello world")
	if err != nil {
		t.Fatalf("session ask returned error: %v", err)
	}
	if !strings.HasPrefix(reply, "Coordinator:") {
		t.Fatalf("expected reply to start with coordinator prefix, got %q", reply)
	}

	generated := rt.NewSession("")
	if generated.ID() == "" {
		t.Fatalf("expected generated session id to be non-empty")
	}
}

func TestRuntimeToolsImmutable(t *testing.T) {
	cfg := Config{
		CoordinatorModel: func(ctx context.Context) (models.Agent, error) {
			return models.NewDummyLLM("Coordinator:"), nil
		},
		MemoryFactory: func(ctx context.Context, _ string) (*memory.MemoryBank, error) {
			return &memory.MemoryBank{}, nil
		},
		Tools: []agent.Tool{&tools.EchoTool{}},
	}
	rt, err := New(context.Background(), cfg)
	if err != nil {
		t.Fatalf("runtime.New returned error: %v", err)
	}

	tools := rt.Tools()
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}
	tools[0] = nil

	toolsAgain := rt.Tools()
	if toolsAgain[0] == nil {
		t.Fatalf("runtime.Tools returned mutable slice")
	}
}
