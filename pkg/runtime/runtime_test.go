package runtime

import (
	"context"
	"errors"
	"reflect"
	"strings"
	"testing"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
)

func TestRuntimeNewValidation(t *testing.T) {
	ctx := context.Background()
	if _, err := New(ctx); err == nil {
		t.Fatalf("expected error when coordinator model is missing")
	}

	failingFactory := func(context.Context, string) (*memory.MemoryBank, error) {
		return nil, errors.New("boom")
	}

	if _, err := New(ctx, WithCoordinatorModel(func(context.Context) (models.Agent, error) {
		return models.NewDummyLLM("demo"), nil
	}), WithMemoryFactory(failingFactory)); err == nil {
		t.Fatalf("expected error when memory factory fails")
	}

	if _, err := New(ctx, WithCoordinatorModel(func(context.Context) (models.Agent, error) {
		return models.NewDummyLLM("demo"), nil
	})); err != nil {
		t.Fatalf("unexpected error creating runtime with defaults: %v", err)
	}
}

func TestRuntimeNewSessionAndAsk(t *testing.T) {
	ctx := context.Background()
	rt, err := New(ctx,
		WithCoordinatorModel(func(context.Context) (models.Agent, error) {
			return models.NewDummyLLM("Coordinator:"), nil
		}),
		WithTools(&tools.EchoTool{}),
	)
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

	active := rt.ActiveSessions()
	if len(active) != 2 {
		t.Fatalf("expected 2 active sessions, got %d", len(active))
	}
}

func TestRuntimeToolsImmutable(t *testing.T) {
	rt, err := New(context.Background(),
		WithCoordinatorModel(func(context.Context) (models.Agent, error) {
			return models.NewDummyLLM("Coordinator:"), nil
		}),
		WithTools(&tools.EchoTool{}),
	)
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

func TestRuntimeActiveSessionsSortedAndRemovable(t *testing.T) {
	rt, err := New(context.Background(),
		WithCoordinatorModel(func(context.Context) (models.Agent, error) {
			return models.NewDummyLLM("Coordinator:"), nil
		}),
	)
	if err != nil {
		t.Fatalf("runtime.New returned error: %v", err)
	}

	rt.NewSession("b")
	rt.NewSession("a")
	rt.NewSession("c")

	got := rt.ActiveSessions()
	want := []string{"a", "b", "c"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expected sorted sessions %v, got %v", want, got)
	}

	rt.RemoveSession("b")
	remaining := rt.ActiveSessions()
	want = []string{"a", "c"}
	if !reflect.DeepEqual(remaining, want) {
		t.Fatalf("expected sessions %v after removal, got %v", want, remaining)
	}

	if _, err := rt.GetSession("missing"); err == nil {
		t.Fatalf("expected error for missing session")
	}
}
