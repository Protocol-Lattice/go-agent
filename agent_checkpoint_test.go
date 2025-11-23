package agent

import (
	"context"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
)

func TestAgentCheckpointAndRestore(t *testing.T) {
	ctx := context.Background()

	// 1. Setup initial agent
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 10)
	// Use DummyEmbedder to avoid external calls and ensure speed
	mem = mem.WithEmbedder(memory.DummyEmbedder{})

	agent, err := New(Options{
		Model:        &stubModel{response: "ok"},
		Memory:       mem,
		SystemPrompt: "Initial Prompt",
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	// 2. Add some state (memory)
	sessionID := "test-session"
	// storeMemory is unexported but accessible in the same package
	agent.storeMemory(sessionID, "user", "Hello world", nil)
	agent.storeMemory(sessionID, "assistant", "Hi there", nil)

	// 3. Checkpoint
	data, err := agent.Checkpoint()
	if err != nil {
		t.Fatalf("Checkpoint failed: %v", err)
	}

	if len(data) == 0 {
		t.Fatal("Checkpoint returned empty data")
	}

	// 4. Create new agent (simulate restart)
	newMem := memory.NewSessionMemory(&memory.MemoryBank{}, 10)
	newMem = newMem.WithEmbedder(memory.DummyEmbedder{})

	newAgent, err := New(Options{
		Model:        &stubModel{response: "ok"},
		Memory:       newMem,
		SystemPrompt: "Default Prompt", // Different from initial
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	// 5. Restore
	if err := newAgent.Restore(data); err != nil {
		t.Fatalf("Restore failed: %v", err)
	}

	// 6. Verify
	if newAgent.systemPrompt != "Initial Prompt" {
		t.Errorf("System prompt not restored. Got %q, want %q", newAgent.systemPrompt, "Initial Prompt")
	}

	// Verify memory
	// We can use RetrieveContext to check if memories are there.
	records, err := newAgent.memory.RetrieveContext(ctx, sessionID, "", 10)
	if err != nil {
		t.Fatalf("RetrieveContext failed: %v", err)
	}

	if len(records) != 2 {
		t.Errorf("Expected 2 memory records, got %d", len(records))
	}

	// Check content
	foundUser := false
	foundAssistant := false
	for _, r := range records {
		if r.Content == "Hello world" {
			foundUser = true
		}
		if r.Content == "Hi there" {
			foundAssistant = true
		}
	}

	if !foundUser {
		t.Error("User memory not found")
	}
	if !foundAssistant {
		t.Error("Assistant memory not found")
	}
}

func TestAgentCheckpointSharedSpaces(t *testing.T) {
	// Setup
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 10).WithEmbedder(memory.DummyEmbedder{})
	// Grant permissions in registry
	mem.Spaces.Grant("team:alpha", "agent-1", memory.SpaceRoleWriter, 0)
	mem.Spaces.Grant("team:beta", "agent-1", memory.SpaceRoleWriter, 0)

	shared := memory.NewSharedSession(mem, "agent-1", "team:alpha")

	agent, _ := New(Options{
		Model:  &stubModel{response: "ok"},
		Memory: mem,
		Shared: shared,
	})

	// Join another space
	if err := agent.Shared.Join("team:beta"); err != nil {
		t.Fatalf("Join failed: %v", err)
	}

	// Checkpoint
	data, err := agent.Checkpoint()
	if err != nil {
		t.Fatalf("Checkpoint failed: %v", err)
	}

	// Restore to new agent
	newMem := memory.NewSessionMemory(&memory.MemoryBank{}, 10).WithEmbedder(memory.DummyEmbedder{})
	// Simulate persistent registry: grant permissions again
	newMem.Spaces.Grant("team:alpha", "agent-1", memory.SpaceRoleWriter, 0)
	newMem.Spaces.Grant("team:beta", "agent-1", memory.SpaceRoleWriter, 0)

	newShared := memory.NewSharedSession(newMem, "agent-1") // No initial spaces
	newAgent, _ := New(Options{
		Model:  &stubModel{response: "ok"},
		Memory: newMem,
		Shared: newShared,
	})

	if err := newAgent.Restore(data); err != nil {
		t.Fatalf("Restore failed: %v", err)
	}

	// Verify spaces
	spaces := newAgent.Shared.Spaces()
	foundAlpha := false
	foundBeta := false
	for _, s := range spaces {
		if s == "team:alpha" {
			foundAlpha = true
		}
		if s == "team:beta" {
			foundBeta = true
		}
	}

	if !foundAlpha {
		t.Error("Expected to be joined to team:alpha")
	}
	if !foundBeta {
		t.Error("Expected to be joined to team:beta")
	}
}
