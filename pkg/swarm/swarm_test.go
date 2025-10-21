package swarm

import (
	"context"
	"testing"
)

func TestNewSwarm_AndGetParticipant(t *testing.T) {
	t.Parallel()

	ps := Participants{
		"researcher": {Alias: "researcher", SessionID: "cli:researcher"},
		"planner":    {Alias: "planner", SessionID: "cli:planner"},
	}
	s := NewSwarm(&ps)

	if got := s.GetParticipant("researcher"); got == nil || got.Alias != "researcher" {
		t.Fatalf("expected researcher participant, got %#v", got)
	}
	if got := s.GetParticipant("summarizer"); got != nil {
		t.Fatalf("expected nil for unknown id, got %#v", got)
	}
}

func TestSwarmSave_NoParticipants_NoPanic(t *testing.T) {
	t.Parallel()

	empty := Participants{}
	s := NewSwarm(&empty)

	// Should not panic even when there are no participants.
	s.Save(context.Background())
}
