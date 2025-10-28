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

func TestSwarm_Join_Leave_Retrieve_Delegate(t *testing.T) {
	t.Parallel()

	// Reuse fakes from participant_test.go
	fs := &fakeShared{spacesVal: []string{"team:core"}}
	fa := &fakeAgent{}

	ps := Participants{
		"r": {Alias: "r", SessionID: "cli:r", Agent: fa, Shared: fs},
	}
	s := NewSwarm(&ps)

	// Join
	if errFlag := s.Join("r", "team:core"); errFlag {
		t.Fatalf("Join should succeed")
	}
	if len(fs.joinCalls) != 1 || fs.joinCalls[0] != "team:core" {
		t.Fatalf("expected Join delegated once, got %#v", fs.joinCalls)
	}

	// Retrieve
	_, _ = s.Retrieve(context.Background(), "r")
	if !fs.retrieveCalled || fs.retrieveQuery != "recent swarm updates" || fs.retrieveK != 5 {
		t.Fatalf("retrieve not delegated correctly: called=%v q=%q k=%d", fs.retrieveCalled, fs.retrieveQuery, fs.retrieveK)
	}

	// Leave
	s.Leave("r", "team:core")
	if len(fs.leaveCalls) != 1 || fs.leaveCalls[0] != "team:core" {
		t.Fatalf("expected Leave delegated once, got %#v", fs.leaveCalls)
	}
}
