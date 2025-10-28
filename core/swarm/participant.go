package swarm

import (
	"context"
	"fmt"

	"github.com/Protocol-Lattice/agent/core/memory"
)

// ---- Exported, narrow interfaces for testability & cross-package impls ----

// SharedSession is a lightweight façade over memory.SharedSession used by swarm participants.
type SharedSession interface {
	Retrieve(ctx context.Context, query string, k int) ([]memory.MemoryRecord, error)
	Leave(space string)
	Join(space string) error
	FlushLocal(ctx context.Context) error
	Spaces() []string
	FlushSpace(ctx context.Context, space string) error
}

// SpaceGranter is the minimal capability a Participant needs from an Agent.
type SpaceGranter interface {
	EnsureSpaceGrants(sessionID string, spaces []string)
}

// ConversationAgent is the minimal surface main.go relies on for chat + space wiring.
type ConversationAgent interface {
	SpaceGranter
	SetSharedSpaces(shared SharedSession)
	Save(ctx context.Context, role, content string) // note: no error to match *agent.Agent
	Generate(ctx context.Context, sessionID, prompt string) (string, error)
}

// Participant binds an agent-like runner with its own session identity and shared-session handle.
type Participant struct {
	Alias     string
	SessionID string
	Agent     ConversationAgent
	Shared    SharedSession
}

type Participants map[string]*Participant

func (participant *Participant) Retrieve(ctx context.Context) ([]memory.MemoryRecord, error) {
	if participant.Shared == nil {
		return nil, nil
	}
	return participant.Shared.Retrieve(ctx, "recent swarm updates", 5)
}

func (participant *Participant) Leave(space string) {
	if participant.Shared == nil {
		return
	}
	participant.Shared.Leave(space)
}

func (participant *Participant) Join(space string) bool {
	// Always ensure grants first (safe even if Agent is nil).
	if participant.Agent != nil {
		participant.Agent.EnsureSpaceGrants(participant.SessionID, []string{space})
	}

	if participant.Shared == nil {
		// Nothing to join; treat as success (no error).
		return false
	}

	if err := participant.Shared.Join(space); err != nil {
		fmt.Printf("Unable to join %s: %v\n", space, err)
		return true // true => error occurred
	}
	return false
}

func (participant *Participant) Save(ctx context.Context) {
	if participant.Shared == nil {
		return
	}
	if err := participant.Shared.FlushLocal(ctx); err != nil {
		fmt.Printf("[%s] flush local: %v\n", participant.Alias, err)
	}
	for _, space := range participant.Shared.Spaces() {
		if err := participant.Shared.FlushSpace(ctx, space); err != nil {
			fmt.Printf("[%s] flush space %s: %v\n", participant.Alias, space, err)
		}
	}
}
