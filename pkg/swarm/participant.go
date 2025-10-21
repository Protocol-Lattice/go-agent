package swarm

import (
	"context"
	"fmt"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

// participant binds an Agent with its own session identity and shared-session handle.
type Participant struct {
	Alias     string
	SessionID string
	Agent     *agent.Agent
	Shared    *memory.SharedSession
}
type Participants map[string]*Participant

func (participant *Participant) Retrieve(ctx context.Context) ([]memory.MemoryRecord, error) {
	return participant.Shared.Retrieve(ctx, "recent swarm updates", 5)
}

func (participant *Participant) Leave(space string) {
	participant.Shared.Leave(space)

}

func (participant *Participant) Join(space string) bool {
	participant.Agent.EnsureSpaceGrants(participant.SessionID, []string{space})
	if err := participant.Shared.Join(space); err != nil {
		fmt.Printf("Unable to join %s: %v\n", space, err)
		return true
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
