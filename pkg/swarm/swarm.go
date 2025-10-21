package swarm

import (
	"context"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

type Swarm struct {
	*Participants
}

func NewSwarm(participants *Participants) *Swarm {
	return &Swarm{
		Participants: participants,
	}
}

func (swarm *Swarm) GetParticipant(id string) *Participant {
	return (*swarm.Participants)[id]
}

func (swarm *Swarm) Save(ctx context.Context) {
	for _, p := range *swarm.Participants {
		p.Save(ctx)
	}
}

func (swarm *Swarm) Retrieve(ctx context.Context, id string) ([]memory.MemoryRecord, error) {
	return swarm.GetParticipant(id).Retrieve(ctx)
}

func (swarm *Swarm) Join(id, space string) bool {
	return swarm.GetParticipant(id).Join(space)
}

func (swarm *Swarm) Leave(id, space string) {
	swarm.GetParticipant(id).Leave(space)
}
