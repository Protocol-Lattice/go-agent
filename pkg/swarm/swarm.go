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

func (swarm *Swarm) Save(ctx context.Context) error {
	for _, p := range *swarm.Participants {
		p.Save(ctx)
	}
	return nil
}

func (swarm *Swarm) Retrieve(ctx context.Context, id string) ([]memory.MemoryRecord, error) {
	p := swarm.GetParticipant(id)
	if p == nil {
		return nil, nil
	}
	return p.Retrieve(ctx)
}

func (swarm *Swarm) Join(id, space string) bool {
	p := swarm.GetParticipant(id)
	if p == nil {
		return true // signal error if participant unknown
	}
	return p.Join(space)
}

func (swarm *Swarm) Leave(id, space string) {
	p := swarm.GetParticipant(id)
	if p == nil {
		return
	}
	p.Leave(space)
}
