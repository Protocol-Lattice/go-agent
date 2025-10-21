package swarm

import (
	"context"
	"testing"
)

// Ensures Save() is a no-op (and does not panic) when Shared == nil.
// This guards against accidental nil-derefs in CLI/demo flows where a
// participant may be partially constructed (e.g., before wiring memory).
func TestParticipantSave_NoShared_NoPanic(t *testing.T) {
	t.Parallel()

	p := &Participant{
		Alias:     "researcher",
		SessionID: "cli:researcher",
		Agent:     nil, // intentionally nil
		Shared:    nil, // critical for this test
	}

	// Should not panic:
	p.Save(context.Background())
}
