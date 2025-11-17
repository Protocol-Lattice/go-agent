package agent

import (
	"context"
	"strings"
)

// Save stores a conversation turn into all shared spaces.
// role should be "user" or "agent".
func (agent *Agent) Save(ctx context.Context, role, content string) {
	if agent.Shared == nil || strings.TrimSpace(content) == "" {
		return
	}
	meta := map[string]string{"role": role}
	for _, sp := range agent.Shared.Spaces() {
		if err := agent.Shared.AddShortTo(sp, content, meta); err != nil {
			continue
		}
		_ = agent.Shared.FlushSpace(ctx, sp) // persist to long-term
	}
}
