package agent

import (
	"context"
	"strconv"
	"strings"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

// Planner coordinates hidden reasoning steps (chain-of-thought or planning)
// before the primary model produces a user-facing answer.
//
// Implementations should keep responses concise and avoid leaking internal
// reasoning into the final assistant output. PlannerOutput is rendered and fed
// back into the main prompt as private notes.
type Planner interface {
	Plan(ctx context.Context, input PlannerInput) (PlannerOutput, error)
}

// PlannerInput bundles the information required to craft a hidden reasoning
// trace for the next assistant reply.
type PlannerInput struct {
	SessionID string
	UserInput string
	Context   []memory.MemoryRecord
}

// PlannerOutput captures hidden reasoning notes. Steps may describe a
// multi-stage plan, while Thoughts and Decision provide broader reflections or
// conclusions. Empty strings are ignored during rendering.
type PlannerOutput struct {
	Thoughts []string
	Steps    []string
	Decision string
}

// IsEmpty reports whether the planner produced any meaningful signal.
func (o PlannerOutput) IsEmpty() bool {
	if len(normaliseEntries(o.Thoughts)) > 0 {
		return false
	}
	if len(normaliseEntries(o.Steps)) > 0 {
		return false
	}
	if strings.TrimSpace(o.Decision) != "" {
		return false
	}
	return true
}

// Format returns a normalised, human-readable trace suitable for private
// planner notes inside the coordinator prompt.
func (o PlannerOutput) Format() string {
	if o.IsEmpty() {
		return ""
	}

	var sb strings.Builder

	if thoughts := normaliseEntries(o.Thoughts); len(thoughts) > 0 {
		sb.WriteString("Thoughts:\n")
		for i, thought := range thoughts {
			if i > 0 {
				sb.WriteByte('\n')
			}
			sb.WriteString(thought)
		}
	}

	if steps := normaliseEntries(o.Steps); len(steps) > 0 {
		if sb.Len() > 0 {
			sb.WriteByte('\n')
			sb.WriteByte('\n')
		}
		sb.WriteString("Plan:\n")
		counter := 1
		for _, step := range steps {
			sb.WriteString(strconv.Itoa(counter))
			sb.WriteString(". ")
			sb.WriteString(step)
			sb.WriteByte('\n')
			counter++
		}
		// Trim trailing newline for cleanliness.
		out := strings.TrimRight(sb.String(), "\n")
		sb.Reset()
		sb.WriteString(out)
	}

	if decision := strings.TrimSpace(o.Decision); decision != "" {
		if sb.Len() > 0 {
			sb.WriteByte('\n')
			sb.WriteByte('\n')
		}
		sb.WriteString("Decision:\n")
		sb.WriteString(decision)
	}

	return strings.TrimSpace(sb.String())
}

func normaliseEntries(items []string) []string {
	out := make([]string, 0, len(items))
	for _, item := range items {
		trimmed := strings.TrimSpace(item)
		if trimmed == "" {
			continue
		}
		out = append(out, trimmed)
	}
	return out
}
