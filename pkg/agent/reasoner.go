package agent

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
)

// Reasoner implements Planner. It can work in two modes:
//  1. heuristic-only (Model == nil) for deterministic, zero-cost planning
//  2. LLM-backed (Model provided) to refine the heuristic into a compact JSON plan
type Reasoner struct {
	model           models.Agent
	maxSteps        int
	enableToolHints bool
	timeout         time.Duration
}

// ReasonerOptions configures the Reasoner.
type ReasonerOptions struct {
	// Model is optional. When nil, Reasoner uses only heuristics.
	Model models.Agent
	// MaxSteps clamps the number of plan steps (default 6).
	MaxSteps int
	// EnableToolHints optionally adds a tool/sub-agent consideration step.
	EnableToolHints bool
	// Timeout bounds a single LLM planning call (default 15s). Ignored in heuristic-only mode.
	Timeout time.Duration
}

// NewReasoner constructs a Planner implementation.
func NewReasoner(opts ReasonerOptions) *Reasoner {
	r := &Reasoner{
		model:           opts.Model,
		maxSteps:        opts.MaxSteps,
		enableToolHints: opts.EnableToolHints,
		timeout:         opts.Timeout,
	}
	if r.maxSteps <= 0 {
		r.maxSteps = 6
	}
	if r.timeout <= 0 {
		r.timeout = 15 * time.Second
	}
	return r
}

// Plan produces a compact hidden plan. If an LLM model is configured, it refines
// a heuristic seed plan into a JSON-structured output; otherwise the heuristic is returned.
func (r *Reasoner) Plan(ctx context.Context, in PlannerInput) (PlannerOutput, error) {
	seed := r.heuristicPlan(in)
	if r.model == nil {
		return seed, nil
	}

	// Build a tight prompt to elicit strictly JSON output.
	prompt := r.buildPrompt(in, seed)

	// Respect existing deadline; otherwise apply a local timeout to avoid blocking.
	if _, hasDeadline := ctx.Deadline(); !hasDeadline && r.timeout > 0 {
		var cancel context.CancelFunc
		ctx, cancel = context.WithTimeout(ctx, r.timeout)
		defer cancel()
	}

	comp, err := r.model.Generate(ctx, prompt)
	if err != nil {
		// Honor caller cancellation/timeouts, otherwise fall back to heuristic.
		if ctx.Err() != nil {
			return PlannerOutput{}, ctx.Err()
		}
		return seed, nil
	}

	parsed := parsePlannerJSON(fmt.Sprint(comp))
	out := mergePlannerOutputs(parsed, seed)
	out.Steps = clamp(out.Steps, r.maxSteps)
	out.Thoughts = clamp(out.Thoughts, 3)
	return dedupTrim(out), nil
}

// ----------------- Heuristic planner -----------------

func (r *Reasoner) heuristicPlan(in PlannerInput) PlannerOutput {
	user := strings.ToLower(in.UserInput)
	steps := []string{}
	thoughts := []string{}
	decision := ""

	// Always consult recent memory when available.
	if len(in.Context) > 0 {
		steps = append(steps, "Skim recent memory for relevant facts")
	}

	switch {
	case hasAny(user, "error", "panic", "stack", "trace", "failed", "exception"):
		thoughts = append(thoughts, "User appears to be troubleshooting a failure.")
		steps = append(steps,
			"Restate the error succinctly",
			"List likely root causes based on the message and context",
			"Suggest minimal steps to isolate the issue",
			"Offer a safe fix or workaround with caveats")
	case hasAny(user, "how to", "steps", "guide", "tutorial", "procedure"):
		thoughts = append(thoughts, "User is requesting a procedure.")
		steps = append(steps,
			"Outline a concise high-level plan",
			"Provide step-by-step instructions",
			"Call out pitfalls and verification checks")
	case hasAny(user, "compare", "vs", "tradeoff", "pros", "cons", "versus"):
		thoughts = append(thoughts, "User requests a comparison.")
		steps = append(steps,
			"Identify the evaluation criteria",
			"Compare options by criteria",
			"Recommend with reasoning and caveats")
	case hasAny(user, "code", "golang", " go ", "snippet", "example", "api"):
		thoughts = append(thoughts, "User may want Go code; focus on idioms, tests, and safety.")
		steps = append(steps,
			"Confirm the desired behavior and constraints",
			"Draft a minimal, working example",
			"Explain key choices, complexity, and next steps")
	default:
		thoughts = append(thoughts, "Default planning: clarify the ask and produce a focused answer.")
		steps = append(steps,
			"Identify the precise ask",
			"Use memory context to enrich the answer",
			"Respond concisely and propose next steps")
	}

	if r.enableToolHints {
		steps = append(steps, "Consider invoking a tool or sub-agent only if it materially improves accuracy or speed")
	}

	decision = "Proceed with the outlined steps; do not expose these notes."
	return PlannerOutput{
		Thoughts: thoughts,
		Steps:    clamp(steps, r.maxSteps),
		Decision: decision,
	}
}

// ----------------- LLM planning prompt -----------------

func (r *Reasoner) buildPrompt(in PlannerInput, seed PlannerOutput) string {
	var sb strings.Builder
	sb.WriteString("You are a compact planner that prepares hidden notes for an assistant.\n")
	sb.WriteString("Return STRICT JSON only with fields: thoughts (array of short strings), steps (array of short strings), decision (short string).\n")
	sb.WriteString("Rules: <=120 words total. No prose outside JSON. No secrets. Keep it crisp.\n\n")

	sb.WriteString("USER_INPUT:\n")
	sb.WriteString(truncate(in.UserInput, 800))
	sb.WriteString("\n\nRECENT_MEMORY:\n")
	for i, rec := range summariseMem(in.Context, 5) {
		sb.WriteString(fmt.Sprintf("%d. %s\n", i+1, rec))
	}

	if !seed.IsEmpty() {
		sb.WriteString("\nSEED_PLAN:\n")
		sb.WriteString(seed.Format())
		sb.WriteString("\n")
	}

	sb.WriteString("\nJSON ONLY:\n")
	sb.WriteString(`{"thoughts":["..."],"steps":["..."],"decision":"..."}`)
	return sb.String()
}

// ----------------- Helpers -----------------

func hasAny(s string, needles ...string) bool {
	for _, n := range needles {
		if strings.Contains(s, n) {
			return true
		}
	}
	return false
}

func truncate(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	if max <= 3 {
		return s[:max]
	}
	return s[:max-3] + "..."
}

func summariseMem(recs []memory.MemoryRecord, max int) []string {
	out := make([]string, 0, len(recs))
	for _, r := range recs {
		c := strings.TrimSpace(r.Content)
		if c == "" {
			continue
		}
		out = append(out, truncate(c, 200))
		if max > 0 && len(out) >= max {
			break
		}
	}
	return out
}

func clamp[T any](in []T, max int) []T {
	if max <= 0 || len(in) <= max {
		return in
	}
	cp := make([]T, max)
	copy(cp, in[:max])
	return cp
}

func dedupTrim(o PlannerOutput) PlannerOutput {
	seen := map[string]struct{}{}
	norm := func(s string) string { return strings.TrimSpace(s) }

	outSteps := make([]string, 0, len(o.Steps))
	for _, s := range o.Steps {
		n := norm(s)
		if n == "" {
			continue
		}
		if _, ok := seen[n]; ok {
			continue
		}
		seen[n] = struct{}{}
		outSteps = append(outSteps, n)
	}

	seen = map[string]struct{}{}
	outThoughts := make([]string, 0, len(o.Thoughts))
	for _, s := range o.Thoughts {
		n := norm(s)
		if n == "" {
			continue
		}
		if _, ok := seen[n]; ok {
			continue
		}
		seen[n] = struct{}{}
		outThoughts = append(outThoughts, n)
	}

	o.Steps = outSteps
	o.Thoughts = outThoughts
	o.Decision = norm(o.Decision)
	return o
}

type plannerJSON struct {
	Thoughts any `json:"thoughts"`
	Steps    any `json:"steps"`
	Decision any `json:"decision"`
}

func parsePlannerJSON(s string) PlannerOutput {
	trim := strings.TrimSpace(s)
	try := func(payload string) (PlannerOutput, bool) {
		var pj plannerJSON
		if err := json.Unmarshal([]byte(payload), &pj); err != nil {
			return PlannerOutput{}, false
		}
		out := PlannerOutput{
			Thoughts: anyToStrings(pj.Thoughts),
			Steps:    anyToStrings(pj.Steps),
			Decision: anyToString(pj.Decision),
		}
		return dedupTrim(out), true
	}

	// First try whole string.
	if out, ok := try(trim); ok {
		return out
	}
	// Try extracting JSON min-slice between the first '{' and last '}'.
	first := strings.Index(trim, "{")
	last := strings.LastIndex(trim, "}")
	if first >= 0 && last > first {
		if out, ok := try(trim[first : last+1]); ok {
			return out
		}
	}
	return PlannerOutput{}
}

func anyToStrings(v any) []string {
	switch t := v.(type) {
	case nil:
		return nil
	case string:
		s := strings.TrimSpace(t)
		if s == "" {
			return nil
		}
		return []string{s}
	case []any:
		out := make([]string, 0, len(t))
		for _, it := range t {
			if str := anyToString(it); str != "" {
				out = append(out, str)
			}
		}
		return out
	case []string:
		out := make([]string, 0, len(t))
		for _, s := range t {
			s = strings.TrimSpace(s)
			if s != "" {
				out = append(out, s)
			}
		}
		return out
	default:
		if s := anyToString(t); s != "" {
			return []string{s}
		}
		return nil
	}
}

func anyToString(v any) string {
	switch t := v.(type) {
	case nil:
		return ""
	case string:
		return strings.TrimSpace(t)
	case float64:
		// Avoid scientific format for integers; but planner shouldn't produce numbers here.
		return strings.TrimSpace(fmt.Sprintf("%g", t))
	default:
		b, _ := json.Marshal(t)
		return strings.TrimSpace(string(b))
	}
}

func mergePlannerOutputs(primary, fallback PlannerOutput) PlannerOutput {
	out := PlannerOutput{}
	if len(primary.Thoughts) == 0 {
		out.Thoughts = append([]string(nil), fallback.Thoughts...)
	} else {
		out.Thoughts = append([]string(nil), primary.Thoughts...)
	}
	if len(primary.Steps) == 0 {
		out.Steps = append([]string(nil), fallback.Steps...)
	} else {
		out.Steps = append([]string(nil), primary.Steps...)
	}
	if strings.TrimSpace(primary.Decision) == "" {
		out.Decision = fallback.Decision
	} else {
		out.Decision = primary.Decision
	}
	return out
}
