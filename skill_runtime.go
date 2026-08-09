package agent

import (
    "context"
    "fmt"
    "strings"

    "github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

// SkillRouting is the request-scoped result of Skill System 2.0 routing.
// It is intentionally immutable after creation so concurrent agent runs do
// not share active skill state.
type SkillRouting struct {
    Matches []SkillMatch
    Skills  []SkillDefinition
    Tools   map[string]struct{}
}

// SkillRegistry returns a registry backed by the Agent's configured skills.
// The registry is rebuilt from disk so edits to SKILL.md are visible on the
// next request without mutating Agent-wide request state.
func (a *Agent) SkillRegistry() (*SkillRegistry, error) {
    if a == nil || a.disableSkills {
        return NewSkillRegistry(), nil
    }
    definitions, err := LoadSkillDefinitions(a.skillsDir)
    if err != nil {
        return nil, err
    }
    registry := NewSkillRegistry()
    for _, skill := range definitions {
        if err := registry.Register(skill); err != nil {
            return nil, err
        }
    }
    return registry, nil
}

// RouteSkills deterministically selects the skills relevant to a request and
// resolves their transitive dependencies.
func (a *Agent) RouteSkills(input string, limit int) (SkillRouting, error) {
    registry, err := a.SkillRegistry()
    if err != nil {
        return SkillRouting{}, err
    }
    matches := registry.Match(input, limit)
    names := make([]string, 0, len(matches))
    for _, match := range matches {
        names = append(names, match.Skill.Name)
    }
    skills, err := registry.ResolveDependencies(names)
    if err != nil {
        return SkillRouting{}, err
    }
    routing := SkillRouting{Matches: matches, Skills: skills, Tools: make(map[string]struct{})}
    for _, skill := range skills {
        for _, name := range skill.Tools {
            routing.Tools[strings.ToLower(strings.TrimSpace(name))] = struct{}{}
        }
    }
    return routing, nil
}

// ActiveToolSpecs restricts tool discovery to the tools declared by the
// selected skills. If no selected skill declares tools, all tools remain
// available, preserving backwards compatibility for general requests.
func (a *Agent) ActiveToolSpecs(routing SkillRouting) []tools.Tool {
    all := a.ToolSpecs()
    if len(routing.Tools) == 0 {
        return all
    }
    filtered := make([]tools.Tool, 0, len(routing.Tools))
    for _, spec := range all {
        if _, ok := routing.Tools[strings.ToLower(strings.TrimSpace(spec.Name))]; ok {
            filtered = append(filtered, spec)
        }
    }
    return filtered
}

// SkillPrompt renders only the selected skills. Dependencies are included
// before dependents, making the prompt deterministic and composable.
func SkillPrompt(routing SkillRouting) string {
    if len(routing.Skills) == 0 {
        return ""
    }
    var b strings.Builder
    b.WriteString("Active project skills:\n")
    b.WriteString("Only the following skills are active for this request. Follow their instructions.\n")
    for _, skill := range routing.Skills {
        b.WriteString("\n### Skill: ")
        b.WriteString(skill.Name)
        if skill.Version != "" {
            b.WriteString(" (v")
            b.WriteString(skill.Version)
            b.WriteString(")")
        }
        b.WriteString("\n")
        if skill.Description != "" {
            b.WriteString("Description: ")
            b.WriteString(skill.Description)
            b.WriteString("\n")
        }
        b.WriteString(skill.Instructions)
        b.WriteString("\n")
    }
    return strings.TrimSpace(b.String())
}

// GenerateWithSkillRouting is the request-scoped Skill System 2.0 entrypoint.
// It routes skills before execution and exposes only their declared tools to
// the tool loop. The legacy Generate API remains unchanged for compatibility.
func (a *Agent) GenerateWithSkillRouting(ctx context.Context, sessionID, userInput string) (any, error) {
    routing, err := a.RouteSkills(userInput, 3)
    if err != nil {
        return nil, fmt.Errorf("route skills: %w", err)
    }
    return a.generateWithRouting(ctx, sessionID, userInput, routing)
}

// ActiveSkillNames returns the selected skills without exposing mutable state.
func (r SkillRouting) ActiveSkillNames() []string {
    names := make([]string, 0, len(r.Skills))
    for _, skill := range r.Skills {
        names = append(names, skill.Name)
    }
    return names
}

// WithSkillRouting attaches immutable routing information to a context for
// integrations that already own an execution pipeline.
type skillRoutingContextKey struct{}

func WithSkillRouting(ctx context.Context, routing SkillRouting) context.Context {
    return context.WithValue(ctx, skillRoutingContextKey{}, routing)
}

func skillRoutingFromContext(ctx context.Context) (SkillRouting, bool) {
    routing, ok := ctx.Value(skillRoutingContextKey{}).(SkillRouting)
    return routing, ok
}
