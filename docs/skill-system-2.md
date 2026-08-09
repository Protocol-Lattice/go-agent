# Skill System 2.0

Skill System 2.0 keeps the existing `.skills` Markdown format while adding a typed registry, deterministic routing, dependencies, tool hints, and evaluation hooks.

## Definition

```go
skill := agent.SkillDefinition{
    Skill: agent.Skill{
        Name: "github-review",
        Description: "Review GitHub changes",
        Instructions: "Review correctness, security, and maintainability.",
    },
    Version: "2",
    Tags: []string{"github", "code-review"},
    Triggers: []string{"review PR", "review pull request"},
    Dependencies: []string{"git-basics"},
    Tools: []string{"github.get_diff", "github.get_pull_request"},
    Enabled: true,
}
```

## Registry

```go
registry := agent.NewSkillRegistry()
_ = registry.Register(skill)

matches := registry.Match("review this pull request", 3)
resolved, err := registry.ResolveDependencies([]string{"github-review"})
```

Routing is deterministic and does not require an LLM: exact skill names score highest, followed by triggers, tags, and description matches.

## Dependencies

Dependencies are resolved transitively in dependency-first order. Cycles and missing dependencies are rejected with explicit errors.

## Evaluation

Skills can attach an evaluator implementing `SkillEvaluator`. This allows Agent Arena to evaluate skill quality independently from the execution runner.

## Compatibility

Existing `.skills/<name>/SKILL.md` files continue to work. `LoadSkillDefinitions` upgrades them to version 2 definitions. `SaveSkill` persists a definition back into the human-editable Markdown format.
