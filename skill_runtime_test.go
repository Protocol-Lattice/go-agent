package agent

import (
    "context"
    "testing"
)

func TestSkillRoutingBuildsActiveToolScope(t *testing.T) {
    r := NewSkillRegistry()
    if err := r.Register(SkillDefinition{Skill: Skill{Name: "github-review", Instructions: "Review GitHub changes"}, Version: "2", Triggers: []string{"review"}, Tools: []string{"github.get_diff", "github.create_review"}, Enabled: true}); err != nil { t.Fatal(err) }
    if err := r.Register(SkillDefinition{Skill: Skill{Name: "git", Instructions: "Git basics"}, Version: "2", Dependencies: []string{"github-review"}, Enabled: true}); err != nil { t.Fatal(err) }

    matches := r.Match("review this PR", 3)
    if len(matches) != 1 || matches[0].Skill.Name != "github-review" { t.Fatalf("unexpected matches: %+v", matches) }
    skills, err := r.ResolveDependencies([]string{"github-review"})
    if err != nil { t.Fatal(err) }
    routing := SkillRouting{Matches: matches, Skills: skills, Tools: map[string]struct{}{"github.get_diff": {}, "github.create_review": {}}
    if len(routing.ActiveSkillNames()) != 1 || routing.ActiveSkillNames()[0] != "github-review" { t.Fatal(routing.ActiveSkillNames()) }
    if got := SkillPrompt(routing); got == "" { t.Fatal("expected active skill prompt") }
    _ = context.Background()
}

func TestSkillPromptIncludesDependenciesInOrder(t *testing.T) {
    routing := SkillRouting{Skills: []SkillDefinition{
        {Skill: Skill{Name: "base", Instructions: "base instructions"}, Version: "2"},
        {Skill: Skill{Name: "review", Instructions: "review instructions"}, Version: "2"},
    }}
    prompt := SkillPrompt(routing)
    base := "### Skill: base"
    review := "### Skill: review"
    if len(prompt) == 0 || indexOf(prompt, base) < 0 || indexOf(prompt, review) < 0 || indexOf(prompt, base) > indexOf(prompt, review) { t.Fatalf("unexpected prompt: %s", prompt) }
}

func indexOf(s, sub string) int {
    for i := 0; i+len(sub) <= len(s); i++ { if s[i:i+len(sub)] == sub { return i } }
    return -1
}
