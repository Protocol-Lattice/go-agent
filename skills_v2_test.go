package agent

import (
    "context"
    "testing"
)

func TestSkillRegistryMatchAndDependencies(t *testing.T) {
    r := NewSkillRegistry()
    if err := r.Register(SkillDefinition{Skill: Skill{Name: "base", Instructions: "base"}, Version: "2", Enabled: true, Tags: []string{"go"}}); err != nil { t.Fatal(err) }
    if err := r.Register(SkillDefinition{Skill: Skill{Name: "review", Instructions: "review"}, Version: "2", Enabled: true, Triggers: []string{"review"}, Dependencies: []string{"base"}}); err != nil { t.Fatal(err) }
    matches := r.Match("please review this go code", 1)
    if len(matches) != 1 || matches[0].Skill.Name != "review" { t.Fatalf("unexpected matches: %+v", matches) }
    resolved, err := r.ResolveDependencies([]string{"review"})
    if err != nil { t.Fatal(err) }
    if len(resolved) != 2 || resolved[0].Name != "base" || resolved[1].Name != "review" { t.Fatalf("unexpected dependency order: %+v", resolved) }
}

func TestSkillRegistryDetectsCycles(t *testing.T) {
    r := NewSkillRegistry()
    for _, skill := range []SkillDefinition{
        {Skill: Skill{Name: "a", Instructions: "a"}, Enabled: true, Dependencies: []string{"b"}},
        {Skill: Skill{Name: "b", Instructions: "b"}, Enabled: true, Dependencies: []string{"a"}},
    } { if err := r.Register(skill); err != nil { t.Fatal(err) } }
    if _, err := r.ResolveDependencies([]string{"a"}); err == nil { t.Fatal("expected cycle error") }
}

func TestSkillRegistryPrefersSpecificSkillNameTokens(t *testing.T) {
    r := NewSkillRegistry()
    if err := r.Register(SkillDefinition{Skill: Skill{Name: "code-review", Description: "Review Go code for correctness.", Instructions: "review"}, Enabled: true}); err != nil { t.Fatal(err) }
    if err := r.Register(SkillDefinition{Skill: Skill{Name: "refactor-readme", Description: "Refactor README documentation.", Instructions: "refactor"}, Enabled: true}); err != nil { t.Fatal(err) }

    matches := r.Match("Use the README refactoring skill, inspect README.md, then modify it", 1)
    if len(matches) != 1 || matches[0].Skill.Name != "refactor-readme" {
        t.Fatalf("expected refactor-readme, got %+v", matches)
    }
    if matches[0].Score < .98 { t.Fatalf("expected strong skill-name match, got %.2f", matches[0].Score) }
}

type testSkillEvaluator struct{}
func (testSkillEvaluator) Evaluate(context.Context, SkillEvaluation) (SkillEvaluationResult, error) { return SkillEvaluationResult{Score: 1, Passed: true}, nil }

func TestSkillEvaluatorContract(t *testing.T) {
    var evaluator SkillEvaluator = testSkillEvaluator{}
    result, err := evaluator.Evaluate(context.Background(), SkillEvaluation{SkillName: "test", Input: "in", Output: "out"})
    if err != nil || !result.Passed || result.Score != 1 { t.Fatalf("unexpected result: %+v, %v", result, err) }
}
