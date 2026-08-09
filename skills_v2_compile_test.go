package agent

import "testing"

func TestSkillSystemV2Compiles(t *testing.T) {
    skill := SkillDefinition{Skill: Skill{Name: "compile", Instructions: "instructions"}, Version: "2", Enabled: true}
    registry := NewSkillRegistry()
    if err := registry.Register(skill); err != nil { t.Fatal(err) }
    if got, ok := registry.Get("compile"); !ok || got.Version != "2" { t.Fatalf("skill not registered: %+v", got) }
}
