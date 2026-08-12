package agent

// WebUISkills returns the currently loaded skills in deterministic order.
// The returned slice is a copy and can be safely serialized by an HTTP UI.
func (a *Agent) WebUISkills() []SkillDefinition {
	if a == nil {
		return nil
	}
	a.skillMu.RLock()
	defer a.skillMu.RUnlock()

	out := make([]SkillDefinition, 0, len(a.skills))
	for _, skill := range a.skills {
		copySkill := skill
		copySkill.Tags = append([]string(nil), skill.Tags...)
		copySkill.Triggers = append([]string(nil), skill.Triggers...)
		copySkill.Dependencies = append([]string(nil), skill.Dependencies...)
		copySkill.Tools = append([]string(nil), skill.Tools...)
		copySkill.Evaluator = nil
		out = append(out, copySkill)
	}
	return out
}

// WebUITools returns the tool specifications currently visible to the agent.
// It intentionally exposes metadata only; invocation remains inside the agent runtime.
func (a *Agent) WebUITools() []ToolSpec {
	if a == nil || a.toolCatalog == nil {
		return nil
	}
	return append([]ToolSpec(nil), a.toolCatalog.Specs()...)
}
