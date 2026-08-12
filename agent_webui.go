package agent

// WebUISkills returns the currently loaded Skill System 2.0 definitions.
func (a *Agent) WebUISkills() []SkillDefinition {
	if a == nil {
		return nil
	}
	a.skillMu.RLock()
	defer a.skillMu.RUnlock()

	out := make([]SkillDefinition, 0, len(a.skills))
	for _, definition := range a.skills {
		// SkillDefinition embeds Skill, so copy the definition itself first.
		copyDefinition := definition
		copyDefinition.Tags = append([]string(nil), definition.Tags...)
		copyDefinition.Triggers = append([]string(nil), definition.Triggers...)
		copyDefinition.Dependencies = append([]string(nil), definition.Dependencies...)
		copyDefinition.Tools = append([]string(nil), definition.Tools...)
		// Evaluators are runtime implementations and must not cross the API boundary.
		copyDefinition.Evaluator = nil
		out = append(out, copyDefinition)
	}
	return out
}

// WebUITools returns tool metadata currently visible to the agent.
// Invocation remains inside the agent runtime.
func (a *Agent) WebUITools() []ToolSpec {
	if a == nil || a.toolCatalog == nil {
		return nil
	}
	return append([]ToolSpec(nil), a.toolCatalog.Specs()...)
}
