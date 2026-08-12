package agent

// WebUISkills returns the skills currently loaded by the Agent as
// SkillDefinition values suitable for the Web UI. The runtime Agent stores
// legacy Skill documents, while SkillDefinition is the richer declarative
// Skill System 2.0 type, so the legacy fields are wrapped without inventing
// v2 metadata that is not present at runtime.
func (a *Agent) WebUISkills() []SkillDefinition {
	if a == nil {
		return nil
	}

	skills := a.Skills()
	out := make([]SkillDefinition, 0, len(skills))
	for _, skill := range skills {
		out = append(out, SkillDefinition{
			Skill:   skill,
			Version: "1",
			Enabled: true,
		})
	}
	return out
}

// WebUITools returns tool metadata currently visible to the Agent.
// Invocation remains inside the agent runtime.
func (a *Agent) WebUITools() []ToolSpec {
	if a == nil || a.toolCatalog == nil {
		return nil
	}
	return a.toolCatalog.Specs()
}
