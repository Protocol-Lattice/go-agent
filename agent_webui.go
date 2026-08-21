package agent

// WebUISkills returns the skills currently loaded by the Agent as
// SkillDefinition values suitable for the Web UI. It uses the same registry
// and validation path as runtime skill routing so the UI cannot advertise a
// different skill set from the execution path.
func (a *Agent) WebUISkills() []SkillDefinition {
	if a == nil {
		return nil
	}

	registry, err := a.SkillRegistry()
	if err == nil {
		return registry.List()
	}

	// Keep the Web UI useful when a malformed skill is present: expose the
	// legacy documents rather than hiding every skill behind one bad definition.
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
