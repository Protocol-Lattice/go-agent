package kit

// Option configures the AgentDevelopmentKit during construction.
type Option func(*AgentDevelopmentKit) error

// WithModule registers a single module with the kit.
func WithModule(module Module) Option {
	return func(kit *AgentDevelopmentKit) error {
		return kit.RegisterModule(module)
	}
}

// WithModules registers multiple modules using a single option invocation.
func WithModules(modules ...Module) Option {
	return func(kit *AgentDevelopmentKit) error {
		for _, module := range modules {
			if err := kit.RegisterModule(module); err != nil {
				return err
			}
		}
		return nil
	}
}

// WithDefaultSystemPrompt overrides the default system prompt used by the kit.
func WithDefaultSystemPrompt(prompt string) Option {
	return func(kit *AgentDevelopmentKit) error {
		kit.SetDefaultSystemPrompt(prompt)
		return nil
	}
}

// WithDefaultContextLimit overrides the default context window size used when
// constructing agents.
func WithDefaultContextLimit(limit int) Option {
	return func(kit *AgentDevelopmentKit) error {
		kit.SetDefaultContextLimit(limit)
		return nil
	}
}

// WithAgentOptions registers default agent options that will be applied to
// every agent created via BuildAgent.
func WithAgentOptions(opts ...AgentOption) Option {
	return func(kit *AgentDevelopmentKit) error {
		for _, opt := range opts {
			kit.UseAgentOption(opt)
		}
		return nil
	}
}
