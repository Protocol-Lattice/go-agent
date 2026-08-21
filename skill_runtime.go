package agent

import (
	"context"
	"fmt"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/models"
	utcp "github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

type SkillRouting struct {
	Matches []SkillMatch
	Skills  []SkillDefinition
	Tools   map[string]struct{}
}

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
			name = strings.ToLower(strings.TrimSpace(name))
			if name != "" {
				routing.Tools[name] = struct{}{}
			}
		}
	}
	return routing, nil
}

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

func SkillPrompt(routing SkillRouting) string {
	if len(routing.Skills) == 0 {
		return ""
	}
	var b strings.Builder
	b.WriteString("Active project skills:\nOnly the following skills are active for this request. Follow their instructions.\n")
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
		b.WriteString("Instructions:\n")
		b.WriteString(skill.Instructions)
		b.WriteString("\n")
	}
	return strings.TrimSpace(b.String())
}

// GenerateWithSkillRouting runs one request with an isolated skill/tool scope.
// It is the safe entry point for hosts such as the Web UI that want skill
// routing without changing the semantics of the lower-level Generate method.
func (a *Agent) GenerateWithSkillRouting(ctx context.Context, sessionID, userInput string) (any, error) {
	routing, err := a.RouteSkills(userInput, 3)
	if err != nil {
		return nil, fmt.Errorf("route skills: %w", err)
	}
	return a.generateWithRouting(ctx, sessionID, userInput, routing)
}

// GenerateWithSkillRoutingWithFiles is the file-aware equivalent used by
// hosts that attach workspace files to a skill-routed request.
func (a *Agent) GenerateWithSkillRoutingWithFiles(ctx context.Context, sessionID, userInput string, files []models.File) (string, error) {
	routing, err := a.RouteSkills(userInput, 3)
	if err != nil {
		return "", fmt.Errorf("route skills: %w", err)
	}
	result, err := a.generateWithRoutingFiles(ctx, sessionID, userInput, routing, files)
	if err != nil {
		return "", err
	}
	return fmt.Sprint(result), nil
}

func (a *Agent) generateWithRouting(ctx context.Context, sessionID, userInput string, routing SkillRouting) (any, error) {
	requestAgent, err := a.newSkillScopedAgent(routing)
	if err != nil {
		return nil, err
	}
	return requestAgent.Generate(ctx, sessionID, userInput)
}

func (a *Agent) generateWithRoutingFiles(ctx context.Context, sessionID, userInput string, routing SkillRouting, files []models.File) (any, error) {
	requestAgent, err := a.newSkillScopedAgent(routing)
	if err != nil {
		return nil, err
	}
	return requestAgent.GenerateWithFiles(ctx, sessionID, userInput, files)
}

func (a *Agent) newSkillScopedAgent(routing SkillRouting) (*Agent, error) {
	activeTools := make([]Tool, 0)
	if len(routing.Tools) > 0 {
		for _, tool := range a.Tools() {
			if _, ok := routing.Tools[strings.ToLower(strings.TrimSpace(tool.Spec().Name))]; ok {
				activeTools = append(activeTools, tool)
			}
		}
	}

	toolCatalog := NewStaticToolCatalog(activeTools)
	requestUTCP := a.UTCPClient
	if len(routing.Tools) > 0 && requestUTCP != nil {
		requestUTCP = &skillFilteredUTCPClient{
			UtcpClientInterface: a.UTCPClient,
			inner:               a.UTCPClient,
			allowed:             routing.Tools,
		}
	}

	var codeMode *codemode.CodeModeUTCP
	if a.CodeMode != nil && (len(routing.Tools) == 0 || hasAllowedTool(routing.Tools, codemode.CodeModeToolName, "codemode.run_code")) {
		if requestUTCP != nil {
			codeMode = codemode.NewCodeModeUTCP(requestUTCP, a.model)
		}
	}

	prompt := strings.TrimSpace(a.systemPrompt)
	if active := SkillPrompt(routing); active != "" {
		if prompt != "" {
			prompt += "\n\n"
		}
		prompt += active
	}

	return New(Options{
		Model:             a.model,
		Memory:            a.memory,
		SystemPrompt:      prompt,
		ContextLimit:      a.contextLimit,
		SkillsDir:         a.skillsDir,
		DisableSkills:     true,
		ToolCatalog:       toolCatalog,
		SubAgentDirectory: a.subAgentDirectory,
		UTCPClient:        requestUTCP,
		CodeMode:          codeMode,
		Shared:            a.Shared,
		AllowUnsafeTools:  a.AllowUnsafeTools,
		Guardrails:        a.Guardrails,
		InputGuardrails:   a.InputGuardrails,
	})
}

func hasAllowedTool(allowed map[string]struct{}, names ...string) bool {
	for _, name := range names {
		if _, ok := allowed[strings.ToLower(strings.TrimSpace(name))]; ok {
			return true
		}
	}
	return false
}

type skillFilteredUTCPClient struct {
	utcp.UtcpClientInterface
	inner   utcp.UtcpClientInterface
	allowed map[string]struct{}
}

func (c *skillFilteredUTCPClient) allowedTool(name string) bool {
	name = strings.ToLower(strings.TrimSpace(name))
	if _, ok := c.allowed[name]; ok {
		return true
	}
	for candidate := range c.allowed {
		if strings.HasSuffix(name, "."+candidate) || strings.HasSuffix(candidate, "."+name) {
			return true
		}
	}
	return false
}

func (c *skillFilteredUTCPClient) SearchTools(query string, limit int) ([]tools.Tool, error) {
	specs, err := c.inner.SearchTools(query, limit)
	if err != nil {
		return nil, err
	}
	filtered := make([]tools.Tool, 0, len(specs))
	for _, spec := range specs {
		if c.allowedTool(spec.Name) {
			filtered = append(filtered, spec)
		}
	}
	return filtered, nil
}

func (c *skillFilteredUTCPClient) CallTool(ctx context.Context, name string, args map[string]any) (any, error) {
	if !c.allowedTool(name) {
		return nil, fmt.Errorf("skill tool access denied: %s", name)
	}
	return c.inner.CallTool(ctx, name, args)
}

func (r SkillRouting) ActiveSkillNames() []string {
	names := make([]string, 0, len(r.Skills))
	for _, skill := range r.Skills {
		names = append(names, skill.Name)
	}
	return names
}

type skillRoutingContextKey struct{}

func WithSkillRouting(ctx context.Context, routing SkillRouting) context.Context {
	return context.WithValue(ctx, skillRoutingContextKey{}, routing)
}

func skillRoutingFromContext(ctx context.Context) (SkillRouting, bool) {
	routing, ok := ctx.Value(skillRoutingContextKey{}).(SkillRouting)
	return routing, ok
}
