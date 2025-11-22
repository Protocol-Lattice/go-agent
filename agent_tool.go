package agent

import (
	"context"
	"fmt"
	"strings"

	utcp "github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/base"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/cli"
	"github.com/universal-tool-calling-protocol/go-utcp/src/repository"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
	"github.com/universal-tool-calling-protocol/go-utcp/src/transports"
)

// SubAgentTool adapts a SubAgent to the Tool interface.
type SubAgentTool struct {
	subAgent SubAgent
}

// NewSubAgentTool creates a new tool that wraps a SubAgent.
func NewSubAgentTool(sa SubAgent) Tool {
	return &SubAgentTool{subAgent: sa}
}

func (t *SubAgentTool) Spec() ToolSpec {
	return ToolSpec{
		Name:        t.subAgent.Name(),
		Description: t.subAgent.Description(),
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"instruction": map[string]any{
					"type":        "string",
					"description": "The instruction or query for the sub-agent.",
				},
			},
			"required": []string{"instruction"},
		},
	}
}

func (t *SubAgentTool) Invoke(ctx context.Context, req ToolRequest) (ToolResponse, error) {
	instruction, ok := req.Arguments["instruction"].(string)
	if !ok {
		return ToolResponse{}, fmt.Errorf("missing or invalid 'instruction' argument")
	}

	result, err := t.subAgent.Run(ctx, instruction)
	if err != nil {
		return ToolResponse{}, err
	}

	return ToolResponse{
		Content: result,
	}, nil
}

// AgentToolAdapter adapts an Agent to the Tool interface.
type AgentToolAdapter struct {
	agent       *Agent
	name        string
	description string
}

type agentCLITransport struct {
	inner repository.ClientTransport
	tools map[string][]tools.Tool
}

func (t *agentCLITransport) RegisterToolProvider(ctx context.Context, prov base.Provider) ([]tools.Tool, error) {
	p, ok := prov.(*cli.CliProvider)
	if !ok {
		if t.inner != nil {
			return t.inner.RegisterToolProvider(ctx, prov)
		}
		return nil, fmt.Errorf("unsupported provider type %T", prov)
	}
	if t.tools == nil {
		t.tools = make(map[string][]tools.Tool)
	}
	list, ok := t.tools[p.Name]
	if !ok {
		if t.inner != nil {
			return t.inner.RegisterToolProvider(ctx, prov)
		}
		return nil, fmt.Errorf("agent tools not found for provider %s", p.Name)
	}
	return list, nil
}

func (t *agentCLITransport) DeregisterToolProvider(ctx context.Context, prov base.Provider) error {
	if p, ok := prov.(*cli.CliProvider); ok {
		if _, ok := t.tools[p.Name]; ok {
			delete(t.tools, p.Name)
			return nil
		}
	}
	if t.inner != nil {
		return t.inner.DeregisterToolProvider(ctx, prov)
	}
	return nil
}

func (t *agentCLITransport) CallTool(ctx context.Context, toolName string, args map[string]any, prov base.Provider, _ *string) (any, error) {
	if p, ok := prov.(*cli.CliProvider); ok {
		if list, ok := t.tools[p.Name]; ok {
			for _, tool := range list {
				if tool.Name == toolName || strings.HasSuffix(tool.Name, "."+toolName) {
					if tool.Handler == nil {
						return nil, fmt.Errorf("tool %s has no handler", toolName)
					}
					return tool.Handler(ctx, args)
				}
			}
		}
		if t.inner != nil {
			return t.inner.CallTool(ctx, toolName, args, prov, nil)
		}
		return nil, fmt.Errorf("tool %s not found for provider %s", toolName, p.Name)
	}
	if t.inner != nil {
		return t.inner.CallTool(ctx, toolName, args, prov, nil)
	}
	return nil, fmt.Errorf("unsupported provider type %T", prov)
}

func (t *agentCLITransport) CallToolStream(ctx context.Context, toolName string, args map[string]any, prov base.Provider) (transports.StreamResult, error) {
	if p, ok := prov.(*cli.CliProvider); ok {
		if _, ok := t.tools[p.Name]; ok {
			return nil, fmt.Errorf("streaming not supported for tool %s (provider %s)", toolName, p.Name)
		}
	}
	if t.inner != nil {
		return t.inner.CallToolStream(ctx, toolName, args, prov)
	}
	return nil, fmt.Errorf("unsupported provider type %T", prov)
}

// NewAgentTool creates a new tool that wraps an Agent.
func NewAgentTool(name, description string, agent *Agent) Tool {
	return &AgentToolAdapter{
		agent:       agent,
		name:        name,
		description: description,
	}
}

func (t *AgentToolAdapter) Spec() ToolSpec {
	return ToolSpec{
		Name:        t.name,
		Description: t.description,
		InputSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"instruction": map[string]any{
					"type":        "string",
					"description": "The instruction or query for the sub-agent.",
				},
			},
			"required": []string{"instruction"},
		},
	}
}

func (t *AgentToolAdapter) Invoke(ctx context.Context, req ToolRequest) (ToolResponse, error) {
	instruction, ok := req.Arguments["instruction"].(string)
	if !ok {
		return ToolResponse{}, fmt.Errorf("missing or invalid 'instruction' argument")
	}

	// Create a sub-session ID to keep context separate but related
	subSessionID := fmt.Sprintf("%s.sub.%s", req.SessionID, t.name)

	result, err := t.agent.Generate(ctx, subSessionID, instruction)
	if err != nil {
		return ToolResponse{}, err
	}

	return ToolResponse{
		Content: fmt.Sprint(result),
	}, nil
}

// AsTool returns a Tool representation of the Agent.
func (a *Agent) AsTool(name, description string) Tool {
	return NewAgentTool(name, description, a)
}

// AsUTCPTool exposes the agent as a UTCP tool with an in-process handler.
// The tool accepts:
// - instruction (required): user query for the agent
// - session_id (optional): custom session id; defaults to a namespaced value derived from the tool name
func (a *Agent) AsUTCPTool(name, description string) tools.Tool {
	providerName := strings.TrimSpace(name)
	if parts := strings.Split(name, "."); len(parts) > 1 {
		providerName = parts[0]
	}
	defaultSession := fmt.Sprintf("%s.session", providerName)
	return tools.Tool{
		Name:        name,
		Description: description,
		Provider: &base.BaseProvider{
			Name:         providerName,
			ProviderType: base.ProviderCLI, // in-process handler, no remote transport
		},
		Inputs: tools.ToolInputOutputSchema{
			Type: "object",
			Properties: map[string]any{
				"instruction": map[string]any{
					"type":        "string",
					"description": "The instruction or query for the agent.",
				},
				"session_id": map[string]any{
					"type":        "string",
					"description": "Optional session id; defaults to the provider-derived session.",
				},
			},
			Required: []string{"instruction"},
		},
		Outputs: tools.ToolInputOutputSchema{
			Type: "object",
			Properties: map[string]any{
				"response":   map[string]any{"type": "string"},
				"session_id": map[string]any{"type": "string"},
			},
		},
		Handler: tools.ToolHandler(func(ctx context.Context, inputs map[string]interface{}) (any, error) {
			rawInstruction, ok := inputs["instruction"].(string)
			if !ok || strings.TrimSpace(rawInstruction) == "" {
				return nil, fmt.Errorf("missing or invalid 'instruction'")
			}

			sessionID, _ := inputs["session_id"].(string)
			sessionID = strings.TrimSpace(sessionID)
			if sessionID == "" {
				sessionID = defaultSession
			}

			execCtx := ctx
			if execCtx == nil {
				execCtx = context.Background()
			}

			out, err := a.Generate(execCtx, sessionID, rawInstruction)
			if err != nil {
				return nil, err
			}

			return fmt.Sprint(out), nil
		}),
	}
}

// RegisterAsUTCPProvider registers the agent as a UTCP tool on the provided client.
// It installs a lightweight in-process transport under the "text" provider type
// to route CallTool invocations directly to the agent's Generate method.
func (a *Agent) RegisterAsUTCPProvider(ctx context.Context, client utcp.UtcpClientInterface, name, description string) error {
	if client == nil {
		return fmt.Errorf("utcp client is nil")
	}

	tool := a.AsUTCPTool(name, description)
	providerName := strings.TrimSpace(name)
	if parts := strings.Split(name, "."); len(parts) > 1 {
		providerName = parts[0]
	}

	tp := &cli.CliProvider{
		BaseProvider: base.BaseProvider{
			Name:         providerName,
			ProviderType: base.ProviderCLI,
		},
	}

	transportsMap := client.GetTransports()
	if transportsMap == nil {
		return fmt.Errorf("utcp client transports map is nil")
	}

	existing := transportsMap[string(base.ProviderCLI)]
	var shim *agentCLITransport
	if maybe, ok := existing.(*agentCLITransport); ok {
		shim = maybe
	} else {
		shim = &agentCLITransport{inner: existing}
		transportsMap[string(base.ProviderCLI)] = shim
	}
	if shim.tools == nil {
		shim.tools = make(map[string][]tools.Tool)
	}
	shim.tools[tp.Name] = []tools.Tool{tool}

	_, err := client.RegisterToolProvider(ctx, tp)
	return err
}
