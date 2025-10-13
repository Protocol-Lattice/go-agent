package tools

import (
	"context"
	"fmt"
	"strings"

	"github.com/Raezil/go-agent-development-kit/pkg/mcp"
)

// MCPInvoker defines the subset of the MCP client used by the tool wrapper.
type MCPInvoker interface {
	CallTool(ctx context.Context, name string, arguments map[string]any) (mcp.CallResult, error)
}

// MCPTool adapts an MCP server tool to the agent.Tool interface.
type MCPTool struct {
	client     MCPInvoker
	remoteName string
	name       string
	desc       string

	buildArgs func(string) map[string]any
}

// MCPToolOption customises the behaviour of the MCP tool wrapper.
type MCPToolOption func(*MCPTool)

// WithMCPDisplayName overrides the name reported to the coordinator. By
// default the remote tool name is used.
func WithMCPDisplayName(name string) MCPToolOption {
	return func(t *MCPTool) {
		if strings.TrimSpace(name) != "" {
			t.name = name
		}
	}
}

// WithMCPArgumentBuilder overrides how input strings are converted into the
// arguments payload sent to the MCP server. The default behaviour forwards the
// string using the key "input".
func WithMCPArgumentBuilder(fn func(string) map[string]any) MCPToolOption {
	return func(t *MCPTool) {
		if fn != nil {
			t.buildArgs = fn
		}
	}
}

// NewMCPTool constructs a tool wrapper for the provided MCP tool definition.
func NewMCPTool(client MCPInvoker, def mcp.ToolDefinition, opts ...MCPToolOption) *MCPTool {
	tool := &MCPTool{
		client:     client,
		remoteName: def.Name,
		name:       def.Name,
		desc:       def.Description,
		buildArgs: func(input string) map[string]any {
			return map[string]any{"input": strings.TrimSpace(input)}
		},
	}

	for _, opt := range opts {
		if opt != nil {
			opt(tool)
		}
	}

	return tool
}

// Name implements agent.Tool.
func (t *MCPTool) Name() string {
	if t == nil {
		return ""
	}
	return t.name
}

// Description implements agent.Tool.
func (t *MCPTool) Description() string {
	if t == nil {
		return ""
	}
	return t.desc
}

// Run invokes the remote MCP tool and returns the concatenated textual
// response. JSON payloads fall back to their string representation when no text
// content is present.
func (t *MCPTool) Run(ctx context.Context, input string) (string, error) {
	if t == nil || t.client == nil {
		return "", fmt.Errorf("mcp tool is not initialised")
	}

	args := t.buildArgs(input)
	result, err := t.client.CallTool(ctx, t.remoteName, args)
	if err != nil {
		return "", err
	}

	output := strings.TrimSpace(result.Text())
	if output == "" {
		output = strings.TrimSpace(result.JSON())
	}
	return output, nil
}
