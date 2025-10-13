// Package mcp exposes helpers that adapt Model Context Protocol servers into
// go-agent-development-kit tools. Use an MCP client implementation (such as
// github.com/mark3labs/mcp-go) to satisfy the Client interface and register the
// returned tools with the runtime.
package mcp

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strings"
	"sync"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
)

// ErrUnsupported indicates that a capability is not implemented by the MCP client.
var ErrUnsupported = errors.New("mcp: capability not supported")

// Client exposes the minimal operations required from an MCP client implementation.
type Client interface {
	// Name returns the human readable identifier for the MCP server.
	Name() string
	// ListTools enumerates the tools exposed by the server.
	ListTools(ctx context.Context) ([]ToolDefinition, error)
	// CallTool invokes a tool with structured arguments.
	CallTool(ctx context.Context, name string, arguments map[string]any) (ToolResponse, error)
	// ListResources returns the static resources exposed by the server, if supported.
	ListResources(ctx context.Context) ([]Resource, error)
	// ReadResource fetches the content of a specific resource, if supported.
	ReadResource(ctx context.Context, uri string) (ResourceContent, error)
	// Close frees any resources allocated by the client.
	Close(ctx context.Context) error
}

// ToolDefinition mirrors the MCP tool description payload.
type ToolDefinition struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// ToolResponse captures a normalised tool response returned from an MCP server.
type ToolResponse struct {
	Content []Content
}

// Content represents a single item returned from a tool invocation.
type Content struct {
	Type     string
	Text     string
	URI      string
	MimeType string
}

// Resource describes a static resource advertised by an MCP server.
type Resource struct {
	URI         string
	Name        string
	Description string
}

// ResourceContent contains the payload for a static MCP resource.
type ResourceContent struct {
	URI      string
	MimeType string
	Text     string
}

// ToolFactory constructs agent tools backed by an MCP client.
type ToolFactory struct {
	client Client
	once   sync.Once
	cached []agent.Tool
	err    error
}

// ClientFactory creates an MCP client instance on demand.
type ClientFactory func(ctx context.Context) (Client, error)

// NewToolFactory initialises a ToolFactory for the supplied client.
func NewToolFactory(client Client) *ToolFactory {
	return &ToolFactory{client: client}
}

// Tools converts MCP tool definitions into agent.Tool implementations.
func (f *ToolFactory) Tools(ctx context.Context) ([]agent.Tool, error) {
	f.once.Do(func() {
		defs, err := f.client.ListTools(ctx)
		if err != nil {
			f.err = fmt.Errorf("list mcp tools: %w", err)
			return
		}

		sort.SliceStable(defs, func(i, j int) bool {
			return strings.ToLower(defs[i].Name) < strings.ToLower(defs[j].Name)
		})

		tools := make([]agent.Tool, 0, len(defs))
		for _, def := range defs {
			def := def
			if strings.TrimSpace(def.Name) == "" {
				continue
			}
			tools = append(tools, &Tool{
				client: f.client,
				definition: ToolDefinition{
					Name:        def.Name,
					Description: def.Description,
					InputSchema: def.InputSchema,
				},
			})
		}
		f.cached = tools
	})
	if f.err != nil {
		return nil, f.err
	}
	out := make([]agent.Tool, len(f.cached))
	copy(out, f.cached)
	return out, nil
}

// Tool wraps an MCP tool definition and proxies invocations to the underlying client.
type Tool struct {
	client     Client
	definition ToolDefinition
}

// Name returns the stable name of the MCP backed tool.
func (t *Tool) Name() string {
	return fmt.Sprintf("mcp:%s:%s", sanitizeName(t.client.Name()), sanitizeName(t.definition.Name))
}

// Description returns a human readable summary of the tool.
func (t *Tool) Description() string {
	desc := strings.TrimSpace(t.definition.Description)
	if desc == "" {
		desc = "Remote tool exposed by a Model Context Protocol server."
	}
	if len(t.definition.InputSchema) > 0 {
		desc = fmt.Sprintf("%s (expects JSON arguments matching the advertised schema)", desc)
	}
	return desc
}

// Run executes the MCP tool via the underlying client.
func (t *Tool) Run(ctx context.Context, input string) (string, error) {
	args, err := parseArguments(input)
	if err != nil {
		return "", err
	}

	resp, err := t.client.CallTool(ctx, t.definition.Name, args)
	if err != nil {
		return "", err
	}
	return formatResponse(resp), nil
}

// ResourceTool exposes MCP resources as an agent tool for quick inspection.
type ResourceTool struct {
	client Client
}

// NewResourceTool constructs a helper tool if the client advertises resource capabilities.
func NewResourceTool(client Client) agent.Tool {
	if client == nil {
		return nil
	}
	return &ResourceTool{client: client}
}

// Name returns the command handle for the resource helper tool.
func (r *ResourceTool) Name() string {
	return fmt.Sprintf("mcp:%s:resources", sanitizeName(r.client.Name()))
}

// Description explains how to use the resource helper tool.
func (r *ResourceTool) Description() string {
	return "List or fetch Model Context Protocol resources. Call with no arguments to list resources or provide a URI to fetch its contents."
}

// Run executes either a resource listing or fetch operation.
func (r *ResourceTool) Run(ctx context.Context, input string) (string, error) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" || strings.EqualFold(trimmed, "list") {
		resources, err := r.client.ListResources(ctx)
		if err != nil {
			if errors.Is(err, ErrUnsupported) {
				return "", fmt.Errorf("resource listing not supported by this MCP server: %w", ErrUnsupported)
			}
			return "", err
		}
		if len(resources) == 0 {
			return "No resources advertised by the MCP server.", nil
		}
		var b strings.Builder
		b.WriteString("Available resources:\n")
		for i, res := range resources {
			name := res.Name
			if name == "" {
				name = res.URI
			}
			desc := strings.TrimSpace(res.Description)
			if desc == "" {
				desc = "(no description provided)"
			}
			fmt.Fprintf(&b, "%d. %s\n   URI: %s\n   %s\n", i+1, name, res.URI, desc)
		}
		return b.String(), nil
	}

	// Attempt to parse JSON input with "uri" key for convenience.
	if strings.HasPrefix(trimmed, "{") {
		var payload map[string]any
		if err := json.Unmarshal([]byte(trimmed), &payload); err == nil {
			if uri, ok := payload["uri"].(string); ok && uri != "" {
				trimmed = uri
			}
		}
	}

	content, err := r.client.ReadResource(ctx, trimmed)
	if err != nil {
		if errors.Is(err, ErrUnsupported) {
			return "", fmt.Errorf("resource fetching not supported by this MCP server: %w", ErrUnsupported)
		}
		return "", err
	}
	mime := content.MimeType
	if mime == "" {
		mime = "text/plain"
	}
	return fmt.Sprintf("Resource %s (%s):\n%s", content.URI, mime, strings.TrimSpace(content.Text)), nil
}

func sanitizeName(name string) string {
	cleaned := strings.ToLower(strings.TrimSpace(name))
	if cleaned == "" {
		return "server"
	}
	var b strings.Builder
	for _, r := range cleaned {
		switch {
		case r >= 'a' && r <= 'z':
			b.WriteRune(r)
		case r >= '0' && r <= '9':
			b.WriteRune(r)
		case r == '-' || r == '_' || r == '.':
			b.WriteRune('-')
		case r == ' ':
			b.WriteRune('-')
		default:
			b.WriteRune('-')
		}
	}
	result := strings.Trim(b.String(), "-")
	if result == "" {
		return "server"
	}
	return result
}

func parseArguments(input string) (map[string]any, error) {
	trimmed := strings.TrimSpace(input)
	if trimmed == "" {
		return map[string]any{}, nil
	}
	if strings.HasPrefix(trimmed, "{") {
		var payload map[string]any
		if err := json.Unmarshal([]byte(trimmed), &payload); err == nil {
			return payload, nil
		}
	}
	return map[string]any{"input": trimmed}, nil
}

func formatResponse(resp ToolResponse) string {
	if len(resp.Content) == 0 {
		return "(no content returned)"
	}
	var b strings.Builder
	for i, item := range resp.Content {
		if i > 0 {
			b.WriteString("\n---\n")
		}
		switch strings.ToLower(item.Type) {
		case "text", "string", "markdown":
			text := strings.TrimSpace(item.Text)
			if text == "" {
				text = "(empty text content)"
			}
			b.WriteString(text)
		case "resource":
			uri := item.URI
			if uri == "" {
				uri = "(unknown URI)"
			}
			mime := item.MimeType
			if mime == "" {
				mime = "application/octet-stream"
			}
			fmt.Fprintf(&b, "Resource reference: %s (%s)", uri, mime)
		default:
			text := strings.TrimSpace(item.Text)
			if text == "" {
				text = "(no payload provided)"
			}
			fmt.Fprintf(&b, "%s: %s", item.Type, text)
		}
	}
	return b.String()
}
