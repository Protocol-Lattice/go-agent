package main

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	mcptools "github.com/Raezil/go-agent-development-kit/pkg/tools/mcp"
	mcpclient "github.com/mark3labs/mcp-go/pkg/client"
)

// loadMCPServer launches an MCP server command and adapts it to the runtime client interface.
func loadMCPServer(ctx context.Context, command string) (mcptools.Client, error) {
	parts, err := splitCommandLine(command)
	if err != nil {
		return nil, fmt.Errorf("parse MCP command: %w", err)
	}
	if len(parts) == 0 {
		return nil, errors.New("mcp command cannot be empty")
	}

	session, err := mcpclient.Start(ctx, mcpclient.Options{
		Command: parts[0],
		Args:    parts[1:],
		Env:     os.Environ(),
	})
	if err != nil {
		return nil, fmt.Errorf("start MCP server: %w", err)
	}
	return &mcpSessionAdapter{session: session}, nil
}

type mcpSessionAdapter struct {
	session *mcpclient.Session
}

func (a *mcpSessionAdapter) Name() string {
	if a.session == nil {
		return ""
	}
	return a.session.Name()
}

func (a *mcpSessionAdapter) ListTools(ctx context.Context) ([]mcptools.ToolDefinition, error) {
	tools, err := a.session.ListTools(ctx)
	if err != nil {
		return nil, err
	}
	defs := make([]mcptools.ToolDefinition, 0, len(tools))
	for _, tool := range tools {
		defs = append(defs, mcptools.ToolDefinition{
			Name:        tool.Name,
			Description: tool.Description,
			InputSchema: tool.InputSchema,
		})
	}
	return defs, nil
}

func (a *mcpSessionAdapter) CallTool(ctx context.Context, name string, arguments map[string]any) (mcptools.ToolResponse, error) {
	content, err := a.session.CallTool(ctx, name, arguments)
	if err != nil {
		return mcptools.ToolResponse{}, err
	}
	items := make([]mcptools.Content, 0, len(content))
	for _, item := range content {
		items = append(items, mcptools.Content{
			Type:     item.Type,
			Text:     item.Text,
			URI:      item.URI,
			MimeType: item.MimeType,
		})
	}
	return mcptools.ToolResponse{Content: items}, nil
}

func (a *mcpSessionAdapter) ListResources(ctx context.Context) ([]mcptools.Resource, error) {
	resources, err := a.session.ListResources(ctx)
	if errors.Is(err, mcpclient.ErrMethodNotFound) {
		return nil, mcptools.ErrUnsupported
	}
	if err != nil {
		return nil, err
	}
	out := make([]mcptools.Resource, 0, len(resources))
	for _, res := range resources {
		out = append(out, mcptools.Resource{
			URI:         res.URI,
			Name:        res.Name,
			Description: res.Description,
		})
	}
	return out, nil
}

func (a *mcpSessionAdapter) ReadResource(ctx context.Context, uri string) (mcptools.ResourceContent, error) {
	data, err := a.session.ReadResource(ctx, uri)
	if errors.Is(err, mcpclient.ErrMethodNotFound) {
		return mcptools.ResourceContent{}, mcptools.ErrUnsupported
	}
	if err != nil {
		return mcptools.ResourceContent{}, err
	}
	return mcptools.ResourceContent{URI: data.URI, MimeType: data.MimeType, Text: data.Text}, nil
}

func (a *mcpSessionAdapter) Close(ctx context.Context) error {
	if a.session == nil {
		return nil
	}
	return a.session.Close(ctx)
}

func splitCommandLine(input string) ([]string, error) {
	var (
		args    []string
		current strings.Builder
		quote   rune
		escape  bool
	)
	for _, r := range input {
		switch {
		case escape:
			current.WriteRune(r)
			escape = false
		case r == '\\':
			escape = true
		case quote != 0:
			if r == quote {
				quote = 0
				continue
			}
			current.WriteRune(r)
		case r == '\'' || r == '"':
			quote = r
		case r == ' ' || r == '\t' || r == '\n':
			if current.Len() > 0 {
				args = append(args, current.String())
				current.Reset()
			}
		default:
			current.WriteRune(r)
		}
	}
	if escape {
		return nil, errors.New("unterminated escape sequence in MCP command")
	}
	if quote != 0 {
		return nil, errors.New("unterminated quote in MCP command")
	}
	if current.Len() > 0 {
		args = append(args, current.String())
	}
	return args, nil
}
