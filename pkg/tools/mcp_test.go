package tools

import (
	"context"
	"errors"
	"testing"

	"github.com/Raezil/go-agent-development-kit/pkg/mcp"
)

type fakeMCPClient struct {
	responses map[string]mcp.CallResult
	err       error
}

func (f *fakeMCPClient) CallTool(ctx context.Context, name string, args map[string]any) (mcp.CallResult, error) {
	if f.err != nil {
		return mcp.CallResult{}, f.err
	}
	res, ok := f.responses[name]
	if !ok {
		return mcp.CallResult{}, errors.New("unknown tool")
	}
	return res, nil
}

func TestMCPTool_RunText(t *testing.T) {
	client := &fakeMCPClient{
		responses: map[string]mcp.CallResult{
			"echo": {Content: []mcp.Content{{Type: "text", Text: "hello"}}},
		},
	}

	tool := NewMCPTool(client, mcp.ToolDefinition{Name: "echo", Description: "Echo"})
	out, err := tool.Run(context.Background(), "ignored")
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if out != "hello" {
		t.Fatalf("unexpected output: %s", out)
	}
}

func TestMCPTool_RunJSONFallback(t *testing.T) {
	client := &fakeMCPClient{
		responses: map[string]mcp.CallResult{
			"json": {Content: []mcp.Content{{Type: "json", Data: []byte(`{"value":42}`)}}},
		},
	}

	tool := NewMCPTool(client, mcp.ToolDefinition{Name: "json", Description: "JSON"})
	out, err := tool.Run(context.Background(), "")
	if err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if out != "{\n  \"value\": 42\n}" {
		t.Fatalf("unexpected output: %s", out)
	}
}

func TestMCPTool_ArgumentBuilder(t *testing.T) {
	var captured map[string]any
	client := &fakeMCPClient{
		responses: map[string]mcp.CallResult{
			"echo": {Content: []mcp.Content{{Type: "text", Text: "ok"}}},
		},
	}

	tool := NewMCPTool(&capturingClient{MCPInvoker: client, capture: func(args map[string]any) {
		captured = args
	}}, mcp.ToolDefinition{Name: "echo"}, WithMCPArgumentBuilder(func(input string) map[string]any {
		return map[string]any{"message": input}
	}))

	if _, err := tool.Run(context.Background(), "hi"); err != nil {
		t.Fatalf("Run error: %v", err)
	}
	if captured["message"] != "hi" {
		t.Fatalf("argument builder not applied: %#v", captured)
	}
}

type capturingClient struct {
	MCPInvoker
	capture func(map[string]any)
}

func (c *capturingClient) CallTool(ctx context.Context, name string, args map[string]any) (mcp.CallResult, error) {
	if c.capture != nil {
		c.capture(args)
	}
	return c.MCPInvoker.CallTool(ctx, name, args)
}
