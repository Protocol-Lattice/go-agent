package runtime

import (
	"context"
	"strings"
	"testing"

	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	mcptools "github.com/Raezil/go-agent-development-kit/pkg/tools/mcp"
)

type fakeMCPClient struct {
	name      string
	closed    bool
	callCount int
}

func (f *fakeMCPClient) Name() string { return f.name }

func (f *fakeMCPClient) ListTools(context.Context) ([]mcptools.ToolDefinition, error) {
	return []mcptools.ToolDefinition{{
		Name:        "ping",
		Description: "Responds with pong",
	}}, nil
}

func (f *fakeMCPClient) CallTool(ctx context.Context, name string, args map[string]any) (mcptools.ToolResponse, error) {
	f.callCount++
	if name != "ping" {
		return mcptools.ToolResponse{}, nil
	}
	payload := "pong"
	if v, ok := args["input"].(string); ok && v != "" {
		payload = "pong: " + v
	}
	return mcptools.ToolResponse{Content: []mcptools.Content{{Type: "text", Text: payload}}}, nil
}

func (f *fakeMCPClient) ListResources(context.Context) ([]mcptools.Resource, error) {
	return nil, mcptools.ErrUnsupported
}

func (f *fakeMCPClient) ReadResource(context.Context, string) (mcptools.ResourceContent, error) {
	return mcptools.ResourceContent{}, mcptools.ErrUnsupported
}

func (f *fakeMCPClient) Close(context.Context) error {
	f.closed = true
	return nil
}

func TestRuntimeIntegratesMCPTools(t *testing.T) {
	ctx := context.Background()
	fake := &fakeMCPClient{name: "Utility"}

	cfg := Config{
		CoordinatorModel: func(ctx context.Context) (models.Agent, error) {
			return models.NewDummyLLM("Coordinator:"), nil
		},
		MemoryFactory: func(ctx context.Context, _ string) (*memory.MemoryBank, error) {
			return &memory.MemoryBank{}, nil
		},
		SessionMemoryBuilder: func(bank *memory.MemoryBank, window int) *memory.SessionMemory {
			return memory.NewSessionMemory(bank, window)
		},
		Tools: []agent.Tool{&agentTestTool{}},
		MCPClients: []mcptools.ClientFactory{
			func(context.Context) (mcptools.Client, error) { return fake, nil },
		},
	}

	rt, err := New(ctx, cfg)
	if err != nil {
		t.Fatalf("runtime.New returned error: %v", err)
	}
	defer rt.Close()

	toolNames := make([]string, 0)
	for _, tool := range rt.Tools() {
		toolNames = append(toolNames, tool.Name())
	}
	found := false
	for _, name := range toolNames {
		if strings.Contains(name, "mcp:utility:ping") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("expected MCP tool to be registered, got %v", toolNames)
	}

	var mcpTool agent.Tool
	for _, tool := range rt.Tools() {
		if tool.Name() == "mcp:utility:ping" {
			mcpTool = tool
			break
		}
	}
	if mcpTool == nil {
		t.Fatalf("expected to resolve MCP tool by name")
	}

	result, err := mcpTool.Run(ctx, "hello")
	if err != nil {
		t.Fatalf("mcp tool run returned error: %v", err)
	}
	if !strings.Contains(result, "pong: hello") {
		t.Fatalf("expected MCP tool response, got %s", result)
	}

	if fake.callCount == 0 {
		t.Fatalf("expected MCP tool to be invoked")
	}

	if err := rt.Close(); err != nil {
		t.Fatalf("runtime.Close returned error: %v", err)
	}
	if !fake.closed {
		t.Fatalf("expected MCP client to be closed")
	}
}

type agentTestTool struct{}

func (t *agentTestTool) Name() string        { return "dummy" }
func (t *agentTestTool) Description() string { return "noop" }
func (t *agentTestTool) Run(context.Context, string) (string, error) {
	return "noop", nil
}
