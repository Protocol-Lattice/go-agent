package agent

import (
	"context"
	"strings"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	utcp "github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/base"
)

func TestSubAgentTool(t *testing.T) {
	ctx := context.Background()
	sub := &stubSubAgent{
		name:        "test_sub",
		description: "A test sub-agent",
	}

	tool := NewSubAgentTool(sub)

	// Test Spec
	spec := tool.Spec()
	if spec.Name != "test_sub" {
		t.Errorf("expected name 'test_sub', got %q", spec.Name)
	}
	if spec.Description != "A test sub-agent" {
		t.Errorf("expected description 'A test sub-agent', got %q", spec.Description)
	}
	if _, ok := spec.InputSchema["properties"]; !ok {
		t.Errorf("expected properties in input schema")
	}

	// Test Invoke
	req := ToolRequest{
		Arguments: map[string]any{
			"instruction": "do something",
		},
	}
	resp, err := tool.Invoke(ctx, req)
	if err != nil {
		t.Fatalf("Invoke returned error: %v", err)
	}
	if resp.Content != "do something" { // stubSubAgent echoes input
		t.Errorf("expected content 'do something', got %q", resp.Content)
	}

	// Test Invoke with missing argument
	reqBad := ToolRequest{
		Arguments: map[string]any{},
	}
	_, err = tool.Invoke(ctx, reqBad)
	if err == nil {
		t.Errorf("expected error for missing instruction")
	}
}

func TestAgentToolAdapter(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{response: "agent response"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	agent, err := New(Options{Model: model, Memory: mem})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	tool := NewAgentTool("agent_tool", "An agent as a tool", agent)

	// Test Spec
	spec := tool.Spec()
	if spec.Name != "agent_tool" {
		t.Errorf("expected name 'agent_tool', got %q", spec.Name)
	}

	// Test Invoke
	req := ToolRequest{
		SessionID: "main_session",
		Arguments: map[string]any{
			"instruction": "hello",
		},
	}
	resp, err := tool.Invoke(ctx, req)
	if err != nil {
		t.Fatalf("Invoke returned error: %v", err)
	}

	// The stubModel returns "agent response | hello" because it appends the prompt
	// But wait, the stubModel implementation in agent_test.go is:
	// return m.response + " | " + prompt, nil
	// So we expect "agent response | hello"

	if !strings.Contains(resp.Content, "agent response") {
		t.Errorf("expected response to contain 'agent response', got %q", resp.Content)
	}
}

func TestAgent_AsTool(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	agent, _ := New(Options{Model: model, Memory: mem})

	tool := agent.AsTool("my_agent", "desc")
	if tool.Spec().Name != "my_agent" {
		t.Errorf("expected tool name 'my_agent', got %q", tool.Spec().Name)
	}
}

func TestAgent_AsUTCPTool(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	agent, _ := New(Options{Model: model, Memory: mem})

	utcpTool := agent.AsUTCPTool("agent.utcp", "desc")
	if utcpTool.Name != "agent.utcp" {
		t.Fatalf("expected tool name agent.utcp, got %q", utcpTool.Name)
	}
	if utcpTool.Provider == nil || utcpTool.Provider.Type() != base.ProviderText {
		t.Fatalf("expected ProviderText provider, got %#v", utcpTool.Provider)
	}

	out, err := utcpTool.Handler(nil, map[string]interface{}{"instruction": "handle this"})
	if err != nil {
		t.Fatalf("unexpected handler error: %v", err)
	}
	resp, ok := out["response"].(string)
	if !ok {
		t.Fatalf("expected response string, got %#v", out["response"])
	}
	if !strings.Contains(resp, "handle this") {
		t.Fatalf("expected agent output to include instruction, got %q", resp)
	}
	if session := out["session_id"]; session == "" {
		t.Fatalf("expected session_id to be populated, got %#v", session)
	}
}

func TestAgent_AsUTCPTool_ValidatesInstruction(t *testing.T) {
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	agent, _ := New(Options{Model: model, Memory: mem})

	utcpTool := agent.AsUTCPTool("agent.utcp", "desc")
	if _, err := utcpTool.Handler(nil, map[string]interface{}{}); err == nil {
		t.Fatalf("expected error for missing instruction")
	}
}

func TestAgent_RegisterAsUTCPProvider(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	agent, _ := New(Options{Model: model, Memory: mem})

	client, err := utcp.NewUTCPClient(ctx, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to create utcp client: %v", err)
	}

	err = agent.RegisterAsUTCPProvider(ctx, client, "local.agent", "desc")
	if err != nil {
		t.Fatalf("register as utcp provider: %v", err)
	}

	out, err := client.CallTool(ctx, "local.agent", map[string]any{"instruction": "ping"})
	if err != nil {
		t.Fatalf("CallTool error: %v", err)
	}
	result, ok := out.(map[string]any)
	if !ok {
		t.Fatalf("expected map result, got %#v", out)
	}
	resp, _ := result["response"].(string)
	if !strings.Contains(resp, "ping") {
		t.Fatalf("expected response to include 'ping', got %q", resp)
	}
}

func TestAgent_RegisterAsUTCPProvider_CallToolWithCustomSession(t *testing.T) {
	ctx := context.Background()
	model := &stubModel{response: "ok"}
	mem := memory.NewSessionMemory(&memory.MemoryBank{}, 0)
	agent, _ := New(Options{Model: model, Memory: mem})

	client, err := utcp.NewUTCPClient(ctx, nil, nil, nil)
	if err != nil {
		t.Fatalf("failed to create utcp client: %v", err)
	}

	if err := agent.RegisterAsUTCPProvider(ctx, client, "local.agent", "desc"); err != nil {
		t.Fatalf("register as utcp provider: %v", err)
	}

	out, err := client.CallTool(ctx, "local.agent", map[string]any{
		"instruction": "hi",
		"session_id":  "custom-session",
	})
	if err != nil {
		t.Fatalf("CallTool error: %v", err)
	}
	result, ok := out.(map[string]any)
	if !ok {
		t.Fatalf("expected map result, got %#v", out)
	}
	if sid, _ := result["session_id"].(string); sid != "custom-session" {
		t.Fatalf("expected session_id to be 'custom-session', got %q", sid)
	}
}
