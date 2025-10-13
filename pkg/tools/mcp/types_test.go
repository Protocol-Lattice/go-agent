package mcp

import (
	"context"
	"errors"
	"strings"
	"testing"
)

type stubClient struct {
	name        string
	toolDefs    []ToolDefinition
	toolErr     error
	callFunc    func(context.Context, string, map[string]any) (ToolResponse, error)
	resources   []Resource
	listErr     error
	readContent ResourceContent
	readErr     error
	closed      bool
}

func (s *stubClient) Name() string {
	if s.name == "" {
		return "Stub"
	}
	return s.name
}

func (s *stubClient) ListTools(context.Context) ([]ToolDefinition, error) {
	if s.toolErr != nil {
		return nil, s.toolErr
	}
	return append([]ToolDefinition(nil), s.toolDefs...), nil
}

func (s *stubClient) CallTool(ctx context.Context, name string, args map[string]any) (ToolResponse, error) {
	if s.callFunc != nil {
		return s.callFunc(ctx, name, args)
	}
	return ToolResponse{}, nil
}

func (s *stubClient) ListResources(context.Context) ([]Resource, error) {
	if s.listErr != nil {
		return nil, s.listErr
	}
	return append([]Resource(nil), s.resources...), nil
}

func (s *stubClient) ReadResource(context.Context, string) (ResourceContent, error) {
	if s.readErr != nil {
		return ResourceContent{}, s.readErr
	}
	return s.readContent, nil
}

func (s *stubClient) Close(context.Context) error {
	s.closed = true
	return nil
}

func TestToolFactoryBuildsTools(t *testing.T) {
	client := &stubClient{
		name: "Calendar Server",
		toolDefs: []ToolDefinition{{
			Name:        "list/events",
			Description: "Return matching events",
		}},
	}
	factory := NewToolFactory(client)

	tools, err := factory.Tools(context.Background())
	if err != nil {
		t.Fatalf("factory.Tools returned error: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(tools))
	}

	if name := tools[0].Name(); name != "mcp:calendar-server:list-events" {
		t.Fatalf("unexpected tool name: %s", name)
	}
	if desc := tools[0].Description(); !strings.Contains(desc, "Return matching events") {
		t.Fatalf("description did not include MCP summary: %s", desc)
	}

	// Ensure slice copies are returned.
	tools[0] = nil
	again, err := factory.Tools(context.Background())
	if err != nil {
		t.Fatalf("factory.Tools second call returned error: %v", err)
	}
	if len(again) != 1 || again[0] == nil {
		t.Fatalf("expected cached tool on second invocation")
	}
}

func TestToolRunParsesArguments(t *testing.T) {
	var captured map[string]any
	client := &stubClient{
		name: "Helpers",
		toolDefs: []ToolDefinition{{
			Name:        "summarise",
			Description: "Summarise input",
		}},
		callFunc: func(_ context.Context, name string, args map[string]any) (ToolResponse, error) {
			captured = args
			if name != "summarise" {
				t.Fatalf("unexpected tool invoked: %s", name)
			}
			return ToolResponse{Content: []Content{{Type: "text", Text: "summary"}}}, nil
		},
	}
	tools, err := NewToolFactory(client).Tools(context.Background())
	if err != nil {
		t.Fatalf("failed to build tools: %v", err)
	}
	if len(tools) != 1 {
		t.Fatalf("expected one tool")
	}

	result, err := tools[0].Run(context.Background(), "tell me a story")
	if err != nil {
		t.Fatalf("tool run returned error: %v", err)
	}
	if result != "summary" {
		t.Fatalf("unexpected result: %s", result)
	}
	if captured["input"] != "tell me a story" {
		t.Fatalf("expected fallback arguments, got %v", captured)
	}

	// JSON input should be parsed as structured payload.
	_, err = tools[0].Run(context.Background(), `{"topic":"go","depth":2}`)
	if err != nil {
		t.Fatalf("tool run with JSON input failed: %v", err)
	}
	if val, ok := captured["topic"].(string); !ok || val != "go" {
		t.Fatalf("expected structured arguments, got %v", captured)
	}
}

func TestResourceToolBehaviour(t *testing.T) {
	client := &stubClient{
		name: "Docs",
		resources: []Resource{{
			URI:         "doc://welcome",
			Name:        "Welcome",
			Description: "Introduction",
		}},
		readContent: ResourceContent{
			URI:      "doc://welcome",
			MimeType: "text/markdown",
			Text:     "# Welcome",
		},
	}
	tool := NewResourceTool(client)
	if tool == nil {
		t.Fatalf("expected resource tool to be created")
	}

	list, err := tool.Run(context.Background(), "")
	if err != nil {
		t.Fatalf("listing resources failed: %v", err)
	}
	if !strings.Contains(list, "doc://welcome") {
		t.Fatalf("expected resource URI in list output: %s", list)
	}

	content, err := tool.Run(context.Background(), `{"uri":"doc://welcome"}`)
	if err != nil {
		t.Fatalf("reading resource failed: %v", err)
	}
	if !strings.Contains(content, "# Welcome") {
		t.Fatalf("expected resource content in output: %s", content)
	}

	client.listErr = ErrUnsupported
	if _, err := tool.Run(context.Background(), ""); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("expected unsupported error when listing, got %v", err)
	}

	client.listErr = nil
	client.readErr = ErrUnsupported
	if _, err := tool.Run(context.Background(), "doc://welcome"); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("expected unsupported error when fetching, got %v", err)
	}
}
