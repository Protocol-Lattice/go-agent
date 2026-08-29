package agent

import (
	"context"
	"fmt"
	"strings"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
	utcpTools "github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

func TestGenerateRoutesRepositoryWorkThroughToolOrchestrator(t *testing.T) {
	agent, utcpClient := newToolRoutingTestAgent(t)

	out, err := agent.Generate(
		context.Background(),
		"session",
		"Inspect the repository with the available tools and report what you find.",
	)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	if utcpClient.callCount != 1 {
		t.Fatalf("UTCP call count = %d, want 1", utcpClient.callCount)
	}
	if utcpClient.lastToolName != "orchestrated.inspect" {
		t.Fatalf(
			"last UTCP tool = %q, want orchestrated.inspect; broad repository work must not be short-circuited by direct CodeMode",
			utcpClient.lastToolName,
		)
	}
	if got, want := fmt.Sprint(out), "utcp says orchestrated.inspect"; got != want {
		t.Fatalf("Generate output = %q, want %q", got, want)
	}
}

func TestLikelyNeedsToolCallDoesNotDropPoliteToolCommands(t *testing.T) {
	a := &Agent{}

	for _, input := range []string{"hello", "Hello!", "thanks", "Good morning."} {
		if a.likelyNeedsToolCall(strings.ToLower(input)) {
			t.Errorf("likelyNeedsToolCall(%q) = true, want false for standalone small talk", input)
		}
	}

	for _, input := range []string{
		"Hello, run the echo tool.",
		"Hey, inspect the repository files.",
		"Thanks, now call the deploy tool.",
	} {
		if !a.likelyNeedsToolCall(strings.ToLower(input)) {
			t.Errorf("likelyNeedsToolCall(%q) = false, want true for a tool command", input)
		}
	}
}

func TestGenerateWithFilesRoutesRepositoryWorkThroughToolOrchestrator(t *testing.T) {
	agent, utcpClient := newToolRoutingTestAgent(t)

	out, err := agent.GenerateWithFiles(
		context.Background(),
		"session",
		"Inspect the repository with the available tools and report what you find.",
		nil,
	)
	if err != nil {
		t.Fatalf("GenerateWithFiles returned error: %v", err)
	}

	if utcpClient.callCount != 1 {
		t.Fatalf("UTCP call count = %d, want 1", utcpClient.callCount)
	}
	if utcpClient.lastToolName != "orchestrated.inspect" {
		t.Fatalf(
			"last UTCP tool = %q, want orchestrated.inspect; broad repository work must not be short-circuited by direct CodeMode",
			utcpClient.lastToolName,
		)
	}
	if got, want := out, "utcp says orchestrated.inspect"; got != want {
		t.Fatalf("GenerateWithFiles output = %q, want %q", got, want)
	}
}

func TestGeneratePreservesExplicitCodeModeRouting(t *testing.T) {
	agent, utcpClient := newToolRoutingTestAgent(t)

	out, err := agent.Generate(
		context.Background(),
		"session",
		"Run code with CodeMode using the echo tool.",
	)
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}

	if utcpClient.callCount != 1 {
		t.Fatalf("UTCP call count = %d, want 1", utcpClient.callCount)
	}
	if utcpClient.lastToolName != "echo" {
		t.Fatalf("last UTCP tool = %q, want echo for an explicit CodeMode request", utcpClient.lastToolName)
	}

	result, ok := out.(codemode.CodeModeResult)
	if !ok {
		t.Fatalf("Generate output type = %T, want codemode.CodeModeResult", out)
	}
	value, ok := result.Value.(string)
	if !ok || !strings.Contains(value, "utcp says echo") {
		t.Fatalf("CodeMode result value = %T(%v), want string containing %q", result.Value, result.Value, "utcp says echo")
	}
}

func TestGenerateWithFilesPreservesExplicitCodeModeRoutingWithoutFiles(t *testing.T) {
	agent, utcpClient := newToolRoutingTestAgent(t)

	out, err := agent.GenerateWithFiles(
		context.Background(),
		"session",
		"Run code with CodeMode using the echo tool.",
		nil,
	)
	if err != nil {
		t.Fatalf("GenerateWithFiles returned error: %v", err)
	}

	if utcpClient.callCount != 1 {
		t.Fatalf("UTCP call count = %d, want 1", utcpClient.callCount)
	}
	if utcpClient.lastToolName != "echo" {
		t.Fatalf("last UTCP tool = %q, want echo for an explicit CodeMode request", utcpClient.lastToolName)
	}
	if !strings.Contains(out, "utcp says echo") {
		t.Fatalf("GenerateWithFiles output = %q, want output containing %q", out, "utcp says echo")
	}
}

func TestGenerateRejectsExplicitCodeModeWhenUnsafeToolsAreDisabled(t *testing.T) {
	model := &dynamicStubModel{responses: map[string]string{}}
	utcpClient := &stubUTCPClient{}
	a, err := New(Options{
		Model:      model,
		Memory:     memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		UTCPClient: utcpClient,
		CodeMode:   codemode.NewCodeModeUTCP(utcpClient, model),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	_, err = a.Generate(context.Background(), "session", "Run code with CodeMode using the echo tool.")
	if err == nil || !strings.Contains(err.Error(), "unauthorized tool execution") {
		t.Fatalf("Generate error = %v, want unauthorized CodeMode error", err)
	}
	if utcpClient.callCount != 0 {
		t.Fatalf("UTCP call count = %d, want 0", utcpClient.callCount)
	}
}

func TestGenerateUsesNormalToolLoopWhenCodeModeIsRestricted(t *testing.T) {
	model := &dynamicStubModel{responses: map[string]string{
		"You are an agentic UTCP tool execution loop": `{"use_tool":true,"tool_name":"echo","arguments":{"input":"hello"}}`,
	}}
	utcpClient := &stubUTCPClient{searchTools: []utcpTools.Tool{{Name: "echo", Description: "Echo input"}}}
	a, err := New(Options{
		Model:      model,
		Memory:     memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		UTCPClient: utcpClient,
		CodeMode:   codemode.NewCodeModeUTCP(utcpClient, model),
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}

	out, err := a.Generate(context.Background(), "session", "Run the echo tool with hello.")
	if err != nil {
		t.Fatalf("Generate returned error: %v", err)
	}
	if got, want := fmt.Sprint(out), "utcp says echo"; got != want {
		t.Fatalf("Generate output = %q, want %q", got, want)
	}
	if utcpClient.callCount != 1 {
		t.Fatalf("UTCP call count = %d, want 1", utcpClient.callCount)
	}
}

func newToolRoutingTestAgent(t *testing.T) (*Agent, *stubUTCPClient) {
	t.Helper()

	codeSnippet := `let result = codemode.CallTool("echo", {"input": "direct"}); result`
	model := &dynamicStubModel{
		responses: map[string]string{
			"You are a strict UTCP CodeMode planner and executor": fmt.Sprintf(
				`{"tools":["echo"],"code":%q,"stream":false}`,
				codeSnippet,
			),
			"You are an agentic UTCP tool execution loop": `{"use_tool":true,"tool_name":"orchestrated.inspect","arguments":{"input":"repository"}}`,
		},
	}
	utcpClient := &stubUTCPClient{
		searchTools: []utcpTools.Tool{
			{Name: "echo", Description: "Direct CodeMode sentinel"},
			{Name: "orchestrated.inspect", Description: "Inspect a repository through the multi-step orchestrator"},
		},
	}

	agent, err := New(Options{
		Model:            model,
		Memory:           memory.NewSessionMemory(&memory.MemoryBank{}, 4),
		UTCPClient:       utcpClient,
		CodeMode:         codemode.NewCodeModeUTCP(utcpClient, model),
		AllowUnsafeTools: true,
	})
	if err != nil {
		t.Fatalf("New returned error: %v", err)
	}
	return agent, utcpClient
}
