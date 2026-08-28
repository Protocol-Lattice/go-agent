package agent

import (
	"context"
	"fmt"
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
	if got, want := fmt.Sprint(out), "utcp says echo"; got != want {
		t.Fatalf("Generate output = %q, want %q", got, want)
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
	if got, want := out, "utcp says echo"; got != want {
		t.Fatalf("GenerateWithFiles output = %q, want %q", got, want)
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
