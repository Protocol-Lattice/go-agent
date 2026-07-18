package agent

import (
	"context"
	"testing"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

func benchmarkAgent(b *testing.B) *Agent {
	b.Helper()
	// Fixed delays make overlap regressions visible without depending on a
	// network model or embedding provider.
	mem := memory.NewSessionMemory(nil, 8).WithEmbedder(&gatedEmbedder{delay: time.Millisecond})
	agent, err := New(Options{
		Model:  &signalingModel{response: "ok", delay: time.Millisecond},
		Memory: mem,
	})
	if err != nil {
		b.Fatalf("New returned error: %v", err)
	}
	return agent
}

func BenchmarkAgentGenerate(b *testing.B) {
	agent := benchmarkAgent(b)
	ctx := context.Background()

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := agent.Generate(ctx, "benchmark", "what is latency?"); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkAgentGenerateWithFiles(b *testing.B) {
	agent := benchmarkAgent(b)
	ctx := context.Background()
	files := []models.File{
		{Name: "one.txt", MIME: "text/plain", Data: []byte("one")},
		{Name: "two.txt", MIME: "text/plain", Data: []byte("two")},
		{Name: "three.txt", MIME: "text/plain", Data: []byte("three")},
	}

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := agent.GenerateWithFiles(ctx, "benchmark", "summarize", files); err != nil {
			b.Fatal(err)
		}
	}
}
