# Lattice

> A batteries-included Agent Development Kit for Go

[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/)
[![CI Status](https://github.com/Raezil/lattice-agent/actions/workflows/go.yml/badge.svg)](https://github.com/Raezil/lattice-agent/actions/workflows/go.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Raezil/lattice-agent.svg)](https://pkg.go.dev/github.com/Raezil/lattice-agent)
[![Go Report Card](https://goreportcard.com/badge/github.com/Raezil/lattice-agent)](https://goreportcard.com/report/github.com/Raezil/lattice-agent)

**Lattice** helps you build AI agents in Go with clean abstractions for LLMs, tool calling, retrieval-augmented memory, and multi-agent coordination. Focus on your domain logic while Lattice handles the orchestration plumbing.

## Why Lattice?

Building production AI agents requires more than just LLM calls. You need:

- **Pluggable LLM providers** that swap without rewriting logic
- **Tool calling** that works across different model APIs
- **Memory systems** that remember context across conversations
- **Multi-agent coordination** for complex workflows
- **Testing infrastructure** that doesn't hit external APIs

Lattice provides all of this with idiomatic Go interfaces and minimal dependencies.

## Features

- 🧩 **Modular Architecture** – Compose agents from reusable modules with declarative configuration
- 🤖 **Multi-Agent Support** – Coordinate specialist agents through a shared catalog and delegation system
- 🔧 **Rich Tooling** – Implement the `Tool` interface once, use everywhere automatically
- 🧠 **Smart Memory** – RAG-powered memory with importance scoring, MMR retrieval, and automatic pruning
- 🔌 **Model Agnostic** – Adapters for Gemini, Anthropic, Ollama, or bring your own
- 📡 **UTCP Ready** – First-class Universal Tool Calling Protocol support

## Quick Start

### Installation

```bash
git clone https://github.com/Raezil/lattice-agent.git
cd lattice-agent
go mod download
```

### Basic Usage

```go
package main

import (
	"context"
	"flag"
	"log"

	"github.com/Raezil/lattice-agent/pkg/adk"
	adkmodules "github.com/Raezil/lattice-agent/pkg/adk/modules"
	"github.com/Raezil/lattice-agent/pkg/agent"
	"github.com/Raezil/lattice-agent/pkg/subagents"

	"github.com/Raezil/lattice-agent/pkg/memory"
	"github.com/Raezil/lattice-agent/pkg/memory/engine"
	"github.com/Raezil/lattice-agent/pkg/models"
	"github.com/Raezil/lattice-agent/pkg/tools"
)

func main() {
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	flag.Parse()
	ctx := context.Background()

	// --- Shared runtime
	researcherModel, err := models.NewGeminiLLM(ctx, "gemini-2.5-pro", "Research summary:")
	if err != nil {
		log.Fatalf("create researcher model: %v", err)
	}
	memOpts := engine.DefaultOptions()

	adkAgent, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
		adk.WithModules(
			adkmodules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				return models.NewGeminiLLM(ctx, "gemini-2.5-pro", "Swarm orchestration:")
			}),
			adkmodules.InQdrantMemory(100000, *qdrantURL, *qdrantCollection, memory.AutoEmbedder(), &memOpts),
			adkmodules.NewToolModule("essentials", adkmodules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
	)
	if err != nil {
		log.Fatal(err)
	}

	agent, err := adkAgent.BuildAgent(ctx)
	if err != nil {
		log.Fatal(err)
	}

	// Use the agent
	resp, err := agent.Generate(ctx, "SessionID", "What is pgvector")
	if err != nil {
		log.Fatal(err)
	}

	log.Println(resp)
}
```

### Running Examples

```bash
# Interactive CLI demo
go run cmd/demo/main.go

# Multi-agent coordination
go run cmd/team/main.go

# Quick start example
go run cmd/quickstart/main.go
```

## Project Structure

```
lattice-agent/
├── cmd/
│   ├── demo/          # Interactive CLI with tools, delegation, and memory
│   ├── quickstart/    # Minimal getting-started example
│   └── team/          # Multi-agent coordination demos
├── pkg/
│   ├── adk/           # Agent Development Kit and module system
│   ├── agent/         # Core coordinator, routing, and delegation
│   ├── memory/        # Memory engine and vector store adapters
│   ├── models/        # LLM provider adapters (Gemini, Ollama, Anthropic)
│   ├── subagents/     # Pre-built specialist agent personas
│   └── tools/         # Built-in tools (echo, calculator, time, etc.)
└── docs/              # Additional documentation
```

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `GOOGLE_API_KEY` | Gemini API credentials | For Gemini models |
| `GEMINI_API_KEY` | Alternative to `GOOGLE_API_KEY` | For Gemini models |
| `DATABASE_URL` | PostgreSQL connection string | For persistent memory |
| `ADK_EMBED_PROVIDER` | Embedding provider override | No (defaults to Gemini) |

### Example Configuration

```bash
export GOOGLE_API_KEY="your-api-key-here"
export DATABASE_URL="postgres://user:pass@localhost:5432/lattice?sslmode=disable"
export ADK_EMBED_PROVIDER="gemini"
```

## Core Concepts

### Memory Engine

Lattice includes a sophisticated memory system with retrieval-augmented generation (RAG):

```go
store := memory.NewInMemoryStore() // or PostgreSQL/Qdrant
engine := memory.NewEngine(store, memory.Options{}).
    WithEmbedder(yourEmbedder)

sessionMemory := memory.NewSessionMemory(
    memory.NewMemoryBankWithStore(store), 
    8, // context window size
).WithEngine(engine)
```

Features:
- **Importance Scoring** – Automatically weights memories by relevance
- **MMR Retrieval** – Maximal Marginal Relevance for diverse results
- **Auto-Pruning** – Removes stale or low-value memories
- **Multiple Backends** – In-memory, PostgreSQL+pgvector, or Qdrant

### Tool System

Create custom tools by implementing a simple interface:

```go
package tools

import (
        "context"
        "fmt"
        "strings"

        "github.com/Raezil/lattice-agent/pkg/agent"
)

// EchoTool repeats the provided input. Useful for testing tool wiring.
type EchoTool struct{}

func (e *EchoTool) Spec() agent.ToolSpec {
        return agent.ToolSpec{
                Name:        "echo",
                Description: "Echoes the provided text back to the caller.",
                InputSchema: map[string]any{
                        "type": "object",
                        "properties": map[string]any{
                                "input": map[string]any{
                                        "type":        "string",
                                        "description": "Text to echo back.",
                                },
                        },
                        "required": []any{"input"},
                },
        }
}

func (e *EchoTool) Invoke(_ context.Context, req agent.ToolRequest) (agent.ToolResponse, error) {
        raw := req.Arguments["input"]
        if raw == nil {
                return agent.ToolResponse{Content: ""}, nil
        }
        return agent.ToolResponse{Content: strings.TrimSpace(fmt.Sprint(raw))}, nil
}
```

Register tools with the module system and they're automatically available to all agents.

### Multi-Agent Coordination

Use **Shared Spaces** to coordinate multiple agents with shared memory

Perfect for:
- Team-based workflows where agents need shared context
- Complex tasks requiring specialist coordination
- Projects with explicit access control requirements

## Development

### Running Tests

```bash
# Run all tests
go test ./...

# Run with coverage
go test -cover ./...

# Run specific package tests
go test ./pkg/memory/...
```

### Code Style

We follow standard Go conventions:
- Use `gofmt` for formatting
- Follow [Effective Go](https://golang.org/doc/effective_go.html) guidelines
- Add tests for new features
- Update documentation when adding capabilities

### Adding New Components

**New LLM Provider:**
1. Implement the `models.LLM` interface in `pkg/models/`
2. Add provider-specific configuration
3. Update documentation and examples

**New Tool:**
1. Implement `agent.Tool` interface in `pkg/tools/`
2. Register with the tool module system
3. Add tests and usage examples

**New Memory Backend:**
1. Implement `memory.VectorStore` interface
2. Add migration scripts if needed
3. Update configuration documentation

## Prerequisites

- **Go** 1.22+ (1.25 recommended)
- **PostgreSQL** 15+ with `pgvector` extension (optional, for persistent memory)
- **API Keys** for your chosen LLM provider

### PostgreSQL Setup (Optional)

For persistent memory with vector search:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

The memory module handles schema migrations automatically.

## Troubleshooting

### Common Issues

**Missing pgvector extension**
```
ERROR: type "vector" does not exist
```
Solution: Run `CREATE EXTENSION vector;` in your PostgreSQL database.

**API key errors**
```
ERROR: authentication failed
```
Solution: Verify your API key is correctly set in the environment where you run the application.

**Tool not found**
```
ERROR: tool "xyz" not registered
```
Solution: Ensure tool names are unique and properly registered in your tool catalog.

### Getting Help

- Check existing [GitHub Issues](https://github.com/Raezil/lattice-agent/issues)
- Review the [examples](./cmd/) for common patterns
- Join discussions in [GitHub Discussions](https://github.com/Raezil/lattice-agent/discussions)

## Contributing

We welcome contributions! Here's how to get started:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Update documentation
5. Submit a pull request

Please ensure:
- Tests pass (`go test ./...`)
- Code is formatted (`gofmt`)
- Documentation is updated
- Commit messages are clear

## License

This project is licensed under the [Apache 2.0 License](./LICENSE).

## Acknowledgments

- Inspired by Google's [Agent Development Kit (Python)](https://github.com/google/adk-python)

---

**Star us on GitHub** if you find Lattice useful! ⭐
