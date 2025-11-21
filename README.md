[![Go Version](https://img.shields.io/badge/Go-1.25-00ADD8?logo=go&logoColor=white)](https://go.dev/dl/)
[![CI Status](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml/badge.svg)](https://github.com/Protocol-Lattice/go-agent/actions/workflows/go.yml)
[![Go Reference](https://pkg.go.dev/badge/github.com/Protocol-Lattice/go-agent.svg)](https://pkg.go.dev/github.com/Protocol-Lattice/go-agent)
[![Go Report Card](https://goreportcard.com/badge/github.com/Protocol-Lattice/go-agent)](https://goreportcard.com/report/github.com/Protocol-Lattice/go-agent)

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
git clone https://github.com/Protocol-Lattice/go-agent.git
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

	"github.com/Protocol-Lattice/go-agent/src/adk"
	adkmodules "github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/subagents"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/Protocol-Lattice/go-agent/src/tools"
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
│   ├── memory/        # Memory engine and vector store adapters
│   ├── models/        # LLM provider adapters (Gemini, Ollama, Anthropic)
│   ├── subagents/     # Pre-built specialist agent personas
├── └── tools/         # Built-in tools (echo, calculator, time, etc.)
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
- **Multiple Backends** – In-memory, PostgreSQL+pgvector,mongodb, neo4j or Qdrant

### Tool System

Create custom tools by implementing a simple interface:

```go
package tools

import (
        "context"
        "fmt"
        "strings"

        "github.com/Protocol-Lattice/go-agent"
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

### Agent Composability

Lattice treats Agents as first-class Tools. This allows you to expose any agent as a tool to another agent, enabling powerful hierarchical or mesh architectures.

**Why use this?**
- **Specialization**: Create small, focused agents (e.g., "Researcher", "Coder", "Reviewer") and orchestrate them with a "Manager" agent.
- **Encapsulation**: Hide complex multi-step workflows behind a simple natural language interface.
- **Scalability**: Build complex systems by composing simple, testable agents.

### How It Works

When you wrap an agent as a tool:
1. **Context Isolation**: The sub-agent runs in its own session (namespaced like `parent_session.sub.tool_name`). It has its own memory and history, preventing the parent's context window from being polluted with the sub-agent's internal thought process.
2. **Recursive Capability**: Since a sub-agent is just a tool, it can have its own tools—including *other* agents. This allows for arbitrarily deep hierarchies (e.g., CEO -> CTO -> Lead Dev -> Coder).
3. **Standard Interface**: The parent agent doesn't know it's talking to another AI. It simply sees a tool that takes an instruction and returns a result. This means you can swap a "Researcher Agent" with a "Google Search Tool" without changing the parent's logic.

```go
// 1. Create a specialist agent
researcher, _ := agent.New(agent.Options{
    SystemPrompt: "You are a researcher. Search for facts.",
    // ...
})

// 2. Create a manager agent that uses the researcher
manager, _ := agent.New(agent.Options{
    SystemPrompt: "You are a manager. Delegate tasks.",
    Tools: []agent.Tool{
        // Expose the researcher as a tool!
        researcher.AsTool("researcher", "Delegates research tasks to the specialist."),
    },
})

// 3. The manager can now call the researcher tool
// User: "Find out why the sky is blue"
// Manager -> calls tool "researcher" -> Researcher Agent runs -> returns result -> Manager answers
```

### Exposing Agents as UTCP Tools

In addition to the internal `Tool` interface, Lattice agents can be exposed as **Universal Tool Calling Protocol (UTCP)** tools. This allows them to be consumed by any UTCP-compliant client, enabling cross-language and cross-platform agent orchestration.

**Key Functions:**

- `agent.AsUTCPTool(name, description)`: Wraps an agent as a standalone UTCP `tools.Tool` struct.
- `agent.RegisterAsUTCPProvider(ctx, client, name, description)`: Automatically registers the agent as a tool provider on a UTCP client.

**Example:**

```go
// 1. Create your specialist agent
researcher, _ := agent.New(agent.Options{
    SystemPrompt: "You are a researcher.",
})

// 2. Initialize a UTCP client
client := utcp.NewClient()

// 3. Register the agent as a UTCP provider
// This makes the agent available as a tool named "researcher.agent"
err := researcher.RegisterAsUTCPProvider(ctx, client, "researcher.agent", "Deep research agent")
if err != nil {
    log.Fatal(err)
}

// 4. Call the agent via UTCP
// The tool accepts 'instruction' and optional 'session_id'
result, err := client.CallTool(ctx, "researcher.agent", map[string]any{
    "instruction": "Analyze the latest trends in AI agents",
}, "researcher", nil)

fmt.Println(result["response"])
```

**Benefits:**
- **Interoperability**: Your Go agents can be called by Python scripts, CLI tools, or other systems using UTCP.
- **Standardization**: Uses the standard UTCP schema for inputs and outputs.
- **Zero Overhead**: Uses an in-process transport when running within the same Go application, avoiding network latency.

## Why Use TOON?

**Token-Oriented Object Notation (TOON)** is integrated into Lattice to dramatically reduce token consumption when passing structured data to and from LLMs. This is especially critical for AI agent workflows where context windows are precious and API costs scale with token usage.

### The Problem with JSON

Traditional JSON is verbose and wastes tokens on repetitive syntax. Consider passing agent memory or tool responses:

```json
{
  "memories": [
    { "id": 1, "content": "User prefers Python", "importance": 0.9, "timestamp": "2025-01-15" },
    { "id": 2, "content": "User is building CLI tools", "importance": 0.85, "timestamp": "2025-01-14" },
    { "id": 3, "content": "User works with PostgreSQL", "importance": 0.8, "timestamp": "2025-01-13" }
  ]
}
```

**Token count:** ~180 tokens

### The TOON Solution

TOON compresses the same data by eliminating redundancy:

```
memories[3]{id,content,importance,timestamp}:
1,User prefers Python,0.9,2025-01-15
2,User is building CLI tools,0.85,2025-01-14
3,User works with PostgreSQL,0.8,2025-01-13
```

**Token count:** ~85 tokens

**Savings: ~53% fewer tokens**

### Why This Matters for AI Agents

1. **Larger Context Windows** – Fit more memories, tool results, and conversation history into the same context limit
2. **Lower API Costs** – Reduce your LLM API bills by up to 50% on structured data
3. **Faster Processing** – Fewer tokens mean faster inference times and lower latency
4. **Better Memory Systems** – Store and retrieve more historical context without hitting token limits
5. **Multi-Agent Communication** – Pass more information between coordinating agents efficiently

### When TOON Shines

TOON is particularly effective for:

- **Agent Memory Banks** – Retrieving and formatting conversation history
- **Tool Responses** – Returning structured data from database queries or API calls
- **Multi-Agent Coordination** – Sharing state between specialist agents
- **Batch Operations** – Processing multiple similar records (users, tasks, logs)
- **RAG Contexts** – Injecting retrieved documents with metadata

### Example: Memory Retrieval

When your agent queries its memory system, TOON can encode dozens of memories in the space where JSON would fit only a handful:

```go
// Retrieve memories
memories := sessionMemory.Retrieve(ctx, "user preferences", 20)

// Encode with TOON for LLM context
encoded, _ := toon.Marshal(memories, toon.WithLengthMarkers(true))

// Pass to LLM with 40-60% fewer tokens than JSON
prompt := fmt.Sprintf("Based on these memories:\n%s\n\nAnswer the user's question.", encoded)
```

### Human-Readable Format

Despite its compactness, TOON remains readable for debugging and development. The format explicitly declares its schema, making it self-documenting:

```
users[2]{id,name,role}:
1,Alice,admin
2,Bob,user
```

You can immediately see: 2 users, with fields id/name/role, followed by their values.

### Getting Started with TOON

Lattice automatically uses TOON for internal data serialization. To use it in your custom tools or memory adapters:

```go
import "github.com/toon-format/toon-go"

// Encode your structs
encoded, err := toon.Marshal(data, toon.WithLengthMarkers(true))

// Decode back to structs
var result MyStruct
err = toon.Unmarshal(encoded, &result)

// Or decode to dynamic maps
var doc map[string]any
err = toon.Unmarshal(encoded, &doc)
```

For more details, see the [TOON specification](https://github.com/toon-format/spec/blob/main/SPEC.md).

---

**Bottom Line:** TOON helps your agents do more with less, turning token budgets into a competitive advantage rather than a constraint.

## 🧠 Tool Orchestrator (LLM-Driven UTCP Tool Selection)

The **Tool Orchestrator** is an intelligent decision engine that lets the LLM choose when and how to call UTCP tools.
It analyzes user input, evaluates available tools, and returns a structured JSON plan describing the next action.

> This brings `go-agent` to the same capability tier as OpenAI’s *tool choice*, but with fully pluggable **UTCP** backends and Go-native execution.

---

### 🎯 What It Does

* Interprets the user’s request
* Loads and renders all available UTCP tools
* Allows the LLM to reason using **TOON-Go**
* Produces a strict JSON decision object:

  ```json
  {
    "use_tool": true,
    "tool_name": "search.files",
    "arguments": { "query": "config" },
    "reason": "User asked to look for configuration files"
  }
  ```
* Executes the chosen tool deterministically

---

### ⚙️ How It Works

1. **Collect Tool Definitions**

   ```go
   rendered := a.renderUtcpToolsForPrompt()
   ```

2. **Build the Orchestration Prompt**

   ```go
   choicePrompt := fmt.Sprintf(`
     You are a UTCP tool selection engine.

     A user asked:
     %q

     You have access to these UTCP tools:
     %s

     You can also discover tools dynamically using:
     search_tools("<query>", <limit>)

     Return ONLY JSON:
     { "use_tool": ..., "tool_name": "...", "arguments": { }, "reason": "..." }
   `, userInput, rendered)
   ```

3. **LLM Makes a Decision (via TOON)**

   * Coordinator executes reasoning
   * Assistant returns the final JSON only

4. **Agent Executes the Tool**

   * `CallTool`
   * `SearchTools`
   * `CallToolStream`

5. **The result becomes the agent’s final response**

---

## 🧩 Why TOON-Go Improves Tool Selection

The orchestrator uses **TOON** as its structured reasoning layer:

* Coordinator → analyzes tool options
* Assistant → returns the strict JSON
* No hallucinated formatting
* Easy to debug via TOON traces
* Session memory stores the entire reasoning trajectory

This yields **stable, deterministic** tool choice behavior.

---

## 🚀 Example

### **User**

> find all files containing “db connection” in the workspace

### **LLM Output**

```json
{
  "use_tool": true,
  "tool_name": "search.files",
  "arguments": {
    "query": "db connection",
    "limit": 20
  },
  "reason": "User wants to search through the workspace files"
}
```

### **Agent Execution**

The `search.files` UTCP tool is invoked, and its direct output is returned to the user.

---

---

## 🧵 Works Seamlessly with CodeMode + Chain

### **CodeMode**

UTCP tool calls can run inside the Go DSL:

```go
r, _ := codemode.CallTool("echo", map[string]any{"input": "hi"})
```

### **ChainStep**

The orchestrator can:

* call a tool
* run a chain step
* discover tools dynamically
* combine reasoning + execution

This makes `go-agent` one of the first Go frameworks with multi-step, LLM-driven tool-routing.


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

- Check existing [GitHub Issues](https://github.com/Protocol-Lattice/go-agent/issues)
- Review the [examples](./cmd/) for common patterns
- Join discussions in [GitHub Discussions](https://github.com/Protocol-Lattice/go-agent/discussions)

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
