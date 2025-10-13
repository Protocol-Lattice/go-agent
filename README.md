# Go Agent Development Kit

A batteries-included starter kit for building production-ready AI agents in Go. The kit now ships with a
composable runtime that wires together language models, retrieval-augmented memory, tools, and specialist
sub-agents so you can focus on domain logic instead of orchestration glue.

## Highlights

- **Runtime orchestration** – `pkg/runtime` exposes a single entry point for creating an agent runtime with
  configurable models, tools, memory, and sub-agents. It returns reusable sessions that encapsulate conversation
  state and memory flushing.
- **Coordinator + specialists** – `pkg/agent` provides the core orchestration logic while `pkg/subagents`
  demonstrates how to bolt on personas such as a researcher that drafts background briefs.
- **Tooling ecosystem** – Implement the `agent.Tool` interface and register it with the runtime. Reference
  implementations (echo, calculator, clock) live in `pkg/tools`.
- **Model Context Protocol connectors** – `pkg/tools/mcp` adapts MCP servers (for example via
  [`mark3labs/mcp-go`](https://github.com/mark3labs/mcp-go)) into runtime tools and resource helpers.
- **Retrieval-augmented memory** – `pkg/memory` combines a short-term window with an optional Postgres + pgvector
  backend for long-term recall. The package gracefully degrades to in-memory only mode when a database is not
  provided (useful for tests and local hacking).
- **Model abstraction** – `pkg/models` defines a tiny interface around `Generate(ctx, prompt)` with adapters for
  Gemini 2.5 Pro and dummy models for offline testing.
- **Command-line demo** – `cmd/demo` is a configurable CLI that spins up the runtime, walks through a scripted
  conversation, and showcases tool usage, delegation, and memory persistence.

## Architecture Overview

```
cmd/demo              # CLI entry point that configures and drives the runtime
pkg/
├── runtime          # High-level runtime + session management
├── agent            # Coordinator agent, tool routing, sub-agent delegation
├── memory           # Short-term cache with optional Postgres/pgvector persistence
├── models           # Gemini, Ollama, Anthropic, Dummy model adapters
├── subagents        # Example researcher persona powered by an LLM
└── tools            # Built-in tools (echo, calculator, time)
```

At the heart of the kit is `runtime.Config`. Provide a database DSN (or your own memory factory), a coordinator
model loader, and any tools/sub-agents you want to expose. The runtime takes care of schema creation, memory
wiring, and returning a `Session` that you can use to `Ask` questions and `Flush` the conversation to long-term
storage.

```go
cfg := runtime.Config{
        DSN:          "postgres://admin:admin@localhost:5432/ragdb?sslmode=disable",
        SchemaPath:   "schema.sql",
        SessionWindow: 8,
        ContextLimit:  6,
        CoordinatorModel: func(ctx context.Context) (models.Agent, error) {
                return models.NewGeminiLLM(ctx, "gemini-2.5-pro", "Coordinator response:" )
        },
        Tools: []agent.Tool{&tools.CalculatorTool{}},
        SubAgents: []agent.SubAgent{subagents.NewResearcher(researcherModel)},
        MCPClients: []mcp.ClientFactory{loadMCPServer},
}
rt, _ := runtime.New(ctx, cfg)
session := rt.NewSession("")
reply, _ := session.Ask(ctx, "How do I wire an agent?")
```

`loadMCPServer` is any factory that returns an implementation of `mcp.Client`. The
[`mark3labs/mcp-go`](https://github.com/mark3labs/mcp-go) project provides a reference Go client—wrap it in a
lightweight adapter that satisfies the interface and the runtime will automatically expose the remote tools and
resources.

## Requirements

- Go 1.22+
- PostgreSQL 15+ with the [`pgvector`](https://github.com/pgvector/pgvector) extension (optional for local
  experimentation, required for long-term memory persistence)
- A Google Gemini API key exported as either `GOOGLE_API_KEY` or `GEMINI_API_KEY`

## Getting Started

1. **Clone the repository**

   ```bash
   git clone https://github.com/Raezil/go-agent-development-kit.git
   cd go-agent-development-kit
   ```

2. **Install dependencies**

   ```bash
   go mod download
   ```

3. **Configure environment**

   ```bash
   export GOOGLE_API_KEY="<your-gemini-api-key>"
   export DATABASE_URL="postgres://admin:admin@localhost:5432/ragdb?sslmode=disable"
   ```

4. **Provision PostgreSQL with pgvector (optional but recommended)**

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

   The CLI will execute [`schema.sql`](schema.sql) automatically if the connection succeeds.

5. **Run the demo conversation**

   ```bash
   go run ./cmd/demo --dsn "$DATABASE_URL"
   ```

   Flags let you customise the coordinator model (`--model`), session identifier (`--session`), context limit
   (`--context`), and short-term memory window (`--window`). Provide additional prompts as positional arguments to
   override the default script.

## Customisation Guide

- **Swap language models** – Implement the `models.Agent` interface or use one of the bundled adapters (Gemini,
  Anthropic, Ollama). Supply a loader function via `runtime.Config.CoordinatorModel`.
- **Add or remove tools** – Create new implementations of `agent.Tool` and append them to `runtime.Config.Tools`.
  Tools are automatically exposed to the coordinator prompt and invocable with the `tool:<name>` convention.
- **Attach MCP servers** – Use `pkg/tools/mcp` to wrap Model Context Protocol clients. Provide one or more
  factories via `runtime.Config.MCPClients` to automatically surface remote tools and resources.
- **Add sub-agents** – Any `agent.SubAgent` can be registered through `runtime.Config.SubAgents`. Delegate tasks in
  conversation with `subagent:<name> do something`.
- **Memory backends** – The runtime defaults to Postgres but you can provide a custom `MemoryFactory` or
  `SessionMemoryBuilder` to plug in alternative storage engines during testing.

## Testing

Run the full test suite:

```bash
go test ./...
```

Unit coverage includes deterministic embeddings, agent orchestration, model adapters, tool behaviours, and the new
runtime/session lifecycle.

## Contributing

Issues and pull requests are welcome! If you add new models, tools, or memory backends, please update the README
and examples so others can benefit from your improvements.
