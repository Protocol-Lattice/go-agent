# Go Agent Development Kit

A batteries-included starter kit for building production-ready AI agents in Go. The kit now ships with a
composable runtime that wires together language models, retrieval-augmented memory, tools, and specialist
sub-agents so you can focus on domain logic instead of orchestration glue.

## Highlights
- **Runtime orchestration** – `pkg/runtime` exposes a single entry point for creating an agent runtime with
  configurable models, tools, memory, and sub-agents. The runtime now uses an ADK-style session manager that keeps
  the active session registry thread-safe and deterministic.
- **Coordinator + specialists** – `pkg/agent` provides the core orchestration logic while `pkg/subagents`
  demonstrates how to bolt on personas such as a researcher that drafts background briefs. Tooling and sub-agents are
  registered through catalog/directory abstractions that mirror the Google ADK terminology.
- **Tooling ecosystem** – Implement the `agent.Tool` interface and register it with the runtime. Reference
  implementations (echo, calculator, clock) live in `pkg/tools`.
- **Advanced memory engine** – `pkg/memory` ships with an `Engine` that layers importance scoring, weighted
  retrieval (similarity/importance/recency/source), maximal marginal relevance diversification, cluster-based
  summarisation, incremental re-embedding, and composable pruning (TTL, max size, LRU × (1-importance),
  deduplication) on top of any vector store.
- **Retrieval-augmented memory** – pkg/memory now supports pluggable vector stores:
  - Postgres + pgvector (first-party)
  - Qdrant (first-party)
  - In-memory (for tests and local experimentation)
- **Model abstraction** – `pkg/models` defines a tiny interface around `Generate(ctx, prompt)` with adapters for
  Gemini 2.5 Pro and dummy models for offline testing.
- **Command-line demo** – `cmd/demo` is a configurable CLI that spins up the runtime, walks through a scripted
  conversation, and showcases tool usage, delegation, and memory persistence.
- **UTCP Support** - The Universal Tool Calling Protocol (UTCP) is a modern, flexible, and scalable standard for defining and interacting with tools across a wide variety of communication protocols. 

## Architecture Overview

```
cmd/demo              # CLI entry point that configures and drives the runtime
pkg/
├── runtime          # High-level runtime + session management
├── agent            # Coordinator agent, tool routing, sub-agent delegation via ToolCatalog/SubAgentDirectory
├── memory           # Short-term cache with optional Postgres/pgvector persistence
├── models           # Gemini, Ollama, Anthropic, Dummy model adapters
├── subagents        # Example researcher persona powered by an LLM
└── tools            # Built-in tools (echo, calculator, time)
```

At the heart of the kit is `runtime.Config`. Provide a database DSN (or your own memory factory), a coordinator
model loader, and any tools/sub-agents you want to expose. The runtime takes care of schema creation, memory wiring,
and returning a `Session` that you can use to `Generate` questions and `Flush` the conversation to long-term storage. Tool
and specialist registration now flows through `agent.ToolCatalog` and `agent.SubAgentDirectory`, matching the core
ADK abstractions and making it easy to override the registries when embedding the runtime into larger systems.

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
}
rt, _ := runtime.New(ctx, cfg)
session := rt.NewSession("")
reply, _ := rt.Generate(ctx, session.ID(), "How do I wire an agent?")
```

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
   export ADK_EMBED_PROVIDER="gemini"
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
- **Add sub-agents** – Any `agent.SubAgent` can be registered through `runtime.Config.SubAgents`. Delegate tasks in
  conversation with `subagent:<name> do something`.
- **Memory backends** – The runtime defaults to Postgres but you can provide a custom `MemoryFactory` or
  `SessionMemoryBuilder` to plug in alternative storage engines during testing.

### Advanced Memory Engine

The `pkg/memory` package now exposes an `Engine` that composes retrieval heuristics, clustering, and pruning on top of
any `VectorStore`. Instantiate it with the store of your choice and pass it to the session memory builder:

```go
store := memory.NewInMemoryStore() // or postgres/qdrant implementation
opts := memory.Options{}
engine := memory.NewEngine(store, opts).WithEmbedder(memory.DummyEmbedder{})

sessionMemory := memory.NewSessionMemory(memory.NewMemoryBankWithStore(store), 8).
        WithEngine(engine)
```

The demo CLI exposes runtime overrides for every option (weights, half-life, TTL, dedupe threshold, etc.). For example:

```bash
go run ./cmd/demo \
  --memory-sim-weight=0.6 \
  --memory-recency-weight=0.2 \
  --memory-half-life=48h \
  --memory-source-boost="pagerduty=1.0,slack=0.6"
```

This keeps configuration runtime-only—no YAML files required—and supports zero-downtime migrations via online-safe
`ALTER TABLE ... IF NOT EXISTS` statements in the Postgres store.

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
