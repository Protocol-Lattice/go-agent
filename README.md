# Go Agent Development Kit

A batteries-included starter project for building multi-modal AI agents in Go. The kit wires together a
coordinator agent, specialist sub-agents, tool calling, and a hybrid short-term / long-term memory system so
that you can focus on the domain logic of your agent instead of the plumbing.

## Features

- **Agent orchestration** – The `pkg/agent` package combines a primary LLM with configurable tools and
  specialist sub-agents. The default coordinator prompt encourages explicit tool usage and delegation.
- **LLM abstraction** – The `pkg/models` package currently ships with a Gemini 2.5 Pro integration powered by
  the official `generative-ai-go` SDK.
- **Memory architecture** – `pkg/memory` provides an in-memory short-term cache backed by PostgreSQL +
  `pgvector` for retrieval-augmented, long-term recall. Session interactions can be flushed to persistent
  storage at any time.
- **Composable tooling** – `pkg/tools` exposes reusable tool interfaces and reference implementations for
  echoing user input, calculator operations, and timestamp lookup.
- **Research sub-agent** – `pkg/subagents` demonstrates how to delegate work to a secondary model persona that
  returns structured research summaries.
- **End-to-end demo** – `main.go` showcases a full conversation loop including tool invocation, sub-agent
  delegation, memory retrieval, and persistence.

## Requirements

- Go 1.22 or later
- PostgreSQL 15+ with the [`pgvector`](https://github.com/pgvector/pgvector) extension installed
- A Google Gemini API key exported as either `GOOGLE_API_KEY` or `GEMINI_API_KEY`

## Quick start

1. **Clone the repository**

   ```bash
   git clone https://github.com/Raezil/go-agent-development-kit.git
   cd go-agent-development-kit
   ```

2. **Provision PostgreSQL with pgvector**

   Create a database (the demo uses `ragdb`) and enable the `pgvector` extension:

   ```sql
   CREATE EXTENSION IF NOT EXISTS vector;
   ```

3. **Configure environment variables**

   ```bash
   export GOOGLE_API_KEY="<your-gemini-api-key>"
   export DATABASE_URL="postgres://admin:admin@localhost:5432/ragdb?sslmode=disable"
   ```

   > The demo reads the connection string directly inside `main.go`. You can either export `DATABASE_URL` and
   > change the code to read from it, or update the hard-coded string to match your environment.

4. **Initialize dependencies**

   ```bash
   go mod download
   ```

5. **Run the demo conversation**

   ```bash
   go run ./...
   ```

   On startup the program creates the database schema from `schema.sql`, spins up the coordinator agent, and
   walks through a short scripted conversation that exercises tool use, research delegation, and memory flush.

## Project structure

```
.
├── main.go              # End-to-end orchestration example
├── schema.sql           # Database schema (pgvector table + indexes)
└── pkg
    ├── agent            # Primary agent implementation and configuration options
    ├── memory           # Short-term cache + Postgres-backed long-term memory
    ├── models           # LLM client integrations (Gemini 2.5 Pro)
    ├── subagents        # Example researcher persona
    └── tools            # Built-in tools: echo, calculator, clock
```

## Customising your agent

- **Switch the model** – Replace the Gemini client in `main.go` with another implementation of the `models.Agent`
  interface. You can create new adapters in `pkg/models` as long as they implement `Generate`.
- **Add tools** – Implement the `agent.Tool` interface and register the tool via the `Options.Tools` slice when
  constructing the agent. See `pkg/tools` for examples.
- **Add sub-agents** – Sub-agents must satisfy the `agent.SubAgent` interface. They can use different prompts or
  even different LLM providers. Register them through `Options.SubAgents` and call them with the
  `subagent:<name>` convention.
- **Tune memory behaviour** – `SessionMemory` accepts a configurable short-term context window. Persisted
  memories are stored in PostgreSQL and retrieved using `pgvector` similarity search.

## Database schema

The provided [`schema.sql`](schema.sql) file ensures the required extension, table, and indexes exist:

- `memory_bank` table stores session messages, optional metadata, and the vector embedding
- GIN / ivfflat indexes accelerate similarity search via the `<->` operator

Run the script against your database before starting the agent, or let `main.go` run `CreateSchema` at startup.

## Testing

The project currently ships without automated tests. When you extend the kit, prefer adding table-driven Go tests
under the relevant package directories and run them with:

```bash
go test ./...
```

## Contributing

Issues and pull requests are welcome! If you add new models, tools, or memory backends, please document the setup
steps in this README to keep the developer experience smooth.
