# Workspace Intelligence

`go-agent` includes a repository-aware Workspace Intelligence layer for coding agents. It builds a structural index of the codebase and can combine symbol search, imports, dependencies, embeddings, and source context into a bounded context for an agent.

```text
Repository
   │
   ├── AST parser
   ├── Symbol index
   ├── Import/dependency graph
   ├── Embeddings (optional)
   └── File metadata
          │
          ▼
    Hybrid Context Builder
          │
          ▼
        Agent
```

## Basic usage

Build an index for a Go repository:

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/Protocol-Lattice/go-agent/workspace"
)

func main() {
    ctx := context.Background()

    index := workspace.NewIndex(workspace.DefaultConfig("."))
    if err := index.Build(ctx); err != nil {
        log.Fatal(err)
    }

    results := index.SearchSymbols(ctx, "Authenticate", 10)
    for _, result := range results {
        fmt.Printf("%s %s:%d (score=%d)\n",
            result.Symbol.Kind,
            result.Symbol.File,
            result.Symbol.StartLine,
            result.Score,
        )
    }
}
```

The default configuration indexes Go files and ignores `.git`, `vendor`, `node_modules`, `dist`, `build`, and `tmp` directories.

## Building agent context

`BuildContext` turns a natural-language task into a bounded set of relevant source files. Structural symbol matches are used first, and semantic retrieval can be enabled when an embedder is configured.

```go
ctx, err := index.BuildContext(context.Background(), workspace.ContextRequest{
    Query:      "Fix the authentication timeout bug",
    MaxBytes:   32 << 10,
    MaxFiles:   8,
    MaxResults: 20,
    Semantic:   true,
})
if err != nil {
    log.Fatal(err)
}

for _, file := range ctx.Files {
    fmt.Printf("--- %s ---\n%s\n", file.Path, file.Content)
}
```

The context builder:

1. searches indexed symbols;
2. optionally performs semantic/vector search;
3. selects relevant files;
4. expands the selection by one dependency hop;
5. enforces a deterministic file and byte budget.

This lets a coding agent retrieve focused repository context instead of loading the whole workspace.

## Semantic search

Embeddings are optional. The workspace package uses a small `Embedder` interface, so an application can connect any embedding provider without coupling the index to a particular vendor.

```go
type Embedder interface {
    Embed(context.Context, string) ([]float32, error)
}
```

Configure the embedder before building the index:

```go
index := workspace.NewIndex(workspace.DefaultConfig("."))
index.SetEmbedder(myEmbedder)

if err := index.Build(ctx); err != nil {
    log.Fatal(err)
}

results, err := index.SearchSemantic(ctx, "authentication timeout", 10)
if err != nil {
    log.Fatal(err)
}
```

Semantic search is optional; symbol and structural search continue to work without an embedding provider.

## Incremental updates

For long-running coding agents, re-index only changed files:

```go
err := index.Update(ctx, []workspace.FileChange{
    {
        Path: "internal/auth/service.go",
        Kind: workspace.ChangeModified,
    },
})
if err != nil {
    log.Fatal(err)
}
```

Deleted files are removed from the file, symbol, import, and embedding indexes:

```go
err := index.Update(ctx, []workspace.FileChange{
    {
        Path: "internal/auth/legacy.go",
        Kind: workspace.ChangeDeleted,
    },
})
```

Only modified files are parsed and, when semantic indexing is enabled, re-embedded.

## Live workspace watcher

Use `workspace.Watcher` to keep the index synchronized with a working tree:

```go
watcher := &workspace.Watcher{
    Index:    index,
    Interval: 500 * time.Millisecond,
}

go func() {
    if err := watcher.Run(ctx); err != nil && !errors.Is(err, context.Canceled) {
        log.Printf("workspace watcher: %v", err)
    }
}()
```

The watcher uses dependency-free polling. It detects created, modified, and deleted files and forwards changes to `Index.Update`.

## Dependencies

The index also exposes imports and module-aware dependency relationships:

```go
imports := index.Imports("internal/auth/service.go")

for _, path := range imports {
    fmt.Println(path)
}

for _, dependency := range index.Dependencies("internal/auth/service.go") {
    fmt.Println("depends on:", dependency)
}
```

This structural graph can be used by higher-level context ranking to expand from a matched symbol into the implementation and its directly related packages.

## Recommended agent pipeline

For coding agents, the intended architecture is:

```text
User task
   │
   ▼
Workspace Index
   ├── symbols
   ├── imports
   ├── dependencies
   └── embeddings
   │
   ▼
Context Builder
   ├── lexical relevance
   ├── semantic relevance
   ├── dependency expansion
   └── token/byte budget
   │
   ▼
Agent
   │
   ├── plan
   ├── edit
   ├── test
   └── validate
```

For production coding agents, keep the index alive for the lifetime of the agent and run the watcher alongside the agent execution loop. This avoids rebuilding the repository index after every request.
