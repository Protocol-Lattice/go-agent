package workspace

import (
	"context"
	"os"
	"path/filepath"
	"testing"
)

func TestIndexBuildSearchAndContext(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "go.mod"), []byte("module example.com/demo\n\ngo 1.25\n"), 0o644); err != nil { t.Fatal(err) }
	if err := os.MkdirAll(filepath.Join(root, "internal", "auth"), 0o755); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(root, "internal", "auth", "auth.go"), []byte(`package auth

type Service struct{}

func (s *Service) Authenticate(user string) bool { return user != "" }
`), 0o644); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte(`package main

import "example.com/demo/internal/auth"

func main() { _ = auth.Service{} }
`), 0o644); err != nil { t.Fatal(err) }

	idx := NewIndex(DefaultConfig(root))
	if err := idx.Build(context.Background()); err != nil { t.Fatal(err) }
	if got := idx.Module(); got != "example.com/demo" { t.Fatalf("module = %q", got) }
	if len(idx.Files()) != 2 { t.Fatalf("files = %d, want 2", len(idx.Files())) }

	results := idx.SearchSymbols(context.Background(), "Authenticate", 10)
	if len(results) == 0 || results[0].Symbol.Name != "Authenticate" { t.Fatalf("unexpected search results: %#v", results) }
	if results[0].Symbol.Kind != SymbolMethod { t.Fatalf("kind = %q, want method", results[0].Symbol.Kind) }

	ctx, err := idx.BuildContext(context.Background(), ContextRequest{Query: "Authenticate", MaxFiles: 2, MaxBytes: 4096})
	if err != nil { t.Fatal(err) }
	if len(ctx.Files) == 0 { t.Fatal("expected context files") }
	if ctx.Files[0].Content == "" { t.Fatal("expected file content") }

	deps := idx.Dependencies("main.go")
	if len(deps) != 1 || deps[0] != "internal/auth/auth.go" { t.Fatalf("dependencies = %#v", deps) }
	dependents := idx.Dependents("internal/auth/auth.go")
	if len(dependents) != 1 || dependents[0] != "main.go" { t.Fatalf("dependents = %#v", dependents) }
}

func TestIndexIgnoresDirectoriesAndUnsupportedFiles(t *testing.T) {
	root := t.TempDir()
	if err := os.MkdirAll(filepath.Join(root, ".git"), 0o755); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(root, "main.go"), []byte("package main\nfunc main() {}\n"), 0o644); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(root, "README.md"), []byte("main"), 0o644); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(root, ".git", "ignored.go"), []byte("package ignored\n"), 0o644); err != nil { t.Fatal(err) }

	idx := NewIndex(DefaultConfig(root))
	if err := idx.Build(context.Background()); err != nil { t.Fatal(err) }
	if len(idx.Files()) != 1 || idx.Files()[0].Path != "main.go" { t.Fatalf("unexpected files: %#v", idx.Files()) }
}
