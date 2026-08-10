package workspace

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"
)

type testEmbedder struct{}

func (testEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if text == "database" || contains(text, "db") { return []float32{0, 1}, nil }
	return []float32{1, 0}, nil
}

func contains(s, sub string) bool { for i := 0; i+len(sub) <= len(s); i++ { if s[i:i+len(sub)] == sub { return true } }; return false }

func TestSemanticSearch(t *testing.T) {
	root := t.TempDir()
	if err := os.WriteFile(filepath.Join(root, "auth.go"), []byte("package auth\nfunc Authenticate() {}\n"), 0o644); err != nil { t.Fatal(err) }
	if err := os.WriteFile(filepath.Join(root, "db.go"), []byte("package db\nfunc Query() {}\n"), 0o644); err != nil { t.Fatal(err) }

	idx := NewIndex(DefaultConfig(root))
	idx.SetEmbedder(testEmbedder{})
	if err := idx.Build(context.Background()); err != nil { t.Fatal(err) }
	results, err := idx.SearchSemantic(context.Background(), "database", 2)
	if err != nil { t.Fatal(err) }
	if len(results) != 2 || results[0].Path != "db.go" || results[0].Score <= results[1].Score { t.Fatalf("results = %#v", results) }
	if math.Abs(float64(results[0].Score-1)) > 0.001 { t.Fatalf("score = %v", results[0].Score) }
}
