package workspace

import (
	"context"
	"math"
	"testing"
)

type testEmbedder struct{}

func (testEmbedder) Embed(_ context.Context, text string) ([]float32, error) {
	if text == "auth" { return []float32{1, 0}, nil }
	if text == "database" { return []float32{0, 1}, nil }
	return []float32{1, 0}, nil
}

func TestSemanticSearch(t *testing.T) {
	root := t.TempDir()
	writeTestFile(t, root, "auth.go", "package auth\nfunc Authenticate() {}\n")
	writeTestFile(t, root, "db.go", "package db\nfunc Query() {}\n")

	idx := NewIndex(DefaultConfig(root))
	idx.SetEmbedder(testEmbedder{})
	if err := idx.Build(context.Background()); err != nil { t.Fatal(err) }
	results, err := idx.SearchSemantic(context.Background(), "database", 2)
	if err != nil { t.Fatal(err) }
	if len(results) != 2 || results[0].Path != "db.go" || results[0].Score <= results[1].Score { t.Fatalf("results = %#v", results) }
	if math.Abs(float64(results[0].Score-1)) > 0.001 { t.Fatalf("score = %v", results[0].Score) }
}
