package store

import (
	"strings"
	"testing"
)

func TestTrimJSON(t *testing.T) {
	cases := map[string]string{
		"[1,2,3]":     "1,2,3",
		"[[nested]]":  "nested",
		"no brackets": "no brackets",
	}
	for input, want := range cases {
		if got := trimJSON(input); got != want {
			t.Fatalf("trimJSON(%q) = %q, want %q", input, got, want)
		}
	}
}

func TestPostgresSimilarityOperatorMatchesCosineIndex(t *testing.T) {
	if postgresCosineDistanceOperator != "<=>" {
		t.Fatalf("postgres similarity operator = %q, want cosine distance operator <=>", postgresCosineDistanceOperator)
	}
	if !strings.Contains(defaultPostgresSchema, "vector_cosine_ops") {
		t.Fatal("default schema must keep the cosine vector index")
	}
}
