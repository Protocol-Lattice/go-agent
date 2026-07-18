package store

import (
	"encoding/json"
	"reflect"
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

func TestFormatVectorMatchesJSONVectorEncoding(t *testing.T) {
	for _, vector := range [][]float32{
		nil,
		{0},
		{1.25, -2, 0, 3.5},
		{1.0 / 3.0, 1e-20, 1e20},
	} {
		encoded := formatVector(vector)
		decoded := parseVector(encoded)
		if !reflect.DeepEqual(decoded, vector) {
			t.Fatalf("formatVector(%v) = %q -> %v", vector, encoded, decoded)
		}
	}
}

func BenchmarkPostgresVectorEncoding(b *testing.B) {
	vector := make([]float32, 768)
	for i := range vector {
		vector[i] = float32(i) / 768
	}
	b.Run("json-roundtrip", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			encoded, _ := json.Marshal(vector)
			if result := vectorFromJSON(encoded); len(result) == 0 {
				b.Fatal("empty vector")
			}
		}
	})
	b.Run("direct", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			if result := formatVector(vector); len(result) == 0 {
				b.Fatal("empty vector")
			}
		}
	})
}

func BenchmarkPostgresVectorParsing(b *testing.B) {
	vector := make([]float32, 768)
	for i := range vector {
		vector[i] = float32(i) / 768
	}
	encoded := formatVector(vector)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if result := parseVector(encoded); len(result) != len(vector) {
			b.Fatalf("parsed %d values, want %d", len(result), len(vector))
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
	if postgresCosineScoreExpression != "1 - (embedding <=> $1::vector)" {
		t.Fatalf("postgres score expression = %q, want cosine similarity", postgresCosineScoreExpression)
	}
}
