package memory

import (
	"context"
	"testing"
)

func TestDummyEmbeddingDeterministic(t *testing.T) {
	first := DummyEmbedding("hello")
	second := DummyEmbedding("hello")
	if len(first) != 768 {
		t.Fatalf("expected embedding length 768, got %d", len(first))
	}
	for i := range first {
		if first[i] != second[i] {
			t.Fatalf("expected deterministic output, mismatch at index %d", i)
		}
	}
}

func TestAddShortTermTrimsToLimit(t *testing.T) {
	sm := NewSessionMemory(&MemoryBank{}, 3)

	for i := 0; i < 5; i++ {
		content := string(rune('A' + i))
		sm.AddShortTerm("session", content, "{}", nil)
	}

	records, err := sm.RetrieveContext(context.Background(), "session", "", 10)
	if err != nil {
		t.Fatalf("retrieve context: %v", err)
	}
	if len(records) != 3 {
		t.Fatalf("expected 3 records retained, got %d", len(records))
	}
	expected := []string{"C", "D", "E"}
	for i, rec := range records {
		if rec.Content != expected[i] {
			t.Fatalf("unexpected record %d content: %q", i, rec.Content)
		}
	}
}
