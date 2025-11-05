package gotoon_test

import (
	"testing"

	"github.com/alpkeskin/gotoon"
)

func TestEncodeExample(t *testing.T) {
	data := map[string]any{
		"users": []map[string]any{
			{"id": 1, "name": "Alice", "role": "admin"},
			{"id": 2, "name": "Bob", "role": "user"},
		},
	}

	encoded, err := gotoon.Encode(data)
	if err != nil {
		t.Fatalf("Encode returned error: %v", err)
	}

	const want = "{\n  users:\n  [\n    {\n      id: 1\n      name: \"Alice\"\n      role: \"admin\"\n    }\n    {\n      id: 2\n      name: \"Bob\"\n      role: \"user\"\n    }\n  ]\n}"

	if encoded != want {
		t.Fatalf("unexpected encoding\nwant:\n%s\n\ngot:\n%s", want, encoded)
	}
}

func TestEncodeOptions(t *testing.T) {
	data := map[string]int{"b": 2, "a": 1}

	encoded, err := gotoon.Encode(data, gotoon.WithIndent("\t"), gotoon.WithSortedKeys(true))
	if err != nil {
		t.Fatalf("Encode returned error: %v", err)
	}

	const want = "{\n\ta: 1\n\tb: 2\n}"
	if encoded != want {
		t.Fatalf("unexpected encoding with options\nwant:\n%s\n\ngot:\n%s", want, encoded)
	}
}
