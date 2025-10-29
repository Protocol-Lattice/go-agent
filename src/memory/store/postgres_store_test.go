package store

import "testing"

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
