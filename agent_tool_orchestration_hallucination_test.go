package agent

import (
	"testing"

	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
)

func TestToolSpecExistsRequiresExactCanonicalName(t *testing.T) {
	specs := []tools.Tool{
		{Name: "github.search_code_search_code_get"},
		{Name: "filesystem.read"},
	}

	cases := []struct {
		name string
		want bool
	}{
		{name: "github.search_code_search_code_get", want: true},
		{name: "github.search_code", want: false},
		{name: "search_code_search_code_get", want: false},
		{name: "github.search_code_search_code_gets", want: false},
		{name: "GitHub.search_code_search_code_get", want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := toolSpecExists(specs, tc.name); got != tc.want {
				t.Fatalf("toolSpecExists(%q) = %v, want %v", tc.name, got, tc.want)
			}
		})
	}
}

func TestAppendCodeModeToolSpecDoesNotDuplicateCanonicalTool(t *testing.T) {
	specs := []tools.Tool{{Name: "filesystem.read"}}

	first := appendCodeModeToolSpec(specs)
	second := appendCodeModeToolSpec(first)

	count := 0
	for _, spec := range second {
		if spec.Name == "codemode.run_code" {
			count++
		}
	}
	if count != 1 {
		t.Fatalf("codemode.run_code count = %d, want 1", count)
	}
}

func TestAppendCodeModeToolSpecPreservesRegisteredTools(t *testing.T) {
	specs := []tools.Tool{{Name: "github.search_code_search_code_get"}}
	got := appendCodeModeToolSpec(specs)

	if !toolSpecExists(got, "github.search_code_search_code_get") {
		t.Fatal("registered canonical tool was lost when adding CodeMode")
	}
	if !toolSpecExists(got, "codemode.run_code") {
		t.Fatal("CodeMode tool was not added")
	}
}
