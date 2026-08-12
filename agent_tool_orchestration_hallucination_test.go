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

func TestValidateCodeModeToolCallsAcceptsCanonicalNames(t *testing.T) {
	a := &Agent{}
	a.toolCatalog = NewStaticToolCatalog(nil)
	_ = a.toolCatalog.Register(testTool("github.search_code_search_code_get"))

	code := `__out = CallTool("github.search_code_search_code_get", map[string]any{"q": "agent"})`
	if err := a.validateCodeModeToolCalls(code); err != nil {
		t.Fatalf("canonical CodeMode tool call rejected: %v", err)
	}
}

func TestValidateCodeModeToolCallsRejectsHallucinatedNames(t *testing.T) {
	a := &Agent{}
	a.toolCatalog = NewStaticToolCatalog(nil)
	_ = a.toolCatalog.Register(testTool("github.search_code_search_code_get"))

	code := `__out = CallTool("github.search_code", map[string]any{"q": "agent"})`
	if err := a.validateCodeModeToolCalls(code); err == nil {
		t.Fatal("hallucinated CodeMode tool call was accepted")
	}
}

func TestValidateCodeModeToolCallsRejectsDynamicNames(t *testing.T) {
	a := &Agent{}
	a.toolCatalog = NewStaticToolCatalog(nil)
	_ = a.toolCatalog.Register(testTool("github.search_code_search_code_get"))

	code := `name := "github.search_code_search_code_get"; __out = CallTool(name, map[string]any{})`
	if err := a.validateCodeModeToolCalls(code); err == nil {
		t.Fatal("dynamic CodeMode tool name was accepted")
	}
}

func TestValidateCodeModeToolCallsRejectsMixedValidAndHallucinatedCalls(t *testing.T) {
	a := &Agent{}
	a.toolCatalog = NewStaticToolCatalog(nil)
	_ = a.toolCatalog.Register(testTool("github.search_code_search_code_get"))

	code := `__out = CallTool("github.search_code_search_code_get", map[string]any{}); __out = CallTool("github.search_code", map[string]any{})`
	if err := a.validateCodeModeToolCalls(code); err == nil {
		t.Fatal("mixed canonical/hallucinated CodeMode calls were accepted")
	}
}

func testTool(name string) Tool {
	return &testToolImpl{name: name}
}

type testToolImpl struct {
	name string
}

func (t *testToolImpl) Spec() ToolSpec {
	return ToolSpec{Name: t.name, InputSchema: map[string]any{"type": "object"}}
}

func (t *testToolImpl) Invoke(_ context.Context, _ ToolRequest) (ToolResponse, error) {
	return ToolResponse{Content: "ok"}, nil
}
