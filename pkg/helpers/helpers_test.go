package helpers

import (
	"context"
	"errors"
	"testing"

	"github.com/Raezil/lattice-agent/pkg/agent"
)

type stubTool struct{ name string }

func (s stubTool) Spec() agent.ToolSpec { return agent.ToolSpec{Name: s.name} }
func (stubTool) Invoke(context.Context, agent.ToolRequest) (agent.ToolResponse, error) {
	return agent.ToolResponse{}, errors.New("not implemented")
}

type stubSubAgent struct{ name string }

func (s stubSubAgent) Name() string                              { return s.name }
func (stubSubAgent) Description() string                         { return "stub" }
func (stubSubAgent) Run(context.Context, string) (string, error) { return "", nil }

func TestParseSourceBoostFlag(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  map[string]float64
	}{
		{"empty", "", nil},
		{"invalid pairs ignored", "alpha,beta=oops", nil},
		{"mix valid and invalid", "alpha=1.2, beta = 0.5, gamma=bad", map[string]float64{"alpha": 1.2, "beta": 0.5}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ParseSourceBoostFlag(tc.input)
			if len(got) != len(tc.want) {
				t.Fatalf("expected %d entries, got %d", len(tc.want), len(got))
			}
			for k, v := range tc.want {
				if got[k] != v {
					t.Fatalf("expected %s -> %v, got %v", k, v, got[k])
				}
			}
		})
	}
}

func TestToolNames(t *testing.T) {
	if got := ToolNames(nil); got != "<none>" {
		t.Fatalf("expected <none> for nil slice, got %q", got)
	}
	tools := []agent.Tool{stubTool{name: "foo"}, stubTool{name: "bar"}}
	if got := ToolNames(tools); got != "foo, bar" {
		t.Fatalf("unexpected tool names: %q", got)
	}
}

func TestSubAgentNames(t *testing.T) {
	if got := SubAgentNames(nil); got != "<none>" {
		t.Fatalf("expected <none> for nil slice, got %q", got)
	}
	sas := []agent.SubAgent{stubSubAgent{name: "alpha"}, stubSubAgent{name: "beta"}}
	if got := SubAgentNames(sas); got != "alpha, beta" {
		t.Fatalf("unexpected sub-agent names: %q", got)
	}
}

func TestParseCSVList(t *testing.T) {
	if got := ParseCSVList("   "); got != nil {
		t.Fatalf("expected nil for whitespace input, got %#v", got)
	}
	list := ParseCSVList("one, two, , three")
	want := []string{"one", "two", "three"}
	if len(list) != len(want) {
		t.Fatalf("expected %d entries, got %d", len(want), len(list))
	}
	for i, v := range want {
		if list[i] != v {
			t.Fatalf("entry %d: expected %q, got %q", i, v, list[i])
		}
	}
}
