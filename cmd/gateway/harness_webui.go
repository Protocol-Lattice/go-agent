package main

import (
	"net/http"

	agent "github.com/Protocol-Lattice/go-agent"
)

// harnessWebSnapshot is the small, UI-oriented projection of go-harness
// concepts exposed by the go-agent gateway. The actual execution still lives
// in the agent runtime; this endpoint only describes the current capabilities
// and the staged workflow so the WebUI can present the same mental model.
type harnessWebSnapshot struct {
	Mode         string            `json:"mode"`
	Architecture []harnessStage    `json:"architecture"`
	Capabilities []string          `json:"capabilities"`
	Workspace    harnessWorkspace  `json:"workspace"`
	Runtime      harnessRuntime    `json:"runtime"`
}

type harnessStage struct {
	ID          string `json:"id"`
	Title       string `json:"title"`
	Description string `json:"description"`
}

type harnessWorkspace struct {
	Indexed bool `json:"indexed"`
	Files   int  `json:"files"`
	Symbols int  `json:"symbols"`
	Graph   bool `json:"graph"`
	AST     bool `json:"ast"`
}

type harnessRuntime struct {
	Provider string `json:"provider"`
	Model    string `json:"model"`
	Skills   int    `json:"skills"`
	Tools    int    `json:"tools"`
}

func handleHarness(runtime *agentRuntime) http.HandlerFunc {
	return func(w http.ResponseWriter, _ *http.Request) {
		ag, client, provider, model := runtime.current()
		toolCount := 0
		if ag != nil {
			toolCount = len(ag.WebUITools())
		}
		if client != nil {
			if tools, err := client.SearchTools("", 100); err == nil {
				toolCount += len(tools)
			}
		}
		skillCount := 0
		if ag != nil {
			skillCount = len(ag.WebUISkills())
		}

		writeJSON(w, http.StatusOK, harnessWebSnapshot{
			Mode: "static",
			Architecture: []harnessStage{
				{ID: "planner", Title: "Planner", Description: "Turn the user request into an executable task plan."},
				{ID: "context", Title: "Context Builder", Description: "Ground the plan in the available workspace, skills, and tools."},
				{ID: "editor", Title: "Editor", Description: "Execute filesystem, CodeMode, and UTCP mutations."},
				{ID: "validator", Title: "Validator", Description: "Run checks and inspect tool results before declaring success."},
				{ID: "repairer", Title: "Repairer", Description: "Feed diagnostics back into the workflow for bounded self-healing."},
			},
			Capabilities: []string{
				"approval-gated tools",
				"deterministic retrieval",
				"symbol-aware editing",
				"workspace diff previews",
				"fast parallel validation",
				"bounded dynamic workflows",
			},
			Workspace: harnessWorkspace{
				Indexed: false,
				Graph:   true,
				AST:     true,
			},
			Runtime: harnessRuntime{
				Provider: provider,
				Model:    model,
				Skills:   skillCount,
				Tools:    toolCount,
			},
		})
	}
}
