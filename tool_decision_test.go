package agent

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
)

func TestOpenRouterJevDecisionLayerSelectsExactTool(t *testing.T) {
	var request openRouterJevRequest
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if got := r.Header.Get("Authorization"); got != "Bearer test-key" {
			t.Fatalf("Authorization = %q, want Bearer test-key", got)
		}
		if err := json.NewDecoder(r.Body).Decode(&request); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"id":"gen-dec-test",
			"provider":"TypeSafe",
			"answers":{
				"next_action":{
					"type":"choice",
					"choice":"tool_001",
					"probabilities":{"tool_000":0.10,"tool_001":0.85,"finish":0.05},
					"confidence":0.80
				}
			}
		}`))
	}))
	defer server.Close()

	layer := NewOpenRouterJevDecisionLayer(OpenRouterJevConfig{
		APIKey:     "test-key",
		Endpoint:   server.URL,
		HTTPClient: server.Client(),
	})

	decision, err := layer.DecideTool(context.Background(), ToolDecisionInput{
		UserInput: "use beta for this request",
		Tools: []ToolDecisionTool{
			{Name: "alpha", Description: "Alpha tool"},
			{Name: "beta", Description: "Beta tool"},
		},
	})
	if err != nil {
		t.Fatalf("DecideTool returned error: %v", err)
	}

	if request.Model != DefaultOpenRouterJevModel {
		t.Fatalf("model = %q, want %q", request.Model, DefaultOpenRouterJevModel)
	}
	question := request.Questions["next_action"]
	if question.Type != "choice" {
		t.Fatalf("question type = %q, want choice", question.Type)
	}
	if len(question.Criteria) != 3 {
		t.Fatalf("criteria count = %d, want 3 including finish", len(question.Criteria))
	}
	if !decision.UseTool || decision.ToolName != "beta" {
		t.Fatalf("decision = %#v, want beta tool", decision)
	}
	if decision.RequestID != "gen-dec-test" || decision.Provider != "TypeSafe" {
		t.Fatalf("decision metadata = %#v", decision)
	}
	if got := decision.Probabilities["beta"]; got != 0.85 {
		t.Fatalf("beta probability = %v, want 0.85", got)
	}
	if got := decision.Probabilities["alpha"]; got != 0.10 {
		t.Fatalf("alpha probability = %v, want 0.10", got)
	}
	if got := decision.Probabilities["finish"]; got != 0.05 {
		t.Fatalf("finish probability = %v, want 0.05", got)
	}
}
