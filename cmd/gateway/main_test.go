package main

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

func newTestRuntime(t *testing.T) *agentRuntime {
	t.Helper()
	model := models.NewDummyLLM("local:")
	mem := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8)
	ag, err := agent.New(agent.Options{
		Model:        model,
		Memory:       mem,
		SystemPrompt: "Test assistant",
		ContextLimit: 8,
	})
	if err != nil {
		t.Fatalf("create test agent: %v", err)
	}
	return &agentRuntime{
		ag:        ag,
		provider:  "dummy",
		model:     "local:",
		subagents: make(map[string]subagentInstance),
	}
}

func TestHandleHealth(t *testing.T) {
	mux := setupMux(newTestRuntime(t))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/health", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var res map[string]any
	if err := json.NewDecoder(w.Body).Decode(&res); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if res["ok"] != true {
		t.Errorf("expected ok=true, got %v", res["ok"])
	}
}

func TestHandleModels(t *testing.T) {
	mux := setupMux(newTestRuntime(t))
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/api/models", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	var res struct {
		CurrentProvider string     `json:"current_provider"`
		Models          []webModel `json:"models"`
	}
	if err := json.NewDecoder(w.Body).Decode(&res); err != nil {
		t.Fatalf("decode JSON: %v", err)
	}
	if res.CurrentProvider != "dummy" {
		t.Errorf("expected current_provider=dummy, got %q", res.CurrentProvider)
	}
	if len(res.Models) == 0 {
		t.Errorf("expected non-empty models catalog")
	}
}

func TestHandleChat_TextAndFiles(t *testing.T) {
	mux := setupMux(newTestRuntime(t))

	// Text message
	body, _ := json.Marshal(chatRequest{Session: "test-session", Message: "ping"})
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/chat", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d (body: %s)", w.Code, w.Body.String())
	}
	var chatRes chatResponse
	if err := json.NewDecoder(w.Body).Decode(&chatRes); err != nil {
		t.Fatalf("decode response: %v", err)
	}
	if chatRes.Session != "test-session" {
		t.Errorf("expected session=test-session, got %q", chatRes.Session)
	}

	// With file attachment
	fileBody, _ := json.Marshal(chatRequest{
		Session: "test-files",
		Message: "analyze",
		Files:   []chatFilePayload{{Name: "main.go", MIME: "text/plain", Data: "package main"}},
	})
	w2 := httptest.NewRecorder()
	mux.ServeHTTP(w2, httptest.NewRequest("POST", "/chat", bytes.NewReader(fileBody)))
	if w2.Code != http.StatusOK {
		t.Fatalf("expected 200 with files, got %d (body: %s)", w2.Code, w2.Body.String())
	}

	// Empty request → 400
	wE := httptest.NewRecorder()
	mux.ServeHTTP(wE, httptest.NewRequest("POST", "/chat", bytes.NewReader([]byte(`{}`))))
	if wE.Code != http.StatusBadRequest {
		t.Errorf("expected 400 for empty request, got %d", wE.Code)
	}
}

func TestHandleStream(t *testing.T) {
	mux := setupMux(newTestRuntime(t))
	body, _ := json.Marshal(chatRequest{Session: "stream-session", Message: "stream test"})
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("POST", "/stream", bytes.NewReader(body)))
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200, got %d", w.Code)
	}
	if !strings.Contains(w.Header().Get("Content-Type"), "text/event-stream") {
		t.Errorf("expected text/event-stream, got %q", w.Header().Get("Content-Type"))
	}
	if !strings.Contains(w.Body.String(), "data: ") {
		t.Errorf("expected SSE data in response, got: %s", w.Body.String())
	}
}

func TestHandleSubagents_CRUD(t *testing.T) {
	mux := setupMux(newTestRuntime(t))

	// List (empty)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, httptest.NewRequest("GET", "/api/subagents", nil))
	if w.Code != http.StatusOK {
		t.Fatalf("GET /api/subagents expected 200, got %d", w.Code)
	}

	// Create
	createBody, _ := json.Marshal(createSubagentRequest{
		Name: "test-reviewer", Description: "Reviews code",
		SystemPrompt: "You are a code reviewer", Provider: "dummy", Model: "local:",
	})
	wC := httptest.NewRecorder()
	mux.ServeHTTP(wC, httptest.NewRequest("POST", "/api/subagents", bytes.NewReader(createBody)))
	if wC.Code != http.StatusCreated {
		t.Fatalf("POST /api/subagents expected 201, got %d (body: %s)", wC.Code, wC.Body.String())
	}

	// List again — must contain the new subagent
	wL := httptest.NewRecorder()
	mux.ServeHTTP(wL, httptest.NewRequest("GET", "/api/subagents", nil))
	var resp struct {
		Subagents []struct {
			Name        string `json:"name"`
			Description string `json:"description"`
		} `json:"subagents"`
	}
	if err := json.NewDecoder(wL.Body).Decode(&resp); err != nil {
		t.Fatalf("decode subagents: %v", err)
	}
	found := false
	for _, sa := range resp.Subagents {
		if sa.Name == "test-reviewer" {
			found = true
			if sa.Description != "Reviews code" {
				t.Errorf("expected description 'Reviews code', got %q", sa.Description)
			}
		}
	}
	if !found {
		t.Errorf("created subagent 'test-reviewer' not found in /api/subagents response")
	}
}
