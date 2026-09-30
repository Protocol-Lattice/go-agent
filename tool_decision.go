package agent

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"time"
)

const (
	DefaultOpenRouterJevModel     = "typesafe/jev-1.13"
	DefaultOpenRouterJevEndpoint  = "https://openrouter.ai/api/alpha/decisions"
	defaultJevDecisionHTTPTimeout = 5 * time.Second
)

// ToolDecisionTool is the minimal tool metadata exposed to a decision layer.
type ToolDecisionTool struct {
	Name        string
	Description string
}

// ToolDecisionInput is the semantic state used to select the next tool.
type ToolDecisionInput struct {
	UserInput          string
	SystemInstructions string
	ConversationMemory string
	Observations       []string
	Tools              []ToolDecisionTool
	Step               int
}

// ToolDecision is the bounded output of a tool-selection decision layer.
type ToolDecision struct {
	UseTool       bool
	ToolName      string
	Confidence    float64
	Probabilities map[string]float64
	RequestID     string
	Provider      string
}

// ToolDecisionLayer chooses the next tool (or completion) without generating
// tool arguments or executing anything.
type ToolDecisionLayer interface {
	DecideTool(context.Context, ToolDecisionInput) (ToolDecision, error)
}

// OpenRouterJevConfig configures Jev as a tool-selection decision layer.
type OpenRouterJevConfig struct {
	APIKey        string
	Model         string
	Endpoint      string
	HTTPClient    *http.Client
	MinConfidence float64
}

// ToolDecisionConfidenceError means Jev returned a valid choice distribution,
// but its confidence did not satisfy the caller's configured execution floor.
type ToolDecisionConfidenceError struct {
	Confidence float64
	Minimum    float64
}

func (e *ToolDecisionConfidenceError) Error() string {
	return fmt.Sprintf("OpenRouter Jev confidence %.3f below minimum %.3f", e.Confidence, e.Minimum)
}

// OpenRouterJevDecisionLayer calls OpenRouter's Decisions API with Jev.
type OpenRouterJevDecisionLayer struct {
	apiKey        string
	model         string
	endpoint      string
	client        *http.Client
	minConfidence float64
}

// NewOpenRouterJevDecisionLayer creates an opt-in Jev 1.13 decision layer.
// APIKey defaults to OPENROUTER_API_KEY and then OPENROUTER_KEY.
func NewOpenRouterJevDecisionLayer(cfg OpenRouterJevConfig) *OpenRouterJevDecisionLayer {
	apiKey := strings.TrimSpace(cfg.APIKey)
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("OPENROUTER_API_KEY"))
	}
	if apiKey == "" {
		apiKey = strings.TrimSpace(os.Getenv("OPENROUTER_KEY"))
	}

	model := strings.TrimSpace(cfg.Model)
	if model == "" {
		model = DefaultOpenRouterJevModel
	}

	endpoint := strings.TrimSpace(cfg.Endpoint)
	if endpoint == "" {
		endpoint = DefaultOpenRouterJevEndpoint
	}

	client := cfg.HTTPClient
	if client == nil {
		client = &http.Client{Timeout: defaultJevDecisionHTTPTimeout}
	}

	minConfidence := cfg.MinConfidence
	if minConfidence < 0 {
		minConfidence = 0
	}
	if minConfidence > 1 {
		minConfidence = 1
	}

	return &OpenRouterJevDecisionLayer{
		apiKey:        apiKey,
		model:         model,
		endpoint:      endpoint,
		client:        client,
		minConfidence: minConfidence,
	}
}

type openRouterJevChoiceQuestion struct {
	Type         string            `json:"type"`
	Instructions string            `json:"instructions"`
	Criteria     map[string]string `json:"criteria"`
}

type openRouterJevRequest struct {
	Model     string                                 `json:"model"`
	State     any                                    `json:"state"`
	Questions map[string]openRouterJevChoiceQuestion `json:"questions"`
}

type openRouterJevChoiceAnswer struct {
	Type          string             `json:"type"`
	Choice        string             `json:"choice"`
	Probabilities map[string]float64 `json:"probabilities"`
	Confidence    float64            `json:"confidence"`
}

type openRouterJevResponse struct {
	ID       string                              `json:"id"`
	Provider string                              `json:"provider"`
	Answers  map[string]openRouterJevChoiceAnswer `json:"answers"`
}

func (l *OpenRouterJevDecisionLayer) DecideTool(ctx context.Context, input ToolDecisionInput) (ToolDecision, error) {
	if l == nil {
		return ToolDecision{}, errors.New("openrouter jev decision layer is nil")
	}
	if strings.TrimSpace(l.apiKey) == "" {
		return ToolDecision{}, errors.New("openrouter jev decision layer requires OPENROUTER_API_KEY or OPENROUTER_KEY")
	}
	if len(input.Tools) == 0 {
		return ToolDecision{}, errors.New("openrouter jev decision layer requires at least one tool")
	}

	criteria := map[string]string{
		"finish": "No further tool is needed. The request can be answered from the current state and previous tool observations.",
	}
	keyToTool := make(map[string]string, len(input.Tools))
	for i, tool := range input.Tools {
		name := strings.TrimSpace(tool.Name)
		if name == "" {
			continue
		}
		key := fmt.Sprintf("tool_%03d", i)
		description := strings.TrimSpace(tool.Description)
		if description == "" {
			description = "No description provided."
		}
		criteria[key] = fmt.Sprintf("Use exact tool %q. %s", name, description)
		keyToTool[key] = name
	}
	if len(keyToTool) == 0 {
		return ToolDecision{}, errors.New("openrouter jev decision layer received no named tools")
	}

	state := map[string]any{
		"user_request":              input.UserInput,
		"system_instructions":       input.SystemInstructions,
		"conversation_memory":       input.ConversationMemory,
		"previous_tool_observations": input.Observations,
		"step":                      input.Step,
	}

	payload := openRouterJevRequest{
		Model: l.model,
		State: state,
		Questions: map[string]openRouterJevChoiceQuestion{
			"next_action": {
				Type: "choice",
				Instructions: "Choose the single best next action for the agent. Prefer an exact tool when additional external information or action is required. Choose finish only when no further tool is needed. Do not invent tools.",
				Criteria: criteria,
			},
		},
	}

	body, err := json.Marshal(payload)
	if err != nil {
		return ToolDecision{}, fmt.Errorf("marshal OpenRouter Jev request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, l.endpoint, bytes.NewReader(body))
	if err != nil {
		return ToolDecision{}, fmt.Errorf("create OpenRouter Jev request: %w", err)
	}
	req.Header.Set("Authorization", "Bearer "+l.apiKey)
	req.Header.Set("Content-Type", "application/json")

	resp, err := l.client.Do(req)
	if err != nil {
		return ToolDecision{}, fmt.Errorf("OpenRouter Jev request: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		limited, _ := io.ReadAll(io.LimitReader(resp.Body, 4<<10))
		return ToolDecision{}, fmt.Errorf("OpenRouter Jev request failed: status=%d body=%s", resp.StatusCode, strings.TrimSpace(string(limited)))
	}

	var decoded openRouterJevResponse
	if err := json.NewDecoder(resp.Body).Decode(&decoded); err != nil {
		return ToolDecision{}, fmt.Errorf("decode OpenRouter Jev response: %w", err)
	}

	answer, ok := decoded.Answers["next_action"]
	if !ok {
		return ToolDecision{}, errors.New("OpenRouter Jev response missing next_action answer")
	}
	if answer.Type != "" && answer.Type != "choice" {
		return ToolDecision{}, fmt.Errorf("OpenRouter Jev returned unexpected answer type %q", answer.Type)
	}
	if l.minConfidence > 0 && answer.Confidence < l.minConfidence {
		return ToolDecision{}, &ToolDecisionConfidenceError{
			Confidence: answer.Confidence,
			Minimum:    l.minConfidence,
		}
	}

	probabilities := make(map[string]float64, len(answer.Probabilities))
	for key, probability := range answer.Probabilities {
		if key == "finish" {
			probabilities["finish"] = probability
			continue
		}
		if toolName, ok := keyToTool[key]; ok {
			probabilities[toolName] = probability
		}
	}

	if answer.Choice == "finish" {
		return ToolDecision{
			UseTool:       false,
			Confidence:    answer.Confidence,
			Probabilities: probabilities,
			RequestID:     decoded.ID,
			Provider:      decoded.Provider,
		}, nil
	}

	toolName, ok := keyToTool[answer.Choice]
	if !ok {
		return ToolDecision{}, fmt.Errorf("OpenRouter Jev selected unknown choice %q", answer.Choice)
	}

	return ToolDecision{
		UseTool:       true,
		ToolName:      toolName,
		Confidence:    answer.Confidence,
		Probabilities: probabilities,
		RequestID:     decoded.ID,
		Provider:      decoded.Provider,
	}, nil
}
