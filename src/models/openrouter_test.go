package models

import (
	"strings"
	"testing"

	"github.com/OpenRouterTeam/go-sdk/models/components"
	"github.com/OpenRouterTeam/go-sdk/optionalnullable"
)

func TestFirstChoiceTextHandlesStructuredAnyContent(t *testing.T) {
	content := components.CreateChatAssistantMessageContentAny(map[string]any{
		"tools":  []any{"echo"},
		"code":   `codemode.CallTool("echo", {"input":"hi"})`,
		"stream": false,
	})

	result := &components.ChatResult{
		Choices: []components.ChatChoice{{
			Message: components.ChatAssistantMessage{
				Content: optionalnullable.From(&content),
				Role:    components.ChatAssistantMessageRoleAssistant,
			},
		}},
	}

	got, err := firstChoiceText(result)
	if err != nil {
		t.Fatalf("firstChoiceText returned error: %v", err)
	}
	for _, want := range []string{`"tools":["echo"]`, `"stream":false`, `"code":"`} {
		if !strings.Contains(got, want) {
			t.Fatalf("firstChoiceText output %q does not contain %q", got, want)
		}
	}
}

func TestFirstChoiceTextHandlesNullContent(t *testing.T) {
	result := &components.ChatResult{
		Choices: []components.ChatChoice{{
			Message: components.ChatAssistantMessage{
				Content: optionalnullable.From[components.ChatAssistantMessageContent](nil),
				Role:    components.ChatAssistantMessageRoleAssistant,
			},
		}},
	}

	_, err := firstChoiceText(result)
	if err == nil || !strings.Contains(err.Error(), "empty response content") {
		t.Fatalf("firstChoiceText error = %v, want empty response content error", err)
	}
}

func TestFirstChoiceTextSurfacesRefusalForNullContent(t *testing.T) {
	refusal := "request rejected"
	result := &components.ChatResult{
		Choices: []components.ChatChoice{{
			Message: components.ChatAssistantMessage{
				Content: optionalnullable.From[components.ChatAssistantMessageContent](nil),
				Refusal: optionalnullable.From(&refusal),
				Role:    components.ChatAssistantMessageRoleAssistant,
			},
		}},
	}

	_, err := firstChoiceText(result)
	if err == nil || !strings.Contains(err.Error(), refusal) {
		t.Fatalf("firstChoiceText error = %v, want refusal text", err)
	}
}
