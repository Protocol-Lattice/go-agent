package models

import (
	"context"
	"errors"
	"os"

	"github.com/sashabaranov/go-openai"
)

type OpenAILLM struct {
	Client       *openai.Client
	Model        string
	PromptPrefix string
}

func NewOpenAILLM(model string, promptPrefix string) *OpenAILLM {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_KEY") // fallback
	}
	client := openai.NewClient(apiKey)
	return &OpenAILLM{Client: client, Model: model, PromptPrefix: promptPrefix}
}

func (o *OpenAILLM) Generate(ctx context.Context, prompt string) (any, error) {
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = o.PromptPrefix + "\n" + prompt
	}

	resp, err := o.Client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: o.Model,
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: fullPrompt,
		}},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 && len(resp.Choices) == 0 {
		// Older lib versions use Choices; guard for both just in case.
		return nil, errors.New("no response from OpenAI")
	}
	// Prefer the standard field
	if len(resp.Choices) > 0 {
		return resp.Choices[0].Message.Content, nil
	}
	return resp.Choices[0].Message.Content, nil
}

func (o *OpenAILLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	combined := combinePromptWithFiles(prompt, files)
	return o.Generate(ctx, combined)
}
