package models

import (
	"context"
	"errors"
	"os"

	"github.com/sashabaranov/go-openai"
)

type OpenAILLM struct {
	Client *openai.Client
	Model  string
}

func NewOpenAILLM(model string) *OpenAILLM {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENAI_KEY") // fallback
	}
	client := openai.NewClient(apiKey)
	return &OpenAILLM{Client: client, Model: model}
}

func (o *OpenAILLM) Generate(ctx context.Context, prompt string) (any, error) {
	resp, err := o.Client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: o.Model,
		Messages: []openai.ChatCompletionMessage{{
			Role:    openai.ChatMessageRoleUser,
			Content: prompt,
		}},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no response from OpenAI")
	}
	return resp.Choices[0].Message.Content, nil
}
