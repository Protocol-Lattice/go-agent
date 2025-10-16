package models

import (
	"context"
	"errors"
	"io"
	"os"
	"strings"

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
	if len(resp.Choices) == 0 {
		return nil, errors.New("no response from OpenAI")
	}
	return resp.Choices[0].Message.Content, nil
}

func (o *OpenAILLM) UploadFiles(ctx context.Context, files []UploadFile) ([]UploadedFile, error) {
	uploads := make([]UploadedFile, 0, len(files))
	for _, file := range files {
		purpose := strings.TrimSpace(file.Purpose)
		if purpose == "" {
			purpose = string(openai.PurposeAssistants)
		}
		if file.Reader == nil && strings.TrimSpace(file.Path) != "" {
			resp, err := o.Client.CreateFile(ctx, openai.FileRequest{
				FilePath: file.Path,
				Purpose:  purpose,
				FileName: strings.TrimSpace(file.Name),
			})
			if err != nil {
				return nil, err
			}
			uploads = append(uploads, UploadedFile{
				ID:        resp.ID,
				Name:      resp.FileName,
				SizeBytes: int64(resp.Bytes),
				MIMEType:  "",
				Provider:  "openai",
				Purpose:   resp.Purpose,
			})
			continue
		}

		resolved, err := file.resolve()
		if err != nil {
			return nil, err
		}
		data, readErr := io.ReadAll(resolved.reader)
		closeErr := resolved.Close()
		if readErr == nil {
			readErr = closeErr
		}
		if readErr != nil {
			return nil, readErr
		}
		resp, err := o.Client.CreateFileBytes(ctx, openai.FileBytesRequest{
			Name:    resolved.name,
			Bytes:   data,
			Purpose: openai.PurposeType(purpose),
		})
		if err != nil {
			return nil, err
		}
		uploads = append(uploads, UploadedFile{
			ID:        resp.ID,
			Name:      resp.FileName,
			SizeBytes: int64(resp.Bytes),
			MIMEType:  resolved.mimeType,
			Provider:  "openai",
			Purpose:   resp.Purpose,
		})
	}
	return uploads, nil
}
