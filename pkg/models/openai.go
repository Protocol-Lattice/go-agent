package models

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
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

// getOpenAIMimeType converts normalized MIME types to OpenAI's expected format
func getOpenAIMimeType(mt string) string {
	mt = strings.ToLower(strings.TrimSpace(mt))
	switch {
	case strings.HasPrefix(mt, "image/"):
		// OpenAI supports: image/jpeg, image/png, image/gif, image/webp
		switch mt {
		case "image/jpeg", "image/jpg":
			return "image/jpeg"
		case "image/png":
			return "image/png"
		case "image/gif":
			return "image/gif"
		case "image/webp":
			return "image/webp"
		default:
			return "" // Unsupported image format
		}
	case strings.HasPrefix(mt, "video/"):
		// OpenAI supports various video formats
		// Common ones: video/mp4, video/mpeg, video/quicktime, video/webm
		return mt
	default:
		return ""
	}
}

func (o *OpenAILLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = o.PromptPrefix + "\n" + prompt
	}

	// Separate files by type
	var textFiles []File
	var mediaFiles []File

	for _, f := range files {
		mt := normalizeMIME(f.Name, f.MIME)

		if isImageOrVideoMIME(mt) && getOpenAIMimeType(mt) != "" {
			mediaFiles = append(mediaFiles, f)
		} else if isTextMIME(mt) {
			textFiles = append(textFiles, f)
		}
	}

	// If no media files, fall back to text-only approach
	if len(mediaFiles) == 0 {
		combined := combinePromptWithFiles(fullPrompt, textFiles)
		return o.Generate(ctx, combined)
	}

	// Build MultiContent message with text and media
	var contentParts []openai.ChatMessagePart

	// Add the text prompt (including inline text files)
	textPrompt := fullPrompt
	if len(textFiles) > 0 {
		textPrompt = combinePromptWithFiles(fullPrompt, textFiles)
	}

	contentParts = append(contentParts, openai.ChatMessagePart{
		Type: openai.ChatMessagePartTypeText,
		Text: textPrompt,
	})

	// Add media files
	for _, f := range mediaFiles {
		mt := normalizeMIME(f.Name, f.MIME)
		openaiMime := getOpenAIMimeType(mt)
		if openaiMime == "" {
			continue
		}

		// Encode as base64
		encoded := base64.StdEncoding.EncodeToString(f.Data)
		dataURL := fmt.Sprintf("data:%s;base64,%s", openaiMime, encoded)

		if strings.HasPrefix(openaiMime, "image/") {
			contentParts = append(contentParts, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL:    dataURL,
					Detail: openai.ImageURLDetailAuto,
				},
			})
		} else if strings.HasPrefix(openaiMime, "video/") {
			// For videos, OpenAI uses the same ImageURL structure
			contentParts = append(contentParts, openai.ChatMessagePart{
				Type: openai.ChatMessagePartTypeImageURL,
				ImageURL: &openai.ChatMessageImageURL{
					URL: dataURL,
				},
			})
		}
	}

	resp, err := o.Client.CreateChatCompletion(ctx, openai.ChatCompletionRequest{
		Model: o.Model,
		Messages: []openai.ChatCompletionMessage{{
			Role:         openai.ChatMessageRoleUser,
			MultiContent: contentParts,
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
