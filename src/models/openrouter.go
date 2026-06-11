package models

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/revrost/go-openrouter"
)

type OpenRouterLLM struct {
	Client       *openrouter.Client
	Model        string
	PromptPrefix string
}

func NewOpenRouterLLM(model string, promptPrefix string) *OpenRouterLLM {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_KEY") // fallback
	}

	client := openrouter.NewClient(apiKey)
	return &OpenRouterLLM{
		Client:       client,
		Model:        model,
		PromptPrefix: promptPrefix,
	}
}

func (o *OpenRouterLLM) Generate(ctx context.Context, prompt string) (any, error) {
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = o.PromptPrefix + "\n" + prompt
	}

	resp, err := o.Client.CreateChatCompletion(ctx, openrouter.ChatCompletionRequest{
		Model: o.Model,
		Messages: []openrouter.ChatCompletionMessage{
			openrouter.UserMessage(fullPrompt),
		},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no response from OpenRouter")
	}
	return resp.Choices[0].Message.Content.Text, nil
}

// GenerateStream uses OpenRouter's streaming chat completion API
func (o *OpenRouterLLM) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = o.PromptPrefix + "\n" + prompt
	}

	stream, err := o.Client.CreateChatCompletionStream(ctx, openrouter.ChatCompletionRequest{
		Model: o.Model,
		Messages: []openrouter.ChatCompletionMessage{
			openrouter.UserMessage(fullPrompt),
		},
	})
	if err != nil {
		return nil, err
	}

	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)
		defer stream.Close()
		var sb strings.Builder
		for {
			resp, err := stream.Recv()
			if err != nil {
				if errors.Is(err, io.EOF) {
					ch <- StreamChunk{Done: true, FullText: sb.String()}
					return
				}
				ch <- StreamChunk{Done: true, FullText: sb.String(), Err: err}
				return
			}
			if len(resp.Choices) > 0 {
				delta := resp.Choices[0].Delta.Content
				if delta != "" {
					sb.WriteString(delta)
					ch <- StreamChunk{Delta: delta}
				}
			}
		}
	}()

	return ch, nil
}

// getOpenRouterMimeType converts normalized MIME types to OpenRouter's expected format
func getOpenRouterMimeType(mt string) string {
	mt = strings.ToLower(strings.TrimSpace(mt))
	switch {
	case strings.HasPrefix(mt, "image/"):
		// OpenRouter supports: image/jpeg, image/png, image/gif, image/webp
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
		// OpenRouter supports various video formats
		return mt
	default:
		return ""
	}
}

func (o *OpenRouterLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	fullPrompt := prompt
	if o.PromptPrefix != "" {
		fullPrompt = o.PromptPrefix + "\n" + prompt
	}

	// Separate files by type
	var textFiles []File
	var imageFiles []File
	var pdfFiles []File

	for _, f := range files {
		mt := normalizeMIME(f.Name, f.MIME)

		if strings.HasPrefix(mt, "image/") && getOpenRouterMimeType(mt) != "" {
			imageFiles = append(imageFiles, f)
		} else if strings.HasPrefix(mt, "application/pdf") {
			pdfFiles = append(pdfFiles, f)
		} else if isTextMIME(mt) {
			textFiles = append(textFiles, f)
		}
	}

	// If no media files, fall back to text-only approach
	if len(imageFiles) == 0 && len(pdfFiles) == 0 {
		combined := combinePromptWithFiles(fullPrompt, textFiles)
		return o.Generate(ctx, combined)
	}

	// Build the text prompt with inline text files
	textPrompt := fullPrompt
	if len(textFiles) > 0 {
		textPrompt = combinePromptWithFiles(fullPrompt, textFiles)
	}

	var msg openrouter.ChatCompletionMessage

	// Handle image files using UserMessageWithImage
	if len(imageFiles) > 0 {
		// Use first image with UserMessageWithImage
		firstImage := imageFiles[0]
		encoded := base64.StdEncoding.EncodeToString(firstImage.Data)
		dataURL := fmt.Sprintf("data:%s;base64,%s", getOpenRouterMimeType(normalizeMIME(firstImage.Name, firstImage.MIME)), encoded)
		msg = openrouter.UserMessageWithImage(textPrompt, dataURL)

		// For additional images, we would need to append to the message
		// This depends on go-openrouter's content structure
	} else if len(pdfFiles) > 0 {
		// Handle PDF files using UserMessageWithPDF
		firstPDF := pdfFiles[0]
		msg = openrouter.UserMessageWithPDF(textPrompt, firstPDF.Name, string(firstPDF.Data))
	} else {
		// Fallback to text-only message
		msg = openrouter.UserMessage(textPrompt)
	}

	resp, err := o.Client.CreateChatCompletion(ctx, openrouter.ChatCompletionRequest{
		Model:    o.Model,
		Messages: []openrouter.ChatCompletionMessage{msg},
	})
	if err != nil {
		return nil, err
	}
	if len(resp.Choices) == 0 {
		return nil, errors.New("no response from OpenRouter")
	}
	return resp.Choices[0].Message.Content.Text, nil
}
