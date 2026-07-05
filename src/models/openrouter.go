package models

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"os"
	"strings"

	openrouter "github.com/OpenRouterTeam/go-sdk"
	"github.com/OpenRouterTeam/go-sdk/models/components"
)

type OpenRouterLLM struct {
	Client       *openrouter.OpenRouter
	Model        string
	PromptPrefix string
}

func NewOpenRouterLLM(model string, promptPrefix string) *OpenRouterLLM {
	apiKey := os.Getenv("OPENROUTER_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("OPENROUTER_KEY") // fallback
	}

	client := openrouter.New(
		openrouter.WithSecurity(apiKey),
	)
	return &OpenRouterLLM{
		Client:       client,
		Model:        model,
		PromptPrefix: promptPrefix,
	}
}

func (o *OpenRouterLLM) buildPrompt(prompt string) string {
	if o.PromptPrefix != "" {
		return o.PromptPrefix + "\n" + prompt
	}
	return prompt
}

// firstChoiceText extracts the assistant's text out of a non-streaming
// ChatResult. Content is normally a plain string (ChatUserMessageContentTypeStr
// equivalent on the response side), but defensively also handles the
// content-parts array shape by concatenating any text parts.
func firstChoiceText(result *components.ChatResult) (string, error) {
	if result == nil || len(result.Choices) == 0 {
		return "", errors.New("no response from OpenRouter")
	}

	content, ok := result.Choices[0].Message.Content.GetOrZero()
	if !ok {
		return "", errors.New("empty response content from OpenRouter")
	}

	if content.Str != nil {
		return *content.Str, nil
	}

	if len(content.ArrayOfChatContentItems) > 0 {
		var sb strings.Builder
		for _, item := range content.ArrayOfChatContentItems {
			if item.ChatContentText != nil {
				sb.WriteString(item.ChatContentText.Text)
			}
		}
		return sb.String(), nil
	}

	return "", errors.New("unsupported response content shape from OpenRouter")
}

func (o *OpenRouterLLM) Generate(ctx context.Context, prompt string) (any, error) {
	fullPrompt := o.buildPrompt(prompt)

	res, err := o.Client.Chat.Send(ctx, components.ChatRequest{
		Model: openrouter.String(o.Model),
		Messages: []components.ChatMessages{
			components.CreateChatMessagesUser(components.ChatUserMessage{
				Content: components.CreateChatUserMessageContentStr(fullPrompt),
			}),
		},
	}, nil)
	if err != nil {
		return nil, err
	}
	if res == nil || res.ChatResult == nil {
		return nil, errors.New("no response from OpenRouter")
	}

	return firstChoiceText(res.ChatResult)
}

// GenerateStream uses OpenRouter's streaming chat completion API.
func (o *OpenRouterLLM) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	fullPrompt := o.buildPrompt(prompt)

	res, err := o.Client.Chat.Send(ctx, components.ChatRequest{
		Model:  openrouter.String(o.Model),
		Stream: openrouter.Bool(true),
		Messages: []components.ChatMessages{
			components.CreateChatMessagesUser(components.ChatUserMessage{
				Content: components.CreateChatUserMessageContentStr(fullPrompt),
			}),
		},
	}, nil)
	if err != nil {
		return nil, err
	}
	if res == nil || res.EventStream == nil {
		return nil, errors.New("no streaming response from OpenRouter")
	}

	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)
		defer res.EventStream.Close()

		var sb strings.Builder
		for res.EventStream.Next() {
			event := res.EventStream.Value()
			if event == nil {
				continue
			}
			chunk := event.Data

			if chunk.Error != nil {
				ch <- StreamChunk{
					Done:     true,
					FullText: sb.String(),
					Err:      fmt.Errorf("openrouter stream error: %s", chunk.Error.Message),
				}
				return
			}
			if len(chunk.Choices) == 0 {
				continue
			}

			delta, ok := chunk.Choices[0].Delta.Content.GetOrZero()
			if ok && delta != "" {
				sb.WriteString(delta)
				ch <- StreamChunk{Delta: delta}
			}
		}

		if err := res.EventStream.Err(); err != nil {
			ch <- StreamChunk{Done: true, FullText: sb.String(), Err: err}
			return
		}
		ch <- StreamChunk{Done: true, FullText: sb.String()}
	}()

	return ch, nil
}

// getOpenRouterMimeType converts normalized MIME types to OpenRouter's expected format.
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
	fullPrompt := o.buildPrompt(prompt)

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

	var content components.ChatUserMessageContent

	switch {
	case len(imageFiles) > 0:
		// Only the first image is attached, matching the original function's behavior.
		firstImage := imageFiles[0]
		encoded := base64.StdEncoding.EncodeToString(firstImage.Data)
		dataURL := fmt.Sprintf(
			"data:%s;base64,%s",
			getOpenRouterMimeType(normalizeMIME(firstImage.Name, firstImage.MIME)),
			encoded,
		)
		content = components.CreateChatUserMessageContentArrayOfChatContentItems([]components.ChatContentItems{
			components.CreateChatContentItemsText(components.ChatContentText{Text: textPrompt}),
			components.CreateChatContentItemsImageURL(components.ChatContentImage{
				ImageURL: components.ChatContentImageImageURL{URL: dataURL},
			}),
		})

	case len(pdfFiles) > 0:
		firstPDF := pdfFiles[0]
		encoded := base64.StdEncoding.EncodeToString(firstPDF.Data)
		dataURL := fmt.Sprintf("data:application/pdf;base64,%s", encoded)
		filename := firstPDF.Name
		content = components.CreateChatUserMessageContentArrayOfChatContentItems([]components.ChatContentItems{
			components.CreateChatContentItemsText(components.ChatContentText{Text: textPrompt}),
			components.CreateChatContentItemsFile(components.ChatContentFile{
				File: components.File{
					FileData: &dataURL,
					Filename: &filename,
				},
			}),
		})

	default:
		content = components.CreateChatUserMessageContentStr(textPrompt)
	}

	res, err := o.Client.Chat.Send(ctx, components.ChatRequest{
		Model: openrouter.String(o.Model),
		Messages: []components.ChatMessages{
			components.CreateChatMessagesUser(components.ChatUserMessage{
				Content: content,
			}),
		},
	}, nil)
	if err != nil {
		return nil, err
	}
	if res == nil || res.ChatResult == nil {
		return nil, errors.New("no response from OpenRouter")
	}

	return firstChoiceText(res.ChatResult)
}
