package models

import (
	"context"
	"encoding/base64"
	"errors"
	"fmt"
	"net/http"
	"os"
	"strings"
	"time"

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
		apiKey = os.Getenv("OPENROUTER_KEY")
	}

	httpClient := &http.Client{}
	client := openrouter.New(
		openrouter.WithSecurity(apiKey),
		openrouter.WithClient(httpClient),
		openrouter.WithTimeout(15*time.Minute),
	)
	return &OpenRouterLLM{Client: client, Model: model, PromptPrefix: promptPrefix}
}

func (o *OpenRouterLLM) buildPrompt(prompt string) string {
	full := prompt
	if o.PromptPrefix != "" {
		full = o.PromptPrefix + "\n" + prompt
	}
	if strings.Contains(full, "codemode.run_code") || strings.Contains(full, "CodeMode") {
		full += "\n\nCRITICAL CODEMODE ABI\n" +
			"- Generated source runs in Yaegi.\n" +
			"- Use codemode.CallTool(\"exact.tool.name\", args) or codemode.CallToolStream(\"exact.tool.name\", args).\n" +
			"- Never use bare CallTool(...).\n" +
			"- CallTool and CallToolStream return generic any/interface values. NEVER use concrete type assertions such as result.(string) unless the exact result type was verified. Prefer any values or fmt.Sprint(result) for textual rendering.\n" +
			"- Keep dependent values in one lexical scope.\n" +
			"- CodeMode snippets are statement-only: no package, imports, or main function.\n" +
			"- Return exactly one JSON object for the planner.\n" +
			"- Do not wrap JSON in Markdown or prose.\n" +
			"- arguments.code must be a JSON string with escaped newlines and quotes.\n"
	}
	return full
}

func normalizePlannerText(text string) string {
	text = strings.TrimSpace(text)
	if strings.HasPrefix(text, "```") && strings.HasSuffix(text, "```") {
		lines := strings.Split(text, "\n")
		if len(lines) >= 2 {
			text = strings.TrimSpace(strings.Join(lines[1:len(lines)-1], "\n"))
		}
	}
	return text
}

func firstChoiceText(result *components.ChatResult) (string, error) {
	if result == nil || len(result.Choices) == 0 {
		return "", errors.New("no response from OpenRouter")
	}
	content, ok := result.Choices[0].Message.Content.GetOrZero()
	if !ok {
		return "", errors.New("empty response content from OpenRouter")
	}
	if content.Str != nil {
		return normalizePlannerText(*content.Str), nil
	}
	if len(content.ArrayOfChatContentItems) > 0 {
		var sb strings.Builder
		for _, item := range content.ArrayOfChatContentItems {
			if item.ChatContentText != nil {
				sb.WriteString(item.ChatContentText.Text)
			}
		}
		return normalizePlannerText(sb.String()), nil
	}
	return "", errors.New("unsupported response content shape from OpenRouter")
}

func (o *OpenRouterLLM) Generate(ctx context.Context, prompt string) (any, error) {
	res, err := o.Client.Chat.Send(ctx, components.ChatRequest{
		Model: openrouter.String(o.Model),
		Messages: []components.ChatMessages{components.CreateChatMessagesUser(components.ChatUserMessage{
			Content: components.CreateChatUserMessageContentStr(o.buildPrompt(prompt)),
		})},
	}, nil)
	if err != nil {
		return nil, err
	}
	if res == nil || res.ChatResult == nil {
		return nil, errors.New("no response from OpenRouter")
	}
	return firstChoiceText(res.ChatResult)
}

func (o *OpenRouterLLM) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	res, err := o.Client.Chat.Send(ctx, components.ChatRequest{
		Model: openrouter.String(o.Model), Stream: openrouter.Bool(true),
		Messages: []components.ChatMessages{components.CreateChatMessagesUser(components.ChatUserMessage{
			Content: components.CreateChatUserMessageContentStr(o.buildPrompt(prompt)),
		})},
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
				ch <- StreamChunk{Done: true, FullText: sb.String(), Err: fmt.Errorf("openrouter stream error: %s", chunk.Error.Message)}
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

func getOpenRouterMimeType(mt string) string {
	mt = strings.ToLower(strings.TrimSpace(mt))
	switch {
	case strings.HasPrefix(mt, "image/"):
		switch mt {
		case "image/jpeg", "image/jpg": return "image/jpeg"
		case "image/png": return "image/png"
		case "image/gif": return "image/gif"
		case "image/webp": return "image/webp"
		default: return ""
		}
	case strings.HasPrefix(mt, "video/"):
		return mt
	default:
		return ""
	}
}

func (o *OpenRouterLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	fullPrompt := o.buildPrompt(prompt)
	var textFiles, imageFiles, pdfFiles []File
	for _, f := range files {
		mt := normalizeMIME(f.Name, f.MIME)
		if strings.HasPrefix(mt, "image/") && getOpenRouterMimeType(mt) != "" { imageFiles = append(imageFiles, f)
		} else if strings.HasPrefix(mt, "application/pdf") { pdfFiles = append(pdfFiles, f)
		} else if isTextMIME(mt) { textFiles = append(textFiles, f) }
	}
	if len(imageFiles) == 0 && len(pdfFiles) == 0 {
		return o.Generate(ctx, combinePromptWithFiles(fullPrompt, textFiles))
	}
	textPrompt := fullPrompt
	if len(textFiles) > 0 { textPrompt = combinePromptWithFiles(fullPrompt, textFiles) }
	var content components.ChatUserMessageContent
	switch {
	case len(imageFiles) > 0:
		firstImage := imageFiles[0]
		encoded := base64.StdEncoding.EncodeToString(firstImage.Data)
		dataURL := fmt.Sprintf("data:%s;base64,%s", getOpenRouterMimeType(normalizeMIME(firstImage.Name, firstImage.MIME)), encoded)
		content = components.CreateChatUserMessageContentArrayOfChatContentItems([]components.ChatContentItems{
			components.CreateChatContentItemsText(components.ChatContentText{Text: textPrompt}),
			components.CreateChatContentItemsImageURL(components.ChatContentImage{ImageURL: components.ChatContentImageImageURL{URL: dataURL}}),
		})
	case len(pdfFiles) > 0:
		firstPDF := pdfFiles[0]
		encoded := base64.StdEncoding.EncodeToString(firstPDF.Data)
		dataURL := fmt.Sprintf("data:application/pdf;base64,%s", encoded)
		filename := firstPDF.Name
		content = components.CreateChatUserMessageContentArrayOfChatContentItems([]components.ChatContentItems{
			components.CreateChatContentItemsText(components.ChatContentText{Text: textPrompt}),
			components.CreateChatContentItemsFile(components.ChatContentFile{File: components.File{FileData: &dataURL, Filename: &filename}}),
		})
	default:
		content = components.CreateChatUserMessageContentStr(textPrompt)
	}
	res, err := o.Client.Chat.Send(ctx, components.ChatRequest{
		Model: openrouter.String(o.Model),
		Messages: []components.ChatMessages{components.CreateChatMessagesUser(components.ChatUserMessage{Content: content})},
	}, nil)
	if err != nil { return nil, err }
	if res == nil || res.ChatResult == nil { return nil, errors.New("no response from OpenRouter") }
	return firstChoiceText(res.ChatResult)
}
