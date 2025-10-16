package models

import (
	"context"
	"errors"
	"fmt"
	"os"
	"strings"

	genai "github.com/google/generative-ai-go/genai"
	"google.golang.org/api/option"
)

// ---------------------------- Google Gemini ----------------------------------

type GeminiLLM struct {
	Client       *genai.Client
	Model        string
	PromptPrefix string
}

func NewGeminiLLM(ctx context.Context, model, promptPrefix string) (*GeminiLLM, error) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		return nil, errors.New("missing GOOGLE_API_KEY or GEMINI_API_KEY")
	}
	client, err := genai.NewClient(ctx, option.WithAPIKey(apiKey))
	if err != nil {
		return nil, fmt.Errorf("gemini init: %w", err)
	}
	return &GeminiLLM{Client: client, Model: model, PromptPrefix: promptPrefix}, nil
}

func (g *GeminiLLM) Generate(ctx context.Context, prompt string) (any, error) {
	model := g.Client.GenerativeModel(g.Model)

	// 🧠 Prepend the role or personality if defined.
	fullPrompt := prompt
	if prefix := strings.TrimSpace(g.PromptPrefix); prefix != "" {
		fullPrompt = fmt.Sprintf("%s %s", prefix, prompt)
	}

	resp, err := model.GenerateContent(ctx, genai.Text(fullPrompt))
	if err != nil {
		return nil, fmt.Errorf("gemini generate: %w", err)
	}
	if len(resp.Candidates) == 0 || resp.Candidates[0].Content == nil {
		return nil, errors.New("gemini: empty response")
	}

	return resp.Candidates[0].Content.Parts[0], nil
}

func (g *GeminiLLM) UploadFiles(ctx context.Context, files []UploadFile) ([]UploadedFile, error) {
	uploads := make([]UploadedFile, 0, len(files))
	for _, file := range files {
		path := strings.TrimSpace(file.Path)
		displayName := strings.TrimSpace(file.Name)
		mimeHint := strings.TrimSpace(file.MIMEType)
		if file.Reader == nil && path != "" {
			gemFile, err := g.Client.UploadFileFromPath(ctx, path, geminiUploadOptions(displayName, mimeHint))
			if err != nil {
				return nil, fmt.Errorf("gemini upload %s: %w", path, err)
			}
			uploads = append(uploads, geminiUploadedFile(gemFile, file.Purpose))
			continue
		}

		resolved, err := file.resolve()
		if err != nil {
			return nil, err
		}
		nameHint := displayName
		if nameHint == "" {
			nameHint = resolved.name
		}
		mime := resolved.mimeType
		if mime == "" {
			mime = mimeHint
		}
		gemFile, uploadErr := g.Client.UploadFile(ctx, "", resolved.reader, geminiUploadOptions(nameHint, mime))
		closeErr := resolved.Close()
		if uploadErr == nil {
			uploadErr = closeErr
		}
		if uploadErr != nil {
			return nil, uploadErr
		}
		uploads = append(uploads, geminiUploadedFile(gemFile, file.Purpose))
	}
	return uploads, nil
}

func geminiUploadOptions(displayName, mimeType string) *genai.UploadFileOptions {
	displayName = strings.TrimSpace(displayName)
	mimeType = strings.TrimSpace(mimeType)
	if displayName == "" && mimeType == "" {
		return nil
	}
	return &genai.UploadFileOptions{DisplayName: displayName, MIMEType: mimeType}
}

func geminiUploadedFile(file *genai.File, purpose string) UploadedFile {
	if file == nil {
		return UploadedFile{Provider: "gemini", Purpose: purpose}
	}
	name := file.DisplayName
	if strings.TrimSpace(name) == "" {
		name = file.Name
	}
	return UploadedFile{
		ID:        file.Name,
		Name:      name,
		SizeBytes: file.SizeBytes,
		MIMEType:  file.MIMEType,
		URI:       file.URI,
		Provider:  "gemini",
		Purpose:   purpose,
	}
}
