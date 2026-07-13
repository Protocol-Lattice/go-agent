package models

import (
	"context"
	"errors"
	"fmt"
	"strings"

	"google.golang.org/genai"
)

// VertexLLM uses Gemini models through Vertex AI. Its client authenticates
// with Application Default Credentials.
type VertexLLM struct {
	Client       *genai.Client
	Model        string
	PromptPrefix string
}

type vertexClientFactory func(context.Context, *genai.ClientConfig) (*genai.Client, error)

// NewVertexLLM creates a Vertex AI model client for project and location.
// Credentials are intentionally omitted so the Google GenAI SDK uses
// Application Default Credentials.
func NewVertexLLM(
	ctx context.Context,
	model string,
	promptPrefix string,
	project string,
	location string,
) (Agent, error) {
	return newVertexLLM(ctx, model, promptPrefix, project, location, genai.NewClient)
}

func newVertexLLM(
	ctx context.Context,
	model string,
	promptPrefix string,
	project string,
	location string,
	newClient vertexClientFactory,
) (Agent, error) {
	project = strings.TrimSpace(project)
	if project == "" {
		return nil, errors.New("vertex: missing GOOGLE_CLOUD_PROJECT")
	}

	location = strings.TrimSpace(location)
	if location == "" {
		return nil, errors.New("vertex: missing GOOGLE_CLOUD_LOCATION/GOOGLE_CLOUD_REGION")
	}

	client, err := newClient(ctx, &genai.ClientConfig{
		Backend:  genai.BackendVertexAI,
		Project:  project,
		Location: location,
	})
	if err != nil {
		return nil, fmt.Errorf("vertex: create client: %w", err)
	}

	return &VertexLLM{
		Client:       client,
		Model:        model,
		PromptPrefix: promptPrefix,
	}, nil
}

func (v *VertexLLM) Generate(ctx context.Context, prompt string) (any, error) {
	resp, err := v.Client.Models.GenerateContent(
		ctx,
		v.Model,
		[]*genai.Content{genai.NewContentFromText(v.fullPrompt(prompt), genai.RoleUser)},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("vertex generate: %w", err)
	}
	return vertexResponseText(resp)
}

func (v *VertexLLM) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	contents := []*genai.Content{
		genai.NewContentFromText(v.fullPrompt(prompt), genai.RoleUser),
	}

	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)

		var full strings.Builder
		for resp, err := range v.Client.Models.GenerateContentStream(ctx, v.Model, contents, nil) {
			if err != nil {
				ch <- StreamChunk{Done: true, FullText: full.String(), Err: err}
				return
			}
			if resp == nil {
				continue
			}

			delta := resp.Text()
			if delta == "" {
				continue
			}
			full.WriteString(delta)
			ch <- StreamChunk{Delta: delta}
		}

		ch <- StreamChunk{Done: true, FullText: full.String()}
	}()

	return ch, nil
}

func (v *VertexLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	parts := v.contentParts(prompt, files)
	resp, err := v.Client.Models.GenerateContent(
		ctx,
		v.Model,
		[]*genai.Content{genai.NewContentFromParts(parts, genai.RoleUser)},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("vertex generate with files: %w", err)
	}
	return vertexResponseText(resp)
}

func (v *VertexLLM) fullPrompt(prompt string) string {
	if strings.TrimSpace(v.PromptPrefix) == "" {
		return prompt
	}
	return v.PromptPrefix + "\n\n" + prompt
}

func (v *VertexLLM) contentParts(prompt string, files []File) []*genai.Part {
	normalized := make([]File, 0, len(files))
	for _, file := range files {
		normalized = append(normalized, File{
			Name: file.Name,
			MIME: normalizeMIME(file.Name, file.MIME),
			Data: file.Data,
		})
	}

	parts := make([]*genai.Part, 0, len(normalized)+2)
	if prefix := strings.TrimSpace(v.PromptPrefix); prefix != "" {
		parts = append(parts, genai.NewPartFromText(prefix))
	}
	parts = append(parts, genai.NewPartFromText(combinePromptWithFiles(prompt, normalized)))

	for _, file := range normalized {
		if len(file.Data) == 0 {
			continue
		}
		mimeType := sanitizeForGemini(file.MIME)
		if mimeType == "" {
			continue
		}
		parts = append(parts, genai.NewPartFromBytes(file.Data, mimeType))
	}

	return parts
}

func vertexResponseText(resp *genai.GenerateContentResponse) (string, error) {
	if resp == nil {
		return "", errors.New("vertex: empty response")
	}
	text := resp.Text()
	if text == "" {
		return "", errors.New("vertex: empty response")
	}
	return text, nil
}
