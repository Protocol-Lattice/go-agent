package models

import (
	"context"
	"errors"
	"strings"
	"testing"

	"google.golang.org/genai"
)

func TestNewVertexLLMUsesVertexBackendAndADC(t *testing.T) {
	var gotConfig *genai.ClientConfig
	fakeClient := &genai.Client{}

	agent, err := newVertexLLM(
		context.Background(),
		"gemini-2.5-flash",
		"prefix",
		" project-id ",
		" europe-west4 ",
		func(_ context.Context, config *genai.ClientConfig) (*genai.Client, error) {
			copyConfig := *config
			gotConfig = &copyConfig
			return fakeClient, nil
		},
	)
	if err != nil {
		t.Fatalf("newVertexLLM() error = %v", err)
	}

	vertex, ok := agent.(*VertexLLM)
	if !ok {
		t.Fatalf("newVertexLLM() returned %T, want *VertexLLM", agent)
	}
	if vertex.Client != fakeClient || vertex.Model != "gemini-2.5-flash" || vertex.PromptPrefix != "prefix" {
		t.Fatalf("unexpected VertexLLM: %#v", vertex)
	}
	if gotConfig == nil {
		t.Fatal("GenAI client factory was not called")
	}
	if gotConfig.Backend != genai.BackendVertexAI {
		t.Fatalf("backend = %v, want %v", gotConfig.Backend, genai.BackendVertexAI)
	}
	if gotConfig.Project != "project-id" || gotConfig.Location != "europe-west4" {
		t.Fatalf("project/location = %q/%q", gotConfig.Project, gotConfig.Location)
	}
	if gotConfig.APIKey != "" || gotConfig.Credentials != nil {
		t.Fatal("Vertex client must leave explicit credentials unset so the SDK uses ADC")
	}
}

func TestNewVertexLLMValidatesProjectAndLocation(t *testing.T) {
	factoryCalled := false
	factory := func(context.Context, *genai.ClientConfig) (*genai.Client, error) {
		factoryCalled = true
		return &genai.Client{}, nil
	}

	_, err := newVertexLLM(context.Background(), "model", "", "", "global", factory)
	if err == nil || !strings.Contains(err.Error(), "GOOGLE_CLOUD_PROJECT") {
		t.Fatalf("missing project error = %v", err)
	}
	_, err = newVertexLLM(context.Background(), "model", "", "project", "", factory)
	if err == nil || !strings.Contains(err.Error(), "GOOGLE_CLOUD_LOCATION") {
		t.Fatalf("missing location error = %v", err)
	}
	if factoryCalled {
		t.Fatal("client factory called for invalid configuration")
	}
}

func TestNewVertexLLMWrapsClientError(t *testing.T) {
	want := errors.New("ADC unavailable")
	_, err := newVertexLLM(
		context.Background(), "model", "", "project", "global",
		func(context.Context, *genai.ClientConfig) (*genai.Client, error) {
			return nil, want
		},
	)
	if !errors.Is(err, want) {
		t.Fatalf("newVertexLLM() error = %v, want wrapped %v", err, want)
	}
}

func TestNewLLMProviderRoutesVertexConfiguration(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "project-id")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
	t.Setenv("GOOGLE_CLOUD_REGION", "ignored-region")

	var gotModel, gotPrefix, gotProject, gotLocation string
	_, err := newLLMProvider(
		context.Background(),
		"vertex",
		"gemini-2.5-flash",
		"prefix",
		func(_ context.Context, model, prefix, project, location string) (Agent, error) {
			gotModel = model
			gotPrefix = prefix
			gotProject = project
			gotLocation = location
			return NewDummyLLM("vertex:"), nil
		},
	)
	if err != nil {
		t.Fatalf("newLLMProvider() error = %v", err)
	}
	if gotModel != "gemini-2.5-flash" || gotPrefix != "prefix" {
		t.Fatalf("model/prefix = %q/%q", gotModel, gotPrefix)
	}
	if gotProject != "project-id" || gotLocation != "us-central1" {
		t.Fatalf("project/location = %q/%q", gotProject, gotLocation)
	}
}

func TestNewLLMProviderVertexFallsBackToRegion(t *testing.T) {
	t.Setenv("GOOGLE_CLOUD_PROJECT", "project-id")
	t.Setenv("GOOGLE_CLOUD_LOCATION", "")
	t.Setenv("GOOGLE_CLOUD_REGION", "europe-west4")

	var gotLocation string
	_, err := newLLMProvider(
		context.Background(), "vertex-ai", "model", "",
		func(_ context.Context, _, _, _, location string) (Agent, error) {
			gotLocation = location
			return NewDummyLLM("vertex:"), nil
		},
	)
	if err != nil {
		t.Fatalf("newLLMProvider() error = %v", err)
	}
	if gotLocation != "europe-west4" {
		t.Fatalf("location = %q, want europe-west4", gotLocation)
	}
}

func TestVertexContentParts(t *testing.T) {
	vertex := &VertexLLM{PromptPrefix: "prefix"}
	parts := vertex.contentParts("describe", []File{
		{Name: "notes.txt", MIME: "text/plain", Data: []byte("hello")},
		{Name: "image.png", MIME: "image/png", Data: []byte{1, 2, 3}},
		{Name: "document.pdf", MIME: "application/pdf", Data: []byte{4, 5, 6}},
	})

	if len(parts) != 3 {
		t.Fatalf("len(parts) = %d, want 3", len(parts))
	}
	if parts[0].Text != "prefix" {
		t.Fatalf("prefix part = %q", parts[0].Text)
	}
	if !strings.Contains(parts[1].Text, "hello") || !strings.Contains(parts[1].Text, "document.pdf") {
		t.Fatalf("text context missing attachment details: %q", parts[1].Text)
	}
	if parts[2].InlineData == nil || parts[2].InlineData.MIMEType != "image/png" {
		t.Fatalf("image part = %#v", parts[2])
	}
}
