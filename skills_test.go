package agent

import (
	"context"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
)

func TestLoadSkillsDiscoversSupportedLayouts(t *testing.T) {
	root := t.TempDir()
	writeSkillTestFile(t, filepath.Join(root, "release.md"), `---
name: release-checklist
description: "Ship safely"
---
Run the release checklist.`)
	writeSkillTestFile(t, filepath.Join(root, "code-review", "SKILL.md"), "Review every changed line.")
	writeSkillTestFile(t, filepath.Join(root, "code-review", "notes.md"), "This is not a skill document.")

	skills, err := LoadSkills(root)
	if err != nil {
		t.Fatalf("LoadSkills() error = %v", err)
	}
	if len(skills) != 2 {
		t.Fatalf("LoadSkills() returned %d skills, want 2: %#v", len(skills), skills)
	}

	byName := make(map[string]Skill, len(skills))
	for _, skill := range skills {
		byName[skill.Name] = skill
	}

	release, ok := byName["release-checklist"]
	if !ok {
		t.Fatalf("front-matter name was not loaded: %#v", skills)
	}
	if release.Description != "Ship safely" || release.Instructions != "Run the release checklist." {
		t.Fatalf("unexpected release skill: %#v", release)
	}

	review, ok := byName["code-review"]
	if !ok {
		t.Fatalf("nested SKILL.md was not loaded: %#v", skills)
	}
	if review.Instructions != "Review every changed line." {
		t.Fatalf("unexpected review instructions: %q", review.Instructions)
	}
}

func TestLoadSkillsAllowsMissingDirectory(t *testing.T) {
	skills, err := LoadSkills(filepath.Join(t.TempDir(), "missing"))
	if err != nil {
		t.Fatalf("LoadSkills() error = %v", err)
	}
	if len(skills) != 0 {
		t.Fatalf("LoadSkills() returned %#v, want no skills", skills)
	}
}

func TestAgentAddsSkillsToAllPromptPaths(t *testing.T) {
	root := t.TempDir()
	writeSkillTestFile(t, filepath.Join(root, "style", "SKILL.md"), "Always answer in haiku.")

	model := &skillPromptModel{}
	a, err := New(Options{
		Model:     model,
		Memory:    memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SkillsDir: root,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if _, err := a.Generate(context.Background(), "skills", "Say hello"); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if _, err := a.GenerateWithFiles(context.Background(), "skills", "Read this", []models.File{{Name: "notes.txt", MIME: "text/plain", Data: []byte("hello")}}); err != nil {
		t.Fatalf("GenerateWithFiles() error = %v", err)
	}
	stream, err := a.GenerateStream(context.Background(), "skills", "Stream hello")
	if err != nil {
		t.Fatalf("GenerateStream() error = %v", err)
	}
	for range stream {
	}

	if len(model.prompts) != 3 {
		t.Fatalf("model received %d prompts, want 3", len(model.prompts))
	}
	for i, prompt := range model.prompts {
		if !strings.Contains(prompt, "Local project skills:") || !strings.Contains(prompt, "Always answer in haiku.") {
			t.Fatalf("prompt %d does not contain the local skill:\n%s", i, prompt)
		}
	}
}

func TestAgentReloadSkills(t *testing.T) {
	root := t.TempDir()
	path := filepath.Join(root, "writing.md")
	writeSkillTestFile(t, path, "Use short sentences.")

	a, err := New(Options{
		Model:     &skillPromptModel{},
		Memory:    memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SkillsDir: root,
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	writeSkillTestFile(t, path, "Use complete sentences.")
	if err := a.ReloadSkills(); err != nil {
		t.Fatalf("ReloadSkills() error = %v", err)
	}

	skills := a.Skills()
	if len(skills) != 1 || skills[0].Instructions != "Use complete sentences." {
		t.Fatalf("ReloadSkills() loaded %#v, want updated skill", skills)
	}
}

func TestAgentAddsSkillsToToolPlanningPrompt(t *testing.T) {
	root := t.TempDir()
	writeSkillTestFile(t, filepath.Join(root, "tools.md"), "Use the approved deployment procedure.")

	model := &skillPromptModel{}
	a, err := New(Options{
		Model:     model,
		Memory:    memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), 8),
		SkillsDir: root,
		Tools: []Tool{&stubTool{spec: ToolSpec{
			Name:        "deploy",
			Description: "Deploy the application.",
			InputSchema: map[string]any{"type": "object"},
		}}},
	})
	if err != nil {
		t.Fatalf("New() error = %v", err)
	}

	if _, err := a.Generate(context.Background(), "skills", "deploy the application"); err != nil {
		t.Fatalf("Generate() error = %v", err)
	}
	if len(model.prompts) < 1 {
		t.Fatal("tool planner did not call the model")
	}
	if plannerPrompt := model.prompts[0]; !strings.Contains(plannerPrompt, "SYSTEM INSTRUCTIONS:") || !strings.Contains(plannerPrompt, "Use the approved deployment procedure.") {
		t.Fatalf("tool planner prompt does not contain local skills:\n%s", plannerPrompt)
	}
}

func writeSkillTestFile(t *testing.T, path, content string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
		t.Fatalf("MkdirAll(%q): %v", path, err)
	}
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("WriteFile(%q): %v", path, err)
	}
}

type skillPromptModel struct {
	prompts []string
}

func (m *skillPromptModel) Generate(_ context.Context, prompt string) (any, error) {
	m.prompts = append(m.prompts, prompt)
	if strings.Contains(prompt, "You are an agentic UTCP tool execution loop") {
		return `{"use_tool":false,"final_answer":"ok"}`, nil
	}
	return "ok", nil
}

func (m *skillPromptModel) GenerateWithFiles(_ context.Context, prompt string, _ []models.File) (any, error) {
	m.prompts = append(m.prompts, prompt)
	if strings.Contains(prompt, "You are an agentic UTCP tool execution loop") {
		return `{"use_tool":false,"final_answer":"ok"}`, nil
	}
	return "ok", nil
}

func (m *skillPromptModel) GenerateStream(_ context.Context, prompt string) (<-chan models.StreamChunk, error) {
	m.prompts = append(m.prompts, prompt)
	stream := make(chan models.StreamChunk, 1)
	stream <- models.StreamChunk{Delta: "ok", FullText: "ok", Done: true}
	close(stream)
	return stream, nil
}
