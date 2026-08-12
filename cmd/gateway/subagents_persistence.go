package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	utcp "github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/plugins/codemode"
)

type persistedSubagent struct {
	ID           string   `json:"id"`
	Name         string   `json:"name"`
	Description  string   `json:"description"`
	SystemPrompt string   `json:"system_prompt"`
	Provider     string   `json:"provider"`
	Model        string   `json:"model"`
	Tools        []string `json:"tools"`
}

type persistedSubagentStore struct {
	Subagents []persistedSubagent `json:"subagents"`
}

func loadPersistedSubagents(ctx context.Context, client utcp.UtcpClientInterface, defaultProvider, defaultModel string) (map[string]*agent.Agent, error) {
	path := "cmd/gateway/web/subagents.json"
	data, err := os.ReadFile(path)
	if os.IsNotExist(err) {
		return map[string]*agent.Agent{}, nil
	}
	if err != nil {
		return nil, err
	}

	var store persistedSubagentStore
	if err := json.Unmarshal(data, &store); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}

	result := make(map[string]*agent.Agent, len(store.Subagents))
	for _, cfg := range store.Subagents {
		name := strings.TrimSpace(cfg.Name)
		if name == "" {
			name = strings.TrimSpace(cfg.ID)
		}
		if name == "" {
			continue
		}

		provider := strings.ToLower(strings.TrimSpace(cfg.Provider))
		if provider == "" {
			provider = defaultProvider
		}
		modelName := strings.TrimSpace(cfg.Model)
		if modelName == "" || modelName == "default" {
			modelName = defaultModel
		}

		var model models.Agent
		if provider == "dummy" {
			model = models.NewDummyLLM(modelName)
		} else {
			model, err = newModel(ctx, provider, modelName)
			if err != nil {
				return nil, fmt.Errorf("load subagent %q: create model: %w", name, err)
			}
		}

		prompt := strings.TrimSpace(cfg.SystemPrompt)
		if prompt == "" {
			prompt = strings.TrimSpace(cfg.Description)
		}
		if prompt == "" {
			prompt = "You are a specialist sub-agent."
		}

		mem := memory.NewSessionMemory(memory.NewMemoryBankWithStore(memory.NewInMemoryStore()), *flagContext)
		sa, err := agent.New(agent.Options{
			Model:        model,
			Memory:       mem,
			SystemPrompt: prompt,
			ContextLimit: *flagContext,
			UTCPClient:   client,
			CodeMode:     codemode.NewCodeModeUTCP(client, model),
		})
		if err != nil {
			return nil, fmt.Errorf("load subagent %q: %w", name, err)
		}

		if client != nil {
			description := strings.TrimSpace(cfg.Description)
			if description == "" {
				description = "Persisted go-agent sub-agent"
			}
			if err := sa.RegisterAsUTCPProvider(ctx, client, name, description); err != nil {
				return nil, fmt.Errorf("load subagent %q: register UTCP provider: %w", name, err)
			}
		}
		result[name] = sa
	}
	return result, nil
}
