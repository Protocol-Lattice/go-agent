package main

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/base"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/cli"
	"github.com/universal-tool-calling-protocol/go-utcp/src/repository"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
	"github.com/universal-tool-calling-protocol/go-utcp/src/transports"
)

// TaskStatus represents the lifecycle of a goal or task.
type TaskStatus string

const (
	StatusPending    TaskStatus = "pending"
	StatusInProgress TaskStatus = "in_progress"
	StatusCompleted  TaskStatus = "completed"
	StatusFailed     TaskStatus = "failed"
)

// TaskRecord represents a persistent goal or task for Claw.
type TaskRecord struct {
	ID          int64      `json:"id"`
	Title       string     `json:"title"`
	Description string     `json:"description"`
	Status      TaskStatus `json:"status"`
	Priority    int        `json:"priority"`
	CreatedAt   time.Time  `json:"created_at"`
	UpdatedAt   time.Time  `json:"updated_at"`
}

// TaskStore manages persistence of tasks to a JSON file.
type TaskStore struct {
	filePath string
	mu       sync.RWMutex
	nextID   int64
	tasks    map[int64]TaskRecord
}

func NewTaskStore(filePath string) *TaskStore {
	store := &TaskStore{
		filePath: filePath,
		tasks:    make(map[int64]TaskRecord),
	}
	store.load()
	return store
}

func (s *TaskStore) load() {
	s.mu.Lock()
	defer s.mu.Unlock()

	data, err := os.ReadFile(s.filePath)
	if err != nil {
		if !os.IsNotExist(err) {
			fmt.Printf("Warning: failed to read task file: %v\n", err)
		}
		return
	}

	var records []TaskRecord
	if err := json.Unmarshal(data, &records); err != nil {
		fmt.Printf("Warning: failed to unmarshal task file: %v\n", err)
		return
	}

	for _, rec := range records {
		s.tasks[rec.ID] = rec
		if rec.ID > s.nextID {
			s.nextID = rec.ID
		}
	}
}

func (s *TaskStore) save() error {
	var records []TaskRecord
	for _, rec := range s.tasks {
		records = append(records, rec)
	}

	// Sort by ID for deterministic file output
	sort.Slice(records, func(i, j int) bool {
		return records[i].ID < records[j].ID
	})

	data, err := json.MarshalIndent(records, "", "  ")
	if err != nil {
		return err
	}

	return os.WriteFile(s.filePath, data, 0644)
}

// Tools returns UTCP tool definitions for managing tasks.
func (s *TaskStore) Tools() []tools.Tool {
	return []tools.Tool{
		{
			Name:        "tasks.create_goal",
			Description: "Create a new long-term goal or task for Claw to track.",
			Inputs: tools.ToolInputOutputSchema{
				Type: "object",
				Properties: map[string]any{
					"title":       map[string]any{"type": "string", "description": "Short title of the goal"},
					"description": map[string]any{"type": "string", "description": "Detailed description of what needs to be done"},
					"priority":    map[string]any{"type": "integer", "description": "Priority level (1-5), 5 being highest", "default": 1},
				},
				Required: []string{"title", "description"},
			},
			Handler: s.handleCreateGoal,
		},
		{
			Name:        "tasks.update_task",
			Description: "Update the status or description of an existing task.",
			Inputs: tools.ToolInputOutputSchema{
				Type: "object",
				Properties: map[string]any{
					"id":          map[string]any{"type": "integer", "description": "The unique ID of the task"},
					"status":      map[string]any{"type": "string", "enum": []string{"pending", "in_progress", "completed", "failed"}},
					"description": map[string]any{"type": "string", "description": "Updated description if needed"},
				},
				Required: []string{"id"},
			},
			Handler: s.handleUpdateTask,
		},
		{
			Name:        "tasks.list_active_tasks",
			Description: "Retrieve a list of all pending or in-progress tasks.",
			Inputs: tools.ToolInputOutputSchema{
				Type:       "object",
				Properties: map[string]any{},
			},
			Handler: s.handleListTasks,
		},
	}
}

func (s *TaskStore) handleCreateGoal(ctx context.Context, inputs map[string]any) (any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	title, _ := inputs["title"].(string)
	desc, _ := inputs["description"].(string)
	priorityVal, ok := inputs["priority"].(float64)
	priority := 1
	if ok {
		priority = int(priorityVal)
	}

	if title == "" || desc == "" {
		return nil, errors.New("title and description are required")
	}

	s.nextID++
	now := time.Now().UTC()
	task := TaskRecord{
		ID:          s.nextID,
		Title:       title,
		Description: desc,
		Status:      StatusPending,
		Priority:    priority,
		CreatedAt:   now,
		UpdatedAt:   now,
	}

	s.tasks[task.ID] = task
	if err := s.save(); err != nil {
		return nil, fmt.Errorf("failed to save task: %w", err)
	}

	return task, nil
}

func (s *TaskStore) handleUpdateTask(ctx context.Context, inputs map[string]any) (any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	idVal, ok := inputs["id"].(float64)
	if !ok {
		return nil, errors.New("valid task ID is required")
	}
	id := int64(idVal)

	task, exists := s.tasks[id]
	if !exists {
		return nil, fmt.Errorf("task with ID %d not found", id)
	}

	if status, ok := inputs["status"].(string); ok {
		task.Status = TaskStatus(status)
	}
	if desc, ok := inputs["description"].(string); ok {
		task.Description = desc
	}

	task.UpdatedAt = time.Now().UTC()
	s.tasks[id] = task

	if err := s.save(); err != nil {
		return nil, fmt.Errorf("failed to save task: %w", err)
	}

	return task, nil
}

func (s *TaskStore) handleListTasks(ctx context.Context, inputs map[string]any) (any, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	var active []TaskRecord
	for _, t := range s.tasks {
		if t.Status == StatusPending || t.Status == StatusInProgress {
			active = append(active, t)
		}
	}

	sort.Slice(active, func(i, j int) bool {
		if active[i].Priority != active[j].Priority {
			return active[i].Priority > active[j].Priority
		}
		return active[i].CreatedAt.Before(active[j].CreatedAt)
	})

	return active, nil
}

// RegisterAsUTCPProvider registers the TaskStore tools on the provided UTCP client.
func (s *TaskStore) RegisterAsUTCPProvider(ctx context.Context, client utcp.UtcpClientInterface) error {
	if client == nil {
		return fmt.Errorf("utcp client is nil")
	}

	providerName := "tasks"
	tp := &cli.CliProvider{
		BaseProvider: base.BaseProvider{
			Name:         providerName,
			ProviderType: base.ProviderCLI,
		},
	}

	transportsMap := client.GetTransports()
	if transportsMap == nil {
		return fmt.Errorf("utcp client transports map is nil")
	}

	existing := transportsMap[string(base.ProviderCLI)]
	var shim *taskCLITransport
	if maybe, ok := existing.(*taskCLITransport); ok {
		shim = maybe
	} else {
		shim = &taskCLITransport{inner: existing}
		transportsMap[string(base.ProviderCLI)] = shim
	}
	if shim.tools == nil {
		shim.tools = make(map[string][]tools.Tool)
	}
	shim.tools[tp.Name] = s.Tools()

	_, err := client.RegisterToolProvider(ctx, tp)
	return err
}

type taskCLITransport struct {
	inner repository.ClientTransport
	tools map[string][]tools.Tool
}

func (t *taskCLITransport) RegisterToolProvider(ctx context.Context, prov base.Provider) ([]tools.Tool, error) {
	p, ok := prov.(*cli.CliProvider)
	if !ok {
		if t.inner != nil {
			return t.inner.RegisterToolProvider(ctx, prov)
		}
		return nil, fmt.Errorf("unsupported provider type %T", prov)
	}
	if t.tools == nil {
		t.tools = make(map[string][]tools.Tool)
	}
	list, ok := t.tools[p.Name]
	if !ok {
		if t.inner != nil {
			return t.inner.RegisterToolProvider(ctx, prov)
		}
		return nil, fmt.Errorf("task tools not found for provider %s", p.Name)
	}
	return list, nil
}

func (t *taskCLITransport) DeregisterToolProvider(ctx context.Context, prov base.Provider) error {
	if p, ok := prov.(*cli.CliProvider); ok {
		if _, ok := t.tools[p.Name]; ok {
			delete(t.tools, p.Name)
			return nil
		}
	}
	if t.inner != nil {
		return t.inner.DeregisterToolProvider(ctx, prov)
	}
	return nil
}

func (t *taskCLITransport) CallTool(ctx context.Context, toolName string, args map[string]any, prov base.Provider, sessionID *string) (any, error) {
	if p, ok := prov.(*cli.CliProvider); ok {
		if list, ok := t.tools[p.Name]; ok {
			for _, tool := range list {
				if tool.Name == toolName || strings.HasSuffix(tool.Name, "."+toolName) {
					if tool.Handler == nil {
						return nil, fmt.Errorf("tool %s has no handler", toolName)
					}
					return tool.Handler(ctx, args)
				}
			}
		}
	}
	if t.inner != nil {
		return t.inner.CallTool(ctx, toolName, args, prov, sessionID)
	}
	return nil, fmt.Errorf("tool %s not found", toolName)
}

func (t *taskCLITransport) CallToolStream(ctx context.Context, toolName string, args map[string]any, prov base.Provider) (transports.StreamResult, error) {
	if t.inner != nil {
		return t.inner.CallToolStream(ctx, toolName, args, prov)
	}
	return nil, fmt.Errorf("streaming not supported for task tools")
}
