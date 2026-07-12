package workflow

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// ErrRunNotFound is returned when a workflow run does not exist in a RunStore.
var ErrRunNotFound = errors.New("workflow run not found")

// ErrRunAlreadyExists is returned when StartRun is called with an existing run ID.
var ErrRunAlreadyExists = errors.New("workflow run already exists")

// RunStatus describes the lifecycle state of a durable workflow run.
type RunStatus string

const (
	// RunStatusRunning identifies a run with pending work. A run remains running
	// after a node error so it can be resumed.
	RunStatusRunning RunStatus = "running"
	// RunStatusCompleted identifies a run that reached a terminal graph output.
	RunStatusCompleted RunStatus = "completed"
	// RunStatusFailed identifies a run that cannot make progress, such as an
	// incomplete join or an invalid persisted node reference.
	RunStatusFailed RunStatus = "failed"
)

// QueuedNode is a node invocation waiting to be executed as part of a durable
// workflow run. Node names are persisted instead of Node implementations so a
// run can be resumed after a process restart.
type QueuedNode struct {
	From  string `json:"from"`
	Node  string `json:"node"`
	Input any    `json:"input"`
}

// JoinState holds the accumulated predecessor outputs for a join node.
type JoinState struct {
	Values       map[string]any `json:"values"`
	FirstArrival time.Time      `json:"first_arrival"`
}

// RunState is the serializable execution state for a durable workflow run.
//
// Inputs, terminal outputs, join values, and ContextState must be JSON
// marshalable when using the supplied stores. Runs have at-least-once node
// execution semantics: if a process stops after a node runs but before its
// transition is saved, resuming the run invokes that node again. Side-effecting
// nodes should therefore be idempotent.
type RunState struct {
	ID           string               `json:"id"`
	SessionID    string               `json:"session_id"`
	InvocationID string               `json:"invocation_id"`
	Status       RunStatus            `json:"status"`
	Queue        []QueuedNode         `json:"queue"`
	Finals       []any                `json:"finals,omitempty"`
	JoinStates   map[string]JoinState `json:"join_states,omitempty"`
	ContextState map[string]any       `json:"context_state,omitempty"`
	Steps        int                  `json:"steps"`
	Result       any                  `json:"result,omitempty"`
	LastError    string               `json:"last_error,omitempty"`
	CreatedAt    time.Time            `json:"created_at"`
	UpdatedAt    time.Time            `json:"updated_at"`
}

// RunStore persists workflow execution state. Implementations must return an
// isolated snapshot from Load and retain an isolated snapshot passed to Create
// or Save, because workflow execution mutates RunState between checkpoints.
type RunStore interface {
	Create(ctx context.Context, state *RunState) error
	Load(ctx context.Context, runID string) (*RunState, error)
	Save(ctx context.Context, state *RunState) error
	Delete(ctx context.Context, runID string) error
}

// InMemoryRunStore is a JSON-round-tripping RunStore suited to tests and
// single-process development. It enforces the same serialization requirements
// as FileRunStore so that a run which works in tests can be persisted later.
type InMemoryRunStore struct {
	mu   sync.RWMutex
	runs map[string]*RunState
}

// NewInMemoryRunStore creates an empty in-memory durable workflow store.
func NewInMemoryRunStore() *InMemoryRunStore {
	return &InMemoryRunStore{runs: make(map[string]*RunState)}
}

func (s *InMemoryRunStore) Create(ctx context.Context, state *RunState) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	copyState, err := cloneRunState(state)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.runs == nil {
		s.runs = make(map[string]*RunState)
	}
	if _, exists := s.runs[copyState.ID]; exists {
		return fmt.Errorf("%w: %s", ErrRunAlreadyExists, copyState.ID)
	}
	s.runs[copyState.ID] = copyState
	return nil
}

func (s *InMemoryRunStore) Load(ctx context.Context, runID string) (*RunState, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	s.mu.RLock()
	state, ok := s.runs[runID]
	s.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("%w: %s", ErrRunNotFound, runID)
	}
	return cloneRunState(state)
}

func (s *InMemoryRunStore) Save(ctx context.Context, state *RunState) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	copyState, err := cloneRunState(state)
	if err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.runs[copyState.ID]; !exists {
		return fmt.Errorf("%w: %s", ErrRunNotFound, copyState.ID)
	}
	s.runs[copyState.ID] = copyState
	return nil
}

func (s *InMemoryRunStore) Delete(ctx context.Context, runID string) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, exists := s.runs[runID]; !exists {
		return fmt.Errorf("%w: %s", ErrRunNotFound, runID)
	}
	delete(s.runs, runID)
	return nil
}

// FileRunStore stores each workflow run as an atomically replaced JSON file.
// It is useful for local deployments and can be replaced with a database-backed
// RunStore in production.
type FileRunStore struct {
	dir string
	mu  sync.RWMutex
}

// NewFileRunStore creates a JSON-backed RunStore rooted at dir.
func NewFileRunStore(dir string) (*FileRunStore, error) {
	dir = strings.TrimSpace(dir)
	if dir == "" {
		return nil, errors.New("workflow run store directory is empty")
	}
	if err := os.MkdirAll(dir, 0o755); err != nil {
		return nil, fmt.Errorf("create workflow run store directory: %w", err)
	}
	return &FileRunStore{dir: dir}, nil
}

func (s *FileRunStore) Create(ctx context.Context, state *RunState) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if s == nil {
		return errors.New("nil workflow file run store")
	}
	path, err := s.path(stateID(state))
	if err != nil {
		return err
	}
	data, err := marshalRunState(state)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := os.Stat(path); err == nil {
		return fmt.Errorf("%w: %s", ErrRunAlreadyExists, state.ID)
	} else if !os.IsNotExist(err) {
		return fmt.Errorf("check workflow run %s: %w", state.ID, err)
	}
	return writeRunFile(path, data)
}

func (s *FileRunStore) Load(ctx context.Context, runID string) (*RunState, error) {
	if err := contextError(ctx); err != nil {
		return nil, err
	}
	if s == nil {
		return nil, errors.New("nil workflow file run store")
	}
	path, err := s.path(runID)
	if err != nil {
		return nil, err
	}

	s.mu.RLock()
	data, err := os.ReadFile(path)
	s.mu.RUnlock()
	if err != nil {
		if os.IsNotExist(err) {
			return nil, fmt.Errorf("%w: %s", ErrRunNotFound, runID)
		}
		return nil, fmt.Errorf("read workflow run %s: %w", runID, err)
	}

	var state RunState
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("decode workflow run %s: %w", runID, err)
	}
	if state.ID != runID {
		return nil, fmt.Errorf("workflow run file %s contains ID %q", runID, state.ID)
	}
	return &state, nil
}

func (s *FileRunStore) Save(ctx context.Context, state *RunState) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if s == nil {
		return errors.New("nil workflow file run store")
	}
	path, err := s.path(stateID(state))
	if err != nil {
		return err
	}
	data, err := marshalRunState(state)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := os.Stat(path); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%w: %s", ErrRunNotFound, state.ID)
		}
		return fmt.Errorf("check workflow run %s: %w", state.ID, err)
	}
	return writeRunFile(path, data)
}

func (s *FileRunStore) Delete(ctx context.Context, runID string) error {
	if err := contextError(ctx); err != nil {
		return err
	}
	if s == nil {
		return errors.New("nil workflow file run store")
	}
	path, err := s.path(runID)
	if err != nil {
		return err
	}

	s.mu.Lock()
	defer s.mu.Unlock()
	if err := os.Remove(path); err != nil {
		if os.IsNotExist(err) {
			return fmt.Errorf("%w: %s", ErrRunNotFound, runID)
		}
		return fmt.Errorf("delete workflow run %s: %w", runID, err)
	}
	return nil
}

func (s *FileRunStore) path(runID string) (string, error) {
	if err := validateRunID(runID); err != nil {
		return "", err
	}
	return filepath.Join(s.dir, runID+".json"), nil
}

func contextError(ctx context.Context) error {
	if ctx == nil {
		return nil
	}
	return ctx.Err()
}

func stateID(state *RunState) string {
	if state == nil {
		return ""
	}
	return state.ID
}

func validateRunID(runID string) error {
	if strings.TrimSpace(runID) == "" {
		return errors.New("workflow run ID is empty")
	}
	for _, r := range runID {
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') || (r >= '0' && r <= '9') || r == '-' || r == '_' || r == '.' {
			continue
		}
		return fmt.Errorf("workflow run ID %q contains unsupported character %q", runID, r)
	}
	return nil
}

func cloneRunState(state *RunState) (*RunState, error) {
	data, err := marshalRunState(state)
	if err != nil {
		return nil, err
	}
	var clone RunState
	if err := json.Unmarshal(data, &clone); err != nil {
		return nil, fmt.Errorf("decode workflow run state: %w", err)
	}
	return &clone, nil
}

func marshalRunState(state *RunState) ([]byte, error) {
	if state == nil {
		return nil, errors.New("nil workflow run state")
	}
	if err := validateRunID(state.ID); err != nil {
		return nil, err
	}
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("encode workflow run %s: %w", state.ID, err)
	}
	return data, nil
}

func writeRunFile(path string, data []byte) (err error) {
	tmp, err := os.CreateTemp(filepath.Dir(path), "."+filepath.Base(path)+"-*")
	if err != nil {
		return fmt.Errorf("create workflow run temp file: %w", err)
	}
	tmpPath := tmp.Name()
	defer func() {
		if tmp != nil {
			_ = tmp.Close()
		}
		if err != nil {
			_ = os.Remove(tmpPath)
		}
	}()

	if _, err = tmp.Write(data); err != nil {
		return fmt.Errorf("write workflow run temp file: %w", err)
	}
	if err = tmp.Sync(); err != nil {
		return fmt.Errorf("sync workflow run temp file: %w", err)
	}
	if err = tmp.Close(); err != nil {
		return fmt.Errorf("close workflow run temp file: %w", err)
	}
	tmp = nil
	if err = os.Rename(tmpPath, path); err != nil {
		return fmt.Errorf("replace workflow run file: %w", err)
	}
	return nil
}
