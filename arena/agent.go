package arena

import (
	"context"
	"fmt"

	agent "github.com/Protocol-Lattice/go-agent"
)

// AgentRunner adapts the native go-agent Agent to the Arena Runner interface.
// SessionID is used to isolate memory between benchmark tasks.
type AgentRunner struct {
	Agent     *agent.Agent
	SessionID string
}

// Run executes the task through Agent.Generate. The task name is included in
// the session id when SessionID is empty, preventing benchmark tasks from
// accidentally sharing conversation memory.
func (r AgentRunner) Run(ctx context.Context, task Task) (RunOutput, error) {
	if r.Agent == nil {
		return RunOutput{}, fmt.Errorf("agent runner requires an agent")
	}
	sessionID := r.SessionID
	if sessionID == "" {
		sessionID = "arena:" + task.Name
	}
	value, err := r.Agent.Generate(ctx, sessionID, task.Input)
	if err != nil {
		return RunOutput{}, err
	}
	return RunOutput{Output: fmt.Sprint(value)}, nil
}
