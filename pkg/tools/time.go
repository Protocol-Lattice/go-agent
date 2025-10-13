package tools

import (
	"context"
	"time"
)

// TimeTool reports the current UTC time in RFC3339 format.
type TimeTool struct{}

func (t *TimeTool) Name() string        { return "time" }
func (t *TimeTool) Description() string { return "Returns the current UTC time." }

func (t *TimeTool) Run(_ context.Context, _ string) (string, error) {
	return time.Now().UTC().Format(time.RFC3339), nil
}
