package mcp

import (
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"strings"
	"sync"
)

// StdioConfig describes how to spawn an MCP server using the stdio transport.
type StdioConfig struct {
	Command string
	Args    []string
	Dir     string
	Env     []string

	// Stderr, when provided, receives the standard error stream of the
	// spawned server process. Defaults to os.Stderr if nil.
	Stderr io.Writer

	Options Options
}

// NewStdioClient starts the configured command and binds the stdin/stdout pipes
// to the MCP client transport. The caller is responsible for invoking Close on
// the returned client when the session should end. Any failure during
// initialisation stops the process and returns an error.
func NewStdioClient(ctx context.Context, cfg StdioConfig) (*Client, error) {
	if strings.TrimSpace(cfg.Command) == "" {
		return nil, errors.New("mcp: stdio command is required")
	}

	cmd := exec.CommandContext(ctx, cfg.Command, cfg.Args...)
	cmd.Dir = cfg.Dir
	if len(cfg.Env) > 0 {
		cmd.Env = append(os.Environ(), cfg.Env...)
	}
	if cfg.Stderr != nil {
		cmd.Stderr = cfg.Stderr
	} else {
		cmd.Stderr = os.Stderr
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: stdout pipe: %w", err)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: stdin pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, fmt.Errorf("mcp: start command: %w", err)
	}

	transport := newStdioTransport(stdin, stdout)
	client, err := NewClient(ctx, transport, cfg.Options)
	if err != nil {
		transport.Close()
		_ = cmd.Process.Kill()
		_ = cmd.Wait()
		return nil, err
	}

	// Ensure the transport is closed when the process exits to unblock any
	// pending reads.
	var once sync.Once
	go func() {
		_ = cmd.Wait()
		once.Do(func() {
			_ = transport.Close()
		})
	}()

	return client, nil
}
