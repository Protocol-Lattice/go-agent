package client

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

var (
	// ErrMethodNotFound mirrors the JSON-RPC error when a server does not implement a capability.
	ErrMethodNotFound = errors.New("mcp: method not found")
	errClientClosed   = errors.New("mcp: client closed")
)

// Tool mirrors a tool definition returned by an MCP server.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// Content captures a single content item from a tool invocation.
type Content struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	URI      string `json:"uri,omitempty"`
	MimeType string `json:"mimeType,omitempty"`
}

// Resource mirrors the metadata for an MCP resource advertisement.
type Resource struct {
	URI         string `json:"uri"`
	Name        string `json:"name,omitempty"`
	Description string `json:"description,omitempty"`
}

// ResourceData captures the payload for a resource fetch.
type ResourceData struct {
	URI      string `json:"uri"`
	MimeType string `json:"mimeType,omitempty"`
	Text     string `json:"text,omitempty"`
}

// Options controls how a stdio MCP session is launched.
type Options struct {
	Command string
	Args    []string
	Env     []string
}

// Session represents an active MCP connection.
type Session struct {
	cmd    *exec.Cmd
	stdin  io.WriteCloser
	stdout io.ReadCloser

	writeMu   sync.Mutex
	pendingMu sync.Mutex
	pending   map[int64]chan rpcResponse
	counter   atomic.Int64

	done chan struct{}
	err  error

	serverName    string
	serverVersion string

	supportsResources bool
}

type rpcRequest struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      *int64      `json:"id,omitempty"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type rpcError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

type rpcResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *int64          `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
}

// Start launches a new stdio MCP session.
func Start(ctx context.Context, opts Options) (*Session, error) {
	if strings.TrimSpace(opts.Command) == "" {
		return nil, errors.New("mcp: command is required")
	}
	cmd := exec.CommandContext(ctx, opts.Command, opts.Args...)
	if len(opts.Env) > 0 {
		cmd.Env = opts.Env
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, fmt.Errorf("mcp: stdin pipe: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		stdin.Close()
		return nil, fmt.Errorf("mcp: stdout pipe: %w", err)
	}
	if err := cmd.Start(); err != nil {
		stdin.Close()
		stdout.Close()
		return nil, fmt.Errorf("mcp: start command: %w", err)
	}

	session := &Session{
		cmd:     cmd,
		stdin:   stdin,
		stdout:  stdout,
		pending: make(map[int64]chan rpcResponse),
		done:    make(chan struct{}),
	}

	go session.readLoop()
	go session.waitForExit()

	if err := session.initialize(ctx); err != nil {
		session.Close(ctx)
		return nil, err
	}

	return session, nil
}

func (s *Session) initialize(ctx context.Context) error {
	params := map[string]interface{}{
		"protocolVersion": "2024-11-05",
		"clientInfo": map[string]string{
			"name":    "go-agent-demo",
			"version": "dev",
		},
		"capabilities": map[string]interface{}{
			"roots":     map[string]interface{}{},
			"tools":     map[string]interface{}{},
			"resources": map[string]interface{}{},
		},
	}
	var result struct {
		ServerInfo struct {
			Name    string `json:"name"`
			Version string `json:"version"`
		} `json:"serverInfo"`
		Capabilities struct {
			Resources map[string]interface{} `json:"resources"`
		} `json:"capabilities"`
	}
	if err := s.call(ctx, "initialize", params, &result); err != nil {
		return fmt.Errorf("mcp: initialize: %w", err)
	}
	s.serverName = result.ServerInfo.Name
	s.serverVersion = result.ServerInfo.Version
	if result.Capabilities.Resources != nil {
		s.supportsResources = true
	}
	// notify initialized but ignore errors.
	_ = s.notify(ctx, "initialized", map[string]interface{}{})
	return nil
}

func (s *Session) readLoop() {
	reader := bufio.NewReader(s.stdout)
	for {
		headers := make(map[string]string)
		for {
			line, err := reader.ReadString('\n')
			if err != nil {
				s.finish(err)
				return
			}
			line = strings.TrimRight(line, "\r\n")
			if line == "" {
				break
			}
			parts := strings.SplitN(line, ":", 2)
			if len(parts) == 2 {
				headers[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
			}
		}
		lengthStr, ok := headers["Content-Length"]
		if !ok {
			s.finish(errors.New("mcp: missing Content-Length header"))
			return
		}
		length, err := strconv.Atoi(lengthStr)
		if err != nil {
			s.finish(fmt.Errorf("mcp: invalid Content-Length: %w", err))
			return
		}
		body := make([]byte, length)
		if _, err := io.ReadFull(reader, body); err != nil {
			s.finish(fmt.Errorf("mcp: read body: %w", err))
			return
		}
		var resp rpcResponse
		if err := json.Unmarshal(body, &resp); err != nil {
			s.finish(fmt.Errorf("mcp: decode response: %w", err))
			return
		}
		if resp.ID == nil {
			continue
		}
		s.pendingMu.Lock()
		ch, ok := s.pending[*resp.ID]
		if ok {
			delete(s.pending, *resp.ID)
		}
		s.pendingMu.Unlock()
		if ok {
			select {
			case ch <- resp:
			default:
			}
		}
	}
}

func (s *Session) waitForExit() {
	err := s.cmd.Wait()
	if err != nil {
		s.finish(err)
	} else {
		s.finish(io.EOF)
	}
}

func (s *Session) finish(err error) {
	s.pendingMu.Lock()
	if s.err == nil {
		s.err = err
	}
	for id, ch := range s.pending {
		delete(s.pending, id)
		select {
		case ch <- rpcResponse{Error: &rpcError{Message: errClientClosed.Error()}}:
		default:
		}
	}
	s.pendingMu.Unlock()
	select {
	case <-s.done:
	default:
		close(s.done)
	}
}

func (s *Session) notify(ctx context.Context, method string, params interface{}) error {
	req := rpcRequest{JSONRPC: "2.0", Method: method, Params: params}
	payload, err := json.Marshal(req)
	if err != nil {
		return err
	}
	return s.writeFrame(payload)
}

func (s *Session) call(ctx context.Context, method string, params interface{}, result interface{}) error {
	id := s.counter.Add(1)
	req := rpcRequest{JSONRPC: "2.0", ID: &id, Method: method, Params: params}
	payload, err := json.Marshal(req)
	if err != nil {
		return err
	}
	respCh := make(chan rpcResponse, 1)
	s.pendingMu.Lock()
	s.pending[id] = respCh
	s.pendingMu.Unlock()
	if err := s.writeFrame(payload); err != nil {
		s.removePending(id)
		return err
	}
	select {
	case <-ctx.Done():
		s.removePending(id)
		return ctx.Err()
	case <-s.done:
		s.removePending(id)
		if s.err != nil {
			return s.err
		}
		return errClientClosed
	case resp := <-respCh:
		if resp.Error != nil {
			if resp.Error.Code == -32601 {
				return ErrMethodNotFound
			}
			return errors.New(resp.Error.Message)
		}
		if result != nil && len(resp.Result) > 0 {
			if err := json.Unmarshal(resp.Result, result); err != nil {
				return err
			}
		}
		return nil
	}
}

func (s *Session) removePending(id int64) {
	s.pendingMu.Lock()
	delete(s.pending, id)
	s.pendingMu.Unlock()
}

func (s *Session) writeFrame(payload []byte) error {
	s.writeMu.Lock()
	defer s.writeMu.Unlock()
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(payload))
	if _, err := io.WriteString(s.stdin, header); err != nil {
		return err
	}
	if _, err := s.stdin.Write(payload); err != nil {
		return err
	}
	return nil
}

// Close terminates the MCP session.
func (s *Session) Close(ctx context.Context) error {
	if s == nil {
		return nil
	}
	// Attempt graceful shutdown.
	shutdownCtx, cancel := context.WithTimeout(ctx, 2*time.Second)
	defer cancel()
	_ = s.call(shutdownCtx, "shutdown", map[string]interface{}{}, nil)
	_ = s.notify(shutdownCtx, "exit", map[string]interface{}{})
	select {
	case <-s.done:
	case <-shutdownCtx.Done():
	}
	if s.cmd.ProcessState == nil || !s.cmd.ProcessState.Exited() {
		_ = s.cmd.Process.Kill()
	}
	return nil
}

// Name returns the MCP server name.
func (s *Session) Name() string {
	if s == nil {
		return ""
	}
	if s.serverName == "" {
		return "mcp-server"
	}
	return s.serverName
}

// ListTools retrieves the available tool definitions.
func (s *Session) ListTools(ctx context.Context) ([]Tool, error) {
	var resp struct {
		Tools []Tool `json:"tools"`
	}
	if err := s.call(ctx, "tools/list", map[string]interface{}{}, &resp); err != nil {
		return nil, err
	}
	return append([]Tool(nil), resp.Tools...), nil
}

// CallTool invokes a named tool on the remote server.
func (s *Session) CallTool(ctx context.Context, name string, arguments map[string]interface{}) ([]Content, error) {
	if arguments == nil {
		arguments = map[string]interface{}{}
	}
	params := map[string]interface{}{
		"name":      name,
		"arguments": arguments,
	}
	var resp struct {
		Content []Content `json:"content"`
	}
	if err := s.call(ctx, "tools/call", params, &resp); err != nil {
		return nil, err
	}
	return append([]Content(nil), resp.Content...), nil
}

// ListResources enumerates server resources if supported.
func (s *Session) ListResources(ctx context.Context) ([]Resource, error) {
	if !s.supportsResources {
		return nil, ErrMethodNotFound
	}
	var resp struct {
		Resources []Resource `json:"resources"`
	}
	if err := s.call(ctx, "resources/list", map[string]interface{}{}, &resp); err != nil {
		if errors.Is(err, ErrMethodNotFound) {
			s.supportsResources = false
			return nil, ErrMethodNotFound
		}
		return nil, err
	}
	return append([]Resource(nil), resp.Resources...), nil
}

// ReadResource fetches the contents of a resource URI.
func (s *Session) ReadResource(ctx context.Context, uri string) (ResourceData, error) {
	if !s.supportsResources {
		return ResourceData{}, ErrMethodNotFound
	}
	params := map[string]interface{}{
		"uri": uri,
	}
	var resp struct {
		Contents []struct {
			URI      string `json:"uri"`
			MimeType string `json:"mimeType"`
			Text     string `json:"text"`
		} `json:"contents"`
	}
	if err := s.call(ctx, "resources/read", params, &resp); err != nil {
		if errors.Is(err, ErrMethodNotFound) {
			s.supportsResources = false
			return ResourceData{}, ErrMethodNotFound
		}
		return ResourceData{}, err
	}
	if len(resp.Contents) == 0 {
		return ResourceData{}, nil
	}
	item := resp.Contents[0]
	return ResourceData{URI: item.URI, MimeType: item.MimeType, Text: item.Text}, nil
}

// Err returns the session terminal error if any.
func (s *Session) Err() error {
	return s.err
}
