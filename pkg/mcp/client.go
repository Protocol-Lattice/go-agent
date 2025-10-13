// Package mcp implements a lightweight Model Context Protocol client that is
// compatible with the mark3lab reference implementation. It focuses on the
// tooling surface area (listing and invoking tools) required by the agent
// runtime.
package mcp

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
)

const (
	// protocolVersion loosely follows the Model Context Protocol releases. The
	// value is intentionally flexible because servers may choose to accept a
	// range of versions. A sensible default keeps the client working out of
	// the box while still allowing tests to override it.
	protocolVersion = "2024-05-01"
)

// ClientInfo describes the calling application when establishing an MCP
// session.
type ClientInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// Options control how the MCP client initialises the remote server.
type Options struct {
	ClientInfo      ClientInfo
	Capabilities    map[string]any
	ProtocolVersion string
}

// ToolDefinition mirrors the subset of the MCP tool schema that the runtime
// requires.
type ToolDefinition struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema,omitempty"`
}

// Content represents a single content part returned from a tool invocation.
type Content struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	Data     json.RawMessage `json:"data,omitempty"`
	MimeType string          `json:"mimeType,omitempty"`
	Href     string          `json:"href,omitempty"`
}

// CallResult captures the structured output of an MCP tool invocation.
type CallResult struct {
	Content []Content `json:"content"`
	IsError bool      `json:"isError,omitempty"`
}

// Text concatenates text parts within the result. Multiple segments are joined
// with a newline to preserve ordering while offering a consumable string.
func (r CallResult) Text() string {
	var segments []string
	for _, part := range r.Content {
		if part.Type != "text" {
			continue
		}
		if trimmed := strings.TrimSpace(part.Text); trimmed != "" {
			segments = append(segments, trimmed)
		}
	}
	return strings.Join(segments, "\n")
}

// JSON returns the first JSON payload embedded inside the call result. The
// output is pretty printed for readability. When no JSON payload exists an
// empty string is returned.
func (r CallResult) JSON() string {
	for _, part := range r.Content {
		if part.Type != "json" || len(part.Data) == 0 {
			continue
		}
		var buf bytes.Buffer
		if err := json.Indent(&buf, part.Data, "", "  "); err != nil {
			return string(part.Data)
		}
		return buf.String()
	}
	return ""
}

// PrimaryText returns the textual interpretation of the result. It prefers the
// aggregated text segments but will fall back to the JSON payload if available.
func (r CallResult) PrimaryText() string {
	if txt := r.Text(); txt != "" {
		return txt
	}
	return r.JSON()
}

// Transport is the underlying message transport used by the MCP client.
type Transport interface {
	Send(ctx context.Context, payload []byte) error
	Receive(ctx context.Context) ([]byte, error)
	Close() error
}

// Client implements a small subset of the Model Context Protocol focused on
// listing and invoking tools.
type Client struct {
	transport    Transport
	info         ClientInfo
	capabilities map[string]any
	protoVersion string

	idCounter atomic.Uint64
	mu        sync.Mutex
	closed    atomic.Bool

	serverInfo ServerInfo
}

// ServerInfo represents the metadata returned by the MCP server during the
// initialise handshake.
type ServerInfo struct {
	Name    string `json:"name"`
	Version string `json:"version"`
}

// NewClient creates an MCP client using the provided transport. The function
// immediately performs the initialise handshake and will close the transport if
// the handshake fails.
func NewClient(ctx context.Context, transport Transport, opts Options) (*Client, error) {
	if transport == nil {
		return nil, errors.New("mcp: transport is nil")
	}

	info := opts.ClientInfo
	if strings.TrimSpace(info.Name) == "" {
		info.Name = "go-agent-development-kit"
	}
	if strings.TrimSpace(info.Version) == "" {
		info.Version = "dev"
	}

	caps := opts.Capabilities
	if caps == nil {
		caps = map[string]any{
			"tools": map[string]bool{
				"list": true,
				"call": true,
			},
		}
	}

	proto := opts.ProtocolVersion
	if strings.TrimSpace(proto) == "" {
		proto = protocolVersion
	}

	client := &Client{
		transport:    transport,
		info:         info,
		capabilities: caps,
		protoVersion: proto,
	}

	if err := client.initialize(ctx); err != nil {
		transport.Close()
		return nil, err
	}

	return client, nil
}

// Close releases the underlying transport. Close is idempotent.
func (c *Client) Close() error {
	if c == nil {
		return nil
	}
	if c.closed.Load() {
		return nil
	}
	c.closed.Store(true)
	return c.transport.Close()
}

// Server returns metadata about the remote MCP server. The information is
// captured during the initialise handshake.
func (c *Client) Server() ServerInfo {
	if c == nil {
		return ServerInfo{}
	}
	return c.serverInfo
}

// ListTools retrieves the complete list of tools exposed by the MCP server.
// The function transparently follows pagination cursors if the server elects to
// paginate the response.
func (c *Client) ListTools(ctx context.Context) ([]ToolDefinition, error) {
	if err := c.ensureOpen(); err != nil {
		return nil, err
	}

	var (
		cursor string
		tools  []ToolDefinition
	)

	for {
		params := map[string]any{}
		if cursor != "" {
			params["cursor"] = cursor
		}

		var resp struct {
			Tools      []ToolDefinition `json:"tools"`
			NextCursor string           `json:"nextCursor,omitempty"`
		}

		if err := c.call(ctx, "tools/list", params, &resp); err != nil {
			return nil, err
		}

		tools = append(tools, resp.Tools...)
		if strings.TrimSpace(resp.NextCursor) == "" {
			break
		}
		cursor = resp.NextCursor
	}

	return tools, nil
}

// CallTool invokes a named tool on the MCP server. The resulting CallResult is
// returned alongside any transport level error. If the server indicates that
// the invocation failed the function returns an error that includes the tool's
// textual output.
func (c *Client) CallTool(ctx context.Context, name string, arguments map[string]any) (CallResult, error) {
	if err := c.ensureOpen(); err != nil {
		return CallResult{}, err
	}
	if strings.TrimSpace(name) == "" {
		return CallResult{}, errors.New("mcp: tool name is required")
	}

	params := map[string]any{
		"name": name,
	}
	if len(arguments) > 0 {
		params["arguments"] = arguments
	}

	var result CallResult
	if err := c.call(ctx, "tools/call", params, &result); err != nil {
		return CallResult{}, err
	}

	if result.IsError {
		message := strings.TrimSpace(result.PrimaryText())
		if message == "" {
			message = "tool reported an error"
		}
		return result, fmt.Errorf("mcp: tool %s failed: %s", name, message)
	}

	return result, nil
}

// Shutdown notifies the server that the client intends to terminate the
// session. Servers may choose to perform additional cleanup when this request
// is received. Shutdown is best effort and errors are returned to the caller so
// they can be logged.
func (c *Client) Shutdown(ctx context.Context) error {
	if err := c.ensureOpen(); err != nil {
		return err
	}
	// A server may respond with an empty result which is perfectly valid for
	// shutdown. We therefore ignore the decoded payload.
	return c.call(ctx, "shutdown", map[string]any{}, &struct{}{})
}

// ensureOpen validates that the client has not been closed.
func (c *Client) ensureOpen() error {
	if c == nil {
		return errors.New("mcp: client is nil")
	}
	if c.closed.Load() {
		return errors.New("mcp: client has been closed")
	}
	return nil
}

func (c *Client) initialize(ctx context.Context) error {
	params := map[string]any{
		"protocolVersion": c.protoVersion,
		"clientInfo":      c.info,
		"capabilities":    c.capabilities,
	}

	var resp struct {
		ProtocolVersion string     `json:"protocolVersion"`
		ServerInfo      ServerInfo `json:"serverInfo"`
	}

	if err := c.call(ctx, "initialize", params, &resp); err != nil {
		return err
	}

	c.serverInfo = resp.ServerInfo
	return nil
}

type request struct {
	JSONRPC string      `json:"jsonrpc"`
	ID      string      `json:"id"`
	Method  string      `json:"method"`
	Params  interface{} `json:"params,omitempty"`
}

type responseEnvelope struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      *string         `json:"id,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *rpcError       `json:"error,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type rpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (c *Client) call(ctx context.Context, method string, params any, out any) error {
	if ctx == nil {
		ctx = context.Background()
	}

	id := strconv.FormatUint(c.idCounter.Add(1), 10)
	payload, err := json.Marshal(request{JSONRPC: "2.0", ID: id, Method: method, Params: params})
	if err != nil {
		return fmt.Errorf("mcp: marshal request: %w", err)
	}

	c.mu.Lock()
	defer c.mu.Unlock()

	if err := ctx.Err(); err != nil {
		return err
	}
	if c.closed.Load() {
		return errors.New("mcp: client has been closed")
	}

	if err := c.transport.Send(ctx, payload); err != nil {
		return err
	}

	for {
		msg, err := c.transport.Receive(ctx)
		if err != nil {
			return err
		}

		var env responseEnvelope
		if err := json.Unmarshal(msg, &env); err != nil {
			return fmt.Errorf("mcp: decode response: %w", err)
		}

		if env.Method != "" {
			// Notification – ignore for now but keep looping to find the
			// response that matches our request id.
			continue
		}

		if env.ID == nil || *env.ID != id {
			// Not the response we are waiting for. Keep looping until we
			// encounter the correct ID.
			continue
		}

		if env.Error != nil {
			return errors.New(env.Error.Message)
		}

		if out != nil && len(env.Result) > 0 {
			if err := json.Unmarshal(env.Result, out); err != nil {
				return fmt.Errorf("mcp: decode result: %w", err)
			}
		}
		return nil
	}
}

// ----------------------------------------------------------------------------
// Transport implementations

type stdioTransport struct {
	reader       *bufio.Reader
	writer       io.Writer
	stdinCloser  io.Closer
	stdoutCloser io.Closer
	writeMu      sync.Mutex
}

func newStdioTransport(stdin io.WriteCloser, stdout io.ReadCloser) Transport {
	return &stdioTransport{
		reader:       bufio.NewReader(stdout),
		writer:       stdin,
		stdinCloser:  stdin,
		stdoutCloser: stdout,
	}
}

func (t *stdioTransport) Send(ctx context.Context, payload []byte) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(payload))

	t.writeMu.Lock()
	defer t.writeMu.Unlock()

	if _, err := io.WriteString(t.writer, header); err != nil {
		return err
	}
	_, err := t.writer.Write(payload)
	return err
}

func (t *stdioTransport) Receive(ctx context.Context) ([]byte, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	length, err := t.readContentLength()
	if err != nil {
		return nil, err
	}

	buf := make([]byte, length)
	if _, err := io.ReadFull(t.reader, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func (t *stdioTransport) Close() error {
	var err error
	if t.stdinCloser != nil {
		if e := t.stdinCloser.Close(); e != nil {
			err = e
		}
	}
	if t.stdoutCloser != nil {
		if e := t.stdoutCloser.Close(); e != nil && err == nil {
			err = e
		}
	}
	return err
}

func (t *stdioTransport) readContentLength() (int, error) {
	length := -1
	for {
		line, err := t.reader.ReadString('\n')
		if err != nil {
			return 0, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			value := strings.TrimSpace(line[len("content-length:"):])
			parsed, err := strconv.Atoi(value)
			if err != nil {
				return 0, fmt.Errorf("mcp: invalid content length: %w", err)
			}
			length = parsed
		}
	}
	if length < 0 {
		return 0, errors.New("mcp: missing Content-Length header")
	}
	return length, nil
}
