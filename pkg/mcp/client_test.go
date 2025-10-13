package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestClientListAndCall(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	clientTransport, server := newInMemoryPair()

	server.handle("initialize", func(id string, params json.RawMessage) (any, *rpcError) {
		return map[string]any{
			"protocolVersion": protocolVersion,
			"serverInfo": map[string]string{
				"name":    "mock-server",
				"version": "1.0.0",
			},
		}, nil
	})
	server.handle("tools/list", func(id string, params json.RawMessage) (any, *rpcError) {
		return map[string]any{
			"tools": []ToolDefinition{{
				Name:        "echo",
				Description: "Echoes the provided input",
			}},
		}, nil
	})
	server.handle("tools/call", func(id string, params json.RawMessage) (any, *rpcError) {
		var payload struct {
			Name      string         `json:"name"`
			Arguments map[string]any `json:"arguments"`
		}
		if err := json.Unmarshal(params, &payload); err != nil {
			return nil, &rpcError{Code: -32602, Message: err.Error()}
		}
		if payload.Name != "echo" {
			return nil, &rpcError{Code: -32001, Message: "unknown tool"}
		}
		input, _ := payload.Arguments["input"].(string)
		return CallResult{
			Content: []Content{{Type: "text", Text: fmt.Sprintf("echo:%s", input)}},
		}, nil
	})

	go server.serve(ctx)

	client, err := NewClient(ctx, clientTransport, Options{})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}
	defer client.Close()

	tools, err := client.ListTools(ctx)
	if err != nil {
		t.Fatalf("ListTools error: %v", err)
	}
	if len(tools) != 1 || tools[0].Name != "echo" {
		t.Fatalf("unexpected tools: %#v", tools)
	}

	result, err := client.CallTool(ctx, "echo", map[string]any{"input": "hello"})
	if err != nil {
		t.Fatalf("CallTool error: %v", err)
	}
	if got := result.Text(); got != "echo:hello" {
		t.Fatalf("unexpected result: %s", got)
	}
}

func TestClientCallToolError(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	clientTransport, server := newInMemoryPair()

	server.handle("initialize", func(id string, params json.RawMessage) (any, *rpcError) {
		return map[string]any{
			"protocolVersion": protocolVersion,
			"serverInfo":      map[string]string{"name": "mock", "version": "1"},
		}, nil
	})
	server.handle("tools/list", func(id string, params json.RawMessage) (any, *rpcError) {
		return map[string]any{"tools": []ToolDefinition{}}, nil
	})
	server.handle("tools/call", func(id string, params json.RawMessage) (any, *rpcError) {
		return CallResult{
			IsError: true,
			Content: []Content{{Type: "text", Text: "failure"}},
		}, nil
	})

	go server.serve(ctx)

	client, err := NewClient(ctx, clientTransport, Options{})
	if err != nil {
		t.Fatalf("NewClient error: %v", err)
	}
	defer client.Close()

	_, err = client.CallTool(ctx, "echo", map[string]any{"input": "hi"})
	if err == nil || !strings.Contains(err.Error(), "failure") {
		t.Fatalf("expected failure error, got %v", err)
	}
}

// ----------------------------------------------------------------------------
// Helpers

type inMemoryServer struct {
	reader   *bufio.Reader
	writer   io.Writer
	handlers map[string]func(id string, params json.RawMessage) (any, *rpcError)
	mu       sync.RWMutex
}

func newInMemoryPair() (Transport, *inMemoryServer) {
	clientRead, serverWrite := io.Pipe()
	serverRead, clientWrite := io.Pipe()

	transport := &stdioTransport{
		reader:       bufio.NewReader(clientRead),
		writer:       clientWrite,
		stdinCloser:  clientWrite,
		stdoutCloser: clientRead,
	}

	server := &inMemoryServer{
		reader:   bufio.NewReader(serverRead),
		writer:   serverWrite,
		handlers: make(map[string]func(id string, params json.RawMessage) (any, *rpcError)),
	}

	return transport, server
}

func (s *inMemoryServer) handle(method string, fn func(id string, params json.RawMessage) (any, *rpcError)) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.handlers[method] = fn
}

func (s *inMemoryServer) serve(ctx context.Context) {
	for {
		payload, err := readFrame(s.reader)
		if err != nil {
			return
		}

		var req request
		if err := json.Unmarshal(payload, &req); err != nil {
			resp := responseEnvelope{JSONRPC: "2.0", ID: &req.ID, Error: &rpcError{Code: -32700, Message: err.Error()}}
			_ = writeFrame(s.writer, resp)
			continue
		}

		s.mu.RLock()
		handler := s.handlers[req.Method]
		s.mu.RUnlock()

		if handler == nil {
			resp := responseEnvelope{JSONRPC: "2.0", ID: &req.ID, Error: &rpcError{Code: -32601, Message: "method not found"}}
			_ = writeFrame(s.writer, resp)
			continue
		}

		result, rpcErr := handler(req.ID, mustRaw(req.Params))
		if rpcErr != nil {
			resp := responseEnvelope{JSONRPC: "2.0", ID: &req.ID, Error: rpcErr}
			_ = writeFrame(s.writer, resp)
			continue
		}

		encoded, err := json.Marshal(result)
		if err != nil {
			resp := responseEnvelope{JSONRPC: "2.0", ID: &req.ID, Error: &rpcError{Code: -32603, Message: err.Error()}}
			_ = writeFrame(s.writer, resp)
			continue
		}

		resp := responseEnvelope{JSONRPC: "2.0", ID: &req.ID, Result: encoded}
		_ = writeFrame(s.writer, resp)
	}
}

func readFrame(r *bufio.Reader) ([]byte, error) {
	length := -1
	for {
		line, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		line = strings.TrimRight(line, "\r\n")
		if line == "" {
			break
		}
		if strings.HasPrefix(strings.ToLower(line), "content-length:") {
			value := strings.TrimSpace(line[len("content-length:"):])
			n, err := strconv.Atoi(value)
			if err != nil {
				return nil, err
			}
			length = n
		}
	}
	if length < 0 {
		return nil, errors.New("missing Content-Length header")
	}
	buf := make([]byte, length)
	if _, err := io.ReadFull(r, buf); err != nil {
		return nil, err
	}
	return buf, nil
}

func writeFrame(w io.Writer, v responseEnvelope) error {
	payload, err := json.Marshal(v)
	if err != nil {
		return err
	}
	header := fmt.Sprintf("Content-Length: %d\r\n\r\n", len(payload))
	if _, err := io.WriteString(w, header); err != nil {
		return err
	}
	_, err = w.Write(payload)
	return err
}

func mustRaw(v any) json.RawMessage {
	if v == nil {
		return nil
	}
	data, _ := json.Marshal(v)
	return data
}
