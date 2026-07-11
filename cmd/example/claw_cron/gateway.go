package main

import (
	"context"
	"fmt"
	"sync"

	"github.com/universal-tool-calling-protocol/go-utcp"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/base"
	"github.com/universal-tool-calling-protocol/go-utcp/src/providers/cli"
	"github.com/universal-tool-calling-protocol/go-utcp/src/repository"
	"github.com/universal-tool-calling-protocol/go-utcp/src/tools"
	"github.com/universal-tool-calling-protocol/go-utcp/src/transports"
)

// PermissionGateway handles human-in-the-loop approvals.
type PermissionGateway struct {
	mu      sync.Mutex
	pending chan permissionReq
}

type permissionReq struct {
	action string
	resp   chan bool
}

func NewPermissionGateway() *PermissionGateway {
	return &PermissionGateway{
		pending: make(chan permissionReq),
	}
}

// RequestChannel returns the channel where pending requests are sent for the main loop to handle.
func (g *PermissionGateway) RequestChannel() <-chan permissionReq {
	return g.pending
}

// Tools returns the UTCP tools for the gateway.
func (g *PermissionGateway) Tools() []tools.Tool {
	return []tools.Tool{
		{
			Name:        "gateway.request_permission",
			Description: "Ask the user for permission to perform a high-stakes or potentially dangerous action.",
			Inputs: tools.ToolInputOutputSchema{
				Type: "object",
				Properties: map[string]any{
					"action": map[string]any{
						"type":        "string",
						"description": "A description of the action you want to perform (e.g., 'Delete file config.json')",
					},
					"reason": map[string]any{
						"type":        "string",
						"description": "The reason why this action is necessary",
					},
				},
				Required: []string{"action", "reason"},
			},
			Handler: g.handleRequestPermission,
		},
	}
}

func (g *PermissionGateway) handleRequestPermission(ctx context.Context, inputs map[string]any) (any, error) {
	action, _ := inputs["action"].(string)
	reason, _ := inputs["reason"].(string)

	prompt := fmt.Sprintf("%s (Reason: %s)", action, reason)

	respCh := make(chan bool)
	req := permissionReq{
		action: prompt,
		resp:   respCh,
	}

	// Send to main loop
	select {
	case g.pending <- req:
	case <-ctx.Done():
		return nil, ctx.Err()
	}

	// Wait for user response
	select {
	case approved := <-respCh:
		if approved {
			return map[string]any{"status": "approved", "message": "User granted permission."}, nil
		}
		return map[string]any{"status": "denied", "message": "User denied permission. DO NOT proceed with this action."}, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// RegisterAsUTCPProvider registers the gateway tools on the provided UTCP client.
func (g *PermissionGateway) RegisterAsUTCPProvider(ctx context.Context, client utcp.UtcpClientInterface) error {
	providerName := "gateway"
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
	var shim *gatewayCLITransport
	if maybe, ok := existing.(*gatewayCLITransport); ok {
		shim = maybe
	} else {
		shim = &gatewayCLITransport{inner: existing}
		transportsMap[string(base.ProviderCLI)] = shim
	}
	if shim.tools == nil {
		shim.tools = make(map[string][]tools.Tool)
	}
	shim.tools[tp.Name] = g.Tools()

	_, err := client.RegisterToolProvider(ctx, tp)
	return err
}

type gatewayCLITransport struct {
	inner repository.ClientTransport
	tools map[string][]tools.Tool
}

func (t *gatewayCLITransport) RegisterToolProvider(ctx context.Context, prov base.Provider) ([]tools.Tool, error) {
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
		return nil, fmt.Errorf("gateway tools not found for provider %s", p.Name)
	}
	return list, nil
}

func (t *gatewayCLITransport) DeregisterToolProvider(ctx context.Context, prov base.Provider) error {
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

func (t *gatewayCLITransport) CallTool(ctx context.Context, toolName string, args map[string]any, prov base.Provider, sessionID *string) (any, error) {
	if p, ok := prov.(*cli.CliProvider); ok {
		if list, ok := t.tools[p.Name]; ok {
			for _, tool := range list {
				if tool.Name == toolName {
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

func (t *gatewayCLITransport) CallToolStream(ctx context.Context, toolName string, args map[string]any, prov base.Provider) (transports.StreamResult, error) {
	if t.inner != nil {
		return t.inner.CallToolStream(ctx, toolName, args, prov)
	}
	return nil, fmt.Errorf("streaming not supported for gateway tools")
}
