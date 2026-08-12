package main

import (
    "context"
    "io"
    "sync"
    "time"

    utcp "github.com/universal-tool-calling-protocol/go-utcp"
    "github.com/universal-tool-calling-protocol/go-utcp/src/providers/base"
    "github.com/universal-tool-calling-protocol/go-utcp/src/tools"
    "github.com/universal-tool-calling-protocol/go-utcp/src/transports"
)

type codeModeEvent struct {
    Type string `json:"type"`
    Tool string `json:"tool,omitempty"`
    Args any `json:"args,omitempty"`
    Result any `json:"result,omitempty"`
    Error string `json:"error,omitempty"`
    Duration int64 `json:"duration_ms,omitempty"`
}

type codeModeEventHub struct { mu sync.RWMutex; subs map[chan codeModeEvent]struct{} }
func newCodeModeEventHub() *codeModeEventHub { return &codeModeEventHub{subs: make(map[chan codeModeEvent]struct{})} }
func (h *codeModeEventHub) subscribe() (<-chan codeModeEvent, func()) { ch:=make(chan codeModeEvent,64); h.mu.Lock(); h.subs[ch]=struct{}{}; h.mu.Unlock(); return ch,func(){h.mu.Lock();if _,ok:=h.subs[ch];ok{delete(h.subs,ch);close(ch)};h.mu.Unlock()} }
func (h *codeModeEventHub) publish(e codeModeEvent) { h.mu.RLock(); defer h.mu.RUnlock(); for ch:=range h.subs { select { case ch<-e: default: } } }
var gatewayCodeModeEvents = newCodeModeEventHub()

type tracingUTCPClient struct { inner utcp.UtcpClientInterface; hub *codeModeEventHub }
func newTracingUTCPClient(inner utcp.UtcpClientInterface, hub *codeModeEventHub) utcp.UtcpClientInterface { if inner==nil{return nil};return &tracingUTCPClient{inner:inner,hub:hub} }
func (c *tracingUTCPClient) RegisterToolProvider(ctx context.Context,p base.Provider)([]tools.Tool,error){return c.inner.RegisterToolProvider(ctx,p)}
func (c *tracingUTCPClient) DeregisterToolProvider(ctx context.Context,name string)error{return c.inner.DeregisterToolProvider(ctx,name)}
func (c *tracingUTCPClient) SearchTools(q string,limit int)([]tools.Tool,error){return c.inner.SearchTools(q,limit)}
func (c *tracingUTCPClient) GetTransports()map[string]utcp.ClientTransport{return c.inner.GetTransports()}
func (c *tracingUTCPClient) CallTool(ctx context.Context,name string,args map[string]any)(any,error){started:=time.Now();c.hub.publish(codeModeEvent{Type:"codemode_tool_start",Tool:name,Args:args});result,err:=c.inner.CallTool(ctx,name,args);e:=codeModeEvent{Type:"codemode_tool_result",Tool:name,Result:result,Duration:time.Since(started).Milliseconds()};if err!=nil{e.Type="codemode_tool_error";e.Error=err.Error()};c.hub.publish(e);return result,err}

type tracingStreamResult struct { inner transports.StreamResult; tool string; hub *codeModeEventHub; started time.Time; finished bool }
func (s *tracingStreamResult) Next()(any,error){item,err:=s.inner.Next();if err!=nil&&!s.finished{s.finished=true;e:=codeModeEvent{Type:"codemode_tool_result",Tool:s.tool,Duration:time.Since(s.started).Milliseconds()};if err!=io.EOF{e.Type="codemode_tool_error";e.Error=err.Error()};s.hub.publish(e)};return item,err}
func (s *tracingStreamResult) Close()error{return s.inner.Close()}
func (c *tracingUTCPClient) CallToolStream(ctx context.Context,name string,args map[string]any)(transports.StreamResult,error){started:=time.Now();c.hub.publish(codeModeEvent{Type:"codemode_tool_start",Tool:name,Args:args});stream,err:=c.inner.CallToolStream(ctx,name,args);if err!=nil{c.hub.publish(codeModeEvent{Type:"codemode_tool_error",Tool:name,Error:err.Error(),Duration:time.Since(started).Milliseconds()});return nil,err};return &tracingStreamResult{inner:stream,tool:name,hub:c.hub,started:started},nil}
