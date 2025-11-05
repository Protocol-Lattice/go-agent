package model

import (
	"errors"
	"fmt"
	json "github.com/alpkeskin/gotoon"
	"strconv"
)

// EdgeType enumerates supported knowledge graph relationships between memories.
type EdgeType string

const (
	EdgeFollows     EdgeType = "follows"
	EdgeExplains    EdgeType = "explains"
	EdgeContradicts EdgeType = "contradicts"
	EdgeDerivedFrom EdgeType = "derived_from"
)

var validEdgeTypes = map[EdgeType]struct{}{
	EdgeFollows:     {},
	EdgeExplains:    {},
	EdgeContradicts: {},
	EdgeDerivedFrom: {},
}

// GraphEdge represents a typed, directed connection between two memory nodes.
type GraphEdge struct {
	Target int64    `json:"target"`
	Type   EdgeType `json:"type"`
}

// Validate ensures the edge definition is usable.
func (g GraphEdge) Validate() error {
	if g.Target == 0 {
		return errors.New("graph edge target is zero")
	}
	if _, ok := validEdgeTypes[g.Type]; !ok {
		return fmt.Errorf("unsupported edge type %q", g.Type)
	}
	return nil
}

// sanitizeGraphEdges normalizes the metadata map and extracts the edge list.
func SanitizeGraphEdges(meta map[string]any) []GraphEdge {
	if meta == nil {
		return nil
	}
	raw, ok := meta["graph_edges"]
	if !ok {
		return nil
	}
	edges := DecodeGraphEdges(raw)
	sanitized := make([]GraphEdge, 0, len(edges))
	for _, edge := range edges {
		if err := edge.Validate(); err != nil {
			continue
		}
		sanitized = append(sanitized, edge)
	}
	if len(sanitized) == 0 {
		delete(meta, "graph_edges")
		return nil
	}
	meta["graph_edges"] = sanitized
	return sanitized
}

// decodeGraphEdges attempts to coerce arbitrary metadata into []GraphEdge.
func DecodeGraphEdges(raw any) []GraphEdge {
	switch v := raw.(type) {
	case []GraphEdge:
		return v
	case []any:
		edges := make([]GraphEdge, 0, len(v))
		for _, item := range v {
			if e := decodeSingleEdge(item); e.Target != 0 && e.Type != "" {
				edges = append(edges, e)
			}
		}
		return edges
	case []map[string]any:
		edges := make([]GraphEdge, 0, len(v))
		for _, item := range v {
			if e := decodeSingleEdge(item); e.Target != 0 && e.Type != "" {
				edges = append(edges, e)
			}
		}
		return edges
	}
	if e := decodeSingleEdge(raw); e.Target != 0 && e.Type != "" {
		return []GraphEdge{e}
	}
	return nil
}

func decodeSingleEdge(raw any) GraphEdge {
	switch edge := raw.(type) {
	case GraphEdge:
		return edge
	case map[string]any:
		var ge GraphEdge
		ge.Target = numericToInt(edge["target"])
		if typ, ok := edge["type"].(string); ok {
			ge.Type = EdgeType(typ)
		}
		return ge
	case []any:
		if len(edge) == 2 {
			var ge GraphEdge
			ge.Target = numericToInt(edge[0])
			if typ, ok := edge[1].(string); ok {
				ge.Type = EdgeType(typ)
			}
			return ge
		}
	}
	return GraphEdge{}
}

func numericToInt(v any) int64 {
	switch t := v.(type) {
	case float64:
		return int64(t)
	case float32:
		return int64(t)
	case int:
		return int64(t)
	case int32:
		return int64(t)
	case int64:
		return t
	case json.Number:
		if i, err := t.Int64(); err == nil {
			return i
		}
	case string:
		if i, err := strconv.ParseInt(t, 10, 64); err == nil {
			return i
		}
	}
	return 0
}

func ValidGraphEdges(meta map[string]any) []GraphEdge {
	if meta == nil {
		return nil
	}
	raw, ok := meta["graph_edges"]
	if !ok {
		return nil
	}
	edges := DecodeGraphEdges(raw)
	if len(edges) == 0 {
		return nil
	}
	out := make([]GraphEdge, 0, len(edges))
	for _, edge := range edges {
		if err := edge.Validate(); err != nil {
			continue
		}
		out = append(out, edge)
	}
	return out
}
