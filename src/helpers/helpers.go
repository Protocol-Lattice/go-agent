package helpers

import (
	"strconv"
	"strings"

	agent "github.com/Protocol-Lattice/go-agent"
)

func ParseSourceBoostFlag(raw string) map[string]float64 {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	boosts := make(map[string]float64)
	pairs := strings.Split(raw, ",")
	for _, pair := range pairs {
		parts := strings.SplitN(pair, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.ToLower(strings.TrimSpace(parts[0]))
		value, err := strconv.ParseFloat(strings.TrimSpace(parts[1]), 64)
		if err != nil {
			continue
		}
		boosts[key] = value
	}
	if len(boosts) == 0 {
		return nil
	}
	return boosts
}

func ToolNames(tools []agent.Tool) string {
	if len(tools) == 0 {
		return "<none>"
	}
	names := make([]string, len(tools))
	for i, tool := range tools {
		names[i] = tool.Spec().Name
	}
	return strings.Join(names, ", ")
}

func SubAgentNames(subagents []agent.SubAgent) string {
	if len(subagents) == 0 {
		return "<none>"
	}
	names := make([]string, len(subagents))
	for i, sa := range subagents {
		names[i] = sa.Name()
	}
	return strings.Join(names, ", ")
}

func ParseCSVList(raw string) []string {
	raw = strings.TrimSpace(raw)
	if raw == "" {
		return nil
	}
	parts := strings.Split(raw, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		s := strings.TrimSpace(p)
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}
