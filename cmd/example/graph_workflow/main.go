package main

import (
	"context"
	"fmt"
	"log"
	"strings"

	"github.com/Protocol-Lattice/go-agent/src/adk/workflow"
	"github.com/Protocol-Lattice/go-agent/src/adk/workflowagent"
)

func main() {
	ctx := context.Background()

	classify := workflow.NewEmittingFunctionNode[string, any]("classify",
		func(_ workflow.Context, input string, emit workflow.EmitFunc) (any, error) {
			route := "LOGISTICS"
			if strings.Contains(strings.ToLower(input), "bug") {
				route = "BUG"
			}
			return nil, emit(&workflow.Event{
				Output: input,
				Routes: []any{route},
			})
		},
		workflow.NodeConfig{Description: "Classifies the request."},
	)

	bugHandler := workflow.NewFunctionNode[string, string]("bug_handler",
		func(_ workflow.Context, input string) (string, error) {
			return "Handling bug: " + input, nil
		},
		workflow.NodeConfig{},
	)

	logisticsHandler := workflow.NewFunctionNode[string, string]("logistics_handler",
		func(_ workflow.Context, input string) (string, error) {
			return "Handling logistics: " + input, nil
		},
		workflow.NodeConfig{},
	)

	edges := workflow.Concat(
		workflow.Chain(workflow.Start, classify),
		[]workflow.Edge{
			{From: classify, To: bugHandler, Route: workflow.StringRoute("BUG")},
			{From: classify, To: logisticsHandler, Route: workflow.Default},
		},
	)

	root, err := workflowagent.New(workflowagent.Config{
		Name:        "routing_workflow",
		Description: "Classifies a request and routes it to a handler.",
		Edges:       edges,
	})
	if err != nil {
		log.Fatal(err)
	}

	out, err := root.Generate(ctx, "demo-session", "There is a bug in checkout")
	if err != nil {
		log.Fatal(err)
	}

	fmt.Println(out)
}
