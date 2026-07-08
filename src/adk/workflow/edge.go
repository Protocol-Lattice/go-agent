package workflow

// Edge connects two workflow nodes. A nil Route is an unconditional edge.
type Edge struct {
	From  Node
	To    Node
	Route Route
}

// Chain wires nodes into a sequential route.
func Chain(nodes ...Node) []Edge {
	if len(nodes) < 2 {
		return nil
	}
	edges := make([]Edge, 0, len(nodes)-1)
	for i := 0; i < len(nodes)-1; i++ {
		edges = append(edges, Edge{From: nodes[i], To: nodes[i+1]})
	}
	return edges
}

// Concat joins multiple edge slices.
func Concat(groups ...[]Edge) []Edge {
	var total int
	for _, group := range groups {
		total += len(group)
	}
	edges := make([]Edge, 0, total)
	for _, group := range groups {
		edges = append(edges, group...)
	}
	return edges
}

// EdgeBuilder provides a fluent way to assemble graph edges.
type EdgeBuilder struct {
	edges []Edge
}

func NewEdgeBuilder() *EdgeBuilder {
	return &EdgeBuilder{}
}

func (b *EdgeBuilder) Add(from, to Node) *EdgeBuilder {
	if b == nil {
		return b
	}
	b.edges = append(b.edges, Edge{From: from, To: to})
	return b
}

func (b *EdgeBuilder) AddRoute(from, to Node, route Route) *EdgeBuilder {
	if b == nil {
		return b
	}
	b.edges = append(b.edges, Edge{From: from, To: to, Route: route})
	return b
}

func (b *EdgeBuilder) AddFanOut(from Node, targets ...Node) *EdgeBuilder {
	for _, target := range targets {
		b.Add(from, target)
	}
	return b
}

func (b *EdgeBuilder) AddFanIn(sources []Node, target Node) *EdgeBuilder {
	for _, source := range sources {
		b.Add(source, target)
	}
	return b
}

func (b *EdgeBuilder) Build() []Edge {
	if b == nil || len(b.edges) == 0 {
		return nil
	}
	out := make([]Edge, len(b.edges))
	copy(out, b.edges)
	return out
}
