package main

import (
	"context"
	"fmt"
	"log"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory"
)

func main() {
	ctx := context.Background()

	store := NewGraphInMemoryStore()
	opts := memory.DefaultOptions()
	opts.MaxSize = 32
	opts.GraphNeighborhoodLimit = 8
	engine := memory.NewEngine(store, opts)
	engine = engine.WithEmbedder(graphAwareEmbedder{})

	sessionID := "graph-lab"

	fmt.Println("Storing graph-aware memories with multi-vector embeddings...")

	primer, err := engine.Store(ctx, sessionID, "Graph neural networks blend structural neighborhoods with learned node representations.", map[string]any{
		"space": "lab",
		"topic": "primer",
	})
	if err != nil {
		log.Fatalf("store primer: %v", err)
	}
	reportRecord("Primer", primer)

	messagePassing, err := engine.Store(ctx, sessionID, "Message passing layers aggregate neighbor states into each node vector to capture connectivity.", map[string]any{
		"topic":       "message_passing",
		"graph_edges": []memory.GraphEdge{{Target: primer.ID, Type: memory.EdgeExplains}},
	})
	if err != nil {
		log.Fatalf("store message passing: %v", err)
	}
	reportRecord("Message Passing", messagePassing)

	readout, err := engine.Store(ctx, sessionID, "Graph pooling summarizes learned node embeddings into graph-level vectors for downstream tasks.", map[string]any{
		"topic":       "readout",
		"graph_edges": []memory.GraphEdge{{Target: messagePassing.ID, Type: memory.EdgeFollows}},
	})
	if err != nil {
		log.Fatalf("store readout: %v", err)
	}
	reportRecord("Readout", readout)

	// Update the graph so each memory references the concepts it depends on.
	connect := func(rec *memory.MemoryRecord, edges []memory.GraphEdge) {
		rec.GraphEdges = edges
		if err := store.UpsertGraph(ctx, *rec, edges); err != nil {
			log.Fatalf("upsert graph for %s: %v", rec.Content, err)
		}
	}

	connect(&primer, []memory.GraphEdge{
		{Target: messagePassing.ID, Type: memory.EdgeExplains},
		{Target: readout.ID, Type: memory.EdgeDerivedFrom},
	})
	connect(&messagePassing, []memory.GraphEdge{
		{Target: primer.ID, Type: memory.EdgeDerivedFrom},
		{Target: readout.ID, Type: memory.EdgeFollows},
	})
	connect(&readout, []memory.GraphEdge{
		{Target: messagePassing.ID, Type: memory.EdgeExplains},
	})

	fmt.Println()
	fmt.Println("Retrieving memories for query: 'How do graph models propagate information across edges?'")
	results, err := engine.Retrieve(ctx, "How do graph models propagate information across edges?", 4)
	if err != nil {
		log.Fatalf("retrieve memories: %v", err)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].WeightedScore > results[j].WeightedScore })
	for idx, rec := range results {
		fmt.Printf("%d. [%0.3f] %s\n", idx+1, rec.WeightedScore, rec.Content)
		if len(rec.MultiEmbeddings) > 0 {
			printMatrix("   auxiliary vectors", rec.MultiEmbeddings)
		}
		if len(rec.GraphEdges) > 0 {
			fmt.Println("   graph edges:")
			for _, edge := range rec.GraphEdges {
				fmt.Printf("     -> %d (%s)\n", edge.Target, edge.Type)
			}
		}
		fmt.Println()
	}
}

// reportRecord prints the stored memory alongside its auxiliary embedding vectors.
func reportRecord(label string, rec memory.MemoryRecord) {
	fmt.Printf("%s stored as memory #%d\n", label, rec.ID)
	fmt.Printf("  content: %s\n", rec.Content)
	if len(rec.MultiEmbeddings) > 0 {
		printMatrix("  auxiliary vectors", rec.MultiEmbeddings)
	}
	if len(rec.GraphEdges) > 0 {
		fmt.Println("  graph edges:")
		for _, edge := range rec.GraphEdges {
			fmt.Printf("    -> %d (%s)\n", edge.Target, edge.Type)
		}
	}
	fmt.Println()
}

// graphAwareEmbedder creates a multi-vector embedding highlighting structural and vector semantics.
type graphAwareEmbedder struct{}

func (graphAwareEmbedder) Embed(ctx context.Context, text string) ([]float32, error) {
	vectors, err := graphAwareEmbedder{}.EmbedMany(ctx, text)
	if err != nil || len(vectors) == 0 {
		return nil, err
	}
	return vectors[0], nil
}

func (graphAwareEmbedder) EmbedMany(_ context.Context, text string) ([][]float32, error) {
	const dims = 8
	primary := make([]float32, dims)
	structure := make([]float32, dims)
	representation := make([]float32, dims)

	tokens := strings.Fields(strings.ToLower(text))
	if len(tokens) == 0 {
		return [][]float32{primary}, nil
	}

	for _, token := range tokens {
		bucket := hashToken(token) % dims
		primary[bucket] += 1
		if isGraphToken(token) {
			structure[(bucket+1)%dims] += 1
		}
		if isVectorToken(token) {
			representation[(bucket+2)%dims] += 1
		}
	}

	normalize := func(vec []float32) {
		var total float32
		for _, val := range vec {
			total += val
		}
		if total == 0 {
			return
		}
		inv := 1 / total
		for i, val := range vec {
			vec[i] = val * inv
		}
	}

	normalize(primary)
	normalize(structure)
	normalize(representation)

	return [][]float32{primary, structure, representation}, nil
}

func hashToken(token string) int {
	var hash uint32 = 2166136261
	for i := 0; i < len(token); i++ {
		hash ^= uint32(token[i])
		hash *= 16777619
	}
	return int(hash)
}

func isGraphToken(token string) bool {
	switch token {
	case "graph", "graphs", "node", "nodes", "edge", "edges", "connectivity", "neighborhood", "structure", "propagate", "propagation":
		return true
	}
	return false
}

func isVectorToken(token string) bool {
	switch token {
	case "vector", "vectors", "embedding", "embeddings", "representation", "representations", "matrix", "matrices", "feature", "features", "state", "states":
		return true
	}
	return false
}

func printMatrix(label string, matrix [][]float32) {
	fmt.Println(label + ":")
	for _, row := range matrix {
		fmt.Print("    [")
		for i, val := range row {
			fmt.Printf("%0.2f", val)
			if i < len(row)-1 {
				fmt.Print(" ")
			}
		}
		fmt.Println("]")
	}
}

// GraphInMemoryStore extends the in-memory store with knowledge-graph traversal.
type GraphInMemoryStore struct {
	base  *memory.InMemoryStore
	mu    sync.RWMutex
	edges map[int64][]memory.GraphEdge
}

func NewGraphInMemoryStore() *GraphInMemoryStore {
	return &GraphInMemoryStore{
		base:  memory.NewInMemoryStore(),
		edges: make(map[int64][]memory.GraphEdge),
	}
}

func (s *GraphInMemoryStore) StoreMemory(ctx context.Context, sessionID, content string, metadata map[string]any, embedding []float32) error {
	return s.base.StoreMemory(ctx, sessionID, content, metadata, embedding)
}

func (s *GraphInMemoryStore) StoreMemoryMulti(ctx context.Context, sessionID, content string, metadata map[string]any, embeddings [][]float32) error {
	return s.base.StoreMemoryMulti(ctx, sessionID, content, metadata, embeddings)
}

func (s *GraphInMemoryStore) SearchMemory(ctx context.Context, queryEmbedding []float32, limit int) ([]memory.MemoryRecord, error) {
	records, err := s.base.SearchMemory(ctx, queryEmbedding, limit)
	if err != nil {
		return nil, err
	}
	s.attachGraph(records)
	return records, nil
}

func (s *GraphInMemoryStore) SearchMemoryMulti(ctx context.Context, queryEmbeddings [][]float32, limit int) ([]memory.MemoryRecord, error) {
	records, err := s.base.SearchMemoryMulti(ctx, queryEmbeddings, limit)
	if err != nil {
		return nil, err
	}
	s.attachGraph(records)
	return records, nil
}

func (s *GraphInMemoryStore) UpdateEmbedding(ctx context.Context, id int64, embedding []float32, lastEmbedded time.Time) error {
	return s.base.UpdateEmbedding(ctx, id, embedding, lastEmbedded)
}

func (s *GraphInMemoryStore) DeleteMemory(ctx context.Context, ids []int64) error {
	if err := s.base.DeleteMemory(ctx, ids); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	for _, id := range ids {
		delete(s.edges, id)
	}
	for id, neighbors := range s.edges {
		filtered := neighbors[:0]
		for _, edge := range neighbors {
			if edge.Target == 0 {
				continue
			}
			remove := false
			for _, removed := range ids {
				if edge.Target == removed {
					remove = true
					break
				}
			}
			if !remove {
				filtered = append(filtered, edge)
			}
		}
		if len(filtered) == 0 {
			delete(s.edges, id)
		} else {
			s.edges[id] = cloneEdges(filtered)
		}
	}
	return nil
}

func (s *GraphInMemoryStore) Iterate(ctx context.Context, fn func(memory.MemoryRecord) bool) error {
	return s.base.Iterate(ctx, func(rec memory.MemoryRecord) bool {
		s.mu.RLock()
		if edges := s.edges[rec.ID]; len(edges) > 0 {
			rec.GraphEdges = cloneEdges(edges)
		}
		s.mu.RUnlock()
		return fn(rec)
	})
}

func (s *GraphInMemoryStore) Count(ctx context.Context) (int, error) {
	return s.base.Count(ctx)
}

func (s *GraphInMemoryStore) UpsertGraph(_ context.Context, record memory.MemoryRecord, edges []memory.GraphEdge) error {
	valid := make([]memory.GraphEdge, 0, len(edges))
	for _, edge := range edges {
		if err := edge.Validate(); err == nil && edge.Target != 0 {
			valid = append(valid, edge)
		}
	}

	s.mu.Lock()
	if len(valid) == 0 {
		delete(s.edges, record.ID)
	} else {
		s.edges[record.ID] = cloneEdges(valid)
	}
	s.mu.Unlock()
	return nil
}

func (s *GraphInMemoryStore) Neighborhood(ctx context.Context, seedIDs []int64, hops, limit int) ([]memory.MemoryRecord, error) {
	if limit <= 0 || hops <= 0 {
		return nil, nil
	}

	s.mu.RLock()
	adjacency := make(map[int64][]memory.GraphEdge, len(s.edges))
	for id, edges := range s.edges {
		adjacency[id] = cloneEdges(edges)
	}
	s.mu.RUnlock()

	visited := make(map[int64]struct{}, len(seedIDs))
	queue := make([]struct {
		id    int64
		depth int
	}, 0, len(seedIDs))
	for _, id := range seedIDs {
		if id == 0 {
			continue
		}
		visited[id] = struct{}{}
		queue = append(queue, struct {
			id    int64
			depth int
		}{id: id, depth: 0})
	}

	neighbors := make([]int64, 0, limit)
	for len(queue) > 0 && len(neighbors) < limit {
		current := queue[0]
		queue = queue[1:]
		if current.depth >= hops {
			continue
		}
		for _, edge := range adjacency[current.id] {
			if edge.Target == 0 {
				continue
			}
			if _, seen := visited[edge.Target]; seen {
				continue
			}
			visited[edge.Target] = struct{}{}
			neighbors = append(neighbors, edge.Target)
			if len(neighbors) >= limit {
				break
			}
			queue = append(queue, struct {
				id    int64
				depth int
			}{id: edge.Target, depth: current.depth + 1})
		}
	}

	if len(neighbors) == 0 {
		return nil, nil
	}

	want := make(map[int64]struct{}, len(neighbors))
	for _, id := range neighbors {
		want[id] = struct{}{}
	}

	results := make([]memory.MemoryRecord, 0, len(neighbors))
	err := s.base.Iterate(ctx, func(rec memory.MemoryRecord) bool {
		if _, ok := want[rec.ID]; !ok {
			return true
		}
		if edges := adjacency[rec.ID]; len(edges) > 0 {
			rec.GraphEdges = cloneEdges(edges)
		}
		results = append(results, rec)
		delete(want, rec.ID)
		return len(results) < len(neighbors)
	})
	if err != nil {
		return nil, err
	}
	return results, nil
}

func (s *GraphInMemoryStore) attachGraph(records []memory.MemoryRecord) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	for i := range records {
		if edges := s.edges[records[i].ID]; len(edges) > 0 {
			records[i].GraphEdges = cloneEdges(edges)
		}
	}
}

func cloneEdges(src []memory.GraphEdge) []memory.GraphEdge {
	if len(src) == 0 {
		return nil
	}
	out := make([]memory.GraphEdge, len(src))
	copy(out, src)
	return out
}
