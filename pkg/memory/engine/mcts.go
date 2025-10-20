package engine

import (
	"context"
	"math"
	"math/rand"
	"sort"
	"strconv"
	"strings"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/memory/model"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/store"
)

type mctsNode struct {
	record       model.MemoryRecord
	parent       *mctsNode
	children     []*mctsNode
	unexpanded   []model.MemoryRecord
	visits       int
	totalReward  float64
	depth        int
	neighborsSet bool
}

func (e *Engine) mctsRefine(ctx context.Context, now time.Time, graph store.GraphStore, query []float32, weights ScoreWeights, seeds []model.MemoryRecord, limit int) []model.MemoryRecord {
	if graph == nil || limit <= 0 || len(seeds) == 0 || e.opts.MCTSSimulations <= 0 {
		return nil
	}

	root := &mctsNode{depth: 0, visits: 1, neighborsSet: true}
	seen := make(map[string]*mctsNode, len(seeds))

	for i := range seeds {
		rec := seeds[i]
		key := recordKey(rec)
		if _, exists := seen[key]; exists {
			continue
		}
		e.computeWeightedScore(now, weights, query, &rec)
		child := &mctsNode{
			record: rec,
			parent: root,
			depth:  1,
		}
		root.children = append(root.children, child)
		seen[key] = child
	}

	if len(root.children) == 0 {
		return nil
	}

	rng := rand.New(rand.NewSource(now.UnixNano()))
	maxDepth := e.opts.MCTSMaxDepth
	if maxDepth <= 0 {
		maxDepth = 1
	}

	for sim := 0; sim < e.opts.MCTSSimulations; sim++ {
		node := root
		path := []*mctsNode{node}

		for {
			if node.depth >= maxDepth {
				break
			}

			if !node.neighborsSet {
				expandNode(ctx, node, graph, e, now, weights, query, seen, limit)
			}

			expanded := false
			for len(node.unexpanded) > 0 {
				nextIdx := len(node.unexpanded) - 1
				candidate := node.unexpanded[nextIdx]
				node.unexpanded = node.unexpanded[:nextIdx]
				key := recordKey(candidate)
				if _, exists := seen[key]; exists {
					continue
				}
				child := &mctsNode{
					record: candidate,
					parent: node,
					depth:  node.depth + 1,
				}
				node.children = append(node.children, child)
				seen[key] = child
				node = child
				path = append(path, node)
				expanded = true
				break
			}
			if expanded {
				break
			}
			if len(node.children) == 0 {
				break
			}
			node = selectChild(node, rng, e.opts.MCTSExploration)
			path = append(path, node)
		}

		reward := node.record.WeightedScore
		if reward == 0 {
			reward = e.computeWeightedScore(now, weights, query, &node.record)
		}

		for _, n := range path {
			n.visits++
			n.totalReward += reward
		}
	}

	candidates := make([]*mctsNode, 0, len(seen))
	for _, node := range seen {
		if node == nil || node.parent == nil {
			continue
		}
		candidates = append(candidates, node)
	}

	if len(candidates) == 0 {
		return nil
	}

	sort.Slice(candidates, func(i, j int) bool {
		avgI := averageReward(candidates[i])
		avgJ := averageReward(candidates[j])
		if avgI != avgJ {
			return avgI > avgJ
		}
		if candidates[i].record.WeightedScore != candidates[j].record.WeightedScore {
			return candidates[i].record.WeightedScore > candidates[j].record.WeightedScore
		}
		return candidates[i].record.CreatedAt.After(candidates[j].record.CreatedAt)
	})

	out := make([]model.MemoryRecord, 0, limit)
	for _, node := range candidates {
		rec := node.record
		avg := averageReward(node)
		if avg > 0 {
			rec.WeightedScore = avg
		}
		out = append(out, rec)
		if len(out) >= limit {
			break
		}
	}

	return out
}

func selectChild(node *mctsNode, rng *rand.Rand, exploration float64) *mctsNode {
	if exploration <= 0 {
		exploration = 1.0
	}
	best := node.children[0]
	bestScore := math.Inf(-1)
	parentVisits := math.Max(float64(node.visits), 1)
	for _, child := range node.children {
		if child.visits == 0 {
			return child
		}
		avg := averageReward(child)
		score := avg + exploration*math.Sqrt(math.Log(parentVisits)/float64(child.visits))
		if score > bestScore {
			best = child
			bestScore = score
		}
	}
	if bestScore == math.Inf(-1) && len(node.children) > 1 {
		return node.children[rng.Intn(len(node.children))]
	}
	return best
}

func averageReward(node *mctsNode) float64 {
	if node == nil {
		return 0
	}
	if node.visits == 0 {
		return node.record.WeightedScore
	}
	return node.totalReward / float64(node.visits)
}

func expandNode(
	ctx context.Context,
	node *mctsNode,
	graph store.GraphStore,
	engine *Engine,
	now time.Time,
	weights ScoreWeights,
	query []float32,
	seen map[string]*mctsNode,
	limit int,
) {
	node.neighborsSet = true
	if node.record.ID == 0 {
		return
	}
	expansion := engine.opts.MCTSExpansion
	if expansion <= 0 {
		expansion = limit
	}
	neighbors, err := graph.Neighborhood(ctx, []int64{node.record.ID}, 1, expansion)
	if err != nil {
		engine.logf("mcts neighborhood lookup failed: %v", err)
		return
	}
	for i := range neighbors {
		rec := neighbors[i]
		key := recordKey(rec)
		if _, exists := seen[key]; exists {
			continue
		}
		engine.computeWeightedScore(now, weights, query, &rec)
		node.unexpanded = append(node.unexpanded, rec)
	}
}

func recordKey(rec model.MemoryRecord) string {
	if rec.ID != 0 {
		return "id:" + strconv.FormatInt(rec.ID, 10)
	}
	return "mem:" + rec.SessionID + "|" + strings.TrimSpace(rec.Content)
}

func dedupeRecords(records []model.MemoryRecord) []model.MemoryRecord {
	if len(records) == 0 {
		return records
	}
	seen := make(map[string]struct{}, len(records))
	out := make([]model.MemoryRecord, 0, len(records))
	for _, rec := range records {
		key := recordKey(rec)
		if _, exists := seen[key]; exists {
			continue
		}
		seen[key] = struct{}{}
		out = append(out, rec)
	}
	return out
}
