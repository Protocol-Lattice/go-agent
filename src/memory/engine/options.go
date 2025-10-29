package engine

import "time"

// ScoreWeights controls the contribution of each scoring component during retrieval.
type ScoreWeights struct {
	Similarity float64
	Keywords   float64
	Importance float64
	Recency    float64
	Source     float64
}

// Options configures the advanced memory engine.
type Options struct {
	Weights                ScoreWeights
	LambdaMMR              float64
	HalfLife               time.Duration
	ClusterSimilarity      float64
	DriftThreshold         float64
	DuplicateSimilarity    float64
	TTL                    time.Duration
	MaxSize                int
	SourceBoost            map[string]float64
	Clock                  func() time.Time
	EnableSummaries        bool
	GraphNeighborhoodHops  int
	GraphNeighborhoodLimit int
}

// DefaultOptions returns the recommended defaults for the advanced memory engine.
func DefaultOptions() Options {
	return Options{
		Weights: ScoreWeights{
			Similarity: 0.45,
			Keywords:   0.20,
			Importance: 0.20,
			Recency:    0.10,
			Source:     0.05,
		},
		LambdaMMR:              0.7,
		HalfLife:               72 * time.Hour,
		ClusterSimilarity:      0.83,
		DriftThreshold:         0.90,
		DuplicateSimilarity:    0.97,
		TTL:                    720 * time.Hour,
		MaxSize:                200_000,
		SourceBoost:            map[string]float64{"default": 1},
		EnableSummaries:        true,
		GraphNeighborhoodHops:  2,
		GraphNeighborhoodLimit: 32,
	}
}

func (o Options) withDefaults() Options {
	defaults := DefaultOptions()
	if o.Weights.Similarity == 0 && o.Weights.Keywords == 0 && o.Weights.Importance == 0 && o.Weights.Recency == 0 && o.Weights.Source == 0 {
		o.Weights = defaults.Weights
	}
	if o.LambdaMMR == 0 {
		o.LambdaMMR = defaults.LambdaMMR
	}
	if o.HalfLife == 0 {
		o.HalfLife = defaults.HalfLife
	}
	if o.ClusterSimilarity == 0 {
		o.ClusterSimilarity = defaults.ClusterSimilarity
	}
	if o.DriftThreshold == 0 {
		o.DriftThreshold = defaults.DriftThreshold
	}
	if o.DuplicateSimilarity == 0 {
		o.DuplicateSimilarity = defaults.DuplicateSimilarity
	}
	if o.TTL == 0 {
		o.TTL = defaults.TTL
	}
	if o.MaxSize == 0 {
		o.MaxSize = defaults.MaxSize
	}
	if o.SourceBoost == nil {
		o.SourceBoost = defaults.SourceBoost
	}
	if o.Clock == nil {
		o.Clock = time.Now
	}
	if o.GraphNeighborhoodHops == 0 {
		o.GraphNeighborhoodHops = defaults.GraphNeighborhoodHops
	}
	if o.GraphNeighborhoodLimit == 0 {
		o.GraphNeighborhoodLimit = defaults.GraphNeighborhoodLimit
	}
	return o
}

func (o Options) normalizedWeights() ScoreWeights {
	total := o.Weights.Similarity + o.Weights.Keywords + o.Weights.Importance + o.Weights.Recency + o.Weights.Source
	if total == 0 {
		return o.Weights
	}
	return ScoreWeights{
		Similarity: o.Weights.Similarity / total,
		Keywords:   o.Weights.Keywords / total,
		Importance: o.Weights.Importance / total,
		Recency:    o.Weights.Recency / total,
		Source:     o.Weights.Source / total,
	}
}
