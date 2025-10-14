package memory

import "sync/atomic"

// Metrics captures lightweight runtime counters for observability.
type Metrics struct {
	stored             atomic.Int64
	retrieved          atomic.Int64
	deduplicated       atomic.Int64
	reembedded         atomic.Int64
	pruned             atomic.Int64
	clustersSummarized atomic.Int64
}

func (m *Metrics) IncStored()             { m.stored.Add(1) }
func (m *Metrics) IncRetrieved(n int)     { m.retrieved.Add(int64(n)) }
func (m *Metrics) IncDeduplicated()       { m.deduplicated.Add(1) }
func (m *Metrics) IncReembedded()         { m.reembedded.Add(1) }
func (m *Metrics) IncPruned(n int)        { m.pruned.Add(int64(n)) }
func (m *Metrics) IncClustersSummarized() { m.clustersSummarized.Add(1) }

// Snapshot returns the current values for reporting/logging.
type MetricsSnapshot struct {
	Stored             int64 `json:"stored"`
	Retrieved          int64 `json:"retrieved"`
	Deduplicated       int64 `json:"deduplicated"`
	Reembedded         int64 `json:"reembedded"`
	Pruned             int64 `json:"pruned"`
	ClustersSummarized int64 `json:"clusters_summarized"`
}

func (m *Metrics) Snapshot() MetricsSnapshot {
	if m == nil {
		return MetricsSnapshot{}
	}
	return MetricsSnapshot{
		Stored:             m.stored.Load(),
		Retrieved:          m.retrieved.Load(),
		Deduplicated:       m.deduplicated.Load(),
		Reembedded:         m.reembedded.Load(),
		Pruned:             m.pruned.Load(),
		ClustersSummarized: m.clustersSummarized.Load(),
	}
}
