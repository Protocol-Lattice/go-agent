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
	ttlExpired         atomic.Int64
	sizeEvicted        atomic.Int64
	recencySamples     atomic.Int64
	recencySumMicros   atomic.Int64
}

func (m *Metrics) IncStored()             { m.stored.Add(1) }
func (m *Metrics) IncRetrieved(n int)     { m.retrieved.Add(int64(n)) }
func (m *Metrics) IncDeduplicated()       { m.deduplicated.Add(1) }
func (m *Metrics) IncReembedded()         { m.reembedded.Add(1) }
func (m *Metrics) IncPruned(n int)        { m.pruned.Add(int64(n)) }
func (m *Metrics) IncClustersSummarized() { m.clustersSummarized.Add(1) }
func (m *Metrics) IncTTLExpired(n int)    { m.ttlExpired.Add(int64(n)) }
func (m *Metrics) IncSizeEvicted(n int)   { m.sizeEvicted.Add(int64(n)) }
func (m *Metrics) ObserveRecency(decay float64) {
	if decay < 0 {
		decay = 0
	}
	if decay > 1 {
		decay = 1
	}
	m.recencySamples.Add(1)
	m.recencySumMicros.Add(int64(decay * 1_000_000))
}

// Snapshot returns the current values for reporting/logging.
type MetricsSnapshot struct {
	Stored             int64   `json:"stored"`
	Retrieved          int64   `json:"retrieved"`
	Deduplicated       int64   `json:"deduplicated"`
	Reembedded         int64   `json:"reembedded"`
	Pruned             int64   `json:"pruned"`
	ClustersSummarized int64   `json:"clusters_summarized"`
	TTLExpired         int64   `json:"ttl_expired"`
	SizeEvicted        int64   `json:"size_evicted"`
	RecencySamples     int64   `json:"recency_samples"`
	RecencyDecayAvg    float64 `json:"recency_decay_avg"`
}

func (m *Metrics) Snapshot() MetricsSnapshot {
	if m == nil {
		return MetricsSnapshot{}
	}
	samples := m.recencySamples.Load()
	sumMicros := m.recencySumMicros.Load()
	avg := 0.0
	if samples > 0 {
		avg = float64(sumMicros) / 1_000_000 / float64(samples)
	}
	return MetricsSnapshot{
		Stored:             m.stored.Load(),
		Retrieved:          m.retrieved.Load(),
		Deduplicated:       m.deduplicated.Load(),
		Reembedded:         m.reembedded.Load(),
		Pruned:             m.pruned.Load(),
		ClustersSummarized: m.clustersSummarized.Load(),
		TTLExpired:         m.ttlExpired.Load(),
		SizeEvicted:        m.sizeEvicted.Load(),
		RecencySamples:     samples,
		RecencyDecayAvg:    avg,
	}
}
