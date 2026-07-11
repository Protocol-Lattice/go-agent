package store

import (
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

// prepareMemoryRecord applies the backend-independent normalization needed
// before a memory is serialized. Persistent stores historically include a
// missing space in metadata; the in-memory store only defaults record.Space.
func prepareMemoryRecord(
	sessionID, content string,
	metadata map[string]any,
	embedding []float32,
	now time.Time,
	persistDefaultSpace bool,
) model.MemoryRecord {
	if persistDefaultSpace {
		if metadata == nil {
			metadata = map[string]any{}
		}
		if _, ok := metadata["space"]; !ok {
			metadata["space"] = sessionID
		}
	}

	importance, source, summary, lastEmbedded, metadataJSON := model.NormalizeMetadata(metadata, now)
	normalizedMetadata := model.DecodeMetadata(metadataJSON)
	space := model.StringFromAny(normalizedMetadata["space"])
	if space == "" {
		space = sessionID
	}

	matrix := model.ValidEmbeddingMatrix(normalizedMetadata)
	storedEmbedding := append([]float32(nil), embedding...)
	if len(storedEmbedding) == 0 {
		for _, vector := range matrix {
			if len(vector) == 0 {
				continue
			}
			storedEmbedding = append([]float32(nil), vector...)
			break
		}
	}

	return model.MemoryRecord{
		SessionID:       sessionID,
		Space:           space,
		Content:         content,
		Metadata:        metadataJSON,
		Embedding:       storedEmbedding,
		EmbeddingMatrix: matrix,
		Importance:      importance,
		Source:          source,
		Summary:         summary,
		CreatedAt:       now,
		LastEmbedded:    lastEmbedded,
		GraphEdges:      model.ValidGraphEdges(normalizedMetadata),
	}
}
