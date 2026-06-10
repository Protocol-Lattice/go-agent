package markdown

import (
	"crypto/md5"
	"crypto/rand"
	"encoding/binary"
	"encoding/hex"
	"encoding/json"
	"strings"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

type Record struct {
	ID           string // String ID for markdown storage
	NumID        int64  // Numeric ID for VectorStore interface (derived from string ID)
	Scope        string
	SessionID    string
	Role         string
	Content      string
	Tags         []string
	Metadata     map[string]any
	Embedding    []float32
	LastEmbedded time.Time
	CreatedAt    time.Time
}

func (r Record) normalized() Record {
	if r.ID == "" {
		r.ID = newID()
	}
	if r.NumID == 0 {
		r.NumID = stringIDToNumID(r.ID)
	}
	if r.Scope == "" {
		r.Scope = "sessions"
	}
	if r.SessionID == "" {
		r.SessionID = "default"
	}
	if r.Role == "" {
		r.Role = "memory"
	}
	if r.CreatedAt.IsZero() {
		r.CreatedAt = time.Now().UTC()
	}
	r.Content = strings.TrimSpace(r.Content)
	if r.Metadata == nil {
		r.Metadata = make(map[string]any)
	}
	return r
}

// toMemoryRecord converts Record to model.MemoryRecord for VectorStore interface
func (r Record) toMemoryRecord() model.MemoryRecord {
	// Marshal metadata to JSON string
	metaStr := ""
	if len(r.Metadata) > 0 {
		if data, err := json.Marshal(r.Metadata); err == nil {
			metaStr = string(data)
		}
	}

	return model.MemoryRecord{
		ID:           r.NumID,
		SessionID:    r.SessionID,
		Content:      r.Content,
		Metadata:     metaStr,
		Embedding:    r.Embedding,
		LastEmbedded: r.LastEmbedded,
		CreatedAt:    r.CreatedAt,
	}
}

// stringIDToNumID converts a hex string ID to int64 using MD5
func stringIDToNumID(id string) int64 {
	hash := md5.Sum([]byte(id))
	return int64(binary.BigEndian.Uint64(hash[:8]))
}

func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return hex.EncodeToString([]byte(time.Now().UTC().Format(time.RFC3339Nano)))
	}
	return hex.EncodeToString(b[:])
}
