package markdown

import (
	"crypto/rand"
	"encoding/hex"
	"strings"
	"time"
)

type Record struct {
	ID        string
	Scope     string
	SessionID string
	Role      string
	Content   string
	Tags      []string
	CreatedAt time.Time
}

func (r Record) normalized() Record {
	if r.ID == "" {
		r.ID = newID()
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
	return r
}

func newID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return hex.EncodeToString([]byte(time.Now().UTC().Format(time.RFC3339Nano)))
	}
	return hex.EncodeToString(b[:])
}
