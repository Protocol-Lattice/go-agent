package runtime

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
)

type sessionManager struct {
	runtime *Runtime

	counter  atomic.Uint64
	mu       sync.RWMutex
	sessions map[string]*Session
}

func newSessionManager(rt *Runtime) *sessionManager {
	return &sessionManager{
		runtime:  rt,
		sessions: make(map[string]*Session),
	}
}

func (m *sessionManager) newSession(id string) *Session {
	id = strings.TrimSpace(id)
	if id == "" {
		id = fmt.Sprintf("session-%d", m.counter.Add(1))
	}
	session := &Session{runtime: m.runtime, id: id}

	m.mu.Lock()
	m.sessions[id] = session
	m.mu.Unlock()

	return session
}

func (m *sessionManager) getSession(id string) (*Session, error) {
	m.mu.RLock()
	session, ok := m.sessions[id]
	m.mu.RUnlock()
	if !ok {
		return nil, fmt.Errorf("session %s not found", id)
	}
	return session, nil
}

func (m *sessionManager) removeSession(id string) {
	m.mu.Lock()
	delete(m.sessions, id)
	m.mu.Unlock()
}

func (m *sessionManager) activeIDs() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	ids := make([]string, 0, len(m.sessions))
	for id := range m.sessions {
		ids = append(ids, id)
	}
	sort.Strings(ids)
	return ids
}
