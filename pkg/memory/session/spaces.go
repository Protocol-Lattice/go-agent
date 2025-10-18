package session

import (
	"errors"
	"sort"
	"strings"
	"sync"
	"time"
)

// SpaceRole defines the access level granted to a principal within a shared space.
type SpaceRole string

const (
	SpaceRoleReader SpaceRole = "reader"
	SpaceRoleWriter SpaceRole = "writer"
	SpaceRoleAdmin  SpaceRole = "admin"
)

var roleOrder = map[SpaceRole]int{
	SpaceRoleReader: 1,
	SpaceRoleWriter: 2,
	SpaceRoleAdmin:  3,
}

// Space represents a collaborative memory namespace shared across sessions.
type Space struct {
	Name      string
	ACL       map[string]SpaceRole
	ExpiresAt time.Time
	CreatedAt time.Time
	UpdatedAt time.Time
}

func (s *Space) clone() *Space {
	if s == nil {
		return nil
	}
	cp := &Space{
		Name:      s.Name,
		ExpiresAt: s.ExpiresAt,
		CreatedAt: s.CreatedAt,
		UpdatedAt: s.UpdatedAt,
	}
	if len(s.ACL) > 0 {
		cp.ACL = make(map[string]SpaceRole, len(s.ACL))
		for k, v := range s.ACL {
			cp.ACL[k] = v
		}
	}
	return cp
}

func (s *Space) roleFor(principal string) SpaceRole {
	if s == nil || principal == "" {
		return ""
	}
	if s.ACL == nil {
		return ""
	}
	return s.ACL[principal]
}

func (s *Space) expired(now time.Time) bool {
	if s == nil || s.ExpiresAt.IsZero() {
		return false
	}
	return now.After(s.ExpiresAt)
}

func (r SpaceRole) allowsWrite() bool {
	return roleOrder[r] >= roleOrder[SpaceRoleWriter]
}

func (r SpaceRole) allowsRead() bool {
	return roleOrder[r] >= roleOrder[SpaceRoleReader]
}

// Errors returned by the space registry.
var (
	ErrSpaceUnknown   = errors.New("space not registered")
	ErrSpaceExpired   = errors.New("space expired")
	ErrSpaceForbidden = errors.New("space access denied")
)

// SpaceRegistry keeps track of collaborative memory spaces, their ACLs and TTLs.
type SpaceRegistry struct {
	mu         sync.RWMutex
	spaces     map[string]*Space
	defaultTTL time.Duration
	clock      func() time.Time
}

// NewSpaceRegistry builds an empty registry with the provided default TTL.
func NewSpaceRegistry(defaultTTL time.Duration) *SpaceRegistry {
	return &SpaceRegistry{
		spaces:     make(map[string]*Space),
		defaultTTL: defaultTTL,
		clock:      time.Now,
	}
}

func (sr *SpaceRegistry) now() time.Time {
	if sr.clock != nil {
		return sr.clock()
	}
	return time.Now()
}

// Upsert creates or updates a space definition.
func (sr *SpaceRegistry) Upsert(name string, ttl time.Duration, acl map[string]SpaceRole) *Space {
	name = strings.TrimSpace(name)
	if name == "" {
		return nil
	}
	sr.mu.Lock()
	defer sr.mu.Unlock()
	now := sr.now()
	space := sr.spaces[name]
	if space == nil {
		space = &Space{Name: name, CreatedAt: now}
	}
	space.UpdatedAt = now
	if ttl <= 0 {
		ttl = sr.defaultTTL
	}
	if ttl > 0 {
		space.ExpiresAt = now.Add(ttl)
	} else {
		space.ExpiresAt = time.Time{}
	}
	if len(acl) > 0 {
		if space.ACL == nil {
			space.ACL = make(map[string]SpaceRole, len(acl))
		}
		for principal, role := range acl {
			principal = strings.TrimSpace(principal)
			if principal == "" {
				continue
			}
			if _, ok := roleOrder[role]; !ok {
				continue
			}
			space.ACL[principal] = role
		}
	}
	sr.spaces[name] = space
	return space.clone()
}

// Grant assigns or updates a role for a principal inside a space.
func (sr *SpaceRegistry) Grant(spaceName, principal string, role SpaceRole, ttl time.Duration) error {
	if _, ok := roleOrder[role]; !ok {
		return errors.New("invalid space role")
	}
	space := sr.Upsert(spaceName, ttl, nil)
	if space == nil {
		return errors.New("space name is empty")
	}
	sr.mu.Lock()
	defer sr.mu.Unlock()
	space = sr.spaces[spaceName]
	if space.ACL == nil {
		space.ACL = map[string]SpaceRole{}
	}
	space.ACL[strings.TrimSpace(principal)] = role
	sr.spaces[spaceName] = space
	return nil
}

// Revoke removes a principal from the space ACL.
func (sr *SpaceRegistry) Revoke(spaceName, principal string) {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	if space := sr.spaces[spaceName]; space != nil && space.ACL != nil {
		delete(space.ACL, strings.TrimSpace(principal))
	}
}

// CanRead reports whether the principal has read access to the space.
func (sr *SpaceRegistry) CanRead(spaceName, principal string) bool {
	return sr.check(spaceName, principal, false) == nil
}

// CanWrite reports whether the principal has write access to the space.
func (sr *SpaceRegistry) CanWrite(spaceName, principal string) bool {
	return sr.check(spaceName, principal, true) == nil
}

// Check enforces access and returns explicit errors.
func (sr *SpaceRegistry) Check(spaceName, principal string, requireWrite bool) error {
	return sr.check(spaceName, principal, requireWrite)
}

func (sr *SpaceRegistry) check(spaceName, principal string, requireWrite bool) error {
	spaceName = strings.TrimSpace(spaceName)
	principal = strings.TrimSpace(principal)
	if spaceName == "" {
		return ErrSpaceUnknown
	}
	if principal == "" {
		return ErrSpaceForbidden
	}
	sr.mu.Lock()
	defer sr.mu.Unlock()
	now := sr.now()
	space := sr.spaces[spaceName]
	if space == nil {
		return ErrSpaceUnknown
	}
	if space.expired(now) {
		delete(sr.spaces, spaceName)
		return ErrSpaceExpired
	}
	role := space.roleFor(principal)
	if role == "" {
		return ErrSpaceForbidden
	}
	if requireWrite && !role.allowsWrite() {
		return ErrSpaceForbidden
	}
	if !requireWrite && !role.allowsRead() {
		return ErrSpaceForbidden
	}
	return nil
}

// List returns active spaces the principal can read.
func (sr *SpaceRegistry) List(principal string) []string {
	principal = strings.TrimSpace(principal)
	if principal == "" {
		return nil
	}
	sr.mu.RLock()
	defer sr.mu.RUnlock()
	now := sr.now()
	out := make([]string, 0, len(sr.spaces))
	for name, space := range sr.spaces {
		if space == nil {
			continue
		}
		if space.expired(now) {
			continue
		}
		role := space.roleFor(principal)
		if role == "" || !role.allowsRead() {
			continue
		}
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}

// Prune removes expired spaces.
func (sr *SpaceRegistry) Prune() []string {
	sr.mu.Lock()
	defer sr.mu.Unlock()
	now := sr.now()
	removed := make([]string, 0)
	for name, space := range sr.spaces {
		if space == nil {
			continue
		}
		if space.expired(now) {
			delete(sr.spaces, name)
			removed = append(removed, name)
		}
	}
	return removed
}
