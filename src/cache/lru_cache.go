package cache

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"sync"
	"time"
)

// CacheEntry holds a cached value with expiration
type CacheEntry struct {
	Value     any
	ExpiresAt time.Time
}

// LRUCache is a thread-safe LRU cache with TTL support
type LRUCache struct {
	mu       sync.RWMutex
	capacity int
	ttl      time.Duration
	items    map[string]*list.Element
	lru      *list.List
}

type entry struct {
	key   string
	value CacheEntry
}

// NewLRUCache creates a new LRU cache with the given capacity and TTL
func NewLRUCache(capacity int, ttl time.Duration) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		ttl:      ttl,
		items:    make(map[string]*list.Element, capacity),
		lru:      list.New(),
	}
}

// Get retrieves a value from the cache
func (c *LRUCache) Get(key string) (any, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	elem, ok := c.items[key]
	if !ok {
		return nil, false
	}

	ent := elem.Value.(*entry)

	// Check if expired
	if time.Now().After(ent.value.ExpiresAt) {
		c.lru.Remove(elem)
		delete(c.items, key)
		return nil, false
	}

	// Move to front (most recently used)
	c.lru.MoveToFront(elem)
	return ent.value.Value, true
}

// Set adds or updates a value in the cache
func (c *LRUCache) Set(key string, value any) {
	c.mu.Lock()
	defer c.mu.Unlock()

	now := time.Now()
	expiresAt := now.Add(c.ttl)

	// Update existing entry
	if elem, ok := c.items[key]; ok {
		c.lru.MoveToFront(elem)
		elem.Value.(*entry).value = CacheEntry{
			Value:     value,
			ExpiresAt: expiresAt,
		}
		return
	}

	// Add new entry
	ent := &entry{
		key: key,
		value: CacheEntry{
			Value:     value,
			ExpiresAt: expiresAt,
		},
	}
	elem := c.lru.PushFront(ent)
	c.items[key] = elem

	// Evict oldest if over capacity
	if c.lru.Len() > c.capacity {
		oldest := c.lru.Back()
		if oldest != nil {
			c.lru.Remove(oldest)
			delete(c.items, oldest.Value.(*entry).key)
		}
	}
}

// Clear removes all entries from the cache
func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.items = make(map[string]*list.Element, c.capacity)
	c.lru.Init()
}

// Len returns the number of items in the cache
func (c *LRUCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lru.Len()
}

// HashKey creates a cache key from a prompt string
func HashKey(prompt string) string {
	h := sha256.Sum256([]byte(prompt))
	return hex.EncodeToString(h[:])
}

// Dump returns a slice of cache entries for persistence
func (c *LRUCache) Dump() map[string]CacheEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()

	dump := make(map[string]CacheEntry, len(c.items))
	for k, elem := range c.items {
		dump[k] = elem.Value.(*entry).value
	}
	return dump
}

// Restore populates the cache from a map of entries
func (c *LRUCache) Restore(dump map[string]CacheEntry) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.lru.Init()
	c.items = make(map[string]*list.Element, c.capacity)

	for k, v := range dump {
		// Check expiry during restore
		if time.Now().After(v.ExpiresAt) {
			continue
		}

		// Add to cache
		ent := &entry{key: k, value: v}
		elem := c.lru.PushFront(ent)
		c.items[k] = elem
	}

	// Enforce capacity
	for c.lru.Len() > c.capacity {
		oldest := c.lru.Back()
		if oldest != nil {
			c.lru.Remove(oldest)
			delete(c.items, oldest.Value.(*entry).key)
		}
	}
}
