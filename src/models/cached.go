package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"time"

	"github.com/Protocol-Lattice/go-agent/src/cache"
)

// CachedLLM wraps an Agent and caches Generate calls.
type CachedLLM struct {
	Agent    Agent
	Cache    *cache.LRUCache
	FilePath string
}

// NewCachedLLM creates a new CachedLLM wrapper.
func NewCachedLLM(agent Agent, size int, ttl time.Duration, filePath string) *CachedLLM {
	c := &CachedLLM{
		Agent:    agent,
		Cache:    cache.NewLRUCache(size, ttl),
		FilePath: filePath,
	}
	if filePath != "" {
		c.load()
	}
	return c
}

func (c *CachedLLM) load() {
	f, err := os.Open(c.FilePath)
	if err != nil {
		return // ignore errors (file not found, etc)
	}
	defer f.Close()

	var dump map[string]cache.CacheEntry
	if err := json.NewDecoder(f).Decode(&dump); err == nil {
		c.Cache.Restore(dump)
	}
}

func (c *CachedLLM) save() {
	if c.FilePath == "" {
		return
	}
	dump := c.Cache.Dump()

	// Atomic write: write to temp, then rename
	tmp := c.FilePath + ".tmp"
	f, err := os.Create(tmp)
	if err != nil {
		return
	}

	if err := json.NewEncoder(f).Encode(dump); err != nil {
		f.Close()
		os.Remove(tmp)
		return
	}
	f.Close()
	os.Rename(tmp, c.FilePath)
}

// Generate checks the cache before calling the underlying agent.
func (c *CachedLLM) Generate(ctx context.Context, prompt string) (any, error) {
	key := cache.HashKey(prompt)
	if val, ok := c.Cache.Get(key); ok {
		return val, nil
	}

	res, err := c.Agent.Generate(ctx, prompt)
	if err != nil {
		return nil, err
	}

	c.Cache.Set(key, res)
	c.save()
	return res, nil
}

// GenerateWithFiles checks the cache (including file hashes) before calling the underlying agent.
func (c *CachedLLM) GenerateWithFiles(ctx context.Context, prompt string, files []File) (any, error) {
	// Create a cache key that includes the prompt and all file contents
	h := sha256.New()
	h.Write([]byte(prompt))
	for _, f := range files {
		h.Write([]byte(f.Name))
		h.Write([]byte(f.MIME))
		h.Write(f.Data)
	}
	key := hex.EncodeToString(h.Sum(nil))

	if val, ok := c.Cache.Get(key); ok {
		return val, nil
	}

	res, err := c.Agent.GenerateWithFiles(ctx, prompt, files)
	if err != nil {
		return nil, err
	}

	c.Cache.Set(key, res)
	c.save()
	return res, nil
}

// GenerateStream passes through to the underlying agent's streaming.
// If the prompt is already cached, it returns a single-chunk stream from cache.
// Otherwise, it streams from the underlying agent and caches the full result when done.
func (c *CachedLLM) GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error) {
	key := cache.HashKey(prompt)
	if val, ok := c.Cache.Get(key); ok {
		ch := make(chan StreamChunk, 1)
		go func() {
			defer close(ch)
			text := fmt.Sprint(val)
			ch <- StreamChunk{Delta: text, Done: true, FullText: text}
		}()
		return ch, nil
	}

	innerCh, err := c.Agent.GenerateStream(ctx, prompt)
	if err != nil {
		return nil, err
	}

	ch := make(chan StreamChunk, 16)
	go func() {
		defer close(ch)
		for chunk := range innerCh {
			ch <- chunk
			if chunk.Done {
				if chunk.FullText != "" && chunk.Err == nil {
					c.Cache.Set(key, chunk.FullText)
					c.save()
				}
			}
		}
	}()

	return ch, nil
}

// TryCreateCachedLLM checks env vars and wraps the agent if caching is enabled.
func TryCreateCachedLLM(agent Agent) Agent {
	sizeStr := os.Getenv("AGENT_LLM_CACHE_SIZE")
	if sizeStr == "" {
		return agent
	}

	size, err := strconv.Atoi(sizeStr)
	if err != nil || size <= 0 {
		return agent
	}

	ttlStr := os.Getenv("AGENT_LLM_CACHE_TTL")
	ttl := 300 * time.Second // default 5 mins
	if ttlStr != "" {
		if sec, err := strconv.Atoi(ttlStr); err == nil && sec > 0 {
			ttl = time.Duration(sec) * time.Second
		}
	}

	path := os.Getenv("AGENT_LLM_CACHE_PATH")
	if path == "" {
		// Default to local directory if not specified, but only if size is set
		path = ".agent_cache.json"
	}

	return NewCachedLLM(agent, size, ttl, path)
}
