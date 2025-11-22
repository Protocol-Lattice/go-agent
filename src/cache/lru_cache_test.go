package cache

import (
	"testing"
	"time"
)

func BenchmarkLRUCache_Set(b *testing.B) {
	cache := NewLRUCache(1000, 5*time.Minute)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Set(HashKey(string(rune(i))), "value")
	}
}

func BenchmarkLRUCache_Get(b *testing.B) {
	cache := NewLRUCache(1000, 5*time.Minute)

	// Populate cache
	for i := 0; i < 100; i++ {
		cache.Set(HashKey(string(rune(i))), "value")
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		cache.Get(HashKey(string(rune(i % 100))))
	}
}

func BenchmarkLRUCache_ConcurrentAccess(b *testing.B) {
	cache := NewLRUCache(1000, 5*time.Minute)

	// Populate cache
	for i := 0; i < 100; i++ {
		cache.Set(HashKey(string(rune(i))), "value")
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			key := HashKey(string(rune(i % 100)))
			if i%2 == 0 {
				cache.Get(key)
			} else {
				cache.Set(key, "value")
			}
			i++
		}
	})
}

func TestLRUCache_Basic(t *testing.T) {
	cache := NewLRUCache(3, time.Hour)

	cache.Set("a", 1)
	cache.Set("b", 2)
	cache.Set("c", 3)

	if val, ok := cache.Get("a"); !ok || val != 1 {
		t.Errorf("expected 1, got %v", val)
	}

	// Add one more, should evict "b" (least recently used)
	cache.Set("d", 4)

	if _, ok := cache.Get("b"); ok {
		t.Error("expected 'b' to be evicted")
	}

	if cache.Len() != 3 {
		t.Errorf("expected cache length 3, got %d", cache.Len())
	}
}

func TestLRUCache_TTL(t *testing.T) {
	cache := NewLRUCache(10, 10*time.Millisecond)

	cache.Set("key", "value")

	if val, ok := cache.Get("key"); !ok || val != "value" {
		t.Error("expected value to be present")
	}

	time.Sleep(20 * time.Millisecond)

	if _, ok := cache.Get("key"); ok {
		t.Error("expected value to be expired")
	}
}
