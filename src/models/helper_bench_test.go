package models

import (
	"strings"
	"sync"
	"testing"
)

func BenchmarkNormalizeMIME(b *testing.B) {
	testCases := []struct {
		name     string
		filename string
		mime     string
	}{
		{"jpeg", "image.jpg", "image/jpeg"},
		{"png", "photo.png", "image/png"},
		{"video", "movie.mp4", "video/mp4"},
		{"text", "readme.md", "text/markdown"},
		{"alias", "pic.jpg", "image/jpg"},            // needs normalization
		{"duplicate", "test.png", "image/image/png"}, // needs fixing
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tc := testCases[i%len(testCases)]
		normalizeMIME(tc.filename, tc.mime)
	}
}

func BenchmarkNormalizeMIME_Concurrent(b *testing.B) {
	testCases := []struct {
		name     string
		filename string
		mime     string
	}{
		{"jpeg", "image.jpg", "image/jpeg"},
		{"png", "photo.png", "image/png"},
		{"video", "movie.mp4", "video/mp4"},
	}

	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		i := 0
		for pb.Next() {
			tc := testCases[i%len(testCases)]
			normalizeMIME(tc.filename, tc.mime)
			i++
		}
	})
}

func BenchmarkCombinePromptWithFiles_Small(b *testing.B) {
	files := []File{
		{Name: "test.txt", MIME: "text/plain", Data: []byte("Hello world")},
		{Name: "image.jpg", MIME: "image/jpeg", Data: make([]byte, 1024)},
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		combinePromptWithFiles("test prompt", files)
	}
}

func BenchmarkCombinePromptWithFiles_Large(b *testing.B) {
	files := make([]File, 10)
	for i := 0; i < 10; i++ {
		files[i] = File{
			Name: "file.txt",
			MIME: "text/plain",
			Data: []byte(strings.Repeat("test data ", 100)),
		}
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		combinePromptWithFiles("test prompt", files)
	}
}

func TestNormalizeMIME_Cache(t *testing.T) {
	// Clear cache
	mimeCacheMu.Lock()
	mimeCache = make(map[string]string)
	mimeCacheMu.Unlock()

	// First call should populate cache
	result1 := normalizeMIME("test.jpg", "image/jpeg")

	// Second call should use cache
	result2 := normalizeMIME("test.jpg", "image/jpeg")

	if result1 != result2 {
		t.Errorf("expected cached result to match, got %s vs %s", result1, result2)
	}

	// Verify cache was populated
	mimeCacheMu.RLock()
	cacheSize := len(mimeCache)
	mimeCacheMu.RUnlock()

	if cacheSize == 0 {
		t.Error("expected cache to be populated")
	}
}

func TestNormalizeMIME_Concurrency(t *testing.T) {
	// Clear cache
	mimeCacheMu.Lock()
	mimeCache = make(map[string]string)
	mimeCacheMu.Unlock()

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			normalizeMIME("test.jpg", "image/jpeg")
		}(i)
	}
	wg.Wait()

	// Should complete without race conditions or panics
}
