package workspace

import (
	"context"
	"io/fs"
	"path/filepath"
	"time"
)

// Watcher is a dependency-free polling filesystem watcher. It keeps the
// workspace index synchronized without requiring an OS-specific dependency.
type Watcher struct {
	Index    *Index
	Interval time.Duration
}

func (w *Watcher) Run(ctx context.Context) error {
	if ctx == nil { ctx = context.Background() }
	if w.Index == nil { return context.Canceled }
	if w.Interval <= 0 { w.Interval = 500 * time.Millisecond }
	state := w.snapshot()
	ticker := time.NewTicker(w.Interval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done(): return ctx.Err()
		case <-ticker.C:
			next := w.snapshot()
			changes := diffSnapshots(state, next)
			if len(changes) > 0 {
				if err := w.Index.Update(ctx, changes); err != nil { return err }
			}
			state = next
		}
	}
}

type fileStamp struct { Size int64; ModUnix int64 }

func (w *Watcher) snapshot() map[string]fileStamp {
	out := make(map[string]fileStamp)
	_ = filepath.Walk(w.Index.root, func(path string, info fs.FileInfo, err error) error {
		if err != nil || info == nil { return nil }
		if info.IsDir() {
			if path != w.Index.root && w.Index.ignored(path) { return filepath.SkipDir }
			return nil
		}
		if !w.Index.supported(path) || info.Size() > w.Index.config.MaxFileSize { return nil }
		rel, err := filepath.Rel(w.Index.root, path); if err != nil { return nil }
		out[filepath.ToSlash(rel)] = fileStamp{Size: info.Size(), ModUnix: info.ModTime().UnixNano()}
		return nil
	})
	return out
}

func diffSnapshots(old, next map[string]fileStamp) []FileChange {
	changes := make([]FileChange, 0)
	for path, stamp := range next {
		previous, ok := old[path]
		if !ok || previous != stamp { changes = append(changes, FileChange{Path: path, Kind: ChangeModified}) }
	}
	for path := range old { if _, ok := next[path]; !ok { changes = append(changes, FileChange{Path: path, Kind: ChangeDeleted}) } }
	return changes
}
