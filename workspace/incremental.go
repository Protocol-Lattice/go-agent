package workspace

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// FileChange describes a filesystem change observed since the last index update.
type FileChange struct { Path string; Kind ChangeKind }
type ChangeKind string
const ( ChangeModified ChangeKind = "modified"; ChangeDeleted ChangeKind = "deleted" )

func (i *Index) Update(ctx context.Context, changes []FileChange) error {
	if ctx == nil { ctx = context.Background() }
	for _, change := range changes {
		if err := ctx.Err(); err != nil { return err }
		path := filepath.Clean(change.Path)
		if !filepath.IsAbs(path) { path = filepath.Join(i.root, path) }
		rel, err := filepath.Rel(i.root, path); if err != nil { return err }
		rel = filepath.ToSlash(rel)
		if change.Kind == ChangeDeleted { i.removeFile(rel); continue }
		if i.ignored(path) || !i.supported(path) { continue }
		st, err := os.Stat(path)
		if os.IsNotExist(err) { i.removeFile(rel); continue }
		if err != nil { return err }
		if st.IsDir() || st.Size() > i.config.MaxFileSize { i.removeFile(rel); continue }
		i.removeFile(rel)
		if err := i.indexFile(path, st.Size()); err != nil { return err }
	}
	i.rebuildSymbolIndex()
	if i.embedder != nil {
		for _, change := range changes {
			if err := ctx.Err(); err != nil { return err }
			path := filepath.ToSlash(change.Path)
			if filepath.IsAbs(change.Path) { if rel, err := filepath.Rel(i.root, change.Path); err == nil { path = filepath.ToSlash(rel) } }
			if change.Kind == ChangeDeleted { delete(i.vectors, path); continue }
			if f, ok := i.files[path]; ok {
				b, err := os.ReadFile(filepath.Join(i.root, filepath.FromSlash(f.Path))); if err != nil { return err }
				v, err := i.embedder.Embed(ctx, string(b)); if err != nil { return err }
				if len(v) == 0 { delete(i.vectors, path) } else { i.vectors[path] = append([]float32(nil), v...) }
			}
		}
	}
	return nil
}

func (i *Index) removeFile(path string) { delete(i.files, filepath.ToSlash(path)); delete(i.imports, filepath.ToSlash(path)); delete(i.vectors, filepath.ToSlash(path)) }
func (i *Index) rebuildSymbolIndex() {
	i.symbols = i.symbols[:0]; i.byName = make(map[string][]Symbol)
	for _, f := range i.files { i.symbols = append(i.symbols, f.Symbols...) }
	sort.Slice(i.symbols, func(a,b int) bool { if i.symbols[a].File != i.symbols[b].File { return i.symbols[a].File < i.symbols[b].File }; return i.symbols[a].StartLine < i.symbols[b].StartLine })
	for _, s := range i.symbols { k := strings.ToLower(s.Name); i.byName[k] = append(i.byName[k], s) }
}
