package workspace

import (
	"path/filepath"
	"sort"
	"strings"
)

// Dependencies returns source files imported by path when their import path
// belongs to the indexed Go module. External dependencies are intentionally
// omitted because they are not represented by source files in this index.
func (i *Index) Dependencies(path string) []string {
	path = filepath.ToSlash(path)
	f, ok := i.files[path]
	if !ok || i.module == "" {
		return nil
	}
	prefix := strings.TrimSuffix(i.module, "/") + "/"
	var out []string
	for _, imp := range f.Imports {
		if !strings.HasPrefix(imp, prefix) { continue }
		suffix := strings.TrimPrefix(imp, prefix)
		candidate := filepath.ToSlash(filepath.Join(suffix, ""))
		for p, indexed := range i.files {
			dir := strings.TrimSuffix(filepath.ToSlash(filepath.Dir(p)), ".")
			if dir == candidate || strings.HasPrefix(dir, candidate) {
				if indexed.Package != "" { out = append(out, p) }
			}
		}
	}
	sort.Strings(out)
		return uniqueStrings(out)
}

// Dependents returns indexed files that import path's package.
func (i *Index) Dependents(path string) []string {
	path = filepath.ToSlash(path)
	target, ok := i.files[path]
	if !ok || i.module == "" { return nil }
	dir := filepath.ToSlash(filepath.Dir(path))
	if dir == "." { dir = "" }
	importPath := strings.TrimSuffix(i.module, "/")
	if dir != "" { importPath += "/" + dir }
	var out []string
	for p, imports := range i.imports {
		for _, imp := range imports {
			if imp == importPath { out = append(out, p); break }
		}
	}
	_ = target
	sort.Strings(out)
	return uniqueStrings(out)
}

func uniqueStrings(values []string) []string {
	if len(values) < 2 { return values }
	out := values[:1]
	for _, v := range values[1:] { if v != out[len(out)-1] { out = append(out, v) } }
	return out
}
