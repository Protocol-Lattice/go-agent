package workspace

import (
	"bufio"
	"context"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Config controls how a WorkspaceIndex scans a repository.
type Config struct {
	Root        string
	Extensions  []string
	IgnoreDirs  []string
	MaxFileSize int64
}

func DefaultConfig(root string) Config {
	return Config{
		Root:        root,
		Extensions:  []string{".go"},
		IgnoreDirs:  []string{".git", "vendor", "node_modules", "dist", "build", "tmp"},
		MaxFileSize: 2 << 20,
	}
}

type File struct {
	Path    string
	Package string
	Bytes   int64
	Symbols []Symbol
	Imports []string
}

type SymbolKind string

const (
	SymbolPackage SymbolKind = "package"
	SymbolType    SymbolKind = "type"
	SymbolFunc    SymbolKind = "function"
	SymbolMethod  SymbolKind = "method"
	SymbolVar     SymbolKind = "var"
	SymbolConst   SymbolKind = "const"
)

type Symbol struct {
	Name      string
	Kind      SymbolKind
	Package   string
	File      string
	StartLine int
	EndLine   int
	Receiver  string
}

type Edge struct {
	From string
	To   string
}

type SearchResult struct {
	Symbol Symbol
	Score  int
}

type ContextRequest struct {
	Query     string
	MaxBytes  int
	MaxFiles  int
	MaxResults int
}

type ContextFile struct {
	Path    string
	Content string
	Symbols []Symbol
}

type Context struct {
	Query   string
	Files   []ContextFile
	Results []SearchResult
}

// Index is an in-memory representation of a source tree. It is deliberately
// storage-agnostic so callers can persist or enrich it with embeddings later.
type Index struct {
	root       string
	config     Config
	files      map[string]File
	symbols    []Symbol
	byName     map[string][]Symbol
	imports    map[string][]string
	module     string
}

func NewIndex(config Config) *Index {
	if config.Root == "" {
		config.Root = "."
	}
	if len(config.Extensions) == 0 {
		config.Extensions = []string{".go"}
	}
	if len(config.IgnoreDirs) == 0 {
		config.IgnoreDirs = DefaultConfig(config.Root).IgnoreDirs
	}
	if config.MaxFileSize <= 0 {
		config.MaxFileSize = 2 << 20
	}
	return &Index{
		root: config.Root, config: config,
		files: make(map[string]File), byName: make(map[string][]Symbol), imports: make(map[string][]string),
	}
}

func (i *Index) Build(ctx context.Context) error {
	if ctx == nil {
		ctx = context.Background()
	}
	i.files = make(map[string]File)
	i.symbols = nil
	i.byName = make(map[string][]Symbol)
	i.imports = make(map[string][]string)
	i.module = readModule(i.root)

	err := filepath.Walk(i.root, func(path string, info os.FileInfo, err error) error {
		if err != nil { return err }
		select { case <-ctx.Done(): return ctx.Err(); default: }
		if info.IsDir() {
			if path != i.root && i.ignored(path) { return filepath.SkipDir }
			return nil
		}
		if info.Size() > i.config.MaxFileSize || !i.supported(path) { return nil }
		return i.indexFile(path, info.Size())
	})
	if err != nil { return err }
	sort.Slice(i.symbols, func(a, b int) bool {
		if i.symbols[a].File == i.symbols[b].File { return i.symbols[a].StartLine < i.symbols[b].StartLine }
		return i.symbols[a].File < i.symbols[b].File
	})
	return nil
}

func (i *Index) indexFile(path string, size int64) error {
	fset := token.NewFileSet()
	f, err := parser.ParseFile(fset, path, nil, parser.ParseComments)
	if err != nil { return fmt.Errorf("parse %s: %w", path, err) }
	rel, err := filepath.Rel(i.root, path)
	if err != nil { return err }
	rel = filepath.ToSlash(rel)
	file := File{Path: rel, Package: f.Name.Name, Bytes: size}
	file.Symbols = append(file.Symbols, Symbol{Name: f.Name.Name, Kind: SymbolPackage, Package: f.Name.Name, File: rel, StartLine: line(fset, f.Pos()), EndLine: line(fset, f.End())})
	for _, imp := range f.Imports {
		p := strings.Trim(imp.Path.Value, "\"")
		file.Imports = append(file.Imports, p)
	}
	ast.Inspect(f, func(n ast.Node) bool {
		var s Symbol
		switch n := n.(type) {
		case *ast.FuncDecl:
			s.Name, s.Package, s.File, s.StartLine, s.EndLine = n.Name.Name, f.Name.Name, rel, line(fset, n.Pos()), line(fset, n.End())
			if n.Recv != nil && len(n.Recv.List) > 0 {
				s.Kind = SymbolMethod
				s.Receiver = receiverName(n.Recv.List[0].Type)
			} else { s.Kind = SymbolFunc }
		case *ast.TypeSpec:
			s = Symbol{Name: n.Name.Name, Kind: SymbolType, Package: f.Name.Name, File: rel, StartLine: line(fset, n.Pos()), EndLine: line(fset, n.End())}
		case *ast.ValueSpec:
			kind := SymbolVar
			if gd, ok := parentGenDecl(n); ok && gd.Tok == token.CONST { kind = SymbolConst }
			for _, name := range n.Names {
				ss := Symbol{Name: name.Name, Kind: kind, Package: f.Name.Name, File: rel, StartLine: line(fset, n.Pos()), EndLine: line(fset, n.End())}
				file.Symbols = append(file.Symbols, ss)
			}
		}
		if s.Name != "" { file.Symbols = append(file.Symbols, s) }
		return true
	})
	i.files[rel] = file
	for _, s := range file.Symbols { i.symbols = append(i.symbols, s); i.byName[strings.ToLower(s.Name)] = append(i.byName[strings.ToLower(s.Name)], s) }
	i.imports[rel] = append([]string(nil), file.Imports...)
	return nil
}

// SearchSymbols performs lightweight lexical ranking over indexed symbols.
func (i *Index) SearchSymbols(_ context.Context, query string, limit int) []SearchResult {
	query = strings.TrimSpace(strings.ToLower(query))
	if query == "" { return nil }
	if limit <= 0 { limit = 20 }
	results := make([]SearchResult, 0)
	for _, s := range i.symbols {
		name := strings.ToLower(s.Name)
		score := 0
		switch {
		case name == query: score = 100
		case strings.HasPrefix(name, query): score = 80
		case strings.Contains(name, query): score = 60
		case strings.Contains(strings.ToLower(s.Package), query): score = 25
		default: continue
		}
		results = append(results, SearchResult{Symbol: s, Score: score})
	}
	sort.SliceStable(results, func(a, b int) bool {
		if results[a].Score != results[b].Score { return results[a].Score > results[b].Score }
		return results[a].Symbol.Name < results[b].Symbol.Name
	})
	if len(results) > limit { results = results[:limit] }
	return results
}

func (i *Index) Files() []File {
	out := make([]File, 0, len(i.files))
	for _, f := range i.files { out = append(out, f) }
	sort.Slice(out, func(a,b int) bool { return out[a].Path < out[b].Path })
	return out
}

func (i *Index) File(path string) (File, bool) { f, ok := i.files[filepath.ToSlash(path)]; return f, ok }
func (i *Index) Symbols() []Symbol { return append([]Symbol(nil), i.symbols...) }
func (i *Index) Imports(path string) []string { return append([]string(nil), i.imports[filepath.ToSlash(path)]...) }
func (i *Index) Module() string { return i.module }

// BuildContext resolves a task to the most relevant source files without
// requiring an embedding provider. It is designed as the deterministic first
// stage before optional semantic/vector retrieval.
func (i *Index) BuildContext(ctx context.Context, req ContextRequest) (Context, error) {
	if req.MaxBytes <= 0 { req.MaxBytes = 64 << 10 }
	if req.MaxFiles <= 0 { req.MaxFiles = 8 }
	if req.MaxResults <= 0 { req.MaxResults = 20 }
	results := i.SearchSymbols(ctx, req.Query, req.MaxResults)
	selected := make(map[string]bool)
	for _, r := range results { if len(selected) >= req.MaxFiles { break }; selected[r.Symbol.File] = true }
	// Fall back to a deterministic lexical file scan when no symbols match.
	if len(selected) == 0 {
		q := strings.ToLower(req.Query)
		for _, f := range i.Files() {
			if len(selected) >= req.MaxFiles { break }
			b, err := os.ReadFile(filepath.Join(i.root, filepath.FromSlash(f.Path))); if err != nil { continue }
			if strings.Contains(strings.ToLower(string(b)), q) { selected[f.Path] = true }
		}
	}
	paths := make([]string, 0, len(selected)); for p := range selected { paths = append(paths, p) }; sort.Strings(paths)
	out := Context{Query: req.Query, Results: results}
	var used int
	for _, p := range paths {
		if err := ctx.Err(); err != nil { return Context{}, err }
		b, err := os.ReadFile(filepath.Join(i.root, filepath.FromSlash(p))); if err != nil { return Context{}, err }
		if used+len(b) > req.MaxBytes { b = b[:max(0, req.MaxBytes-used)] }
		f := i.files[p]
		out.Files = append(out.Files, ContextFile{Path: p, Content: string(b), Symbols: f.Symbols})
		used += len(b)
		if used >= req.MaxBytes { break }
	}
	return out, nil
}

func (i *Index) supported(path string) bool { for _, ext := range i.config.Extensions { if strings.EqualFold(filepath.Ext(path), ext) { return true } }; return false }
func (i *Index) ignored(path string) bool { for _, d := range i.config.IgnoreDirs { if path == filepath.Join(i.root, d) || strings.HasPrefix(path, filepath.Join(i.root, d)+string(os.PathSeparator)) { return true } }; return false }
func line(fset *token.FileSet, pos token.Pos) int { return fset.Position(pos).Line }
func receiverName(expr ast.Expr) string { switch t := expr.(type) { case *ast.Ident: return t.Name; case *ast.StarExpr: if id, ok := t.X.(*ast.Ident); ok { return id.Name }; case *ast.IndexExpr: if id, ok := t.X.(*ast.Ident); ok { return id.Name } }; return "" }
func parentGenDecl(*ast.ValueSpec) (*ast.GenDecl, bool) { return nil, false }
func readModule(root string) string { f, err := os.Open(filepath.Join(root, "go.mod")); if err != nil { return "" }; defer f.Close(); s := bufio.NewScanner(f); for s.Scan() { fields := strings.Fields(s.Text()); if len(fields) == 2 && fields[0] == "module" { return fields[1] } }; return "" }

func max(a, b int) int { if a > b { return a }; return b }
