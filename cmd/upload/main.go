// main.go
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/Raezil/go-agent-development-kit/pkg/adk"
	"github.com/Raezil/go-agent-development-kit/pkg/adk/modules"
	"github.com/Raezil/go-agent-development-kit/pkg/agent"
	"github.com/Raezil/go-agent-development-kit/pkg/memory"
	"github.com/Raezil/go-agent-development-kit/pkg/memory/engine"
	"github.com/Raezil/go-agent-development-kit/pkg/models"
	"github.com/Raezil/go-agent-development-kit/pkg/subagents"
	"github.com/Raezil/go-agent-development-kit/pkg/tools"
	contextio "github.com/Raezil/go-agent-development-kit/pkg/upload"
)

// ---- output structs ----

type ChunkView struct {
	Index int               `json:"index"`
	Text  string            `json:"text,omitempty"`
	Meta  map[string]string `json:"meta,omitempty"`
}

type UploadItem struct {
	File       string      `json:"file"`
	SizeBytes  int64       `json:"size_bytes"`
	MIME       string      `json:"mime"`
	Space      string      `json:"space"`
	Scope      string      `json:"scope"`
	Tags       []string    `json:"tags"`
	ChunkIDs   []string    `json:"chunk_ids"`
	ChunkCount int         `json:"chunk_count"`
	Chunks     []ChunkView `json:"chunks,omitempty"`
	Err        string      `json:"error,omitempty"`
}

type UploadReport struct {
	Items       []UploadItem `json:"items"`
	TotalChunks int          `json:"total_chunks"`
	Failed      int          `json:"failed"`
}

func main() {
	// flags
	space := flag.String("space", "team:shared", "Target shared space (e.g. team:shared)")
	scope := flag.String("scope", "space", "Context scope: local|space|global")
	ttl := flag.String("ttl", "", "Optional TTL (e.g. 168h). Empty = no TTL")
	tagsCSV := flag.String("tags", "", "Comma-separated tags")
	storeDir := flag.String("store-dir", "./data/uploads", "Directory to persist original files")
	redact := flag.Bool("redact", false, "Enable simple regex PII redaction before embedding")
	mimeOverride := flag.String("mime", "", "Force MIME type for all inputs (optional)")
	quiet := flag.Bool("quiet", false, "Suppress per-file logs; only print final JSON report")
	printText := flag.Bool("print-text", true, "Include extracted chunk text in the JSON report")
	textMax := flag.Int("text-max", 300, "If >0, max characters per chunk to include in JSON")
	qdrantURL := flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection := flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
	modelName := flag.String("model", "gemini-2.5-pro", "Gemini model ID")

	flag.Parse()

	if flag.NArg() == 0 {
		fmt.Println("Usage: go run [-tags=pdf] main.go [flags] <file1> [file2 ...]")
		flag.PrintDefaults()
		os.Exit(2)
	}

	// wire providers (replace stubs with real integrations later)
	embed := newYourEmbeddingsProvider() // wrap your real embeddings (OpenAI/Gemini/etc.)
	writer := newYourMemoryWriter()      // write into your memory engine (Qdrant/pgvector)
	store := contextio.FSStore{BaseDir: *storeDir}
	ing := contextio.NewDefaultIngestor(embed, writer, store)
	// ensure PDFs allowed (constructor already can include ".pdf" if you added it)
	if ing.AllowedExtsLower != nil {
		ing.AllowedExtsLower[".pdf"] = struct{}{}
	}
	if *redact {
		ing.Redactor = contextio.NewDefaultRedactor()
	}

	var ttlPtr *time.Duration
	if *ttl != "" {
		d, err := time.ParseDuration(*ttl)
		check(err)
		ttlPtr = &d
	}
	tags := splitCSV(*tagsCSV)
	sc := scopeFrom(*scope)

	report := UploadReport{Items: make([]UploadItem, 0, flag.NArg())}
	exitCode := 0

	for _, path := range flag.Args() {
		item := UploadItem{
			File:  path,
			Space: *space,
			Scope: *scope,
			Tags:  append([]string(nil), tags...),
		}

		data, err := os.ReadFile(path)
		if err != nil {
			item.Err = fmt.Sprintf("read: %v", err)
			report.Items = append(report.Items, item)
			exitCode = 1
			if !*quiet {
				fmt.Fprintf(os.Stderr, "✖ read %s: %v\n", path, err)
			}
			continue
		}
		item.SizeBytes = int64(len(data))

		mime := *mimeOverride
		if mime == "" {
			mime = detectMIME(data)
		}
		item.MIME = mime

		// Optional: preview extracted chunks (same logic as ingestor path)
		var previewChunks []ChunkView
		if *printText {
			doc := &contextio.Document{
				Name:      filepath.Base(path),
				MIME:      mime,
				SizeBytes: int64(len(data)),
				Source:    "upload://local",
				Reader:    bytes.NewReader(data),
			}
			var ex contextio.Extractor
			for _, cand := range ing.Extractors {
				if cand.Supports(doc.MIME) {
					ex = cand
					break
				}
			}
			if ex == nil {
				item.Err = "no extractor for MIME: " + doc.MIME + " (did you run with -tags=pdf?)"
				report.Items = append(report.Items, item)
				exitCode = 1
				if !*quiet {
					fmt.Fprintf(os.Stderr, "✖ ingest %s: %v\n", path, item.Err)
				}
				continue
			}
			blocks, err := ex.Extract(doc)
			if err != nil {
				item.Err = "extract: " + err.Error()
				report.Items = append(report.Items, item)
				exitCode = 1
				if !*quiet {
					fmt.Fprintf(os.Stderr, "✖ extract %s: %v\n", path, err)
				}
				continue
			}
			chunks, err := ing.Chunker.Chunk(blocks)
			if err != nil {
				item.Err = "chunk: " + err.Error()
				report.Items = append(report.Items, item)
				exitCode = 1
				if !*quiet {
					fmt.Fprintf(os.Stderr, "✖ chunk %s: %v\n", path, err)
				}
				continue
			}
			// optional redaction preview
			if ing.Redactor != nil {
				for i := range chunks {
					if red, changed := ing.Redactor.Redact(chunks[i].Text); changed {
						chunks[i].Text = red
						if chunks[i].Meta == nil {
							chunks[i].Meta = map[string]string{}
						}
						chunks[i].Meta["redacted"] = "true"
					}
				}
			}
			previewChunks = make([]ChunkView, 0, len(chunks))
			for i := range chunks {
				txt := chunks[i].Text
				if *textMax > 0 && len(txt) > *textMax {
					txt = txt[:*textMax]
				}
				previewChunks = append(previewChunks, ChunkView{
					Index: chunks[i].Index,
					Text:  txt,
					Meta:  chunks[i].Meta,
				})
			}
		}

		// Persist via full ingest pipeline (embeds + write to memory/vector)
		ids, err := ing.IngestReader(
			filepath.Base(path),
			mime,
			bytes.NewReader(data),
			contextio.IngestOptions{
				Space: *space,
				Scope: sc,
				Tags:  tags,
				TTL:   ttlPtr,
				Meta:  map[string]string{"filename": filepath.Base(path)},
			},
		)
		if err != nil {
			item.Err = err.Error()
			report.Items = append(report.Items, item)
			exitCode = 1
			if !*quiet {
				fmt.Fprintf(os.Stderr, "✖ ingest %s: %v\n", path, err)
			}
			continue
		}

		item.ChunkIDs = ids
		item.ChunkCount = len(ids)
		if *printText {
			item.Chunks = previewChunks
		}
		report.TotalChunks += len(ids)
		report.Items = append(report.Items, item)

		if !*quiet {
			fmt.Printf("✔ %s → %d chunks indexed into %q (scope=%s)\n", path, len(ids), *space, *scope)
		}
	}

	for _, it := range report.Items {
		if it.Err != "" {
			report.Failed++
		}
	}
	ctx := context.Background()

	// Final machine-readable report
	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	if err := enc.Encode(report); err != nil {
		fmt.Fprintf(os.Stderr, "error: cannot encode JSON report: %v\n", err)
	}
	researcherModel, err := models.NewGeminiLLM(ctx, *modelName, "Research summary:")
	if err != nil {
		log.Fatalf("create researcher model: %v", err)
	}
	memOpts := engine.DefaultOptions()
	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		adk.WithSubAgents(subagents.NewResearcher(researcherModel)),
		adk.WithModules(
			modules.NewModelModule("gemini-model", func(_ context.Context) (models.Agent, error) {
				return models.NewGeminiLLM(ctx, *modelName, "Swarm orchestration:")
			}),
			modules.InQdrantMemory(100000, *qdrantURL, *qdrantCollection, memory.AutoEmbedder(), &memOpts),
			modules.NewToolModule("essentials", modules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
		adk.WithIngestor(ing),
	)
	agCore, err := kit.BuildAgent(ctx)
	if err != nil {
		log.Fatalf("build agent %q: %v", "alias", err)
	}
	var atts []agent.Attachment
	for _, p := range flag.Args() {
		data, err := os.ReadFile(p)
		if err != nil {
			log.Printf("skip attach %s: %v", p, err)
			continue
		}
		atts = append(atts, agent.Attachment{
			Name:   filepath.Base(p),
			MIME:   detectMIME(data),      // or your MIME detector
			Reader: bytes.NewReader(data), // or Bytes: data, depending on your struct
			// Optional: Size, Meta, Tags, etc., if your struct supports them
		})
	}
	res, err := agCore.GenerateWithAttachments(ctx, *space, "Summarize uploaded files", atts)
	if err != nil {
		log.Fatalf("generate with attachments: %v", err)
	}
	printAgentResponse(res)

	// Optional: persist for later inspection
	_ = os.WriteFile("summary.txt", []byte(res), 0644)

	fmt.Println(res)
	os.Exit(exitCode)
}
func printAgentResponse(res string) {
	fmt.Println("\n--- Agent Response ---")
	// Pretty-print if it looks like JSON; otherwise just print raw
	if json.Valid([]byte(res)) {
		var v any
		if err := json.Unmarshal([]byte(res), &v); err == nil {
			b, _ := json.MarshalIndent(v, "", "  ")
			fmt.Println(string(b))
		} else {
			fmt.Println(res)
		}
	} else {
		fmt.Println(res)
	}
}

// ---- helpers ----

func splitCSV(s string) []string {
	if s == "" {
		return nil
	}
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if v := strings.TrimSpace(p); v != "" {
			out = append(out, v)
		}
	}
	return out
}

func scopeFrom(s string) contextio.Scope {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "local":
		return contextio.ScopeLocal
	case "global":
		return contextio.ScopeGlobal
	default:
		return contextio.ScopeSpace
	}
}

func detectMIME(data []byte) string {
	if len(data) == 0 {
		return "application/octet-stream"
	}
	head := data
	if len(head) > 512 {
		head = head[:512]
	}
	return http.DetectContentType(head)
}

func check(err error) {
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(1)
	}
}

// ---- stub providers (replace with real ones) ----

type yourEmbedder struct{}

func newYourEmbeddingsProvider() *yourEmbedder { return &yourEmbedder{} }

func (e *yourEmbedder) EmbedTexts(texts []string) ([][]float32, error) {
	const dim = 8
	out := make([][]float32, len(texts))
	for i := range texts {
		v := make([]float32, dim)
		for j := 0; j < dim; j++ {
			v[j] = float32((len(texts[i])+j)%7) / 10.0
		}
		out[i] = v
	}
	return out, nil
}

type yourWriter struct{}

func newYourMemoryWriter() *yourWriter { return &yourWriter{} }

// Replace with real upserts to Qdrant/pgvector + graph edges (Document->Chunk[i]).
func (w *yourWriter) WriteChunks(space string, scope contextio.Scope, chunks []contextio.Chunk, vectors [][]float32, ttl *time.Duration, docMeta map[string]string) ([]string, error) {
	ids := make([]string, len(chunks))
	for i := range chunks {
		ids[i] = fmt.Sprintf("%s#%d", chunks[i].DocName, chunks[i].Index)
	}
	return ids, nil
}
