// main.go — fixed agent-mode with provider flag
// Runs your Agent with optional file context using GenerateWithFiles.
// - Defaults to Gemini, but you can switch provider/model via flags.
// - Loads trailing CLI args as files; text is inlined, non-text is referenced.
//
// Examples:
//
//	export GOOGLE_API_KEY=...   # or GEMINI_API_KEY
//	go run . -message "Summarize the attachments" context/notes.md
//
//	export OPENAI_API_KEY=...
//	go run . -provider openai -model gpt-4o-mini -message "Brief" docs/notes.md
//
// Qdrant (memory) defaults:
//
//	-url http://localhost:6333 -collection adk_memories
package main

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"mime"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"time"
	"unicode/utf8"

	agent "github.com/Protocol-Lattice/go-agent"
	"github.com/Protocol-Lattice/go-agent/src/adk"
	"github.com/Protocol-Lattice/go-agent/src/adk/modules"
	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/Protocol-Lattice/go-agent/src/tools"
)

var (
	flagProvider     = flag.String("provider", "gemini", "LLM provider: openai|gemini|anthropic|ollama|dummy")
	flagModel        = flag.String("model", "gemini-2.5-pro", "Model ID for the selected provider")
	flagPrefix       = flag.String("prefix", "", "Optional system/prompt prefix")
	flagSession      = flag.String("session", "default", "Session ID for conversation continuity")
	flagMessage      = flag.String("message", "", "User message (ignored if -stdin is set)")
	flagStdin        = flag.Bool("stdin", false, "Read user message from STDIN")
	flagJSON         = flag.Bool("json", false, "Print JSON {response, provider, model}")
	flagTimeout      = flag.Duration("timeout", 90*time.Second, "Overall request timeout")
	qdrantURL        = flag.String("qdrant-url", "http://localhost:6333", "Qdrant base URL")
	qdrantCollection = flag.String("qdrant-collection", "adk_memories", "Qdrant collection name")
)

func main() {
	flag.Parse()

	ctx, cancel := context.WithTimeout(context.Background(), *flagTimeout)
	defer cancel()

	// 1) Message
	msg, err := getMessage(*flagMessage, *flagStdin, os.Stdin)
	if err != nil {
		fail(err)
	}

	// 2) Files
	files, err := loadFiles(flag.Args()...)
	if err != nil {
		fail(err)
	}
	if strings.TrimSpace(msg) == "" && len(files) == 0 {
		fail(errors.New("no message and no files provided"))
	}

	// 3) Build ADK with a model provider module bound to your flags
	memOpts := engine.DefaultOptions()

	kit, err := adk.New(ctx,
		adk.WithDefaultSystemPrompt("You orchestrate a helpful assistant team."),
		adk.WithModules(
			modules.NewModelModule("llm", func(c context.Context) (models.Agent, error) {
				// Provider-agnostic: openai|gemini|anthropic|ollama|dummy
				return models.NewLLMProvider(c, strings.ToLower(*flagProvider), *flagModel, "Swarm orchestration:")
			}),
			modules.InQdrantMemory(100000, *qdrantURL, *qdrantCollection, memory.AutoEmbedder(), &memOpts),
			modules.NewToolModule("essentials", modules.StaticToolProvider([]agent.Tool{&tools.EchoTool{}}, nil)),
		),
	)
	if err != nil {
		fail(fmt.Errorf("adk.New: %w", err))
	}

	// 4) Build an agent and run one turn with ephemeral files
	ag, err := kit.BuildAgent(ctx)
	if err != nil {
		fail(fmt.Errorf("build agent: %w", err))
	}

	out, err := ag.GenerateWithFiles(ctx, *flagSession, withPrefix(*flagPrefix, *flagProvider, *flagModel, msg), files)
	if err != nil {
		fail(err)
	}

	// 5) Print
	if *flagJSON {
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(map[string]any{
			"response": out,
			"provider": *flagProvider,
			"model":    *flagModel,
		})
		return
	}
	fmt.Println(out)
}

func getMessage(flagMsg string, useStdin bool, r io.Reader) (string, error) {
	if useStdin {
		var b strings.Builder
		sc := bufio.NewScanner(r)
		for sc.Scan() {
			b.WriteString(sc.Text())
			b.WriteByte('\n')
		}
		if err := sc.Err(); err != nil {
			return "", err
		}
		return strings.TrimRight(b.String(), "\n"), nil
	}
	return flagMsg, nil
}

// loadFiles converts paths → []models.File with best-effort MIME detection.
func loadFiles(paths ...string) ([]models.File, error) {
	var out []models.File
	for _, p := range paths {
		if strings.TrimSpace(p) == "" {
			continue
		}
		data, err := os.ReadFile(p)
		if err != nil {
			return nil, fmt.Errorf("read %s: %w", p, err)
		}
		m := mime.TypeByExtension(strings.ToLower(filepath.Ext(p)))
		if m == "" {
			peek := data
			if len(peek) > 512 {
				peek = peek[:512]
			}
			m = http.DetectContentType(peek)
		}
		if (m == "" || m == "application/octet-stream") && isLikelyText(data) {
			m = "text/plain; charset=utf-8"
		}
		out = append(out, models.File{Name: filepath.Base(p), MIME: m, Data: data})
	}
	return out, nil
}

func isLikelyText(b []byte) bool {
	if len(b) == 0 || !utf8.Valid(b) {
		return false
	}
	const max = 1024
	limit := len(b)
	if limit > max {
		limit = max
	}
	nul := 0
	for i := 0; i < limit; i++ {
		if b[i] == 0 {
			nul++
			if nul > 1 {
				return false
			}
		}
	}
	return true
}

// withPrefix optionally prepends a small header (provider/model) before the message.
func withPrefix(prefix, provider, model, msg string) string {
	prefix = strings.TrimSpace(prefix)
	if prefix == "" {
		return msg
	}
	return fmt.Sprintf("%s\n\n[provider=%s model=%s]\n\n%s", prefix, provider, model, msg)
}

func fail(err error) {
	fmt.Fprintln(os.Stderr, "error:", err)
	os.Exit(1)
}
