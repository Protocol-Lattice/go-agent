package agent

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/Protocol-Lattice/go-agent/src/memory"
	"github.com/Protocol-Lattice/go-agent/src/models"
	"github.com/alpkeskin/gotoon"
)

var (
	roleUserRe      = regexp.MustCompile(`(?mi)^(?:User|User\s*\(quoted\))\s*:`)
	roleSystemRe    = regexp.MustCompile(`(?mi)^(?:System|System\s*\(quoted\))\s*:`)
	roleAssistantRe = regexp.MustCompile(`(?mi)^(?:Assistant|Assistant\s*\(quoted\))\s*:`)
	roleMemoryRe    = regexp.MustCompile(`(?mi)^Conversation memory`)
)

// buildPrompt assembles the full assistant prompt for normal LLM generation.
// It does NOT include Toon markup. It NEVER formats for tool calls.
// It simply injects system prompt, retrieved memory, and file context.
func (a *Agent) buildPrompt(
	ctx context.Context,
	sessionID string,
	userInput string,
) (string, error) {
	userInput = sanitizeInput(userInput)

	queryType := classifyQuery(userInput)

	var (
		records []memory.MemoryRecord
		err     error
	)

	switch queryType {
	case QueryMath:
		// Math usually does not need retrieved conversation context.

	case QueryShortFactoid:
		limit := intMin(a.contextLimit/2, 3)
		if limit > 0 {
			records, err = a.retrieveContext(ctx, sessionID, userInput, limit)
			if err != nil {
				return "", fmt.Errorf("retrieve context: %w", err)
			}
		}

	case QueryComplex:
		if a.contextLimit > 0 {
			records, err = a.retrieveContext(ctx, sessionID, userInput, a.contextLimit)
			if err != nil {
				return "", fmt.Errorf("retrieve context: %w", err)
			}
		}

	default:
		// Unknown query type: keep prompt lean and avoid accidental noisy retrieval.
	}

	var sb strings.Builder
	sb.Grow(4096)

	if strings.TrimSpace(a.systemPrompt) != "" {
		sb.WriteString(strings.TrimSpace(a.systemPrompt))
		sb.WriteString("\n\n")
	}

	if len(records) > 0 {
		sb.WriteString("Conversation memory (TOON):\n")
		sb.WriteString(a.renderMemory(records))
		sb.WriteString("\n\n")
	}

	files, err := a.RetrieveAttachmentFiles(ctx, sessionID, a.contextLimit)
	if err != nil {
		return "", fmt.Errorf("retrieve attachment files: %w", err)
	}

	if len(files) > 0 {
		sb.WriteString(a.buildAttachmentPrompt("Session attachments (rehydrated)", files))
		sb.WriteString("\n\n")
	}

	sb.WriteString("User: ")
	sb.WriteString(userInput)
	sb.WriteString("\n")

	return sb.String(), nil
}

func intMin(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// renderMemory formats retrieved memory records into a clean, token-efficient list.
func (a *Agent) renderMemory(records []memory.MemoryRecord) string {
	if len(records) == 0 {
		return "(no stored memory)\n"
	}

	entries := make([]map[string]any, 0, len(records))
	var fallback strings.Builder
	counter := 0
	for _, rec := range records {
		content := strings.TrimSpace(rec.Content)
		if content == "" {
			continue
		}
		counter++
		role := metadataRole(rec.Metadata)
		space := rec.Space
		if space == "" {
			space = rec.SessionID
		}
		entry := map[string]any{
			"id":          counter,
			"role":        role,
			"space":       space,
			"score":       rec.Score,
			"importance":  rec.Importance,
			"source":      rec.Source,
			"summary":     rec.Summary,
			"content":     content,
			"last_update": rec.LastEmbedded.UTC().Format(time.RFC3339Nano),
		}
		if rec.LastEmbedded.IsZero() {
			delete(entry, "last_update")
		}
		entries = append(entries, entry)
		fallback.WriteString(fmt.Sprintf("%d. [%s] %s\n", counter, role, escapePromptContent(content)))
	}
	if len(entries) == 0 {
		return "(no stored memory)\n"
	}
	if toon := encodeTOONBlock(map[string]any{"memories": entries}); toon != "" {
		return toon + "\n"
	}
	return fallback.String()
}

func escapePromptContent(s string) string {
	s = strings.ReplaceAll(s, "`", "'")
	s = roleUserRe.ReplaceAllString(s, "User (quoted):")
	s = roleSystemRe.ReplaceAllString(s, "System (quoted):")
	s = roleAssistantRe.ReplaceAllString(s, "Assistant (quoted):")
	s = roleMemoryRe.ReplaceAllString(s, "Conversation memory (quoted):")
	return s
}

func sanitizeInput(s string) string {
	s = strings.TrimSpace(s)
	s = roleUserRe.ReplaceAllString(s, "User (quoted):")
	s = roleSystemRe.ReplaceAllString(s, "System (quoted):")
	s = roleAssistantRe.ReplaceAllString(s, "Assistant (quoted):")
	s = roleMemoryRe.ReplaceAllString(s, "Conversation memory (quoted):")
	return s
}

func (a *Agent) storeAttachmentMemories(sessionID string, files []models.File) {
	waitMemoryStoreTasks(a.startAttachmentMemoryStores(sessionID, files))
}

// startAttachmentMemoryStores prepares every attachment concurrently while
// preserving file order when the tasks are committed.
func (a *Agent) startAttachmentMemoryStores(sessionID string, files []models.File) []*memoryStoreTask {
	tasks := make([]*memoryStoreTask, 0, len(files))
	for i, file := range files {
		name := strings.TrimSpace(file.Name)
		if name == "" {
			name = fmt.Sprintf("file_%d", i+1)
		}
		mime := strings.TrimSpace(file.MIME)
		content := buildAttachmentMemoryContent(name, mime, file.Data)
		extra := map[string]string{
			"source":   "file_upload",
			"filename": name,
		}
		if mime != "" {
			extra["mime"] = mime
		}
		if size := len(file.Data); size > 0 {
			extra["size_bytes"] = strconv.Itoa(size)
		}
		if len(file.Data) > 0 {
			extra["data_base64"] = base64.StdEncoding.EncodeToString(file.Data)
		}
		if isTextAttachment(mime, file.Data) {
			extra["text"] = "true"
		} else {
			extra["text"] = "false"
		}
		tasks = append(tasks, a.startMemoryStore(sessionID, "attachment", content, extra))
	}
	return tasks
}

func waitMemoryStoreTasks(tasks []*memoryStoreTask) {
	for _, task := range tasks {
		task.Wait()
	}
}

// RetrieveAttachmentFiles returns attachment files stored for the session.
// It reconstructs the original bytes from base64-encoded metadata, making it
// suitable for binary assets such as images and videos.
func (a *Agent) RetrieveAttachmentFiles(ctx context.Context, sessionID string, limit int) ([]models.File, error) {
	if a == nil || a.memory == nil {
		return nil, nil
	}
	if limit <= 0 {
		limit = a.contextLimit
		if limit <= 0 {
			limit = 8
		}
	}

	records, err := a.memory.RetrieveContext(ctx, sessionID, "", limit)
	if err != nil {
		return nil, err
	}

	var attachments []models.File
	for _, record := range records {
		file, ok := attachmentFromRecord(record)
		if !ok {
			continue
		}
		attachments = append(attachments, file)
	}

	return attachments, nil
}

func attachmentFromRecord(record memory.MemoryRecord) (models.File, bool) {
	if strings.TrimSpace(record.Metadata) == "" {
		return models.File{}, false
	}

	var payload struct {
		Role       string `json:"role"`
		Filename   string `json:"filename"`
		MIME       string `json:"mime"`
		DataBase64 string `json:"data_base64"`
	}
	if err := json.Unmarshal([]byte(record.Metadata), &payload); err != nil {
		return models.File{}, false
	}
	if payload.Role != "attachment" {
		return models.File{}, false
	}

	name := payload.Filename
	if name == "" {
		name = "attachment"
	}

	var data []byte
	if payload.DataBase64 != "" {
		raw, err := base64.StdEncoding.DecodeString(payload.DataBase64)
		if err != nil {
			return models.File{}, false
		}
		data = raw
	} else {
		data = extractTextAttachment(record.Content)
	}

	return models.File{Name: name, MIME: payload.MIME, Data: data}, true
}

func extractTextAttachment(content string) []byte {
	idx := strings.Index(content, ":\n")
	if idx == -1 {
		return nil
	}
	return []byte(content[idx+2:])
}

func isTextAttachment(mime string, data []byte) bool {
	mt := strings.ToLower(strings.TrimSpace(mime))
	switch {
	case strings.HasPrefix(mt, "text/"):
		return true
	case mt == "application/json",
		mt == "application/xml",
		mt == "application/x-yaml",
		mt == "application/yaml",
		mt == "text/markdown",
		mt == "text/x-markdown":
		return true
	}
	if len(data) == 0 {
		return true
	}
	return utf8.Valid(data)
}

func buildAttachmentMemoryContent(name, mime string, data []byte) string {
	display := strings.TrimSpace(name)
	if display == "" {
		display = "attachment"
	}
	descriptor := display
	if m := strings.TrimSpace(mime); m != "" {
		descriptor = fmt.Sprintf("%s (%s)", display, m)
	}
	if len(data) == 0 {
		return fmt.Sprintf("Attachment %s [empty file]", descriptor)
	}
	if isTextAttachment(mime, data) {
		var sb strings.Builder
		sb.Grow(len(data) + len(descriptor) + 32)
		sb.WriteString("Attachment ")
		sb.WriteString(descriptor)
		sb.WriteString(":\n")
		sb.Write(data)
		return sb.String()
	}
	return fmt.Sprintf("Attachment %s [%d bytes of non-text content]", descriptor, len(data))
}

func fileBackedWorkspaceRules(files []models.File) string {
	if len(files) == 0 {
		return ""
	}

	var sb strings.Builder
	sb.WriteString("Workspace file-selection rules:\n")
	sb.WriteString("Available attached workspace paths (authoritative for existing files):\n")

	seen := make(map[string]bool, len(files))
	for i, f := range files {
		name := strings.TrimSpace(f.Name)
		if name == "" {
			name = fmt.Sprintf("attachment_%d", i+1)
		}
		key := strings.ToLower(name)
		if seen[key] {
			continue
		}
		seen[key] = true
		sb.WriteString("- ")
		sb.WriteString(name)
		sb.WriteString("\n")
	}

	sb.WriteString("For fixes, refactors, and edits to existing code, choose targets from the attached path list first.\n")
	sb.WriteString("If the attached paths include the relevant existing source file, use that file as the refactor target instead of creating a new source path.\n")
	sb.WriteString("Treat paths mentioned as examples, e.g. after \"for example\", \"e.g.\", \"such as\", or \"like\", as illustrative unless they are attached paths or the user explicitly asks to create that exact path.\n")
	sb.WriteString("New companion files are allowed only when the task requires them, such as tests or docs, and should correspond to the chosen existing file when possible.\n")
	return sb.String()
}

// buildAttachmentPrompt renders a compact, token-conscious list of files.
// It never inlines non-text bytes. For text files, it shows a short preview.
func (a *Agent) buildAttachmentPrompt(title string, files []models.File) string {
	if len(files) == 0 {
		return ""
	}
	entries := make([]map[string]any, 0, len(files))
	for i, f := range files {
		name := strings.TrimSpace(f.Name)
		if name == "" {
			name = fmt.Sprintf("attachment_%d", i+1)
		}
		mime := strings.TrimSpace(f.MIME)
		if mime == "" {
			mime = "application/octet-stream"
		}
		sizeBytes := len(f.Data)
		isText := isTextAttachment(mime, f.Data)
		preview := ""
		if isText && len(f.Data) > 0 {
			preview = previewText(mime, f.Data)
		}
		entry := map[string]any{
			"id":         i + 1,
			"name":       name,
			"mime":       mime,
			"size_bytes": sizeBytes,
			"text":       isText,
		}
		if preview != "" {
			entry["preview"] = preview
		}
		entries = append(entries, entry)
	}

	var sb strings.Builder
	sb.WriteString("\n\n")
	sb.WriteString(title)
	sb.WriteString(":\n")
	if toon := encodeTOONBlock(map[string]any{"files": entries}); toon != "" {
		sb.WriteString(indentBlock(toon, "  "))
		sb.WriteString("\n")
	} else {
		sb.WriteString(renderAttachmentFallback(files))
	}
	return sb.String()
}

func renderAttachmentFallback(files []models.File) string {
	var sb strings.Builder
	for i, file := range files {
		name := strings.TrimSpace(file.Name)
		if name == "" {
			name = fmt.Sprintf("attachment_%d", i+1)
		}
		mime := strings.TrimSpace(file.MIME)
		if mime == "" {
			mime = "application/octet-stream"
		}

		sb.WriteString(fmt.Sprintf("- %s (%s, %s)", name, mime, humanSize(len(file.Data))))
		if isTextAttachment(mime, file.Data) && len(file.Data) > 0 {
			sb.WriteString("\n  preview:\n  ")
			sb.WriteString(escapePromptContent(previewText(mime, file.Data)))
		}
		sb.WriteString("\n")
	}
	return sb.String()
}

func humanSize(n int) string {
	const (
		KB = 1024
		MB = 1024 * KB
		GB = 1024 * MB
	)
	switch {
	case n >= GB:
		return fmt.Sprintf("%.2f GB", float64(n)/float64(GB))
	case n >= MB:
		return fmt.Sprintf("%.2f MB", float64(n)/float64(MB))
	case n >= KB:
		return fmt.Sprintf("%.2f KB", float64(n)/float64(KB))
	default:
		return fmt.Sprintf("%d B", n)
	}
}

// previewText returns a short snippet from text attachments (max ~1KB) to save tokens.
func previewText(_ string, data []byte) string {
	const maxPreview = 1024
	txt := string(data)
	return truncate(txt, maxPreview)
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	// Try to cut on a boundary to avoid mid-rune issues for safety.
	if max > 3 {
		return s[:max-3] + "..."
	}
	return s[:max]
}

func encodeTOONBlock(value any) string {
	if value == nil {
		return ""
	}
	if encoded, err := gotoon.Encode(value); err == nil {
		return strings.TrimSpace(encoded)
	}
	if fallback, err := json.MarshalIndent(value, "", "  "); err == nil {
		return strings.TrimSpace(string(fallback))
	}
	return ""
}

func indentBlock(text, prefix string) string {
	text = strings.TrimRight(text, "\n")
	if text == "" {
		return ""
	}
	lines := strings.Split(text, "\n")
	for i := range lines {
		lines[i] = prefix + lines[i]
	}
	return strings.Join(lines, "\n")
}
