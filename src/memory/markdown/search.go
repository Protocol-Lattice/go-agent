package markdown

import (
	"encoding/base64"
	"encoding/json"
	"regexp"
	"sort"
	"strings"
	"time"
)

type scoredRecord struct {
	Record Record
	Score  int
}

var wordRE = regexp.MustCompile(`[a-zA-Z0-9_./:-]+`)

func scoreRecords(records []Record, query string) []scoredRecord {
	terms := wordRE.FindAllString(strings.ToLower(query), -1)
	if len(terms) == 0 {
		return nil
	}

	var scored []scoredRecord

	for _, rec := range records {
		haystack := strings.ToLower(rec.Role + " " + strings.Join(rec.Tags, " ") + " " + rec.Content)

		score := 0
		for _, term := range terms {
			if strings.Contains(haystack, term) {
				score += 10
			}
		}

		if strings.Contains(haystack, strings.ToLower(query)) {
			score += 25
		}

		if score == 0 {
			continue
		}

		// Small recency boost.
		if time.Since(rec.CreatedAt) < 24*time.Hour {
			score += 3
		}

		scored = append(scored, scoredRecord{
			Record: rec,
			Score:  score,
		})
	}

	sort.SliceStable(scored, func(i, j int) bool {
		if scored[i].Score == scored[j].Score {
			return scored[i].Record.CreatedAt.After(scored[j].Record.CreatedAt)
		}
		return scored[i].Score > scored[j].Score
	})

	return scored
}

func parseBlocks(doc string) []Record {
	lines := readLines(doc)

	var out []Record
	var current *Record
	var body strings.Builder
	inBlock := false
	inContent := false

	for _, line := range lines {
		if strings.HasPrefix(line, "<!-- memory:") {
			rec := parseMeta(line)
			current = &rec
			body.Reset()
			inBlock = true
			inContent = false
			continue
		}

		if strings.TrimSpace(line) == "<!-- /memory -->" {
			if inBlock && current != nil {
				current.Content = strings.TrimSpace(body.String())
				out = append(out, current.normalized())
			}
			current = nil
			body.Reset()
			inBlock = false
			inContent = false
			continue
		}

		if !inBlock {
			continue
		}

		if strings.HasPrefix(line, "## ") && !inContent {
			inContent = true
			continue
		}

		if inContent {
			body.WriteString(line)
			body.WriteByte('\n')
		}
	}

	return out
}

func parseMeta(line string) Record {
	line = strings.TrimPrefix(line, "<!-- memory:")
	line = strings.TrimSuffix(line, "-->")
	line = strings.TrimSpace(line)

	fields := map[string]string{}

	for _, part := range strings.Fields(line) {
		k, v, ok := strings.Cut(part, "=")
		if !ok {
			continue
		}
		fields[k] = v
	}

	var rec Record
	rec.ID = fields["id"]
	rec.Role = fields["role"]

	if tags := fields["tags"]; tags != "" {
		for _, tag := range strings.Split(tags, ",") {
			tag = strings.TrimSpace(tag)
			if tag != "" {
				rec.Tags = append(rec.Tags, tag)
			}
		}
	}

	if ts := fields["ts"]; ts != "" {
		if parsed, err := time.Parse(time.RFC3339, ts); err == nil {
			rec.CreatedAt = parsed
		}
	}

	// Decode embedding
	if embStr := fields["embedding"]; embStr != "" {
		if data, err := base64.StdEncoding.DecodeString(embStr); err == nil {
			var embedding []float32
			if err := json.Unmarshal(data, &embedding); err == nil {
				rec.Embedding = embedding
			}
		}
	}

	// Decode metadata
	if metaStr := fields["metadata"]; metaStr != "" {
		if data, err := base64.StdEncoding.DecodeString(metaStr); err == nil {
			var metadata map[string]any
			if err := json.Unmarshal(data, &metadata); err == nil {
				rec.Metadata = metadata
			}
		}
	}

	return rec
}
