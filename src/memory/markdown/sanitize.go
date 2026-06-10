package markdown

import (
	"fmt"
	"strings"
)

func renderBlock(rec Record) string {
	rec = rec.normalized()

	var b strings.Builder

	fmt.Fprintf(
		&b,
		"<!-- memory:id=%s ts=%s role=%s tags=%s -->\n\n",
		escapeMeta(rec.ID),
		rec.CreatedAt.UTC().Format("2006-01-02T15:04:05Z07:00"),
		escapeMeta(rec.Role),
		escapeMeta(strings.Join(rec.Tags, ",")),
	)

	fmt.Fprintf(&b, "## %s\n\n", heading(rec.Role))
	b.WriteString(strings.TrimSpace(rec.Content))
	b.WriteString("\n\n<!-- /memory -->\n\n")

	return b.String()
}

func heading(role string) string {
	role = strings.TrimSpace(role)
	if role == "" {
		return "Memory"
	}

	return strings.ToUpper(role[:1]) + role[1:]
}

func escapeMeta(v string) string {
	v = strings.ReplaceAll(v, "\n", " ")
	v = strings.ReplaceAll(v, "\r", " ")
	v = strings.ReplaceAll(v, "-->", "—>")
	return strings.TrimSpace(v)
}
