package uploads

import (
	"regexp"
	"strings"
)

var whitespaceRegexp = regexp.MustCompile(`\s+`)

func normalizeWhitespace(text string) string {
	trimmed := strings.TrimSpace(text)
	if trimmed == "" {
		return ""
	}
	return whitespaceRegexp.ReplaceAllString(trimmed, " ")
}
