package models

import (
	"os"
	"regexp"
	"strings"
	"testing"
)

func discardStderr(t *testing.T) func() {
	t.Helper()
	original := os.Stderr
	devnull, err := os.OpenFile(os.DevNull, os.O_WRONLY, 0)
	if err != nil {
		t.Fatalf("failed to open devnull: %v", err)
	}
	os.Stderr = devnull
	return func() {
		os.Stderr = original
		_ = devnull.Close()
	}
}

func TestSanitizeForGemini(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"empty", "", ""},
		{"basic png", "image/png", "image/png"},
		{"png with params", "image/png; charset=binary", "image/png"},
		{"double prefix", "image/image/png", "image/png"},
		{"jpeg alias", "IMAGE/JPG", "image/jpeg"},
		{"video mov", "video/mov", "video/quicktime"},
		{"unsupported", "application/pdf", ""},
		{"double video prefix", "video/video/mp4", "video/mp4"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sanitizeForGemini(tc.input); got != tc.want {
				t.Fatalf("sanitizeForGemini(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestSanitizeForAnthropic(t *testing.T) {
	cases := []struct {
		name  string
		input string
		want  string
	}{
		{"empty", "", ""},
		{"basic jpeg", "image/jpeg", "image/jpeg"},
		{"jpeg alias", "image/jpg", "image/jpeg"},
		{"with params", "image/png; something", "image/png"},
		{"double prefix", "image/image/webp", "image/webp"},
		{"unsupported", "video/mp4", ""},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := sanitizeForAnthropic(tc.input); got != tc.want {
				t.Fatalf("sanitizeForAnthropic(%q) = %q, want %q", tc.input, got, tc.want)
			}
		})
	}
}

func TestNormalizeMIME(t *testing.T) {
	restore := discardStderr(t)
	defer restore()

	cases := []struct {
		name string
		file string
		mime string
		want string
	}{
		{"empty everything", "noext", "", ""},
		{"from extension", "report.md", "", "text/markdown"},
		{"alias jpeg", "photo", "image/jpg", "image/jpeg"},
		{"double prefix", "diagram.png", "image/image/png", "image/png"},
		{"invalid without slash", "clip.mp4", "video", "video/mp4"},
		{"with params", "vector.svg", "image/svg+xml; charset=utf-8", "image/svg+xml"},
		{"already clean", "data.bin", "application/octet-stream", "application/octet-stream"},
		{"suffix slash", "notes.txt", "text/plain/", "text/plain"},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := normalizeMIME(tc.file, tc.mime); got != tc.want {
				t.Fatalf("normalizeMIME(%q, %q) = %q, want %q", tc.file, tc.mime, got, tc.want)
			}
		})
	}
}

func TestIsTextMIME(t *testing.T) {
	cases := []struct {
		input string
		want  bool
	}{
		{"text/plain", true},
		{" application/json ", true},
		{"text/markdown", true},
		{"application/pdf", false},
		{"", false},
	}

	for _, tc := range cases {
		if got := isTextMIME(tc.input); got != tc.want {
			t.Fatalf("isTextMIME(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestIsImageOrVideoMIME(t *testing.T) {
	cases := []struct {
		input string
		want  bool
	}{
		{"image/png", true},
		{" video/mp4 ", true},
		{"text/plain", false},
		{"", false},
	}

	for _, tc := range cases {
		if got := isImageOrVideoMIME(tc.input); got != tc.want {
			t.Fatalf("isImageOrVideoMIME(%q) = %v, want %v", tc.input, got, tc.want)
		}
	}
}

func TestGetOpenAIMimeType(t *testing.T) {
	cases := []struct {
		input string
		want  string
	}{
		{"image/jpeg", "image/jpeg"},
		{"image/jpg", "image/jpeg"},
		{"image/png", "image/png"},
		{"image/webp", "image/webp"},
		{"image/bmp", ""},
		{"video/mp4", "video/mp4"},
		{" application/pdf ", ""},
	}

	for _, tc := range cases {
		if got := getOpenAIMimeType(tc.input); got != tc.want {
			t.Fatalf("getOpenAIMimeType(%q) = %q, want %q", tc.input, got, tc.want)
		}
	}
}

func TestCombinePromptWithFiles_DefaultNameAndNonText(t *testing.T) {
	base := "Summarize"
	files := []File{
		{Name: "", MIME: "text/plain", Data: []byte("hello")},
		{Name: "clip.mp4", MIME: "video/mp4", Data: []byte{0x00, 0x01}},
	}

	combined := combinePromptWithFiles(base, files)

	if !isTextMIME("text/plain") {
		t.Fatalf("sanity: expected text mime to be true")
	}

	if got, want := combinePromptWithFiles("only base", nil), "only base"; got != want {
		t.Fatalf("combinePromptWithFiles without files = %q, want %q", got, want)
	}

	if !containsAllRegex(combined, []string{
		regexp.QuoteMeta(base),
		"ATTACHMENTS CONTEXT",
		`<<<FILE file_1(?: \[text/plain\])?>>>:?`, // MIME optional, colon optional
		regexp.QuoteMeta("[Non-text attachment] clip.mp4 (video/mp4)"),
	}) {
		t.Fatalf("combined output missing expected segments:\n%s", combined)
	}
}

// containsAllRegex returns true if every pattern (Go regexp) matches somewhere in s.
func containsAllRegex(s string, patterns []string) bool {
	for _, p := range patterns {
		re := regexp.MustCompile(p)
		if !re.MatchString(s) {
			return false
		}
	}
	return true
}

func containsAll(haystack string, needles []string) bool {
	for _, n := range needles {
		if !strings.Contains(haystack, n) {
			return false
		}
	}
	return true
}
