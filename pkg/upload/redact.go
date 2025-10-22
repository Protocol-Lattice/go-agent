package upload

import "regexp"

type RegexRedactor struct {
	rx []*regexp.Regexp
}

func NewDefaultRedactor() *RegexRedactor {
	return &RegexRedactor{
		rx: []*regexp.Regexp{
			regexp.MustCompile(`\b[\w\.-]+@[\w\.-]+\.\w+\b`),                      // emails
			regexp.MustCompile(`\b(?:\+?\d{1,3}[\s-]?)?(?:\d{3}[\s-]?){2,4}\d\b`), // phones-ish
		},
	}
}

func (r *RegexRedactor) Redact(s string) (string, bool) {
	changed := false
	out := s
	for _, re := range r.rx {
		if re.MatchString(out) {
			out = re.ReplaceAllString(out, "[REDACTED]")
			changed = true
		}
	}
	return out, changed
}
