package agent

import "regexp"

var (
	unqualifiedCallToolRE       = regexp.MustCompile(`(?m)(^|[^.[:alnum:]_])CallToolStream\s*\(`)
	unqualifiedCallToolAnyRE    = regexp.MustCompile(`(?m)(^|[^.[:alnum:]_])CallTool\s*\(`)
)

// normalizeCodeModeSource repairs legacy/generated CodeMode snippets that use
// CallTool(...) instead of the actual CodeMode API codemode.CallTool(...).
// Qualified calls are left unchanged.
func normalizeCodeModeSource(code string) string {
	code = unqualifiedCallToolRE.ReplaceAllString(code, `${1}codemode.CallToolStream(`)
	code = unqualifiedCallToolAnyRE.ReplaceAllString(code, `${1}codemode.CallTool(`)
	return code
}
