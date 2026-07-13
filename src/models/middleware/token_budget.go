package middleware

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"sync"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

// ErrTokenBudgetExceeded indicates that a model call would exceed, or has
// already generated output beyond, its configured estimated token budget.
var ErrTokenBudgetExceeded = errors.New("model token budget exceeded")

// TokenEstimator estimates the number of tokens in text. Provider tokenizers
// can be supplied when exact accounting is required.
type TokenEstimator func(string) int64

// ApproximateTokenCount estimates one token per four UTF-8 bytes. It is a
// portable fallback, not provider billing data.
func ApproximateTokenCount(text string) int64 {
	if text == "" {
		return 0
	}
	return int64((len(text) + 3) / 4)
}

// TokenBudgetError describes a rejected budget charge.
type TokenBudgetError struct {
	Phase     string
	Max       int64
	Used      int64
	Requested int64
}

func (e *TokenBudgetError) Error() string {
	return fmt.Sprintf(
		"%s: phase=%s max=%d used=%d requested=%d",
		ErrTokenBudgetExceeded,
		e.Phase,
		e.Max,
		e.Used,
		e.Requested,
	)
}

// Unwrap supports errors.Is(err, ErrTokenBudgetExceeded).
func (e *TokenBudgetError) Unwrap() error { return ErrTokenBudgetExceeded }

// TokenBudget is a concurrency-safe, reusable estimated-token allowance.
// Input charges are rejected before a model call. Output charges are recorded
// after generation, so Used may exceed Max when a non-streaming provider
// returns more output than remained.
type TokenBudget struct {
	mu        sync.Mutex
	max       int64
	used      int64
	estimator TokenEstimator
}

// NewTokenBudget creates a token budget. A nil estimator uses
// ApproximateTokenCount.
func NewTokenBudget(max int64, estimator TokenEstimator) (*TokenBudget, error) {
	if max <= 0 {
		return nil, errors.New("token budget maximum must be greater than zero")
	}
	if estimator == nil {
		estimator = ApproximateTokenCount
	}
	return &TokenBudget{max: max, estimator: estimator}, nil
}

// Max returns the configured allowance.
func (b *TokenBudget) Max() int64 {
	if b == nil {
		return 0
	}
	return b.max
}

// Used returns the estimated tokens charged so far.
func (b *TokenBudget) Used() int64 {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.used
}

// Remaining returns the estimated tokens still available, clamped to zero.
func (b *TokenBudget) Remaining() int64 {
	if b == nil {
		return 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	remaining := b.max - b.used
	if remaining < 0 {
		return 0
	}
	return remaining
}

// Reset clears all charges. It is safe to call concurrently, though callers
// should normally reset only when no requests using the budget are active.
func (b *TokenBudget) Reset() {
	if b == nil {
		return
	}
	b.mu.Lock()
	b.used = 0
	b.mu.Unlock()
}

type tokenBudgetContextKey struct{}

// ContextWithTokenBudget associates a per-run budget with ctx. It overrides
// the fallback budget configured on TokenBudgetPolicy.
func ContextWithTokenBudget(ctx context.Context, budget *TokenBudget) context.Context {
	if ctx == nil {
		ctx = context.Background()
	}
	return context.WithValue(ctx, tokenBudgetContextKey{}, budget)
}

// TokenBudgetFromContext returns the budget associated with ctx.
func TokenBudgetFromContext(ctx context.Context) (*TokenBudget, bool) {
	if ctx == nil {
		return nil, false
	}
	budget, ok := ctx.Value(tokenBudgetContextKey{}).(*TokenBudget)
	return budget, ok && budget != nil
}

// TokenBudgetPolicy enforces the context budget when present, otherwise
// Budget. A nil fallback is valid and makes the policy context-only.
type TokenBudgetPolicy struct {
	Budget *TokenBudget
}

// Wrap applies the token-budget policy.
func (p TokenBudgetPolicy) Wrap(next models.Agent) (models.Agent, error) {
	if next == nil {
		return nil, errNilModel
	}
	return &tokenBudgetAgent{next: next, fallback: p.Budget}, nil
}

type tokenBudgetAgent struct {
	next     models.Agent
	fallback *TokenBudget
}

func (a *tokenBudgetAgent) wrappedModel() models.Agent { return a.next }

func (a *tokenBudgetAgent) Generate(ctx context.Context, prompt string) (any, error) {
	budget := a.budget(ctx)
	if budget == nil {
		return a.next.Generate(ctx, prompt)
	}
	if err := budget.chargeInput(budget.estimate(prompt)); err != nil {
		return nil, err
	}
	result, err := a.next.Generate(ctx, prompt)
	if err != nil {
		return nil, err
	}
	if err := budget.chargeOutput(budget.estimate(fmt.Sprint(result))); err != nil {
		return nil, err
	}
	return result, nil
}

func (a *tokenBudgetAgent) GenerateWithFiles(ctx context.Context, prompt string, files []models.File) (any, error) {
	budget := a.budget(ctx)
	if budget == nil {
		return a.next.GenerateWithFiles(ctx, prompt, files)
	}
	inputTokens := budget.estimate(prompt) + estimateFileTokens(budget, files)
	if err := budget.chargeInput(inputTokens); err != nil {
		return nil, err
	}
	result, err := a.next.GenerateWithFiles(ctx, prompt, files)
	if err != nil {
		return nil, err
	}
	if err := budget.chargeOutput(budget.estimate(fmt.Sprint(result))); err != nil {
		return nil, err
	}
	return result, nil
}

func (a *tokenBudgetAgent) GenerateWithTools(ctx context.Context, prompt string, tools []models.ToolDefinition) (models.ToolCallResponse, error) {
	native, err := nativeModel(a.next)
	if err != nil {
		return models.ToolCallResponse{}, err
	}
	budget := a.budget(ctx)
	if budget == nil {
		return native.GenerateWithTools(ctx, prompt, tools)
	}
	inputTokens := budget.estimate(prompt)
	if encoded, marshalErr := json.Marshal(tools); marshalErr == nil {
		inputTokens += budget.estimate(string(encoded))
	}
	if err := budget.chargeInput(inputTokens); err != nil {
		return models.ToolCallResponse{}, err
	}
	result, err := native.GenerateWithTools(ctx, prompt, tools)
	if err != nil {
		return models.ToolCallResponse{}, err
	}
	encoded, err := json.Marshal(result)
	if err != nil {
		encoded = []byte(result.Content)
	}
	if err := budget.chargeOutput(budget.estimate(string(encoded))); err != nil {
		return models.ToolCallResponse{}, err
	}
	return result, nil
}

func (a *tokenBudgetAgent) GenerateStream(ctx context.Context, prompt string) (<-chan models.StreamChunk, error) {
	budget := a.budget(ctx)
	if budget == nil {
		return a.next.GenerateStream(ctx, prompt)
	}
	if err := budget.chargeInput(budget.estimate(prompt)); err != nil {
		return nil, err
	}
	if ctx == nil {
		ctx = context.Background()
	}
	streamCtx, cancel := context.WithCancel(ctx)
	inner, err := a.next.GenerateStream(streamCtx, prompt)
	if err != nil {
		cancel()
		return nil, err
	}
	if inner == nil {
		cancel()
		return nil, errors.New("model returned a nil stream")
	}

	out := make(chan models.StreamChunk, 16)
	go func() {
		defer close(out)
		defer cancel()
		var actual strings.Builder
		var forwarded strings.Builder
		var charged int64
		for {
			select {
			case <-streamCtx.Done():
				if ctx.Err() != nil {
					out <- models.StreamChunk{Done: true, FullText: forwarded.String(), Err: ctx.Err()}
				}
				return
			case chunk, ok := <-inner:
				if !ok {
					return
				}

				if chunk.Delta != "" {
					actual.WriteString(chunk.Delta)
				}
				actualText := actual.String()
				if chunk.Done && chunk.FullText != "" {
					actualText = chunk.FullText
				}
				total := budget.estimate(actualText)
				increment := total - charged
				if increment < 0 {
					increment = 0
				}
				if err := budget.chargeOutput(increment); err != nil {
					cancel()
					out <- models.StreamChunk{Done: true, FullText: forwarded.String(), Err: err}
					return
				}
				charged += increment

				if !sendStreamChunk(streamCtx, out, chunk) {
					return
				}
				if chunk.Delta != "" {
					forwarded.WriteString(chunk.Delta)
				}
				if chunk.Done || chunk.Err != nil {
					return
				}
			}
		}
	}()
	return out, nil
}

func (a *tokenBudgetAgent) budget(ctx context.Context) *TokenBudget {
	if budget, ok := TokenBudgetFromContext(ctx); ok {
		return budget
	}
	return a.fallback
}

func (b *TokenBudget) estimate(text string) int64 {
	estimator := b.estimator
	if estimator == nil {
		estimator = ApproximateTokenCount
	}
	tokens := estimator(text)
	if tokens < 0 {
		return 0
	}
	return tokens
}

func (b *TokenBudget) chargeInput(tokens int64) error {
	if tokens < 0 {
		tokens = 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	if b.used+tokens > b.max {
		return &TokenBudgetError{
			Phase:     "input",
			Max:       b.max,
			Used:      b.used,
			Requested: tokens,
		}
	}
	b.used += tokens
	return nil
}

func (b *TokenBudget) chargeOutput(tokens int64) error {
	if tokens < 0 {
		tokens = 0
	}
	b.mu.Lock()
	defer b.mu.Unlock()
	b.used += tokens
	if b.used > b.max {
		return &TokenBudgetError{
			Phase:     "output",
			Max:       b.max,
			Used:      b.used,
			Requested: tokens,
		}
	}
	return nil
}

func estimateFileTokens(budget *TokenBudget, files []models.File) int64 {
	var total int64
	for _, file := range files {
		if fileLooksTextual(file) {
			total += budget.estimate(string(file.Data))
		}
	}
	return total
}

func fileLooksTextual(file models.File) bool {
	mime := strings.ToLower(strings.TrimSpace(file.MIME))
	if semi := strings.IndexByte(mime, ';'); semi >= 0 {
		mime = strings.TrimSpace(mime[:semi])
	}
	if mime == "" || mime == "application/octet-stream" {
		ext := strings.ToLower(filepath.Ext(file.Name))
		switch ext {
		case ".txt", ".md", ".markdown", ".json", ".jsonl", ".xml", ".yaml", ".yml", ".toml", ".csv", ".tsv", ".go", ".js", ".ts", ".jsx", ".tsx", ".py", ".rs", ".java", ".c", ".h", ".cpp", ".hpp", ".sql", ".html", ".css", ".sh":
			return true
		default:
			return mime == "" && len(file.Data) > 0
		}
	}
	if strings.HasPrefix(mime, "text/") {
		return true
	}
	switch mime {
	case "application/json", "application/ld+json", "application/xml", "application/javascript", "application/sql", "application/yaml", "application/x-yaml", "application/toml":
		return true
	default:
		return strings.HasSuffix(mime, "+json") || strings.HasSuffix(mime, "+xml")
	}
}
