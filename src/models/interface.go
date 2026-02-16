package models

import "context"

// File is a lightweight in-memory attachment.
// Name is used for display; MIME should be best-effort (e.g., "text/markdown").
type File struct {
	Name string
	MIME string
	Data []byte
}

// StreamChunk represents a single piece of a streaming LLM response.
// When Done is true, the stream is complete and FullText holds the aggregated output.
// When Err is non-nil, the stream encountered an error.
type StreamChunk struct {
	Delta    string // incremental text token
	Done     bool   // true on the final chunk
	FullText string // aggregated text (populated only on the final chunk)
	Err      error  // non-nil if the stream encountered a fatal error
}

type Agent interface {
	Generate(context.Context, string) (any, error)
	GenerateWithFiles(context.Context, string, []File) (any, error)

	// GenerateStream returns a channel that yields incremental text chunks.
	// The final chunk has Done=true and FullText set to the complete response.
	// If the provider doesn't support streaming natively, it falls back to
	// a single-chunk response wrapping Generate.
	GenerateStream(ctx context.Context, prompt string) (<-chan StreamChunk, error)
}
