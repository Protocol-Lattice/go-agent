package models

import "context"

// File is a lightweight in-memory attachment.
// Name is used for display; MIME should be best-effort (e.g., "text/markdown").
type File struct {
	Name string
	MIME string
	Data []byte
}

type Agent interface {
	Generate(context.Context, string) (any, error)
	GenerateWithFiles(context.Context, string, []File) (any, error)
}
