package models

import (
	"context"
	"errors"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// Agent defines the core contract that all model adapters must satisfy.
type Agent interface {
	Generate(context.Context, string) (any, error)
	UploadFiles(context.Context, []UploadFile) ([]UploadedFile, error)
}

// UploadFile represents a local or in-memory file that should be uploaded to a
// model provider. Either Path or Reader must be supplied. Optional metadata can
// be provided via Name, MIMEType, and Purpose.
type UploadFile struct {
	Name     string
	Path     string
	Reader   io.Reader
	MIMEType string
	Purpose  string
}

// UploadedFile captures the identifier and metadata returned from a provider
// when a file upload succeeds. The Data field is populated for providers that
// return the raw bytes (e.g. Ollama image uploads) instead of a remote handle.
type UploadedFile struct {
	ID        string
	Name      string
	SizeBytes int64
	MIMEType  string
	URI       string
	Provider  string
	Purpose   string
	Data      []byte
}

type resolvedUpload struct {
	reader   *uploadReader
	name     string
	mimeType string
	path     string
	size     int64
}

// Close releases the underlying reader for the resolved upload.
func (r *resolvedUpload) Close() error {
	if r == nil || r.reader == nil {
		return nil
	}
	return r.reader.Close()
}

// uploadReader wraps an io.ReadCloser to expose filename and MIME type hints so
// SDKs that inspect optional interfaces (e.g. Anthropic) can populate metadata.
type uploadReader struct {
	io.ReadCloser
	name string
	mime string
}

func (r *uploadReader) Filename() string { return r.name }

func (r *uploadReader) Name() string { return r.name }

func (r *uploadReader) ContentType() string {
	if r.mime != "" {
		return r.mime
	}
	return "application/octet-stream"
}

func (f UploadFile) resolve() (*resolvedUpload, error) {
	var (
		rc   io.ReadCloser
		size int64
		path string
	)

	if f.Reader != nil {
		if closer, ok := f.Reader.(io.ReadCloser); ok {
			rc = closer
		} else {
			rc = io.NopCloser(f.Reader)
		}
	} else {
		if strings.TrimSpace(f.Path) == "" {
			return nil, errors.New("upload file requires either Path or Reader")
		}
		file, err := os.Open(f.Path)
		if err != nil {
			return nil, err
		}
		rc = file
		path = f.Path
		if info, err := file.Stat(); err == nil {
			size = info.Size()
		}
	}

	name := strings.TrimSpace(f.Name)
	if name == "" {
		switch {
		case path != "":
			name = filepath.Base(path)
		case f.Reader != nil:
			if named, ok := f.Reader.(interface{ Name() string }); ok {
				name = filepath.Base(strings.TrimSpace(named.Name()))
			}
		}
	}
	if name == "" {
		name = "upload"
	}

	mimeType := strings.TrimSpace(f.MIMEType)
	if mimeType == "" {
		if typed, ok := f.Reader.(interface{ ContentType() string }); ok {
			mimeType = strings.TrimSpace(typed.ContentType())
		}
	}

	return &resolvedUpload{
		reader:   &uploadReader{ReadCloser: rc, name: name, mime: mimeType},
		name:     name,
		mimeType: mimeType,
		path:     path,
		size:     size,
	}, nil
}
