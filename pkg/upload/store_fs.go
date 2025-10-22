package upload

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"time"
)

type FSStore struct {
	BaseDir string // e.g., "./data/uploads"
}

func (s FSStore) Put(doc *Document) (string, error) {
	if err := os.MkdirAll(s.BaseDir, 0o755); err != nil {
		return "", err
	}
	ts := time.Now().UTC().Format("20060102T150405Z")
	name := fmt.Sprintf("%s_%s", ts, sanitizeName(doc.Name))
	path := filepath.Join(s.BaseDir, name)
	f, err := os.Create(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	if _, err := doc.Reader.Seek(0, io.SeekStart); err != nil {
		return "", err
	}
	if _, err := io.Copy(f, doc.Reader); err != nil {
		return "", err
	}
	return "file://" + path, nil
}

func sanitizeName(n string) string {
	out := make([]rune, 0, len(n))
	for _, r := range n {
		if r == '/' || r == '\\' {
			out = append(out, '_')
			continue
		}
		out = append(out, r)
	}
	return string(out)
}
