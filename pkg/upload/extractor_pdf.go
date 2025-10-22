package upload

import (
	"bytes"
	"io"
	"strconv"
	"strings"

	"github.com/ledongthuc/pdf"
)

// PDFExtractor implements Extractor for application/pdf.
// Build with: go build -tags=pdf ./...
type PDFExtractor struct{}

func (PDFExtractor) Supports(m string) bool {
	return strings.EqualFold(m, "application/pdf")
}

func (PDFExtractor) Extract(doc *Document) ([]string, error) {
	// Ensure we have an io.ReaderAt and correct size for the PDF reader.
	var ra io.ReaderAt
	if r, ok := doc.Reader.(io.ReaderAt); ok {
		ra = r
	} else {
		// Fallback: buffer into memory to obtain a ReaderAt
		if _, err := doc.Reader.Seek(0, io.SeekStart); err != nil {
			return nil, err
		}
		buf, err := io.ReadAll(doc.Reader)
		if err != nil {
			return nil, err
		}
		br := bytes.NewReader(buf)
		ra = br
		doc.Reader = br
		doc.SizeBytes = int64(len(buf))
	}

	rdr, err := pdf.NewReader(ra, doc.SizeBytes)
	if err != nil {
		return nil, err
	}

	n := rdr.NumPage()
	out := make([]string, 0, n)

	for i := 1; i <= n; i++ {
		// ledongthuc/pdf Page(i) commonly returns a value (pdf.Page), not a pointer.
		pg := rdr.Page(i)

		txt, err := pg.GetPlainText(nil)
		if err != nil {
			// Image-only or problematic page — skip gracefully.
			continue
		}
		s := strings.TrimSpace(txt)
		if s == "" {
			continue
		}
		// Prefix with page number for better retrieval/reranking context.
		out = append(out, "Page "+strconv.Itoa(i)+"\n"+s)
	}

	return out, nil
}
