package upload

func defaultExtractors() []Extractor {
	// PDF first, then text (so PDFs don’t fall through)
	return []Extractor{PDFExtractor{}, TextExtractor{}}
}
