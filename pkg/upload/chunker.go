package upload

import "unicode/utf8"

type FixedChunker struct {
	MaxRunes int // e.g., 1200
	Overlap  int // e.g., 120
}

func (c FixedChunker) Chunk(blocks []string) ([]Chunk, error) {
	if c.MaxRunes <= 0 {
		c.MaxRunes, c.Overlap = 1200, 120
	}
	out := make([]Chunk, 0, len(blocks)*2)
	idx := 0
	for _, b := range blocks {
		runes := []rune(b)
		for i := 0; i < len(runes); i += (c.MaxRunes - c.Overlap) {
			end := i + c.MaxRunes
			if end > len(runes) {
				end = len(runes)
			}
			ch := string(runes[i:end])
			// rough token hint (~ 0.75 * runes)
			out = append(out, Chunk{Index: idx, Text: ch, TokenHint: int(float64(utf8.RuneCountInString(ch)) * 0.75)})
			idx++
			if end == len(runes) {
				break
			}
		}
	}
	return out, nil
}
