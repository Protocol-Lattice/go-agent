package engine

import (
	"bufio"
	"container/heap"
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"time"
	"unicode"
	"unicode/utf8"

	"github.com/Protocol-Lattice/go-agent/src/memory/model"
)

const (
	pruneDeleteBatchSize        = 1024
	pruneDeletionSpoolPrefix    = "go-agent-prune-"
	pruneDeletionSpoolRecordLen = 9 // int64 ID plus one TTL flag byte.
)

type pendingDeletion struct {
	id  int64
	ttl bool
}

// deletionSpool keeps deletion candidates off-heap while a store is being
// iterated. Store implementations can keep locks or cursors open in Iterate,
// so deletions must not be applied until iteration has completed. The spool
// bounds retained deletion IDs to one delete batch during replay.
type deletionSpool struct {
	file   *os.File
	writer *bufio.Writer
	path   string
}

func (s *deletionSpool) append(deletion pendingDeletion) error {
	if s.file == nil {
		file, err := os.CreateTemp("", pruneDeletionSpoolPrefix)
		if err != nil {
			return fmt.Errorf("create prune deletion spool: %w", err)
		}
		s.file = file
		s.writer = bufio.NewWriterSize(file, 64*1024)
		s.path = file.Name()
	}

	var record [pruneDeletionSpoolRecordLen]byte
	binary.LittleEndian.PutUint64(record[:8], uint64(deletion.id))
	if deletion.ttl {
		record[8] = 1
	}
	for remaining := record[:]; len(remaining) > 0; {
		n, err := s.writer.Write(remaining)
		if err != nil {
			return fmt.Errorf("write prune deletion spool: %w", err)
		}
		if n == 0 {
			return fmt.Errorf("write prune deletion spool: %w", io.ErrShortWrite)
		}
		remaining = remaining[n:]
	}
	return nil
}

func (s *deletionSpool) rewind() error {
	if s.file == nil {
		return nil
	}
	if err := s.writer.Flush(); err != nil {
		return fmt.Errorf("flush prune deletion spool: %w", err)
	}
	if _, err := s.file.Seek(0, io.SeekStart); err != nil {
		return fmt.Errorf("rewind prune deletion spool: %w", err)
	}
	return nil
}

func (s *deletionSpool) nextBatch() ([]pendingDeletion, error) {
	if s.file == nil {
		return nil, nil
	}

	batch := make([]pendingDeletion, 0, pruneDeleteBatchSize)
	var record [pruneDeletionSpoolRecordLen]byte
	for len(batch) < pruneDeleteBatchSize {
		_, err := io.ReadFull(s.file, record[:])
		if errors.Is(err, io.EOF) {
			return batch, nil
		}
		if err != nil {
			return nil, fmt.Errorf("read prune deletion spool: %w", err)
		}
		batch = append(batch, pendingDeletion{
			id:  int64(binary.LittleEndian.Uint64(record[:8])),
			ttl: record[8] != 0,
		})
	}
	return batch, nil
}

func (s *deletionSpool) closeAndRemove() error {
	if s.file == nil {
		return nil
	}

	flushErr := s.writer.Flush()
	closeErr := s.file.Close()
	removeErr := os.Remove(s.path)
	s.file = nil
	s.writer = nil
	s.path = ""
	return errors.Join(flushErr, closeErr, removeErr)
}

type pruneCandidate struct {
	id         int64
	createdAt  time.Time
	importance float64
	content    string
	metadata   string
}

// Prune applies TTL, size and deduplication policies.
func (e *Engine) Prune(ctx context.Context) (err error) {
	if e.store == nil {
		return nil
	}

	now := e.clock().UTC()
	seen := make(map[string]int64, 1024)
	candidates := make([]pruneCandidate, 0, 1024)
	spool := deletionSpool{}
	defer func() {
		err = errors.Join(err, spool.closeAndRemove())
	}()
	var spoolErr error
	survivors := 0

	if err := e.store.Iterate(ctx, func(rec model.MemoryRecord) bool {
		if !rec.CreatedAt.IsZero() && now.Sub(rec.CreatedAt) > e.opts.TTL {
			spoolErr = spool.append(pendingDeletion{id: rec.ID, ttl: true})
			return spoolErr == nil
		}

		key := canonicalKey(rec.Content)
		if _, ok := seen[key]; ok {
			spoolErr = spool.append(pendingDeletion{id: rec.ID})
			if spoolErr != nil {
				return false
			}
			if e.metrics != nil {
				e.metrics.IncDeduplicated()
			}
			return true
		}
		seen[key] = rec.ID

		candidates = append(candidates, pruneCandidate{
			id:         rec.ID,
			createdAt:  rec.CreatedAt,
			importance: rec.Importance,
			content:    rec.Content,
			metadata:   rec.Metadata,
		})
		survivors++
		return true
	}); err != nil {
		return err
	}
	if spoolErr != nil {
		return spoolErr
	}

	// Store implementations may hold locks or database cursors while Iterate is
	// running. Defer every mutation until iteration has fully completed.
	if err := e.deletePrunedRecords(ctx, &spool); err != nil {
		return err
	}

	if survivors <= e.opts.MaxSize {
		return nil
	}

	overflow := survivors - e.opts.MaxSize
	h := make(minHeap, 0, overflow)
	heap.Init(&h)

	for i := range candidates {
		candidate := &candidates[i]
		importance := candidate.importance
		if importance == 0 {
			importance = importanceScore(candidate.content, model.DecodeMetadata(candidate.metadata))
			candidate.importance = importance
		}
		ageHours := now.Sub(candidate.createdAt).Hours() + 1
		score := ageHours * (1 - importance)

		if len(h) < overflow {
			heap.Push(&h, item{id: candidate.id, score: score})
		} else if score > h[0].score {
			h[0] = item{id: candidate.id, score: score}
			heap.Fix(&h, 0)
		}
	}

	evict := make([]int64, 0, h.Len())
	for h.Len() > 0 {
		evict = append(evict, heap.Pop(&h).(item).id)
	}

	return e.deleteSizeEvictions(ctx, evict)
}

func (e *Engine) deletePrunedRecords(ctx context.Context, spool *deletionSpool) error {
	if err := spool.rewind(); err != nil {
		return err
	}
	for {
		batch, err := spool.nextBatch()
		if err != nil {
			return err
		}
		if len(batch) == 0 {
			return nil
		}
		ids := make([]int64, len(batch))
		ttlCount := 0
		for i, deletion := range batch {
			ids[i] = deletion.id
			if deletion.ttl {
				ttlCount++
			}
		}

		if err := e.store.DeleteMemory(ctx, ids); err != nil {
			return err
		}
		if e.metrics != nil {
			e.metrics.IncPruned(len(ids))
			if ttlCount > 0 {
				e.metrics.IncTTLExpired(ttlCount)
			}
		}
	}
}

func (e *Engine) deleteSizeEvictions(ctx context.Context, ids []int64) error {
	for start := 0; start < len(ids); start += pruneDeleteBatchSize {
		end := min(start+pruneDeleteBatchSize, len(ids))
		batch := ids[start:end]
		if err := e.store.DeleteMemory(ctx, batch); err != nil {
			return err
		}
		if e.metrics != nil {
			e.metrics.IncPruned(len(batch))
			e.metrics.IncSizeEvicted(len(batch))
		}
	}
	return nil
}

// canonicalKey lowercases and trims whitespace in a single pass to reduce allocations.
func canonicalKey(s string) string {
	start, end := 0, len(s)
	for start < end {
		r, size := utf8.DecodeRuneInString(s[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		start += size
	}
	for start < end {
		r, size := utf8.DecodeLastRuneInString(s[start:end])
		if !unicode.IsSpace(r) {
			break
		}
		end -= size
	}
	if start >= end {
		return ""
	}

	var b strings.Builder
	b.Grow(end - start)
	for _, r := range s[start:end] {
		b.WriteRune(unicode.ToLower(r))
	}
	return b.String()
}

// item and minHeap retain the largest prune scores, which are evicted first.
type item struct {
	id    int64
	score float64
}

type minHeap []item

func (h minHeap) Len() int           { return len(h) }
func (h minHeap) Less(i, j int) bool { return h[i].score < h[j].score }
func (h minHeap) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *minHeap) Push(x any)        { *h = append(*h, x.(item)) }
func (h *minHeap) Pop() any {
	old := *h
	n := len(old)
	x := old[n-1]
	*h = old[:n-1]
	return x
}
