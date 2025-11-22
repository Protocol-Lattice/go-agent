package concurrent

import (
	"context"
	"sync"
)

// WorkerPool manages a pool of workers for concurrent operations
type WorkerPool struct {
	maxWorkers int
	sem        chan struct{}
}

// NewWorkerPool creates a new worker pool with the specified max workers
func NewWorkerPool(maxWorkers int) *WorkerPool {
	if maxWorkers <= 0 {
		maxWorkers = 10
	}
	return &WorkerPool{
		maxWorkers: maxWorkers,
		sem:        make(chan struct{}, maxWorkers),
	}
}

// Do executes a function with worker pool concurrency control
func (wp *WorkerPool) Do(ctx context.Context, fn func() error) error {
	select {
	case <-ctx.Done():
		return ctx.Err()
	case wp.sem <- struct{}{}:
		defer func() { <-wp.sem }()
		return fn()
	}
}

// ParallelMap executes a function on each item in parallel and returns results
func ParallelMap[T, R any](ctx context.Context, items []T, fn func(T) (R, error), maxConcurrency int) ([]R, error) {
	if len(items) == 0 {
		return nil, nil
	}

	if maxConcurrency <= 0 {
		maxConcurrency = 10
	}

	results := make([]R, len(items))
	errors := make([]error, len(items))

	var wg sync.WaitGroup
	sem := make(chan struct{}, maxConcurrency)

	for i, item := range items {
		wg.Add(1)
		go func(idx int, val T) {
			defer wg.Done()

			select {
			case <-ctx.Done():
				errors[idx] = ctx.Err()
				return
			case sem <- struct{}{}:
				defer func() { <-sem }()
				results[idx], errors[idx] = fn(val)
			}
		}(i, item)
	}

	wg.Wait()

	// Check for errors
	for _, err := range errors {
		if err != nil {
			return results, err
		}
	}

	return results, nil
}

// ParallelForEach executes a function on each item in parallel
func ParallelForEach[T any](ctx context.Context, items []T, fn func(T) error, maxConcurrency int) error {
	if len(items) == 0 {
		return nil
	}

	if maxConcurrency <= 0 {
		maxConcurrency = 10
	}

	var wg sync.WaitGroup
	sem := make(chan struct{}, maxConcurrency)
	errChan := make(chan error, len(items))

	for _, item := range items {
		wg.Add(1)
		go func(val T) {
			defer wg.Done()

			select {
			case <-ctx.Done():
				errChan <- ctx.Err()
				return
			case sem <- struct{}{}:
				defer func() { <-sem }()
				if err := fn(val); err != nil {
					errChan <- err
				}
			}
		}(item)
	}

	wg.Wait()
	close(errChan)

	// Return first error if any
	for err := range errChan {
		if err != nil {
			return err
		}
	}

	return nil
}
