package concurrent

import (
	"context"
	"sync"
)

const defaultMaxConcurrency = 10

// WorkerPool manages a pool of workers for concurrent operations
type WorkerPool struct {
	sem chan struct{}
}

// NewWorkerPool creates a new worker pool with the specified max workers
func NewWorkerPool(maxWorkers int) *WorkerPool {
	maxWorkers = normalizedMaxConcurrency(maxWorkers)
	return &WorkerPool{
		sem: make(chan struct{}, maxWorkers),
	}
}

// Do executes a function with worker pool concurrency control
func (wp *WorkerPool) Do(ctx context.Context, fn func() error) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	select {
	case <-ctx.Done():
		return ctx.Err()
	case wp.sem <- struct{}{}:
		defer func() { <-wp.sem }()
		if err := ctx.Err(); err != nil {
			return err
		}
		return fn()
	}
}

// ParallelMap executes a function on each item in parallel and returns results
func ParallelMap[T, R any](ctx context.Context, items []T, fn func(T) (R, error), maxConcurrency int) ([]R, error) {
	if len(items) == 0 {
		return nil, nil
	}

	results := make([]R, len(items))
	errors := executeIndexed(ctx, len(items), maxConcurrency, func(index int) error {
		var err error
		results[index], err = fn(items[index])
		return err
	})
	if err := firstError(errors); err != nil {
		return results, err
	}

	return results, nil
}

// ParallelForEach executes a function on each item in parallel
func ParallelForEach[T any](ctx context.Context, items []T, fn func(T) error, maxConcurrency int) error {
	if len(items) == 0 {
		return nil
	}

	return firstError(executeIndexed(ctx, len(items), maxConcurrency, func(index int) error {
		return fn(items[index])
	}))
}

func executeIndexed(ctx context.Context, itemCount, maxConcurrency int, fn func(int) error) []error {
	errors := make([]error, itemCount)
	workerCount := normalizedMaxConcurrency(maxConcurrency)
	if workerCount > itemCount {
		workerCount = itemCount
	}

	indices := make(chan int)
	var workers sync.WaitGroup
	workers.Add(workerCount)
	for range workerCount {
		go func() {
			defer workers.Done()
			for index := range indices {
				select {
				case <-ctx.Done():
					errors[index] = ctx.Err()
				default:
					errors[index] = fn(index)
				}
			}
		}()
	}

	nextIndex := 0
dispatch:
	for ; nextIndex < itemCount; nextIndex++ {
		select {
		case <-ctx.Done():
			break dispatch
		case indices <- nextIndex:
		}
	}
	close(indices)

	if err := ctx.Err(); err != nil {
		for ; nextIndex < itemCount; nextIndex++ {
			errors[nextIndex] = err
		}
	}

	workers.Wait()
	return errors
}

func normalizedMaxConcurrency(maxConcurrency int) int {
	if maxConcurrency <= 0 {
		return defaultMaxConcurrency
	}
	return maxConcurrency
}

func firstError(errors []error) error {
	for _, err := range errors {
		if err != nil {
			return err
		}
	}

	return nil
}
