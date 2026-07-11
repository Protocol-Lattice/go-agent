package concurrent

import (
	"context"
	"errors"
	"slices"
	"sync/atomic"
	"testing"
	"time"
)

type parallelRunner func(context.Context, []int, func(int) error, int) error

func parallelRunners() map[string]parallelRunner {
	return map[string]parallelRunner{
		"map": func(ctx context.Context, items []int, fn func(int) error, maxConcurrency int) error {
			_, err := ParallelMap(ctx, items, func(item int) (int, error) {
				return item, fn(item)
			}, maxConcurrency)
			return err
		},
		"for each": ParallelForEach[int],
	}
}

func TestWorkerPoolBoundsConcurrency(t *testing.T) {
	pool := NewWorkerPool(1)
	firstStarted := make(chan struct{})
	releaseFirst := make(chan struct{})
	secondStarted := make(chan struct{})
	errors := make(chan error, 2)

	go func() {
		errors <- pool.Do(context.Background(), func() error {
			close(firstStarted)
			<-releaseFirst
			return nil
		})
	}()
	select {
	case <-firstStarted:
	case <-time.After(time.Second):
		t.Fatal("first operation did not start")
	}

	go func() {
		errors <- pool.Do(context.Background(), func() error {
			close(secondStarted)
			return nil
		})
	}()

	select {
	case <-secondStarted:
		t.Fatal("second operation started before the worker was released")
	case <-time.After(20 * time.Millisecond):
	}

	close(releaseFirst)
	select {
	case <-secondStarted:
	case <-time.After(time.Second):
		t.Fatal("second operation did not start after the worker was released")
	}
	for range 2 {
		select {
		case err := <-errors:
			if err != nil {
				t.Fatalf("WorkerPool.Do() error = %v, want nil", err)
			}
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for worker completion")
		}
	}
}

func TestWorkerPoolHonorsCanceledContext(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	called := false

	err := NewWorkerPool(0).Do(ctx, func() error {
		called = true
		return nil
	})
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("WorkerPool.Do() error = %v, want %v", err, context.Canceled)
	}
	if called {
		t.Fatal("WorkerPool.Do() invoked callback for canceled context")
	}
}

func TestParallelMapPreservesResultOrder(t *testing.T) {
	items := []int{10, 20, 30}
	started := make(chan int, len(items))
	release := []chan struct{}{make(chan struct{}), make(chan struct{}), make(chan struct{})}
	type outcome struct {
		results []int
		err     error
	}
	done := make(chan outcome, 1)

	go func() {
		results, err := ParallelMap(context.Background(), items, func(item int) (int, error) {
			index := item/10 - 1
			started <- index
			<-release[index]
			return item + 1, nil
		}, len(items))
		done <- outcome{results: results, err: err}
	}()

	for range items {
		select {
		case <-started:
		case <-time.After(time.Second):
			t.Fatal("timed out waiting for map operations to start")
		}
	}
	close(release[2])
	close(release[0])
	close(release[1])

	select {
	case got := <-done:
		if got.err != nil {
			t.Fatalf("ParallelMap() error = %v, want nil", got.err)
		}
		want := []int{11, 21, 31}
		if !slices.Equal(got.results, want) {
			t.Fatalf("ParallelMap() results = %v, want %v", got.results, want)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for ParallelMap")
	}
}

func TestParallelFunctionsReturnFirstErrorByInputOrder(t *testing.T) {
	firstErr := errors.New("first input error")
	secondErr := errors.New("second input error")

	for name, run := range parallelRunners() {
		t.Run(name, func(t *testing.T) {
			secondStarted := make(chan struct{})
			err := run(context.Background(), []int{0, 1}, func(item int) error {
				if item == 0 {
					<-secondStarted
					return firstErr
				}
				close(secondStarted)
				return secondErr
			}, 2)

			if !errors.Is(err, firstErr) {
				t.Fatalf("parallel operation error = %v, want %v", err, firstErr)
			}
		})
	}
}

func TestParallelFunctionsEmptyInput(t *testing.T) {
	var mapCalls atomic.Int32
	mapResults, err := ParallelMap(context.Background(), []int(nil), func(int) (int, error) {
		mapCalls.Add(1)
		return 0, nil
	}, 1)
	if err != nil {
		t.Fatalf("ParallelMap() error = %v, want nil", err)
	}
	if mapResults != nil {
		t.Fatalf("ParallelMap() results = %#v, want nil", mapResults)
	}
	if got := mapCalls.Load(); got != 0 {
		t.Fatalf("ParallelMap called function %d times for empty input, want 0", got)
	}

	var forEachCalls atomic.Int32
	err = ParallelForEach(context.Background(), []int(nil), func(int) error {
		forEachCalls.Add(1)
		return nil
	}, 1)
	if err != nil {
		t.Fatalf("ParallelForEach() error = %v, want nil", err)
	}
	if got := forEachCalls.Load(); got != 0 {
		t.Fatalf("ParallelForEach called function %d times for empty input, want 0", got)
	}
}

func TestParallelFunctionsBoundConcurrency(t *testing.T) {
	tests := []struct {
		name           string
		maxConcurrency int
		wantPeak       int32
	}{
		{name: "explicit limit", maxConcurrency: 3, wantPeak: 3},
		{name: "zero uses default", maxConcurrency: 0, wantPeak: 10},
		{name: "negative uses default", maxConcurrency: -1, wantPeak: 10},
	}

	items := make([]int, 24)
	for runnerName, run := range parallelRunners() {
		for _, test := range tests {
			t.Run(runnerName+"/"+test.name, func(t *testing.T) {
				started := make(chan struct{}, len(items))
				release := make(chan struct{})
				done := make(chan error, 1)
				var active atomic.Int32
				var peak atomic.Int32

				go func() {
					done <- run(context.Background(), items, func(int) error {
						current := active.Add(1)
						for observed := peak.Load(); current > observed && !peak.CompareAndSwap(observed, current); observed = peak.Load() {
						}
						started <- struct{}{}
						<-release
						active.Add(-1)
						return nil
					}, test.maxConcurrency)
				}()

				for i := int32(0); i < test.wantPeak; i++ {
					select {
					case <-started:
					case <-time.After(time.Second):
						close(release)
						t.Fatal("timed out waiting for operations to reach concurrency limit")
					}
				}

				select {
				case <-started:
					close(release)
					t.Fatalf("more than %d operations started concurrently", test.wantPeak)
				case <-time.After(20 * time.Millisecond):
				}

				close(release)
				select {
				case err := <-done:
					if err != nil {
						t.Fatalf("parallel operation error = %v, want nil", err)
					}
				case <-time.After(time.Second):
					t.Fatal("timed out waiting for parallel operation")
				}
				if got := peak.Load(); got != test.wantPeak {
					t.Fatalf("peak concurrency = %d, want %d", got, test.wantPeak)
				}
			})
		}
	}
}

func TestParallelFunctionsReturnContextCancellation(t *testing.T) {
	items := make([]int, 8)
	for name, run := range parallelRunners() {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			started := make(chan struct{}, len(items))
			done := make(chan error, 1)

			go func() {
				done <- run(ctx, items, func(int) error {
					started <- struct{}{}
					<-ctx.Done()
					return ctx.Err()
				}, 2)
			}()

			for range 2 {
				select {
				case <-started:
				case <-time.After(time.Second):
					cancel()
					t.Fatal("timed out waiting for operations to start")
				}
			}
			cancel()

			select {
			case err := <-done:
				if !errors.Is(err, context.Canceled) {
					t.Fatalf("parallel operation error = %v, want %v", err, context.Canceled)
				}
			case <-time.After(time.Second):
				t.Fatal("timed out waiting for canceled parallel operation")
			}
		})
	}
}
