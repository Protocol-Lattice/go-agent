package memory

import (
	"context"
	"fmt"
)

func ExampleNewEngine() {
	store := NewInMemoryStore()
	engine := NewEngine(store, Options{}).WithEmbedder(DummyEmbedder{})
	ctx := context.Background()

	engine.Store(ctx, "demo", "Track onboarding progress", map[string]any{"source": "notion"})
	engine.Store(ctx, "demo", "Customer reported login issue", map[string]any{"source": "support"})

	records, _ := engine.Retrieve(ctx, "demo", "login", 1)
	fmt.Println(len(records) > 0)
	// Output: true
}
