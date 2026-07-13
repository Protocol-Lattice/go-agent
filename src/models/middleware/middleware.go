// Package middleware provides composable policies for model calls.
package middleware

import (
	"errors"
	"fmt"

	"github.com/Protocol-Lattice/go-agent/src/models"
)

var errNilModel = errors.New("model middleware requires a model")

// Middleware wraps a model with one policy. Implementations should preserve
// models.ToolCallingAgent when possible and return models.ErrToolCallingUnsupported
// when the wrapped model does not support native tool calls.
type Middleware interface {
	Wrap(models.Agent) (models.Agent, error)
}

// MiddlewareFunc adapts a function into Middleware.
type MiddlewareFunc func(models.Agent) (models.Agent, error)

// Wrap applies f to next.
func (f MiddlewareFunc) Wrap(next models.Agent) (models.Agent, error) {
	if f == nil {
		return nil, errors.New("model middleware function is nil")
	}
	return f(next)
}

// Wrap composes policies around base. The first policy is the outermost: it
// sees the request first and the response last.
func Wrap(base models.Agent, policies ...Middleware) (models.Agent, error) {
	if base == nil {
		return nil, errNilModel
	}

	wrapped := base
	for i := len(policies) - 1; i >= 0; i-- {
		if policies[i] == nil {
			return nil, fmt.Errorf("model middleware %d is nil", i)
		}
		next, err := policies[i].Wrap(wrapped)
		if err != nil {
			return nil, fmt.Errorf("wrap model middleware %d: %w", i, err)
		}
		if next == nil {
			return nil, fmt.Errorf("model middleware %d returned a nil model", i)
		}
		wrapped = next
	}
	return wrapped, nil
}

func nativeModel(next models.Agent) (models.ToolCallingAgent, error) {
	if !supportsNativeToolCalling(next) {
		return nil, fmt.Errorf("%w: wrapped model", models.ErrToolCallingUnsupported)
	}
	native, ok := next.(models.ToolCallingAgent)
	if !ok {
		return nil, fmt.Errorf("%w: wrapped model", models.ErrToolCallingUnsupported)
	}
	return native, nil
}

type wrappedModel interface {
	wrappedModel() models.Agent
}

func supportsNativeToolCalling(next models.Agent) bool {
	for next != nil {
		wrapper, ok := next.(wrappedModel)
		if !ok {
			_, ok = next.(models.ToolCallingAgent)
			return ok
		}
		next = wrapper.wrappedModel()
	}
	return false
}
