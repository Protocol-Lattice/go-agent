package workflow

import "fmt"

// Route selects successor edges based on emitted Event.Routes.
type Route interface {
	Match(Event) bool
	isDefault() bool
}

type routeFunc struct {
	match   func(Event) bool
	def     bool
	display string
}

func (r routeFunc) Match(ev Event) bool {
	if r.match == nil {
		return false
	}
	return r.match(ev)
}

func (r routeFunc) isDefault() bool { return r.def }

func (r routeFunc) String() string {
	if r.display != "" {
		return r.display
	}
	if r.def {
		return "default"
	}
	return "route"
}

// Default matches only when no non-default route from the same node matches.
var Default Route = routeFunc{
	def:     true,
	display: "default",
	match: func(Event) bool {
		return true
	},
}

// StringRoute matches an emitted string route value.
func StringRoute(value string) Route {
	return valueRoute[string](value)
}

// IntRoute matches an emitted int route value.
func IntRoute(value int) Route {
	return valueRoute[int](value)
}

// BoolRoute matches an emitted bool route value.
func BoolRoute(value bool) Route {
	return valueRoute[bool](value)
}

// MultiRoute matches any emitted route value in values.
func MultiRoute[T comparable](values ...T) Route {
	allowed := make(map[T]struct{}, len(values))
	for _, value := range values {
		allowed[value] = struct{}{}
	}
	return routeFunc{
		display: fmt.Sprintf("multi(%d)", len(values)),
		match: func(ev Event) bool {
			for _, route := range ev.Routes {
				typed, ok := route.(T)
				if !ok {
					continue
				}
				if _, found := allowed[typed]; found {
					return true
				}
			}
			return false
		},
	}
}

func valueRoute[T comparable](value T) Route {
	return routeFunc{
		display: fmt.Sprint(value),
		match: func(ev Event) bool {
			for _, route := range ev.Routes {
				typed, ok := route.(T)
				if ok && typed == value {
					return true
				}
			}
			return false
		},
	}
}
