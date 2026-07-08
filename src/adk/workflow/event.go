package workflow

// Event carries node output and optional route selections to successor edges.
type Event struct {
	Output   any
	Message  string
	Routes   []any
	Metadata map[string]string
}

// EmitFunc lets a node publish one or more explicit workflow events.
type EmitFunc func(*Event) error

func eventResult(ev Event) any {
	if ev.Message != "" {
		return ev.Message
	}
	return ev.Output
}
