package agent

import "os"

func init() {
	// Keep repository mutations bounded by default: inspect, mutate, verify,
	// then stop. Operators can override this with utcp_tool_loop_max_steps.
	if os.Getenv("utcp_tool_loop_max_steps") == "" {
		_ = os.Setenv("utcp_tool_loop_max_steps", "4")
	}
}
