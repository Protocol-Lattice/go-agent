package agent

import "testing"

func TestShouldUseCodeModeForMutationRequests(t *testing.T) {
	cases := []struct {
		name       string
		userInput  string
		fileBacked bool
		want       bool
	}{
		{name: "mutation", userInput: "Refactor README.md", want: true},
		{name: "non mutation", userInput: "Explain README.md", want: true},
		{name: "file backed", userInput: "Refactor the attached README.md", fileBacked: true, want: false},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := shouldUseCodeMode(tc.userInput, tc.fileBacked); got != tc.want {
				t.Fatalf("shouldUseCodeMode(%q, %v) = %v, want %v", tc.userInput, tc.fileBacked, got, tc.want)
			}
		})
	}
}
