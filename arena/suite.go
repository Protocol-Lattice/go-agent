package arena

import "context"

// Competitor names a runner participating in the same task suite.
type Competitor struct {
	Name   string
	Runner Runner
}

// SuiteResult contains per-task results and an aggregate leaderboard entry.
type SuiteResult struct {
	Competitor string
	Results    []Result
	Summary    Summary
}

// RunSuite runs the same task set against multiple competitors and returns
// deterministic leaderboard-ready results. Each competitor gets an isolated
// Arena instance, while task definitions remain shared.
func RunSuite(ctx context.Context, tasks []Task, competitors []Competitor, concurrency int) []SuiteResult {
	results := make([]SuiteResult, 0, len(competitors))
	for _, competitor := range competitors {
		if competitor.Runner == nil {
			results = append(results, SuiteResult{
				Competitor: competitor.Name,
				Results: []Result{{TaskName: "<runner>", Error: errMissingRunner()}},
				Summary: Summarize([]Result{{TaskName: "<runner>", Error: errMissingRunner()}}),
			})
			continue
		}
		runnerResults := (&Arena{Runner: competitor.Runner}).RunAll(ctx, tasks, concurrency)
		results = append(results, SuiteResult{
			Competitor: competitor.Name,
			Results:    runnerResults,
			Summary:    Summarize(runnerResults),
		})
	}
	return results
}

// RankSuite converts suite results into the same deterministic ordering used
// by Rank.
func RankSuite(results []SuiteResult) []LeaderboardEntry {
	entries := make([]LeaderboardEntry, 0, len(results))
	for _, result := range results {
		entries = append(entries, LeaderboardEntry{Name: result.Competitor, Summary: result.Summary})
	}
	return Rank(entries)
}
