package swarm

import (
	"context"
	"errors"
	"testing"

	"github.com/Protocol-Lattice/go-agent/src/memory"
)

// --- Fakes ---

type fakeShared struct {
	retrieveCalled bool
	retrieveQuery  string
	retrieveK      int

	leaveCalls []string

	joinCalls []string
	joinErr   error

	flushLocalCalls int

	spacesVal []string

	flushSpaceCalls []string
}

func (f *fakeShared) Retrieve(ctx context.Context, query string, k int) ([]memory.MemoryRecord, error) {
	f.retrieveCalled = true
	f.retrieveQuery = query
	f.retrieveK = k
	return nil, nil
}
func (f *fakeShared) Leave(space string) { f.leaveCalls = append(f.leaveCalls, space) }
func (f *fakeShared) Join(space string) error {
	f.joinCalls = append(f.joinCalls, space)
	return f.joinErr
}
func (f *fakeShared) FlushLocal(ctx context.Context) error {
	f.flushLocalCalls++
	return nil
}
func (f *fakeShared) Spaces() []string { return f.spacesVal }
func (f *fakeShared) FlushSpace(ctx context.Context, space string) error {
	f.flushSpaceCalls = append(f.flushSpaceCalls, space)
	return nil
}

type fakeAgent struct {
	grants []struct {
		sessionID string
		spaces    []string
	}
	sharedSetCalls int

	saves []struct {
		role, content string
	}
	generations []struct {
		sessionID string
		prompt    string
	}
	genResp string
	genErr  error
}

func (f *fakeAgent) EnsureSpaceGrants(sessionID string, spaces []string) {
	f.grants = append(f.grants, struct {
		sessionID string
		spaces    []string
	}{sessionID, append([]string(nil), spaces...)})
}
func (f *fakeAgent) SetSharedSpaces(shared SharedSession) { f.sharedSetCalls++ }
func (f *fakeAgent) Save(ctx context.Context, role, content string) {
	f.saves = append(f.saves, struct {
		role, content string
	}{role, content})
}
func (f *fakeAgent) Generate(ctx context.Context, sessionID, prompt string) (string, error) {
	f.generations = append(f.generations, struct {
		sessionID string
		prompt    string
	}{sessionID, prompt})
	return f.genResp, f.genErr
}

// --- Tests ---

// Ensures Save() is a no-op (and does not panic) when Shared == nil.
func TestParticipantSave_NoShared_NoPanic(t *testing.T) {
	t.Parallel()

	p := &Participant{
		Alias:     "researcher",
		SessionID: "cli:researcher",
		Agent:     nil, // intentionally nil
		Shared:    nil, // critical for this test
	}

	// Should not panic:
	p.Save(context.Background())
}

func TestParticipant_Retrieve_DelegatesFixedQueryAndK(t *testing.T) {
	t.Parallel()

	fs := &fakeShared{}
	p := &Participant{
		Alias:     "researcher",
		SessionID: "cli:researcher",
		Shared:    fs,
	}

	_, _ = p.Retrieve(context.Background())

	if !fs.retrieveCalled {
		t.Fatalf("expected Retrieve to be called on shared session")
	}
	if fs.retrieveQuery != "recent swarm updates" || fs.retrieveK != 5 {
		t.Fatalf("unexpected args: query=%q k=%d", fs.retrieveQuery, fs.retrieveK)
	}
}

func TestParticipant_Leave_ForwardsSpace(t *testing.T) {
	t.Parallel()

	fs := &fakeShared{}
	p := &Participant{
		Alias:     "planner",
		SessionID: "cli:planner",
		Shared:    fs,
	}

	p.Leave("team:core")

	if len(fs.leaveCalls) != 1 || fs.leaveCalls[0] != "team:core" {
		t.Fatalf("expected single Leave('team:core'), got %#v", fs.leaveCalls)
	}
}

func TestParticipant_Join_GrantsThenJoins_ReturnsBoolOnError(t *testing.T) {
	t.Parallel()

	// success path
	fs1 := &fakeShared{}
	fa1 := &fakeAgent{}
	p1 := &Participant{
		Alias:     "summarizer",
		SessionID: "cli:summarizer",
		Agent:     fa1,
		Shared:    fs1,
	}

	errFlag := p1.Join("team:shared")
	if errFlag {
		t.Fatalf("expected false on success")
	}
	// Ensure grants called once with proper args
	if len(fa1.grants) != 1 || fa1.grants[0].sessionID != "cli:summarizer" || len(fa1.grants[0].spaces) != 1 || fa1.grants[0].spaces[0] != "team:shared" {
		t.Fatalf("unexpected grants: %#v", fa1.grants)
	}
	if len(fs1.joinCalls) != 1 || fs1.joinCalls[0] != "team:shared" {
		t.Fatalf("expected Join('team:shared'), got %#v", fs1.joinCalls)
	}

	// error path
	fs2 := &fakeShared{joinErr: errors.New("boom")}
	fa2 := &fakeAgent{}
	p2 := &Participant{
		Alias:     "planner",
		SessionID: "cli:planner",
		Agent:     fa2,
		Shared:    fs2,
	}

	errFlag2 := p2.Join("team:core")
	if !errFlag2 {
		t.Fatalf("expected true on error")
	}
	if len(fa2.grants) != 1 || fa2.grants[0].sessionID != "cli:planner" || fa2.grants[0].spaces[0] != "team:core" {
		t.Fatalf("unexpected grants on error path: %#v", fa2.grants)
	}
	if len(fs2.joinCalls) != 1 || fs2.joinCalls[0] != "team:core" {
		t.Fatalf("expected Join('team:core') call even on error, got %#v", fs2.joinCalls)
	}
}

func TestParticipant_Save_FlushesLocalAndAllSpaces(t *testing.T) {
	t.Parallel()

	fs := &fakeShared{spacesVal: []string{"team:core", "team:shared"}}
	p := &Participant{
		Alias:     "researcher",
		SessionID: "cli:researcher",
		Shared:    fs,
	}

	p.Save(context.Background())

	if fs.flushLocalCalls != 1 {
		t.Fatalf("expected one FlushLocal call, got %d", fs.flushLocalCalls)
	}
	if len(fs.flushSpaceCalls) != 2 {
		t.Fatalf("expected two FlushSpace calls, got %d", len(fs.flushSpaceCalls))
	}
	if fs.flushSpaceCalls[0] != "team:core" || fs.flushSpaceCalls[1] != "team:shared" {
		t.Fatalf("unexpected FlushSpace order/args: %#v", fs.flushSpaceCalls)
	}
}
