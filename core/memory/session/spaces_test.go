package session

import (
	"slices"
	"testing"
	"time"
)

func TestSpaceCloneAndRole(t *testing.T) {
	now := time.Now()
	space := &Space{
		Name:      "team",
		ACL:       map[string]SpaceRole{"alice": SpaceRoleWriter},
		ExpiresAt: now.Add(time.Hour),
		CreatedAt: now,
		UpdatedAt: now,
	}
	clone := space.clone()
	if clone == space || &clone.ACL == &space.ACL {
		t.Fatal("expected deep clone of space")
	}
	if clone.roleFor("alice") != SpaceRoleWriter {
		t.Fatalf("unexpected role: %v", clone.roleFor("alice"))
	}
	if !space.roleFor("alice").allowsWrite() {
		t.Fatal("expected writer role to allow writes")
	}
	if space.roleFor("bob") != "" {
		t.Fatal("expected unknown principal to have empty role")
	}
}

func TestSpaceRegistryUpsertAndGrant(t *testing.T) {
	sr := NewSpaceRegistry(time.Hour)
	sr.clock = func() time.Time { return time.Unix(0, 0) }
	space := sr.Upsert(" team ", 0, map[string]SpaceRole{"alice": SpaceRoleAdmin})
	if space == nil || space.Name != "team" {
		t.Fatalf("unexpected space: %#v", space)
	}
	if err := sr.Grant("team", "bob", SpaceRoleWriter, time.Minute); err != nil {
		t.Fatalf("unexpected grant error: %v", err)
	}
	if !sr.CanRead("team", "bob") || !sr.CanWrite("team", "bob") {
		t.Fatal("expected granted writer to have read/write access")
	}
	sr.Revoke("team", "bob")
	if sr.CanRead("team", "bob") {
		t.Fatal("expected revoke to remove access")
	}
}

func TestSpaceRegistryCheckErrors(t *testing.T) {
	sr := NewSpaceRegistry(time.Minute)
	sr.clock = func() time.Time { return time.Unix(0, 0) }
	if err := sr.Check("", "alice", false); err != ErrSpaceUnknown {
		t.Fatalf("expected ErrSpaceUnknown, got %v", err)
	}
	sr.Upsert("team", time.Second, map[string]SpaceRole{"alice": SpaceRoleReader})
	sr.clock = func() time.Time { return time.Unix(10, 0) }
	if err := sr.Check("team", "alice", false); err != ErrSpaceExpired {
		t.Fatalf("expected ErrSpaceExpired, got %v", err)
	}
	sr.clock = func() time.Time { return time.Unix(0, 0) }
	sr.Upsert("team", 0, map[string]SpaceRole{"alice": SpaceRoleReader})
	if err := sr.Check("team", "", false); err != ErrSpaceForbidden {
		t.Fatalf("expected ErrSpaceForbidden for empty principal, got %v", err)
	}
	if err := sr.Check("team", "alice", true); err != ErrSpaceForbidden {
		t.Fatalf("expected ErrSpaceForbidden for insufficient role, got %v", err)
	}
}

func TestSpaceRegistryListAndPrune(t *testing.T) {
	sr := NewSpaceRegistry(time.Minute)
	now := time.Unix(0, 0)
	sr.clock = func() time.Time { return now }
	sr.Upsert("alpha", time.Minute, map[string]SpaceRole{"alice": SpaceRoleReader})
	sr.Upsert("beta", 0, map[string]SpaceRole{"alice": SpaceRoleWriter})
	sr.Upsert("gamma", time.Second, map[string]SpaceRole{"bob": SpaceRoleAdmin})

	list := sr.List("alice")
	if len(list) != 2 || list[0] != "alpha" || list[1] != "beta" {
		t.Fatalf("unexpected list of spaces: %v", list)
	}

	sr.clock = func() time.Time { return now.Add(2 * time.Minute) }
	removed := sr.Prune()
	slices.Sort(removed)
	expected := []string{"alpha", "beta", "gamma"}
	if !slices.Equal(removed, expected) {
		t.Fatalf("expected expired spaces %v, got %v", expected, removed)
	}
}
