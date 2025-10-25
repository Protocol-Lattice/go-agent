//go:build neo4j

package store

import (
	"context"

	neo4j "github.com/neo4j/neo4j-go-driver/v5/neo4j"
)

type driverWrapper struct {
	driver neo4j.DriverWithContext
}

// WrapNeo4jDriver adapts the official Neo4j Go driver so it can be used with NewNeo4jStore.
func WrapNeo4jDriver(driver neo4j.DriverWithContext) neo4jDriver {
	if driver == nil {
		return nil
	}
	return &driverWrapper{driver: driver}
}

func (d *driverWrapper) NewSession(ctx context.Context, config Neo4jSessionConfig) (neo4jSession, error) {
	sessionConfig := neo4j.SessionConfig{DatabaseName: config.DatabaseName}
	switch config.AccessMode {
	case AccessModeWrite:
		sessionConfig.AccessMode = neo4j.AccessModeWrite
	case AccessModeRead:
		sessionConfig.AccessMode = neo4j.AccessModeRead
	}
	session, err := d.driver.NewSession(ctx, sessionConfig)
	if err != nil {
		return nil, err
	}
	return &sessionWrapper{session: session}, nil
}

func (d *driverWrapper) Close(ctx context.Context) error {
	return d.driver.Close(ctx)
}

type sessionWrapper struct {
	session neo4j.SessionWithContext
}

func (s *sessionWrapper) BeginTransaction(ctx context.Context) (neo4jTransaction, error) {
	tx, err := s.session.BeginTransaction(ctx)
	if err != nil {
		return nil, err
	}
	return &transactionWrapper{tx: tx}, nil
}

func (s *sessionWrapper) Run(ctx context.Context, query string, params map[string]any) (neo4jResult, error) {
	res, err := s.session.Run(ctx, query, params)
	if err != nil {
		return nil, err
	}
	return &resultWrapper{result: res}, nil
}

func (s *sessionWrapper) Close(ctx context.Context) error {
	return s.session.Close(ctx)
}

type transactionWrapper struct {
	tx neo4j.ExplicitTransaction
}

func (t *transactionWrapper) Run(ctx context.Context, query string, params map[string]any) (neo4jResult, error) {
	res, err := t.tx.Run(ctx, query, params)
	if err != nil {
		return nil, err
	}
	return &resultWrapper{result: res}, nil
}

func (t *transactionWrapper) Commit(ctx context.Context) error {
	return t.tx.Commit(ctx)
}

func (t *transactionWrapper) Rollback(ctx context.Context) error {
	return t.tx.Rollback(ctx)
}

func (t *transactionWrapper) Close(ctx context.Context) error {
	return t.tx.Close(ctx)
}

type resultWrapper struct {
	result neo4j.ResultWithContext
}

func (r *resultWrapper) Next(ctx context.Context) bool {
	return r.result.Next(ctx)
}

func (r *resultWrapper) Record() neo4jRecord {
	rec := r.result.Record()
	if rec == nil {
		return nil
	}
	return recordWrapper{record: rec}
}

func (r *resultWrapper) Err() error {
	return r.result.Err()
}

func (r *resultWrapper) Close(ctx context.Context) error {
	return r.result.Close(ctx)
}

type recordWrapper struct {
	record *neo4j.Record
}

func (r recordWrapper) Get(key string) (any, bool) {
	if r.record == nil {
		return nil, false
	}
	return r.record.Get(key)
}
