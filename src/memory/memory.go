package memory

import (
	embedpkg "github.com/Protocol-Lattice/go-agent/src/memory/embed"
	memengine "github.com/Protocol-Lattice/go-agent/src/memory/engine"
	"github.com/Protocol-Lattice/go-agent/src/memory/model"
	sessionpkg "github.com/Protocol-Lattice/go-agent/src/memory/session"
	storepkg "github.com/Protocol-Lattice/go-agent/src/memory/store"
)

// Type aliases preserving the original public API.
type (
	Engine              = memengine.Engine
	Options             = memengine.Options
	ScoreWeights        = memengine.ScoreWeights
	Metrics             = memengine.Metrics
	MetricsSnapshot     = memengine.MetricsSnapshot
	Summarizer          = memengine.Summarizer
	HeuristicSummarizer = memengine.HeuristicSummarizer

	MemoryRecord = model.MemoryRecord
	GraphEdge    = model.GraphEdge
	EdgeType     = model.EdgeType

	MemoryBank    = sessionpkg.MemoryBank
	SessionMemory = sessionpkg.SessionMemory
	SpaceRegistry = sessionpkg.SpaceRegistry
	SpaceRole     = sessionpkg.SpaceRole
	Space         = sessionpkg.Space
	SharedSession = sessionpkg.SharedSession

	VectorStore       = storepkg.VectorStore
	SchemaInitializer = storepkg.SchemaInitializer
	GraphStore        = storepkg.GraphStore

	InMemoryStore           = storepkg.InMemoryStore
	PostgresStore           = storepkg.PostgresStore
	QdrantStore             = storepkg.QdrantStore
	Neo4jStore              = storepkg.Neo4jStore
	Distance                = storepkg.Distance
	CreateCollectionRequest = storepkg.CreateCollectionRequest

	Embedder      = embedpkg.Embedder
	DummyEmbedder = embedpkg.DummyEmbedder
)

const (
	EdgeFollows     = model.EdgeFollows
	EdgeExplains    = model.EdgeExplains
	EdgeContradicts = model.EdgeContradicts
	EdgeDerivedFrom = model.EdgeDerivedFrom

	SpaceRoleReader = sessionpkg.SpaceRoleReader
	SpaceRoleWriter = sessionpkg.SpaceRoleWriter
	SpaceRoleAdmin  = sessionpkg.SpaceRoleAdmin
)

var (
	ErrNotSupported = embedpkg.ErrNotSupported

	NewEngine              = memengine.NewEngine
	DefaultOptions         = memengine.DefaultOptions
	NewMemoryBank          = sessionpkg.NewMemoryBank
	NewMemoryBankWithStore = sessionpkg.NewMemoryBankWithStore
	NewSessionMemory       = sessionpkg.NewSessionMemory
	NewSharedSession       = sessionpkg.NewSharedSession
	NewSpaceRegistry       = sessionpkg.NewSpaceRegistry

	AutoEmbedder        = embedpkg.AutoEmbedder
	DummyEmbedding      = embedpkg.DummyEmbedding
	NewOpenAIEmbedder   = embedpkg.NewOpenAIEmbedder
	NewVertexAIEmbedder = embedpkg.NewVertexAIEmbedder
	NewOllamaEmbedder   = embedpkg.NewOllamaEmbedder
	NewClaudeEmbedder   = embedpkg.NewClaudeEmbedder

	NewInMemoryStore = storepkg.NewInMemoryStore
	NewPostgresStore = storepkg.NewPostgresStore
	NewQdrantStore   = storepkg.NewQdrantStore
	NewNeo4jStore    = storepkg.NewNeo4jStore
)
