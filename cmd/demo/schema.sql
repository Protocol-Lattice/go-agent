CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_bank (
    id BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL,
    content TEXT NOT NULL,
    metadata JSONB,
    embedding vector(768),
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS memory_session_idx ON memory_bank (session_id);
CREATE INDEX IF NOT EXISTS memory_embedding_idx ON memory_bank USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS importance DOUBLE PRECISION DEFAULT 0;
ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS source TEXT DEFAULT '';
ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS summary TEXT DEFAULT '';
ALTER TABLE memory_bank ADD COLUMN IF NOT EXISTS last_embedded TIMESTAMPTZ DEFAULT NOW();

CREATE TABLE IF NOT EXISTS memory_nodes (
    memory_id BIGINT PRIMARY KEY REFERENCES memory_bank(id) ON DELETE CASCADE,
    space TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS memory_edges (
    from_memory BIGINT NOT NULL REFERENCES memory_bank(id) ON DELETE CASCADE,
    to_memory BIGINT NOT NULL REFERENCES memory_bank(id) ON DELETE CASCADE,
    edge_type TEXT NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    PRIMARY KEY (from_memory, to_memory, edge_type)
);

CREATE INDEX IF NOT EXISTS memory_edges_to_idx ON memory_edges (to_memory);