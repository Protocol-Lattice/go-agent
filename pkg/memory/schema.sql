CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS memory_bank (
    id SERIAL PRIMARY KEY,
    session_id TEXT,
    content TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS memory_embedding_idx
ON memory_bank USING ivfflat (embedding vector_l2_ops)
WITH (lists = 100);

CREATE INDEX IF NOT EXISTS memory_session_idx ON memory_bank (session_id);
CREATE INDEX IF NOT EXISTS memory_metadata_idx ON memory_bank USING GIN (metadata);
