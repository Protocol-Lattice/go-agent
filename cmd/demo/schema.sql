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