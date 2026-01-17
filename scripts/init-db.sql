-- Entity Resolution Database Initialization
-- This script runs when PostgreSQL container starts

-- Create extensions
CREATE EXTENSION IF NOT EXISTS pg_trgm;
CREATE EXTENSION IF NOT EXISTS fuzzystrmatch;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Set similarity threshold for trigram matching
SET pg_trgm.similarity_threshold = 0.3;

-- Create persons table
CREATE TABLE IF NOT EXISTS persons (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    first_name VARCHAR(100) NOT NULL,
    middle_name VARCHAR(100),
    last_name VARCHAR(100) NOT NULL,
    birth_date DATE,
    death_date DATE,
    ssn_last4 VARCHAR(4),
    ssn_hash VARCHAR(64),
    city VARCHAR(100),
    state VARCHAR(2),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),
    source_system VARCHAR(50) DEFAULT 'synthetic'
);

-- Create match_jobs table for bulk processing
CREATE TABLE IF NOT EXISTS match_jobs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    status VARCHAR(20) DEFAULT 'pending',
    total_records INTEGER DEFAULT 0,
    processed_records INTEGER DEFAULT 0,
    matched_records INTEGER DEFAULT 0,
    original_filename VARCHAR(255),
    file_format VARCHAR(20),
    results_path VARCHAR(500),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    user_id VARCHAR(255)
);

-- Create users table for authentication (Phase 3)
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255),
    picture VARCHAR(500),
    provider VARCHAR(50) NOT NULL,
    provider_id VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_admin BOOLEAN DEFAULT false,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    last_login TIMESTAMPTZ,
    UNIQUE(provider, provider_id)
);

-- B-tree indexes for exact matching
CREATE INDEX IF NOT EXISTS idx_persons_first_name ON persons(first_name);
CREATE INDEX IF NOT EXISTS idx_persons_last_name ON persons(last_name);
CREATE INDEX IF NOT EXISTS idx_persons_birth_date ON persons(birth_date);
CREATE INDEX IF NOT EXISTS idx_persons_state ON persons(state);
CREATE INDEX IF NOT EXISTS idx_persons_ssn_last4 ON persons(ssn_last4);
CREATE INDEX IF NOT EXISTS idx_persons_ssn_hash ON persons(ssn_hash);

-- Composite indexes for blocking strategies
CREATE INDEX IF NOT EXISTS idx_persons_blocking_name_dob ON persons(last_name, birth_date);
CREATE INDEX IF NOT EXISTS idx_persons_blocking_state_name ON persons(state, last_name);

-- Note: GIN indexes for trigram matching should be created after data load
-- They are slow to build and update, but fast for queries
-- See seed_database.py for index creation

-- Function to update timestamp
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for updated_at
DROP TRIGGER IF EXISTS trigger_persons_updated_at ON persons;
CREATE TRIGGER trigger_persons_updated_at
    BEFORE UPDATE ON persons
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Grant permissions (for production, use more restrictive permissions)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;

-- Verify setup
DO $$
BEGIN
    RAISE NOTICE 'Entity Resolution database initialized successfully';
    RAISE NOTICE 'Extensions: pg_trgm, fuzzystrmatch, uuid-ossp';
END $$;
