-- Migration tracking table
-- This table keeps track of which migrations have been applied

CREATE TABLE IF NOT EXISTS schema_migrations (
    version varchar(255) PRIMARY KEY,
    executed_at timestamptz NOT NULL DEFAULT now(),
    checksum varchar(64),
    execution_time_ms integer,
    rolled_back_at timestamptz
);

-- Index for faster lookups
CREATE INDEX idx_schema_migrations_executed_at ON schema_migrations(executed_at DESC);

-- Add comment
COMMENT ON TABLE schema_migrations IS 'Tracks applied database migrations';
COMMENT ON COLUMN schema_migrations.version IS 'Migration file name or version number';
COMMENT ON COLUMN schema_migrations.executed_at IS 'When the migration was applied';
COMMENT ON COLUMN schema_migrations.checksum IS 'MD5 checksum of migration file for integrity';
COMMENT ON COLUMN schema_migrations.execution_time_ms IS 'How long the migration took to execute';
COMMENT ON COLUMN schema_migrations.rolled_back_at IS 'When the migration was rolled back (if applicable)';