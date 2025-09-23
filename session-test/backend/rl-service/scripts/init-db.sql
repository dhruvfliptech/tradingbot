-- RL Service Database Initialization
-- ==================================

-- Create database schema for RL service (optional persistence)

-- Create extension for UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schema for RL service
CREATE SCHEMA IF NOT EXISTS rl_service;

-- Set search path
SET search_path TO rl_service, public;

-- Table for storing agent configurations
CREATE TABLE IF NOT EXISTS agent_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    config_name VARCHAR(255) NOT NULL,
    config_data JSONB NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    
    UNIQUE(user_id, config_name)
);

-- Table for storing prediction history
CREATE TABLE IF NOT EXISTS prediction_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    symbol VARCHAR(50) NOT NULL,
    request_data JSONB NOT NULL,
    response_data JSONB NOT NULL,
    model_used VARCHAR(100) NOT NULL,
    processing_time_ms REAL NOT NULL,
    confidence REAL NOT NULL,
    fallback_used BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing model performance metrics
CREATE TABLE IF NOT EXISTS model_performance (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_name VARCHAR(100) NOT NULL,
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_metadata JSONB,
    measurement_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(model_name, metric_name, measurement_time)
);

-- Table for storing alerts
CREATE TABLE IF NOT EXISTS alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    alert_id VARCHAR(255) NOT NULL UNIQUE,
    title VARCHAR(255) NOT NULL,
    description TEXT NOT NULL,
    severity VARCHAR(50) NOT NULL,
    source VARCHAR(100) NOT NULL,
    metadata JSONB,
    resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Table for storing A/B test configurations
CREATE TABLE IF NOT EXISTS ab_test_configs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    test_name VARCHAR(255) NOT NULL UNIQUE,
    enabled BOOLEAN DEFAULT true,
    rl_traffic_percentage REAL NOT NULL CHECK (rl_traffic_percentage >= 0 AND rl_traffic_percentage <= 1),
    config_data JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Table for storing user assignment to A/B tests
CREATE TABLE IF NOT EXISTS ab_test_assignments (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id VARCHAR(255) NOT NULL,
    test_name VARCHAR(255) NOT NULL,
    assignment VARCHAR(100) NOT NULL, -- 'control' or 'treatment'
    assigned_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, test_name),
    FOREIGN KEY (test_name) REFERENCES ab_test_configs(test_name)
);

-- Table for storing system metrics
CREATE TABLE IF NOT EXISTS system_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value REAL NOT NULL,
    metric_labels JSONB,
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_agent_configs_user_id ON agent_configs(user_id);
CREATE INDEX IF NOT EXISTS idx_agent_configs_active ON agent_configs(is_active);

CREATE INDEX IF NOT EXISTS idx_prediction_history_user_id ON prediction_history(user_id);
CREATE INDEX IF NOT EXISTS idx_prediction_history_symbol ON prediction_history(symbol);
CREATE INDEX IF NOT EXISTS idx_prediction_history_created_at ON prediction_history(created_at);
CREATE INDEX IF NOT EXISTS idx_prediction_history_model_used ON prediction_history(model_used);

CREATE INDEX IF NOT EXISTS idx_model_performance_model_name ON model_performance(model_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_metric_name ON model_performance(metric_name);
CREATE INDEX IF NOT EXISTS idx_model_performance_time ON model_performance(measurement_time);

CREATE INDEX IF NOT EXISTS idx_alerts_resolved ON alerts(resolved);
CREATE INDEX IF NOT EXISTS idx_alerts_severity ON alerts(severity);
CREATE INDEX IF NOT EXISTS idx_alerts_created_at ON alerts(created_at);

CREATE INDEX IF NOT EXISTS idx_ab_test_assignments_user_id ON ab_test_assignments(user_id);
CREATE INDEX IF NOT EXISTS idx_ab_test_assignments_test_name ON ab_test_assignments(test_name);

CREATE INDEX IF NOT EXISTS idx_system_metrics_name_time ON system_metrics(metric_name, recorded_at);

-- Create views for common queries
CREATE OR REPLACE VIEW prediction_summary AS
SELECT 
    user_id,
    symbol,
    model_used,
    COUNT(*) as total_predictions,
    AVG(confidence) as avg_confidence,
    AVG(processing_time_ms) as avg_processing_time,
    SUM(CASE WHEN fallback_used THEN 1 ELSE 0 END) as fallback_count,
    DATE_TRUNC('hour', created_at) as hour_bucket
FROM prediction_history 
WHERE created_at >= NOW() - INTERVAL '24 hours'
GROUP BY user_id, symbol, model_used, hour_bucket
ORDER BY hour_bucket DESC;

CREATE OR REPLACE VIEW model_performance_summary AS
SELECT 
    model_name,
    metric_name,
    AVG(metric_value) as avg_value,
    MIN(metric_value) as min_value,
    MAX(metric_value) as max_value,
    COUNT(*) as measurement_count,
    DATE_TRUNC('hour', measurement_time) as hour_bucket
FROM model_performance 
WHERE measurement_time >= NOW() - INTERVAL '24 hours'
GROUP BY model_name, metric_name, hour_bucket
ORDER BY hour_bucket DESC;

CREATE OR REPLACE VIEW active_alerts_summary AS
SELECT 
    severity,
    source,
    COUNT(*) as alert_count,
    MIN(created_at) as oldest_alert,
    MAX(created_at) as newest_alert
FROM alerts 
WHERE resolved = false
GROUP BY severity, source
ORDER BY severity, source;

-- Function to clean up old data
CREATE OR REPLACE FUNCTION cleanup_old_data(retention_days INTEGER DEFAULT 30)
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER := 0;
    temp_count INTEGER;
BEGIN
    -- Clean up old prediction history
    DELETE FROM prediction_history 
    WHERE created_at < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Clean up old model performance data
    DELETE FROM model_performance 
    WHERE measurement_time < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Clean up old resolved alerts
    DELETE FROM alerts 
    WHERE resolved = true AND resolved_at < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    -- Clean up old system metrics
    DELETE FROM system_metrics 
    WHERE recorded_at < NOW() - (retention_days || ' days')::INTERVAL;
    GET DIAGNOSTICS temp_count = ROW_COUNT;
    deleted_count := deleted_count + temp_count;
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Function to get user statistics
CREATE OR REPLACE FUNCTION get_user_statistics(input_user_id VARCHAR(255))
RETURNS TABLE (
    total_predictions BIGINT,
    avg_confidence REAL,
    avg_processing_time REAL,
    fallback_rate REAL,
    most_used_model VARCHAR(100),
    last_prediction_time TIMESTAMP WITH TIME ZONE
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COUNT(*)::BIGINT as total_predictions,
        AVG(ph.confidence)::REAL as avg_confidence,
        AVG(ph.processing_time_ms)::REAL as avg_processing_time,
        (SUM(CASE WHEN ph.fallback_used THEN 1 ELSE 0 END)::REAL / COUNT(*)::REAL) as fallback_rate,
        (
            SELECT ph2.model_used 
            FROM prediction_history ph2 
            WHERE ph2.user_id = input_user_id 
            GROUP BY ph2.model_used 
            ORDER BY COUNT(*) DESC 
            LIMIT 1
        ) as most_used_model,
        MAX(ph.created_at) as last_prediction_time
    FROM prediction_history ph
    WHERE ph.user_id = input_user_id
    AND ph.created_at >= NOW() - INTERVAL '7 days';
END;
$$ LANGUAGE plpgsql;

-- Insert default A/B test configuration
INSERT INTO ab_test_configs (test_name, enabled, rl_traffic_percentage, config_data)
VALUES (
    'default_rl_test',
    true,
    0.1,
    '{"description": "Default RL vs Fallback A/B test", "created_by": "system"}'
) ON CONFLICT (test_name) DO NOTHING;

-- Grant permissions to the rl_user
GRANT USAGE ON SCHEMA rl_service TO rl_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA rl_service TO rl_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA rl_service TO rl_user;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA rl_service TO rl_user;

-- Set default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA rl_service GRANT ALL ON TABLES TO rl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA rl_service GRANT ALL ON SEQUENCES TO rl_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA rl_service GRANT EXECUTE ON FUNCTIONS TO rl_user;

-- Create a cleanup job (this would typically be run by a cron job or scheduler)
-- For demonstration purposes, we'll create a function that can be called periodically
CREATE OR REPLACE FUNCTION schedule_cleanup()
RETURNS void AS $$
BEGIN
    -- This function can be called by an external scheduler
    -- to perform regular cleanup of old data
    PERFORM cleanup_old_data(30); -- Keep 30 days of data
END;
$$ LANGUAGE plpgsql;

-- Log successful initialization
DO $$
BEGIN
    RAISE NOTICE 'RL Service database schema initialized successfully';
END $$;