-- Backtesting and Strategy Validation Database Schema
-- Extension for Composer MCP integration

-- Strategy Definitions Table
CREATE TABLE IF NOT EXISTS strategy_definitions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    name text NOT NULL,
    description text,
    strategy_type text NOT NULL CHECK (strategy_type IN ('momentum', 'mean_reversion', 'arbitrage', 'ml_based', 'custom')),
    version text NOT NULL DEFAULT '1.0.0',
    parameters jsonb NOT NULL DEFAULT '{}',
    entry_rules jsonb NOT NULL DEFAULT '[]',
    exit_rules jsonb NOT NULL DEFAULT '[]',
    risk_management jsonb NOT NULL DEFAULT '{}',
    is_active boolean DEFAULT true,
    is_validated boolean DEFAULT false,
    validation_score decimal(5,4),
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(user_id, name, version)
);

-- Backtest Configurations Table
CREATE TABLE IF NOT EXISTS backtest_configs (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    strategy_id uuid NOT NULL REFERENCES strategy_definitions(id) ON DELETE CASCADE,
    name text NOT NULL,
    symbols text[] NOT NULL,
    start_date timestamptz NOT NULL,
    end_date timestamptz NOT NULL,
    initial_capital decimal(20,8) NOT NULL DEFAULT 10000,
    timeframe text NOT NULL DEFAULT '1h' CHECK (timeframe IN ('1m', '5m', '15m', '1h', '4h', '1d')),
    commission_rate decimal(8,6) DEFAULT 0.001,
    slippage_rate decimal(8,6) DEFAULT 0.0005,
    strategy_parameters jsonb DEFAULT '{}',
    risk_settings jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Backtest Results Table
CREATE TABLE IF NOT EXISTS backtest_results (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    config_id uuid NOT NULL REFERENCES backtest_configs(id) ON DELETE CASCADE,
    strategy_id uuid NOT NULL REFERENCES strategy_definitions(id) ON DELETE CASCADE,
    external_backtest_id text UNIQUE, -- ID from Composer MCP
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress decimal(5,4) DEFAULT 0,
    
    -- Performance Metrics
    total_return decimal(10,6),
    sharpe_ratio decimal(8,4),
    sortino_ratio decimal(8,4),
    calmar_ratio decimal(8,4),
    max_drawdown decimal(8,4),
    win_rate decimal(5,4),
    profit_factor decimal(8,4),
    volatility decimal(8,4),
    beta decimal(8,4),
    alpha decimal(8,4),
    
    -- Trade Metrics
    total_trades integer,
    winning_trades integer,
    losing_trades integer,
    avg_win decimal(10,6),
    avg_loss decimal(10,6),
    largest_win decimal(10,6),
    largest_loss decimal(10,6),
    consecutive_wins integer,
    consecutive_losses integer,
    avg_trade_duration interval,
    total_commissions decimal(20,8),
    
    -- Risk Metrics
    var_95 decimal(10,6), -- Value at Risk 95%
    var_99 decimal(10,6), -- Value at Risk 99%
    expected_shortfall decimal(10,6),
    tracking_error decimal(8,4),
    information_ratio decimal(8,4),
    downside_deviation decimal(8,4),
    
    -- Raw Results (JSON storage)
    detailed_metrics jsonb DEFAULT '{}',
    equity_curve jsonb DEFAULT '[]',
    drawdown_periods jsonb DEFAULT '[]',
    monthly_returns jsonb DEFAULT '{}',
    
    error_message text,
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Backtest Trades Table (Individual trades from backtests)
CREATE TABLE IF NOT EXISTS backtest_trades (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    backtest_result_id uuid NOT NULL REFERENCES backtest_results(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    trade_number integer NOT NULL,
    symbol text NOT NULL,
    side text NOT NULL CHECK (side IN ('buy', 'sell')),
    entry_price decimal(20,8) NOT NULL,
    exit_price decimal(20,8),
    quantity decimal(20,8) NOT NULL,
    entry_time timestamptz NOT NULL,
    exit_time timestamptz,
    pnl decimal(20,8),
    pnl_percent decimal(8,4),
    commission decimal(20,8) DEFAULT 0,
    slippage decimal(20,8) DEFAULT 0,
    hold_duration interval,
    entry_reason text,
    exit_reason text,
    confidence decimal(5,4),
    indicators jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now()
);

-- Strategy Optimization Table
CREATE TABLE IF NOT EXISTS strategy_optimizations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    strategy_id uuid NOT NULL REFERENCES strategy_definitions(id) ON DELETE CASCADE,
    config_id uuid NOT NULL REFERENCES backtest_configs(id) ON DELETE CASCADE,
    external_optimization_id text UNIQUE, -- ID from Composer MCP
    
    optimization_method text NOT NULL CHECK (optimization_method IN ('grid', 'genetic', 'bayesian', 'random')),
    objective_function text NOT NULL CHECK (objective_function IN ('sharpe', 'return', 'profit_factor', 'sortino', 'calmar')),
    parameters_to_optimize text[] NOT NULL,
    iterations integer NOT NULL DEFAULT 100,
    
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    progress decimal(5,4) DEFAULT 0,
    
    -- Best Results
    best_parameters jsonb,
    best_performance decimal(10,6),
    best_metrics jsonb,
    
    -- All Results
    optimization_results jsonb DEFAULT '[]',
    parameter_importance jsonb DEFAULT '{}',
    
    error_message text,
    started_at timestamptz,
    completed_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Historical Data Cache Table
CREATE TABLE IF NOT EXISTS historical_data_cache (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol text NOT NULL,
    timeframe text NOT NULL CHECK (timeframe IN ('1m', '5m', '15m', '1h', '4h', '1d')),
    start_date timestamptz NOT NULL,
    end_date timestamptz NOT NULL,
    data_hash text NOT NULL, -- Hash of the data for integrity
    data_size integer NOT NULL, -- Number of data points
    compression_type text DEFAULT 'gzip',
    compressed_data bytea, -- Compressed OHLCV data
    indicators jsonb DEFAULT '{}', -- Pre-calculated indicators
    last_accessed timestamptz DEFAULT now(),
    expires_at timestamptz,
    created_at timestamptz DEFAULT now(),
    UNIQUE(symbol, timeframe, start_date, end_date)
);

-- Market Regime Analysis Table
CREATE TABLE IF NOT EXISTS market_regimes (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    symbols text[] NOT NULL,
    start_date timestamptz NOT NULL,
    end_date timestamptz NOT NULL,
    regime_type text NOT NULL CHECK (regime_type IN ('bull', 'bear', 'sideways', 'volatile')),
    characteristics jsonb NOT NULL DEFAULT '{}', -- volatility, trend, momentum
    confidence decimal(5,4) NOT NULL,
    analyzed_at timestamptz DEFAULT now(),
    created_at timestamptz DEFAULT now()
);

-- Strategy Performance in Different Regimes
CREATE TABLE IF NOT EXISTS strategy_regime_performance (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id uuid NOT NULL REFERENCES strategy_definitions(id) ON DELETE CASCADE,
    regime_id uuid NOT NULL REFERENCES market_regimes(id) ON DELETE CASCADE,
    backtest_result_id uuid NOT NULL REFERENCES backtest_results(id) ON DELETE CASCADE,
    
    period_start timestamptz NOT NULL,
    period_end timestamptz NOT NULL,
    
    -- Performance in this regime
    return_in_regime decimal(10,6),
    sharpe_in_regime decimal(8,4),
    max_drawdown_in_regime decimal(8,4),
    trades_in_regime integer,
    win_rate_in_regime decimal(5,4),
    
    -- Relative performance vs benchmark
    excess_return decimal(10,6),
    outperformance_ratio decimal(8,4),
    
    created_at timestamptz DEFAULT now(),
    UNIQUE(strategy_id, regime_id, backtest_result_id)
);

-- Adaptive Threshold Training Data
CREATE TABLE IF NOT EXISTS adaptive_threshold_training (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    strategy_id uuid REFERENCES strategy_definitions(id) ON DELETE CASCADE,
    backtest_result_id uuid REFERENCES backtest_results(id) ON DELETE CASCADE,
    
    training_period_start timestamptz NOT NULL,
    training_period_end timestamptz NOT NULL,
    
    -- Input features
    market_conditions jsonb NOT NULL DEFAULT '{}',
    strategy_parameters jsonb NOT NULL DEFAULT '{}',
    threshold_values jsonb NOT NULL DEFAULT '{}',
    
    -- Target outcomes
    performance_score decimal(8,4) NOT NULL,
    sharpe_ratio decimal(8,4),
    max_drawdown decimal(8,4),
    win_rate decimal(5,4),
    
    -- Training metadata
    data_quality_score decimal(5,4),
    regime_type text,
    volatility_bucket text CHECK (volatility_bucket IN ('low', 'medium', 'high')),
    
    created_at timestamptz DEFAULT now()
);

-- Strategy Validation Results
CREATE TABLE IF NOT EXISTS strategy_validations (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    strategy_id uuid NOT NULL REFERENCES strategy_definitions(id) ON DELETE CASCADE,
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    
    validation_type text NOT NULL CHECK (validation_type IN ('syntax', 'logic', 'performance', 'risk', 'comprehensive')),
    status text NOT NULL CHECK (status IN ('passed', 'failed', 'warning')),
    score decimal(5,4),
    
    -- Validation Results
    errors jsonb DEFAULT '[]',
    warnings jsonb DEFAULT '[]',
    recommendations jsonb DEFAULT '[]',
    
    -- Performance Validation (if applicable)
    min_sharpe_required decimal(8,4),
    actual_sharpe decimal(8,4),
    min_win_rate_required decimal(5,4),
    actual_win_rate decimal(5,4),
    max_drawdown_allowed decimal(8,4),
    actual_max_drawdown decimal(8,4),
    
    validated_at timestamptz DEFAULT now(),
    expires_at timestamptz,
    created_at timestamptz DEFAULT now()
);

-- Indexes for Performance
CREATE INDEX IF NOT EXISTS idx_strategy_definitions_user_active ON strategy_definitions(user_id, is_active);
CREATE INDEX IF NOT EXISTS idx_backtest_configs_user_strategy ON backtest_configs(user_id, strategy_id);
CREATE INDEX IF NOT EXISTS idx_backtest_results_user_status ON backtest_results(user_id, status);
CREATE INDEX IF NOT EXISTS idx_backtest_results_external_id ON backtest_results(external_backtest_id);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_result_symbol ON backtest_trades(backtest_result_id, symbol);
CREATE INDEX IF NOT EXISTS idx_backtest_trades_user_time ON backtest_trades(user_id, entry_time DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_optimizations_user_status ON strategy_optimizations(user_id, status);
CREATE INDEX IF NOT EXISTS idx_historical_data_cache_symbol_timeframe ON historical_data_cache(symbol, timeframe, start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_historical_data_cache_accessed ON historical_data_cache(last_accessed);
CREATE INDEX IF NOT EXISTS idx_market_regimes_dates ON market_regimes(start_date, end_date);
CREATE INDEX IF NOT EXISTS idx_adaptive_threshold_training_user ON adaptive_threshold_training(user_id, training_period_start DESC);
CREATE INDEX IF NOT EXISTS idx_strategy_validations_strategy ON strategy_validations(strategy_id, validated_at DESC);

-- Enable Row Level Security
ALTER TABLE strategy_definitions ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_configs ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_results ENABLE ROW LEVEL SECURITY;
ALTER TABLE backtest_trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_optimizations ENABLE ROW LEVEL SECURITY;
ALTER TABLE adaptive_threshold_training ENABLE ROW LEVEL SECURITY;
ALTER TABLE strategy_validations ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can manage own strategies" ON strategy_definitions
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own backtest configs" ON backtest_configs
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own backtest results" ON backtest_results
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own backtest trades" ON backtest_trades
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own optimizations" ON strategy_optimizations
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own training data" ON adaptive_threshold_training
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own validations" ON strategy_validations
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Historical data cache and market regimes are readable by all authenticated users
CREATE POLICY "Authenticated users can read historical data cache" ON historical_data_cache
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Authenticated users can read market regimes" ON market_regimes
    FOR SELECT TO authenticated USING (true);

CREATE POLICY "Authenticated users can read strategy regime performance" ON strategy_regime_performance
    FOR SELECT TO authenticated USING (true);

-- Admin policies for cache management
CREATE POLICY "Service role can manage historical data cache" ON historical_data_cache
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role can manage market regimes" ON market_regimes
    FOR ALL TO service_role USING (true) WITH CHECK (true);

-- Triggers for updated_at columns
CREATE TRIGGER trg_strategy_definitions_updated_at BEFORE UPDATE ON strategy_definitions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_backtest_configs_updated_at BEFORE UPDATE ON backtest_configs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_backtest_results_updated_at BEFORE UPDATE ON backtest_results
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_strategy_optimizations_updated_at BEFORE UPDATE ON strategy_optimizations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Functions for data cleanup
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS void AS $$
BEGIN
    DELETE FROM historical_data_cache 
    WHERE expires_at IS NOT NULL AND expires_at < now();
    
    DELETE FROM strategy_validations 
    WHERE expires_at IS NOT NULL AND expires_at < now();
END;
$$ LANGUAGE plpgsql;

-- Function to calculate strategy performance score
CREATE OR REPLACE FUNCTION calculate_performance_score(
    p_sharpe_ratio decimal,
    p_max_drawdown decimal,
    p_win_rate decimal,
    p_profit_factor decimal
) RETURNS decimal AS $$
DECLARE
    score decimal := 0;
BEGIN
    -- Weighted performance score (0-100)
    score := (
        COALESCE(p_sharpe_ratio, 0) * 30 +
        (1 - COALESCE(p_max_drawdown, 1)) * 25 +
        COALESCE(p_win_rate, 0) * 20 +
        LEAST(COALESCE(p_profit_factor, 0), 3) / 3 * 25
    );
    
    RETURN GREATEST(0, LEAST(100, score));
END;
$$ LANGUAGE plpgsql;

-- Scheduled job function (to be called by cron)
CREATE OR REPLACE FUNCTION scheduled_cache_cleanup()
RETURNS void AS $$
BEGIN
    -- Clean up expired cache entries
    PERFORM cleanup_expired_cache();
    
    -- Update last_accessed for frequently used cache entries
    UPDATE historical_data_cache 
    SET last_accessed = now() 
    WHERE last_accessed > now() - interval '1 hour';
    
    -- Log cleanup activity
    INSERT INTO system_logs (level, message, created_at) 
    VALUES ('INFO', 'Cache cleanup completed', now());
EXCEPTION WHEN OTHERS THEN
    -- Log errors but don't fail
    INSERT INTO system_logs (level, message, error_details, created_at) 
    VALUES ('ERROR', 'Cache cleanup failed', SQLERRM, now());
END;
$$ LANGUAGE plpgsql;