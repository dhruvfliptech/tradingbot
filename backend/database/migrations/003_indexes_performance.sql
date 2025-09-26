-- Performance Optimization Indexes
-- Add after initial data load to avoid slowing down inserts

-- Trading Performance Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_user_created
    ON orders(user_id, created_at DESC)
    WHERE status != 'canceled';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_symbol_time
    ON orders(symbol, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_user_status
    ON positions(user_id, status)
    WHERE status = 'open';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_trades_position_time
    ON trades(position_id, executed_at DESC);

-- Signal Analysis Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_symbol_confidence
    ON trading_signals(symbol, confidence DESC)
    WHERE executed = false;

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_signals_session_time
    ON trading_signals(session_id, generated_at DESC);

-- Performance Metrics Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_daily_performance_user_date
    ON daily_performance(user_id, date DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_performance_metrics_session
    ON performance_metrics(session_id, calculated_at DESC);

-- Backtesting Indexes
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_trades_result_time
    ON backtest_trades(result_id, timestamp DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_backtest_results_strategy_score
    ON backtest_results(strategy_id, sharpe_ratio DESC NULLS LAST);

-- Partial Indexes for Common Queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_active_sessions
    ON trading_sessions(user_id, started_at DESC)
    WHERE status IN ('active', 'paused');

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_pending_orders
    ON orders(user_id, created_at DESC)
    WHERE status = 'pending';

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_profitable_trades
    ON trades(user_id, executed_at DESC)
    WHERE profit > 0;

-- Composite Indexes for Complex Queries
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_composite
    ON orders(user_id, symbol, status, created_at DESC);

CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_positions_composite
    ON positions(user_id, symbol, status, opened_at DESC);

-- Text Search Indexes (if needed)
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_orders_notes_search
    ON orders USING gin(to_tsvector('english', notes))
    WHERE notes IS NOT NULL;

-- JSON Indexes for Configuration Fields
CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_session_config_watchlist
    ON trading_sessions USING gin((config->'watchlist'))
    WHERE config IS NOT NULL;

-- Statistics Update
ANALYZE orders;
ANALYZE positions;
ANALYZE trades;
ANALYZE trading_signals;
ANALYZE trading_sessions;