-- Trading Bot Backend Database Schema
-- Extension of existing Supabase schema

-- Trading Sessions Table
CREATE TABLE IF NOT EXISTS trading_sessions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    status text NOT NULL DEFAULT 'stopped' CHECK (status IN ('running', 'stopped', 'paused', 'error')),
    started_at timestamptz,
    stopped_at timestamptz,
    watchlist text[] DEFAULT '{}',
    settings jsonb DEFAULT '{}',
    total_trades integer DEFAULT 0,
    total_pnl decimal(20,8) DEFAULT 0,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Trading Signals Table
CREATE TABLE IF NOT EXISTS trading_signals (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id uuid REFERENCES trading_sessions(id) ON DELETE SET NULL,
    symbol text NOT NULL,
    action text NOT NULL CHECK (action IN ('BUY', 'SELL', 'HOLD')),
    confidence decimal(5,4) NOT NULL,
    reason text,
    price_target decimal(20,8),
    stop_loss decimal(20,8),
    current_price decimal(20,8),
    indicators jsonb DEFAULT '{}',
    executed boolean DEFAULT false,
    created_at timestamptz DEFAULT now()
);

-- Enhanced Orders Table
CREATE TABLE IF NOT EXISTS orders (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id uuid REFERENCES trading_sessions(id) ON DELETE SET NULL,
    signal_id uuid REFERENCES trading_signals(id) ON DELETE SET NULL,
    external_order_id text,
    symbol text NOT NULL,
    side text NOT NULL CHECK (side IN ('buy', 'sell')),
    order_type text NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    quantity decimal(20,8) NOT NULL,
    price decimal(20,8),
    filled_quantity decimal(20,8) DEFAULT 0,
    avg_fill_price decimal(20,8),
    status text NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'partially_filled', 'cancelled', 'rejected')),
    fees decimal(20,8) DEFAULT 0,
    created_at timestamptz DEFAULT now(),
    filled_at timestamptz,
    cancelled_at timestamptz
);

-- Positions Table
CREATE TABLE IF NOT EXISTS positions (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    symbol text NOT NULL,
    quantity decimal(20,8) NOT NULL,
    avg_cost decimal(20,8) NOT NULL,
    current_price decimal(20,8),
    unrealized_pnl decimal(20,8),
    realized_pnl decimal(20,8) DEFAULT 0,
    opened_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(user_id, symbol)
);

-- Trades Table (Completed trades)
CREATE TABLE IF NOT EXISTS trades (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    session_id uuid REFERENCES trading_sessions(id) ON DELETE SET NULL,
    signal_id uuid REFERENCES trading_signals(id) ON DELETE SET NULL,
    buy_order_id uuid REFERENCES orders(id),
    sell_order_id uuid REFERENCES orders(id),
    symbol text NOT NULL,
    entry_price decimal(20,8) NOT NULL,
    exit_price decimal(20,8) NOT NULL,
    quantity decimal(20,8) NOT NULL,
    pnl decimal(20,8) NOT NULL,
    pnl_percent decimal(8,4) NOT NULL,
    fees decimal(20,8) DEFAULT 0,
    hold_duration interval,
    opened_at timestamptz NOT NULL,
    closed_at timestamptz DEFAULT now()
);

-- Market Data Table (Time series)
CREATE TABLE IF NOT EXISTS market_data (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    symbol text NOT NULL,
    price decimal(20,8) NOT NULL,
    volume decimal(20,8),
    market_cap bigint,
    change_24h decimal(8,4),
    indicators jsonb DEFAULT '{}',
    timestamp timestamptz NOT NULL,
    created_at timestamptz DEFAULT now()
);

-- Adaptive Threshold Data
CREATE TABLE IF NOT EXISTS adaptive_thresholds (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    symbol text,
    parameter_name text NOT NULL,
    current_value decimal(10,6) NOT NULL,
    initial_value decimal(10,6) NOT NULL,
    performance_window integer DEFAULT 100,
    adaptation_rate decimal(6,4) DEFAULT 0.01,
    last_performance decimal(8,4),
    adaptation_count integer DEFAULT 0,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(user_id, symbol, parameter_name)
);

-- API Keys Table (Encrypted)
CREATE TABLE IF NOT EXISTS user_api_keys (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider text NOT NULL CHECK (provider IN ('alpaca', 'binance', 'coinbase')),
    encrypted_api_key text NOT NULL,
    encrypted_secret_key text NOT NULL,
    is_paper_trading boolean DEFAULT true,
    is_active boolean DEFAULT true,
    last_used_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now(),
    UNIQUE(user_id, provider)
);

-- User Settings Table
CREATE TABLE IF NOT EXISTS user_settings (
    user_id uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    trading_settings jsonb DEFAULT '{}',
    notification_settings jsonb DEFAULT '{}',
    risk_settings jsonb DEFAULT '{}',
    ui_preferences jsonb DEFAULT '{}',
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_trading_sessions_user_status ON trading_sessions(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trading_signals_user_created ON trading_signals(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_orders_user_status ON orders(user_id, status);
CREATE INDEX IF NOT EXISTS idx_trades_user_closed ON trades(user_id, closed_at DESC);
CREATE INDEX IF NOT EXISTS idx_market_data_symbol_time ON market_data(symbol, timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_positions_user_symbol ON positions(user_id, symbol);
CREATE INDEX IF NOT EXISTS idx_adaptive_thresholds_user_symbol ON adaptive_thresholds(user_id, symbol);

-- Enable Row Level Security
ALTER TABLE trading_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_signals ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE adaptive_thresholds ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_api_keys ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can manage own trading sessions" ON trading_sessions
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own trading signals" ON trading_signals
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own orders" ON orders
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own positions" ON positions
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own trades" ON trades
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own adaptive thresholds" ON adaptive_thresholds
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own API keys" ON user_api_keys
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can manage own settings" ON user_settings
    FOR ALL TO authenticated USING (auth.uid() = user_id) WITH CHECK (auth.uid() = user_id);

-- Market data is readable by all authenticated users
CREATE POLICY "Authenticated users can read market data" ON market_data
    FOR SELECT TO authenticated USING (true);

-- Functions for updating timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at columns
CREATE TRIGGER trg_trading_sessions_updated_at BEFORE UPDATE ON trading_sessions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_adaptive_thresholds_updated_at BEFORE UPDATE ON adaptive_thresholds
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_user_api_keys_updated_at BEFORE UPDATE ON user_api_keys
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trg_user_settings_updated_at BEFORE UPDATE ON user_settings
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();