-- Supabase Database Setup Script
-- Run this in your Supabase SQL Editor: https://sjtulkkhxojiitpjhgrt.supabase.co

-- 1. API Keys Table
CREATE TABLE IF NOT EXISTS public.api_keys (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,
    key_name VARCHAR(100) NOT NULL,
    encrypted_value TEXT NOT NULL,
    is_active BOOLEAN DEFAULT true,
    last_validated_at TIMESTAMP WITH TIME ZONE,
    validation_status VARCHAR(20) DEFAULT 'pending',
    validation_error TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    CONSTRAINT unique_user_provider_key UNIQUE (user_id, provider, key_name)
);

CREATE INDEX IF NOT EXISTS idx_api_keys_user_provider ON public.api_keys(user_id, provider);
ALTER TABLE public.api_keys ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Users can view their own API keys" ON public.api_keys
    FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert their own API keys" ON public.api_keys
    FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update their own API keys" ON public.api_keys
    FOR UPDATE USING (auth.uid() = user_id);
CREATE POLICY "Users can delete their own API keys" ON public.api_keys
    FOR DELETE USING (auth.uid() = user_id);

-- 2. Trade History Table
CREATE TABLE IF NOT EXISTS trade_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  symbol VARCHAR(20) NOT NULL,
  side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell', 'short', 'cover')),
  quantity DECIMAL(20, 8) NOT NULL,
  entry_price DECIMAL(20, 8) NOT NULL,
  exit_price DECIMAL(20, 8),
  order_id VARCHAR(100),
  alpaca_order_id VARCHAR(100),
  filled_at TIMESTAMPTZ,
  execution_status VARCHAR(20) DEFAULT 'pending',
  realized_pnl DECIMAL(20, 8),
  unrealized_pnl DECIMAL(20, 8),
  fees DECIMAL(20, 8) DEFAULT 0,
  position_size_percent DECIMAL(5, 2),
  risk_amount DECIMAL(20, 8),
  confidence_score DECIMAL(5, 2),
  risk_reward_ratio DECIMAL(5, 2),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  closed_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_trade_history_user_created ON trade_history(user_id, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_trade_history_symbol ON trade_history(symbol);
CREATE INDEX IF NOT EXISTS idx_trade_history_status ON trade_history(execution_status);
ALTER TABLE trade_history ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view own trade history" ON trade_history
  FOR ALL USING (auth.uid() = user_id);

-- 3. User Settings Table
CREATE TABLE IF NOT EXISTS user_settings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
  trading_start_time TIME DEFAULT '09:00:00',
  trading_end_time TIME DEFAULT '18:00:00',
  trading_timezone VARCHAR(50) DEFAULT 'America/New_York',
  trading_enabled BOOLEAN DEFAULT true,
  per_trade_risk_percent DECIMAL(5, 2) DEFAULT 1.0,
  max_position_size_percent DECIMAL(5, 2) DEFAULT 10.0,
  max_drawdown_percent DECIMAL(5, 2) DEFAULT 15.0,
  volatility_tolerance VARCHAR(10) DEFAULT 'medium',
  confidence_threshold DECIMAL(5, 2) DEFAULT 75.0,
  risk_reward_minimum DECIMAL(5, 2) DEFAULT 3.0,
  shorting_enabled BOOLEAN DEFAULT false,
  margin_enabled BOOLEAN DEFAULT false,
  max_leverage DECIMAL(5, 2) DEFAULT 1.0,
  unorthodox_strategies BOOLEAN DEFAULT false,
  agent_pauses_remaining INTEGER DEFAULT 2,
  agent_pauses_reset_date DATE,
  last_pause_reason TEXT,
  last_pause_at TIMESTAMPTZ,
  email_notifications BOOLEAN DEFAULT true,
  critical_alerts_only BOOLEAN DEFAULT false,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can manage own settings" ON user_settings
  FOR ALL USING (auth.uid() = user_id);

-- 4. Virtual Portfolio Table
CREATE TABLE IF NOT EXISTS virtual_portfolios (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  portfolio_name VARCHAR(100) DEFAULT 'Main Portfolio',
  initial_balance DECIMAL(20, 2) DEFAULT 50000.00,
  current_balance DECIMAL(20, 2) DEFAULT 50000.00,
  total_positions INTEGER DEFAULT 0,
  open_positions JSONB DEFAULT '[]',
  total_realized_pnl DECIMAL(20, 2) DEFAULT 0,
  total_unrealized_pnl DECIMAL(20, 2) DEFAULT 0,
  total_fees_paid DECIMAL(20, 2) DEFAULT 0,
  peak_balance DECIMAL(20, 2) DEFAULT 50000.00,
  max_drawdown DECIMAL(20, 2) DEFAULT 0,
  total_trades INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  losing_trades INTEGER DEFAULT 0,
  win_rate DECIMAL(5, 2),
  average_win DECIMAL(20, 2),
  average_loss DECIMAL(20, 2),
  profit_factor DECIMAL(10, 2),
  sharpe_ratio DECIMAL(10, 4),
  sortino_ratio DECIMAL(10, 4),
  shadow_balance DECIMAL(20, 2) DEFAULT 50000.00,
  shadow_pnl DECIMAL(20, 2) DEFAULT 0,
  user_impact_percent DECIMAL(10, 2) DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  last_reset_at TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS idx_virtual_portfolios_user ON virtual_portfolios(user_id);
ALTER TABLE virtual_portfolios ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can manage own portfolios" ON virtual_portfolios
  FOR ALL USING (auth.uid() = user_id);

-- 5. Daily Performance Table
CREATE TABLE IF NOT EXISTS daily_performance (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  portfolio_id UUID REFERENCES virtual_portfolios(id) ON DELETE CASCADE,
  snapshot_date DATE NOT NULL,
  opening_balance DECIMAL(20, 2),
  closing_balance DECIMAL(20, 2),
  high_balance DECIMAL(20, 2),
  low_balance DECIMAL(20, 2),
  daily_pnl DECIMAL(20, 2),
  daily_pnl_percent DECIMAL(10, 4),
  cumulative_pnl DECIMAL(20, 2),
  trades_count INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  losing_trades INTEGER DEFAULT 0,
  daily_volatility DECIMAL(10, 4),
  max_position_size DECIMAL(20, 2),
  risk_adjusted_return DECIMAL(10, 4),
  created_at TIMESTAMPTZ DEFAULT NOW(),
  UNIQUE(portfolio_id, snapshot_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_performance_date ON daily_performance(portfolio_id, snapshot_date DESC);
ALTER TABLE daily_performance ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Users can view own performance" ON daily_performance
  FOR ALL USING (auth.uid() = user_id);

-- 6. Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_api_keys_updated_at BEFORE UPDATE ON public.api_keys
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_trade_history_updated_at BEFORE UPDATE ON trade_history
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_user_settings_updated_at BEFORE UPDATE ON user_settings
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_virtual_portfolios_updated_at BEFORE UPDATE ON virtual_portfolios
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 7. Initial setup function for new users
CREATE OR REPLACE FUNCTION setup_new_user()
RETURNS TRIGGER AS $$
BEGIN
  -- Create default settings
  INSERT INTO user_settings (user_id)
  VALUES (NEW.id)
  ON CONFLICT (user_id) DO NOTHING;
  
  -- Create default virtual portfolio
  INSERT INTO virtual_portfolios (user_id)
  VALUES (NEW.id);
  
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION setup_new_user();

-- Success message
SELECT 'Database setup completed successfully!' as message;
