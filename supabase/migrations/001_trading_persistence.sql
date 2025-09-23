-- Migration: Trading Bot v2.0 Data Persistence Layer
-- Description: Creates tables for trade history, audit logs, user settings, and virtual portfolios

-- 1. Trade History Table
CREATE TABLE IF NOT EXISTS trade_history (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Trade Details
  symbol VARCHAR(20) NOT NULL,
  side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell', 'short', 'cover')),
  quantity DECIMAL(20, 8) NOT NULL,
  entry_price DECIMAL(20, 8) NOT NULL,
  exit_price DECIMAL(20, 8),
  
  -- Execution Details
  order_id VARCHAR(100),
  alpaca_order_id VARCHAR(100),
  filled_at TIMESTAMPTZ,
  execution_status VARCHAR(20) DEFAULT 'pending',
  
  -- P&L Tracking
  realized_pnl DECIMAL(20, 8),
  unrealized_pnl DECIMAL(20, 8),
  fees DECIMAL(20, 8) DEFAULT 0,
  
  -- Risk Metrics at Time of Trade
  position_size_percent DECIMAL(5, 2),
  risk_amount DECIMAL(20, 8),
  confidence_score DECIMAL(5, 2),
  risk_reward_ratio DECIMAL(5, 2),
  
  -- Meta
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  closed_at TIMESTAMPTZ,
  
  -- Indexing
  INDEX idx_trade_history_user_created (user_id, created_at DESC),
  INDEX idx_trade_history_symbol (symbol),
  INDEX idx_trade_history_status (execution_status)
);

-- 2. Audit Logs Table
CREATE TABLE IF NOT EXISTS audit_logs (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Event Details
  event_type VARCHAR(50) NOT NULL, -- 'ai_decision', 'user_override', 'setting_change', 'trade_execution', 'agent_pause'
  event_category VARCHAR(50), -- 'trading', 'risk', 'settings', 'system'
  
  -- Decision Context
  symbol VARCHAR(20),
  action VARCHAR(20), -- 'buy', 'sell', 'hold', 'skip'
  confidence_score DECIMAL(5, 2),
  
  -- AI Reasoning (JSONB for flexibility)
  ai_reasoning JSONB,
  /* Example structure:
  {
    "signals": {
      "technical": { "rsi": 65, "macd": "bullish", "ma_trend": "up" },
      "momentum": { "volume": 1500000, "change_24h": 5.2 },
      "sentiment": { "fear_greed": 72, "whale_activity": "accumulation" }
    },
    "risk_assessment": {
      "volatility": "medium",
      "drawdown_risk": 0.05,
      "position_correlation": 0.3
    },
    "decision_factors": [
      { "factor": "RSI", "weight": 0.2, "contribution": 0.15 },
      { "factor": "Volume", "weight": 0.15, "contribution": 0.12 }
    ]
  }
  */
  
  -- User Context
  old_value JSONB,
  new_value JSONB,
  user_reason TEXT,
  
  -- System Context
  market_conditions JSONB,
  portfolio_state JSONB,
  
  -- Meta
  created_at TIMESTAMPTZ DEFAULT NOW(),
  session_id VARCHAR(100),
  ip_address INET,
  
  -- Indexing
  INDEX idx_audit_logs_user_created (user_id, created_at DESC),
  INDEX idx_audit_logs_event_type (event_type),
  INDEX idx_audit_logs_symbol (symbol)
);

-- 3. User Settings Table
CREATE TABLE IF NOT EXISTS user_settings (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE UNIQUE,
  
  -- Trading Window
  trading_start_time TIME DEFAULT '09:00:00',
  trading_end_time TIME DEFAULT '18:00:00',
  trading_timezone VARCHAR(50) DEFAULT 'America/New_York',
  trading_enabled BOOLEAN DEFAULT true,
  
  -- Risk Management
  per_trade_risk_percent DECIMAL(5, 2) DEFAULT 1.0, -- 1% default
  max_position_size_percent DECIMAL(5, 2) DEFAULT 10.0,
  max_drawdown_percent DECIMAL(5, 2) DEFAULT 15.0,
  volatility_tolerance VARCHAR(10) DEFAULT 'medium', -- 'low', 'medium', 'high'
  confidence_threshold DECIMAL(5, 2) DEFAULT 65.0,
  risk_reward_minimum DECIMAL(5, 2) DEFAULT 3.0, -- 1:3 ratio
  
  -- Strategy Settings
  shorting_enabled BOOLEAN DEFAULT false,
  margin_enabled BOOLEAN DEFAULT false,
  max_leverage DECIMAL(5, 2) DEFAULT 1.0,
  unorthodox_strategies BOOLEAN DEFAULT false,
  
  -- Agent Controls
  agent_pauses_remaining INTEGER DEFAULT 2,
  agent_pauses_reset_date DATE,
  last_pause_reason TEXT,
  last_pause_at TIMESTAMPTZ,
  
  -- Notification Preferences
  email_notifications BOOLEAN DEFAULT true,
  critical_alerts_only BOOLEAN DEFAULT false,
  
  -- Meta
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- 4. Virtual Portfolio Table
CREATE TABLE IF NOT EXISTS virtual_portfolios (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  
  -- Portfolio Details
  portfolio_name VARCHAR(100) DEFAULT 'Main Portfolio',
  initial_balance DECIMAL(20, 2) DEFAULT 50000.00,
  current_balance DECIMAL(20, 2) DEFAULT 50000.00,
  
  -- Position Tracking
  total_positions INTEGER DEFAULT 0,
  open_positions JSONB DEFAULT '[]',
  /* Example structure:
  [
    {
      "symbol": "BTC",
      "quantity": 0.5,
      "entry_price": 65000,
      "current_price": 67000,
      "unrealized_pnl": 1000,
      "opened_at": "2024-01-15T10:30:00Z"
    }
  ]
  */
  
  -- Performance Metrics
  total_realized_pnl DECIMAL(20, 2) DEFAULT 0,
  total_unrealized_pnl DECIMAL(20, 2) DEFAULT 0,
  total_fees_paid DECIMAL(20, 2) DEFAULT 0,
  peak_balance DECIMAL(20, 2) DEFAULT 50000.00,
  max_drawdown DECIMAL(20, 2) DEFAULT 0,
  
  -- Statistics
  total_trades INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  losing_trades INTEGER DEFAULT 0,
  win_rate DECIMAL(5, 2),
  average_win DECIMAL(20, 2),
  average_loss DECIMAL(20, 2),
  profit_factor DECIMAL(10, 2),
  sharpe_ratio DECIMAL(10, 4),
  sortino_ratio DECIMAL(10, 4),
  
  -- Shadow Portfolio (AI-only decisions for comparison)
  shadow_balance DECIMAL(20, 2) DEFAULT 50000.00,
  shadow_pnl DECIMAL(20, 2) DEFAULT 0,
  user_impact_percent DECIMAL(10, 2) DEFAULT 0, -- Difference between actual and shadow
  
  -- Meta
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW(),
  last_reset_at TIMESTAMPTZ,
  
  -- Indexing
  INDEX idx_virtual_portfolios_user (user_id)
);

-- 5. Daily Performance Snapshots
CREATE TABLE IF NOT EXISTS daily_performance (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  portfolio_id UUID REFERENCES virtual_portfolios(id) ON DELETE CASCADE,
  
  -- Date
  snapshot_date DATE NOT NULL,
  
  -- Balances
  opening_balance DECIMAL(20, 2),
  closing_balance DECIMAL(20, 2),
  high_balance DECIMAL(20, 2),
  low_balance DECIMAL(20, 2),
  
  -- P&L
  daily_pnl DECIMAL(20, 2),
  daily_pnl_percent DECIMAL(10, 4),
  cumulative_pnl DECIMAL(20, 2),
  
  -- Trading Activity
  trades_count INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  losing_trades INTEGER DEFAULT 0,
  
  -- Risk Metrics
  daily_volatility DECIMAL(10, 4),
  max_position_size DECIMAL(20, 2),
  risk_adjusted_return DECIMAL(10, 4),
  
  -- Meta
  created_at TIMESTAMPTZ DEFAULT NOW(),
  
  -- Ensure one snapshot per day per portfolio
  UNIQUE(portfolio_id, snapshot_date),
  INDEX idx_daily_performance_date (portfolio_id, snapshot_date DESC)
);

-- 6. Create update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply triggers
CREATE TRIGGER update_trade_history_updated_at BEFORE UPDATE ON trade_history
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_settings_updated_at BEFORE UPDATE ON user_settings
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_virtual_portfolios_updated_at BEFORE UPDATE ON virtual_portfolios
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 7. Row Level Security (RLS)
ALTER TABLE trade_history ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_settings ENABLE ROW LEVEL SECURITY;
ALTER TABLE virtual_portfolios ENABLE ROW LEVEL SECURITY;
ALTER TABLE daily_performance ENABLE ROW LEVEL SECURITY;

-- RLS Policies
CREATE POLICY "Users can view own trade history" ON trade_history
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own audit logs" ON audit_logs
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own settings" ON user_settings
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can manage own portfolios" ON virtual_portfolios
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can view own performance" ON daily_performance
  FOR ALL USING (auth.uid() = user_id);

-- 8. Initial setup function for new users
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

-- Trigger for new user setup
CREATE TRIGGER on_auth_user_created
  AFTER INSERT ON auth.users
  FOR EACH ROW EXECUTE FUNCTION setup_new_user();

-- 9. Indexes for performance
CREATE INDEX idx_trade_history_performance ON trade_history(user_id, created_at DESC) WHERE execution_status = 'completed';
CREATE INDEX idx_audit_logs_recent ON audit_logs(created_at DESC) WHERE created_at > NOW() - INTERVAL '30 days';
CREATE INDEX idx_daily_performance_recent ON daily_performance(snapshot_date DESC) WHERE snapshot_date > CURRENT_DATE - INTERVAL '90 days';

COMMENT ON TABLE trade_history IS 'Complete history of all trades executed by the trading bot';
COMMENT ON TABLE audit_logs IS 'Audit trail for all AI decisions and user interventions';
COMMENT ON TABLE user_settings IS 'User-configurable settings for trading agent behavior';
COMMENT ON TABLE virtual_portfolios IS 'Virtual portfolio tracking with $50K baseline';
COMMENT ON TABLE daily_performance IS 'Daily snapshots for performance tracking and calendar view';