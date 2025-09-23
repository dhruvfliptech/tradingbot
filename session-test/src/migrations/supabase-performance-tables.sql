-- Migration for Trading Bot Performance Tracking Tables
-- Run these SQL commands in your Supabase SQL editor

-- 1. Strategy Performance Table
CREATE TABLE IF NOT EXISTS strategy_performance (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  strategy_name TEXT NOT NULL,
  total_trades INTEGER DEFAULT 0,
  winning_trades INTEGER DEFAULT 0,
  losing_trades INTEGER DEFAULT 0,
  total_pnl DECIMAL(15,2) DEFAULT 0,
  win_rate DECIMAL(5,2) DEFAULT 0,
  average_win DECIMAL(15,2) DEFAULT 0,
  average_loss DECIMAL(15,2) DEFAULT 0,
  profit_factor DECIMAL(8,4) DEFAULT 0,
  sharpe_ratio DECIMAL(8,4) DEFAULT 0,
  max_drawdown DECIMAL(15,2) DEFAULT 0,
  last_trade_date TIMESTAMP WITH TIME ZONE,
  confidence_avg DECIMAL(5,2) DEFAULT 0,
  risk_reward_avg DECIMAL(8,4) DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, strategy_name)
);

-- 2. Trade Attribution Table
CREATE TABLE IF NOT EXISTS trade_attribution (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  trade_id TEXT NOT NULL,
  strategy_name TEXT NOT NULL,
  confidence DECIMAL(5,2) NOT NULL,
  expected_outcome TEXT CHECK (expected_outcome IN ('win', 'loss')),
  actual_outcome TEXT CHECK (actual_outcome IN ('win', 'loss', 'pending')),
  accuracy_score DECIMAL(5,2) DEFAULT 0,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, trade_id)
);

-- 3. Performance Snapshots Table
CREATE TABLE IF NOT EXISTS performance_snapshots (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  snapshot_date DATE NOT NULL,
  total_portfolio_value DECIMAL(15,2) NOT NULL,
  strategies_data JSONB DEFAULT '{}',
  market_data JSONB DEFAULT '{}',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, snapshot_date)
);

-- 4. Enhanced Trade History Table (add strategy attribution column)
ALTER TABLE trade_history 
ADD COLUMN IF NOT EXISTS strategy_attribution TEXT DEFAULT 'unknown';

-- Add MAE and MFE columns for advanced metrics
ALTER TABLE trade_history 
ADD COLUMN IF NOT EXISTS mae DECIMAL(15,2), -- Maximum Adverse Excursion
ADD COLUMN IF NOT EXISTS mfe DECIMAL(15,2); -- Maximum Favorable Excursion

-- 5. Position Management Tables
CREATE TABLE IF NOT EXISTS position_rules (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  position_id TEXT NOT NULL,
  rule_type TEXT NOT NULL CHECK (rule_type IN ('take_profit', 'stop_loss', 'trailing_stop', 'time_exit')),
  trigger_price DECIMAL(15,8),
  trigger_percentage DECIMAL(5,2),
  quantity_percentage DECIMAL(5,2) NOT NULL,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS dca_orders (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  target_price DECIMAL(15,8) NOT NULL,
  quantity DECIMAL(15,8) NOT NULL,
  max_entries INTEGER NOT NULL,
  current_entries INTEGER DEFAULT 0,
  price_step_percentage DECIMAL(5,2) NOT NULL,
  is_active BOOLEAN DEFAULT TRUE,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- 6. Correlation Data Table
CREATE TABLE IF NOT EXISTS correlation_data (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  btc_sp500 DECIMAL(6,4),
  btc_gold DECIMAL(6,4),
  btc_dxy DECIMAL(6,4),
  btc_volatility DECIMAL(8,4),
  market_regime TEXT CHECK (market_regime IN ('risk_on', 'risk_off', 'mixed', 'uncertain')),
  position_size_multiplier DECIMAL(4,2),
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, date)
);

-- 7. News Sentiment Data Table
CREATE TABLE IF NOT EXISTS news_sentiment (
  id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  date DATE NOT NULL,
  overall_sentiment INTEGER CHECK (overall_sentiment BETWEEN -100 AND 100),
  confidence INTEGER CHECK (confidence BETWEEN 0 AND 100),
  news_count INTEGER DEFAULT 0,
  regulatory_risk INTEGER CHECK (regulatory_risk BETWEEN 0 AND 100),
  market_sentiment TEXT CHECK (market_sentiment IN ('bullish', 'bearish', 'neutral')),
  key_themes JSONB DEFAULT '[]',
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id, date)
);

-- 8. Indexes for performance
CREATE INDEX IF NOT EXISTS idx_strategy_performance_user_strategy ON strategy_performance(user_id, strategy_name);
CREATE INDEX IF NOT EXISTS idx_trade_attribution_user_trade ON trade_attribution(user_id, trade_id);
CREATE INDEX IF NOT EXISTS idx_trade_attribution_strategy ON trade_attribution(strategy_name);
CREATE INDEX IF NOT EXISTS idx_performance_snapshots_user_date ON performance_snapshots(user_id, snapshot_date);
CREATE INDEX IF NOT EXISTS idx_trade_history_strategy ON trade_history(strategy_attribution);
CREATE INDEX IF NOT EXISTS idx_position_rules_user_position ON position_rules(user_id, position_id);
CREATE INDEX IF NOT EXISTS idx_dca_orders_user_symbol ON dca_orders(user_id, symbol);
CREATE INDEX IF NOT EXISTS idx_correlation_data_user_date ON correlation_data(user_id, date);
CREATE INDEX IF NOT EXISTS idx_news_sentiment_user_date ON news_sentiment(user_id, date);

-- 9. Row Level Security (RLS) Policies
ALTER TABLE strategy_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE trade_attribution ENABLE ROW LEVEL SECURITY;
ALTER TABLE performance_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE position_rules ENABLE ROW LEVEL SECURITY;
ALTER TABLE dca_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE correlation_data ENABLE ROW LEVEL SECURITY;
ALTER TABLE news_sentiment ENABLE ROW LEVEL SECURITY;

-- Create policies for user data isolation
CREATE POLICY "Users can only see their own strategy performance" ON strategy_performance
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own trade attribution" ON trade_attribution
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own performance snapshots" ON performance_snapshots
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own position rules" ON position_rules
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own DCA orders" ON dca_orders
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own correlation data" ON correlation_data
  FOR ALL USING (auth.uid() = user_id);

CREATE POLICY "Users can only see their own news sentiment" ON news_sentiment
  FOR ALL USING (auth.uid() = user_id);

-- 10. Functions for automatic updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers for updated_at columns
CREATE TRIGGER update_strategy_performance_updated_at BEFORE UPDATE ON strategy_performance
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_trade_attribution_updated_at BEFORE UPDATE ON trade_attribution
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_performance_snapshots_updated_at BEFORE UPDATE ON performance_snapshots
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_dca_orders_updated_at BEFORE UPDATE ON dca_orders
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- 11. View for strategy performance summary
CREATE OR REPLACE VIEW strategy_performance_summary AS
SELECT 
  sp.user_id,
  sp.strategy_name,
  sp.total_trades,
  sp.win_rate,
  sp.total_pnl,
  sp.profit_factor,
  sp.sharpe_ratio,
  COALESCE(ta.accuracy, 0) as prediction_accuracy,
  sp.last_trade_date,
  sp.updated_at
FROM strategy_performance sp
LEFT JOIN (
  SELECT 
    user_id,
    strategy_name,
    AVG(accuracy_score) as accuracy
  FROM trade_attribution 
  WHERE actual_outcome != 'pending'
  GROUP BY user_id, strategy_name
) ta ON sp.user_id = ta.user_id AND sp.strategy_name = ta.strategy_name;

-- 12. Function to calculate R-multiples
CREATE OR REPLACE FUNCTION calculate_r_multiple(
  entry_price DECIMAL(15,8),
  exit_price DECIMAL(15,8),
  stop_loss DECIMAL(15,8),
  position_side TEXT
) RETURNS DECIMAL(8,4) AS $$
DECLARE
  risk_amount DECIMAL(15,8);
  actual_pnl DECIMAL(15,8);
BEGIN
  IF stop_loss IS NULL OR stop_loss = 0 THEN
    RETURN NULL;
  END IF;
  
  -- Calculate risk amount (distance to stop loss)
  IF position_side = 'buy' THEN
    risk_amount := entry_price - stop_loss;
    actual_pnl := exit_price - entry_price;
  ELSE
    risk_amount := stop_loss - entry_price;
    actual_pnl := entry_price - exit_price;
  END IF;
  
  -- Return R-multiple
  IF risk_amount > 0 THEN
    RETURN actual_pnl / risk_amount;
  ELSE
    RETURN NULL;
  END IF;
END;
$$ LANGUAGE plpgsql;

-- 13. Sample data insertion function (for testing)
CREATE OR REPLACE FUNCTION insert_sample_performance_data(target_user_id UUID)
RETURNS void AS $$
BEGIN
  -- Insert sample strategy performance
  INSERT INTO strategy_performance (
    user_id, strategy_name, total_trades, winning_trades, losing_trades,
    total_pnl, win_rate, average_win, average_loss, profit_factor
  ) VALUES 
  (target_user_id, 'liquidity', 25, 15, 10, 1250.50, 60.00, 125.75, 75.25, 1.67),
  (target_user_id, 'smartMoney', 30, 20, 10, 1875.25, 66.67, 140.50, 87.75, 1.85),
  (target_user_id, 'volumeProfile', 20, 12, 8, 950.75, 60.00, 115.25, 68.50, 1.52),
  (target_user_id, 'microstructure', 35, 22, 13, 2100.00, 62.86, 135.25, 89.25, 1.76)
  ON CONFLICT (user_id, strategy_name) DO NOTHING;
  
  -- Insert sample performance snapshot for today
  INSERT INTO performance_snapshots (
    user_id, snapshot_date, total_portfolio_value,
    strategies_data, market_data
  ) VALUES (
    target_user_id, 
    CURRENT_DATE, 
    52500.75,
    '{"liquidity": {"pnl": 250.50, "trades": 5}, "smartMoney": {"pnl": 375.25, "trades": 7}}',
    '{"btc_price": 43250.75, "market_trend": "bullish", "volatility": 0.045}'
  ) ON CONFLICT (user_id, snapshot_date) DO NOTHING;
END;
$$ LANGUAGE plpgsql;

-- Usage example:
-- SELECT insert_sample_performance_data(auth.uid());

COMMENT ON TABLE strategy_performance IS 'Tracks performance metrics for each trading strategy';
COMMENT ON TABLE trade_attribution IS 'Links trades to the strategies that generated them';
COMMENT ON TABLE performance_snapshots IS 'Daily portfolio snapshots with strategy breakdown';
COMMENT ON TABLE position_rules IS 'Rules for automated position management';
COMMENT ON TABLE dca_orders IS 'Dollar cost averaging orders';
COMMENT ON TABLE correlation_data IS 'Cross-asset correlation tracking';
COMMENT ON TABLE news_sentiment IS 'News sentiment analysis data';