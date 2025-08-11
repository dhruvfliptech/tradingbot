-- Performance metrics storage for trading analytics
-- Authenticated-only access via RLS

CREATE TABLE IF NOT EXISTS performance_metrics (
  id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
  window text NOT NULL DEFAULT 'overall', -- 'overall' | 'daily' | 'weekly' etc.
  window_date date,                      -- used when window is daily/weekly
  total_return_percent double precision DEFAULT 0,
  day_return_percent double precision DEFAULT 0,
  sharpe double precision DEFAULT 0,
  total_trades integer DEFAULT 0,
  win_rate double precision DEFAULT 0,
  avg_trade_return double precision DEFAULT 0,
  best_trade_return double precision DEFAULT 0,
  worst_trade_return double precision DEFAULT 0,
  created_at timestamptz DEFAULT now(),
  updated_at timestamptz DEFAULT now(),
  UNIQUE(user_id, window, window_date)
);

ALTER TABLE performance_metrics ENABLE ROW LEVEL SECURITY;

-- RLS: Authenticated users can manage only their own rows
CREATE POLICY "Users can manage own performance metrics"
  ON performance_metrics
  FOR ALL
  TO authenticated
  USING (auth.uid() = user_id)
  WITH CHECK (auth.uid() = user_id);

-- Trigger to maintain updated_at
CREATE OR REPLACE FUNCTION perf_metrics_set_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = now();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_perf_metrics_updated_at
  BEFORE UPDATE ON performance_metrics
  FOR EACH ROW
  EXECUTE FUNCTION perf_metrics_set_updated_at();


