export interface Position {
  symbol: string;
  qty: string;
  market_value: number;
  cost_basis: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  side: 'long' | 'short';
  name: string;
  image: string;
}

export interface Order {
  id: string;
  symbol: string;
  qty: string;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit';
  status: 'pending' | 'filled' | 'canceled';
  filled_qty: string;
  filled_avg_price: number;
  submitted_at: string;
  filled_at: string | null;
  limit_price: number | null;
}

export interface Account {
  id: string;
  balance_usd: number;
  balance_btc: number;
  portfolio_value: number;
  available_balance: number;
  total_trades: number;
}

export interface CryptoData {
  id: string;
  symbol: string;
  name: string;
  image: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  volume24h?: number;
  high: number;
  low: number;
  high24h?: number;
  low24h?: number;
  market_cap: number;
  market_cap_rank: number;
  price_change_24h: number;
  price_change_percentage_24h: number;
  circulating_supply: number;
  total_supply: number;
  max_supply: number;
  ath: number;
  ath_change_percentage: number;
  last_updated: string;
}

export interface TradingSignal {
  symbol: string;
  name?: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  reason?: string;
  timestamp: string;
  price?: number;
  price_target?: number;
  stop_loss?: number;
  current_price?: number;
  rsi?: number;
  macd?: number;
  ma20?: number;
  ma50?: number;
  volume_indicator?: string;
  trend?: 'Bullish' | 'Bearish' | 'Neutral';
  maIndicator?: 'bullish' | 'bearish' | 'neutral';
  technicalScore?: number;
  validationDetails?: any;
  strategySignals?: {
    liquidity?: any;
    smartMoney?: any;
    volumeProfile?: any;
    microstructure?: any;
  };
}

export interface PerformanceMetrics {
  totalReturn: number;
  totalReturnPercent: number;
  dayReturn: number;
  dayReturnPercent: number;
  totalTrades: number;
  winRate: number;
  avgTradeReturn: number;
  bestTrade: number;
  worstTrade: number;
}

export interface FearGreedIndex {
  value: number;
  value_classification: string;
  timestamp: string;
  time_until_update: string;
}