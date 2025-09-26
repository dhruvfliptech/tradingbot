// Trading System Type Definitions

export interface TradingSignal {
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  reasoning: string;
  timestamp: Date;
  metadata?: {
    signalCount?: number;
    strategies?: string[];
    votes?: Record<string, number>;
    indicators?: any;
  };
}

export interface Order {
  id: string;
  userId: string;
  symbol: string;
  side: 'buy' | 'sell';
  type: 'market' | 'limit' | 'stop' | 'stop_limit';
  quantity: number;
  limitPrice?: number;
  stopPrice?: number;
  status: 'pending' | 'filled' | 'partially_filled' | 'failed' | 'cancelled';
  createdAt: Date;
  filledAt?: Date;
  cancelledAt?: Date;
  error?: string;
  signal?: TradingSignal;
  positionId?: string;
  timeInForce?: 'GTC' | 'IOC' | 'FOK';
  reduceOnly?: boolean;
  postOnly?: boolean;
}

export interface ExecutedOrder extends Order {
  price: number;
  filledQuantity: number;
  commission: number;
  commissionAsset: string;
  executionTime: number;
  avgPrice?: number;
  trades?: Trade[];
}

export interface Trade {
  id: string;
  orderId: string;
  price: number;
  quantity: number;
  commission: number;
  commissionAsset: string;
  time: Date;
  isMaker: boolean;
}

export interface Position {
  id: string;
  userId: string;
  symbol: string;
  side: 'long' | 'short';
  entryPrice: number;
  quantity: number;
  currentPrice: number;
  markPrice?: number;
  unrealizedPnL: number;
  realizedPnL: number;
  status: 'open' | 'closing' | 'closed';
  openedAt: Date;
  closedAt?: Date;
  closePrice?: number;
  stopLoss?: number;
  takeProfit?: number;
  trailingStop?: number;
  trailingStopDistance?: number;
  highWaterMark?: number;
  commission?: number;
  orderId: string;
  leverage?: number;
  marginType?: 'isolated' | 'cross';
  liquidationPrice?: number;
}

export interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  changePercent?: number;
  changeValue?: number;
  marketCap?: number;
  bid: number;
  ask: number;
  bidSize?: number;
  askSize?: number;
  high24h?: number;
  low24h?: number;
  open24h?: number;
  volatility?: number;
  timestamp?: Date;
}

export interface OrderBook {
  symbol: string;
  bids: PriceLevel[];
  asks: PriceLevel[];
  timestamp: Date;
}

export interface PriceLevel {
  price: number;
  quantity: number;
  orders?: number;
}

export interface TradingSettings {
  // Position Management
  maxPositions: number;
  maxPositionSize: number; // As percentage of portfolio
  minPositionSize?: number;
  defaultPositionSize?: number;

  // Risk Management
  stopLossPercent: number;
  takeProfitPercent: number;
  trailingStopPercent?: number;
  maxDrawdown: number;
  maxDailyLoss?: number;
  maxConsecutiveLosses?: number;

  // Trading Parameters
  minConfidence: number;
  orderType: 'market' | 'limit';
  slippage?: number;
  limitOrderOffset?: number;

  // Strategy Settings
  watchlist: string[];
  enabledStrategies: string[];
  strategyWeights?: Record<string, number>;

  // Validator Settings
  validatorEnabled: boolean;
  validatorThreshold?: number;
  strategyWeightBalance: number; // 0-1, balance between strategies and validators

  // Performance Targets
  weeklyTarget: number;
  monthlyTarget?: number;
  profitTarget?: number;

  // Trading Hours
  tradingHours?: {
    enabled: boolean;
    startHour: number;
    endHour: number;
    timezone: string;
  };

  // Advanced Settings
  compoundProfits?: boolean;
  rebalanceFrequency?: 'daily' | 'weekly' | 'monthly' | 'never';
  correlationLimit?: number;
  newsTrading?: boolean;
  weekendTrading?: boolean;
}

export interface PerformanceMetrics {
  period: string;
  totalPnL: number;
  totalPnLPercent: number;
  winRate: number;
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
  profitFactor: number;
  sharpeRatio: number;
  sortinoRatio?: number;
  calmarRatio?: number;
  maxDrawdown: number;
  maxDrawdownDuration?: number;
  openPositions: number;
  closedPositions: number;
  bestTrade: number;
  worstTrade: number;
  avgHoldingTime?: number;
  expectancy?: number;
  kellyFraction?: number;
}

export interface RiskMetrics {
  portfolioValue: number;
  currentExposure: number;
  exposureRatio: number;
  openPositions: number;
  marketVolatility: number;
  dailyPnL: number;
  consecutiveLosses: number;
  riskScore: number; // 0-100
  var95?: number; // Value at Risk 95%
  var99?: number; // Value at Risk 99%
  stressTestResult?: number;
}

export interface AccountInfo {
  userId: string;
  balance: number;
  equity: number;
  marginUsed: number;
  marginFree: number;
  unrealizedPnL: number;
  realizedPnL: number;
  totalAssets?: Record<string, number>;
  updateTime: Date;
}

export interface BacktestResult {
  id: string;
  strategy: string;
  period: {
    start: Date;
    end: Date;
  };
  settings: TradingSettings;
  performance: PerformanceMetrics;
  trades: ExecutedOrder[];
  equity: number[];
  drawdown: number[];
  signals: TradingSignal[];
}

export interface Alert {
  id: string;
  userId: string;
  type: 'trade' | 'risk' | 'system' | 'performance' | 'error';
  severity: 'info' | 'warning' | 'critical';
  title: string;
  message: string;
  data?: any;
  timestamp: Date;
  acknowledged?: boolean;
}

export interface AuditLog {
  id: string;
  userId: string;
  eventType: string;
  action: string;
  details: any;
  ipAddress?: string;
  userAgent?: string;
  timestamp: Date;
}

export interface MarketRegime {
  type: 'bull' | 'bear' | 'sideways' | 'high_volatility' | 'mean_reverting' | 'momentum';
  confidence: number;
  indicators: {
    trend?: 'up' | 'down' | 'neutral';
    volatility?: 'low' | 'medium' | 'high';
    volume?: 'low' | 'normal' | 'high';
    sentiment?: 'bearish' | 'neutral' | 'bullish';
  };
  updatedAt: Date;
}

export interface StrategyPerformance {
  strategyName: string;
  signals: number;
  trades: number;
  winRate: number;
  avgReturn: number;
  sharpeRatio: number;
  contribution: number; // Contribution to total PnL
  lastSignal?: Date;
  isActive: boolean;
}

// Enums for better type safety
export enum OrderStatus {
  PENDING = 'pending',
  FILLED = 'filled',
  PARTIALLY_FILLED = 'partially_filled',
  FAILED = 'failed',
  CANCELLED = 'cancelled'
}

export enum PositionStatus {
  OPEN = 'open',
  CLOSING = 'closing',
  CLOSED = 'closed'
}

export enum MarketTrend {
  BULLISH = 'bullish',
  BEARISH = 'bearish',
  NEUTRAL = 'neutral'
}

export enum RiskLevel {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

// WebSocket Event Types
export interface WSEvent {
  type: string;
  data: any;
  timestamp: Date;
}

export interface WSOrderEvent extends WSEvent {
  type: 'order_update';
  data: {
    order: Order;
    action: 'created' | 'filled' | 'cancelled' | 'failed';
  };
}

export interface WSPositionEvent extends WSEvent {
  type: 'position_update';
  data: {
    position: Position;
    action: 'opened' | 'updated' | 'closed';
  };
}

export interface WSMarketEvent extends WSEvent {
  type: 'market_update';
  data: {
    symbol: string;
    price: number;
    volume: number;
    timestamp: Date;
  };
}

export interface WSAlertEvent extends WSEvent {
  type: 'alert';
  data: Alert;
}

// API Response Types
export interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: Date;
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number;
    pageSize: number;
    totalItems: number;
    totalPages: number;
    hasNext: boolean;
    hasPrevious: boolean;
  };
}

// Validation Types
export interface ValidationResult {
  isValid: boolean;
  errors?: ValidationError[];
}

export interface ValidationError {
  field: string;
  message: string;
  code?: string;
}

// Export all types as a namespace for easier importing
export namespace TradingTypes {
  export type Signal = TradingSignal;
  export type OrderType = Order;
  export type PositionType = Position;
  export type Settings = TradingSettings;
  export type Market = MarketData;
  export type Performance = PerformanceMetrics;
  export type Risk = RiskMetrics;
  export type Account = AccountInfo;
}