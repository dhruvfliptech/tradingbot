// Basic models for the trading system

export interface TradingSession {
  id: string;
  userId: string;
  status: 'active' | 'paused' | 'stopped';
  startedAt: Date;
  lastActivity: Date;
  settings: TradingSettings;
  activeTrades: number;
  totalPnL: number;
}

export interface TradingSignal {
  id: string;
  symbol: string;
  action: 'BUY' | 'SELL' | 'HOLD';
  confidence: number;
  price: number;
  timestamp: Date;
  reasoning: string;
  technicalIndicators: Record<string, any>;
  marketConditions: Record<string, any>;
}

export interface Order {
  id: string;
  userId: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  type: 'market' | 'limit' | 'stop';
  status: 'pending' | 'filled' | 'canceled' | 'rejected';
  filledQuantity: number;
  filledPrice: number;
  createdAt: Date;
  updatedAt: Date;
}

export interface TradingSettings {
  confidenceThreshold: number;
  maxPositionSize: number;
  riskPercentage: number;
  cooldownMinutes: number;
  adaptiveThresholdEnabled: boolean;
  maxConcurrentTrades: number;
}

export interface User {
  id: string;
  email: string;
  createdAt: Date;
  updatedAt: Date;
  preferences: Record<string, any>;
}

export interface Portfolio {
  id: string;
  userId: string;
  totalValue: number;
  availableCash: number;
  positions: Position[];
  createdAt: Date;
  updatedAt: Date;
}

export interface Position {
  id: string;
  symbol: string;
  quantity: number;
  averagePrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  createdAt: Date;
}
