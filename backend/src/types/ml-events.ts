/**
 * ML/AI Event Types and Schemas
 * Defines all events for ML/AI pipeline communication
 */

export enum MLEventType {
  // Market Events (Published by Trading Engine)
  MARKET_DATA_UPDATE = 'market.data.update',
  TRADE_EXECUTED = 'trade.executed',
  POSITION_OPENED = 'position.opened',
  POSITION_CLOSED = 'position.closed',
  PERFORMANCE_UPDATE = 'performance.update',

  // ML Service Events (Published by Adaptive Threshold)
  THRESHOLD_ADJUSTED = 'ml.threshold.adjusted',
  RISK_LIMIT_UPDATE = 'ml.risk.limit.update',
  ADAPTATION_TRIGGERED = 'ml.adaptation.triggered',
  ML_PREDICTION_READY = 'ml.prediction.ready',

  // RL Service Events (Published by Decision Server)
  RL_SIGNAL_GENERATED = 'rl.signal.generated',
  RL_CONFIDENCE_UPDATE = 'rl.confidence.update',
  RL_ACTION_RECOMMENDED = 'rl.action.recommended',
  RL_MODEL_UPDATED = 'rl.model.updated',

  // Feedback Events
  PREDICTION_OUTCOME = 'feedback.prediction.outcome',
  PERFORMANCE_FEEDBACK = 'feedback.performance',

  // System Events
  SERVICE_HEALTH_CHECK = 'system.health.check',
  SERVICE_STATUS_UPDATE = 'system.status.update',
  EMERGENCY_STOP = 'system.emergency.stop'
}

export interface BaseEvent {
  id: string;
  type: MLEventType;
  timestamp: Date;
  userId: string;
  version: string; // For schema versioning
  correlationId?: string; // For tracing related events
  metadata?: Record<string, any>;
}

// Market Events
export interface MarketDataUpdateEvent extends BaseEvent {
  type: MLEventType.MARKET_DATA_UPDATE;
  data: {
    symbol: string;
    price: number;
    volume: number;
    changePercent: number;
    bid: number;
    ask: number;
    indicators: {
      rsi?: number;
      macd?: number;
      bollinger?: { upper: number; middle: number; lower: number };
      volume_profile?: any;
    };
  };
}

export interface TradeExecutedEvent extends BaseEvent {
  type: MLEventType.TRADE_EXECUTED;
  data: {
    orderId: string;
    symbol: string;
    side: 'buy' | 'sell';
    quantity: number;
    price: number;
    executedAt: Date;
    strategy?: string;
    mlSignalId?: string;
    rlSignalId?: string;
  };
}

export interface PositionOpenedEvent extends BaseEvent {
  type: MLEventType.POSITION_OPENED;
  data: {
    positionId: string;
    symbol: string;
    quantity: number;
    entryPrice: number;
    stopLoss?: number;
    takeProfit?: number;
    strategy?: string;
  };
}

export interface PositionClosedEvent extends BaseEvent {
  type: MLEventType.POSITION_CLOSED;
  data: {
    positionId: string;
    symbol: string;
    quantity: number;
    entryPrice: number;
    exitPrice: number;
    pnl: number;
    pnlPercent: number;
    holdingPeriod: number; // in seconds
    closeReason: 'take_profit' | 'stop_loss' | 'signal' | 'manual' | 'emergency';
  };
}

export interface PerformanceUpdateEvent extends BaseEvent {
  type: MLEventType.PERFORMANCE_UPDATE;
  data: {
    period: 'hourly' | 'daily' | 'weekly';
    metrics: {
      totalReturn: number;
      sharpeRatio: number;
      winRate: number;
      avgWin: number;
      avgLoss: number;
      maxDrawdown: number;
      totalTrades: number;
      profitableTrades: number;
    };
  };
}

// ML Service Events
export interface ThresholdAdjustedEvent extends BaseEvent {
  type: MLEventType.THRESHOLD_ADJUSTED;
  data: {
    parameter: string;
    oldValue: number;
    newValue: number;
    reason: string;
    confidence: number;
    affectedStrategies: string[];
  };
}

export interface RiskLimitUpdateEvent extends BaseEvent {
  type: MLEventType.RISK_LIMIT_UPDATE;
  data: {
    limitType: 'position_size' | 'max_positions' | 'daily_loss' | 'exposure';
    oldLimit: number;
    newLimit: number;
    reason: string;
    marketCondition: 'volatile' | 'trending' | 'ranging' | 'uncertain';
  };
}

export interface MLPredictionReadyEvent extends BaseEvent {
  type: MLEventType.ML_PREDICTION_READY;
  data: {
    predictionId: string;
    symbol: string;
    prediction: {
      action: 'buy' | 'sell' | 'hold';
      confidence: number;
      priceTarget?: number;
      stopLoss?: number;
      timeHorizon: number; // minutes
    };
    features: Record<string, number>;
    modelVersion: string;
  };
}

// RL Service Events
export interface RLSignalGeneratedEvent extends BaseEvent {
  type: MLEventType.RL_SIGNAL_GENERATED;
  data: {
    signalId: string;
    symbol: string;
    action: 'buy' | 'sell' | 'hold';
    confidence: number;
    reasoning: string;
    expectedReturn: number;
    riskScore: number;
    modelInfo: {
      agentType: string;
      version: string;
      trainingEpisodes: number;
    };
  };
}

export interface RLConfidenceUpdateEvent extends BaseEvent {
  type: MLEventType.RL_CONFIDENCE_UPDATE;
  data: {
    symbol: string;
    previousConfidence: number;
    currentConfidence: number;
    factors: {
      marketRegime: number;
      recentPerformance: number;
      volatility: number;
      modelUncertainty: number;
    };
  };
}

// Feedback Events
export interface PredictionOutcomeEvent extends BaseEvent {
  type: MLEventType.PREDICTION_OUTCOME;
  data: {
    predictionId: string;
    signalId: string;
    predicted: {
      action: string;
      confidence: number;
      expectedReturn: number;
    };
    actual: {
      return: number;
      executionPrice: number;
      slippage: number;
    };
    accuracy: number;
    profitLoss: number;
  };
}

// Type Guards
export function isMarketDataUpdate(event: BaseEvent): event is MarketDataUpdateEvent {
  return event.type === MLEventType.MARKET_DATA_UPDATE;
}

export function isTradeExecuted(event: BaseEvent): event is TradeExecutedEvent {
  return event.type === MLEventType.TRADE_EXECUTED;
}

export function isThresholdAdjusted(event: BaseEvent): event is ThresholdAdjustedEvent {
  return event.type === MLEventType.THRESHOLD_ADJUSTED;
}

export function isRLSignalGenerated(event: BaseEvent): event is RLSignalGeneratedEvent {
  return event.type === MLEventType.RL_SIGNAL_GENERATED;
}

// Event Factory
export class MLEventFactory {
  static createMarketDataUpdate(
    userId: string,
    symbol: string,
    data: Partial<MarketDataUpdateEvent['data']>
  ): MarketDataUpdateEvent {
    return {
      id: `mdu_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: MLEventType.MARKET_DATA_UPDATE,
      timestamp: new Date(),
      userId,
      version: '1.0.0',
      data: {
        symbol,
        price: 0,
        volume: 0,
        changePercent: 0,
        bid: 0,
        ask: 0,
        indicators: {},
        ...data
      }
    };
  }

  static createTradeExecuted(
    userId: string,
    orderId: string,
    data: Partial<TradeExecutedEvent['data']>
  ): TradeExecutedEvent {
    return {
      id: `te_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: MLEventType.TRADE_EXECUTED,
      timestamp: new Date(),
      userId,
      version: '1.0.0',
      data: {
        orderId,
        symbol: '',
        side: 'buy',
        quantity: 0,
        price: 0,
        executedAt: new Date(),
        ...data
      }
    };
  }

  static createRLSignal(
    userId: string,
    symbol: string,
    data: Partial<RLSignalGeneratedEvent['data']>
  ): RLSignalGeneratedEvent {
    return {
      id: `rls_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      type: MLEventType.RL_SIGNAL_GENERATED,
      timestamp: new Date(),
      userId,
      version: '1.0.0',
      data: {
        signalId: `sig_${Date.now()}`,
        symbol,
        action: 'hold',
        confidence: 0,
        reasoning: '',
        expectedReturn: 0,
        riskScore: 0,
        modelInfo: {
          agentType: 'ensemble',
          version: '1.0.0',
          trainingEpisodes: 0
        },
        ...data
      }
    };
  }
}

export type MLEvent =
  | MarketDataUpdateEvent
  | TradeExecutedEvent
  | PositionOpenedEvent
  | PositionClosedEvent
  | PerformanceUpdateEvent
  | ThresholdAdjustedEvent
  | RiskLimitUpdateEvent
  | MLPredictionReadyEvent
  | RLSignalGeneratedEvent
  | RLConfidenceUpdateEvent
  | PredictionOutcomeEvent;