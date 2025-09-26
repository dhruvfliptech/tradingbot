/**
 * Trading Engine ML Integration
 * Integrates the Trading Engine with ML/RL services via event bus
 */

import { EventEmitter } from 'events';
import logger from '../../utils/logger';
import { MLEventBus, getMLEventBus } from './MLEventBus';
import {
  MLEventType,
  MLEventFactory,
  MarketDataUpdateEvent,
  TradeExecutedEvent,
  PositionOpenedEvent,
  PositionClosedEvent,
  PerformanceUpdateEvent,
  RLSignalGeneratedEvent,
  ThresholdAdjustedEvent,
  MLPredictionReadyEvent,
  PredictionOutcomeEvent
} from '../../types/ml-events';
import {
  Order,
  Position,
  MarketData,
  TradingSignal,
  TradingSettings
} from '../../types/trading';

interface MLIntegrationConfig {
  enabled: boolean;
  publishMarketData: boolean;
  publishTrades: boolean;
  publishPerformance: boolean;
  subscribeToMLSignals: boolean;
  subscribeToRLSignals: boolean;
  feedbackEnabled: boolean;
}

export class TradingEngineMLIntegration extends EventEmitter {
  private eventBus: MLEventBus;
  private config: MLIntegrationConfig;
  private signalCache: Map<string, RLSignalGeneratedEvent | MLPredictionReadyEvent> = new Map();
  private predictionTracking: Map<string, { prediction: any; executedAt?: Date }> = new Map();
  private performanceBuffer: any[] = [];
  private performanceFlushInterval: NodeJS.Timeout | null = null;

  constructor(config?: Partial<MLIntegrationConfig>) {
    super();

    this.config = {
      enabled: true,
      publishMarketData: true,
      publishTrades: true,
      publishPerformance: true,
      subscribeToMLSignals: true,
      subscribeToRLSignals: true,
      feedbackEnabled: true,
      ...config
    };

    this.eventBus = getMLEventBus();

    if (this.config.enabled) {
      this.initialize();
    }
  }

  /**
   * Initialize ML integration
   */
  private async initialize(): Promise<void> {
    try {
      // Start the event bus
      await this.eventBus.start();

      // Subscribe to ML/RL events
      if (this.config.subscribeToMLSignals) {
        this.subscribeToMLEvents();
      }

      if (this.config.subscribeToRLSignals) {
        this.subscribeToRLEvents();
      }

      // Start performance tracking
      if (this.config.publishPerformance) {
        this.startPerformanceTracking();
      }

      logger.info('TradingEngineMLIntegration initialized successfully');

    } catch (error) {
      logger.error('Failed to initialize TradingEngineMLIntegration:', error);
      throw error;
    }
  }

  /**
   * Subscribe to ML service events
   */
  private subscribeToMLEvents(): void {
    // Listen for threshold adjustments
    this.eventBus.onEvent(MLEventType.THRESHOLD_ADJUSTED, async (event) => {
      const thresholdEvent = event as ThresholdAdjustedEvent;
      logger.info(`Threshold adjusted: ${thresholdEvent.data.parameter} from ${thresholdEvent.data.oldValue} to ${thresholdEvent.data.newValue}`);

      // Emit for Trading Engine to update its settings
      this.emit('thresholdAdjusted', {
        parameter: thresholdEvent.data.parameter,
        value: thresholdEvent.data.newValue,
        confidence: thresholdEvent.data.confidence
      });
    });

    // Listen for ML predictions
    this.eventBus.onEvent(MLEventType.ML_PREDICTION_READY, async (event) => {
      const predictionEvent = event as MLPredictionReadyEvent;

      // Cache the prediction
      this.signalCache.set(predictionEvent.data.predictionId, predictionEvent);

      // Convert to trading signal
      const signal = this.convertMLPredictionToSignal(predictionEvent);

      // Emit for Trading Engine
      this.emit('mlSignalGenerated', signal);
    });

    // Listen for risk limit updates
    this.eventBus.onEvent(MLEventType.RISK_LIMIT_UPDATE, async (event) => {
      logger.info('Risk limit update received:', event);
      this.emit('riskLimitUpdate', event);
    });
  }

  /**
   * Subscribe to RL service events
   */
  private subscribeToRLEvents(): void {
    // Listen for RL signals
    this.eventBus.onEvent(MLEventType.RL_SIGNAL_GENERATED, async (event) => {
      const rlSignal = event as RLSignalGeneratedEvent;

      // Cache the signal
      this.signalCache.set(rlSignal.data.signalId, rlSignal);

      // Track for feedback
      if (this.config.feedbackEnabled) {
        this.predictionTracking.set(rlSignal.data.signalId, {
          prediction: rlSignal.data
        });
      }

      // Convert to trading signal
      const signal = this.convertRLSignalToTradingSignal(rlSignal);

      // Emit for Trading Engine
      this.emit('rlSignalGenerated', signal);

      logger.info(`RL signal generated for ${rlSignal.data.symbol}: ${rlSignal.data.action} with confidence ${rlSignal.data.confidence}`);
    });

    // Listen for confidence updates
    this.eventBus.onEvent(MLEventType.RL_CONFIDENCE_UPDATE, async (event) => {
      logger.info('RL confidence update:', event);
      this.emit('confidenceUpdate', event);
    });
  }

  /**
   * Publish market data update
   */
  async publishMarketData(
    userId: string,
    marketData: MarketData,
    indicators?: any
  ): Promise<void> {
    if (!this.config.publishMarketData) return;

    try {
      const event = MLEventFactory.createMarketDataUpdate(
        userId,
        marketData.symbol,
        {
          price: marketData.price,
          volume: marketData.volume,
          changePercent: marketData.changePercent || 0,
          bid: marketData.bid,
          ask: marketData.ask,
          indicators: indicators || {}
        }
      );

      await this.eventBus.publishEvent(event);

    } catch (error) {
      logger.error('Failed to publish market data:', error);
    }
  }

  /**
   * Publish trade execution
   */
  async publishTradeExecution(
    userId: string,
    order: Order,
    mlSignalId?: string,
    rlSignalId?: string
  ): Promise<void> {
    if (!this.config.publishTrades) return;

    try {
      const event = MLEventFactory.createTradeExecuted(
        userId,
        order.id,
        {
          symbol: order.symbol,
          side: order.side as 'buy' | 'sell',
          quantity: parseFloat(order.qty),
          price: order.filled_avg_price || order.limit_price || 0,
          executedAt: new Date(order.filled_at || order.created_at),
          strategy: order.strategy,
          mlSignalId,
          rlSignalId
        }
      );

      await this.eventBus.publishEvent(event);

      // Track execution for feedback
      if (this.config.feedbackEnabled && (mlSignalId || rlSignalId)) {
        const signalId = mlSignalId || rlSignalId || '';
        const tracking = this.predictionTracking.get(signalId);
        if (tracking) {
          tracking.executedAt = new Date();
        }
      }

    } catch (error) {
      logger.error('Failed to publish trade execution:', error);
    }
  }

  /**
   * Publish position opened
   */
  async publishPositionOpened(
    userId: string,
    position: Position
  ): Promise<void> {
    try {
      const event: PositionOpenedEvent = {
        id: `po_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: MLEventType.POSITION_OPENED,
        timestamp: new Date(),
        userId,
        version: '1.0.0',
        data: {
          positionId: position.id || position.symbol,
          symbol: position.symbol,
          quantity: parseFloat(position.qty),
          entryPrice: position.avg_entry_price || 0,
          stopLoss: position.stop_loss,
          takeProfit: position.take_profit,
          strategy: position.strategy
        }
      };

      await this.eventBus.publishEvent(event);

    } catch (error) {
      logger.error('Failed to publish position opened:', error);
    }
  }

  /**
   * Publish position closed with feedback
   */
  async publishPositionClosed(
    userId: string,
    position: Position,
    exitPrice: number,
    closeReason: 'take_profit' | 'stop_loss' | 'signal' | 'manual' | 'emergency'
  ): Promise<void> {
    try {
      const pnl = (exitPrice - (position.avg_entry_price || 0)) * parseFloat(position.qty);
      const pnlPercent = ((exitPrice - (position.avg_entry_price || 0)) / (position.avg_entry_price || 1)) * 100;

      const event: PositionClosedEvent = {
        id: `pc_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: MLEventType.POSITION_CLOSED,
        timestamp: new Date(),
        userId,
        version: '1.0.0',
        data: {
          positionId: position.id || position.symbol,
          symbol: position.symbol,
          quantity: parseFloat(position.qty),
          entryPrice: position.avg_entry_price || 0,
          exitPrice,
          pnl,
          pnlPercent,
          holdingPeriod: Date.now() - new Date(position.created_at).getTime(),
          closeReason
        }
      };

      await this.eventBus.publishEvent(event);

      // Send prediction outcome feedback
      if (this.config.feedbackEnabled) {
        await this.sendPredictionFeedback(position, pnl);
      }

    } catch (error) {
      logger.error('Failed to publish position closed:', error);
    }
  }

  /**
   * Send prediction outcome feedback
   */
  private async sendPredictionFeedback(
    position: Position,
    actualPnl: number
  ): Promise<void> {
    // Find related predictions
    for (const [signalId, tracking] of this.predictionTracking.entries()) {
      if (tracking.prediction.symbol === position.symbol && tracking.executedAt) {
        const timeDiff = Date.now() - tracking.executedAt.getTime();

        // If executed within last hour, consider it related
        if (timeDiff < 3600000) {
          const event: PredictionOutcomeEvent = {
            id: `pof_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            type: MLEventType.PREDICTION_OUTCOME,
            timestamp: new Date(),
            userId: position.user_id || '',
            version: '1.0.0',
            data: {
              predictionId: signalId,
              signalId: signalId,
              predicted: {
                action: tracking.prediction.action,
                confidence: tracking.prediction.confidence,
                expectedReturn: tracking.prediction.expectedReturn || 0
              },
              actual: {
                return: actualPnl,
                executionPrice: position.avg_entry_price || 0,
                slippage: 0 // Calculate if needed
              },
              accuracy: actualPnl > 0 ? tracking.prediction.confidence : 1 - tracking.prediction.confidence,
              profitLoss: actualPnl
            }
          };

          await this.eventBus.publishEvent(event);

          // Remove from tracking
          this.predictionTracking.delete(signalId);
        }
      }
    }
  }

  /**
   * Publish performance update
   */
  async publishPerformanceUpdate(
    userId: string,
    period: 'hourly' | 'daily' | 'weekly',
    metrics: any
  ): Promise<void> {
    if (!this.config.publishPerformance) return;

    try {
      const event: PerformanceUpdateEvent = {
        id: `pu_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        type: MLEventType.PERFORMANCE_UPDATE,
        timestamp: new Date(),
        userId,
        version: '1.0.0',
        data: {
          period,
          metrics: {
            totalReturn: metrics.totalReturn || 0,
            sharpeRatio: metrics.sharpeRatio || 0,
            winRate: metrics.winRate || 0,
            avgWin: metrics.avgWin || 0,
            avgLoss: metrics.avgLoss || 0,
            maxDrawdown: metrics.maxDrawdown || 0,
            totalTrades: metrics.totalTrades || 0,
            profitableTrades: metrics.profitableTrades || 0
          }
        }
      };

      await this.eventBus.publishEvent(event);

    } catch (error) {
      logger.error('Failed to publish performance update:', error);
    }
  }

  /**
   * Start performance tracking
   */
  private startPerformanceTracking(): void {
    // Flush performance metrics every 5 minutes
    this.performanceFlushInterval = setInterval(async () => {
      if (this.performanceBuffer.length > 0) {
        // Calculate aggregate metrics
        const metrics = this.calculatePerformanceMetrics(this.performanceBuffer);

        // Publish hourly update
        await this.publishPerformanceUpdate('system', 'hourly', metrics);

        // Clear buffer
        this.performanceBuffer = [];
      }
    }, 5 * 60 * 1000); // 5 minutes
  }

  /**
   * Convert ML prediction to trading signal
   */
  private convertMLPredictionToSignal(prediction: MLPredictionReadyEvent): TradingSignal {
    return {
      id: prediction.data.predictionId,
      symbol: prediction.data.symbol,
      action: prediction.data.prediction.action.toUpperCase() as 'BUY' | 'SELL' | 'HOLD',
      confidence: prediction.data.prediction.confidence,
      strategy: 'ml_adaptive',
      reasoning: `ML prediction with ${(prediction.data.prediction.confidence * 100).toFixed(1)}% confidence`,
      riskScore: 1 - prediction.data.prediction.confidence,
      expectedReturn: ((prediction.data.prediction.priceTarget || 0) - 1) * 100,
      timeHorizon: prediction.data.prediction.timeHorizon || 60,
      stopLoss: prediction.data.prediction.stopLoss,
      takeProfit: prediction.data.prediction.priceTarget,
      timestamp: prediction.timestamp
    };
  }

  /**
   * Convert RL signal to trading signal
   */
  private convertRLSignalToTradingSignal(rlSignal: RLSignalGeneratedEvent): TradingSignal {
    return {
      id: rlSignal.data.signalId,
      symbol: rlSignal.data.symbol,
      action: rlSignal.data.action.toUpperCase() as 'BUY' | 'SELL' | 'HOLD',
      confidence: rlSignal.data.confidence,
      strategy: `rl_${rlSignal.data.modelInfo.agentType}`,
      reasoning: rlSignal.data.reasoning,
      riskScore: rlSignal.data.riskScore,
      expectedReturn: rlSignal.data.expectedReturn,
      timeHorizon: 60, // Default 60 minutes
      timestamp: rlSignal.timestamp
    };
  }

  /**
   * Calculate performance metrics
   */
  private calculatePerformanceMetrics(buffer: any[]): any {
    // Implement performance calculation logic
    const trades = buffer.filter(item => item.type === 'trade');
    const wins = trades.filter(t => t.pnl > 0);
    const losses = trades.filter(t => t.pnl < 0);

    return {
      totalReturn: trades.reduce((sum, t) => sum + t.pnl, 0),
      winRate: trades.length > 0 ? wins.length / trades.length : 0,
      avgWin: wins.length > 0 ? wins.reduce((sum, t) => sum + t.pnl, 0) / wins.length : 0,
      avgLoss: losses.length > 0 ? losses.reduce((sum, t) => sum + t.pnl, 0) / losses.length : 0,
      totalTrades: trades.length,
      profitableTrades: wins.length
    };
  }

  /**
   * Get integration status
   */
  getStatus(): any {
    return {
      enabled: this.config.enabled,
      eventBusConnected: this.eventBus !== null,
      signalsCached: this.signalCache.size,
      predictionsTracking: this.predictionTracking.size,
      performanceBufferSize: this.performanceBuffer.length
    };
  }

  /**
   * Stop ML integration
   */
  async stop(): Promise<void> {
    if (this.performanceFlushInterval) {
      clearInterval(this.performanceFlushInterval);
    }

    await this.eventBus.stop();

    logger.info('TradingEngineMLIntegration stopped');
  }
}

// Singleton instance
let mlIntegrationInstance: TradingEngineMLIntegration | null = null;

export function getTradingEngineMLIntegration(): TradingEngineMLIntegration {
  if (!mlIntegrationInstance) {
    mlIntegrationInstance = new TradingEngineMLIntegration();
  }
  return mlIntegrationInstance;
}