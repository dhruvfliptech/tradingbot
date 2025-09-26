/**
 * AutoTradingService - Backend implementation of automated trading bot
 * Handles all trading logic, strategy execution, and order placement
 * Replaces frontend tradingAgentV2.ts
 */

import { EventEmitter } from 'events';
import logger from '../../utils/logger';
import { StateManager } from './StateManager';
import { MarketDataService } from './MarketDataService';
import { SignalProcessor } from './SignalProcessor';
import { RiskManager } from './RiskManager';
import { OrderExecutor } from './OrderExecutor';
import { PositionManager } from './PositionManager';
import { getTradingEngineMLIntegration } from '../ml/TradingEngineMLIntegration';
import {
  TradingSignal,
  MarketData,
  Order,
  Position,
  TradingSettings,
  Account
} from '../../types/trading';

export interface AutoTradingConfig {
  userId: string;
  settings: TradingSettings;
  watchlist: string[];
  cycleIntervalMs: number;
  cooldownMinutes: number;
  maxOpenPositions: number;
  riskBudgetUsd: number;
  confidenceThreshold: number;
}

export interface TradingEvent {
  type: 'status' | 'analysis' | 'decision' | 'order_submitted' | 'order_error' | 'cycle_complete' | 'error';
  data: any;
  timestamp: Date;
  userId: string;
}

interface TradingState {
  isActive: boolean;
  lastCycleTime: number;
  cyclesExecuted: number;
  signalsGenerated: number;
  tradesExecuted: number;
  errors: number;
  lastError?: string;
  positions: Position[];
  account?: Account;
}

export class AutoTradingService extends EventEmitter {
  private config: AutoTradingConfig;
  private state: TradingState;
  private cycleTimer: NodeJS.Timeout | null = null;
  private lastTradeBySymbol: Map<string, number> = new Map();

  // Service dependencies
  private stateManager: StateManager;
  private marketDataService: MarketDataService;
  private signalProcessor: SignalProcessor;
  private riskManager: RiskManager;
  private orderExecutor: OrderExecutor;
  private positionManager: PositionManager;
  private mlIntegration: any;

  constructor(config: AutoTradingConfig) {
    super();
    this.config = config;

    this.state = {
      isActive: false,
      lastCycleTime: 0,
      cyclesExecuted: 0,
      signalsGenerated: 0,
      tradesExecuted: 0,
      errors: 0,
      positions: []
    };

    // Initialize services
    this.stateManager = new StateManager();
    this.marketDataService = new MarketDataService();
    this.signalProcessor = new SignalProcessor();
    this.riskManager = new RiskManager();
    this.orderExecutor = new OrderExecutor();
    this.positionManager = new PositionManager();

    // Initialize ML integration if available
    try {
      this.mlIntegration = getTradingEngineMLIntegration();
    } catch (error) {
      logger.warn('ML integration not available:', error);
    }
  }

  /**
   * Start the auto-trading bot
   */
  async start(): Promise<void> {
    if (this.state.isActive) {
      logger.warn(`AutoTrading already active for user ${this.config.userId}`);
      return;
    }

    try {
      logger.info(`Starting AutoTrading for user ${this.config.userId}`);

      // Load account info
      this.state.account = await this.orderExecutor.getAccount(this.config.userId);

      // Load current positions
      this.state.positions = await this.positionManager.getPositions(this.config.userId);

      // Set active state
      this.state.isActive = true;

      // Emit status event
      this.emitEvent('status', {
        active: true,
        account: this.state.account,
        positions: this.state.positions.length
      });

      // Start trading cycle
      await this.runTradingCycle();

      // Schedule recurring cycles
      this.cycleTimer = setInterval(
        () => this.runTradingCycle(),
        this.config.cycleIntervalMs
      );

      logger.info(`AutoTrading started successfully for user ${this.config.userId}`);

    } catch (error) {
      logger.error(`Failed to start AutoTrading for user ${this.config.userId}:`, error);
      this.state.isActive = false;
      throw error;
    }
  }

  /**
   * Stop the auto-trading bot
   */
  async stop(): Promise<void> {
    if (!this.state.isActive) {
      logger.warn(`AutoTrading already stopped for user ${this.config.userId}`);
      return;
    }

    logger.info(`Stopping AutoTrading for user ${this.config.userId}`);

    // Clear cycle timer
    if (this.cycleTimer) {
      clearInterval(this.cycleTimer);
      this.cycleTimer = null;
    }

    // Set inactive state
    this.state.isActive = false;

    // Emit status event
    this.emitEvent('status', {
      active: false,
      cyclesExecuted: this.state.cyclesExecuted,
      tradesExecuted: this.state.tradesExecuted
    });

    logger.info(`AutoTrading stopped for user ${this.config.userId}`);
  }

  /**
   * Run a single trading cycle
   */
  private async runTradingCycle(): Promise<void> {
    if (!this.state.isActive) return;

    const cycleStart = Date.now();

    try {
      logger.debug(`Starting trading cycle for user ${this.config.userId}`);

      // 1. Update account and positions
      const [account, positions] = await Promise.all([
        this.orderExecutor.getAccount(this.config.userId),
        this.positionManager.getPositions(this.config.userId)
      ]);

      this.state.account = account;
      this.state.positions = positions;

      // 2. Get market data for watchlist
      const marketData = await this.marketDataService.getMarketData(this.config.watchlist);

      // Publish market data to ML pipeline
      if (this.mlIntegration) {
        for (const data of marketData) {
          await this.mlIntegration.publishMarketData(
            this.config.userId,
            data
          );
        }
      }

      // 3. Generate trading signals
      const marketRegime = await this.getMarketRegime();
      const signals = await this.signalProcessor.generateSignals(
        marketData,
        this.config.settings,
        marketRegime
      );

      this.state.signalsGenerated += signals.length;

      // Emit analysis event
      this.emitEvent('analysis', {
        evaluated: marketData.length,
        signals: signals.length,
        topSignals: signals.slice(0, 3).map(s => ({
          symbol: s.symbol,
          action: s.action,
          confidence: s.confidence
        }))
      });

      // 4. Process signals and execute trades
      for (const signal of signals) {
        await this.processSignal(signal, marketData);
      }

      // 5. Update positions (check stop losses, take profits)
      await this.updatePositions();

      // Update cycle metrics
      this.state.cyclesExecuted++;
      this.state.lastCycleTime = Date.now() - cycleStart;

      // Emit cycle complete event
      this.emitEvent('cycle_complete', {
        duration: this.state.lastCycleTime,
        signalsGenerated: signals.length,
        cyclesExecuted: this.state.cyclesExecuted
      });

      logger.debug(`Trading cycle completed in ${this.state.lastCycleTime}ms`);

    } catch (error) {
      logger.error(`Trading cycle error for user ${this.config.userId}:`, error);
      this.state.errors++;
      this.state.lastError = error instanceof Error ? error.message : 'Unknown error';

      this.emitEvent('error', {
        message: this.state.lastError,
        cycle: this.state.cyclesExecuted
      });
    }
  }

  /**
   * Process a trading signal
   */
  private async processSignal(signal: TradingSignal, marketData: MarketData[]): Promise<void> {
    // Check confidence threshold
    if (signal.confidence < this.config.confidenceThreshold) {
      logger.debug(`Signal confidence ${signal.confidence} below threshold ${this.config.confidenceThreshold}`);
      return;
    }

    // Check if we should execute (cooldown, position limits, etc.)
    if (!this.shouldExecuteTrade(signal)) {
      return;
    }

    // Find market data for symbol
    const symbolData = marketData.find(m => m.symbol === signal.symbol);
    if (!symbolData) {
      logger.warn(`No market data for signal symbol ${signal.symbol}`);
      return;
    }

    // Run risk checks
    const riskApproved = await this.riskManager.validateTrade({
      signal,
      account: this.state.account!,
      positions: this.state.positions,
      settings: this.config.settings
    });

    if (!riskApproved) {
      logger.info(`Risk check failed for ${signal.symbol}`);
      this.emitEvent('decision', {
        symbol: signal.symbol,
        action: 'REJECTED',
        reason: 'Risk check failed',
        confidence: signal.confidence
      });
      return;
    }

    // Execute the trade
    await this.executeTrade(signal, symbolData);
  }

  /**
   * Check if we should execute a trade
   */
  private shouldExecuteTrade(signal: TradingSignal): boolean {
    // Check cooldown
    const lastTrade = this.lastTradeBySymbol.get(signal.symbol);
    if (lastTrade) {
      const cooldownMs = this.config.cooldownMinutes * 60 * 1000;
      if (Date.now() - lastTrade < cooldownMs) {
        logger.debug(`Cooldown active for ${signal.symbol}`);
        return false;
      }
    }

    // Check if we already have a position in this symbol
    const hasPosition = this.state.positions.some(
      p => p.symbol.toLowerCase() === signal.symbol.toLowerCase()
    );
    if (hasPosition) {
      logger.debug(`Already have position in ${signal.symbol}`);
      return false;
    }

    // Check max open positions
    if (this.state.positions.length >= this.config.maxOpenPositions) {
      logger.debug(`Max open positions reached: ${this.state.positions.length}/${this.config.maxOpenPositions}`);
      return false;
    }

    return true;
  }

  /**
   * Execute a trade based on signal
   */
  private async executeTrade(signal: TradingSignal, marketData: MarketData): Promise<void> {
    try {
      // Calculate position size
      const accountValue = this.state.account?.equity || 50000;
      const positionValue = Math.min(this.config.riskBudgetUsd, accountValue * 0.1);
      const quantity = Math.floor((positionValue / marketData.price) * 100) / 100;

      if (quantity < 0.01) {
        logger.debug(`Position size too small for ${signal.symbol}: ${quantity}`);
        return;
      }

      // Emit decision event
      this.emitEvent('decision', {
        symbol: signal.symbol,
        action: signal.action,
        confidence: signal.confidence,
        price: marketData.price,
        quantity,
        reason: signal.reasoning || 'Signal generated'
      });

      // Place order
      const order = await this.orderExecutor.executeOrder({
        userId: this.config.userId,
        symbol: signal.symbol,
        side: signal.action.toLowerCase() as 'buy' | 'sell',
        quantity,
        type: 'market',
        metadata: {
          signalId: signal.id,
          confidence: signal.confidence,
          strategy: signal.strategy
        }
      });

      // Update last trade time
      this.lastTradeBySymbol.set(signal.symbol, Date.now());
      this.state.tradesExecuted++;

      // Track with ML integration
      if (this.mlIntegration) {
        await this.mlIntegration.publishTradeExecution(
          this.config.userId,
          order,
          undefined,
          signal.id
        );
      }

      // Emit order submitted event
      this.emitEvent('order_submitted', {
        symbol: signal.symbol,
        side: signal.action.toLowerCase(),
        quantity,
        orderId: order.id,
        price: order.filled_avg_price || marketData.price
      });

      logger.info(`Trade executed for ${signal.symbol}: ${quantity} @ ${marketData.price}`);

    } catch (error) {
      logger.error(`Failed to execute trade for ${signal.symbol}:`, error);

      this.emitEvent('order_error', {
        symbol: signal.symbol,
        message: error instanceof Error ? error.message : 'Trade execution failed'
      });

      throw error;
    }
  }

  /**
   * Update existing positions (check stop loss, take profit, etc.)
   */
  private async updatePositions(): Promise<void> {
    for (const position of this.state.positions) {
      try {
        // Check if position needs management (stop loss, take profit, trailing stop)
        const action = await this.positionManager.evaluatePosition(
          position,
          this.config.settings
        );

        if (action) {
          logger.info(`Position action for ${position.symbol}: ${action.type}`);

          // Execute position action (close, partial close, adjust stops)
          await this.positionManager.executeAction(
            this.config.userId,
            position,
            action
          );
        }
      } catch (error) {
        logger.error(`Error updating position ${position.symbol}:`, error);
      }
    }
  }

  /**
   * Get current market regime
   */
  private async getMarketRegime(): Promise<any> {
    // Simple market regime detection
    // In production, this would use more sophisticated analysis
    return {
      trend: 'neutral',
      volatility: 'normal',
      volume: 'average'
    };
  }

  /**
   * Emit trading event
   */
  private emitEvent(type: string, data: any): void {
    const event: TradingEvent = {
      type: type as any,
      data,
      timestamp: new Date(),
      userId: this.config.userId
    };

    this.emit('tradingEvent', event);

    // Also emit specific event type
    this.emit(type, data);
  }

  /**
   * Get current trading status
   */
  getStatus(): TradingState & { config: AutoTradingConfig } {
    return {
      ...this.state,
      config: this.config
    };
  }

  /**
   * Update trading configuration
   */
  updateConfig(updates: Partial<AutoTradingConfig>): void {
    this.config = {
      ...this.config,
      ...updates
    };

    logger.info(`Trading config updated for user ${this.config.userId}`);
  }

  /**
   * Emergency stop - immediately halt all trading
   */
  async emergencyStop(): Promise<void> {
    logger.warn(`EMERGENCY STOP triggered for user ${this.config.userId}`);

    // Stop trading
    await this.stop();

    // Close all positions if configured
    if (this.config.settings.emergencyCloseAll) {
      for (const position of this.state.positions) {
        try {
          await this.positionManager.closePosition(
            this.config.userId,
            position.id || position.symbol,
            'emergency'
          );
        } catch (error) {
          logger.error(`Failed to close position ${position.symbol} during emergency stop:`, error);
        }
      }
    }

    this.emitEvent('emergency_stop', {
      reason: 'Manual emergency stop',
      positionsClosed: this.config.settings.emergencyCloseAll
    });
  }
}