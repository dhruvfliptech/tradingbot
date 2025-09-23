import { EventEmitter } from 'events';
import { TradingSession, TradingSignal, Order } from '../../models';
import { MarketDataService } from '../market-data/MarketDataService';
import { AnalyticsService } from '../analytics/AnalyticsService';
import { RiskManager } from './RiskManager';
import { OrderManager } from './OrderManager';
import { WebSocketServer } from '../../websocket/WebSocketServer';
import { TradingQueue } from '../../queues/TradingQueue';
import logger from '../../utils/logger';

export interface TradingSettings {
  confidenceThreshold: number;
  maxPositionSize: number;
  riskPercentage: number;
  cooldownMinutes: number;
  adaptiveThresholdEnabled: boolean;
  maxConcurrentTrades: number;
}

export interface TradingStatus {
  active: boolean;
  uptime: number;
  lastTrade?: string;
  activePairs: number;
  pendingOrders: number;
  totalTrades: number;
  totalPnl: number;
}

export class TradingService extends EventEmitter {
  private activeSessions: Map<string, TradingSession> = new Map();
  private marketDataService: MarketDataService;
  private analyticsService: AnalyticsService;
  private riskManager: RiskManager;
  private orderManager: OrderManager;
  private webSocketServer: WebSocketServer;
  private tradingQueue: TradingQueue;
  private tickIntervals: Map<string, NodeJS.Timeout> = new Map();

  constructor() {
    super();
    this.marketDataService = new MarketDataService();
    this.analyticsService = new AnalyticsService();
    this.riskManager = new RiskManager();
    this.orderManager = new OrderManager();
    this.webSocketServer = WebSocketServer.getInstance();
    this.tradingQueue = new TradingQueue();
  }

  async start(userId: string, watchlist: string[], settings: TradingSettings): Promise<void> {
    try {
      logger.info(`Starting trading session for user ${userId}`);

      // Check if user already has an active session
      if (this.activeSessions.has(userId)) {
        throw new Error('Trading session already active for this user');
      }

      // Create new trading session
      const session = await TradingSession.create({
        userId,
        status: 'running',
        startedAt: new Date(),
        watchlist,
        settings,
        totalTrades: 0,
        totalPnl: 0
      });

      this.activeSessions.set(userId, session);

      // Start trading cycle
      await this.startTradingCycle(userId, session);

      // Emit status update
      this.webSocketServer.emitToUser(userId, 'bot_status_changed', {
        active: true,
        reason: 'Started by user',
        timestamp: new Date().toISOString()
      });

      logger.info(`Trading session started for user ${userId}`);

    } catch (error) {
      logger.error(`Error starting trading session for user ${userId}:`, error);
      throw error;
    }
  }

  async stop(userId: string): Promise<void> {
    try {
      logger.info(`Stopping trading session for user ${userId}`);

      const session = this.activeSessions.get(userId);
      if (!session) {
        throw new Error('No active trading session found');
      }

      // Clear trading cycle interval
      const interval = this.tickIntervals.get(userId);
      if (interval) {
        clearInterval(interval);
        this.tickIntervals.delete(userId);
      }

      // Update session status
      await session.update({
        status: 'stopped',
        stoppedAt: new Date()
      });

      // Remove from active sessions
      this.activeSessions.delete(userId);

      // Emit status update
      this.webSocketServer.emitToUser(userId, 'bot_status_changed', {
        active: false,
        reason: 'Stopped by user',
        timestamp: new Date().toISOString()
      });

      logger.info(`Trading session stopped for user ${userId}`);

    } catch (error) {
      logger.error(`Error stopping trading session for user ${userId}:`, error);
      throw error;
    }
  }

  async getStatus(userId: string): Promise<TradingStatus> {
    try {
      const session = this.activeSessions.get(userId);
      
      if (!session) {
        return {
          active: false,
          uptime: 0,
          activePairs: 0,
          pendingOrders: 0,
          totalTrades: 0,
          totalPnl: 0
        };
      }

      const uptime = session.startedAt ? 
        Date.now() - session.startedAt.getTime() : 0;

      // Get pending orders count
      const pendingOrders = await Order.count({
        where: {
          userId,
          status: 'pending'
        }
      });

      // Get recent trade info
      const lastTrade = await Order.findOne({
        where: { userId, status: 'filled' },
        order: [['filledAt', 'DESC']]
      });

      return {
        active: session.status === 'running',
        uptime: Math.floor(uptime / 1000), // in seconds
        lastTrade: lastTrade?.filledAt?.toISOString(),
        activePairs: session.watchlist.length,
        pendingOrders,
        totalTrades: session.totalTrades,
        totalPnl: session.totalPnl
      };

    } catch (error) {
      logger.error(`Error getting trading status for user ${userId}:`, error);
      throw error;
    }
  }

  async getSignals(userId: string, options: {
    limit?: number;
    symbol?: string;
  }): Promise<TradingSignal[]> {
    try {
      const whereClause: any = { userId };
      
      if (options.symbol) {
        whereClause.symbol = options.symbol;
      }

      return await TradingSignal.findAll({
        where: whereClause,
        order: [['createdAt', 'DESC']],
        limit: options.limit || 10
      });

    } catch (error) {
      logger.error(`Error getting signals for user ${userId}:`, error);
      throw error;
    }
  }

  async updateSettings(userId: string, settings: Partial<TradingSettings>): Promise<void> {
    try {
      const session = this.activeSessions.get(userId);
      if (!session) {
        throw new Error('No active trading session found');
      }

      const updatedSettings = { ...session.settings, ...settings };
      
      await session.update({ settings: updatedSettings });
      
      logger.info(`Trading settings updated for user ${userId}`);

    } catch (error) {
      logger.error(`Error updating settings for user ${userId}:`, error);
      throw error;
    }
  }

  async getPerformance(userId: string, period: string) {
    try {
      return await this.analyticsService.getPerformanceMetrics(userId, period);
    } catch (error) {
      logger.error(`Error getting performance for user ${userId}:`, error);
      throw error;
    }
  }

  private async startTradingCycle(userId: string, session: TradingSession): Promise<void> {
    // Initial cycle run
    await this.runTradingCycle(userId, session);

    // Set up recurring trading cycle (45 seconds, same as current implementation)
    const interval = setInterval(async () => {
      try {
        await this.runTradingCycle(userId, session);
      } catch (error) {
        logger.error(`Error in trading cycle for user ${userId}:`, error);
        
        // Emit error to user
        this.webSocketServer.emitToUser(userId, 'trading_error', {
          message: 'Trading cycle error',
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }, 45000); // 45 seconds

    this.tickIntervals.set(userId, interval);
  }

  private async runTradingCycle(userId: string, session: TradingSession): Promise<void> {
    if (session.status !== 'running') {
      return;
    }

    try {
      logger.debug(`Running trading cycle for user ${userId}`);

      // 1. Fetch market data for watchlist
      const marketData = await this.marketDataService.getCryptoData(session.watchlist);
      
      // 2. Generate trading signals
      const signals = await this.analyticsService.generateSignals(userId, marketData, session.settings);
      
      // 3. Save signals to database
      for (const signalData of signals) {
        const signal = await TradingSignal.create({
          userId,
          sessionId: session.id,
          symbol: signalData.symbol,
          action: signalData.action,
          confidence: signalData.confidence,
          reason: signalData.reason,
          priceTarget: signalData.priceTarget,
          stopLoss: signalData.stopLoss,
          currentPrice: signalData.currentPrice,
          indicators: signalData.indicators
        });

        // Emit signal to user
        this.webSocketServer.emitToUser(userId, 'signal_generated', {
          signal: {
            symbol: signal.symbol,
            action: signal.action,
            confidence: signal.confidence,
            reason: signal.reason,
            timestamp: signal.createdAt.toISOString()
          }
        });
      }

      // 4. Process high-confidence signals for trading
      const tradableSignals = signals.filter(s => 
        s.confidence >= session.settings.confidenceThreshold && 
        s.action !== 'HOLD'
      );

      // 5. Execute trades through risk manager
      for (const signal of tradableSignals) {
        await this.processTradeSignal(userId, session, signal, marketData);
      }

      // 6. Emit analysis summary
      const top3Signals = signals
        .sort((a, b) => b.confidence - a.confidence)
        .slice(0, 3)
        .map(s => ({
          symbol: s.symbol,
          action: s.action,
          confidence: s.confidence
        }));

      this.webSocketServer.emitToUser(userId, 'analysis_complete', {
        evaluated: signals.length,
        top: top3Signals,
        timestamp: new Date().toISOString()
      });

      logger.debug(`Trading cycle completed for user ${userId}. Signals: ${signals.length}`);

    } catch (error) {
      logger.error(`Trading cycle error for user ${userId}:`, error);
      throw error;
    }
  }

  private async processTradeSignal(
    userId: string, 
    session: TradingSession, 
    signal: any, 
    marketData: any[]
  ): Promise<void> {
    try {
      // Check cooldown period
      const lastTrade = await Order.findOne({
        where: { 
          userId, 
          symbol: signal.symbol,
          status: 'filled'
        },
        order: [['filledAt', 'DESC']]
      });

      if (lastTrade && lastTrade.filledAt) {
        const timeSinceLastTrade = Date.now() - lastTrade.filledAt.getTime();
        const cooldownMs = session.settings.cooldownMinutes * 60 * 1000;
        
        if (timeSinceLastTrade < cooldownMs) {
          logger.debug(`Signal for ${signal.symbol} skipped due to cooldown`);
          return;
        }
      }

      // Check if adaptive threshold service should be consulted
      if (session.settings.adaptiveThresholdEnabled) {
        const shouldTrade = await this.checkAdaptiveThreshold(userId, signal);
        if (!shouldTrade) {
          logger.debug(`Signal for ${signal.symbol} filtered out by adaptive threshold`);
          return;
        }
      }

      // Risk management evaluation
      const riskEvaluation = await this.riskManager.evaluateSignal(userId, signal, marketData);
      
      if (!riskEvaluation.proceed) {
        logger.info(`Trade rejected by risk manager: ${riskEvaluation.reason}`);
        
        this.webSocketServer.emitToUser(userId, 'trade_rejected', {
          symbol: signal.symbol,
          reason: riskEvaluation.reason,
          timestamp: new Date().toISOString()
        });
        return;
      }

      // Place order through order manager
      const orderResult = await this.orderManager.placeOrder(userId, {
        sessionId: session.id,
        signalId: signal.id,
        symbol: signal.symbol,
        side: signal.action.toLowerCase(),
        orderType: 'market',
        quantity: riskEvaluation.quantity,
        price: signal.currentPrice
      });

      // Update session stats
      await session.increment('totalTrades');

      // Emit order confirmation
      this.webSocketServer.emitToUser(userId, 'order_placed', {
        orderId: orderResult.id,
        symbol: signal.symbol,
        side: signal.action.toLowerCase(),
        quantity: riskEvaluation.quantity,
        timestamp: new Date().toISOString()
      });

      logger.info(`Order placed for user ${userId}: ${signal.symbol} ${signal.action} ${riskEvaluation.quantity}`);

    } catch (error) {
      logger.error(`Error processing trade signal for user ${userId}:`, error);
      
      this.webSocketServer.emitToUser(userId, 'order_error', {
        symbol: signal.symbol,
        error: error.message,
        timestamp: new Date().toISOString()
      });
    }
  }

  private async checkAdaptiveThreshold(userId: string, signal: any): Promise<boolean> {
    try {
      // Call ML service to evaluate signal
      const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://ml-service:5000';
      const response = await fetch(`${mlServiceUrl}/api/v1/evaluate/${userId}`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          signal: {
            symbol: signal.symbol,
            confidence: signal.confidence,
            rsi: signal.indicators?.rsi,
            change_percent: signal.changePercent,
            volume: signal.volume,
            action: signal.action
          }
        })
      });

      if (!response.ok) {
        logger.warn(`ML service unavailable, proceeding with trade`);
        return true; // Fallback to allow trade if ML service is down
      }

      const result = await response.json();
      return result.data?.should_trade ?? true;

    } catch (error) {
      logger.error(`Error checking adaptive threshold:`, error);
      return true; // Fallback to allow trade on error
    }
  }

  // Cleanup method for graceful shutdown
  async shutdown(): Promise<void> {
    logger.info('Shutting down TradingService...');
    
    // Stop all active sessions
    for (const [userId] of this.activeSessions) {
      try {
        await this.stop(userId);
      } catch (error) {
        logger.error(`Error stopping session for user ${userId}:`, error);
      }
    }

    // Clear all intervals
    for (const [userId, interval] of this.tickIntervals) {
      clearInterval(interval);
    }
    this.tickIntervals.clear();
    
    logger.info('TradingService shutdown complete');
  }
}