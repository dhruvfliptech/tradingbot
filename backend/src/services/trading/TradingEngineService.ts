import { EventEmitter } from 'events';
import { DatabaseService } from '../database/DatabaseService';
import { OrderExecutor } from './OrderExecutor';
import { PositionManager } from './PositionManager';
import { StateManager } from './StateManager';
import { SignalProcessor } from './SignalProcessor';
import { MarketDataService } from './MarketDataService';
import { RiskManager } from './RiskManager';
import logger from '../../utils/logger';
import { TradingSignal, Order, Position, MarketData, TradingSettings, AgentEvent } from '../../types/trading';

export interface TradingSession {
  id: string;
  userId: string;
  status: 'active' | 'paused' | 'stopped';
  startedAt: Date;
  lastActivity: Date;
  settings: TradingSettings;
  activeTrades: number;
  totalPnL: number;
  weeklyTarget: number;
  currentWeekPnL: number;
}

export class TradingEngineService extends EventEmitter {
  private static instance: TradingEngineService;

  private orderExecutor: OrderExecutor;
  private positionManager: PositionManager;
  private stateManager: StateManager;
  private signalProcessor: SignalProcessor;
  private marketDataService: MarketDataService;
  private riskManager: RiskManager;
  private databaseService: DatabaseService;

  private sessions: Map<string, TradingSession> = new Map();
  private tickIntervals: Map<string, NodeJS.Timeout> = new Map();
  private isInitialized: boolean = false;

  // Trading parameters
  private readonly TICK_INTERVAL = 45000; // 45 seconds
  private readonly MIN_TRADE_INTERVAL = 3600000; // 1 hour between trades per symbol
  private readonly DEFAULT_WATCHLIST = ['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano'];

  private constructor() {
    super();
    this.orderExecutor = new OrderExecutor();
    this.positionManager = new PositionManager();
    this.stateManager = new StateManager();
    this.signalProcessor = new SignalProcessor();
    this.marketDataService = new MarketDataService();
    this.riskManager = new RiskManager();
    this.databaseService = DatabaseService.getInstance();
  }

  public static getInstance(): TradingEngineService {
    if (!TradingEngineService.instance) {
      TradingEngineService.instance = new TradingEngineService();
    }
    return TradingEngineService.instance;
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Initialize all sub-services
      await this.stateManager.initialize();
      await this.marketDataService.initialize();
      await this.positionManager.initialize();
      await this.orderExecutor.initialize();
      await this.riskManager.initialize();

      // Set up event listeners
      this.setupEventListeners();

      // Restore active sessions from database
      await this.restoreActiveSessions();

      this.isInitialized = true;
      logger.info('Trading Engine Service initialized successfully');
    } catch (error) {
      logger.error('Failed to initialize Trading Engine Service:', error);
      throw error;
    }
  }

  private setupEventListeners(): void {
    // Listen to order events
    this.orderExecutor.on('orderFilled', (order) => this.handleOrderFilled(order));
    this.orderExecutor.on('orderFailed', (order, error) => this.handleOrderFailed(order, error));

    // Listen to position events
    this.positionManager.on('positionOpened', (position) => this.handlePositionOpened(position));
    this.positionManager.on('positionClosed', (position) => this.handlePositionClosed(position));

    // Listen to risk events
    this.riskManager.on('riskLimitExceeded', (userId, reason) => this.handleRiskLimitExceeded(userId, reason));
    this.riskManager.on('emergencyStop', (userId) => this.handleEmergencyStop(userId));
  }

  async startSession(userId: string, settings?: TradingSettings): Promise<TradingSession> {
    try {
      // Check if session already exists
      if (this.sessions.has(userId)) {
        const existingSession = this.sessions.get(userId)!;
        if (existingSession.status === 'active') {
          logger.warn(`Session already active for user ${userId}`);
          return existingSession;
        }
      }

      // Load user settings from database or use provided ones
      const userSettings = settings || await this.loadUserSettings(userId);

      // Create new session
      const session: TradingSession = {
        id: `session_${userId}_${Date.now()}`,
        userId,
        status: 'active',
        startedAt: new Date(),
        lastActivity: new Date(),
        settings: userSettings,
        activeTrades: 0,
        totalPnL: 0,
        weeklyTarget: userSettings.weeklyTarget || 4, // 4% default
        currentWeekPnL: 0
      };

      // Store session
      this.sessions.set(userId, session);
      await this.stateManager.saveSession(session);

      // Start trading loop
      this.startTradingLoop(userId);

      // Emit event
      this.emit('sessionStarted', { userId, session });

      // Log to audit
      await this.logAuditEvent(userId, 'SESSION_START', { sessionId: session.id });

      logger.info(`Trading session started for user ${userId}`, { sessionId: session.id });
      return session;

    } catch (error) {
      logger.error(`Failed to start session for user ${userId}:`, error);
      throw error;
    }
  }

  async stopSession(userId: string): Promise<void> {
    try {
      const session = this.sessions.get(userId);
      if (!session) {
        logger.warn(`No session found for user ${userId}`);
        return;
      }

      // Stop trading loop
      this.stopTradingLoop(userId);

      // Close all open positions
      await this.closeAllPositions(userId);

      // Cancel all pending orders
      await this.cancelAllOrders(userId);

      // Update session status
      session.status = 'stopped';
      session.lastActivity = new Date();
      await this.stateManager.saveSession(session);

      // Remove from active sessions
      this.sessions.delete(userId);

      // Emit event
      this.emit('sessionStopped', { userId, session });

      // Log to audit
      await this.logAuditEvent(userId, 'SESSION_STOP', {
        sessionId: session.id,
        totalPnL: session.totalPnL
      });

      logger.info(`Trading session stopped for user ${userId}`, { sessionId: session.id });

    } catch (error) {
      logger.error(`Failed to stop session for user ${userId}:`, error);
      throw error;
    }
  }

  async pauseSession(userId: string): Promise<void> {
    const session = this.sessions.get(userId);
    if (!session) throw new Error(`No session found for user ${userId}`);

    session.status = 'paused';
    this.stopTradingLoop(userId);
    await this.stateManager.saveSession(session);

    this.emit('sessionPaused', { userId, session });
    logger.info(`Trading session paused for user ${userId}`);
  }

  async resumeSession(userId: string): Promise<void> {
    const session = this.sessions.get(userId);
    if (!session) throw new Error(`No session found for user ${userId}`);

    session.status = 'active';
    this.startTradingLoop(userId);
    await this.stateManager.saveSession(session);

    this.emit('sessionResumed', { userId, session });
    logger.info(`Trading session resumed for user ${userId}`);
  }

  private startTradingLoop(userId: string): void {
    // Clear any existing interval
    this.stopTradingLoop(userId);

    // Execute immediately
    this.executeTradingCycle(userId);

    // Set up interval
    const interval = setInterval(() => {
      this.executeTradingCycle(userId);
    }, this.TICK_INTERVAL);

    this.tickIntervals.set(userId, interval);
  }

  private stopTradingLoop(userId: string): void {
    const interval = this.tickIntervals.get(userId);
    if (interval) {
      clearInterval(interval);
      this.tickIntervals.delete(userId);
    }
  }

  private async executeTradingCycle(userId: string): Promise<void> {
    const session = this.sessions.get(userId);
    if (!session || session.status !== 'active') return;

    try {
      logger.debug(`Executing trading cycle for user ${userId}`);

      // Update session activity
      session.lastActivity = new Date();

      // Get market data
      const watchlist = session.settings.watchlist || this.DEFAULT_WATCHLIST;
      const marketData = await this.marketDataService.getMarketData(watchlist);

      // Check risk limits
      const riskCheck = await this.riskManager.checkRiskLimits(userId, session);
      if (!riskCheck.passed) {
        logger.warn(`Risk limits exceeded for user ${userId}: ${riskCheck.reason}`);
        this.emit('riskLimitExceeded', { userId, reason: riskCheck.reason });
        return;
      }

      // Generate trading signals
      const signals = await this.signalProcessor.generateSignals(
        marketData,
        session.settings,
        await this.stateManager.getMarketRegime()
      );

      // Filter signals based on confidence and other criteria
      const validSignals = await this.filterSignals(userId, signals, session.settings);

      // Execute trades for valid signals
      for (const signal of validSignals) {
        await this.executeSignal(userId, signal, session);
      }

      // Update positions
      await this.updatePositions(userId);

      // Save session state
      await this.stateManager.saveSession(session);

      // Emit cycle complete event
      this.emit('tradingCycleComplete', {
        userId,
        signalsEvaluated: signals.length,
        tradesExecuted: validSignals.length
      });

    } catch (error) {
      logger.error(`Trading cycle error for user ${userId}:`, error);
      this.emit('tradingCycleError', { userId, error });
    }
  }

  private async filterSignals(
    userId: string,
    signals: TradingSignal[],
    settings: TradingSettings
  ): Promise<TradingSignal[]> {
    const filtered: TradingSignal[] = [];
    const lastTrades = await this.stateManager.getLastTrades(userId);

    for (const signal of signals) {
      // Check confidence threshold
      if (signal.confidence < (settings.minConfidence || 0.7)) {
        continue;
      }

      // Check if enough time has passed since last trade for this symbol
      const lastTradeTime = lastTrades[signal.symbol];
      if (lastTradeTime && Date.now() - lastTradeTime < this.MIN_TRADE_INTERVAL) {
        continue;
      }

      // Check position limits
      const positions = await this.positionManager.getPositions(userId);
      if (positions.length >= (settings.maxPositions || 5)) {
        continue;
      }

      // Additional validation from validators
      const validationResult = await this.signalProcessor.validateSignal(signal, settings);
      if (!validationResult.isValid) {
        logger.debug(`Signal validation failed for ${signal.symbol}: ${validationResult.reason}`);
        continue;
      }

      filtered.push(signal);
    }

    return filtered;
  }

  private async executeSignal(
    userId: string,
    signal: TradingSignal,
    session: TradingSession
  ): Promise<void> {
    try {
      // Calculate position size
      const positionSize = await this.riskManager.calculatePositionSize(
        userId,
        signal,
        session.settings
      );

      if (positionSize <= 0) {
        logger.warn(`Position size is 0 for signal ${signal.symbol}, skipping`);
        return;
      }

      // Create order
      const order: Order = {
        id: `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        userId,
        symbol: signal.symbol,
        side: signal.action === 'BUY' ? 'buy' : 'sell',
        type: session.settings.orderType || 'market',
        quantity: positionSize,
        status: 'pending',
        createdAt: new Date(),
        signal: signal
      };

      // Execute order
      const executedOrder = await this.orderExecutor.executeOrder(order);

      // Update last trade time
      await this.stateManager.updateLastTrade(userId, signal.symbol);

      // Log to audit
      await this.logAuditEvent(userId, 'ORDER_EXECUTED', {
        orderId: executedOrder.id,
        symbol: signal.symbol,
        side: order.side,
        quantity: order.quantity,
        confidence: signal.confidence
      });

      // Emit event
      this.emit('orderExecuted', { userId, order: executedOrder, signal });

    } catch (error) {
      logger.error(`Failed to execute signal for ${signal.symbol}:`, error);
      this.emit('orderExecutionFailed', { userId, signal, error });
    }
  }

  private async updatePositions(userId: string): Promise<void> {
    const positions = await this.positionManager.getPositions(userId);
    const session = this.sessions.get(userId);
    if (!session) return;

    for (const position of positions) {
      // Check stop loss and take profit
      const shouldClose = await this.riskManager.shouldClosePosition(position, session.settings);

      if (shouldClose) {
        await this.closePosition(userId, position.id, shouldClose.reason);
      }
    }

    // Update session PnL
    const totalPnL = await this.positionManager.calculateTotalPnL(userId);
    session.totalPnL = totalPnL;
    session.currentWeekPnL = await this.positionManager.calculateWeeklyPnL(userId);
  }

  private async closePosition(userId: string, positionId: string, reason: string): Promise<void> {
    try {
      const position = await this.positionManager.getPosition(positionId);
      if (!position) return;

      // Create closing order
      const order: Order = {
        id: `close_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        userId,
        symbol: position.symbol,
        side: position.side === 'long' ? 'sell' : 'buy',
        type: 'market',
        quantity: position.quantity,
        status: 'pending',
        createdAt: new Date(),
        positionId: positionId
      };

      // Execute closing order
      const executedOrder = await this.orderExecutor.executeOrder(order);

      // Update position
      await this.positionManager.closePosition(positionId, executedOrder.price!);

      // Log to audit
      await this.logAuditEvent(userId, 'POSITION_CLOSED', {
        positionId,
        reason,
        pnl: position.unrealizedPnL
      });

      logger.info(`Position ${positionId} closed for user ${userId}, reason: ${reason}`);

    } catch (error) {
      logger.error(`Failed to close position ${positionId}:`, error);
    }
  }

  private async closeAllPositions(userId: string): Promise<void> {
    const positions = await this.positionManager.getPositions(userId);
    for (const position of positions) {
      await this.closePosition(userId, position.id, 'SESSION_STOP');
    }
  }

  private async cancelAllOrders(userId: string): Promise<void> {
    const orders = await this.orderExecutor.getPendingOrders(userId);
    for (const order of orders) {
      await this.orderExecutor.cancelOrder(order.id);
    }
  }

  private async loadUserSettings(userId: string): Promise<TradingSettings> {
    try {
      const settings = await this.databaseService.getUserSettings(userId);
      return settings || this.getDefaultSettings();
    } catch (error) {
      logger.error(`Failed to load user settings for ${userId}:`, error);
      return this.getDefaultSettings();
    }
  }

  private getDefaultSettings(): TradingSettings {
    return {
      maxPositions: 5,
      maxPositionSize: 0.1, // 10% of portfolio
      minConfidence: 0.7,
      stopLossPercent: 0.05, // 5%
      takeProfitPercent: 0.15, // 15%
      trailingStopPercent: 0.03, // 3%
      weeklyTarget: 4, // 4% weekly target
      maxDrawdown: 0.15, // 15% max drawdown
      orderType: 'market',
      watchlist: this.DEFAULT_WATCHLIST,
      enabledStrategies: ['liquidity', 'smartMoney', 'volumeProfile', 'microstructure'],
      validatorEnabled: true,
      strategyWeightBalance: 0.5 // 50/50 balance between strategies and validators
    };
  }

  private async restoreActiveSessions(): Promise<void> {
    try {
      const activeSessions = await this.stateManager.getActiveSessions();
      for (const session of activeSessions) {
        this.sessions.set(session.userId, session);
        if (session.status === 'active') {
          this.startTradingLoop(session.userId);
        }
      }
      logger.info(`Restored ${activeSessions.length} active sessions`);
    } catch (error) {
      logger.error('Failed to restore active sessions:', error);
    }
  }

  private async logAuditEvent(userId: string, eventType: string, data: any): Promise<void> {
    try {
      await this.databaseService.logAuditEvent({
        userId,
        eventType,
        data,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to log audit event:', error);
    }
  }

  // Event handlers
  private async handleOrderFilled(order: Order): Promise<void> {
    logger.info(`Order filled: ${order.id}`);
    this.emit('orderFilled', order);
  }

  private async handleOrderFailed(order: Order, error: any): Promise<void> {
    logger.error(`Order failed: ${order.id}`, error);
    this.emit('orderFailed', { order, error });
  }

  private async handlePositionOpened(position: Position): Promise<void> {
    logger.info(`Position opened: ${position.id}`);
    this.emit('positionOpened', position);
  }

  private async handlePositionClosed(position: Position): Promise<void> {
    logger.info(`Position closed: ${position.id}, PnL: ${position.realizedPnL}`);
    this.emit('positionClosed', position);
  }

  private async handleRiskLimitExceeded(userId: string, reason: string): Promise<void> {
    logger.warn(`Risk limit exceeded for user ${userId}: ${reason}`);
    await this.pauseSession(userId);
  }

  private async handleEmergencyStop(userId: string): Promise<void> {
    logger.error(`Emergency stop triggered for user ${userId}`);
    await this.stopSession(userId);
  }

  // Public API methods
  async getSession(userId: string): Promise<TradingSession | null> {
    return this.sessions.get(userId) || null;
  }

  async getPositions(userId: string): Promise<Position[]> {
    return this.positionManager.getPositions(userId);
  }

  async getOrders(userId: string, status?: string): Promise<Order[]> {
    return this.orderExecutor.getOrders(userId, status);
  }

  async updateSettings(userId: string, settings: Partial<TradingSettings>): Promise<void> {
    const session = this.sessions.get(userId);
    if (!session) throw new Error(`No session found for user ${userId}`);

    session.settings = { ...session.settings, ...settings };
    await this.stateManager.saveSession(session);

    this.emit('settingsUpdated', { userId, settings });
  }

  async getPerformanceMetrics(userId: string, period: string = '24h'): Promise<any> {
    return this.positionManager.getPerformanceMetrics(userId, period);
  }

  async emergencyStopAll(): Promise<void> {
    logger.warn('Emergency stop all triggered');
    for (const [userId] of this.sessions) {
      await this.stopSession(userId);
    }
  }
}

export default TradingEngineService;