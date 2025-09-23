import { EventEmitter } from 'events';
import WebSocket from 'ws';
import fetch from 'node-fetch';
import logger from '../../utils/logger';

export interface BacktestConfig {
  strategyId: string;
  symbols: string[];
  startDate: string;
  endDate: string;
  initialCapital: number;
  parameters: Record<string, any>;
  riskSettings: {
    maxPositionSize: number;
    stopLoss: number;
    takeProfit: number;
  };
}

export interface BacktestResult {
  id: string;
  strategyId: string;
  performance: {
    totalReturn: number;
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    profitFactor: number;
    sortino: number;
    calmar: number;
    volatility: number;
  };
  trades: Trade[];
  metrics: PerformanceMetrics;
  riskAnalysis: RiskAnalysis;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  startedAt: Date;
  completedAt?: Date;
  error?: string;
}

export interface Trade {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  entryPrice: number;
  exitPrice?: number;
  entryTime: Date;
  exitTime?: Date;
  pnl?: number;
  pnlPercent?: number;
  commission: number;
  reason: string;
  confidence: number;
}

export interface PerformanceMetrics {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  consecutiveWins: number;
  consecutiveLosses: number;
  avgTradeDuration: number;
  totalCommissions: number;
}

export interface RiskAnalysis {
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  expectedShortfall: number;
  beta: number;
  alpha: number;
  trackingError: number;
  informationRatio: number;
  downsideDeviation: number;
}

export interface HistoricalData {
  symbol: string;
  timestamp: Date;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
  indicators?: Record<string, number>;
}

export interface StrategyDefinition {
  id: string;
  name: string;
  description: string;
  parameters: StrategyParameter[];
  entryRules: string[];
  exitRules: string[];
  riskManagement: RiskRule[];
  version: string;
}

export interface StrategyParameter {
  name: string;
  type: 'number' | 'boolean' | 'string' | 'array';
  defaultValue: any;
  min?: number;
  max?: number;
  description: string;
}

export interface RiskRule {
  type: 'stop_loss' | 'take_profit' | 'position_size' | 'correlation' | 'exposure';
  parameter: string;
  value: number;
  condition: 'greater_than' | 'less_than' | 'equal_to';
}

export class ComposerService extends EventEmitter {
  private mcpUrl: string;
  private ws: WebSocket | null = null;
  private isConnected: boolean = false;
  private reconnectAttempts: number = 0;
  private maxReconnectAttempts: number = 5;
  private reconnectDelay: number = 1000;
  private activeBacktests: Map<string, BacktestResult> = new Map();
  private messageHandlers: Map<string, (data: any) => void> = new Map();

  constructor(mcpUrl: string = 'https://ai.composer.trade/mcp') {
    super();
    this.mcpUrl = mcpUrl;
    this.setupMessageHandlers();
  }

  private setupMessageHandlers(): void {
    this.messageHandlers.set('backtest_progress', this.handleBacktestProgress.bind(this));
    this.messageHandlers.set('backtest_completed', this.handleBacktestCompleted.bind(this));
    this.messageHandlers.set('backtest_error', this.handleBacktestError.bind(this));
    this.messageHandlers.set('historical_data', this.handleHistoricalData.bind(this));
  }

  async connect(): Promise<void> {
    try {
      logger.info('Connecting to Composer MCP server...');
      
      const wsUrl = this.mcpUrl.replace('https://', 'wss://').replace('http://', 'ws://');
      this.ws = new WebSocket(wsUrl);

      return new Promise((resolve, reject) => {
        if (!this.ws) {
          reject(new Error('WebSocket not initialized'));
          return;
        }

        this.ws.on('open', () => {
          logger.info('Connected to Composer MCP server');
          this.isConnected = true;
          this.reconnectAttempts = 0;
          this.emit('connected');
          resolve();
        });

        this.ws.on('message', this.handleMessage.bind(this));

        this.ws.on('close', () => {
          logger.warn('Disconnected from Composer MCP server');
          this.isConnected = false;
          this.emit('disconnected');
          this.attemptReconnect();
        });

        this.ws.on('error', (error) => {
          logger.error('Composer MCP WebSocket error:', error);
          this.emit('error', error);
          reject(error);
        });

        // Timeout for connection
        setTimeout(() => {
          if (!this.isConnected) {
            reject(new Error('Connection timeout'));
          }
        }, 10000);
      });
    } catch (error) {
      logger.error('Error connecting to Composer MCP:', error);
      throw error;
    }
  }

  private async attemptReconnect(): Promise<void> {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);
    
    logger.info(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(async () => {
      try {
        await this.connect();
      } catch (error) {
        logger.error('Reconnection failed:', error);
      }
    }, delay);
  }

  private handleMessage(data: Buffer): void {
    try {
      const message = JSON.parse(data.toString());
      const handler = this.messageHandlers.get(message.type);
      
      if (handler) {
        handler(message.data);
      } else {
        logger.warn('Unknown message type from Composer MCP:', message.type);
      }
    } catch (error) {
      logger.error('Error parsing Composer MCP message:', error);
    }
  }

  private handleBacktestProgress(data: any): void {
    const { backtestId, progress, currentDate, performance } = data;
    
    if (this.activeBacktests.has(backtestId)) {
      this.emit('backtest_progress', {
        backtestId,
        progress,
        currentDate,
        performance
      });
    }
  }

  private handleBacktestCompleted(data: any): void {
    const { backtestId, result } = data;
    
    if (this.activeBacktests.has(backtestId)) {
      const backtest = this.activeBacktests.get(backtestId)!;
      backtest.status = 'completed';
      backtest.completedAt = new Date();
      backtest.performance = result.performance;
      backtest.trades = result.trades;
      backtest.metrics = result.metrics;
      backtest.riskAnalysis = result.riskAnalysis;
      
      this.emit('backtest_completed', backtest);
    }
  }

  private handleBacktestError(data: any): void {
    const { backtestId, error } = data;
    
    if (this.activeBacktests.has(backtestId)) {
      const backtest = this.activeBacktests.get(backtestId)!;
      backtest.status = 'failed';
      backtest.error = error;
      backtest.completedAt = new Date();
      
      this.emit('backtest_error', backtest);
    }
  }

  private handleHistoricalData(data: any): void {
    this.emit('historical_data', data);
  }

  private sendMessage(type: string, data: any): void {
    if (!this.isConnected || !this.ws) {
      throw new Error('Not connected to Composer MCP server');
    }

    const message = {
      type,
      data,
      timestamp: new Date().toISOString()
    };

    this.ws.send(JSON.stringify(message));
  }

  async runBacktest(config: BacktestConfig): Promise<string> {
    try {
      logger.info(`Starting backtest for strategy ${config.strategyId}`);

      const backtestId = `bt_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      
      const backtestResult: BacktestResult = {
        id: backtestId,
        strategyId: config.strategyId,
        performance: {
          totalReturn: 0,
          sharpeRatio: 0,
          maxDrawdown: 0,
          winRate: 0,
          profitFactor: 0,
          sortino: 0,
          calmar: 0,
          volatility: 0
        },
        trades: [],
        metrics: {
          totalTrades: 0,
          winningTrades: 0,
          losingTrades: 0,
          avgWin: 0,
          avgLoss: 0,
          largestWin: 0,
          largestLoss: 0,
          consecutiveWins: 0,
          consecutiveLosses: 0,
          avgTradeDuration: 0,
          totalCommissions: 0
        },
        riskAnalysis: {
          var95: 0,
          var99: 0,
          expectedShortfall: 0,
          beta: 0,
          alpha: 0,
          trackingError: 0,
          informationRatio: 0,
          downsideDeviation: 0
        },
        status: 'running',
        startedAt: new Date()
      };

      this.activeBacktests.set(backtestId, backtestResult);

      this.sendMessage('run_backtest', {
        backtestId,
        config
      });

      return backtestId;
    } catch (error) {
      logger.error('Error starting backtest:', error);
      throw error;
    }
  }

  async cancelBacktest(backtestId: string): Promise<void> {
    try {
      if (!this.activeBacktests.has(backtestId)) {
        throw new Error('Backtest not found');
      }

      this.sendMessage('cancel_backtest', { backtestId });
      
      const backtest = this.activeBacktests.get(backtestId)!;
      backtest.status = 'cancelled';
      backtest.completedAt = new Date();

      logger.info(`Cancelled backtest ${backtestId}`);
    } catch (error) {
      logger.error('Error cancelling backtest:', error);
      throw error;
    }
  }

  async getBacktestStatus(backtestId: string): Promise<BacktestResult | null> {
    return this.activeBacktests.get(backtestId) || null;
  }

  async getHistoricalData(
    symbols: string[],
    startDate: string,
    endDate: string,
    timeframe: '1m' | '5m' | '15m' | '1h' | '4h' | '1d' = '1h'
  ): Promise<HistoricalData[]> {
    try {
      logger.info(`Fetching historical data for ${symbols.join(', ')} from ${startDate} to ${endDate}`);

      // Use HTTP API for historical data requests
      const response = await fetch(`${this.mcpUrl}/api/historical-data`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbols,
          startDate,
          endDate,
          timeframe
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result.data || [];
    } catch (error) {
      logger.error('Error fetching historical data:', error);
      throw error;
    }
  }

  async validateStrategy(strategy: StrategyDefinition): Promise<{
    isValid: boolean;
    errors: string[];
    warnings: string[];
  }> {
    try {
      const response = await fetch(`${this.mcpUrl}/api/validate-strategy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ strategy })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('Error validating strategy:', error);
      throw error;
    }
  }

  async getAvailableStrategies(): Promise<StrategyDefinition[]> {
    try {
      const response = await fetch(`${this.mcpUrl}/api/strategies`, {
        method: 'GET',
        headers: {
          'Content-Type': 'application/json',
        }
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const result = await response.json();
      return result.data || [];
    } catch (error) {
      logger.error('Error fetching available strategies:', error);
      throw error;
    }
  }

  async optimizeStrategy(
    strategyId: string,
    config: BacktestConfig,
    optimizationParams: {
      parameters: string[];
      method: 'grid' | 'genetic' | 'bayesian';
      iterations: number;
      objectiveFunction: 'sharpe' | 'return' | 'profit_factor' | 'sortino';
    }
  ): Promise<{
    optimizationId: string;
    bestParameters: Record<string, any>;
    bestPerformance: number;
    results: Array<{
      parameters: Record<string, any>;
      performance: number;
      metrics: PerformanceMetrics;
    }>;
  }> {
    try {
      const response = await fetch(`${this.mcpUrl}/api/optimize-strategy`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          strategyId,
          config,
          optimization: optimizationParams
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('Error optimizing strategy:', error);
      throw error;
    }
  }

  async getMarketRegimes(
    symbols: string[],
    startDate: string,
    endDate: string
  ): Promise<{
    regimes: Array<{
      period: { start: string; end: string };
      type: 'bull' | 'bear' | 'sideways' | 'volatile';
      characteristics: {
        volatility: number;
        trend: number;
        momentum: number;
      };
    }>;
  }> {
    try {
      const response = await fetch(`${this.mcpUrl}/api/market-regimes`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          symbols,
          startDate,
          endDate
        })
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      return await response.json();
    } catch (error) {
      logger.error('Error fetching market regimes:', error);
      throw error;
    }
  }

  disconnect(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
    logger.info('Disconnected from Composer MCP server');
  }

  isConnectedToMcp(): boolean {
    return this.isConnected;
  }

  getActiveBacktests(): BacktestResult[] {
    return Array.from(this.activeBacktests.values());
  }
}