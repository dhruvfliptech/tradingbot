import { EventEmitter } from 'events';
import logger from '../../utils/logger';
import { TradingSignal, Position, TradingSettings } from '../../types/trading';
import { TradingSession } from './TradingEngineService';
import { PositionManager } from './PositionManager';
import { StateManager } from './StateManager';

interface RiskCheck {
  passed: boolean;
  reason?: string;
  metrics?: any;
}

interface PositionCloseSignal {
  shouldClose: boolean;
  reason: string;
  urgency: 'immediate' | 'normal' | 'monitoring';
}

export class RiskManager extends EventEmitter {
  private positionManager: PositionManager;
  private stateManager: StateManager;

  // Risk parameters
  private readonly MAX_PORTFOLIO_RISK = 0.15; // 15% max portfolio risk
  private readonly MAX_POSITION_RISK = 0.10; // 10% max per position
  private readonly MAX_DAILY_LOSS = 0.05; // 5% max daily loss
  private readonly MAX_CONSECUTIVE_LOSSES = 5;
  private readonly HOURLY_VOLATILITY_LIMIT = 0.05; // 5% hourly volatility limit
  private readonly CORRELATION_LIMIT = 0.7; // Max correlation between positions

  // Tracking variables
  private dailyPnL: Map<string, number> = new Map();
  private consecutiveLosses: Map<string, number> = new Map();
  private lastVolatilityCheck: number = 0;
  private marketVolatility: number = 0;

  constructor() {
    super();
    this.positionManager = new PositionManager();
    this.stateManager = new StateManager();
  }

  async initialize(): Promise<void> {
    try {
      // Initialize position manager if not already done
      await this.positionManager.initialize();

      // Start risk monitoring
      this.startRiskMonitoring();

      logger.info('Risk Manager initialized');
    } catch (error) {
      logger.error('Failed to initialize Risk Manager:', error);
      throw error;
    }
  }

  async checkRiskLimits(userId: string, session: TradingSession): Promise<RiskCheck> {
    try {
      const checks: RiskCheck[] = [];

      // Run all risk checks in parallel
      const [
        drawdownCheck,
        exposureCheck,
        volatilityCheck,
        consecutiveLossCheck,
        dailyLossCheck,
        correlationCheck
      ] = await Promise.all([
        this.checkDrawdown(userId, session),
        this.checkExposure(userId, session),
        this.checkVolatility(),
        this.checkConsecutiveLosses(userId),
        this.checkDailyLoss(userId),
        this.checkCorrelation(userId)
      ]);

      checks.push(
        drawdownCheck,
        exposureCheck,
        volatilityCheck,
        consecutiveLossCheck,
        dailyLossCheck,
        correlationCheck
      );

      // Find any failed checks
      const failedCheck = checks.find(check => !check.passed);
      if (failedCheck) {
        logger.warn(`Risk check failed for user ${userId}: ${failedCheck.reason}`);
        this.emit('riskLimitExceeded', userId, failedCheck.reason);
        return failedCheck;
      }

      return {
        passed: true,
        metrics: {
          drawdown: drawdownCheck.metrics?.drawdown || 0,
          exposure: exposureCheck.metrics?.exposure || 0,
          volatility: volatilityCheck.metrics?.volatility || 0,
          dailyPnL: dailyLossCheck.metrics?.dailyPnL || 0
        }
      };

    } catch (error) {
      logger.error(`Risk check error for user ${userId}:`, error);
      return {
        passed: false,
        reason: 'Risk check system error'
      };
    }
  }

  private async checkDrawdown(userId: string, session: TradingSession): Promise<RiskCheck> {
    const totalPnL = session.totalPnL;
    const portfolioValue = await this.getPortfolioValue(userId);
    const drawdown = portfolioValue > 0 ? Math.abs(Math.min(0, totalPnL / portfolioValue)) : 0;

    if (drawdown > session.settings.maxDrawdown) {
      return {
        passed: false,
        reason: `Drawdown ${(drawdown * 100).toFixed(2)}% exceeds limit ${(session.settings.maxDrawdown * 100).toFixed(2)}%`,
        metrics: { drawdown }
      };
    }

    return {
      passed: true,
      metrics: { drawdown }
    };
  }

  private async checkExposure(userId: string, session: TradingSession): Promise<RiskCheck> {
    const positions = await this.positionManager.getOpenPositions(userId);
    const portfolioValue = await this.getPortfolioValue(userId);

    let totalExposure = 0;
    for (const position of positions) {
      totalExposure += position.quantity * position.currentPrice;
    }

    const exposureRatio = portfolioValue > 0 ? totalExposure / portfolioValue : 0;

    if (exposureRatio > 0.6) { // 60% max exposure
      return {
        passed: false,
        reason: `Exposure ${(exposureRatio * 100).toFixed(2)}% exceeds 60% limit`,
        metrics: { exposure: exposureRatio }
      };
    }

    return {
      passed: true,
      metrics: { exposure: exposureRatio }
    };
  }

  private async checkVolatility(): Promise<RiskCheck> {
    // Check hourly volatility
    const now = Date.now();
    if (now - this.lastVolatilityCheck < 3600000) { // Use cached value for 1 hour
      if (this.marketVolatility > this.HOURLY_VOLATILITY_LIMIT) {
        return {
          passed: false,
          reason: `Market volatility ${(this.marketVolatility * 100).toFixed(2)}% exceeds ${(this.HOURLY_VOLATILITY_LIMIT * 100).toFixed(2)}% limit`,
          metrics: { volatility: this.marketVolatility }
        };
      }
    }

    // Calculate new volatility (simplified)
    this.marketVolatility = await this.calculateMarketVolatility();
    this.lastVolatilityCheck = now;

    if (this.marketVolatility > this.HOURLY_VOLATILITY_LIMIT) {
      return {
        passed: false,
        reason: `Market volatility ${(this.marketVolatility * 100).toFixed(2)}% exceeds ${(this.HOURLY_VOLATILITY_LIMIT * 100).toFixed(2)}% limit`,
        metrics: { volatility: this.marketVolatility }
      };
    }

    return {
      passed: true,
      metrics: { volatility: this.marketVolatility }
    };
  }

  private async checkConsecutiveLosses(userId: string): Promise<RiskCheck> {
    const losses = this.consecutiveLosses.get(userId) || 0;

    if (losses >= this.MAX_CONSECUTIVE_LOSSES) {
      return {
        passed: false,
        reason: `${losses} consecutive losses exceeds limit of ${this.MAX_CONSECUTIVE_LOSSES}`,
        metrics: { consecutiveLosses: losses }
      };
    }

    return {
      passed: true,
      metrics: { consecutiveLosses: losses }
    };
  }

  private async checkDailyLoss(userId: string): Promise<RiskCheck> {
    const dailyPnL = this.dailyPnL.get(userId) || 0;
    const portfolioValue = await this.getPortfolioValue(userId);
    const dailyLossRatio = portfolioValue > 0 ? Math.abs(Math.min(0, dailyPnL / portfolioValue)) : 0;

    if (dailyLossRatio > this.MAX_DAILY_LOSS) {
      return {
        passed: false,
        reason: `Daily loss ${(dailyLossRatio * 100).toFixed(2)}% exceeds ${(this.MAX_DAILY_LOSS * 100).toFixed(2)}% limit`,
        metrics: { dailyPnL, dailyLossRatio }
      };
    }

    return {
      passed: true,
      metrics: { dailyPnL, dailyLossRatio }
    };
  }

  private async checkCorrelation(userId: string): Promise<RiskCheck> {
    const positions = await this.positionManager.getOpenPositions(userId);

    if (positions.length < 2) {
      return { passed: true, metrics: { maxCorrelation: 0 } };
    }

    // Simplified correlation check - in production, use actual price correlation
    const symbols = positions.map(p => p.symbol);
    const correlations = await this.calculateCorrelations(symbols);
    const maxCorrelation = Math.max(...correlations);

    if (maxCorrelation > this.CORRELATION_LIMIT) {
      return {
        passed: false,
        reason: `Position correlation ${maxCorrelation.toFixed(2)} exceeds ${this.CORRELATION_LIMIT} limit`,
        metrics: { maxCorrelation }
      };
    }

    return {
      passed: true,
      metrics: { maxCorrelation }
    };
  }

  async calculatePositionSize(
    userId: string,
    signal: TradingSignal,
    settings: TradingSettings
  ): Promise<number> {
    try {
      const portfolioValue = await this.getPortfolioValue(userId);
      if (portfolioValue <= 0) return 0;

      // Kelly Criterion with safety factor
      const winRate = 0.55; // Historical win rate (would be calculated from actual data)
      const avgWinLoss = 2; // Average win/loss ratio
      const kellyFraction = (winRate * avgWinLoss - (1 - winRate)) / avgWinLoss;
      const safetyFactor = 0.25; // Use 25% of Kelly
      const kellySize = kellyFraction * safetyFactor;

      // Adjust for confidence
      const confidenceAdjustment = Math.pow(signal.confidence, 2);

      // Calculate base position size
      let positionSize = portfolioValue * kellySize * confidenceAdjustment;

      // Apply max position size limit
      const maxSize = portfolioValue * (settings.maxPositionSize || this.MAX_POSITION_RISK);
      positionSize = Math.min(positionSize, maxSize);

      // Adjust for volatility
      const volatilityAdjustment = 1 / (1 + this.marketVolatility * 10);
      positionSize *= volatilityAdjustment;

      // Adjust for existing exposure
      const currentExposure = await this.getCurrentExposure(userId);
      const remainingCapacity = portfolioValue * 0.6 - currentExposure; // 60% max total exposure
      positionSize = Math.min(positionSize, remainingCapacity);

      // Convert to quantity based on price
      const quantity = positionSize / signal.price;

      return Math.max(0, quantity);

    } catch (error) {
      logger.error('Error calculating position size:', error);
      return 0;
    }
  }

  async shouldClosePosition(
    position: Position,
    settings: TradingSettings
  ): Promise<PositionCloseSignal | null> {
    // Check stop loss
    if (position.stopLoss) {
      if (position.side === 'long' && position.currentPrice <= position.stopLoss) {
        return {
          shouldClose: true,
          reason: 'Stop loss triggered',
          urgency: 'immediate'
        };
      }
      if (position.side === 'short' && position.currentPrice >= position.stopLoss) {
        return {
          shouldClose: true,
          reason: 'Stop loss triggered',
          urgency: 'immediate'
        };
      }
    }

    // Check take profit
    if (position.takeProfit) {
      if (position.side === 'long' && position.currentPrice >= position.takeProfit) {
        return {
          shouldClose: true,
          reason: 'Take profit reached',
          urgency: 'normal'
        };
      }
      if (position.side === 'short' && position.currentPrice <= position.takeProfit) {
        return {
          shouldClose: true,
          reason: 'Take profit reached',
          urgency: 'normal'
        };
      }
    }

    // Check time-based stop (positions open too long)
    const positionAge = Date.now() - position.openedAt.getTime();
    const maxAge = 7 * 24 * 60 * 60 * 1000; // 7 days
    if (positionAge > maxAge) {
      return {
        shouldClose: true,
        reason: 'Position age exceeds 7 days',
        urgency: 'normal'
      };
    }

    // Check drawdown on position
    const positionDrawdown = position.unrealizedPnL < 0 ?
      Math.abs(position.unrealizedPnL / (position.quantity * position.entryPrice)) : 0;

    if (positionDrawdown > 0.2) { // 20% position drawdown
      return {
        shouldClose: true,
        reason: `Position drawdown ${(positionDrawdown * 100).toFixed(2)}% exceeds 20%`,
        urgency: 'immediate'
      };
    }

    return null;
  }

  private async getPortfolioValue(userId: string): Promise<number> {
    // This would integrate with the actual broker to get account value
    // For now, return a default value
    return 10000;
  }

  private async getCurrentExposure(userId: string): Promise<number> {
    const positions = await this.positionManager.getOpenPositions(userId);
    let totalExposure = 0;

    for (const position of positions) {
      totalExposure += position.quantity * position.currentPrice;
    }

    return totalExposure;
  }

  private async calculateMarketVolatility(): Promise<number> {
    // Simplified volatility calculation
    // In production, this would use actual market data
    return Math.random() * 0.03; // Random value between 0-3%
  }

  private async calculateCorrelations(symbols: string[]): Promise<number[]> {
    // Simplified correlation calculation
    // In production, this would use actual price correlation matrices
    const correlations: number[] = [];

    for (let i = 0; i < symbols.length; i++) {
      for (let j = i + 1; j < symbols.length; j++) {
        // Mock correlation value
        correlations.push(Math.random() * 0.5);
      }
    }

    return correlations;
  }

  updateDailyPnL(userId: string, pnl: number): void {
    this.dailyPnL.set(userId, pnl);
  }

  updateConsecutiveLosses(userId: string, isWin: boolean): void {
    if (isWin) {
      this.consecutiveLosses.set(userId, 0);
    } else {
      const current = this.consecutiveLosses.get(userId) || 0;
      this.consecutiveLosses.set(userId, current + 1);
    }
  }

  private startRiskMonitoring(): void {
    // Monitor risk metrics every 30 seconds
    setInterval(async () => {
      try {
        // Update market volatility
        this.marketVolatility = await this.calculateMarketVolatility();

        // Emit risk metrics
        this.emit('riskMetricsUpdate', {
          marketVolatility: this.marketVolatility,
          timestamp: new Date()
        });

      } catch (error) {
        logger.error('Risk monitoring error:', error);
      }
    }, 30000);
  }

  async emergencyStopLoss(userId: string): Promise<void> {
    logger.error(`Emergency stop loss triggered for user ${userId}`);

    // Close all positions immediately
    const positions = await this.positionManager.getOpenPositions(userId);
    for (const position of positions) {
      this.emit('emergencyClose', {
        userId,
        positionId: position.id,
        reason: 'Emergency stop loss'
      });
    }

    // Emit emergency stop event
    this.emit('emergencyStop', userId);
  }

  async getRiskMetrics(userId: string): Promise<any> {
    const positions = await this.positionManager.getOpenPositions(userId);
    const portfolioValue = await this.getPortfolioValue(userId);
    const currentExposure = await this.getCurrentExposure(userId);

    const metrics = {
      portfolioValue,
      currentExposure,
      exposureRatio: portfolioValue > 0 ? currentExposure / portfolioValue : 0,
      openPositions: positions.length,
      marketVolatility: this.marketVolatility,
      dailyPnL: this.dailyPnL.get(userId) || 0,
      consecutiveLosses: this.consecutiveLosses.get(userId) || 0,
      riskScore: this.calculateRiskScore(userId, positions)
    };

    return metrics;
  }

  private calculateRiskScore(userId: string, positions: Position[]): number {
    // Risk score from 0 (lowest) to 100 (highest)
    let score = 0;

    // Factor in number of positions
    score += Math.min(positions.length * 5, 25); // Max 25 points

    // Factor in exposure
    const exposurePoints = positions.length > 0 ? 30 : 0; // Max 30 points
    score += exposurePoints;

    // Factor in consecutive losses
    const losses = this.consecutiveLosses.get(userId) || 0;
    score += losses * 10; // 10 points per loss

    // Factor in market volatility
    score += this.marketVolatility * 500; // Max ~25 points at 5% volatility

    return Math.min(100, Math.max(0, score));
  }

  cleanup(): void {
    this.dailyPnL.clear();
    this.consecutiveLosses.clear();
  }
}

export default RiskManager;