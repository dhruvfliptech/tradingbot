import { EventEmitter } from 'events';
import { DatabaseService } from '../database/DatabaseService';
import { BacktestResult, Trade } from '../composer/ComposerService';
import logger from '../../utils/logger';

export interface PerformanceMetrics {
  // Return Metrics
  totalReturn: number;
  annualizedReturn: number;
  cumulativeReturn: number;
  
  // Risk Metrics
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  maxDrawdown: number;
  maxDrawdownDuration: number;
  volatility: number;
  downsideDeviation: number;
  
  // Trade Metrics
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  profitFactor: number;
  avgWin: number;
  avgLoss: number;
  largestWin: number;
  largestLoss: number;
  avgTradeDuration: number;
  
  // Risk Analytics
  var95: number; // Value at Risk 95%
  var99: number; // Value at Risk 99%
  expectedShortfall: number;
  beta: number;
  alpha: number;
  trackingError: number;
  informationRatio: number;
  
  // Advanced Metrics
  omegaRatio: number;
  kurtosis: number;
  skewness: number;
  tailRatio: number;
  
  // Streak Analytics
  maxWinStreak: number;
  maxLossStreak: number;
  currentStreak: number;
  streakType: 'win' | 'loss' | 'none';
  
  // Time-based Analytics
  monthlyReturns: Record<string, number>;
  yearlyReturns: Record<string, number>;
  bestMonth: { month: string; return: number };
  worstMonth: { month: string; return: number };
  
  // Regime Performance
  regimePerformance?: Record<string, {
    return: number;
    sharpe: number;
    maxDrawdown: number;
    trades: number;
  }>;
}

export interface DrawdownPeriod {
  start: Date;
  end: Date;
  duration: number; // in days
  maxDrawdown: number;
  recovery: Date | null;
  recovered: boolean;
}

export interface EquityPoint {
  date: Date;
  equity: number;
  drawdown: number;
  returns: number;
}

export interface PerformanceReport {
  summary: PerformanceMetrics;
  equityCurve: EquityPoint[];
  drawdownPeriods: DrawdownPeriod[];
  tradeAnalysis: TradeAnalysis;
  riskAnalysis: RiskAnalysis;
  regimeAnalysis?: RegimeAnalysis;
}

export interface TradeAnalysis {
  trades: Trade[];
  winningTrades: Trade[];
  losingTrades: Trade[];
  tradesBySymbol: Record<string, Trade[]>;
  tradesByTimeOfDay: Record<string, Trade[]>;
  tradesByDayOfWeek: Record<string, Trade[]>;
  averageHoldTime: number;
  profitFactorBySymbol: Record<string, number>;
  winRateBySymbol: Record<string, number>;
}

export interface RiskAnalysis {
  correlationMatrix: Record<string, Record<string, number>>;
  positionSizes: number[];
  leverageAnalysis: {
    maxLeverage: number;
    avgLeverage: number;
    leverageDistribution: Record<string, number>;
  };
  exposureAnalysis: {
    maxExposure: number;
    avgExposure: number;
    exposureByAsset: Record<string, number>;
  };
}

export interface RegimeAnalysis {
  regimes: Array<{
    period: { start: Date; end: Date };
    type: 'bull' | 'bear' | 'sideways' | 'volatile';
    performance: PerformanceMetrics;
    characteristics: {
      volatility: number;
      trend: number;
      momentum: number;
    };
  }>;
  bestRegime: string;
  worstRegime: string;
  consistencyScore: number;
}

export class PerformanceMetricsService extends EventEmitter {
  private databaseService: DatabaseService;

  constructor() {
    super();
    this.databaseService = new DatabaseService();
  }

  /**
   * Calculate comprehensive performance metrics from backtest result
   */
  async calculatePerformanceMetrics(backtestResult: BacktestResult): Promise<PerformanceMetrics> {
    try {
      const trades = backtestResult.trades;
      if (!trades || trades.length === 0) {
        throw new Error('No trades available for analysis');
      }

      // Calculate basic return metrics
      const returns = this.calculateReturns(trades);
      const totalReturn = this.calculateTotalReturn(returns);
      const annualizedReturn = this.calculateAnnualizedReturn(returns, this.getTradingDays(trades));
      
      // Calculate risk metrics
      const volatility = this.calculateVolatility(returns);
      const sharpeRatio = this.calculateSharpeRatio(returns, 0.02); // 2% risk-free rate
      const sortinoRatio = this.calculateSortinoRatio(returns, 0.02);
      const maxDrawdown = this.calculateMaxDrawdown(trades);
      const downsideDeviation = this.calculateDownsideDeviation(returns, 0);

      // Calculate trade metrics
      const winningTrades = trades.filter(t => (t.pnl || 0) > 0);
      const losingTrades = trades.filter(t => (t.pnl || 0) < 0);
      const winRate = winningTrades.length / trades.length;
      const avgWin = winningTrades.length > 0 ? 
        winningTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / winningTrades.length : 0;
      const avgLoss = losingTrades.length > 0 ? 
        Math.abs(losingTrades.reduce((sum, t) => sum + (t.pnl || 0), 0) / losingTrades.length) : 0;
      const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0;

      // Calculate advanced metrics
      const var95 = this.calculateVaR(returns, 0.95);
      const var99 = this.calculateVaR(returns, 0.99);
      const expectedShortfall = this.calculateExpectedShortfall(returns, 0.95);
      const omegaRatio = this.calculateOmegaRatio(returns, 0);
      const kurtosis = this.calculateKurtosis(returns);
      const skewness = this.calculateSkewness(returns);

      // Calculate time-based metrics
      const monthlyReturns = this.calculateMonthlyReturns(trades);
      const yearlyReturns = this.calculateYearlyReturns(trades);
      const bestMonth = this.getBestMonth(monthlyReturns);
      const worstMonth = this.getWorstMonth(monthlyReturns);

      // Calculate streak analytics
      const streakAnalysis = this.calculateStreakAnalysis(trades);

      const metrics: PerformanceMetrics = {
        totalReturn,
        annualizedReturn,
        cumulativeReturn: totalReturn,
        sharpeRatio,
        sortinoRatio,
        calmarRatio: annualizedReturn / Math.abs(maxDrawdown),
        maxDrawdown,
        maxDrawdownDuration: this.calculateMaxDrawdownDuration(trades),
        volatility,
        downsideDeviation,
        totalTrades: trades.length,
        winningTrades: winningTrades.length,
        losingTrades: losingTrades.length,
        winRate,
        profitFactor,
        avgWin,
        avgLoss,
        largestWin: Math.max(...trades.map(t => t.pnl || 0)),
        largestLoss: Math.min(...trades.map(t => t.pnl || 0)),
        avgTradeDuration: this.calculateAvgTradeDuration(trades),
        var95,
        var99,
        expectedShortfall,
        beta: 0, // Would need market benchmark data
        alpha: 0, // Would need market benchmark data
        trackingError: 0, // Would need benchmark data
        informationRatio: 0, // Would need benchmark data
        omegaRatio,
        kurtosis,
        skewness,
        tailRatio: Math.abs(var95 / var99),
        maxWinStreak: streakAnalysis.maxWinStreak,
        maxLossStreak: streakAnalysis.maxLossStreak,
        currentStreak: streakAnalysis.currentStreak,
        streakType: streakAnalysis.streakType,
        monthlyReturns,
        yearlyReturns,
        bestMonth,
        worstMonth
      };

      return metrics;

    } catch (error) {
      logger.error('Error calculating performance metrics:', error);
      throw error;
    }
  }

  /**
   * Generate comprehensive performance report
   */
  async generatePerformanceReport(backtestResult: BacktestResult): Promise<PerformanceReport> {
    try {
      const metrics = await this.calculatePerformanceMetrics(backtestResult);
      const equityCurve = this.calculateEquityCurve(backtestResult.trades);
      const drawdownPeriods = this.calculateDrawdownPeriods(equityCurve);
      const tradeAnalysis = this.analyzeTradePatterns(backtestResult.trades);
      const riskAnalysis = this.analyzeRisk(backtestResult.trades);

      return {
        summary: metrics,
        equityCurve,
        drawdownPeriods,
        tradeAnalysis,
        riskAnalysis
      };

    } catch (error) {
      logger.error('Error generating performance report:', error);
      throw error;
    }
  }

  /**
   * Save performance metrics to database
   */
  async savePerformanceMetrics(
    userId: string,
    backtestId: string,
    metrics: PerformanceMetrics
  ): Promise<void> {
    try {
      // Update backtest result with calculated metrics
      await this.databaseService.updateBacktestResult(backtestId, {
        totalReturn: metrics.totalReturn,
        sharpeRatio: metrics.sharpeRatio,
        sortinoRatio: metrics.sortinoRatio,
        calmarRatio: metrics.calmarRatio,
        maxDrawdown: metrics.maxDrawdown,
        winRate: metrics.winRate,
        profitFactor: metrics.profitFactor,
        volatility: metrics.volatility,
        var95: metrics.var95,
        var99: metrics.var99,
        expectedShortfall: metrics.expectedShortfall,
        detailedMetrics: metrics,
        monthlyReturns: metrics.monthlyReturns
      });

      logger.info(`Performance metrics saved for backtest ${backtestId}`);

    } catch (error) {
      logger.error('Error saving performance metrics:', error);
      throw error;
    }
  }

  // Private calculation methods

  private calculateReturns(trades: Trade[]): number[] {
    return trades.map(trade => (trade.pnlPercent || 0) / 100);
  }

  private calculateTotalReturn(returns: number[]): number {
    return returns.reduce((total, ret) => total * (1 + ret), 1) - 1;
  }

  private calculateAnnualizedReturn(returns: number[], tradingDays: number): number {
    const totalReturn = this.calculateTotalReturn(returns);
    const years = tradingDays / 252; // Assuming 252 trading days per year
    return Math.pow(1 + totalReturn, 1 / years) - 1;
  }

  private calculateVolatility(returns: number[]): number {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    return Math.sqrt(variance * 252); // Annualized
  }

  private calculateSharpeRatio(returns: number[], riskFreeRate: number): number {
    const excessReturns = returns.map(ret => ret - riskFreeRate / 252);
    const avgExcessReturn = excessReturns.reduce((sum, ret) => sum + ret, 0) / excessReturns.length;
    const volatility = this.calculateVolatility(excessReturns);
    return volatility > 0 ? (avgExcessReturn * 252) / volatility : 0;
  }

  private calculateSortinoRatio(returns: number[], targetReturn: number): number {
    const excessReturns = returns.map(ret => ret - targetReturn / 252);
    const avgExcessReturn = excessReturns.reduce((sum, ret) => sum + ret, 0) / excessReturns.length;
    const downsideDeviation = this.calculateDownsideDeviation(returns, targetReturn / 252);
    return downsideDeviation > 0 ? (avgExcessReturn * 252) / downsideDeviation : 0;
  }

  private calculateDownsideDeviation(returns: number[], targetReturn: number): number {
    const downsideReturns = returns.filter(ret => ret < targetReturn);
    if (downsideReturns.length === 0) return 0;
    
    const mean = targetReturn;
    const variance = downsideReturns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    return Math.sqrt(variance * 252);
  }

  private calculateMaxDrawdown(trades: Trade[]): number {
    const equityCurve = this.calculateEquityCurve(trades);
    let maxDrawdown = 0;
    let peak = equityCurve[0]?.equity || 0;

    for (const point of equityCurve) {
      if (point.equity > peak) {
        peak = point.equity;
      }
      const drawdown = (peak - point.equity) / peak;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }

    return maxDrawdown;
  }

  private calculateMaxDrawdownDuration(trades: Trade[]): number {
    const equityCurve = this.calculateEquityCurve(trades);
    const drawdownPeriods = this.calculateDrawdownPeriods(equityCurve);
    
    if (drawdownPeriods.length === 0) return 0;
    
    return Math.max(...drawdownPeriods.map(period => period.duration));
  }

  private calculateVaR(returns: number[], confidence: number): number {
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidence) * sortedReturns.length);
    return sortedReturns[index] || 0;
  }

  private calculateExpectedShortfall(returns: number[], confidence: number): number {
    const varValue = this.calculateVaR(returns, confidence);
    const tailReturns = returns.filter(ret => ret <= varValue);
    return tailReturns.length > 0 ?
      tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length : 0;
  }

  private calculateOmegaRatio(returns: number[], threshold: number): number {
    const gainsSum = returns.filter(ret => ret > threshold).reduce((sum, ret) => sum + (ret - threshold), 0);
    const lossesSum = returns.filter(ret => ret < threshold).reduce((sum, ret) => sum + Math.abs(ret - threshold), 0);
    return lossesSum > 0 ? gainsSum / lossesSum : 0;
  }

  private calculateKurtosis(returns: number[]): number {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    const fourthMoment = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 4), 0) / returns.length;
    return variance > 0 ? fourthMoment / Math.pow(variance, 2) - 3 : 0;
  }

  private calculateSkewness(returns: number[]): number {
    const mean = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 2), 0) / returns.length;
    const thirdMoment = returns.reduce((sum, ret) => sum + Math.pow(ret - mean, 3), 0) / returns.length;
    return variance > 0 ? thirdMoment / Math.pow(variance, 1.5) : 0;
  }

  private calculateEquityCurve(trades: Trade[]): EquityPoint[] {
    const curve: EquityPoint[] = [];
    let equity = 10000; // Starting capital
    let peak = equity;

    for (const trade of trades.sort((a, b) => new Date(a.entryTime).getTime() - new Date(b.entryTime).getTime())) {
      equity += trade.pnl || 0;
      
      if (equity > peak) {
        peak = equity;
      }
      
      const drawdown = (peak - equity) / peak;
      const returns = trade.pnlPercent ? trade.pnlPercent / 100 : 0;

      curve.push({
        date: new Date(trade.exitTime || trade.entryTime),
        equity,
        drawdown,
        returns
      });
    }

    return curve;
  }

  private calculateDrawdownPeriods(equityCurve: EquityPoint[]): DrawdownPeriod[] {
    const periods: DrawdownPeriod[] = [];
    let inDrawdown = false;
    let drawdownStart: Date | null = null;
    let maxDrawdownInPeriod = 0;

    for (let i = 0; i < equityCurve.length; i++) {
      const point = equityCurve[i];
      
      if (point.drawdown > 0.01 && !inDrawdown) {
        // Start of drawdown period
        inDrawdown = true;
        drawdownStart = point.date;
        maxDrawdownInPeriod = point.drawdown;
      } else if (point.drawdown > maxDrawdownInPeriod) {
        maxDrawdownInPeriod = point.drawdown;
      } else if (point.drawdown === 0 && inDrawdown) {
        // End of drawdown period
        if (drawdownStart) {
          const duration = (point.date.getTime() - drawdownStart.getTime()) / (1000 * 60 * 60 * 24);
          periods.push({
            start: drawdownStart,
            end: point.date,
            duration,
            maxDrawdown: maxDrawdownInPeriod,
            recovery: point.date,
            recovered: true
          });
        }
        inDrawdown = false;
        drawdownStart = null;
        maxDrawdownInPeriod = 0;
      }
    }

    // Handle ongoing drawdown
    if (inDrawdown && drawdownStart) {
      const lastPoint = equityCurve[equityCurve.length - 1];
      const duration = (lastPoint.date.getTime() - drawdownStart.getTime()) / (1000 * 60 * 60 * 24);
      periods.push({
        start: drawdownStart,
        end: lastPoint.date,
        duration,
        maxDrawdown: maxDrawdownInPeriod,
        recovery: null,
        recovered: false
      });
    }

    return periods;
  }

  private analyzeTradePatterns(trades: Trade[]): TradeAnalysis {
    const winningTrades = trades.filter(t => (t.pnl || 0) > 0);
    const losingTrades = trades.filter(t => (t.pnl || 0) < 0);
    
    // Group trades by symbol
    const tradesBySymbol = trades.reduce((groups, trade) => {
      if (!groups[trade.symbol]) groups[trade.symbol] = [];
      groups[trade.symbol].push(trade);
      return groups;
    }, {} as Record<string, Trade[]>);

    // Group trades by time of day
    const tradesByTimeOfDay = trades.reduce((groups, trade) => {
      const hour = new Date(trade.entryTime).getHours();
      const timeSlot = `${hour}:00`;
      if (!groups[timeSlot]) groups[timeSlot] = [];
      groups[timeSlot].push(trade);
      return groups;
    }, {} as Record<string, Trade[]>);

    // Group trades by day of week
    const tradesByDayOfWeek = trades.reduce((groups, trade) => {
      const dayNames = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'];
      const day = dayNames[new Date(trade.entryTime).getDay()];
      if (!groups[day]) groups[day] = [];
      groups[day].push(trade);
      return groups;
    }, {} as Record<string, Trade[]>);

    // Calculate profit factor and win rate by symbol
    const profitFactorBySymbol: Record<string, number> = {};
    const winRateBySymbol: Record<string, number> = {};

    for (const [symbol, symbolTrades] of Object.entries(tradesBySymbol)) {
      const wins = symbolTrades.filter(t => (t.pnl || 0) > 0);
      const losses = symbolTrades.filter(t => (t.pnl || 0) < 0);
      
      winRateBySymbol[symbol] = wins.length / symbolTrades.length;
      
      const avgWin = wins.length > 0 ? wins.reduce((sum, t) => sum + (t.pnl || 0), 0) / wins.length : 0;
      const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((sum, t) => sum + (t.pnl || 0), 0) / losses.length) : 0;
      profitFactorBySymbol[symbol] = avgLoss > 0 ? avgWin / avgLoss : 0;
    }

    return {
      trades,
      winningTrades,
      losingTrades,
      tradesBySymbol,
      tradesByTimeOfDay,
      tradesByDayOfWeek,
      averageHoldTime: this.calculateAvgTradeDuration(trades),
      profitFactorBySymbol,
      winRateBySymbol
    };
  }

  private analyzeRisk(trades: Trade[]): RiskAnalysis {
    // Calculate position sizes
    const positionSizes = trades.map(t => t.quantity);
    
    // Group by symbol for exposure analysis
    const exposureByAsset: Record<string, number> = {};
    for (const trade of trades) {
      if (!exposureByAsset[trade.symbol]) exposureByAsset[trade.symbol] = 0;
      exposureByAsset[trade.symbol] += Math.abs(trade.quantity * trade.entryPrice);
    }

    return {
      correlationMatrix: {}, // Would need price correlation data
      positionSizes,
      leverageAnalysis: {
        maxLeverage: 1, // Placeholder
        avgLeverage: 1, // Placeholder
        leverageDistribution: { '1x': 100 }
      },
      exposureAnalysis: {
        maxExposure: Math.max(...Object.values(exposureByAsset)),
        avgExposure: Object.values(exposureByAsset).reduce((sum, exp) => sum + exp, 0) / Object.keys(exposureByAsset).length,
        exposureByAsset
      }
    };
  }

  private calculateMonthlyReturns(trades: Trade[]): Record<string, number> {
    const monthlyReturns: Record<string, number> = {};
    
    for (const trade of trades) {
      const date = new Date(trade.exitTime || trade.entryTime);
      const monthKey = `${date.getFullYear()}-${(date.getMonth() + 1).toString().padStart(2, '0')}`;
      
      if (!monthlyReturns[monthKey]) monthlyReturns[monthKey] = 0;
      monthlyReturns[monthKey] += (trade.pnlPercent || 0) / 100;
    }
    
    return monthlyReturns;
  }

  private calculateYearlyReturns(trades: Trade[]): Record<string, number> {
    const yearlyReturns: Record<string, number> = {};
    
    for (const trade of trades) {
      const year = new Date(trade.exitTime || trade.entryTime).getFullYear().toString();
      
      if (!yearlyReturns[year]) yearlyReturns[year] = 0;
      yearlyReturns[year] += (trade.pnlPercent || 0) / 100;
    }
    
    return yearlyReturns;
  }

  private getBestMonth(monthlyReturns: Record<string, number>): { month: string; return: number } {
    const entries = Object.entries(monthlyReturns);
    if (entries.length === 0) return { month: '', return: 0 };
    
    const best = entries.reduce((best, current) => current[1] > best[1] ? current : best);
    return { month: best[0], return: best[1] };
  }

  private getWorstMonth(monthlyReturns: Record<string, number>): { month: string; return: number } {
    const entries = Object.entries(monthlyReturns);
    if (entries.length === 0) return { month: '', return: 0 };
    
    const worst = entries.reduce((worst, current) => current[1] < worst[1] ? current : worst);
    return { month: worst[0], return: worst[1] };
  }

  private calculateStreakAnalysis(trades: Trade[]): {
    maxWinStreak: number;
    maxLossStreak: number;
    currentStreak: number;
    streakType: 'win' | 'loss' | 'none';
  } {
    let maxWinStreak = 0;
    let maxLossStreak = 0;
    let currentWinStreak = 0;
    let currentLossStreak = 0;
    let currentStreak = 0;
    let streakType: 'win' | 'loss' | 'none' = 'none';

    for (const trade of trades.sort((a, b) => new Date(a.entryTime).getTime() - new Date(b.entryTime).getTime())) {
      const isWin = (trade.pnl || 0) > 0;
      
      if (isWin) {
        currentWinStreak++;
        currentLossStreak = 0;
        maxWinStreak = Math.max(maxWinStreak, currentWinStreak);
      } else {
        currentLossStreak++;
        currentWinStreak = 0;
        maxLossStreak = Math.max(maxLossStreak, currentLossStreak);
      }
    }

    // Determine current streak
    if (currentWinStreak > 0) {
      currentStreak = currentWinStreak;
      streakType = 'win';
    } else if (currentLossStreak > 0) {
      currentStreak = currentLossStreak;
      streakType = 'loss';
    }

    return {
      maxWinStreak,
      maxLossStreak,
      currentStreak,
      streakType
    };
  }

  private calculateAvgTradeDuration(trades: Trade[]): number {
    const durations = trades
      .filter(t => t.exitTime && t.entryTime)
      .map(t => new Date(t.exitTime!).getTime() - new Date(t.entryTime).getTime());
    
    if (durations.length === 0) return 0;
    
    const avgDurationMs = durations.reduce((sum, duration) => sum + duration, 0) / durations.length;
    return avgDurationMs / (1000 * 60 * 60); // Convert to hours
  }

  private getTradingDays(trades: Trade[]): number {
    if (trades.length === 0) return 0;
    
    const firstTrade = trades.sort((a, b) => new Date(a.entryTime).getTime() - new Date(b.entryTime).getTime())[0];
    const lastTrade = trades.sort((a, b) => new Date(b.entryTime).getTime() - new Date(a.entryTime).getTime())[0];
    
    const daysDiff = (new Date(lastTrade.entryTime).getTime() - new Date(firstTrade.entryTime).getTime()) / (1000 * 60 * 60 * 24);
    return Math.max(1, daysDiff);
  }
}