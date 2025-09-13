/**
 * Portfolio Analytics Service
 * Calculates advanced portfolio metrics including Sharpe Ratio, Sortino, Calmar, etc.
 */

import { virtualPortfolioService } from './persistence/virtualPortfolioService';
import { tradeHistoryService } from './persistence/tradeHistoryService';

export interface PortfolioMetrics {
  // Basic Metrics
  totalValue: number;
  totalReturn: number;
  totalReturnPercent: number;
  
  // Risk-Adjusted Returns
  sharpeRatio: number;
  sortinoRatio: number;
  calmarRatio: number;
  
  // Risk Metrics
  volatility: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  currentDrawdown: number;
  currentDrawdownPercent: number;
  downsideDeviation: number;
  
  // Performance Metrics
  winRate: number;
  profitFactor: number;
  averageWin: number;
  averageLoss: number;
  largestWin: number;
  largestLoss: number;
  
  // Trading Activity
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  averageTradeDuration: number; // in hours
  tradesPerDay: number;
  
  // Efficiency Metrics
  expectancy: number;
  payoffRatio: number;
  recoveryFactor: number;
  
  // Time-based Returns
  dailyReturn: number;
  weeklyReturn: number;
  monthlyReturn: number;
  yearlyReturn: number;
  
  // Risk/Reward
  averageRiskRewardRatio: number;
  actualRiskRewardRatio: number;
}

export interface PerformanceSnapshot {
  date: string;
  value: number;
  dailyReturn: number;
  cumulativeReturn: number;
}

class PortfolioAnalyticsService {
  private readonly RISK_FREE_RATE = 0.05; // 5% annual risk-free rate
  private readonly INITIAL_CAPITAL = 50000;
  
  /**
   * Calculate all portfolio metrics
   */
  async calculateMetrics(days: number = 30): Promise<PortfolioMetrics> {
    try {
      // Get portfolio history
      const snapshots = await virtualPortfolioService.getDailySnapshots(days);
      const trades = await tradeHistoryService.getRecentTrades(1000);
      const currentPortfolio = await virtualPortfolioService.getPortfolio();
      
      // Basic metrics
      const totalValue = currentPortfolio?.total_value || this.INITIAL_CAPITAL;
      const totalReturn = totalValue - this.INITIAL_CAPITAL;
      const totalReturnPercent = (totalReturn / this.INITIAL_CAPITAL) * 100;
      
      // Calculate daily returns
      const dailyReturns: number[] = [];
      for (let i = 1; i < snapshots.length; i++) {
        const dailyReturn = ((snapshots[i].total_value - snapshots[i - 1].total_value) / snapshots[i - 1].total_value) * 100;
        dailyReturns.push(dailyReturn);
      }
      
      // Risk metrics
      const volatility = this.calculateVolatility(dailyReturns);
      const { maxDrawdown, maxDrawdownPercent, currentDrawdown, currentDrawdownPercent } = 
        this.calculateDrawdown(snapshots);
      const downsideDeviation = this.calculateDownsideDeviation(dailyReturns);
      
      // Risk-adjusted returns
      const sharpeRatio = this.calculateSharpeRatio(dailyReturns, volatility);
      const sortinoRatio = this.calculateSortinoRatio(dailyReturns, downsideDeviation);
      const calmarRatio = this.calculateCalmarRatio(totalReturnPercent, maxDrawdownPercent);
      
      // Trade analysis
      const tradeAnalysis = this.analyzeeTrades(trades);
      
      // Time-based returns
      const timeReturns = this.calculateTimeBasedReturns(snapshots);
      
      // Efficiency metrics
      const expectancy = this.calculateExpectancy(tradeAnalysis);
      const payoffRatio = tradeAnalysis.averageWin / Math.abs(tradeAnalysis.averageLoss || 1);
      const recoveryFactor = totalReturn / Math.abs(maxDrawdown || 1);
      
      return {
        // Basic
        totalValue,
        totalReturn,
        totalReturnPercent,
        
        // Risk-adjusted
        sharpeRatio,
        sortinoRatio,
        calmarRatio,
        
        // Risk
        volatility,
        maxDrawdown,
        maxDrawdownPercent,
        currentDrawdown,
        currentDrawdownPercent,
        downsideDeviation,
        
        // Performance
        ...tradeAnalysis,
        
        // Efficiency
        expectancy,
        payoffRatio,
        recoveryFactor,
        
        // Time-based
        ...timeReturns
      };
    } catch (error) {
      console.error('Error calculating portfolio metrics:', error);
      return this.getDefaultMetrics();
    }
  }

  /**
   * Calculate Sharpe Ratio
   * (Return - Risk Free Rate) / Standard Deviation
   */
  private calculateSharpeRatio(dailyReturns: number[], volatility: number): number {
    if (dailyReturns.length === 0 || volatility === 0) return 0;
    
    const avgDailyReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
    const annualizedReturn = avgDailyReturn * 252; // 252 trading days
    const annualizedVolatility = volatility * Math.sqrt(252);
    
    if (annualizedVolatility === 0) return 0;
    
    const sharpe = (annualizedReturn - this.RISK_FREE_RATE) / annualizedVolatility;
    return Math.round(sharpe * 100) / 100;
  }

  /**
   * Calculate Sortino Ratio
   * (Return - Risk Free Rate) / Downside Deviation
   */
  private calculateSortinoRatio(dailyReturns: number[], downsideDeviation: number): number {
    if (dailyReturns.length === 0 || downsideDeviation === 0) return 0;
    
    const avgDailyReturn = dailyReturns.reduce((a, b) => a + b, 0) / dailyReturns.length;
    const annualizedReturn = avgDailyReturn * 252;
    const annualizedDownside = downsideDeviation * Math.sqrt(252);
    
    if (annualizedDownside === 0) return 0;
    
    const sortino = (annualizedReturn - this.RISK_FREE_RATE) / annualizedDownside;
    return Math.round(sortino * 100) / 100;
  }

  /**
   * Calculate Calmar Ratio
   * Annual Return / Max Drawdown
   */
  private calculateCalmarRatio(totalReturnPercent: number, maxDrawdownPercent: number): number {
    if (maxDrawdownPercent === 0) return 0;
    
    // Annualize the return (assuming the period is less than a year)
    const annualizedReturn = totalReturnPercent; // Simplified
    const calmar = annualizedReturn / Math.abs(maxDrawdownPercent);
    return Math.round(calmar * 100) / 100;
  }

  /**
   * Calculate volatility (standard deviation of returns)
   */
  private calculateVolatility(returns: number[]): number {
    if (returns.length < 2) return 0;
    
    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const squaredDiffs = returns.map(r => Math.pow(r - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / returns.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Calculate downside deviation (volatility of negative returns only)
   */
  private calculateDownsideDeviation(returns: number[]): number {
    const negativeReturns = returns.filter(r => r < 0);
    if (negativeReturns.length < 2) return 0;
    
    const mean = negativeReturns.reduce((a, b) => a + b, 0) / negativeReturns.length;
    const squaredDiffs = negativeReturns.map(r => Math.pow(r - mean, 2));
    const variance = squaredDiffs.reduce((a, b) => a + b, 0) / negativeReturns.length;
    
    return Math.sqrt(variance);
  }

  /**
   * Calculate drawdown metrics
   */
  private calculateDrawdown(snapshots: any[]): {
    maxDrawdown: number;
    maxDrawdownPercent: number;
    currentDrawdown: number;
    currentDrawdownPercent: number;
  } {
    if (snapshots.length === 0) {
      return { maxDrawdown: 0, maxDrawdownPercent: 0, currentDrawdown: 0, currentDrawdownPercent: 0 };
    }

    let peak = snapshots[0].total_value;
    let maxDrawdown = 0;
    let maxDrawdownPercent = 0;
    
    for (const snapshot of snapshots) {
      if (snapshot.total_value > peak) {
        peak = snapshot.total_value;
      }
      
      const drawdown = peak - snapshot.total_value;
      const drawdownPercent = (drawdown / peak) * 100;
      
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
        maxDrawdownPercent = drawdownPercent;
      }
    }
    
    // Current drawdown
    const currentValue = snapshots[snapshots.length - 1].total_value;
    const currentDrawdown = peak - currentValue;
    const currentDrawdownPercent = (currentDrawdown / peak) * 100;
    
    return {
      maxDrawdown,
      maxDrawdownPercent: Math.round(maxDrawdownPercent * 100) / 100,
      currentDrawdown,
      currentDrawdownPercent: Math.round(currentDrawdownPercent * 100) / 100
    };
  }

  /**
   * Analyze trades for performance metrics
   */
  private analyzeeTrades(trades: any[]): {
    winRate: number;
    profitFactor: number;
    averageWin: number;
    averageLoss: number;
    largestWin: number;
    largestLoss: number;
    totalTrades: number;
    winningTrades: number;
    losingTrades: number;
    averageTradeDuration: number;
    tradesPerDay: number;
    averageRiskRewardRatio: number;
    actualRiskRewardRatio: number;
  } {
    if (trades.length === 0) {
      return {
        winRate: 0,
        profitFactor: 0,
        averageWin: 0,
        averageLoss: 0,
        largestWin: 0,
        largestLoss: 0,
        totalTrades: 0,
        winningTrades: 0,
        losingTrades: 0,
        averageTradeDuration: 0,
        tradesPerDay: 0,
        averageRiskRewardRatio: 0,
        actualRiskRewardRatio: 0
      };
    }

    const completedTrades = trades.filter(t => t.exit_price && t.pnl !== null);
    const winningTrades = completedTrades.filter(t => t.pnl > 0);
    const losingTrades = completedTrades.filter(t => t.pnl < 0);
    
    const totalWins = winningTrades.reduce((sum, t) => sum + t.pnl, 0);
    const totalLosses = Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0));
    
    const winRate = completedTrades.length > 0 
      ? (winningTrades.length / completedTrades.length) * 100 
      : 0;
    
    const profitFactor = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0;
    
    const averageWin = winningTrades.length > 0 
      ? totalWins / winningTrades.length 
      : 0;
    
    const averageLoss = losingTrades.length > 0 
      ? totalLosses / losingTrades.length 
      : 0;
    
    const largestWin = winningTrades.length > 0 
      ? Math.max(...winningTrades.map(t => t.pnl)) 
      : 0;
    
    const largestLoss = losingTrades.length > 0 
      ? Math.min(...losingTrades.map(t => t.pnl)) 
      : 0;
    
    // Calculate average trade duration
    let totalDuration = 0;
    let durationCount = 0;
    
    for (const trade of completedTrades) {
      if (trade.entry_time && trade.exit_time) {
        const duration = new Date(trade.exit_time).getTime() - new Date(trade.entry_time).getTime();
        totalDuration += duration;
        durationCount++;
      }
    }
    
    const averageTradeDuration = durationCount > 0 
      ? totalDuration / durationCount / (1000 * 60 * 60) // Convert to hours
      : 0;
    
    // Calculate trades per day
    if (trades.length > 0) {
      const firstTrade = new Date(trades[trades.length - 1].created_at);
      const lastTrade = new Date(trades[0].created_at);
      const daysDiff = Math.max(1, (lastTrade.getTime() - firstTrade.getTime()) / (1000 * 60 * 60 * 24));
      const tradesPerDay = trades.length / daysDiff;
      
      // Risk/Reward ratios
      const avgRiskReward = trades.reduce((sum, t) => sum + (t.risk_reward_ratio || 0), 0) / trades.length;
      const actualRiskReward = averageLoss > 0 ? averageWin / averageLoss : 0;
      
      return {
        winRate: Math.round(winRate * 100) / 100,
        profitFactor: Math.round(profitFactor * 100) / 100,
        averageWin,
        averageLoss,
        largestWin,
        largestLoss,
        totalTrades: trades.length,
        winningTrades: winningTrades.length,
        losingTrades: losingTrades.length,
        averageTradeDuration: Math.round(averageTradeDuration * 10) / 10,
        tradesPerDay: Math.round(tradesPerDay * 10) / 10,
        averageRiskRewardRatio: Math.round(avgRiskReward * 100) / 100,
        actualRiskRewardRatio: Math.round(actualRiskReward * 100) / 100
      };
    }
    
    return {
      winRate: 0,
      profitFactor: 0,
      averageWin: 0,
      averageLoss: 0,
      largestWin: 0,
      largestLoss: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      averageTradeDuration: 0,
      tradesPerDay: 0,
      averageRiskRewardRatio: 0,
      actualRiskRewardRatio: 0
    };
  }

  /**
   * Calculate expectancy (average expected profit per trade)
   */
  private calculateExpectancy(tradeAnalysis: any): number {
    const { winRate, averageWin, averageLoss } = tradeAnalysis;
    const winProbability = winRate / 100;
    const loseProbability = 1 - winProbability;
    
    const expectancy = (winProbability * averageWin) - (loseProbability * averageLoss);
    return Math.round(expectancy * 100) / 100;
  }

  /**
   * Calculate time-based returns
   */
  private calculateTimeBasedReturns(snapshots: any[]): {
    dailyReturn: number;
    weeklyReturn: number;
    monthlyReturn: number;
    yearlyReturn: number;
  } {
    if (snapshots.length < 2) {
      return { dailyReturn: 0, weeklyReturn: 0, monthlyReturn: 0, yearlyReturn: 0 };
    }

    // Daily return (last day)
    const dailyReturn = snapshots.length >= 2
      ? ((snapshots[snapshots.length - 1].total_value - snapshots[snapshots.length - 2].total_value) / 
         snapshots[snapshots.length - 2].total_value) * 100
      : 0;

    // Weekly return (last 7 days)
    const weekStart = Math.max(0, snapshots.length - 7);
    const weeklyReturn = ((snapshots[snapshots.length - 1].total_value - snapshots[weekStart].total_value) / 
                          snapshots[weekStart].total_value) * 100;

    // Monthly return (last 30 days)
    const monthStart = Math.max(0, snapshots.length - 30);
    const monthlyReturn = ((snapshots[snapshots.length - 1].total_value - snapshots[monthStart].total_value) / 
                           snapshots[monthStart].total_value) * 100;

    // Yearly return (annualized based on available data)
    const firstValue = snapshots[0].total_value;
    const lastValue = snapshots[snapshots.length - 1].total_value;
    const totalReturn = ((lastValue - firstValue) / firstValue);
    const daysElapsed = snapshots.length;
    const yearlyReturn = totalReturn * (365 / daysElapsed) * 100;

    return {
      dailyReturn: Math.round(dailyReturn * 100) / 100,
      weeklyReturn: Math.round(weeklyReturn * 100) / 100,
      monthlyReturn: Math.round(monthlyReturn * 100) / 100,
      yearlyReturn: Math.round(yearlyReturn * 100) / 100
    };
  }

  /**
   * Get default metrics when data is unavailable
   */
  private getDefaultMetrics(): PortfolioMetrics {
    return {
      totalValue: this.INITIAL_CAPITAL,
      totalReturn: 0,
      totalReturnPercent: 0,
      sharpeRatio: 0,
      sortinoRatio: 0,
      calmarRatio: 0,
      volatility: 0,
      maxDrawdown: 0,
      maxDrawdownPercent: 0,
      currentDrawdown: 0,
      currentDrawdownPercent: 0,
      downsideDeviation: 0,
      winRate: 0,
      profitFactor: 0,
      averageWin: 0,
      averageLoss: 0,
      largestWin: 0,
      largestLoss: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      averageTradeDuration: 0,
      tradesPerDay: 0,
      expectancy: 0,
      payoffRatio: 0,
      recoveryFactor: 0,
      dailyReturn: 0,
      weeklyReturn: 0,
      monthlyReturn: 0,
      yearlyReturn: 0,
      averageRiskRewardRatio: 0,
      actualRiskRewardRatio: 0
    };
  }

  /**
   * Get performance snapshots for charting
   */
  async getPerformanceHistory(days: number = 30): Promise<PerformanceSnapshot[]> {
    const snapshots = await virtualPortfolioService.getDailySnapshots(days);
    const history: PerformanceSnapshot[] = [];
    
    for (let i = 0; i < snapshots.length; i++) {
      const snapshot = snapshots[i];
      const dailyReturn = i > 0 
        ? ((snapshot.total_value - snapshots[i - 1].total_value) / snapshots[i - 1].total_value) * 100
        : 0;
      
      const cumulativeReturn = ((snapshot.total_value - this.INITIAL_CAPITAL) / this.INITIAL_CAPITAL) * 100;
      
      history.push({
        date: snapshot.snapshot_date,
        value: snapshot.total_value,
        dailyReturn,
        cumulativeReturn
      });
    }
    
    return history;
  }
}

// Export singleton instance
export const portfolioAnalytics = new PortfolioAnalyticsService();
export default portfolioAnalytics;