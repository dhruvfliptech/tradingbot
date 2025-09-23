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

  // Advanced Metrics
  maximumAdverseExcursion: number; // MAE - largest loss during winning trades
  maximumFavorableExcursion: number; // MFE - largest profit during losing trades
  averageMAE: number; // Average MAE across all trades
  averageMFE: number; // Average MFE across all trades
  rMultipleDistribution: { [key: string]: number }; // Distribution of R-multiples
  expectancy: number; // Mathematical expectancy
  kellyCriterion: number; // Optimal position size percentage
  efficiencyRatio: number; // How much of potential profit was captured
  tradeQuality: number; // Overall trade quality score (0-100)
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
      
      // Advanced metrics calculation
      const advancedMetrics = this.calculateAdvancedMetrics(trades);
      
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
        ...timeReturns,
        
        // Advanced metrics
        ...advancedMetrics
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
   * Calculate advanced trading metrics (MAE, MFE, R-multiples, etc.)
   */
  private calculateAdvancedMetrics(trades: any[]): {
    maximumAdverseExcursion: number;
    maximumFavorableExcursion: number;
    averageMAE: number;
    averageMFE: number;
    rMultipleDistribution: { [key: string]: number };
    expectancy: number;
    kellyCriterion: number;
    efficiencyRatio: number;
    tradeQuality: number;
  } {
    if (trades.length === 0) {
      return {
        maximumAdverseExcursion: 0,
        maximumFavorableExcursion: 0,
        averageMAE: 0,
        averageMFE: 0,
        rMultipleDistribution: {},
        expectancy: 0,
        kellyCriterion: 0,
        efficiencyRatio: 0,
        tradeQuality: 0
      };
    }

    const completedTrades = trades.filter(t => t.exit_price && t.pnl !== null);
    const winningTrades = completedTrades.filter(t => t.pnl > 0);
    const losingTrades = completedTrades.filter(t => t.pnl <= 0);

    // Calculate MAE and MFE
    let totalMAE = 0;
    let totalMFE = 0;
    let maxMAE = 0;
    let maxMFE = 0;
    const rMultiples: number[] = [];

    for (const trade of completedTrades) {
      // MAE: Maximum Adverse Excursion (stored in database or calculated)
      const mae = trade.mae || this.estimateMAE(trade);
      totalMAE += mae;
      maxMAE = Math.max(maxMAE, mae);

      // MFE: Maximum Favorable Excursion (stored in database or calculated)
      const mfe = trade.mfe || this.estimateMFE(trade);
      totalMFE += mfe;
      maxMFE = Math.max(maxMFE, mfe);

      // Calculate R-multiple if we have stop loss information
      if (trade.stop_loss || trade.risk_amount) {
        const rMultiple = this.calculateRMultiple(trade);
        if (rMultiple !== null) {
          rMultiples.push(rMultiple);
        }
      }
    }

    const averageMAE = completedTrades.length > 0 ? totalMAE / completedTrades.length : 0;
    const averageMFE = completedTrades.length > 0 ? totalMFE / completedTrades.length : 0;

    // R-multiple distribution
    const rMultipleDistribution = this.calculateRMultipleDistribution(rMultiples);

    // Mathematical expectancy
    const winRate = winningTrades.length / completedTrades.length;
    const avgWin = winningTrades.length > 0 ? winningTrades.reduce((sum, t) => sum + t.pnl, 0) / winningTrades.length : 0;
    const avgLoss = losingTrades.length > 0 ? Math.abs(losingTrades.reduce((sum, t) => sum + t.pnl, 0) / losingTrades.length) : 0;
    const expectancy = (winRate * avgWin) - ((1 - winRate) * avgLoss);

    // Kelly Criterion for optimal position sizing
    const kellyCriterion = avgLoss > 0 ? Math.max(0, (winRate - ((1 - winRate) / (avgWin / avgLoss))) * 100) : 0;

    // Efficiency Ratio: How much of the potential profit was captured
    const efficiencyRatio = averageMFE > 0 ? (avgWin / averageMFE) * 100 : 0;

    // Trade Quality Score (0-100)
    const tradeQuality = this.calculateTradeQuality(winRate, expectancy, efficiencyRatio, kellyCriterion);

    return {
      maximumAdverseExcursion: maxMAE,
      maximumFavorableExcursion: maxMFE,
      averageMAE: Math.round(averageMAE * 100) / 100,
      averageMFE: Math.round(averageMFE * 100) / 100,
      rMultipleDistribution,
      expectancy: Math.round(expectancy * 100) / 100,
      kellyCriterion: Math.round(kellyCriterion * 10) / 10,
      efficiencyRatio: Math.round(efficiencyRatio * 10) / 10,
      tradeQuality: Math.round(tradeQuality)
    };
  }

  /**
   * Estimate MAE for trades without recorded data
   */
  private estimateMAE(trade: any): number {
    // For winning trades, estimate MAE as a percentage of the trade size
    if (trade.pnl > 0) {
      return Math.abs(trade.entry_price * trade.quantity * 0.02); // Estimate 2% adverse move
    }
    // For losing trades, MAE is approximately the loss amount
    return Math.abs(trade.pnl);
  }

  /**
   * Estimate MFE for trades without recorded data
   */
  private estimateMFE(trade: any): number {
    // For losing trades, estimate MFE as a small favorable move
    if (trade.pnl <= 0) {
      return Math.abs(trade.entry_price * trade.quantity * 0.015); // Estimate 1.5% favorable move
    }
    // For winning trades, MFE is approximately the profit (could have been higher)
    return Math.abs(trade.pnl) * 1.2; // Estimate 20% more profit was possible
  }

  /**
   * Calculate R-multiple for a trade
   */
  private calculateRMultiple(trade: any): number | null {
    let riskAmount = 0;

    if (trade.risk_amount) {
      riskAmount = trade.risk_amount;
    } else if (trade.stop_loss && trade.entry_price) {
      riskAmount = Math.abs(trade.entry_price - trade.stop_loss) * trade.quantity;
    } else {
      return null;
    }

    if (riskAmount === 0) return null;

    return trade.pnl / riskAmount;
  }

  /**
   * Calculate R-multiple distribution
   */
  private calculateRMultipleDistribution(rMultiples: number[]): { [key: string]: number } {
    const distribution: { [key: string]: number } = {
      'Loss (-2R to 0R)': 0,
      'Small Win (0R to 1R)': 0,
      'Good Win (1R to 2R)': 0,
      'Great Win (2R to 3R)': 0,
      'Exceptional Win (3R+)': 0
    };

    for (const r of rMultiples) {
      if (r < 0) {
        distribution['Loss (-2R to 0R)']++;
      } else if (r < 1) {
        distribution['Small Win (0R to 1R)']++;
      } else if (r < 2) {
        distribution['Good Win (1R to 2R)']++;
      } else if (r < 3) {
        distribution['Great Win (2R to 3R)']++;
      } else {
        distribution['Exceptional Win (3R+)']++;
      }
    }

    // Convert to percentages
    const total = rMultiples.length;
    if (total > 0) {
      for (const key in distribution) {
        distribution[key] = Math.round((distribution[key] / total) * 100);
      }
    }

    return distribution;
  }

  /**
   * Calculate overall trade quality score
   */
  private calculateTradeQuality(
    winRate: number,
    expectancy: number,
    efficiencyRatio: number,
    kellyCriterion: number
  ): number {
    // Normalize and weight different components
    const winRateScore = Math.min(100, winRate * 100);
    const expectancyScore = Math.min(100, Math.max(0, (expectancy + 100) / 2)); // Assuming expectancy range -100 to +100
    const efficiencyScore = Math.min(100, efficiencyRatio);
    const kellyScore = Math.min(100, kellyCriterion);

    // Weighted average (you can adjust weights based on importance)
    const qualityScore = (
      winRateScore * 0.25 +
      expectancyScore * 0.35 +
      efficiencyScore * 0.25 +
      kellyScore * 0.15
    );

    return qualityScore;
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
      actualRiskRewardRatio: 0,
      maximumAdverseExcursion: 0,
      maximumFavorableExcursion: 0,
      averageMAE: 0,
      averageMFE: 0,
      rMultipleDistribution: {},
      kellyCriterion: 0,
      efficiencyRatio: 0,
      tradeQuality: 0
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