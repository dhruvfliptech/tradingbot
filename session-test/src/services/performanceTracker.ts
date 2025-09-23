import { supabase } from '../lib/supabase';

export interface StrategyPerformance {
  strategy_name: string;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  total_pnl: number;
  win_rate: number;
  average_win: number;
  average_loss: number;
  profit_factor: number;
  sharpe_ratio: number;
  max_drawdown: number;
  last_trade_date: string;
  confidence_avg: number;
  risk_reward_avg: number;
}

export interface TradeAttribution {
  trade_id: string;
  strategy_name: string;
  confidence: number;
  expected_outcome: 'win' | 'loss';
  actual_outcome: 'win' | 'loss' | 'pending';
  accuracy_score: number;
}

export interface PerformanceSnapshot {
  snapshot_date: string;
  total_portfolio_value: number;
  strategy_breakdown: {
    [strategyName: string]: {
      trades_count: number;
      pnl_contribution: number;
      win_rate: number;
      avg_confidence: number;
    };
  };
  market_conditions: {
    btc_price: number;
    market_trend: 'bullish' | 'bearish' | 'sideways';
    volatility: number;
    fear_greed_index: number;
  };
}

class PerformanceTracker {
  private readonly STRATEGY_NAMES = ['liquidity', 'smartMoney', 'volumeProfile', 'microstructure'];

  /**
   * Record strategy attribution for a trade
   */
  async recordTradeAttribution(
    tradeId: string,
    strategyName: string,
    confidence: number,
    expectedOutcome: 'win' | 'loss'
  ): Promise<void> {
    try {
      // Try Supabase first, fall back to localStorage
      const { error } = await supabase
        .from('trade_attribution')
        .insert({
          trade_id: tradeId,
          strategy_name: strategyName,
          confidence: confidence,
          expected_outcome: expectedOutcome,
          actual_outcome: 'pending',
          accuracy_score: 0,
          created_at: new Date().toISOString()
        });

      if (error) throw error;

    } catch (error) {
      console.error('Error recording trade attribution to Supabase, storing locally:', error);
      
      // Fallback to localStorage
      const localKey = `trade_attribution_${tradeId}`;
      const attribution: TradeAttribution = {
        trade_id: tradeId,
        strategy_name: strategyName,
        confidence: confidence,
        expected_outcome: expectedOutcome,
        actual_outcome: 'pending',
        accuracy_score: 0
      };
      
      localStorage.setItem(localKey, JSON.stringify(attribution));
    }
  }

  /**
   * Update trade outcome when trade is closed
   */
  async updateTradeOutcome(tradeId: string, actualPnl: number): Promise<void> {
    const actualOutcome: 'win' | 'loss' = actualPnl > 0 ? 'win' : 'loss';
    
    try {
      // Get existing attribution
      const { data: attribution, error: fetchError } = await supabase
        .from('trade_attribution')
        .select('*')
        .eq('trade_id', tradeId)
        .maybeSingle();

      if (fetchError) throw fetchError;

      if (attribution) {
        // Calculate accuracy score
        const accuracyScore = attribution.expected_outcome === actualOutcome ? 100 : 0;
        
        const { error: updateError } = await supabase
          .from('trade_attribution')
          .update({
            actual_outcome: actualOutcome,
            accuracy_score: accuracyScore,
            updated_at: new Date().toISOString()
          })
          .eq('trade_id', tradeId);

        if (updateError) throw updateError;
      }

    } catch (error) {
      console.error('Error updating trade outcome in Supabase:', error);
      
      // Try localStorage fallback
      const localKey = `trade_attribution_${tradeId}`;
      const storedData = localStorage.getItem(localKey);
      
      if (storedData) {
        const attribution: TradeAttribution = JSON.parse(storedData);
        attribution.actual_outcome = actualOutcome;
        attribution.accuracy_score = attribution.expected_outcome === actualOutcome ? 100 : 0;
        localStorage.setItem(localKey, JSON.stringify(attribution));
      }
    }
  }

  /**
   * Calculate performance metrics for each strategy
   */
  async calculateStrategyPerformance(): Promise<StrategyPerformance[]> {
    const performances: StrategyPerformance[] = [];

    for (const strategyName of this.STRATEGY_NAMES) {
      try {
        // Try to get data from Supabase first
        const performance = await this.getStrategyPerformanceFromSupabase(strategyName);
        performances.push(performance);
      } catch (error) {
        console.error(`Error getting performance for ${strategyName} from Supabase:`, error);
        
        // Fallback to localStorage calculation
        const localPerformance = await this.getStrategyPerformanceFromLocal(strategyName);
        performances.push(localPerformance);
      }
    }

    return performances;
  }

  /**
   * Get strategy performance from Supabase
   */
  private async getStrategyPerformanceFromSupabase(strategyName: string): Promise<StrategyPerformance> {
    // Get trade attributions for this strategy
    const { data: attributions, error: attrError } = await supabase
      .from('trade_attribution')
      .select(`
        *,
        trade_history (
          pnl,
          quantity,
          entry_price,
          exit_price,
          created_at,
          confidence_score,
          risk_reward_ratio
        )
      `)
      .eq('strategy_name', strategyName)
      .not('actual_outcome', 'eq', 'pending');

    if (attrError) throw attrError;

    return this.calculatePerformanceMetrics(strategyName, attributions || []);
  }

  /**
   * Get strategy performance from localStorage
   */
  private async getStrategyPerformanceFromLocal(strategyName: string): Promise<StrategyPerformance> {
    const attributions: any[] = [];
    
    // Scan localStorage for attributions
    for (let i = 0; i < localStorage.length; i++) {
      const key = localStorage.key(i);
      if (key?.startsWith('trade_attribution_')) {
        const data = localStorage.getItem(key);
        if (data) {
          const attribution = JSON.parse(data);
          if (attribution.strategy_name === strategyName && attribution.actual_outcome !== 'pending') {
            attributions.push(attribution);
          }
        }
      }
    }

    return this.calculatePerformanceMetrics(strategyName, attributions);
  }

  /**
   * Calculate performance metrics from attribution data
   */
  private calculatePerformanceMetrics(strategyName: string, attributions: any[]): StrategyPerformance {
    const trades = attributions.filter(attr => attr.trade_history || attr.pnl !== undefined);
    const totalTrades = trades.length;
    
    if (totalTrades === 0) {
      return {
        strategy_name: strategyName,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        total_pnl: 0,
        win_rate: 0,
        average_win: 0,
        average_loss: 0,
        profit_factor: 0,
        sharpe_ratio: 0,
        max_drawdown: 0,
        last_trade_date: '',
        confidence_avg: 0,
        risk_reward_avg: 0
      };
    }

    const winningTrades = trades.filter(trade => {
      const pnl = trade.trade_history?.pnl || trade.pnl || 0;
      return pnl > 0;
    });
    
    const losingTrades = trades.filter(trade => {
      const pnl = trade.trade_history?.pnl || trade.pnl || 0;
      return pnl <= 0;
    });

    const totalPnl = trades.reduce((sum, trade) => {
      const pnl = trade.trade_history?.pnl || trade.pnl || 0;
      return sum + pnl;
    }, 0);

    const totalWins = winningTrades.reduce((sum, trade) => {
      const pnl = trade.trade_history?.pnl || trade.pnl || 0;
      return sum + pnl;
    }, 0);

    const totalLosses = Math.abs(losingTrades.reduce((sum, trade) => {
      const pnl = trade.trade_history?.pnl || trade.pnl || 0;
      return sum + pnl;
    }, 0));

    const winRate = (winningTrades.length / totalTrades) * 100;
    const averageWin = winningTrades.length > 0 ? totalWins / winningTrades.length : 0;
    const averageLoss = losingTrades.length > 0 ? totalLosses / losingTrades.length : 0;
    const profitFactor = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0;

    // Calculate confidence average
    const confidenceAvg = trades.reduce((sum, trade) => {
      return sum + (trade.confidence || trade.trade_history?.confidence_score || 0);
    }, 0) / totalTrades;

    // Calculate risk/reward average
    const riskRewardAvg = trades.reduce((sum, trade) => {
      return sum + (trade.trade_history?.risk_reward_ratio || 1.0);
    }, 0) / totalTrades;

    // Get last trade date
    const lastTradeDate = trades.length > 0 
      ? trades.sort((a, b) => {
          const dateA = new Date(a.trade_history?.created_at || a.created_at || 0);
          const dateB = new Date(b.trade_history?.created_at || b.created_at || 0);
          return dateB.getTime() - dateA.getTime();
        })[0].trade_history?.created_at || ''
      : '';

    return {
      strategy_name: strategyName,
      total_trades: totalTrades,
      winning_trades: winningTrades.length,
      losing_trades: losingTrades.length,
      total_pnl: totalPnl,
      win_rate: winRate,
      average_win: averageWin,
      average_loss: averageLoss,
      profit_factor: profitFactor,
      sharpe_ratio: 0, // Would need more historical data
      max_drawdown: 0, // Would need more historical data
      last_trade_date: lastTradeDate,
      confidence_avg: confidenceAvg,
      risk_reward_avg: riskRewardAvg
    };
  }

  /**
   * Save daily performance snapshot
   */
  async savePerformanceSnapshot(snapshot: PerformanceSnapshot): Promise<void> {
    try {
      const { error } = await supabase
        .from('performance_snapshots')
        .upsert({
          snapshot_date: snapshot.snapshot_date,
          total_portfolio_value: snapshot.total_portfolio_value,
          strategies_data: snapshot.strategy_breakdown,
          market_data: snapshot.market_conditions,
          updated_at: new Date().toISOString()
        }, { onConflict: 'snapshot_date' });

      if (error) throw error;

    } catch (error) {
      console.error('Error saving performance snapshot to Supabase:', error);
      
      // Fallback to localStorage
      const localKey = `performance_snapshot_${snapshot.snapshot_date}`;
      localStorage.setItem(localKey, JSON.stringify(snapshot));
    }
  }

  /**
   * Get performance snapshots for a date range
   */
  async getPerformanceSnapshots(days: number = 30): Promise<PerformanceSnapshot[]> {
    try {
      const startDate = new Date();
      startDate.setDate(startDate.getDate() - days);

      const { data, error } = await supabase
        .from('performance_snapshots')
        .select('*')
        .gte('snapshot_date', startDate.toISOString().split('T')[0])
        .order('snapshot_date', { ascending: true });

      if (error) throw error;

      return (data || []).map(row => ({
        snapshot_date: row.snapshot_date,
        total_portfolio_value: row.total_portfolio_value,
        strategy_breakdown: row.strategies_data || {},
        market_conditions: row.market_data || {
          btc_price: 0,
          market_trend: 'sideways' as const,
          volatility: 0,
          fear_greed_index: 50
        }
      }));

    } catch (error) {
      console.error('Error getting performance snapshots from Supabase:', error);
      
      // Fallback to localStorage
      const snapshots: PerformanceSnapshot[] = [];
      
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i);
        if (key?.startsWith('performance_snapshot_')) {
          const data = localStorage.getItem(key);
          if (data) {
            snapshots.push(JSON.parse(data));
          }
        }
      }
      
      return snapshots.sort((a, b) => 
        new Date(a.snapshot_date).getTime() - new Date(b.snapshot_date).getTime()
      );
    }
  }

  /**
   * Get strategy accuracy scores
   */
  async getStrategyAccuracy(): Promise<{ [strategyName: string]: number }> {
    const accuracy: { [strategyName: string]: number } = {};

    for (const strategyName of this.STRATEGY_NAMES) {
      try {
        const { data, error } = await supabase
          .from('trade_attribution')
          .select('accuracy_score')
          .eq('strategy_name', strategyName)
          .not('actual_outcome', 'eq', 'pending');

        if (error) throw error;

        if (data && data.length > 0) {
          const avgAccuracy = data.reduce((sum, item) => sum + item.accuracy_score, 0) / data.length;
          accuracy[strategyName] = avgAccuracy;
        } else {
          accuracy[strategyName] = 0;
        }

      } catch (error) {
        console.error(`Error getting accuracy for ${strategyName}:`, error);
        accuracy[strategyName] = 0;
      }
    }

    return accuracy;
  }

  /**
   * Get best performing strategy over time period
   */
  async getBestStrategy(days: number = 30): Promise<{ name: string; performance: StrategyPerformance } | null> {
    const performances = await this.calculateStrategyPerformance();
    
    if (performances.length === 0) return null;

    // Filter strategies with recent activity
    const recentStrategies = performances.filter(p => {
      if (!p.last_trade_date) return false;
      const lastTrade = new Date(p.last_trade_date);
      const cutoff = new Date();
      cutoff.setDate(cutoff.getDate() - days);
      return lastTrade >= cutoff;
    });

    if (recentStrategies.length === 0) return null;

    // Sort by profit factor (or total PnL if profit factor is similar)
    const best = recentStrategies.sort((a, b) => {
      if (Math.abs(a.profit_factor - b.profit_factor) < 0.1) {
        return b.total_pnl - a.total_pnl;
      }
      return b.profit_factor - a.profit_factor;
    })[0];

    return {
      name: best.strategy_name,
      performance: best
    };
  }
}

export const performanceTracker = new PerformanceTracker();
export default performanceTracker;