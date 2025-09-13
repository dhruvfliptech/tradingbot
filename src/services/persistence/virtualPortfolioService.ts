import { supabase } from '../../lib/supabase';
import { tradeHistoryService, TradeRecord } from './tradeHistoryService';

export interface VirtualPosition {
  symbol: string;
  quantity: number;
  entry_price: number;
  current_price: number;
  unrealized_pnl: number;
  opened_at: string;
  trade_id?: string;
}

export interface VirtualPortfolio {
  id?: string;
  user_id?: string;
  portfolio_name: string;
  initial_balance: number;
  current_balance: number;
  total_positions: number;
  open_positions: VirtualPosition[];
  total_realized_pnl: number;
  total_unrealized_pnl: number;
  total_fees_paid: number;
  peak_balance: number;
  max_drawdown: number;
  total_trades: number;
  winning_trades: number;
  losing_trades: number;
  win_rate: number;
  average_win: number;
  average_loss: number;
  profit_factor: number;
  sharpe_ratio: number;
  sortino_ratio: number;
  shadow_balance: number;
  shadow_pnl: number;
  user_impact_percent: number;
  created_at?: string;
  updated_at?: string;
  last_reset_at?: string;
}

export interface DailySnapshot {
  id?: string;
  portfolio_id: string;
  snapshot_date: string;
  opening_balance: number;
  closing_balance: number;
  high_balance: number;
  low_balance: number;
  daily_pnl: number;
  daily_pnl_percent: number;
  cumulative_pnl: number;
  trades_count: number;
  winning_trades: number;
  losing_trades: number;
  daily_volatility: number;
  max_position_size: number;
  risk_adjusted_return: number;
}

class VirtualPortfolioService {
  private readonly PORTFOLIO_TABLE = 'virtual_portfolios';
  private readonly DAILY_PERFORMANCE_TABLE = 'daily_performance';
  private readonly INITIAL_BALANCE = 50000;
  private portfolio: VirtualPortfolio | null = null;
  private shadowPortfolio: VirtualPortfolio | null = null;
  private updateTimer: number | null = null;

  /**
   * Initialize or get existing portfolio
   */
  async initializePortfolio(): Promise<VirtualPortfolio> {
    try {
      // Try to get existing portfolio
      const { data: existing, error: fetchError } = await supabase
        .from(this.PORTFOLIO_TABLE)
        .select('*')
        .single();

      if (existing && !fetchError) {
        this.portfolio = existing;
        // Parse JSON fields
        if (typeof existing.open_positions === 'string') {
          this.portfolio.open_positions = JSON.parse(existing.open_positions);
        }
        return this.portfolio;
      }

      // Create new portfolio
      const newPortfolio: Partial<VirtualPortfolio> = {
        portfolio_name: 'Main Portfolio',
        initial_balance: this.INITIAL_BALANCE,
        current_balance: this.INITIAL_BALANCE,
        total_positions: 0,
        open_positions: [],
        total_realized_pnl: 0,
        total_unrealized_pnl: 0,
        total_fees_paid: 0,
        peak_balance: this.INITIAL_BALANCE,
        max_drawdown: 0,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        win_rate: 0,
        average_win: 0,
        average_loss: 0,
        profit_factor: 0,
        sharpe_ratio: 0,
        sortino_ratio: 0,
        shadow_balance: this.INITIAL_BALANCE,
        shadow_pnl: 0,
        user_impact_percent: 0
      };

      const { data: created, error: createError } = await supabase
        .from(this.PORTFOLIO_TABLE)
        .insert([newPortfolio])
        .select()
        .single();

      if (createError) {
        console.error('Error creating portfolio:', createError);
        throw createError;
      }

      this.portfolio = created;
      this.portfolio.open_positions = [];
      
      // Create initial daily snapshot
      await this.createDailySnapshot();
      
      return this.portfolio;
    } catch (error) {
      console.error('Failed to initialize portfolio:', error);
      // Return a default portfolio in memory
      return this.getDefaultPortfolio();
    }
  }

  /**
   * Reset portfolio to initial state
   */
  async resetPortfolio(archiveExisting: boolean = true): Promise<VirtualPortfolio> {
    try {
      if (archiveExisting && this.portfolio) {
        // Archive current portfolio data
        await this.archivePortfolio();
      }

      // Reset to initial values
      const resetData: Partial<VirtualPortfolio> = {
        current_balance: this.INITIAL_BALANCE,
        total_positions: 0,
        open_positions: [],
        total_realized_pnl: 0,
        total_unrealized_pnl: 0,
        total_fees_paid: 0,
        peak_balance: this.INITIAL_BALANCE,
        max_drawdown: 0,
        total_trades: 0,
        winning_trades: 0,
        losing_trades: 0,
        win_rate: 0,
        average_win: 0,
        average_loss: 0,
        profit_factor: 0,
        sharpe_ratio: 0,
        sortino_ratio: 0,
        shadow_balance: this.INITIAL_BALANCE,
        shadow_pnl: 0,
        user_impact_percent: 0,
        last_reset_at: new Date().toISOString()
      };

      const { data: updated, error } = await supabase
        .from(this.PORTFOLIO_TABLE)
        .update(resetData)
        .eq('id', this.portfolio?.id)
        .select()
        .single();

      if (error) {
        console.error('Error resetting portfolio:', error);
        throw error;
      }

      this.portfolio = updated;
      this.portfolio.open_positions = [];
      
      // Create new daily snapshot for reset
      await this.createDailySnapshot();
      
      return this.portfolio;
    } catch (error) {
      console.error('Failed to reset portfolio:', error);
      return this.getDefaultPortfolio();
    }
  }

  /**
   * Open a new position
   */
  async openPosition(trade: TradeRecord): Promise<void> {
    if (!this.portfolio) {
      await this.initializePortfolio();
    }

    const position: VirtualPosition = {
      symbol: trade.symbol,
      quantity: trade.quantity,
      entry_price: trade.entry_price,
      current_price: trade.entry_price,
      unrealized_pnl: 0,
      opened_at: trade.created_at || new Date().toISOString(),
      trade_id: trade.id
    };

    // Calculate position cost
    const positionCost = trade.quantity * trade.entry_price + (trade.fees || 0);
    
    // Update portfolio
    this.portfolio!.open_positions.push(position);
    this.portfolio!.current_balance -= positionCost;
    this.portfolio!.total_positions += 1;
    this.portfolio!.total_trades += 1;
    this.portfolio!.total_fees_paid += trade.fees || 0;

    await this.savePortfolio();
    await this.updateMetrics();
  }

  /**
   * Close an existing position
   */
  async closePosition(symbol: string, exitPrice: number, fees: number = 0): Promise<number> {
    if (!this.portfolio) {
      await this.initializePortfolio();
    }

    const positionIndex = this.portfolio!.open_positions.findIndex(p => p.symbol === symbol);
    if (positionIndex === -1) {
      throw new Error(`No open position found for ${symbol}`);
    }

    const position = this.portfolio!.open_positions[positionIndex];
    
    // Calculate P&L
    const grossPnL = (exitPrice - position.entry_price) * position.quantity;
    const netPnL = grossPnL - fees;
    
    // Update portfolio
    this.portfolio!.open_positions.splice(positionIndex, 1);
    this.portfolio!.current_balance += (position.quantity * exitPrice - fees);
    this.portfolio!.total_realized_pnl += netPnL;
    this.portfolio!.total_fees_paid += fees;
    this.portfolio!.total_positions -= 1;
    
    // Update win/loss stats
    if (netPnL > 0) {
      this.portfolio!.winning_trades += 1;
      const currentAvgWin = this.portfolio!.average_win;
      const currentWins = this.portfolio!.winning_trades - 1;
      this.portfolio!.average_win = (currentAvgWin * currentWins + netPnL) / this.portfolio!.winning_trades;
    } else {
      this.portfolio!.losing_trades += 1;
      const currentAvgLoss = this.portfolio!.average_loss;
      const currentLosses = this.portfolio!.losing_trades - 1;
      this.portfolio!.average_loss = (currentAvgLoss * currentLosses + Math.abs(netPnL)) / this.portfolio!.losing_trades;
    }
    
    // Update win rate
    const totalClosed = this.portfolio!.winning_trades + this.portfolio!.losing_trades;
    this.portfolio!.win_rate = totalClosed > 0 ? (this.portfolio!.winning_trades / totalClosed) * 100 : 0;
    
    // Update profit factor
    const totalWins = this.portfolio!.winning_trades * this.portfolio!.average_win;
    const totalLosses = this.portfolio!.losing_trades * this.portfolio!.average_loss;
    this.portfolio!.profit_factor = totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0;

    // Update trade history
    if (position.trade_id) {
      await tradeHistoryService.updateTrade(position.trade_id, {
        exit_price: exitPrice,
        realized_pnl: netPnL,
        closed_at: new Date().toISOString()
      });
    }

    await this.savePortfolio();
    await this.updateMetrics();
    
    return netPnL;
  }

  /**
   * Update current prices for open positions
   */
  async updatePositionPrices(prices: Map<string, number>): Promise<void> {
    if (!this.portfolio || this.portfolio.open_positions.length === 0) {
      return;
    }

    let totalUnrealizedPnL = 0;
    
    for (const position of this.portfolio.open_positions) {
      const currentPrice = prices.get(position.symbol.toUpperCase());
      if (currentPrice) {
        position.current_price = currentPrice;
        position.unrealized_pnl = (currentPrice - position.entry_price) * position.quantity;
        totalUnrealizedPnL += position.unrealized_pnl;
      }
    }
    
    this.portfolio.total_unrealized_pnl = totalUnrealizedPnL;
    
    // Calculate current total value
    const totalValue = this.portfolio.current_balance + totalUnrealizedPnL;
    
    // Update peak and drawdown
    if (totalValue > this.portfolio.peak_balance) {
      this.portfolio.peak_balance = totalValue;
    }
    
    const drawdown = ((this.portfolio.peak_balance - totalValue) / this.portfolio.peak_balance) * 100;
    if (drawdown > this.portfolio.max_drawdown) {
      this.portfolio.max_drawdown = drawdown;
    }
    
    // Update unrealized P&L in trade history
    await tradeHistoryService.updateUnrealizedPnL(prices);
    
    await this.savePortfolio();
  }

  /**
   * Calculate performance metrics
   */
  async updateMetrics(): Promise<void> {
    if (!this.portfolio) return;

    // Get daily snapshots for calculations
    const snapshots = await this.getDailySnapshots(30); // Last 30 days
    
    if (snapshots.length < 2) return;
    
    // Calculate daily returns
    const returns: number[] = [];
    for (let i = 1; i < snapshots.length; i++) {
      const dailyReturn = snapshots[i].daily_pnl_percent;
      returns.push(dailyReturn);
    }
    
    if (returns.length === 0) return;
    
    // Calculate Sharpe Ratio (assuming 0% risk-free rate)
    const avgReturn = returns.reduce((sum, r) => sum + r, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - avgReturn, 2), 0) / returns.length;
    const stdDev = Math.sqrt(variance);
    this.portfolio.sharpe_ratio = stdDev > 0 ? (avgReturn / stdDev) * Math.sqrt(252) : 0; // Annualized
    
    // Calculate Sortino Ratio (downside deviation)
    const downsideReturns = returns.filter(r => r < 0);
    if (downsideReturns.length > 0) {
      const downsideVariance = downsideReturns.reduce((sum, r) => sum + Math.pow(r, 2), 0) / downsideReturns.length;
      const downsideDev = Math.sqrt(downsideVariance);
      this.portfolio.sortino_ratio = downsideDev > 0 ? (avgReturn / downsideDev) * Math.sqrt(252) : 0;
    }
    
    await this.savePortfolio();
  }

  /**
   * Update shadow portfolio (AI-only decisions)
   */
  async updateShadowPortfolio(aiDecision: 'buy' | 'sell', symbol: string, quantity: number, price: number): Promise<void> {
    if (!this.portfolio) return;
    
    // Simple shadow tracking - just track balance changes
    if (aiDecision === 'buy') {
      this.portfolio.shadow_balance -= quantity * price;
    } else {
      // Assume we're selling at current price
      this.portfolio.shadow_balance += quantity * price;
    }
    
    this.portfolio.shadow_pnl = this.portfolio.shadow_balance - this.INITIAL_BALANCE;
    
    // Calculate user impact
    const actualPnL = this.portfolio.total_realized_pnl + this.portfolio.total_unrealized_pnl;
    const shadowPnL = this.portfolio.shadow_pnl;
    const difference = actualPnL - shadowPnL;
    this.portfolio.user_impact_percent = shadowPnL !== 0 ? (difference / Math.abs(shadowPnL)) * 100 : 0;
    
    await this.savePortfolio();
  }

  /**
   * Create daily performance snapshot
   */
  async createDailySnapshot(): Promise<void> {
    if (!this.portfolio) return;
    
    const today = new Date().toISOString().split('T')[0];
    
    // Check if snapshot already exists for today
    const { data: existing } = await supabase
      .from(this.DAILY_PERFORMANCE_TABLE)
      .select('id')
      .eq('portfolio_id', this.portfolio.id)
      .eq('snapshot_date', today)
      .single();
    
    if (existing) return; // Already have today's snapshot
    
    const totalValue = this.portfolio.current_balance + this.portfolio.total_unrealized_pnl;
    
    // Get yesterday's snapshot for comparison
    const yesterday = new Date(Date.now() - 86400000).toISOString().split('T')[0];
    const { data: yesterdaySnapshot } = await supabase
      .from(this.DAILY_PERFORMANCE_TABLE)
      .select('closing_balance')
      .eq('portfolio_id', this.portfolio.id)
      .eq('snapshot_date', yesterday)
      .single();
    
    const openingBalance = yesterdaySnapshot?.closing_balance || this.portfolio.initial_balance;
    const dailyPnL = totalValue - openingBalance;
    const dailyPnLPercent = openingBalance > 0 ? (dailyPnL / openingBalance) * 100 : 0;
    
    const snapshot: Partial<DailySnapshot> = {
      portfolio_id: this.portfolio.id,
      snapshot_date: today,
      opening_balance: openingBalance,
      closing_balance: totalValue,
      high_balance: totalValue, // Will be updated throughout the day
      low_balance: totalValue,
      daily_pnl: dailyPnL,
      daily_pnl_percent: dailyPnLPercent,
      cumulative_pnl: this.portfolio.total_realized_pnl + this.portfolio.total_unrealized_pnl,
      trades_count: 0, // Will be incremented during the day
      winning_trades: 0,
      losing_trades: 0,
      daily_volatility: 0, // Will be calculated at end of day
      max_position_size: Math.max(...this.portfolio.open_positions.map(p => p.quantity * p.current_price), 0),
      risk_adjusted_return: 0 // Will be calculated at end of day
    };
    
    await supabase
      .from(this.DAILY_PERFORMANCE_TABLE)
      .insert([snapshot]);
  }

  /**
   * Get daily performance snapshots
   */
  async getDailySnapshots(days: number = 30): Promise<DailySnapshot[]> {
    const startDate = new Date(Date.now() - days * 86400000).toISOString().split('T')[0];
    
    const { data, error } = await supabase
      .from(this.DAILY_PERFORMANCE_TABLE)
      .select('*')
      .eq('portfolio_id', this.portfolio?.id)
      .gte('snapshot_date', startDate)
      .order('snapshot_date', { ascending: true });
    
    if (error) {
      console.error('Error fetching daily snapshots:', error);
      return [];
    }
    
    return data || [];
  }

  /**
   * Get portfolio value over time
   */
  async getPortfolioHistory(days: number = 30): Promise<{ date: string; value: number; pnl: number }[]> {
    const snapshots = await this.getDailySnapshots(days);
    
    return snapshots.map(s => ({
      date: s.snapshot_date,
      value: s.closing_balance,
      pnl: s.daily_pnl
    }));
  }

  // Helper methods
  private async savePortfolio(): Promise<void> {
    if (!this.portfolio) return;
    
    // Debounce saves
    if (this.updateTimer) {
      clearTimeout(this.updateTimer);
    }
    
    this.updateTimer = window.setTimeout(async () => {
      const { error } = await supabase
        .from(this.PORTFOLIO_TABLE)
        .update({
          ...this.portfolio,
          open_positions: JSON.stringify(this.portfolio!.open_positions),
          updated_at: new Date().toISOString()
        })
        .eq('id', this.portfolio!.id);
      
      if (error) {
        console.error('Error saving portfolio:', error);
      }
    }, 1000);
  }

  private async archivePortfolio(): Promise<void> {
    // In a real implementation, you'd copy the current portfolio to an archive table
    console.log('Archiving portfolio data...');
  }

  private getDefaultPortfolio(): VirtualPortfolio {
    return {
      portfolio_name: 'Main Portfolio',
      initial_balance: this.INITIAL_BALANCE,
      current_balance: this.INITIAL_BALANCE,
      total_positions: 0,
      open_positions: [],
      total_realized_pnl: 0,
      total_unrealized_pnl: 0,
      total_fees_paid: 0,
      peak_balance: this.INITIAL_BALANCE,
      max_drawdown: 0,
      total_trades: 0,
      winning_trades: 0,
      losing_trades: 0,
      win_rate: 0,
      average_win: 0,
      average_loss: 0,
      profit_factor: 0,
      sharpe_ratio: 0,
      sortino_ratio: 0,
      shadow_balance: this.INITIAL_BALANCE,
      shadow_pnl: 0,
      user_impact_percent: 0
    };
  }

  /**
   * Get current portfolio
   */
  getPortfolio(): VirtualPortfolio | null {
    return this.portfolio;
  }
}

// Export singleton instance
export const virtualPortfolioService = new VirtualPortfolioService();