import { supabase } from '../../lib/supabase';

export interface TradeRecord {
  id?: string;
  user_id?: string;
  symbol: string;
  side: 'buy' | 'sell' | 'short' | 'cover';
  quantity: number;
  entry_price: number;
  exit_price?: number;
  order_id?: string;
  alpaca_order_id?: string;
  filled_at?: string;
  execution_status: 'pending' | 'filled' | 'partial' | 'cancelled' | 'failed';
  realized_pnl?: number;
  unrealized_pnl?: number;
  fees?: number;
  position_size_percent?: number;
  risk_amount?: number;
  confidence_score?: number;
  risk_reward_ratio?: number;
  created_at?: string;
  updated_at?: string;
  closed_at?: string;
}

export interface TradeFilters {
  symbol?: string;
  side?: string;
  status?: string;
  startDate?: Date;
  endDate?: Date;
  limit?: number;
}

export interface TradeStats {
  totalTrades: number;
  winningTrades: number;
  losingTrades: number;
  winRate: number;
  totalPnL: number;
  averageWin: number;
  averageLoss: number;
  profitFactor: number;
  largestWin: number;
  largestLoss: number;
}

class TradeHistoryService {
  private readonly TABLE_NAME = 'trade_history';
  private cache: Map<string, TradeRecord[]> = new Map();
  private statsCache: Map<string, TradeStats> = new Map();
  private cacheTimeout = 30000; // 30 seconds

  /**
   * Record a new trade
   */
  async recordTrade(trade: TradeRecord): Promise<TradeRecord | null> {
    try {
      const { data, error } = await supabase
        .from(this.TABLE_NAME)
        .insert([trade])
        .select()
        .single();

      if (error) {
        console.error('Error recording trade:', error);
        return null;
      }

      // Invalidate cache
      this.invalidateCache();
      
      // Emit event for real-time updates
      this.emitTradeEvent('new_trade', data);
      
      return data;
    } catch (error) {
      console.error('Failed to record trade:', error);
      return null;
    }
  }

  /**
   * Update an existing trade (e.g., when closing a position)
   */
  async updateTrade(tradeId: string, updates: Partial<TradeRecord>): Promise<TradeRecord | null> {
    try {
      // Calculate realized P&L if closing
      if (updates.exit_price && !updates.realized_pnl) {
        const trade = await this.getTrade(tradeId);
        if (trade) {
          const pnl = this.calculatePnL(
            trade.side,
            trade.quantity,
            trade.entry_price,
            updates.exit_price,
            updates.fees || 0
          );
          updates.realized_pnl = pnl;
          updates.closed_at = new Date().toISOString();
        }
      }

      const { data, error } = await supabase
        .from(this.TABLE_NAME)
        .update(updates)
        .eq('id', tradeId)
        .select()
        .single();

      if (error) {
        console.error('Error updating trade:', error);
        return null;
      }

      this.invalidateCache();
      this.emitTradeEvent('trade_updated', data);
      
      return data;
    } catch (error) {
      console.error('Failed to update trade:', error);
      return null;
    }
  }

  /**
   * Get a single trade by ID
   */
  async getTrade(tradeId: string): Promise<TradeRecord | null> {
    try {
      const { data, error } = await supabase
        .from(this.TABLE_NAME)
        .select('*')
        .eq('id', tradeId)
        .single();

      if (error) {
        console.error('Error fetching trade:', error);
        return null;
      }

      return data;
    } catch (error) {
      console.error('Failed to fetch trade:', error);
      return null;
    }
  }

  /**
   * Get trade history with filters
   */
  async getTradeHistory(filters?: TradeFilters): Promise<TradeRecord[]> {
    try {
      // Check cache first
      const cacheKey = JSON.stringify(filters || {});
      const cached = this.cache.get(cacheKey);
      if (cached && this.isCacheValid(cacheKey)) {
        return cached;
      }

      let query = supabase
        .from(this.TABLE_NAME)
        .select('*')
        .order('created_at', { ascending: false });

      // Apply filters
      if (filters?.symbol) {
        query = query.eq('symbol', filters.symbol);
      }
      if (filters?.side) {
        query = query.eq('side', filters.side);
      }
      if (filters?.status) {
        query = query.eq('execution_status', filters.status);
      }
      if (filters?.startDate) {
        query = query.gte('created_at', filters.startDate.toISOString());
      }
      if (filters?.endDate) {
        query = query.lte('created_at', filters.endDate.toISOString());
      }
      if (filters?.limit) {
        query = query.limit(filters.limit);
      }

      const { data, error } = await query;

      if (error) {
        console.error('Error fetching trade history:', error);
        return [];
      }

      // Update cache
      this.cache.set(cacheKey, data || []);
      setTimeout(() => this.cache.delete(cacheKey), this.cacheTimeout);

      return data || [];
    } catch (error) {
      console.error('Failed to fetch trade history:', error);
      return [];
    }
  }

  /**
   * Get recent trades (alias for getTradeHistory with limit)
   */
  async getRecentTrades(limit: number = 50): Promise<TradeRecord[]> {
    return this.getTradeHistory({ limit });
  }

  /**
   * Get open positions
   */
  async getOpenPositions(): Promise<TradeRecord[]> {
    try {
      const { data, error } = await supabase
        .from(this.TABLE_NAME)
        .select('*')
        .is('exit_price', null)
        .eq('execution_status', 'filled')
        .order('created_at', { ascending: false });

      if (error) {
        console.error('Error fetching open positions:', error);
        return [];
      }

      return data || [];
    } catch (error) {
      console.error('Failed to fetch open positions:', error);
      return [];
    }
  }

  /**
   * Calculate trading statistics
   */
  async getTradeStatistics(startDate?: Date, endDate?: Date): Promise<TradeStats> {
    try {
      // Check cache
      const cacheKey = `stats_${startDate?.toISOString()}_${endDate?.toISOString()}`;
      const cached = this.statsCache.get(cacheKey);
      if (cached && this.isCacheValid(cacheKey)) {
        return cached;
      }

      let query = supabase
        .from(this.TABLE_NAME)
        .select('*')
        .eq('execution_status', 'filled')
        .not('realized_pnl', 'is', null);

      if (startDate) {
        query = query.gte('created_at', startDate.toISOString());
      }
      if (endDate) {
        query = query.lte('created_at', endDate.toISOString());
      }

      const { data: trades, error } = await query;

      if (error) {
        console.error('Error calculating statistics:', error);
        return this.getEmptyStats();
      }

      if (!trades || trades.length === 0) {
        return this.getEmptyStats();
      }

      const winningTrades = trades.filter(t => (t.realized_pnl || 0) > 0);
      const losingTrades = trades.filter(t => (t.realized_pnl || 0) < 0);
      
      const totalWins = winningTrades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0);
      const totalLosses = Math.abs(losingTrades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0));

      const stats: TradeStats = {
        totalTrades: trades.length,
        winningTrades: winningTrades.length,
        losingTrades: losingTrades.length,
        winRate: trades.length > 0 ? (winningTrades.length / trades.length) * 100 : 0,
        totalPnL: trades.reduce((sum, t) => sum + (t.realized_pnl || 0), 0),
        averageWin: winningTrades.length > 0 ? totalWins / winningTrades.length : 0,
        averageLoss: losingTrades.length > 0 ? totalLosses / losingTrades.length : 0,
        profitFactor: totalLosses > 0 ? totalWins / totalLosses : totalWins > 0 ? Infinity : 0,
        largestWin: Math.max(...winningTrades.map(t => t.realized_pnl || 0), 0),
        largestLoss: Math.min(...losingTrades.map(t => t.realized_pnl || 0), 0),
      };

      // Cache the stats
      this.statsCache.set(cacheKey, stats);
      setTimeout(() => this.statsCache.delete(cacheKey), this.cacheTimeout);

      return stats;
    } catch (error) {
      console.error('Failed to calculate statistics:', error);
      return this.getEmptyStats();
    }
  }

  /**
   * Export trade history to CSV
   */
  async exportToCSV(filters?: TradeFilters): Promise<string> {
    const trades = await this.getTradeHistory(filters);
    
    if (trades.length === 0) {
      return '';
    }

    const headers = [
      'Date',
      'Symbol',
      'Side',
      'Quantity',
      'Entry Price',
      'Exit Price',
      'P&L',
      'Fees',
      'Status',
      'Confidence',
      'Risk/Reward'
    ];

    const rows = trades.map(t => [
      new Date(t.created_at || '').toLocaleString(),
      t.symbol,
      t.side,
      t.quantity.toString(),
      t.entry_price.toString(),
      t.exit_price?.toString() || '',
      t.realized_pnl?.toString() || '',
      t.fees?.toString() || '0',
      t.execution_status,
      t.confidence_score?.toString() || '',
      t.risk_reward_ratio?.toString() || ''
    ]);

    const csv = [
      headers.join(','),
      ...rows.map(row => row.join(','))
    ].join('\n');

    return csv;
  }

  /**
   * Subscribe to real-time trade updates
   */
  subscribeToTrades(callback: (trade: TradeRecord) => void) {
    const subscription = supabase
      .channel('trade_updates')
      .on(
        'postgres_changes',
        {
          event: '*',
          schema: 'public',
          table: this.TABLE_NAME
        },
        (payload) => {
          callback(payload.new as TradeRecord);
          this.invalidateCache();
        }
      )
      .subscribe();

    return () => {
      subscription.unsubscribe();
    };
  }

  /**
   * Batch update unrealized P&L for open positions
   */
  async updateUnrealizedPnL(currentPrices: Map<string, number>): Promise<void> {
    try {
      const openPositions = await this.getOpenPositions();
      
      const updates = openPositions.map(async (position) => {
        const currentPrice = currentPrices.get(position.symbol.toUpperCase());
        if (!currentPrice) return;

        const unrealizedPnl = this.calculatePnL(
          position.side,
          position.quantity,
          position.entry_price,
          currentPrice,
          0
        );

        return this.updateTrade(position.id!, {
          unrealized_pnl: unrealizedPnl
        });
      });

      await Promise.all(updates);
    } catch (error) {
      console.error('Failed to update unrealized P&L:', error);
    }
  }

  // Helper methods
  private calculatePnL(
    side: string,
    quantity: number,
    entryPrice: number,
    exitPrice: number,
    fees: number
  ): number {
    let pnl = 0;
    
    if (side === 'buy') {
      pnl = (exitPrice - entryPrice) * quantity;
    } else if (side === 'sell' || side === 'short') {
      pnl = (entryPrice - exitPrice) * quantity;
    } else if (side === 'cover') {
      // Cover is closing a short position
      pnl = (entryPrice - exitPrice) * quantity;
    }
    
    return pnl - fees;
  }

  private invalidateCache(): void {
    this.cache.clear();
    this.statsCache.clear();
  }

  private isCacheValid(key: string): boolean {
    // Simple time-based cache validation
    // In production, you might want more sophisticated cache invalidation
    return true;
  }

  private emitTradeEvent(event: string, data: any): void {
    // Emit custom event for UI updates
    window.dispatchEvent(new CustomEvent('trade-update', {
      detail: { event, data }
    }));
  }

  private getEmptyStats(): TradeStats {
    return {
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
      winRate: 0,
      totalPnL: 0,
      averageWin: 0,
      averageLoss: 0,
      profitFactor: 0,
      largestWin: 0,
      largestLoss: 0
    };
  }
}

// Export singleton instance
export const tradeHistoryService = new TradeHistoryService();