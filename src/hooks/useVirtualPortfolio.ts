import { useState, useEffect } from 'react';
import { virtualPortfolioService, VirtualPortfolio } from '../services/persistence/virtualPortfolioService';
import { tradeHistoryService } from '../services/persistence/tradeHistoryService';
import { statePersistenceService } from '../services/persistence/statePersistenceService';

export interface PortfolioStats {
  totalValue: number;
  cashBalance: number;
  positionsValue: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  totalPnL: number;
  totalPnLPercent: number;
  winRate: number;
  profitFactor: number;
  sharpeRatio: number;
  maxDrawdown: number;
  userImpact: number;
}

export const useVirtualPortfolio = () => {
  const [portfolio, setPortfolio] = useState<VirtualPortfolio | null>(null);
  const [stats, setStats] = useState<PortfolioStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    loadPortfolio();
    
    // Subscribe to trade updates
    const unsubscribeTrades = tradeHistoryService.subscribeToTrades(() => {
      loadPortfolio();
    });
    
    // Update prices every 30 seconds
    const priceInterval = setInterval(() => {
      updatePrices();
    }, 30000);
    
    // Listen for portfolio updates
    const handlePortfolioUpdate = () => {
      loadPortfolio();
    };
    window.addEventListener('portfolio-update', handlePortfolioUpdate);
    
    return () => {
      unsubscribeTrades();
      clearInterval(priceInterval);
      window.removeEventListener('portfolio-update', handlePortfolioUpdate);
    };
  }, []);

  const loadPortfolio = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const portfolioData = await virtualPortfolioService.initializePortfolio();
      setPortfolio(portfolioData);
      
      // Calculate stats
      if (portfolioData) {
        const totalValue = portfolioData.current_balance + portfolioData.total_unrealized_pnl;
        const positionsValue = portfolioData.open_positions.reduce(
          (sum, pos) => sum + (pos.quantity * pos.current_price),
          0
        );
        
        // Get today's snapshot for daily P&L
        const today = new Date().toISOString().split('T')[0];
        const snapshots = await virtualPortfolioService.getDailySnapshots(1);
        const todaySnapshot = snapshots.find(s => s.snapshot_date === today);
        
        const dailyPnL = todaySnapshot?.daily_pnl || 0;
        const dailyPnLPercent = todaySnapshot?.daily_pnl_percent || 0;
        
        const totalPnL = portfolioData.total_realized_pnl + portfolioData.total_unrealized_pnl;
        const totalPnLPercent = ((totalValue - portfolioData.initial_balance) / portfolioData.initial_balance) * 100;
        
        const calculatedStats: PortfolioStats = {
          totalValue,
          cashBalance: portfolioData.current_balance,
          positionsValue,
          dailyPnL,
          dailyPnLPercent,
          totalPnL,
          totalPnLPercent,
          winRate: portfolioData.win_rate,
          profitFactor: portfolioData.profit_factor,
          sharpeRatio: portfolioData.sharpe_ratio,
          maxDrawdown: portfolioData.max_drawdown,
          userImpact: portfolioData.user_impact_percent
        };
        
        setStats(calculatedStats);
      }
    } catch (err) {
      console.error('Failed to load portfolio:', err);
      setError('Failed to load portfolio data');
    } finally {
      setLoading(false);
    }
  };

  const updatePrices = async () => {
    if (!portfolio || portfolio.open_positions.length === 0) {
      return;
    }
    
    try {
      // Get current prices from localStorage (set by main app)
      const persistedState = statePersistenceService.getState();
      const prices = persistedState.lastPrices;
      
      if (prices) {
        const priceMap = new Map(Object.entries(prices));
        await virtualPortfolioService.updatePositionPrices(priceMap);
        await loadPortfolio(); // Reload to get updated values
      }
    } catch (err) {
      console.error('Failed to update prices:', err);
    }
  };

  const resetPortfolio = async () => {
    if (!confirm('Are you sure you want to reset the portfolio to $50,000? This will archive all existing trades.')) {
      return;
    }
    
    try {
      setLoading(true);
      await virtualPortfolioService.resetPortfolio(true);
      await loadPortfolio();
    } catch (err) {
      console.error('Failed to reset portfolio:', err);
      setError('Failed to reset portfolio');
    } finally {
      setLoading(false);
    }
  };

  const getPerformanceHistory = async (days: number = 30) => {
    try {
      return await virtualPortfolioService.getPortfolioHistory(days);
    } catch (err) {
      console.error('Failed to get performance history:', err);
      return [];
    }
  };

  const getDailySnapshots = async (days: number = 30) => {
    try {
      return await virtualPortfolioService.getDailySnapshots(days);
    } catch (err) {
      console.error('Failed to get daily snapshots:', err);
      return [];
    }
  };

  return {
    portfolio,
    stats,
    loading,
    error,
    resetPortfolio,
    getPerformanceHistory,
    getDailySnapshots,
    reload: loadPortfolio
  };
};