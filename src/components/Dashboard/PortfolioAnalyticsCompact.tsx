import React, { useState, useEffect } from 'react';
import { TrendingUp, Award, Target, Shield } from 'lucide-react';
import { portfolioAnalytics, PortfolioMetrics } from '../../services/portfolioAnalytics';

export const PortfolioAnalyticsCompact: React.FC = () => {
  const [metrics, setMetrics] = useState<PortfolioMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadMetrics();
    const interval = setInterval(loadMetrics, 60000);
    return () => clearInterval(interval);
  }, []);

  const loadMetrics = async () => {
    try {
      const data = await portfolioAnalytics.calculateMetrics(30);
      setMetrics(data);
    } catch (error) {
      console.error('Failed to load metrics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading || !metrics) {
    return (
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="text-gray-400">Loading analytics...</div>
      </div>
    );
  }

  const getSharpeColor = (sharpe: number) => {
    if (sharpe >= 2) return 'text-green-400';
    if (sharpe >= 1) return 'text-yellow-400';
    return 'text-red-400';
  };

  return (
    <div className="bg-gray-800 rounded-lg p-6 h-full overflow-y-auto">
      <h2 className="text-xl font-bold text-white mb-6">Portfolio Analytics</h2>
      
      {/* Key Metrics */}
      <div className="grid grid-cols-2 gap-4 mb-6">
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center text-gray-400 text-xs mb-1">
            <Award className="h-3 w-3 mr-1" />
            Sharpe Ratio
          </div>
          <div className={`text-2xl font-bold ${getSharpeColor(metrics.sharpeRatio)}`}>
            {metrics.sharpeRatio.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center text-gray-400 text-xs mb-1">
            <Target className="h-3 w-3 mr-1" />
            Win Rate
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.winRate.toFixed(1)}%
          </div>
        </div>
        
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center text-gray-400 text-xs mb-1">
            <TrendingUp className="h-3 w-3 mr-1" />
            Profit Factor
          </div>
          <div className="text-2xl font-bold text-white">
            {metrics.profitFactor.toFixed(2)}
          </div>
        </div>
        
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="flex items-center text-gray-400 text-xs mb-1">
            <Shield className="h-3 w-3 mr-1" />
            Max Drawdown
          </div>
          <div className="text-2xl font-bold text-red-400">
            -{metrics.maxDrawdownPercent.toFixed(1)}%
          </div>
        </div>
      </div>
      
      {/* Performance Summary */}
      <div className="bg-gray-700 rounded-lg p-4">
        <h3 className="text-white font-medium mb-3">Performance</h3>
        <div className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-gray-400">Total Return</span>
            <span className={metrics.totalReturnPercent >= 0 ? 'text-green-400' : 'text-red-400'}>
              {metrics.totalReturnPercent >= 0 ? '+' : ''}{metrics.totalReturnPercent.toFixed(2)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Monthly</span>
            <span className={metrics.monthlyReturn >= 0 ? 'text-green-400' : 'text-red-400'}>
              {metrics.monthlyReturn >= 0 ? '+' : ''}{metrics.monthlyReturn.toFixed(2)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Total Trades</span>
            <span className="text-white">{metrics.totalTrades}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Expectancy</span>
            <span className={metrics.expectancy >= 0 ? 'text-green-400' : 'text-red-400'}>
              ${metrics.expectancy.toFixed(2)}
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};