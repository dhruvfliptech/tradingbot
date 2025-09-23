import React, { useEffect, useState } from 'react';
import { Trophy, TrendingUp, TrendingDown, Activity } from 'lucide-react';
import { metricsService, ComputedMetrics } from '../../services/metricsService';

export const PerformanceAnalytics: React.FC = () => {
  const [metrics, setMetrics] = useState<ComputedMetrics | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [updating, setUpdating] = useState<boolean>(false);

  const load = async () => {
    setLoading(true);
    try {
      const existing = await metricsService.getOverall();
      if (existing) setMetrics(existing);
      else {
        const computed = await metricsService.computeFromOrders();
        await metricsService.upsertOverall(computed);
        setMetrics(computed);
      }
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const recompute = async () => {
    setUpdating(true);
    try {
      const computed = await metricsService.computeFromOrders();
      await metricsService.upsertOverall(computed);
      setMetrics(computed);
    } finally {
      setUpdating(false);
    }
  };

  const Stat: React.FC<{ label: string; value: string; icon?: React.ReactNode; hint?: string }> = ({ label, value, icon, hint }) => (
    <div className="bg-gray-700 rounded-lg p-4 flex items-center justify-between">
      <div>
        <div className="text-sm text-gray-300">{label}</div>
        <div className="text-xl font-semibold text-white">{value}</div>
        {hint && <div className="text-xs text-gray-400 mt-1">{hint}</div>}
      </div>
      {icon}
    </div>
  );

  return (
    <div className="h-full min-h-[260px] overflow-y-auto bg-gray-800 rounded-lg border border-gray-700 p-6 pr-2">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white">Performance Analytics</h2>
        <button
          onClick={recompute}
          disabled={updating}
          className="px-3 py-1.5 bg-indigo-600 hover:bg-indigo-500 text-white text-sm rounded disabled:opacity-60"
        >
          {updating ? 'Updating…' : 'Recompute'}
        </button>
      </div>
      {loading ? (
        <div className="space-y-3">
          {[...Array(3)].map((_, i) => (
            <div key={i} className="bg-gray-700 rounded p-4 animate-pulse h-14" />
          ))}
        </div>
      ) : !metrics ? (
        <div className="text-gray-400 text-sm">No metrics yet.</div>
      ) : (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          <Stat label="Total Return" value={`${metrics.totalReturnPercent.toFixed(2)}%`} icon={<TrendingUp className="h-5 w-5 text-green-400" />} />
          <Stat label="Day Return" value={`${metrics.dayReturnPercent.toFixed(2)}%`} icon={<TrendingDown className="h-5 w-5 text-blue-400" />} />
          <Stat label="Sharpe Ratio" value={`${metrics.sharpe.toFixed(2)}`} icon={<Activity className="h-5 w-5 text-yellow-400" />} hint="Approx. based on per-trade series (v1)" />
          <Stat label="Total Trades" value={`${metrics.totalTrades}`} icon={<Trophy className="h-5 w-5 text-purple-400" />} />
          <Stat label="Win Rate" value={`${Math.round(metrics.winRate * 100)}%`} />
          <Stat label="Avg Trade Return" value={`${metrics.avgTradeReturn.toFixed(2)}%`} />
          <Stat label="Best Trade" value={`${metrics.bestTradeReturn.toFixed(2)}%`} />
          <Stat label="Worst Trade" value={`${metrics.worstTradeReturn.toFixed(2)}%`} />
        </div>
      )}
      <div className="text-xs text-gray-500 mt-4">
        v1 metrics use placeholders for realized PnL pairing. We’ll enhance with proper trade pairing and MTM soon.
      </div>
    </div>
  );
};


