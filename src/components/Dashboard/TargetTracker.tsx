import React, { useState, useEffect } from 'react';
import { 
  Target, 
  TrendingUp, 
  TrendingDown, 
  Calendar,
  Clock,
  AlertTriangle,
  CheckCircle,
  Award,
  Activity
} from 'lucide-react';
import { tradingAgentV2 } from '../../services/tradingAgentV2';
import { useVirtualPortfolio } from '../../hooks/useVirtualPortfolio';

interface WeeklyTarget {
  target: number;
  current: number;
  progress: number;
  daysRemaining: number;
  riskLevel: 'conservative' | 'moderate' | 'aggressive';
  projectedReturn: number;
  status: 'on_track' | 'ahead' | 'behind' | 'exceeded';
}

export const TargetTracker: React.FC = () => {
  const [weeklyTarget, setWeeklyTarget] = useState<WeeklyTarget | null>(null);
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const { stats: portfolioStats } = useVirtualPortfolio();

  useEffect(() => {
    let mounted = true; // Flag to prevent state updates after unmount

    const updateTargetData = () => {
      if (!mounted) return; // Exit if component was unmounted

      try {
        const progress = tradingAgentV2.getWeeklyProgress();
        
        // Calculate days remaining in week (Monday to Sunday)
        const now = new Date();
        const dayOfWeek = now.getDay(); // 0 = Sunday, 1 = Monday, etc.
        const daysUntilSunday = dayOfWeek === 0 ? 0 : 7 - dayOfWeek;
        
        // Calculate current week's P&L from portfolio stats
        const currentWeekPnL = portfolioStats?.weeklyPnL || 0;
        const currentWeekPnLPercent = portfolioStats?.weeklyPnLPercent || 0;
        
        // Calculate projected return based on current rate
        const daysPassed = 7 - daysUntilSunday;
        const dailyRate = daysPassed > 0 ? currentWeekPnLPercent / daysPassed : 0;
        const projectedReturn = dailyRate * 7;
        
        // Determine status
        let status: WeeklyTarget['status'] = 'on_track';
        const progressRatio = currentWeekPnLPercent / progress.target;
        
        if (currentWeekPnLPercent >= progress.target) {
          status = 'exceeded';
        } else if (progressRatio > 1.1) {
          status = 'ahead';
        } else if (progressRatio < 0.7) {
          status = 'behind';
        }
        
        // Determine risk level based on progress and time remaining
        let riskLevel: WeeklyTarget['riskLevel'] = 'moderate';
        if (status === 'exceeded' || (status === 'ahead' && daysUntilSunday > 2)) {
          riskLevel = 'conservative';
        } else if (status === 'behind' && daysUntilSunday < 3) {
          riskLevel = 'aggressive';
        }

        if (mounted) { // Only update state if still mounted
          setWeeklyTarget({
            target: progress.target,
            current: currentWeekPnLPercent,
            progress: progressRatio,
            daysRemaining: daysUntilSunday,
            riskLevel,
            projectedReturn,
            status
          });
          
          setLastUpdate(new Date());
        }
      } catch (error) {
        console.error('Error updating target tracker data:', error);
      }
    };

    // Update immediately
    updateTargetData();

    // Update every minute
    const interval = setInterval(updateTargetData, 60000);

    return () => {
      mounted = false; // Mark component as unmounted
      clearInterval(interval); // Clear the interval
    };
  }, [portfolioStats]);

  const getProgressColor = (progress: number, status: string) => {
    if (status === 'exceeded') return 'bg-green-500';
    if (status === 'ahead') return 'bg-blue-500';
    if (status === 'behind') return 'bg-red-500';
    return 'bg-yellow-500';
  };

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'exceeded':
        return <Award className="h-4 w-4 text-green-400" />;
      case 'ahead':
        return <TrendingUp className="h-4 w-4 text-blue-400" />;
      case 'behind':
        return <TrendingDown className="h-4 w-4 text-red-400" />;
      case 'on_track':
        return <CheckCircle className="h-4 w-4 text-yellow-400" />;
      default:
        return <Activity className="h-4 w-4 text-gray-400" />;
    }
  };

  const getStatusMessage = (target: WeeklyTarget) => {
    switch (target.status) {
      case 'exceeded':
        return `Target exceeded! Consider reducing risk to preserve gains.`;
      case 'ahead':
        return `Ahead of schedule. Current pace looks good.`;
      case 'behind':
        return `Behind target. ${target.daysRemaining < 3 ? 'Consider increasing position sizes.' : 'Still time to catch up.'}`;
      case 'on_track':
        return `On track to meet weekly target.`;
      default:
        return 'Monitoring weekly progress...';
    }
  };

  const getRiskLevelColor = (riskLevel: string) => {
    switch (riskLevel) {
      case 'conservative':
        return 'text-green-400 bg-green-900/30';
      case 'moderate':
        return 'text-yellow-400 bg-yellow-900/30';
      case 'aggressive':
        return 'text-red-400 bg-red-900/30';
      default:
        return 'text-gray-400 bg-gray-900/30';
    }
  };

  if (!weeklyTarget) {
    return (
      <div className="bg-gray-700 rounded-lg p-4">
        <div className="flex items-center space-x-2 mb-4">
          <Target className="h-5 w-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">Weekly Target Tracker</h3>
        </div>
        <div className="text-center py-8">
          <Activity className="h-8 w-8 text-gray-500 mx-auto mb-2 animate-pulse" />
          <p className="text-gray-500">Loading target data...</p>
        </div>
      </div>
    );
  }

  const progressPercentage = Math.min(100, Math.max(0, weeklyTarget.progress * 100));

  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Target className="h-5 w-5 text-purple-400" />
          <h3 className="text-lg font-semibold text-white">Weekly Target Tracker</h3>
        </div>
        <div className="text-xs text-gray-400">
          Updated: {lastUpdate.toLocaleTimeString()}
        </div>
      </div>

      <div className="space-y-4">
        {/* Progress Overview */}
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              {getStatusIcon(weeklyTarget.status)}
              <span className="text-sm font-medium text-white capitalize">
                {weeklyTarget.status.replace('_', ' ')}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <Calendar className="h-4 w-4 text-gray-400" />
              <span className="text-sm text-gray-300">
                {weeklyTarget.daysRemaining} days left
              </span>
            </div>
          </div>
          
          <div className="text-xs text-gray-400 mb-3">
            {getStatusMessage(weeklyTarget)}
          </div>

          {/* Progress Bar */}
          <div className="space-y-2">
            <div className="flex justify-between text-sm">
              <span className="text-gray-300">Progress</span>
              <span className="text-white font-medium">
                {weeklyTarget.current.toFixed(2)}% / {weeklyTarget.target.toFixed(1)}%
              </span>
            </div>
            <div className="w-full bg-gray-600 rounded-full h-2">
              <div 
                className={`h-2 rounded-full transition-all duration-500 ${getProgressColor(weeklyTarget.progress, weeklyTarget.status)}`}
                style={{ width: `${progressPercentage}%` }}
              />
            </div>
            <div className="text-xs text-gray-400 text-center">
              {progressPercentage.toFixed(1)}% of target achieved
            </div>
          </div>
        </div>

        {/* Stats Grid */}
        <div className="grid grid-cols-2 gap-3">
          <div className="bg-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-400">Current Return</span>
              <TrendingUp className={`h-3 w-3 ${weeklyTarget.current >= 0 ? 'text-green-400' : 'text-red-400'}`} />
            </div>
            <div className={`text-lg font-bold ${weeklyTarget.current >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {weeklyTarget.current >= 0 ? '+' : ''}{weeklyTarget.current.toFixed(2)}%
            </div>
          </div>

          <div className="bg-gray-800 rounded-lg p-3">
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs text-gray-400">Projected Return</span>
              <Activity className="h-3 w-3 text-blue-400" />
            </div>
            <div className={`text-lg font-bold ${weeklyTarget.projectedReturn >= weeklyTarget.target ? 'text-green-400' : 'text-yellow-400'}`}>
              {weeklyTarget.projectedReturn >= 0 ? '+' : ''}{weeklyTarget.projectedReturn.toFixed(2)}%
            </div>
          </div>
        </div>

        {/* Risk Level */}
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <AlertTriangle className="h-4 w-4 text-orange-400" />
              <span className="text-sm text-gray-300">Suggested Risk Level</span>
            </div>
            <span className={`px-2 py-1 rounded text-xs font-medium capitalize ${getRiskLevelColor(weeklyTarget.riskLevel)}`}>
              {weeklyTarget.riskLevel}
            </span>
          </div>
          
          <div className="mt-2 text-xs text-gray-400">
            {weeklyTarget.riskLevel === 'conservative' && 
              'Target met or exceeded. Focus on capital preservation.'}
            {weeklyTarget.riskLevel === 'moderate' && 
              'Balanced approach. Maintain current strategy.'}
            {weeklyTarget.riskLevel === 'aggressive' && 
              'Behind target. Consider larger positions or higher-confidence trades.'}
          </div>
        </div>

        {/* Time Analysis */}
        <div className="bg-gray-800 rounded-lg p-3">
          <div className="flex items-center space-x-2 mb-2">
            <Clock className="h-4 w-4 text-blue-400" />
            <span className="text-sm font-medium text-white">Time Analysis</span>
          </div>
          
          <div className="grid grid-cols-2 gap-4 text-xs">
            <div>
              <span className="text-gray-400">Week Progress:</span>
              <div className="text-white font-medium">
                {7 - weeklyTarget.daysRemaining} / 7 days
              </div>
            </div>
            <div>
              <span className="text-gray-400">Daily Avg Needed:</span>
              <div className="text-white font-medium">
                {weeklyTarget.daysRemaining > 0 
                  ? ((weeklyTarget.target - weeklyTarget.current) / weeklyTarget.daysRemaining).toFixed(2)
                  : '0.00'
                }%
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Footer */}
      <div className="mt-4 pt-3 border-t border-gray-600 text-xs text-gray-500 text-center">
        Target: {weeklyTarget.target}% weekly return • Resets every Monday
      </div>
    </div>
  );
};

export default TargetTracker;