import React, { useState, useEffect } from 'react';
import { 
  Target, 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Zap, 
  BarChart3,
  Brain,
  Waves,
  AlertTriangle,
  CheckCircle,
  Clock,
  Minus
} from 'lucide-react';
import { tradingAgentV2 } from '../../services/tradingAgentV2';

interface StrategySignal {
  action: string;
  confidence: number;
  symbol?: string;
  timestamp?: string;
  reasoning?: string[];
  urgency?: string;
  signalType?: string;
  targetLevel?: number;
  currentPrice?: number;
}

interface StrategySignalsProps {
  // No props needed - component fetches data from trading agent
}

export const StrategySignals: React.FC<StrategySignalsProps> = () => {
  const [strategyData, setStrategyData] = useState<Map<string, any>>(new Map());
  const [lastUpdate, setLastUpdate] = useState<Date>(new Date());
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let mounted = true; // Flag to prevent state updates after unmount

    const updateData = () => {
      if (!mounted) return; // Exit if component was unmounted
      
      try {
        const signals = tradingAgentV2.getStrategySignals();
        if (mounted) { // Only update state if still mounted
          setStrategyData(new Map(signals));
          setLastUpdate(new Date());
          setLoading(false);
        }
      } catch (error) {
        console.error('Error updating strategy signals:', error);
      }
    };

    // Update immediately
    updateData();

    // Update every 30 seconds
    const interval = setInterval(updateData, 30000);

    return () => {
      mounted = false; // Mark component as unmounted
      clearInterval(interval); // Clear the interval
    };
  }, []);

  const getSignalIcon = (strategyType: string) => {
    switch (strategyType) {
      case 'liquidity':
        return <Waves className="h-4 w-4" />;
      case 'smartMoney':
        return <Brain className="h-4 w-4" />;
      case 'volumeProfile':
        return <BarChart3 className="h-4 w-4" />;
      case 'microstructure':
        return <Activity className="h-4 w-4" />;
      default:
        return <Zap className="h-4 w-4" />;
    }
  };

  const getSignalColor = (confidence: number, action: string) => {
    if (action === 'watch' || action === 'wait' || action === 'neutral' || action === 'avoid') {
      return 'text-gray-400';
    }
    
    if (confidence >= 80) return 'text-green-400';
    if (confidence >= 60) return 'text-yellow-400';
    if (confidence >= 40) return 'text-orange-400';
    return 'text-red-400';
  };

  const getActionIcon = (action: string) => {
    if (action.includes('buy') || action.includes('long') || action === 'follow_smart_money') {
      return <TrendingUp className="h-3 w-3" />;
    }
    if (action.includes('sell') || action.includes('short') || action === 'fade_retail') {
      return <TrendingDown className="h-3 w-3" />;
    }
    if (action === 'wait' || action === 'watch') {
      return <Clock className="h-3 w-3" />;
    }
    if (action === 'avoid') {
      return <AlertTriangle className="h-3 w-3" />;
    }
    return <Minus className="h-3 w-3" />;
  };

  const formatAction = (action: string) => {
    return action
      .replace(/_/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .toLowerCase()
      .replace(/\b\w/g, l => l.toUpperCase());
  };

  const getUrgencyColor = (urgency?: string) => {
    switch (urgency) {
      case 'immediate':
        return 'bg-red-900/50 text-red-300';
      case 'high':
        return 'bg-orange-900/50 text-orange-300';
      case 'medium':
        return 'bg-yellow-900/50 text-yellow-300';
      case 'low':
        return 'bg-green-900/50 text-green-300';
      default:
        return 'bg-gray-900/50 text-gray-300';
    }
  };

  const renderSignalCard = (
    symbol: string,
    strategyType: string,
    signal: StrategySignal | null,
    strategyName: string
  ) => {
    if (!signal) {
      return (
        <div key={`${symbol}-${strategyType}`} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
          <div className="flex items-center justify-between mb-2">
            <div className="flex items-center space-x-2">
              {getSignalIcon(strategyType)}
              <span className="text-sm font-medium text-white">{strategyName}</span>
            </div>
            <span className="text-xs text-gray-500">{symbol.toUpperCase()}</span>
          </div>
          <div className="text-xs text-gray-500">No signal</div>
        </div>
      );
    }

    const confidence = signal.confidence || 0;
    const action = signal.action || 'neutral';
    const signalColor = getSignalColor(confidence, action);

    return (
      <div key={`${symbol}-${strategyType}`} className="bg-gray-800 rounded-lg p-3 border border-gray-700 hover:border-gray-600 transition-colors">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-2">
            <div className={signalColor}>
              {getSignalIcon(strategyType)}
            </div>
            <span className="text-sm font-medium text-white">{strategyName}</span>
          </div>
          <div className="flex items-center space-x-2">
            {signal.urgency && (
              <span className={`px-2 py-1 rounded text-xs ${getUrgencyColor(signal.urgency)}`}>
                {signal.urgency}
              </span>
            )}
            <span className="text-xs text-gray-500">{symbol.toUpperCase()}</span>
          </div>
        </div>

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-2">
              <div className={signalColor}>
                {getActionIcon(action)}
              </div>
              <span className={`text-sm font-medium ${signalColor}`}>
                {formatAction(action)}
              </span>
            </div>
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${confidence >= 70 ? 'bg-green-400' : confidence >= 50 ? 'bg-yellow-400' : 'bg-red-400'}`} />
              <span className="text-sm text-gray-300">{confidence}%</span>
            </div>
          </div>

          {signal.signalType && (
            <div className="text-xs text-gray-400">
              Type: {signal.signalType.replace(/_/g, ' ')}
            </div>
          )}

          {signal.targetLevel && signal.currentPrice && (
            <div className="text-xs text-gray-400">
              Target: ${signal.targetLevel.toFixed(4)} 
              {signal.currentPrice && (
                <span className="ml-2">
                  ({((signal.targetLevel - signal.currentPrice) / signal.currentPrice * 100).toFixed(1)}%)
                </span>
              )}
            </div>
          )}

          {signal.reasoning && signal.reasoning.length > 0 && (
            <div className="text-xs text-gray-500 bg-gray-900/50 rounded p-2 max-h-20 overflow-y-auto">
              {signal.reasoning[0]} {/* Show first reasoning point */}
              {signal.reasoning.length > 1 && (
                <span className="text-gray-600"> +{signal.reasoning.length - 1} more</span>
              )}
            </div>
          )}
        </div>
      </div>
    );
  };

  return (
    <div className="bg-gray-700 rounded-lg p-4">
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-2">
          <Target className="h-5 w-5 text-blue-400" />
          <h3 className="text-lg font-semibold text-white">Strategy Signals</h3>
        </div>
        <div className="text-xs text-gray-400">
          Last update: {lastUpdate.toLocaleTimeString()}
        </div>
      </div>

      {loading ? (
        <div className="space-y-4">
          {/* Loading skeleton */}
          {[1, 2].map((i) => (
            <div key={i} className="space-y-3">
              <div className="h-4 bg-gray-600 rounded animate-pulse w-24"></div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {[1, 2, 3, 4].map((j) => (
                  <div key={j} className="bg-gray-800 rounded-lg p-3 border border-gray-700">
                    <div className="space-y-2">
                      <div className="h-4 bg-gray-600 rounded animate-pulse w-32"></div>
                      <div className="h-3 bg-gray-700 rounded animate-pulse w-20"></div>
                      <div className="h-8 bg-gray-700 rounded animate-pulse"></div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      ) : strategyData.size === 0 ? (
        <div className="text-center py-8">
          <Activity className="h-8 w-8 text-gray-500 mx-auto mb-2" />
          <p className="text-gray-500">No strategy signals available</p>
          <p className="text-xs text-gray-600 mt-1">Signals will appear when the trading agent is active</p>
        </div>
      ) : (
        <div className="space-y-4">
          {Array.from(strategyData.entries()).map(([symbol, signals]) => (
            <div key={symbol} className="space-y-3">
              <div className="text-sm font-medium text-gray-300 border-b border-gray-600 pb-1">
                {symbol.toUpperCase()} Strategies
              </div>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {renderSignalCard(symbol, 'liquidity', signals.liquidity, 'Liquidity Hunt')}
                {renderSignalCard(symbol, 'smartMoney', signals.smartMoney, 'Smart Money')}
                {renderSignalCard(symbol, 'volumeProfile', signals.volumeProfile, 'Volume Profile')}
                {renderSignalCard(symbol, 'microstructure', signals.microstructure, 'Microstructure')}
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="mt-4 pt-3 border-t border-gray-600">
        <div className="flex items-center justify-between text-xs text-gray-500">
          <div className="flex items-center space-x-4">
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-green-400" />
              <span>High Confidence (70%+)</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-yellow-400" />
              <span>Medium (50-69%)</span>
            </div>
            <div className="flex items-center space-x-1">
              <div className="w-2 h-2 rounded-full bg-red-400" />
              <span>Low (&lt;50%)</span>
            </div>
          </div>
          <div className="text-gray-600">
            Real-time strategy analysis
          </div>
        </div>
      </div>
    </div>
  );
};

export default StrategySignals;