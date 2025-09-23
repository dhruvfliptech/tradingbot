import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, AlertTriangle } from 'lucide-react';
import { FearGreedIndex as FearGreedIndexData } from '../../types/trading';
import { coinGeckoService } from '../../services/coinGeckoService';

export const FearGreedIndexWidget: React.FC = () => {
  const [fearGreedData, setFearGreedData] = useState<FearGreedIndexData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchFearGreedIndex = async () => {
      try {
        const data = await coinGeckoService.getFearGreedIndex();
        setFearGreedData(data);
      } catch (error) {
        console.error('Failed to fetch Fear & Greed Index:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchFearGreedIndex();
    // Update every 5 minutes
    const interval = setInterval(fetchFearGreedIndex, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const getIndexColor = (value: number) => {
    if (value <= 25) return 'text-red-400';
    if (value <= 45) return 'text-orange-400';
    if (value <= 55) return 'text-yellow-400';
    if (value <= 75) return 'text-green-400';
    return 'text-green-500';
  };

  const getIndexIcon = (classification: string) => {
    const lowerClass = classification.toLowerCase();
    if (lowerClass.includes('fear')) return <TrendingDown className="h-5 w-5" />;
    if (lowerClass.includes('greed')) return <TrendingUp className="h-5 w-5" />;
    return <Minus className="h-5 w-5" />;
  };

  const getBackgroundGradient = (value: number) => {
    if (value <= 25) return 'from-red-900/20 to-red-800/20';
    if (value <= 45) return 'from-orange-900/20 to-orange-800/20';
    if (value <= 55) return 'from-yellow-900/20 to-yellow-800/20';
    if (value <= 75) return 'from-green-900/20 to-green-800/20';
    return 'from-green-900/20 to-green-700/20';
  };

  if (loading) {
    return (
      <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-3/4 mb-4"></div>
          <div className="h-16 bg-gray-700 rounded"></div>
        </div>
      </div>
    );
  }

  if (!fearGreedData) {
    return (
      <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
        <div className="flex items-center text-yellow-400 mb-2">
          <AlertTriangle className="h-5 w-5 mr-2" />
          <h3 className="text-lg font-semibold">Fear & Greed Index</h3>
        </div>
        <p className="text-gray-400">Unable to load index data</p>
      </div>
    );
  }

  return (
    <div className={`h-full overflow-y-auto pr-2 bg-gradient-to-br ${getBackgroundGradient(fearGreedData.value)} bg-gray-800 rounded-lg p-6 border border-gray-700`}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Fear & Greed Index</h3>
        <div className={`${getIndexColor(fearGreedData.value)}`}>
          {getIndexIcon(fearGreedData.value_classification)}
        </div>
      </div>
      
      <div className="text-center">
        <div className={`text-4xl font-bold mb-2 ${getIndexColor(fearGreedData.value)}`}>
          {fearGreedData.value}
        </div>
        <div className={`text-lg font-medium mb-4 ${getIndexColor(fearGreedData.value)}`}>
          {fearGreedData.value_classification}
        </div>
        
        {/* Progress Bar */}
        <div className="w-full bg-gray-700 rounded-full h-3 mb-4">
          <div
            className={`h-3 rounded-full transition-all duration-500 ${
              fearGreedData.value <= 25 ? 'bg-red-500' :
              fearGreedData.value <= 45 ? 'bg-orange-500' :
              fearGreedData.value <= 55 ? 'bg-yellow-500' :
              fearGreedData.value <= 75 ? 'bg-green-500' : 'bg-green-400'
            }`}
            style={{ width: `${fearGreedData.value}%` }}
          ></div>
        </div>
        
        {/* Scale Labels */}
        <div className="flex justify-between text-xs text-gray-400 mb-3">
          <span>Extreme Fear</span>
          <span>Neutral</span>
          <span>Extreme Greed</span>
        </div>
        
        <p className="text-xs text-gray-400">
          Last updated: {new Date(parseInt(fearGreedData.timestamp) * 1000).toLocaleString()}
        </p>
      </div>
    </div>
  );
};

export default FearGreedIndexWidget;