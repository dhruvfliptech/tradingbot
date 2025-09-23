import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Globe, Activity, Coins, Building2, Fuel } from 'lucide-react';
import { coinGeckoService } from '../../services/coinGeckoService';

interface GlobalMarketData {
  active_cryptocurrencies: number;
  markets: number;
  total_market_cap: { usd: number };
  total_volume: { usd: number };
  market_cap_percentage: { btc: number; eth: number };
  market_cap_change_percentage_24h_usd: number;
}

export const GlobalMarketHeader: React.FC = () => {
  const [globalData, setGlobalData] = useState<GlobalMarketData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchGlobalData = async () => {
      try {
        const data = await coinGeckoService.getGlobalMarketData();
        setGlobalData(data);
      } catch (error) {
        console.error('Failed to fetch global market data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchGlobalData();
    // Update every 5 minutes to reduce API calls and avoid rate limiting
    const interval = setInterval(fetchGlobalData, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, []);

  const formatNumber = (num: number): string => {
    if (num >= 1e12) return `$${(num / 1e12).toFixed(2)}T`;
    if (num >= 1e9) return `$${(num / 1e9).toFixed(2)}B`;
    if (num >= 1e6) return `$${(num / 1e6).toFixed(2)}M`;
    return `$${num.toLocaleString()}`;
  };

  const formatPercentage = (num: number): string => {
    return `${num >= 0 ? '+' : ''}${num.toFixed(1)}%`;
  };

  if (loading || !globalData) {
    return (
      <div className="bg-gray-800 border-b border-gray-700 py-2">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-center">
            <div className="animate-pulse text-gray-400 text-sm">Loading global market data...</div>
          </div>
        </div>
      </div>
    );
  }

  const marketCapChange = globalData.market_cap_change_percentage_24h_usd;
  const isPositive = marketCapChange >= 0;

  return (
    <div className="bg-gray-800 border-b border-gray-700 py-1 sm:py-2">
      <div className="max-w-[1600px] mx-auto px-2 sm:px-4 lg:px-8">
        <div className="flex items-center justify-start text-xs sm:text-sm text-gray-300 overflow-x-auto scrollbar-hide">
          <div className="flex items-center space-x-3 sm:space-x-6 min-w-max">
            {/* Coins */}
            <div className="flex items-center">
              <Coins className="h-3 w-3 sm:h-4 sm:w-4 mr-1 text-blue-400" />
              <span className="text-gray-400">Coins:</span>
              <span className="ml-1 text-white font-medium">
                {globalData.active_cryptocurrencies.toLocaleString()}
              </span>
            </div>

            {/* Exchanges */}
            <div className="hidden sm:flex items-center">
              <Building2 className="h-3 w-3 sm:h-4 sm:w-4 mr-1 text-green-400" />
              <span className="text-gray-400">Exchanges:</span>
              <span className="ml-1 text-white font-medium">
                {globalData.markets.toLocaleString()}
              </span>
            </div>

            {/* Market Cap */}
            <div className="flex items-center">
              <Globe className="h-3 w-3 sm:h-4 sm:w-4 mr-1 text-purple-400" />
              <span className="text-gray-400 hidden sm:inline">Market Cap:</span>
              <span className="text-gray-400 sm:hidden">Cap:</span>
              <span className="ml-1 text-white font-medium">
                {formatNumber(globalData.total_market_cap.usd)}
              </span>
              <span className={`ml-1 flex items-center ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                {isPositive ? <TrendingUp className="h-2 w-2 sm:h-3 sm:w-3 mr-1" /> : <TrendingDown className="h-2 w-2 sm:h-3 sm:w-3 mr-1" />}
                {formatPercentage(marketCapChange)}
              </span>
            </div>

            {/* 24h Volume */}
            <div className="flex items-center">
              <Activity className="h-3 w-3 sm:h-4 sm:w-4 mr-1 text-yellow-400" />
              <span className="text-gray-400">Vol:</span>
              <span className="ml-1 text-white font-medium">
                {formatNumber(globalData.total_volume.usd)}
              </span>
            </div>

            {/* Dominance */}
            <div className="flex items-center">
              <span className="text-gray-400">Dominance:</span>
              <span className="ml-1 text-orange-400 font-medium">
                BTC {globalData.market_cap_percentage.btc.toFixed(1)}%
              </span>
              <span className="ml-1 sm:ml-2 text-blue-400 font-medium">
                ETH {globalData.market_cap_percentage.eth.toFixed(1)}%
              </span>
            </div>

            {/* Gas Price (Mock for now since CoinGecko doesn't provide this) */}
            <div className="hidden lg:flex items-center">
              <Fuel className="h-3 w-3 sm:h-4 sm:w-4 mr-1 text-indigo-400" />
              <span className="text-gray-400">⛽ Gas:</span>
              <span className="ml-1 text-white font-medium">
                {Math.floor(Math.random() * 50 + 20)} GWEI
              </span>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};