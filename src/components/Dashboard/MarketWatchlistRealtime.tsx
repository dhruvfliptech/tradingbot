import React from 'react';
import { TrendingUp, TrendingDown, Volume2, Wifi, WifiOff } from 'lucide-react';
import { useRealtimeMarketData } from '../../hooks/useSocket';

interface MarketWatchlistRealtimeProps {
  symbols?: string[];
}

export const MarketWatchlistRealtime: React.FC<MarketWatchlistRealtimeProps> = ({
  symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA']
}) => {
  const { marketData, lastUpdate, isConnected } = useRealtimeMarketData(symbols);

  const formatPrice = (price: number) => {
    if (price >= 1000) return `$${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
    if (price >= 1) return `$${price.toFixed(2)}`;
    return `$${price.toFixed(6)}`;
  };

  const formatVolume = (volume: number) => {
    if (volume >= 1000000000) return `$${(volume / 1000000000).toFixed(1)}B`;
    if (volume >= 1000000) return `$${(volume / 1000000).toFixed(1)}M`;
    if (volume >= 1000) return `$${(volume / 1000).toFixed(1)}K`;
    return `$${volume.toFixed(0)}`;
  };

  const formatMarketCap = (marketCap: number) => {
    if (marketCap >= 1000000000) return `$${(marketCap / 1000000000).toFixed(1)}B`;
    if (marketCap >= 1000000) return `$${(marketCap / 1000000).toFixed(1)}M`;
    return `$${(marketCap / 1000).toFixed(1)}K`;
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <div className="flex items-center justify-between mb-6">
        <h2 className="text-xl font-bold text-white">Live Market Data</h2>
        <div className="flex items-center space-x-2">
          <div className="flex items-center text-xs text-gray-400">
            {isConnected ? (
              <>
                <Wifi className="h-3 w-3 mr-1 text-green-400" />
                <span className="text-green-400">Live</span>
              </>
            ) : (
              <>
                <WifiOff className="h-3 w-3 mr-1 text-red-400" />
                <span className="text-red-400">Offline</span>
              </>
            )}
          </div>
          {lastUpdate && (
            <div className="text-xs text-gray-500">
              {new Date(lastUpdate).toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>

      <div className="space-y-4">
        {symbols.map((symbol) => {
          const data = marketData.get(symbol);

          if (!data) {
            return (
              <div key={symbol} className="bg-gray-700 rounded-lg p-4">
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="text-white font-semibold text-sm sm:text-lg">{symbol}</h3>
                    <p className="text-xs text-gray-400">Loading...</p>
                  </div>
                </div>
              </div>
            );
          }

          const changePercent = data.changePercent || 0;
          const isPositive = changePercent >= 0;

          return (
            <div key={symbol} className="bg-gray-700 rounded-lg p-4 transition-all hover:bg-gray-600">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center mb-2">
                    {data.image && (
                      <img src={data.image} alt={data.name} className="w-6 h-6 sm:w-8 sm:h-8 rounded-full mr-2" />
                    )}
                    <div>
                      <h3 className="text-white font-semibold text-sm sm:text-lg">{symbol}</h3>
                      <p className="text-xs text-gray-400">{data.name || symbol}</p>
                    </div>
                  </div>
                  <p className="text-lg sm:text-2xl font-bold text-white">
                    {formatPrice(data.price)}
                  </p>
                </div>
                <div className="text-right">
                  <div className={`flex items-center ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                    {isPositive ? (
                      <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 mr-1" />
                    ) : (
                      <TrendingDown className="h-3 w-3 sm:h-4 sm:w-4 mr-1" />
                    )}
                    <span className="font-semibold text-sm sm:text-base">
                      {isPositive ? '+' : ''}{changePercent.toFixed(2)}%
                    </span>
                  </div>
                  {data.change !== undefined && (
                    <div className={`text-sm ${isPositive ? 'text-green-400' : 'text-red-400'}`}>
                      {isPositive ? '+' : ''}${Math.abs(data.change).toFixed(2)}
                    </div>
                  )}
                </div>
              </div>

              <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4 text-xs sm:text-sm">
                {data.marketCap && (
                  <div>
                    <p className="text-gray-400">Market Cap</p>
                    <p className="text-white">{formatMarketCap(data.marketCap)}</p>
                  </div>
                )}
                {data.high24h && (
                  <div>
                    <p className="text-gray-400">24h High</p>
                    <p className="text-white">{formatPrice(data.high24h)}</p>
                  </div>
                )}
                {data.low24h && (
                  <div className="hidden sm:block">
                    <p className="text-gray-400">24h Low</p>
                    <p className="text-white">{formatPrice(data.low24h)}</p>
                  </div>
                )}
                {data.volume24h && (
                  <div className="hidden sm:block">
                    <p className="text-gray-400 flex items-center">
                      <Volume2 className="h-3 w-3 mr-1" />
                      Volume
                    </p>
                    <p className="text-white">{formatVolume(data.volume24h)}</p>
                  </div>
                )}
              </div>

              {/* Real-time indicator */}
              {data.lastUpdate && (
                <div className="mt-2 flex items-center justify-end">
                  <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse mr-1"></div>
                  <span className="text-xs text-gray-500">
                    Updated {new Date(data.lastUpdate).toLocaleTimeString()}
                  </span>
                </div>
              )}
            </div>
          );
        })}
      </div>

      {marketData.size === 0 && !isConnected && (
        <div className="text-center py-8">
          <WifiOff className="h-8 w-8 text-gray-500 mx-auto mb-2" />
          <p className="text-gray-400">Connecting to real-time market data...</p>
        </div>
      )}
    </div>
  );
};