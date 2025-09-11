import React from 'react';
import { TrendingUp, TrendingDown, Volume2 } from 'lucide-react';
import { CryptoData } from '../../types/trading';

interface MarketWatchlistProps {
  cryptoData: CryptoData[];
}

export const MarketWatchlist: React.FC<MarketWatchlistProps> = ({ cryptoData }) => {
  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <h2 className="text-xl font-bold text-white mb-6">Crypto Watchlist</h2>
      <div className="space-y-4">
        {cryptoData.map((crypto) => (
          <div key={crypto.symbol} className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <div className="flex items-center mb-2">
                  <img src={crypto.image} alt={crypto.name} className="w-6 h-6 sm:w-8 sm:h-8 rounded-full mr-2" />
                  <div>
                    <h3 className="text-white font-semibold text-sm sm:text-lg">{crypto.symbol}</h3>
                    <p className="text-xs text-gray-400">{crypto.name}</p>
                  </div>
                </div>
                <p className="text-lg sm:text-2xl font-bold text-white">${crypto.price.toLocaleString()}</p>
              </div>
              <div className="text-right">
                <div className={`flex items-center ${crypto.change >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {crypto.change >= 0 ? <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 mr-1" /> : <TrendingDown className="h-3 w-3 sm:h-4 sm:w-4 mr-1" />}
                  <span className="font-semibold text-sm sm:text-base">
                    {crypto.change >= 0 ? '+' : ''}${crypto.change.toFixed(2)}
                  </span>
                </div>
                <div className={`text-sm sm:text-base ${crypto.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                  {crypto.changePercent >= 0 ? '+' : ''}{crypto.changePercent.toFixed(2)}%
                </div>
              </div>
            </div>
            <div className="mt-3 grid grid-cols-2 sm:grid-cols-4 gap-2 sm:gap-4 text-xs sm:text-sm">
              <div>
                <p className="text-gray-400">Market Cap</p>
                <p className="text-white">${(crypto.market_cap / 1000000000).toFixed(1)}B</p>
              </div>
              <div>
                <p className="text-gray-400">24h High</p>
                <p className="text-white">${crypto.high.toLocaleString()}</p>
              </div>
              <div className="hidden sm:block">
                <p className="text-gray-400">24h Low</p>
                <p className="text-white">${crypto.low.toLocaleString()}</p>
              </div>
              <div className="hidden sm:block">
                <p className="text-gray-400 flex items-center">
                  <Volume2 className="h-3 w-3 mr-1" />
                  Volume
                </p>
                <p className="text-white">${(crypto.volume / 1000000000).toFixed(1)}B</p>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};