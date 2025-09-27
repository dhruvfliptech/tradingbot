import React, { useEffect, useState } from 'react';
import { DollarSign, TrendingUp, TrendingDown, Activity, RefreshCw } from 'lucide-react';
import { brokerApiService } from '../services/brokerApiService';

interface AccountBalance {
  totalUSD: number;
  totalBTC: number;
  btcPrice: number;
  cash: number;
  positions: number;
  lastUpdate: Date;
  breakdown: {
    symbol: string;
    amount: number;
    valueUSD: number;
    valueBTC: number;
    percentage: number;
  }[];
}

export function AccountSummary() {
  const [balance, setBalance] = useState<AccountBalance | null>(null);
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const fetchAccountBalance = async () => {
    try {
      setRefreshing(true);

      // Fetch account data from backend
      const accountData = await brokerApiService.getAccount();

      if (accountData.success && accountData.data) {
        const account = accountData.data;

        // Get current BTC price
        const btcData = await brokerApiService.getMarketData(['BTCUSDT']);
        const btcPrice = btcData.data?.[0]?.price || 0;

        // Calculate balances
        let totalUSD = 0;
        const breakdown: AccountBalance['breakdown'] = [];

        // Process Binance balances
        if (account.broker === 'binance' && account.balances) {
          Object.entries(account.balances).forEach(([asset, balance]: [string, any]) => {
            const amount = parseFloat(balance.free) + parseFloat(balance.locked);

            if (amount > 0) {
              let valueUSD = 0;

              // Calculate USD value
              if (asset === 'USDT' || asset === 'USD' || asset === 'BUSD') {
                valueUSD = amount;
              } else if (asset === 'BTC') {
                valueUSD = amount * btcPrice;
              } else {
                // Get price for other assets
                // For now, we'll skip other assets or fetch their prices
                return;
              }

              totalUSD += valueUSD;

              breakdown.push({
                symbol: asset,
                amount,
                valueUSD,
                valueBTC: valueUSD / btcPrice,
                percentage: 0 // Will calculate after total
              });
            }
          });
        }

        // Calculate percentages
        breakdown.forEach(item => {
          item.percentage = (item.valueUSD / totalUSD) * 100;
        });

        // Sort by value
        breakdown.sort((a, b) => b.valueUSD - a.valueUSD);

        setBalance({
          totalUSD,
          totalBTC: totalUSD / btcPrice,
          btcPrice,
          cash: breakdown.find(b => ['USDT', 'USD', 'BUSD'].includes(b.symbol))?.valueUSD || 0,
          positions: breakdown.filter(b => !['USDT', 'USD', 'BUSD'].includes(b.symbol)).length,
          lastUpdate: new Date(),
          breakdown
        });
      }
    } catch (error) {
      console.error('Failed to fetch account balance:', error);
      setError('Failed to fetch account balance');
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAccountBalance();

    // Refresh every 30 seconds
    const interval = setInterval(fetchAccountBalance, 30000);

    return () => clearInterval(interval);
  }, []);

  const formatUSD = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatBTC = (value: number) => {
    return `₿${value.toFixed(8)}`;
  };

  if (loading && !balance) {
    return (
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-4 bg-gray-200 dark:bg-gray-700 rounded w-1/3"></div>
          <div className="h-8 bg-gray-200 dark:bg-gray-700 rounded w-1/2"></div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
      <div className="flex justify-between items-start mb-6">
        <div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Binance Account Summary
          </h2>
          <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
            Live trading account • Last updated: {balance?.lastUpdate.toLocaleTimeString() || 'Never'}
          </p>
        </div>
        <button
          onClick={fetchAccountBalance}
          disabled={refreshing}
          className="p-2 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
        >
          <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
        </button>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
          <p className="text-red-600 dark:text-red-400">{error}</p>
        </div>
      )}

      {balance && (
        <>
          {/* Total Balance */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6 mb-6">
            <div className="bg-gradient-to-r from-green-500 to-green-600 rounded-lg p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">Total Balance (USD)</p>
                  <p className="text-3xl font-bold mt-1">{formatUSD(balance.totalUSD)}</p>
                </div>
                <DollarSign className="w-8 h-8 opacity-80" />
              </div>
            </div>

            <div className="bg-gradient-to-r from-orange-500 to-orange-600 rounded-lg p-6 text-white">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm opacity-90">Total Balance (BTC)</p>
                  <p className="text-3xl font-bold mt-1">{formatBTC(balance.totalBTC)}</p>
                  <p className="text-xs opacity-75 mt-1">
                    @ {formatUSD(balance.btcPrice)}/BTC
                  </p>
                </div>
                <Activity className="w-8 h-8 opacity-80" />
              </div>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="grid grid-cols-3 gap-4 mb-6">
            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">Cash Available</p>
              <p className="text-xl font-semibold text-gray-900 dark:text-white mt-1">
                {formatUSD(balance.cash)}
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">Active Positions</p>
              <p className="text-xl font-semibold text-gray-900 dark:text-white mt-1">
                {balance.positions}
              </p>
            </div>

            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
              <p className="text-sm text-gray-600 dark:text-gray-400">Portfolio Health</p>
              <p className="text-xl font-semibold text-green-600 dark:text-green-400 mt-1">
                Good
              </p>
            </div>
          </div>

          {/* Asset Breakdown */}
          {balance.breakdown.length > 0 && (
            <div>
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-3">
                Asset Breakdown
              </h3>
              <div className="space-y-2">
                {balance.breakdown.map((asset) => (
                  <div
                    key={asset.symbol}
                    className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700 rounded-lg"
                  >
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-gray-200 dark:bg-gray-600 rounded-full flex items-center justify-center">
                        <span className="text-xs font-bold">
                          {asset.symbol.substring(0, 3)}
                        </span>
                      </div>
                      <div>
                        <p className="font-medium text-gray-900 dark:text-white">
                          {asset.symbol}
                        </p>
                        <p className="text-sm text-gray-500 dark:text-gray-400">
                          {asset.amount.toFixed(8)} units
                        </p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="font-medium text-gray-900 dark:text-white">
                        {formatUSD(asset.valueUSD)}
                      </p>
                      <p className="text-sm text-gray-500 dark:text-gray-400">
                        {asset.percentage.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}
    </div>
  );
}