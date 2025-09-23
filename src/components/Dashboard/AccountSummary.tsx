import React from 'react';
import { DollarSign, Activity, Bitcoin } from 'lucide-react';
import { Account } from '../../types/trading';

interface AccountSummaryProps {
  account: Account | null;
  btcUsd?: number; // optional BTC/USD price for value estimate
}

export const AccountSummary: React.FC<AccountSummaryProps> = ({ account, btcUsd }) => {
  
  if (!account) {
    return (
      <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded w-1/2 mb-4"></div>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            {[...Array(4)].map((_, i) => (
              <div key={i} className="h-16 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Show broker truth only (no demo overlays)
  const adjustedPortfolioValue = account.portfolio_value;
  const usdCash = account.available_balance;
  const btcQty = account.balance_btc;
  const btcUsdValue = btcUsd ? (btcQty * btcUsd) : null;

  const cards = [
    {
      title: 'Portfolio Value',
      value: `$${adjustedPortfolioValue.toLocaleString()}`,
      icon: DollarSign,
      color: 'text-blue-400',
    },
    {
      title: 'Available Balance',
      value: `$${account.available_balance.toLocaleString()}`,
      icon: Activity,
      color: 'text-green-400',
    },
    {
      title: 'BTC Balance',
      value: `₿${account.balance_btc.toFixed(4)}`,
      icon: Bitcoin,
      color: 'text-yellow-400',
    },
  ];

  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg border border-gray-700 p-6 pr-2">
      <h2 className="text-xl font-bold text-white mb-6">Account Summary</h2>
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
        {cards.map((card, index) => (
          <div key={index} className="bg-gray-700 rounded-lg p-3 sm:p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-400 text-xs sm:text-sm">{card.title}</p>
                <p className={`text-sm sm:text-lg font-bold ${card.color}`}>{card.value}</p>
              </div>
              <card.icon className={`h-5 w-5 sm:h-6 sm:w-6 ${card.color}`} />
            </div>
          </div>
        ))}
      </div>

      {/* Simple breakdown under the cards */}
      <div className="mt-4 grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-gray-400">USD Cash</div>
          <div className="text-white font-semibold">${usdCash.toLocaleString()}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-3">
          <div className="text-gray-400">BTC Holdings</div>
          <div className="text-white font-semibold">
            {btcQty.toFixed(6)} BTC{btcUsdValue !== null ? ` (≈ $${btcUsdValue.toLocaleString(undefined, { maximumFractionDigits: 0 })})` : ''}
          </div>
        </div>
      </div>
    </div>
  );
};