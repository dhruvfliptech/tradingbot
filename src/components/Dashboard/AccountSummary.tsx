import React from 'react';
import { DollarSign, TrendingUp, TrendingDown, Activity, Bitcoin, Bot } from 'lucide-react';
import { Account } from '../../types/trading';
import { getTradingBotReport } from '../../services/tradingBotReportService';

interface AccountSummaryProps {
  account: Account | null;
}

export const AccountSummary: React.FC<AccountSummaryProps> = ({ account }) => {
  const botReport = getTradingBotReport();
  
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

  // Use realistic portfolio values that reflect bot trading performance
  const botPnL = botReport.totalPnL;
  const adjustedPortfolioValue = account.portfolio_value + botPnL;
  const totalReturn = adjustedPortfolioValue - account.portfolio_value;
  const totalReturnPercent = (totalReturn / account.portfolio_value) * 100;

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
    {
      title: 'Bot P&L (3 days)',
      value: `${botPnL >= 0 ? '+' : ''}$${botPnL.toLocaleString()}`,
      icon: Bot,
      color: botPnL >= 0 ? 'text-green-400' : 'text-red-400',
    },
  ];

  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
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
    </div>
  );
};