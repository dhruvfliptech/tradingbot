import React, { useState, useEffect } from 'react';
import { getTradingBotReport, TradingBotReport, DailyPerformance, Trade } from '../../services/tradingBotReportService';

interface TradingBotReportProps {
  className?: string;
}

const TradingBotReportComponent: React.FC<TradingBotReportProps> = ({ className = '' }) => {
  const [report, setReport] = useState<TradingBotReport | null>(null);
  const [selectedDay, setSelectedDay] = useState<string>('');
  const [showTradeDetails, setShowTradeDetails] = useState(false);

  useEffect(() => {
    const botReport = getTradingBotReport();
    setReport(botReport);
    if (botReport.dailyPerformance.length > 0) {
      setSelectedDay(botReport.dailyPerformance[botReport.dailyPerformance.length - 1].date);
    }
  }, []);

  if (!report) {
    return (
      <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
        <div className="animate-pulse">
          <div className="h-6 bg-gray-700 rounded mb-4 w-48"></div>
          <div className="space-y-3">
            <div className="h-4 bg-gray-700 rounded w-3/4"></div>
            <div className="h-4 bg-gray-700 rounded w-1/2"></div>
          </div>
        </div>
      </div>
    );
  }

  const selectedDayData = report.dailyPerformance.find(day => day.date === selectedDay);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(value);
  };

  const formatDate = (dateStr: string) => {
    return new Date(dateStr).toLocaleDateString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric'
    });
  };

  const getPnLColor = (pnl: number) => {
    if (pnl > 0) return 'text-green-400';
    if (pnl < 0) return 'text-red-400';
    return 'text-gray-400';
  };

  return (
    <div className={`bg-gray-800 rounded-lg p-6 ${className}`}>
      {/* Header */}
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-2xl font-bold text-white mb-2">Trading Bot Performance Report</h2>
          <p className="text-gray-400">{report.period}</p>
        </div>
        <div className="text-right">
          <div className="text-sm text-gray-400">Total P&L</div>
          <div className={`text-2xl font-bold ${getPnLColor(report.totalPnL)}`}>
            {formatCurrency(report.totalPnL)}
          </div>
        </div>
      </div>

      {/* Overall Statistics */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Total Trades</div>
          <div className="text-xl font-bold text-white">{report.totalTrades}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Win Rate</div>
          <div className="text-xl font-bold text-green-400">{report.winRate}%</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Total Volume</div>
          <div className="text-xl font-bold text-white">{formatCurrency(report.totalVolume)}</div>
        </div>
        <div className="bg-gray-700 rounded-lg p-4">
          <div className="text-sm text-gray-400 mb-1">Avg Daily P&L</div>
          <div className={`text-xl font-bold ${getPnLColor(report.totalPnL / report.dailyPerformance.length)}`}>
            {formatCurrency(report.totalPnL / report.dailyPerformance.length)}
          </div>
        </div>
      </div>

      {/* Daily Performance */}
      <div className="mb-6">
        <h3 className="text-lg font-semibold text-white mb-4">Daily Performance</h3>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {report.dailyPerformance.map((day) => (
            <div
              key={day.date}
              className={`bg-gray-700 rounded-lg p-4 cursor-pointer transition-colors ${
                selectedDay === day.date ? 'ring-2 ring-blue-500' : 'hover:bg-gray-600'
              }`}
              onClick={() => setSelectedDay(day.date)}
            >
              <div className="text-sm text-gray-400 mb-2">{formatDate(day.date)}</div>
              <div className={`text-lg font-bold mb-1 ${getPnLColor(day.totalPnL)}`}>
                {formatCurrency(day.totalPnL)}
              </div>
              <div className="text-sm text-gray-400">
                {day.trades.length} trades • {day.winRate}% win rate
              </div>
              <div className="text-xs text-gray-500 mt-1">
                Volume: {formatCurrency(day.totalVolume)}
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Current Positions */}
      {report.currentPositions.length > 0 && (
        <div className="mb-6">
          <h3 className="text-lg font-semibold text-white mb-4">Current Positions</h3>
          <div className="bg-gray-700 rounded-lg overflow-hidden">
            <table className="w-full">
              <thead className="bg-gray-600">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-medium text-gray-300">Symbol</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-gray-300">Quantity</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-gray-300">Avg Price</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-gray-300">Current Price</th>
                  <th className="px-4 py-3 text-right text-sm font-medium text-gray-300">Unrealized P&L</th>
                </tr>
              </thead>
              <tbody>
                {report.currentPositions.map((position) => (
                  <tr key={position.symbol} className="border-t border-gray-600">
                    <td className="px-4 py-3 text-white font-medium">{position.symbol}</td>
                    <td className="px-4 py-3 text-right text-white">{position.quantity}</td>
                    <td className="px-4 py-3 text-right text-white">{formatCurrency(position.avgPrice)}</td>
                    <td className="px-4 py-3 text-right text-white">{formatCurrency(position.currentPrice)}</td>
                    <td className={`px-4 py-3 text-right font-medium ${getPnLColor(position.unrealizedPnL)}`}>
                      {formatCurrency(position.unrealizedPnL)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Selected Day Details */}
      {selectedDayData && (
        <div>
          <div className="flex justify-between items-center mb-4">
            <h3 className="text-lg font-semibold text-white">
              {formatDate(selectedDay)} Trade Details
            </h3>
            <button
              onClick={() => setShowTradeDetails(!showTradeDetails)}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors text-sm"
            >
              {showTradeDetails ? 'Hide Details' : 'Show Details'}
            </button>
          </div>

          {/* Day Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">P&L</div>
              <div className={`text-lg font-bold ${getPnLColor(selectedDayData.totalPnL)}`}>
                {formatCurrency(selectedDayData.totalPnL)}
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Best Trade</div>
              <div className="text-lg font-bold text-green-400">
                {formatCurrency(selectedDayData.bestTrade)}
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Worst Trade</div>
              <div className="text-lg font-bold text-red-400">
                {formatCurrency(selectedDayData.worstTrade)}
              </div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-xs text-gray-400 mb-1">Volume</div>
              <div className="text-lg font-bold text-white">
                {formatCurrency(selectedDayData.totalVolume)}
              </div>
            </div>
          </div>

          {/* Trade Details Table */}
          {showTradeDetails && (
            <div className="bg-gray-700 rounded-lg overflow-hidden">
              <div className="max-h-96 overflow-y-auto">
                <table className="w-full">
                  <thead className="bg-gray-600 sticky top-0">
                    <tr>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-300">Time</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-300">Symbol</th>
                      <th className="px-3 py-2 text-center text-xs font-medium text-gray-300">Type</th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-300">Quantity</th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-300">Price</th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-300">Value</th>
                      <th className="px-3 py-2 text-right text-xs font-medium text-gray-300">P&L</th>
                      <th className="px-3 py-2 text-left text-xs font-medium text-gray-300">Strategy</th>
                    </tr>
                  </thead>
                  <tbody>
                    {selectedDayData.trades.map((trade) => (
                      <tr key={trade.id} className="border-t border-gray-600 hover:bg-gray-600">
                        <td className="px-3 py-2 text-xs text-gray-300">
                          {new Date(trade.timestamp).toLocaleTimeString('en-US', { 
                            hour: '2-digit', 
                            minute: '2-digit' 
                          })}
                        </td>
                        <td className="px-3 py-2 text-xs text-white font-medium">{trade.symbol}</td>
                        <td className="px-3 py-2 text-center">
                          <span className={`px-2 py-1 rounded text-xs font-medium ${
                            trade.type === 'BUY' 
                              ? 'bg-green-600 text-white' 
                              : 'bg-red-600 text-white'
                          }`}>
                            {trade.type}
                          </span>
                        </td>
                        <td className="px-3 py-2 text-right text-xs text-white">{trade.quantity}</td>
                        <td className="px-3 py-2 text-right text-xs text-white">{formatCurrency(trade.price)}</td>
                        <td className="px-3 py-2 text-right text-xs text-white">{formatCurrency(trade.value)}</td>
                        <td className={`px-3 py-2 text-right text-xs font-medium ${
                          trade.pnl ? getPnLColor(trade.pnl) : 'text-gray-400'
                        }`}>
                          {trade.pnl ? formatCurrency(trade.pnl) : '-'}
                        </td>
                        <td className="px-3 py-2 text-xs text-gray-400">{trade.strategy}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default TradingBotReportComponent;
