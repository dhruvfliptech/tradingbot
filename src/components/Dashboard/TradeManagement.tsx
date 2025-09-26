import React, { useState, useEffect, useCallback } from 'react';
import {
  TrendingUp,
  TrendingDown,
  X,
  Edit3,
  AlertTriangle,
  DollarSign,
  Calculator,
  Target,
  StopCircle,
  Plus,
  Minus,
  Check,
} from 'lucide-react';
import { tradingProviderService } from '../../services/tradingProviderService';
import { tradingBotService } from '../../services/tradingBotService';
import { tradeHistoryService } from '../../services/persistence/tradeHistoryService';
import { useTradingProvider } from '../../hooks/useTradingProvider';
import { Position as BrokerPosition } from '../../types/trading';

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  quantity: number;
  entry_price: number;
  current_price: number;
  pnl: number;
  pnl_percent: number;
  stop_loss?: number;
  take_profit?: number;
  status: 'open' | 'closing';
}

interface ManualTradeForm {
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  order_type: 'market' | 'limit';
  limit_price?: number;
}

export const TradeManagement: React.FC = () => {
  const { activeProvider, providers } = useTradingProvider();
  const providerMeta = providers.find((meta) => meta.id === activeProvider);

  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [editingPosition, setEditingPosition] = useState<string | null>(null);
  const [manualTrade, setManualTrade] = useState<ManualTradeForm>({
    symbol: '',
    side: 'buy',
    quantity: 0,
    order_type: 'market',
  });
  const [riskPercent, setRiskPercent] = useState(2);
  const [accountValue, setAccountValue] = useState(50000);
  const [showCloseAllConfirm, setShowCloseAllConfirm] = useState(false);
  const [isSubmittingTrade, setIsSubmittingTrade] = useState(false);
  const [notifications, setNotifications] = useState<string[]>([]);

  const formatPositions = (brokerPositions: BrokerPosition[]): Position[] => {
    return brokerPositions.map((pos) => {
      const normalizedSymbol = tradingProviderService.normalizeSymbol(pos.symbol);
      const quantity = Math.abs(parseFloat(pos.qty));
      const marketValue = Number(pos.market_value) || 0;
      const costBasis = Number(pos.cost_basis) || 0;
      const entryPrice = quantity > 0 ? costBasis / quantity : 0;
      const currentPrice = quantity > 0 ? marketValue / quantity : 0;
      const unrealized = Number(pos.unrealized_pl ?? marketValue - costBasis);
      const unrealizedPercent =
        pos.unrealized_plpc !== undefined && pos.unrealized_plpc !== null
          ? Number(pos.unrealized_plpc) * 100
          : costBasis > 0
            ? (unrealized / costBasis) * 100
            : 0;

      return {
        id: `${normalizedSymbol}-${pos.side}`,
        symbol: normalizedSymbol,
        side: pos.side,
        quantity,
        entry_price: entryPrice,
        current_price: currentPrice,
        pnl: unrealized,
        pnl_percent: unrealizedPercent,
        status: 'open',
      };
    });
  };

  const loadPositions = useCallback(async () => {
    try {
      const brokerPositions = await tradingProviderService.getPositions();
      setPositions(formatPositions(brokerPositions));
    } catch (error) {
      console.error('Error loading positions:', error);
    } finally {
      setIsLoading(false);
    }
  }, []);

  const loadAccountInfo = useCallback(async () => {
    try {
      const account = await tradingProviderService.getAccount();
      setAccountValue(account.portfolio_value);
    } catch (error) {
      console.error('Error loading account info:', error);
    }
  }, []);

  useEffect(() => {
    setIsLoading(true);
    loadPositions();
    loadAccountInfo();
    const interval = setInterval(loadPositions, 10000);
    return () => clearInterval(interval);
  }, [activeProvider, loadPositions, loadAccountInfo]);

  const calculatePositionSize = (price: number): number => {
    const riskAmount = accountValue * (riskPercent / 100);
    return Math.floor((riskAmount / price) * 100) / 100;
  };

  const closePosition = async (positionId: string) => {
    const position = positions.find((p) => p.id === positionId);
    if (!position) return;

    try {
      setPositions((prev) =>
        prev.map((p) => (p.id === positionId ? { ...p, status: 'closing' } : p))
      );

      await tradingProviderService.placeOrder({
        symbol: position.symbol,
        qty: position.quantity,
        side: position.side === 'long' ? 'sell' : 'buy',
        order_type: 'market',
        time_in_force: 'day',
      });

      addNotification(`Position ${position.symbol} closing order submitted`);
      setTimeout(loadPositions, 2000);
    } catch (error) {
      console.error('Error closing position:', error);
      const message = error instanceof Error ? error.message : 'Unknown error';
      addNotification(`Error closing ${position.symbol}: ${message}`);
      setPositions((prev) =>
        prev.map((p) => (p.id === positionId ? { ...p, status: 'open' } : p))
      );
    }
  };

  const updateStopLoss = async (positionId: string, stopLoss: number) => {
    const position = positions.find((p) => p.id === positionId);
    if (!position) return;

    setPositions((prev) =>
      prev.map((p) => (p.id === positionId ? { ...p, stop_loss: stopLoss } : p))
    );
    addNotification(`Stop loss updated for ${position.symbol}: $${stopLoss.toFixed(2)}`);
  };

  const updateTakeProfit = async (positionId: string, takeProfit: number) => {
    const position = positions.find((p) => p.id === positionId);
    if (!position) return;

    setPositions((prev) =>
      prev.map((p) => (p.id === positionId ? { ...p, take_profit: takeProfit } : p))
    );
    addNotification(`Take profit updated for ${position.symbol}: $${takeProfit.toFixed(2)}`);
  };

  const closeAllPositions = async () => {
    try {
      for (const position of positions) {
        await closePosition(position.id);
      }
      addNotification('All positions closing...');
      setShowCloseAllConfirm(false);
    } catch (error) {
      console.error('Error closing all positions:', error);
      const message = error instanceof Error ? error.message : 'Unknown error';
      addNotification(`Error closing all positions: ${message}`);
    }
  };

  const submitManualTrade = async () => {
    if (!manualTrade.symbol || manualTrade.quantity <= 0) {
      addNotification('Please fill in all required fields');
      return;
    }

    setIsSubmittingTrade(true);
    const normalizedSymbol = tradingProviderService.normalizeSymbol(manualTrade.symbol.toUpperCase());

    try {
      const order = await tradingProviderService.placeOrder({
        symbol: normalizedSymbol,
        qty: manualTrade.quantity,
        side: manualTrade.side,
        order_type: manualTrade.order_type,
        time_in_force: 'gtc', // Use 'gtc' for crypto orders (required by Alpaca)
        limit_price: manualTrade.order_type === 'limit' ? manualTrade.limit_price ?? undefined : undefined,
      });

      await tradeHistoryService.recordTrade({
        symbol: normalizedSymbol,
        side: manualTrade.side,
        quantity: manualTrade.quantity,
        entry_price:
          order.filled_avg_price || manualTrade.limit_price || 0,
        execution_status: 'pending',
        confidence_score: 100,
        risk_reward_ratio: 1.0,
        position_size_percent:
          accountValue > 0
            ? ((manualTrade.limit_price || order.filled_avg_price || 0) * manualTrade.quantity * 100) /
              accountValue
            : 0,
        risk_amount: (manualTrade.limit_price || order.filled_avg_price || 0) * manualTrade.quantity,
        alpaca_order_id: order.id,
      });

      addNotification(`Manual ${manualTrade.side} order submitted for ${normalizedSymbol}`);

      setManualTrade({
        symbol: '',
        side: 'buy',
        quantity: 0,
        order_type: 'market',
      });

      setTimeout(loadPositions, 2000);
    } catch (error) {
      console.error('Error submitting manual trade:', error);
      const message = error instanceof Error ? error.message : 'Unknown error';
      addNotification(`Error submitting trade: ${message}`);
    }
    setIsSubmittingTrade(false);
  };

  const suggestedPositionSize =
    manualTrade.order_type === 'limit' && manualTrade.limit_price && manualTrade.limit_price > 0
      ? calculatePositionSize(manualTrade.limit_price)
      : 0;

  const addNotification = (message: string) => {
    setNotifications((prev) => [...prev, message]);
    setTimeout(() => {
      setNotifications((prev) => prev.slice(1));
    }, 5000);
  };

  if (isLoading) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-700 rounded w-1/4" />
          <div className="space-y-3">
            {[1, 2, 3].map((i) => (
              <div key={i} className="h-16 bg-gray-800 rounded" />
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 space-y-6 h-full overflow-y-auto">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-xl font-semibold text-white flex items-center">
            <TrendingUp className="w-6 h-6 mr-2 text-green-400" />
            Trade Management
          </h3>
          {providerMeta && (
            <span className="text-xs text-gray-400">Broker: {providerMeta.label}</span>
          )}
        </div>
        {positions.length > 0 && (
          <button
            onClick={() => setShowCloseAllConfirm(true)}
            className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-white text-sm font-medium transition-colors flex items-center"
          >
            <StopCircle className="w-4 h-4 mr-2" />
            Close All Positions
          </button>
        )}
      </div>

      {/* Positions Table */}
      <div className="grid gap-4">
        {positions.length === 0 ? (
          <div className="text-center py-10 border border-dashed border-gray-700 rounded-lg">
            <TrendingDown className="w-10 h-10 text-gray-500 mx-auto mb-2" />
            <p className="text-gray-400">No active positions</p>
          </div>
        ) : (
          positions.map((position) => (
            <div key={position.id} className="bg-gray-800 rounded-lg p-4 border border-gray-700">
              <div className="flex items-center justify-between">
                <div>
                  <div className="flex items-center space-x-2">
                    <span className="text-white font-semibold text-lg">{position.symbol}</span>
                    <span
                      className={`text-xs px-2 py-1 rounded-full ${
                        position.side === 'long' ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400'
                      }`}
                    >
                      {position.side.toUpperCase()}
                    </span>
                  </div>
                  <div className="text-sm text-gray-400 mt-1">
                    Qty: {position.quantity} @ ${position.entry_price.toFixed(2)}
                  </div>
                </div>

                <div className="text-right">
                  <div
                    className={`text-lg font-semibold ${
                      position.pnl >= 0 ? 'text-green-400' : 'text-red-400'
                    }`}
                  >
                    ${position.pnl.toFixed(2)}
                  </div>
                  <div
                    className={`text-sm ${position.pnl_percent >= 0 ? 'text-green-400' : 'text-red-400'}`}
                  >
                    {position.pnl_percent.toFixed(2)}%
                  </div>
                </div>
              </div>

              <div className="mt-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
                <div className="bg-gray-900 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Current Price</div>
                  <div className="text-white text-lg font-semibold">
                    ${position.current_price.toFixed(2)}
                  </div>
                </div>
                <div className="bg-gray-900 rounded-lg p-3">
                  <div className="text-xs text-gray-400">Market Value</div>
                  <div className="text-white text-lg font-semibold">
                    ${(position.current_price * position.quantity).toFixed(2)}
                  </div>
                </div>
                <div className="bg-gray-900 rounded-lg p-3">
                  <div className="text-xs text-gray-400 flex items-center">
                    <Target className="w-3 h-3 mr-1" /> Take Profit
                  </div>
                  {editingPosition === `${position.id}-tp` ? (
                    <div className="flex items-center space-x-2 mt-2">
                      <input
                        type="number"
                        className="bg-gray-800 text-white rounded px-2 py-1 text-sm w-full"
                        value={position.take_profit || ''}
                        onChange={(e) => updateTakeProfit(position.id, Number(e.target.value))}
                      />
                      <button
                        onClick={() => setEditingPosition(null)}
                        className="p-2 bg-gray-700 rounded"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setEditingPosition(`${position.id}-tp`)}
                      className="flex items-center text-sm text-blue-400 hover:text-blue-300 mt-2"
                    >
                      <Edit3 className="w-4 h-4 mr-1" />
                      {position.take_profit ? `$${position.take_profit.toFixed(2)}` : 'Set target'}
                    </button>
                  )}
                </div>
                <div className="bg-gray-900 rounded-lg p-3">
                  <div className="text-xs text-gray-400 flex items-center">
                    <AlertTriangle className="w-3 h-3 mr-1" /> Stop Loss
                  </div>
                  {editingPosition === `${position.id}-sl` ? (
                    <div className="flex items-center space-x-2 mt-2">
                      <input
                        type="number"
                        className="bg-gray-800 text-white rounded px-2 py-1 text-sm w-full"
                        value={position.stop_loss || ''}
                        onChange={(e) => updateStopLoss(position.id, Number(e.target.value))}
                      />
                      <button
                        onClick={() => setEditingPosition(null)}
                        className="p-2 bg-gray-700 rounded"
                      >
                        <Check className="w-4 h-4" />
                      </button>
                    </div>
                  ) : (
                    <button
                      onClick={() => setEditingPosition(`${position.id}-sl`)}
                      className="flex items-center text-sm text-yellow-400 hover:text-yellow-300 mt-2"
                    >
                      <Edit3 className="w-4 h-4 mr-1" />
                      {position.stop_loss ? `$${position.stop_loss.toFixed(2)}` : 'Set stop'}
                    </button>
                  )}
                </div>
              </div>

              <div className="mt-4 flex items-center justify-between">
                <div className="text-sm text-gray-400 flex items-center space-x-3">
                  <div className="flex items-center">
                    <Calculator className="w-4 h-4 mr-1" />
                    <span>
                      Risk per unit: ${
                        position.entry_price > 0
                          ? (position.entry_price - (position.stop_loss || position.entry_price * 0.98)).toFixed(2)
                          : '—'
                      }
                    </span>
                  </div>
                  <div className="flex items-center">
                    <DollarSign className="w-4 h-4 mr-1" />
                    <span>Total risk: ${((position.stop_loss || position.entry_price) * position.quantity).toFixed(2)}</span>
                  </div>
                </div>

                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => closePosition(position.id)}
                    className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm flex items-center"
                  >
                    <X className="w-4 h-4 mr-1" />
                    Close Position
                  </button>
                </div>
              </div>
            </div>
          ))
        )}
      </div>

      {/* Manual Trade Entry */}
      <div className="bg-gray-800 rounded-lg p-5 border border-gray-700">
        <div className="flex items-center justify-between mb-4">
          <div>
            <h4 className="text-lg font-semibold text-white flex items-center">
              <TrendingDown className="w-5 h-5 mr-2 text-blue-400" />
              Manual Trade Entry
            </h4>
            {providerMeta && (
              <p className="text-xs text-gray-400">Orders will be sent via {providerMeta.label}</p>
            )}
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-3">
          <div>
            <label className="text-xs text-gray-400 mb-2 block">Symbol</label>
            <input
              type="text"
              value={manualTrade.symbol}
              onChange={(e) => setManualTrade({ ...manualTrade, symbol: e.target.value })}
              placeholder="BTCUSD or BTCUSDT"
              className="w-full bg-gray-900 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-2 block">Side</label>
            <div className="flex bg-gray-900 rounded-lg overflow-hidden">
              {['buy', 'sell'].map((side) => (
                <button
                  key={side}
                  onClick={() => setManualTrade({ ...manualTrade, side: side as 'buy' | 'sell' })}
                  className={`flex-1 px-3 py-2 text-sm font-medium transition-colors ${
                    manualTrade.side === side
                      ? side === 'buy'
                        ? 'bg-green-600 text-white'
                        : 'bg-red-600 text-white'
                      : 'text-gray-300'
                  }`}
                >
                  {side.toUpperCase()}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-2 block">Quantity</label>
            <input
              type="number"
              min={0}
              value={manualTrade.quantity}
              onChange={(e) => setManualTrade({ ...manualTrade, quantity: Number(e.target.value) })}
              className="w-full bg-gray-900 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
          <div>
            <label className="text-xs text-gray-400 mb-2 block">Order Type</label>
            <select
              value={manualTrade.order_type}
              onChange={(e) => setManualTrade({ ...manualTrade, order_type: e.target.value as 'market' | 'limit' })}
              className="w-full bg-gray-900 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="market">Market</option>
              <option value="limit">Limit</option>
            </select>
          </div>
        </div>

        {manualTrade.order_type === 'limit' && (
          <div className="mt-3">
            <label className="text-xs text-gray-400 mb-2 block">Limit Price</label>
            <input
              type="number"
              min={0}
              value={manualTrade.limit_price || ''}
              onChange={(e) => setManualTrade({ ...manualTrade, limit_price: Number(e.target.value) })}
              className="w-full bg-gray-900 text-white rounded-lg px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>
        )}

        <div className="mt-4 flex flex-col md:flex-row md:items-center md:justify-between space-y-3 md:space-y-0">
          <div className="text-sm text-gray-400 flex items-center space-x-2">
            <span className="flex items-center">
              <Calculator className="w-4 h-4 mr-1" />
              Position size at {riskPercent}% risk: ${
                suggestedPositionSize > 0
                  ? suggestedPositionSize.toFixed(2)
                  : 'Set limit price'
              }
            </span>
            <div className="flex items-center space-x-1">
              <button
                onClick={() => setRiskPercent((prev) => Math.max(0.5, prev - 0.5))}
                className="p-1 bg-gray-900 rounded"
              >
                <Minus className="w-3 h-3" />
              </button>
              <span>{riskPercent}% risk</span>
              <button
                onClick={() => setRiskPercent((prev) => Math.min(10, prev + 0.5))}
                className="p-1 bg-gray-900 rounded"
              >
                <Plus className="w-3 h-3" />
              </button>
            </div>
          </div>
          <button
            onClick={submitManualTrade}
            disabled={isSubmittingTrade}
            className="px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg text-sm font-medium flex items-center justify-center"
          >
            {isSubmittingTrade ? 'Submitting…' : 'Submit Trade'}
          </button>
        </div>
      </div>

      {showCloseAllConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-60 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 w-full max-w-md">
            <h4 className="text-lg font-semibold text-white mb-2">Close All Positions?</h4>
            <p className="text-gray-300 text-sm">
              This will attempt to close every open position through {providerMeta?.label || 'your broker'}.
            </p>
            <div className="mt-4 flex justify-end space-x-3">
              <button
                onClick={() => setShowCloseAllConfirm(false)}
                className="px-3 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm"
              >
                Cancel
              </button>
              <button
                onClick={closeAllPositions}
                className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm"
              >
                Confirm Close All
              </button>
            </div>
          </div>
        </div>
      )}

      {notifications.length > 0 && (
        <div className="fixed bottom-6 right-6 space-y-2 z-40">
          {notifications.map((note, index) => (
            <div key={`${note}-${index}`} className="bg-gray-900 border border-gray-700 text-white px-4 py-3 rounded-lg shadow-lg">
              {note}
            </div>
          ))}
        </div>
      )}

      <style jsx>{`
        /* Custom scrollbar styling */
        .overflow-y-auto::-webkit-scrollbar {
          width: 6px;
        }

        .overflow-y-auto::-webkit-scrollbar-track {
          background: #1F2937;
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb {
          background: #4B5563;
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb:hover {
          background: #6B7280;
        }
      `}</style>
    </div>
  );
};


