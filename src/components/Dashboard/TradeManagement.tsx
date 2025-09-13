import React, { useState, useEffect } from 'react';
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
  Minus
} from 'lucide-react';
import { alpacaService } from '../../services/alpacaService';
import { tradeHistoryService } from '../../services/persistence/tradeHistoryService';

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

const TradeManagement: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [editingPosition, setEditingPosition] = useState<string | null>(null);
  const [manualTrade, setManualTrade] = useState<ManualTradeForm>({
    symbol: '',
    side: 'buy',
    quantity: 0,
    order_type: 'market'
  });
  const [riskPercent, setRiskPercent] = useState(2);
  const [accountValue, setAccountValue] = useState(50000);
  const [showCloseAllConfirm, setShowCloseAllConfirm] = useState(false);
  const [isSubmittingTrade, setIsSubmittingTrade] = useState(false);
  const [notifications, setNotifications] = useState<string[]>([]);

  useEffect(() => {
    loadPositions();
    loadAccountInfo();
    const interval = setInterval(loadPositions, 10000); // Update every 10 seconds
    return () => clearInterval(interval);
  }, []);

  const loadPositions = async () => {
    try {
      const alpacaPositions = await alpacaService.getPositions();
      const trades = await tradeHistoryService.getRecentTrades(100);
      
      // Convert Alpaca positions to our format
      const formattedPositions: Position[] = alpacaPositions.map(pos => ({
        id: pos.asset_id || pos.symbol,
        symbol: pos.symbol,
        side: parseFloat(pos.qty) > 0 ? 'long' : 'short',
        quantity: Math.abs(parseFloat(pos.qty)),
        entry_price: parseFloat(pos.avg_entry_price || '0'),
        current_price: parseFloat(pos.market_value || '0') / Math.abs(parseFloat(pos.qty)),
        pnl: parseFloat(pos.unrealized_pl || '0'),
        pnl_percent: parseFloat(pos.unrealized_plpc || '0') * 100,
        status: 'open'
      }));
      
      setPositions(formattedPositions);
      setIsLoading(false);
    } catch (error) {
      console.error('Error loading positions:', error);
      setIsLoading(false);
    }
  };

  const loadAccountInfo = async () => {
    try {
      const account = await alpacaService.getAccount();
      setAccountValue(account.portfolio_value);
    } catch (error) {
      console.error('Error loading account info:', error);
    }
  };

  const calculatePositionSize = (price: number): number => {
    const riskAmount = accountValue * (riskPercent / 100);
    return Math.floor((riskAmount / price) * 100) / 100;
  };

  const closePosition = async (positionId: string) => {
    const position = positions.find(p => p.id === positionId);
    if (!position) return;

    try {
      setPositions(prev => 
        prev.map(p => p.id === positionId ? { ...p, status: 'closing' } : p)
      );

      await alpacaService.placeOrder({
        symbol: position.symbol,
        qty: position.quantity,
        side: position.side === 'long' ? 'sell' : 'buy',
        type: 'market',
        time_in_force: 'day'
      });

      addNotification(`Position ${position.symbol} closing order submitted`);
      setTimeout(loadPositions, 2000); // Refresh after 2 seconds
    } catch (error) {
      console.error('Error closing position:', error);
      addNotification(`Error closing ${position.symbol}: ${error.message}`);
      // Revert status
      setPositions(prev => 
        prev.map(p => p.id === positionId ? { ...p, status: 'open' } : p)
      );
    }
  };

  const updateStopLoss = async (positionId: string, stopLoss: number) => {
    const position = positions.find(p => p.id === positionId);
    if (!position) return;

    try {
      // For simplicity, we'll store this locally and in future iterations
      // this would integrate with Alpaca's bracket orders or alerts
      setPositions(prev =>
        prev.map(p => p.id === positionId ? { ...p, stop_loss: stopLoss } : p)
      );
      
      addNotification(`Stop loss updated for ${position.symbol}: $${stopLoss.toFixed(2)}`);
    } catch (error) {
      console.error('Error updating stop loss:', error);
      addNotification(`Error updating stop loss: ${error.message}`);
    }
  };

  const updateTakeProfit = async (positionId: string, takeProfit: number) => {
    const position = positions.find(p => p.id === positionId);
    if (!position) return;

    try {
      setPositions(prev =>
        prev.map(p => p.id === positionId ? { ...p, take_profit: takeProfit } : p)
      );
      
      addNotification(`Take profit updated for ${position.symbol}: $${takeProfit.toFixed(2)}`);
    } catch (error) {
      console.error('Error updating take profit:', error);
      addNotification(`Error updating take profit: ${error.message}`);
    }
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
      addNotification(`Error closing all positions: ${error.message}`);
    }
  };

  const submitManualTrade = async () => {
    if (!manualTrade.symbol || manualTrade.quantity <= 0) {
      addNotification('Please fill in all required fields');
      return;
    }

    setIsSubmittingTrade(true);
    try {
      const order = await alpacaService.placeOrder({
        symbol: manualTrade.symbol.toUpperCase(),
        qty: manualTrade.quantity,
        side: manualTrade.side,
        type: manualTrade.order_type,
        time_in_force: 'day',
        limit_price: manualTrade.order_type === 'limit' ? manualTrade.limit_price : undefined
      });

      // Record in trade history
      await tradeHistoryService.recordTrade({
        symbol: manualTrade.symbol.toUpperCase(),
        side: manualTrade.side,
        quantity: manualTrade.quantity,
        entry_price: manualTrade.limit_price || 0,
        execution_status: 'pending',
        confidence_score: 100, // Manual trade
        risk_reward_ratio: 1.0,
        position_size_percent: (manualTrade.quantity * (manualTrade.limit_price || 0) / accountValue) * 100,
        risk_amount: manualTrade.quantity * (manualTrade.limit_price || 0)
      });

      addNotification(`Manual ${manualTrade.side} order submitted for ${manualTrade.symbol}`);
      
      // Reset form
      setManualTrade({
        symbol: '',
        side: 'buy',
        quantity: 0,
        order_type: 'market'
      });

      setTimeout(loadPositions, 2000);
    } catch (error) {
      console.error('Error submitting manual trade:', error);
      addNotification(`Error submitting trade: ${error.message}`);
    }
    setIsSubmittingTrade(false);
  };

  const addNotification = (message: string) => {
    setNotifications(prev => [...prev, message]);
    setTimeout(() => {
      setNotifications(prev => prev.slice(1));
    }, 5000);
  };

  if (isLoading) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="animate-pulse space-y-4">
          <div className="h-6 bg-gray-700 rounded w-1/4"></div>
          <div className="space-y-3">
            {[1, 2, 3].map(i => (
              <div key={i} className="h-16 bg-gray-800 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg p-6 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold text-white flex items-center">
          <TrendingUp className="w-6 h-6 mr-2 text-green-400" />
          Trade Management
        </h3>
        {positions.length > 0 && (
          <button
            onClick={() => setShowCloseAllConfirm(true)}
            className="bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-white text-sm font-medium transition-colors flex items-center"
          >
            <X className="w-4 h-4 mr-1" />
            Close All Positions
          </button>
        )}
      </div>

      {/* Notifications */}
      {notifications.length > 0 && (
        <div className="space-y-2">
          {notifications.map((notification, index) => (
            <div key={index} className="bg-blue-900 border border-blue-700 text-blue-200 px-4 py-2 rounded-lg text-sm">
              {notification}
            </div>
          ))}
        </div>
      )}

      {/* Position Size Calculator */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center mb-3">
          <Calculator className="w-5 h-5 mr-2 text-blue-400" />
          <h4 className="text-lg font-medium text-white">Position Size Calculator</h4>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
          <div>
            <label className="block text-gray-400 mb-1">Risk Percentage</label>
            <div className="flex items-center space-x-2">
              <input
                type="range"
                min="0.5"
                max="10"
                step="0.5"
                value={riskPercent}
                onChange={(e) => setRiskPercent(parseFloat(e.target.value))}
                className="flex-1"
              />
              <span className="text-white w-12">{riskPercent}%</span>
            </div>
          </div>
          <div>
            <label className="block text-gray-400 mb-1">Account Value</label>
            <div className="text-white font-medium">${accountValue.toLocaleString()}</div>
          </div>
          <div>
            <label className="block text-gray-400 mb-1">Risk Amount</label>
            <div className="text-white font-medium">${(accountValue * riskPercent / 100).toLocaleString()}</div>
          </div>
        </div>
      </div>

      {/* Open Positions */}
      <div className="space-y-4">
        <h4 className="text-lg font-medium text-white">
          Open Positions ({positions.length})
        </h4>
        
        {positions.length === 0 ? (
          <div className="bg-gray-800 rounded-lg p-8 text-center">
            <TrendingUp className="w-12 h-12 mx-auto text-gray-600 mb-4" />
            <p className="text-gray-400">No open positions</p>
          </div>
        ) : (
          <div className="space-y-3">
            {positions.map((position) => (
              <div key={position.id} className="bg-gray-800 rounded-lg p-4">
                <div className="flex items-center justify-between mb-3">
                  <div className="flex items-center space-x-3">
                    <div className={`p-2 rounded-full ${
                      position.side === 'long' ? 'bg-green-900 text-green-300' : 'bg-red-900 text-red-300'
                    }`}>
                      {position.side === 'long' ? <TrendingUp className="w-4 h-4" /> : <TrendingDown className="w-4 h-4" />}
                    </div>
                    <div>
                      <h5 className="font-medium text-white">{position.symbol}</h5>
                      <p className="text-sm text-gray-400">
                        {position.side.toUpperCase()} • {position.quantity} shares
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`font-medium ${position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                      ${position.pnl.toFixed(2)} ({position.pnl_percent.toFixed(2)}%)
                    </div>
                    <div className="text-sm text-gray-400">
                      ${position.current_price.toFixed(2)}
                    </div>
                  </div>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-3">
                  <div className="space-y-2">
                    <label className="text-xs text-gray-400">Entry Price</label>
                    <div className="text-white">${position.entry_price.toFixed(2)}</div>
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs text-gray-400">Stop Loss</label>
                    {editingPosition === position.id ? (
                      <input
                        type="number"
                        step="0.01"
                        defaultValue={position.stop_loss || ''}
                        className="w-full bg-gray-700 text-white px-2 py-1 rounded text-sm"
                        onBlur={(e) => {
                          if (e.target.value) {
                            updateStopLoss(position.id, parseFloat(e.target.value));
                          }
                          setEditingPosition(null);
                        }}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter') {
                            e.currentTarget.blur();
                          }
                        }}
                        autoFocus
                      />
                    ) : (
                      <div 
                        className="text-white cursor-pointer hover:text-blue-400 flex items-center"
                        onClick={() => setEditingPosition(position.id)}
                      >
                        {position.stop_loss ? `$${position.stop_loss.toFixed(2)}` : 'Not set'}
                        <Edit3 className="w-3 h-3 ml-1" />
                      </div>
                    )}
                  </div>
                  <div className="space-y-2">
                    <label className="text-xs text-gray-400">Take Profit</label>
                    {editingPosition === `${position.id}_tp` ? (
                      <input
                        type="number"
                        step="0.01"
                        defaultValue={position.take_profit || ''}
                        className="w-full bg-gray-700 text-white px-2 py-1 rounded text-sm"
                        onBlur={(e) => {
                          if (e.target.value) {
                            updateTakeProfit(position.id, parseFloat(e.target.value));
                          }
                          setEditingPosition(null);
                        }}
                        onKeyPress={(e) => {
                          if (e.key === 'Enter') {
                            e.currentTarget.blur();
                          }
                        }}
                        autoFocus
                      />
                    ) : (
                      <div 
                        className="text-white cursor-pointer hover:text-blue-400 flex items-center"
                        onClick={() => setEditingPosition(`${position.id}_tp`)}
                      >
                        {position.take_profit ? `$${position.take_profit.toFixed(2)}` : 'Not set'}
                        <Edit3 className="w-3 h-3 ml-1" />
                      </div>
                    )}
                  </div>
                </div>

                <div className="flex justify-end">
                  <button
                    onClick={() => closePosition(position.id)}
                    disabled={position.status === 'closing'}
                    className="bg-red-600 hover:bg-red-700 disabled:bg-red-800 px-3 py-1 rounded text-white text-sm font-medium transition-colors flex items-center"
                  >
                    {position.status === 'closing' ? (
                      <>
                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-white mr-1"></div>
                        Closing...
                      </>
                    ) : (
                      <>
                        <X className="w-3 h-3 mr-1" />
                        Close Position
                      </>
                    )}
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Manual Trade Entry */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center mb-4">
          <Plus className="w-5 h-5 mr-2 text-blue-400" />
          <h4 className="text-lg font-medium text-white">Manual Trade Entry</h4>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4 mb-4">
          <div>
            <label className="block text-sm text-gray-400 mb-1">Symbol</label>
            <input
              type="text"
              placeholder="e.g., BTCUSD"
              value={manualTrade.symbol}
              onChange={(e) => setManualTrade(prev => ({ ...prev, symbol: e.target.value.toUpperCase() }))}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-400 mb-1">Side</label>
            <select
              value={manualTrade.side}
              onChange={(e) => setManualTrade(prev => ({ ...prev, side: e.target.value as 'buy' | 'sell' }))}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded"
            >
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
          </div>
          
          <div>
            <label className="block text-sm text-gray-400 mb-1">Quantity</label>
            <input
              type="number"
              step="0.01"
              placeholder="0.00"
              value={manualTrade.quantity || ''}
              onChange={(e) => setManualTrade(prev => ({ ...prev, quantity: parseFloat(e.target.value) || 0 }))}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded"
            />
          </div>
          
          <div>
            <label className="block text-sm text-gray-400 mb-1">Order Type</label>
            <select
              value={manualTrade.order_type}
              onChange={(e) => setManualTrade(prev => ({ ...prev, order_type: e.target.value as 'market' | 'limit' }))}
              className="w-full bg-gray-700 text-white px-3 py-2 rounded"
            >
              <option value="market">Market</option>
              <option value="limit">Limit</option>
            </select>
          </div>
        </div>

        {manualTrade.order_type === 'limit' && (
          <div className="mb-4">
            <label className="block text-sm text-gray-400 mb-1">Limit Price</label>
            <input
              type="number"
              step="0.01"
              placeholder="0.00"
              value={manualTrade.limit_price || ''}
              onChange={(e) => setManualTrade(prev => ({ ...prev, limit_price: parseFloat(e.target.value) || undefined }))}
              className="w-full md:w-1/4 bg-gray-700 text-white px-3 py-2 rounded"
            />
          </div>
        )}

        <button
          onClick={submitManualTrade}
          disabled={isSubmittingTrade || !manualTrade.symbol || manualTrade.quantity <= 0}
          className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 px-6 py-2 rounded-lg text-white font-medium transition-colors flex items-center"
        >
          {isSubmittingTrade ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
              Submitting...
            </>
          ) : (
            <>
              <Plus className="w-4 h-4 mr-2" />
              Submit Trade
            </>
          )}
        </button>
      </div>

      {/* Close All Confirmation Modal */}
      {showCloseAllConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md mx-4">
            <div className="flex items-center mb-4">
              <AlertTriangle className="w-6 h-6 text-red-400 mr-3" />
              <h3 className="text-lg font-semibold text-white">Confirm Close All Positions</h3>
            </div>
            <p className="text-gray-300 mb-6">
              Are you sure you want to close all {positions.length} open positions? This action cannot be undone.
            </p>
            <div className="flex space-x-3">
              <button
                onClick={() => setShowCloseAllConfirm(false)}
                className="flex-1 bg-gray-600 hover:bg-gray-700 px-4 py-2 rounded-lg text-white transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={closeAllPositions}
                className="flex-1 bg-red-600 hover:bg-red-700 px-4 py-2 rounded-lg text-white transition-colors"
              >
                Close All
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default TradeManagement;