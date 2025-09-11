import React, { useState } from 'react';
import { Play, Square } from 'lucide-react';
import { tradingAgent } from '../../services/tradingAgent';
import { alpacaService } from '../../services/alpacaService';

interface TradingControlsProps {
  onOrderPlaced?: () => void;
}

export const TradingControls: React.FC<TradingControlsProps> = ({ onOrderPlaced }) => {
  const [agentStatus, setAgentStatus] = useState(tradingAgent.getStatus());
  const [orderForm, setOrderForm] = useState({
    symbol: '',
    qty: '',
    side: 'buy' as 'buy' | 'sell',
    order_type: 'market' as 'market' | 'limit',
    limit_price: '',
  });

  const handleStartAgent = () => {
    tradingAgent.start();
    setAgentStatus(tradingAgent.getStatus());
  };

  const handleStopAgent = () => {
    tradingAgent.stop();
    setAgentStatus(tradingAgent.getStatus());
  };

  const normalizeSymbol = (raw: string): string => {
    const cleaned = raw.toUpperCase().replace(/[^A-Z]/g, '');
    if (!cleaned) return '';
    if (cleaned.endsWith('USD')) return cleaned;
    return `${cleaned}USD`;
  };

  const quickBuy = async (symbolBase: string, qty: string) => {
    try {
      const symbol = normalizeSymbol(symbolBase);
      await alpacaService.placeOrder({
        symbol,
        qty,
        side: 'buy',
        order_type: 'market',
      });
      if (onOrderPlaced) onOrderPlaced();
      alert(`Market BUY placed: ${qty} ${symbol}`);
    } catch (err) {
      console.error('Quick buy failed', err);
      alert('Failed to place quick buy order.');
    }
  };

  const sellAllPositions = async () => {
    try {
      const positions = await alpacaService.getPositions();
      if (!positions || positions.length === 0) {
        alert('No open positions to sell.');
        return;
      }
      for (const p of positions) {
        const symbol = normalizeSymbol((p as any).symbol || '');
        const qty = (p as any).qty || '0';
        if (!symbol || qty === '0') continue;
        await alpacaService.placeOrder({
          symbol,
          qty,
          side: 'sell',
          order_type: 'market',
        });
      }
      if (onOrderPlaced) onOrderPlaced();
      alert('Submitted market sell orders for all positions.');
    } catch (err) {
      console.error('Sell all failed', err);
      alert('Failed to sell all positions.');
    }
  };

  // Favorites (local)
  type Favorite = { id: string; label: string; side: 'buy' | 'sell'; symbolBase: string; qty: string };
  const [favorites, setFavorites] = useState<Favorite[]>(() => {
    try {
      const raw = localStorage.getItem('trade-favorites');
      return raw ? (JSON.parse(raw) as Favorite[]) : [];
    } catch {
      return [];
    }
  });
  const persistFavorites = (next: Favorite[]) => {
    setFavorites(next);
    localStorage.setItem('trade-favorites', JSON.stringify(next));
  };
  const runFavorite = async (fav: Favorite) => {
    try {
      const symbol = normalizeSymbol(fav.symbolBase);
      await alpacaService.placeOrder({ symbol, qty: fav.qty, side: fav.side, order_type: 'market' });
      if (onOrderPlaced) onOrderPlaced();
      alert(`${fav.label}: ${fav.side.toUpperCase()} ${fav.qty} ${symbol}`);
    } catch (err) {
      console.error('Favorite trade failed', err);
      alert('Failed to execute favorite trade.');
    }
  };
  const removeFavorite = (id: string) => persistFavorites(favorites.filter(f => f.id !== id));
  const [newFav, setNewFav] = useState<Favorite>({ id: '', label: '', side: 'buy', symbolBase: 'ETH', qty: '1' });
  const addFavorite = () => {
    if (!newFav.label.trim()) return alert('Add a label');
    const id = `${Date.now()}`;
    const next = [...favorites, { ...newFav, id }];
    persistFavorites(next);
    setNewFav({ id: '', label: '', side: 'buy', symbolBase: 'ETH', qty: '1' });
  };

  const handlePlaceOrder = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      const symbol = normalizeSymbol(orderForm.symbol);
      if (!symbol) throw new Error('Invalid symbol');
      if (orderForm.order_type === 'limit' && !orderForm.limit_price) {
        throw new Error('Limit price is required for limit orders');
      }

      await alpacaService.placeOrder({
        symbol,
        qty: orderForm.qty,
        side: orderForm.side,
        order_type: orderForm.order_type,
        limit_price: orderForm.limit_price ? parseFloat(orderForm.limit_price) : null,
      });
      if (onOrderPlaced) onOrderPlaced();
      
      // Reset form
      setOrderForm({
        symbol: '',
        qty: '',
        side: 'buy',
        order_type: 'market',
        limit_price: '',
      });
      
      // Show success message
      alert('Order placed successfully!');
    } catch (error) {
      console.error('Error placing order:', error);
      alert('Failed to place order. Please try again.');
    }
  };

  return (
    <div className="h-full overflow-y-auto space-y-6 pr-2">
      {/* Trading Agent Control */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-lg sm:text-xl font-bold text-white mb-4">Crypto Trading Agent</h2>
        <div className="mb-6">
          <div className="flex items-center justify-between mb-4">
            <div className={`w-3 h-3 rounded-full mr-3 ${agentStatus.active ? 'bg-green-400' : 'bg-red-400'}`}></div>
            <span className="text-white font-medium text-lg">
              Status: {agentStatus.active ? 'Active' : 'Inactive'}
            </span>
            <div className="flex space-x-3">
              <button
                onClick={handleStartAgent}
                disabled={agentStatus.active}
                className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 text-white rounded-lg transition-colors font-medium"
              >
                <Play className="h-4 w-4 mr-2" />
                Start
              </button>
              <button
                onClick={handleStopAgent}
                disabled={!agentStatus.active}
                className="flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white rounded-lg transition-colors font-medium"
              >
                <Square className="h-4 w-4 mr-2" />
                Stop
              </button>
            </div>
          </div>
        </div>
        <div className="grid grid-cols-2 gap-4">
          <div className="bg-gray-700 p-4 rounded-lg">
            <p className="text-gray-400 text-sm mb-1">Signals Generated</p>
            <p className="text-white font-bold text-2xl">{agentStatus.signalsCount}</p>
          </div>
          <div className="bg-gray-700 p-4 rounded-lg">
            <p className="text-gray-400 text-sm mb-1">Watchlist Size</p>
            <p className="text-white font-bold text-2xl">{agentStatus.watchlistSize}</p>
          </div>
        </div>
      </div>

      {/* Manual Trading */}
      <div className="bg-gray-800 rounded-lg p-6">
        <h2 className="text-lg sm:text-xl font-bold text-white mb-4">Manual Crypto Trading</h2>
        {/* Quick Actions */}
        <div className="mb-4 grid grid-cols-1 sm:grid-cols-3 gap-2">
          <button
            type="button"
            onClick={() => quickBuy('ETH', '1')}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"
          >
            Buy 1 ETH (Market)
          </button>
          <button
            type="button"
            onClick={() => quickBuy('BTC', '0.1')}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"
          >
            Buy 0.1 BTC (Market)
          </button>
          <button
            type="button"
            onClick={sellAllPositions}
            className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm"
          >
            Sell All Positions (Market)
          </button>
          <button
            type="button"
            onClick={() => alpacaService.placeOrder({ symbol: normalizeSymbol('ETH'), qty: '1', side: 'sell', order_type: 'market' }).then(() => { if (onOrderPlaced) onOrderPlaced(); alert('Market SELL placed: 1 ETH'); }).catch(err => { console.error(err); alert('Failed to sell 1 ETH.'); })}
            className="px-3 py-2 bg-red-600 hover:bg-red-700 text-white rounded-lg text-sm"
          >
            Sell 1 ETH (Market)
          </button>
          <button
            type="button"
            onClick={() => quickBuy('SOL', '10')}
            className="px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg text-sm"
          >
            Buy 10 SOL (Market)
          </button>
        </div>

        {/* Favorite Trades */}
        <div className="mb-4">
          <h3 className="text-white font-semibold mb-2">Favorite Trades</h3>
          {favorites.length > 0 ? (
            <div className="flex flex-wrap gap-2">
              {favorites.map(f => (
                <div key={f.id} className={`flex items-center rounded-lg overflow-hidden border ${f.side === 'buy' ? 'border-blue-600' : 'border-red-600'}`}>
                  <button onClick={() => runFavorite(f)} className={`${f.side === 'buy' ? 'bg-blue-600 hover:bg-blue-700' : 'bg-red-600 hover:bg-red-700'} text-white px-3 py-2 text-xs`}>{f.label}</button>
                  <button onClick={() => removeFavorite(f.id)} className="px-2 text-gray-300 hover:text-white text-xs bg-gray-700">✕</button>
                </div>
              ))}
            </div>
          ) : (
            <div className="text-gray-400 text-xs">No favorites yet.</div>
          )}
          <div className="mt-2 grid grid-cols-1 sm:grid-cols-5 gap-2 text-xs">
            <input value={newFav.label} onChange={e => setNewFav({ ...newFav, label: e.target.value })} placeholder="Label (e.g., Buy 5 ADA)" className="px-2 py-2 bg-gray-700 text-white rounded" />
            <select value={newFav.side} onChange={e => setNewFav({ ...newFav, side: e.target.value as 'buy' | 'sell' })} className="px-2 py-2 bg-gray-700 text-white rounded">
              <option value="buy">Buy</option>
              <option value="sell">Sell</option>
            </select>
            <input value={newFav.symbolBase} onChange={e => setNewFav({ ...newFav, symbolBase: e.target.value })} placeholder="Symbol (BTC/ETH)" className="px-2 py-2 bg-gray-700 text-white rounded" />
            <input value={newFav.qty} onChange={e => setNewFav({ ...newFav, qty: e.target.value })} placeholder="Qty" className="px-2 py-2 bg-gray-700 text-white rounded" />
            <button type="button" onClick={addFavorite} className="px-3 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded">Add Favorite</button>
          </div>
        </div>
        <form onSubmit={handlePlaceOrder} className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 text-xs sm:text-sm mb-2">Symbol</label>
              <input
                type="text"
                value={orderForm.symbol}
                onChange={(e) => setOrderForm({ ...orderForm, symbol: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                placeholder="BTC or BTCUSD"
                required
              />
              <p className="mt-1 text-xs text-gray-400">Tip: You can enter base symbol (e.g., BTC, ETH) — it will be sent as USD pair (BTCUSD, ETHUSD).</p>
            </div>
            <div>
              <label className="block text-gray-400 text-xs sm:text-sm mb-2">Quantity</label>
              <input
                type="number"
                value={orderForm.qty}
                onChange={(e) => setOrderForm({ ...orderForm, qty: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                placeholder="0.001"
                required
              />
              <p className="mt-1 text-xs text-gray-400">Small fractional sizes are supported for crypto on Alpaca paper trading.</p>
            </div>
          </div>
          
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
            <div>
              <label className="block text-gray-400 text-xs sm:text-sm mb-2">Side</label>
              <select
                value={orderForm.side}
                onChange={(e) => setOrderForm({ ...orderForm, side: e.target.value as 'buy' | 'sell' })}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value="buy">Buy</option>
                <option value="sell">Sell</option>
              </select>
            </div>
            <div>
              <label className="block text-gray-400 text-xs sm:text-sm mb-2">Order Type</label>
              <select
                value={orderForm.order_type}
                onChange={(e) => setOrderForm({ ...orderForm, order_type: e.target.value as 'market' | 'limit' })}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
              >
                <option value="market">Market</option>
                <option value="limit">Limit</option>
              </select>
            </div>
          </div>

          {orderForm.order_type === 'limit' && (
            <div>
              <label className="block text-gray-400 text-xs sm:text-sm mb-2">Limit Price</label>
              <input
                type="number"
                step="0.01"
                value={orderForm.limit_price}
                onChange={(e) => setOrderForm({ ...orderForm, limit_price: e.target.value })}
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                placeholder="175.50"
                required
              />
            </div>
          )}

          <button
            type="submit"
            className={`w-full py-3 rounded-lg font-semibold transition-colors text-sm sm:text-base ${
              orderForm.side === 'buy' 
                ? 'bg-green-600 hover:bg-green-700 text-white' 
                : 'bg-red-600 hover:bg-red-700 text-white'
            }`}
          >
            {orderForm.side === 'buy' ? 'Place Buy Order' : 'Place Sell Order'}
          </button>
        </form>
      </div>

    </div>
  );
};