import React, { useState } from 'react';
import { Wifi, WifiOff, Activity, TrendingUp, AlertCircle } from 'lucide-react';
import {
  useSocket,
  useRealtimePositions,
  useRealtimeOrders,
  useRealtimeMarketData,
  useRealtimeOrderBook,
  useRealtimeAlerts
} from '../hooks/useSocket';

export const SocketIOTest: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const symbols = ['BTC', 'ETH', 'BNB', 'SOL', 'ADA'];

  // Use all Socket.IO hooks
  const baseSocket = useSocket({
    onConnectionStatus: (status) => console.log('Connection status:', status),
  });

  const { positions, isConnected: positionsConnected } = useRealtimePositions();
  const { orders, isConnected: ordersConnected } = useRealtimeOrders();
  const { marketData, lastUpdate: marketLastUpdate, isConnected: marketConnected } = useRealtimeMarketData(symbols);
  const { orderBook, lastUpdate: orderBookLastUpdate, isConnected: orderBookConnected } = useRealtimeOrderBook(selectedSymbol);
  const { alerts, isConnected: alertsConnected } = useRealtimeAlerts();

  const handleStartTrading = async () => {
    try {
      const response = await baseSocket.startTrading({
        maxPositions: 5,
        riskPerTrade: 0.02
      });
      console.log('Start trading response:', response);
    } catch (error) {
      console.error('Failed to start trading:', error);
    }
  };

  const handleStopTrading = async () => {
    try {
      const response = await baseSocket.stopTrading();
      console.log('Stop trading response:', response);
    } catch (error) {
      console.error('Failed to stop trading:', error);
    }
  };

  const handleEmergencyStop = async () => {
    try {
      const response = await baseSocket.emergencyStop();
      console.log('Emergency stop response:', response);
    } catch (error) {
      console.error('Failed to emergency stop:', error);
    }
  };

  const ConnectionStatus = ({ label, connected }: { label: string; connected: boolean }) => (
    <div className="flex items-center space-x-2">
      {connected ? (
        <Wifi className="h-4 w-4 text-green-400" />
      ) : (
        <WifiOff className="h-4 w-4 text-red-400" />
      )}
      <span className={connected ? 'text-green-400' : 'text-red-400'}>
        {label}: {connected ? 'Connected' : 'Disconnected'}
      </span>
    </div>
  );

  return (
    <div className="min-h-screen bg-gray-900 text-white p-8">
      <div className="max-w-7xl mx-auto">
        <h1 className="text-3xl font-bold mb-8">Socket.IO Integration Test</h1>

        {/* Connection Status */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Connection Status</h2>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
            <ConnectionStatus label="Base Socket" connected={baseSocket.isConnected} />
            <ConnectionStatus label="Positions" connected={positionsConnected} />
            <ConnectionStatus label="Orders" connected={ordersConnected} />
            <ConnectionStatus label="Market Data" connected={marketConnected} />
            <ConnectionStatus label="Order Book" connected={orderBookConnected} />
            <ConnectionStatus label="Alerts" connected={alertsConnected} />
          </div>
          <div className="mt-4 text-sm text-gray-400">
            Connection Status: {baseSocket.connectionStatus}
          </div>
        </div>

        {/* Trading Controls */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <h2 className="text-xl font-semibold mb-4">Trading Controls</h2>
          <div className="flex space-x-4">
            <button
              onClick={handleStartTrading}
              className="px-4 py-2 bg-green-600 hover:bg-green-700 rounded-lg transition-colors"
              disabled={!baseSocket.isConnected}
            >
              Start Trading
            </button>
            <button
              onClick={handleStopTrading}
              className="px-4 py-2 bg-yellow-600 hover:bg-yellow-700 rounded-lg transition-colors"
              disabled={!baseSocket.isConnected}
            >
              Stop Trading
            </button>
            <button
              onClick={handleEmergencyStop}
              className="px-4 py-2 bg-red-600 hover:bg-red-700 rounded-lg transition-colors"
              disabled={!baseSocket.isConnected}
            >
              Emergency Stop
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Positions */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <TrendingUp className="h-5 w-5 mr-2" />
              Positions ({positions.length})
            </h2>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {positions.length === 0 ? (
                <p className="text-gray-400">No positions</p>
              ) : (
                positions.map((position: any) => (
                  <div key={position.id} className="bg-gray-700 rounded p-3">
                    <div className="flex justify-between">
                      <span className="font-medium">{position.symbol}</span>
                      <span className={position.pnl >= 0 ? 'text-green-400' : 'text-red-400'}>
                        ${position.pnl?.toFixed(2) || '0.00'}
                      </span>
                    </div>
                    <div className="text-sm text-gray-400">
                      Qty: {position.quantity} | Entry: ${position.entryPrice}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Orders */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <Activity className="h-5 w-5 mr-2" />
              Recent Orders ({orders.length})
            </h2>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {orders.length === 0 ? (
                <p className="text-gray-400">No orders</p>
              ) : (
                orders.slice(0, 5).map((order: any) => (
                  <div key={order.id} className="bg-gray-700 rounded p-3">
                    <div className="flex justify-between">
                      <span className="font-medium">{order.symbol}</span>
                      <span className={`text-sm px-2 py-1 rounded ${
                        order.status === 'filled' ? 'bg-green-600' :
                        order.status === 'failed' ? 'bg-red-600' :
                        'bg-yellow-600'
                      }`}>
                        {order.status}
                      </span>
                    </div>
                    <div className="text-sm text-gray-400">
                      {order.side} {order.quantity} @ ${order.price}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>

          {/* Market Data */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4">
              Market Data
              {marketLastUpdate && (
                <span className="text-sm text-gray-400 ml-2">
                  Last: {new Date(marketLastUpdate).toLocaleTimeString()}
                </span>
              )}
            </h2>
            <div className="space-y-2">
              {symbols.map(symbol => {
                const data = marketData.get(symbol);
                return (
                  <div key={symbol} className="flex justify-between items-center bg-gray-700 rounded p-2">
                    <span className="font-medium">{symbol}</span>
                    {data ? (
                      <div className="text-right">
                        <div>${data.price?.toFixed(2) || 'N/A'}</div>
                        <div className={`text-sm ${data.changePercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                          {data.changePercent >= 0 ? '+' : ''}{data.changePercent?.toFixed(2) || '0.00'}%
                        </div>
                      </div>
                    ) : (
                      <span className="text-gray-400">Loading...</span>
                    )}
                  </div>
                );
              })}
            </div>
          </div>

          {/* Alerts */}
          <div className="bg-gray-800 rounded-lg p-6">
            <h2 className="text-xl font-semibold mb-4 flex items-center">
              <AlertCircle className="h-5 w-5 mr-2" />
              Alerts ({alerts.length})
            </h2>
            <div className="space-y-2 max-h-64 overflow-y-auto">
              {alerts.length === 0 ? (
                <p className="text-gray-400">No alerts</p>
              ) : (
                alerts.slice(0, 5).map((alert: any) => (
                  <div key={alert.id} className={`rounded p-3 ${
                    alert.severity === 'critical' ? 'bg-red-900' :
                    alert.severity === 'warning' ? 'bg-yellow-900' :
                    'bg-gray-700'
                  }`}>
                    <div className="font-medium">{alert.title}</div>
                    <div className="text-sm text-gray-300">{alert.message}</div>
                    <div className="text-xs text-gray-400 mt-1">
                      {new Date(alert.timestamp).toLocaleTimeString()}
                    </div>
                  </div>
                ))
              )}
            </div>
          </div>
        </div>

        {/* Order Book */}
        <div className="bg-gray-800 rounded-lg p-6 mt-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-xl font-semibold">Order Book</h2>
            <select
              value={selectedSymbol}
              onChange={(e) => setSelectedSymbol(e.target.value)}
              className="bg-gray-700 text-white px-3 py-1 rounded"
            >
              {symbols.map(symbol => (
                <option key={symbol} value={symbol}>{symbol}</option>
              ))}
            </select>
          </div>
          {orderBook ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <h3 className="text-green-400 font-medium mb-2">Bids</h3>
                <div className="space-y-1">
                  {orderBook.bids?.slice(0, 5).map((bid: any, i: number) => (
                    <div key={i} className="flex justify-between text-sm bg-gray-700 rounded px-2 py-1">
                      <span>${bid[0]}</span>
                      <span className="text-gray-400">{bid[1]}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div>
                <h3 className="text-red-400 font-medium mb-2">Asks</h3>
                <div className="space-y-1">
                  {orderBook.asks?.slice(0, 5).map((ask: any, i: number) => (
                    <div key={i} className="flex justify-between text-sm bg-gray-700 rounded px-2 py-1">
                      <span>${ask[0]}</span>
                      <span className="text-gray-400">{ask[1]}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <p className="text-gray-400">No order book data</p>
          )}
          {orderBookLastUpdate && (
            <div className="text-xs text-gray-400 mt-2">
              Last update: {new Date(orderBookLastUpdate).toLocaleTimeString()}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};