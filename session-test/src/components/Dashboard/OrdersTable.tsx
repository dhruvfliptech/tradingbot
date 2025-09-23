import React from 'react';
import { Clock, CheckCircle, XCircle, AlertCircle } from 'lucide-react';
import { Order } from '../../types/trading';
import { format } from 'date-fns';

interface OrdersTableProps {
  orders: Order[];
}

export const OrdersTable: React.FC<OrdersTableProps> = ({ orders }) => {
  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'filled':
        return <CheckCircle className="h-4 w-4 text-green-400" />;
      case 'canceled':
      case 'rejected':
        return <XCircle className="h-4 w-4 text-red-400" />;
      case 'new':
      case 'pending_new':
        return <Clock className="h-4 w-4 text-blue-400" />;
      default:
        return <AlertCircle className="h-4 w-4 text-yellow-400" />;
    }
  };

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'filled':
        return 'text-green-400';
      case 'canceled':
      case 'rejected':
        return 'text-red-400';
      case 'new':
      case 'pending_new':
        return 'text-blue-400';
      default:
        return 'text-yellow-400';
    }
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <h2 className="text-xl font-bold text-white mb-6">Recent Orders</h2>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-gray-400 text-sm border-b border-gray-700">
              <th className="text-left py-3">Symbol</th>
              <th className="text-left py-3">Side</th>
              <th className="text-right py-3">Qty</th>
              <th className="text-right py-3">Type</th>
              <th className="text-right py-3">Price</th>
              <th className="text-left py-3">Status</th>
              <th className="text-left py-3">Time</th>
            </tr>
          </thead>
          <tbody>
            {orders.map((order) => (
              <tr key={order.id} className="border-b border-gray-700 hover:bg-gray-700/50">
                <td className="py-3">
                  <span className="font-medium text-white">{order.symbol}</span>
                </td>
                <td className="py-3">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${
                    order.side === 'buy' ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'
                  }`}>
                    {order.side.toUpperCase()}
                  </span>
                </td>
                <td className="text-right py-3 text-gray-300">{order.qty}</td>
                <td className="text-right py-3 text-gray-300 capitalize">{order.order_type}</td>
                <td className="text-right py-3 text-gray-300">
                  {order.filled_at && order.filled_avg_price > 0
                    ? `$${order.filled_avg_price.toLocaleString()}`
                    : order.limit_price
                    ? `$${order.limit_price.toLocaleString()}`
                    : 'Market'
                  }
                </td>
                <td className="py-3">
                  <div className="flex items-center">
                    {getStatusIcon(order.status)}
                    <span className={`ml-2 text-sm capitalize ${getStatusColor(order.status)}`}>
                      {order.status.replace('_', ' ')}
                    </span>
                  </div>
                </td>
                <td className="py-3 text-gray-300 text-sm">
                  {format(new Date(order.submitted_at), 'HH:mm:ss')}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};