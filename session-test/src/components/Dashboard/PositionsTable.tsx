import React from 'react';
import { TrendingUp, TrendingDown } from 'lucide-react';
import { Position } from '../../types/trading';

interface PositionsTableProps {
  positions: Position[];
}

export const PositionsTable: React.FC<PositionsTableProps> = ({ positions }) => {
  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <h2 className="text-xl font-bold text-white mb-6">Current Positions</h2>
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="text-gray-400 text-sm border-b border-gray-700">
              <th className="text-left py-3">Symbol</th>
              <th className="text-right py-3 hidden sm:table-cell">Qty</th>
              <th className="text-right py-3">Value</th>
              <th className="text-right py-3 hidden md:table-cell">Cost Basis</th>
              <th className="text-right py-3">P&L</th>
              <th className="text-right py-3 hidden sm:table-cell">P&L %</th>
            </tr>
          </thead>
          <tbody>
            {positions.map((position) => {
              const pnl = parseFloat(position.unrealized_pl);
              const pnlPercent = parseFloat(position.unrealized_plpc) * 100;
              
              return (
                <tr key={position.symbol} className="border-b border-gray-700 hover:bg-gray-700/50">
                  <td className="py-3">
                    <div className="flex items-center">
                      <img src={position.image} alt={position.name} className="w-5 h-5 sm:w-6 sm:h-6 rounded-full mr-2" />
                      <span className="font-medium text-white">{position.symbol}</span>
                      <span className="ml-2 text-xs text-gray-400 hidden lg:inline">{position.name}</span>
                    </div>
                  </td>
                  <td className="text-right py-3 text-gray-300 hidden sm:table-cell">{position.qty}</td>
                  <td className="text-right py-3 text-gray-300">
                    <div className="text-sm sm:text-base">${position.market_value.toLocaleString()}</div>
                    <div className="text-xs text-gray-400 sm:hidden">{position.qty}</div>
                  </td>
                  <td className="text-right py-3 text-gray-300 hidden md:table-cell">
                    ${position.cost_basis.toLocaleString()}
                  </td>
                  <td className={`text-right py-3 flex items-center justify-end ${position.unrealized_pl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {position.unrealized_pl >= 0 ? <TrendingUp className="h-3 w-3 sm:h-4 sm:w-4 mr-1" /> : <TrendingDown className="h-3 w-3 sm:h-4 sm:w-4 mr-1" />}
                    <div className="text-sm sm:text-base">${position.unrealized_pl >= 0 ? '+' : ''}{position.unrealized_pl.toFixed(2)}</div>
                  </td>
                  <td className={`text-right py-3 hidden sm:table-cell ${position.unrealized_plpc >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {position.unrealized_plpc >= 0 ? '+' : ''}{(position.unrealized_plpc * 100).toFixed(2)}%
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        {positions.length === 0 && (
          <div className="text-center text-gray-400 py-8">
            <p>No positions found</p>
            <p className="text-sm mt-2">Start trading to see your positions here</p>
          </div>
        )}
      </div>
    </div>
  );
};