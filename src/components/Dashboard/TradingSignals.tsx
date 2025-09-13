import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Minus, Clock, Target, Shield, BarChart3, Activity, CheckCircle, XCircle } from 'lucide-react';
import { TradingSignal } from '../../types/trading';
import { tradingAgentV2 } from '../../services/tradingAgentV2';
import { CryptoData } from '../../types/trading';

interface TradingSignalsProps {
  cryptoData?: CryptoData[];
}

export const TradingSignals: React.FC<TradingSignalsProps> = ({ cryptoData = [] }) => {

  const [signals, setSignals] = useState<TradingSignal[]>([]);

  useEffect(() => {
    // Get signals from the trading agent v2
    const updateSignals = () => {
      const currentSignals = tradingAgentV2.getSignals();
      setSignals(currentSignals);
    };
    
    // Initial load
    updateSignals();
    
    // Subscribe to agent updates
    const unsubscribe = tradingAgentV2.subscribe((event) => {
      if (event.type === 'analysis') {
        updateSignals();
      }
    });
    
    return () => unsubscribe();
  }, [cryptoData]);

  const getActionIcon = (action: string) => {
    switch (action) {
      case 'BUY':
        return <TrendingUp className="h-4 w-4" />;
      case 'SELL':
        return <TrendingDown className="h-4 w-4" />;
      default:
        return <Minus className="h-4 w-4" />;
    }
  };

  const getActionColor = (action: string) => {
    switch (action) {
      case 'BUY':
        return 'text-green-400';
      case 'SELL':
        return 'text-red-400';
      default:
        return 'text-yellow-400';
    }
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <h2 className="text-xl font-bold text-white mb-6">Trading Signals</h2>
      <div className="space-y-4">
        {signals.map((signal, index) => (
          <div key={index} className="bg-gray-700 rounded-lg p-4">
            <div className="flex items-center justify-between mb-3">
              <div className="flex items-center">
                <div className={`${getActionColor(signal.action)} mr-3`}>
                  {getActionIcon(signal.action)}
                </div>
                <div>
                  <h3 className="text-white font-semibold">{signal.symbol}</h3>
                  <div className="flex items-center mt-1">
                    <span className={`text-sm font-medium ${getActionColor(signal.action)}`}>
                      {signal.action}
                    </span>
                    <span className="text-gray-400 text-sm ml-2">
                      Confidence: {signal.confidence.toFixed(0)}%
                    </span>
                  </div>
                </div>
              </div>
              <div className="text-right">
                <div className="flex items-center text-gray-400 text-sm">
                  <Clock className="h-3 w-3 mr-1" />
                  {new Date(signal.timestamp).toLocaleTimeString()}
                </div>
                <div className="w-16 bg-gray-600 rounded-full h-2 mt-2">
                  <div
                    className={`h-2 rounded-full ${
                      signal.confidence > 70 ? 'bg-green-400' : 
                      signal.confidence > 50 ? 'bg-yellow-400' : 'bg-red-400'
                    }`}
                    style={{ width: `${signal.confidence}%` }}
                  ></div>
                </div>
              </div>
            </div>
            
            {signal.reason && <p className="text-gray-300 text-sm mb-3">{signal.reason}</p>}
            
            {/* Technical Indicators */}
            <div className="bg-gray-600 rounded-lg p-3 mb-3">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <BarChart3 className="h-4 w-4 text-blue-400 mr-2" />
                  <span className="text-white text-sm font-medium">Technical Analysis</span>
                </div>
                {signal.technicalScore && (
                  <span className="text-xs text-gray-300">
                    Score: {signal.technicalScore.toFixed(0)}/100
                  </span>
                )}
              </div>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 text-xs">
                <div>
                  <p className="text-gray-400">RSI (14)</p>
                  <p className={`font-medium ${
                    (signal.rsi || 0) > 70 ? 'text-red-400' : 
                    (signal.rsi || 0) < 30 ? 'text-green-400' : 'text-yellow-400'
                  }`}>
                    {(signal.rsi || 0).toFixed(1)}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">MACD</p>
                  <p className={`font-medium ${(signal.macd || 0) >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                    {(signal.macd || 0) >= 0 ? '+' : ''}{(signal.macd || 0).toFixed(3)}
                  </p>
                </div>
                <div>
                  <p className="text-gray-400">MA Trend</p>
                  <p className={`font-medium ${
                    signal.maIndicator === 'bullish' ? 'text-green-400' :
                    signal.maIndicator === 'bearish' ? 'text-red-400' : 'text-yellow-400'
                  }`}>
                    {signal.maIndicator || 'neutral'}
                  </p>
                </div>
                {signal.ma20 && (
                  <div>
                    <p className="text-gray-400">MA (20)</p>
                    <p className={`font-medium ${
                      (signal.current_price || signal.price || 0) > signal.ma20 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      ${signal.ma20.toLocaleString()}
                    </p>
                  </div>
                )}
                {signal.ma50 && (
                  <div>
                    <p className="text-gray-400">MA (50)</p>
                    <p className={`font-medium ${
                      (signal.current_price || signal.price || 0) > signal.ma50 ? 'text-green-400' : 'text-red-400'
                    }`}>
                      ${signal.ma50.toLocaleString()}
                    </p>
                  </div>
                )}
                {signal.volume_indicator && (
                  <div>
                    <p className="text-gray-400">Volume</p>
                    <p className="text-white font-medium">
                      {signal.volume_indicator}
                    </p>
                  </div>
                )}
              </div>
            </div>
            
            {/* Validator Results */}
            {signal.validationDetails && (
              <div className="bg-gray-600 rounded-lg p-3 mb-3">
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center">
                    <Shield className="h-4 w-4 text-purple-400 mr-2" />
                    <span className="text-white text-sm font-medium">6-Validator System</span>
                  </div>
                  <span className={`text-xs font-medium ${
                    signal.validationDetails.passed ? 'text-green-400' : 'text-red-400'
                  }`}>
                    {signal.validationDetails.passed ? 'PASSED' : 'FAILED'}
                  </span>
                </div>
                <div className="grid grid-cols-2 gap-2 text-xs">
                  {Object.entries(signal.validationDetails.validators || {}).map(([name, result]: [string, any]) => (
                    <div key={name} className="flex items-center justify-between">
                      <span className="text-gray-400 capitalize">{name}:</span>
                      <div className="flex items-center">
                        {result.passed ? 
                          <CheckCircle className="h-3 w-3 text-green-400 mr-1" /> : 
                          <XCircle className="h-3 w-3 text-red-400 mr-1" />
                        }
                        <span className={result.passed ? 'text-green-400' : 'text-red-400'}>
                          {result.score.toFixed(0)}%
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
                {signal.validationDetails.recommendation && (
                  <div className="mt-2 pt-2 border-t border-gray-500">
                    <span className="text-xs text-gray-400">Recommendation: </span>
                    <span className={`text-xs font-medium ${
                      signal.validationDetails.recommendation.includes('buy') ? 'text-green-400' :
                      signal.validationDetails.recommendation.includes('sell') ? 'text-red-400' :
                      'text-yellow-400'
                    }`}>
                      {signal.validationDetails.recommendation.toUpperCase()}
                    </span>
                  </div>
                )}
              </div>
            )}
            
            {(signal.price_target || signal.stop_loss) && (
              <div className="flex space-x-4 text-sm">
                {signal.price_target && (
                  <div className="flex items-center text-green-400">
                    <Target className="h-3 w-3 mr-1" />
                    Target: ${signal.price_target.toFixed(2)}
                  </div>
                )}
                {signal.stop_loss && (
                  <div className="flex items-center text-red-400">
                    <Shield className="h-3 w-3 mr-1" />
                    Stop: ${signal.stop_loss.toFixed(2)}
                  </div>
                )}
              </div>
            )}
          </div>
        ))}
        
        {signals.length === 0 && (
          <div className="text-center text-gray-400 py-8">
            <p>No trading signals available</p>
            <p className="text-sm mt-2">Start the trading agent to generate signals</p>
          </div>
        )}
      </div>
    </div>
  );
};