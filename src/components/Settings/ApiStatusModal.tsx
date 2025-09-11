import React, { useState, useEffect } from 'react';
import { X, CheckCircle, XCircle, AlertCircle, RefreshCw, Settings, Wifi, WifiOff } from 'lucide-react';
import { alpacaService } from '../../services/alpacaService';
import { coinGeckoService } from '../../services/coinGeckoService';
import { supabase } from '../../lib/supabase';

interface ApiStatusModalProps {
  isOpen: boolean;
  onClose: () => void;
}

interface ApiStatus {
  name: string;
  status: 'connected' | 'error' | 'checking';
  message: string;
  lastChecked: Date;
  icon: React.ReactNode;
  color: string;
}

export const ApiStatusModal: React.FC<ApiStatusModalProps> = ({ isOpen, onClose }) => {
  const [apiStatuses, setApiStatuses] = useState<ApiStatus[]>([
    {
      name: 'Alpaca Trading API',
      status: 'checking',
      message: 'Checking connection...',
      lastChecked: new Date(),
      icon: <RefreshCw className="h-5 w-5 animate-spin" />,
      color: 'text-yellow-400',
    },
    {
      name: 'CoinGecko Market Data',
      status: 'checking',
      message: 'Checking connection...',
      lastChecked: new Date(),
      icon: <RefreshCw className="h-5 w-5 animate-spin" />,
      color: 'text-yellow-400',
    },
    {
      name: 'Supabase Database',
      status: 'checking',
      message: 'Checking connection...',
      lastChecked: new Date(),
      icon: <RefreshCw className="h-5 w-5 animate-spin" />,
      color: 'text-yellow-400',
    },
  ]);

  const checkApiStatus = async () => {
    const newStatuses = [...apiStatuses];

    // Check Alpaca API
    try {
      await alpacaService.getAccount();
      newStatuses[0] = {
        name: 'Alpaca Trading API',
        status: 'connected',
        message: 'Connected to paper trading account',
        lastChecked: new Date(),
        icon: <CheckCircle className="h-5 w-5" />,
        color: 'text-green-400',
      };
    } catch (error) {
      newStatuses[0] = {
        name: 'Alpaca Trading API',
        status: 'error',
        message: error instanceof Error ? error.message : 'Connection failed',
        lastChecked: new Date(),
        icon: <XCircle className="h-5 w-5" />,
        color: 'text-red-400',
      };
    }

    // Check CoinGecko API
    try {
      await coinGeckoService.getCryptoData(['bitcoin']);
      newStatuses[1] = {
        name: 'CoinGecko Market Data',
        status: 'connected',
        message: 'Real-time crypto data available',
        lastChecked: new Date(),
        icon: <CheckCircle className="h-5 w-5" />,
        color: 'text-green-400',
      };
    } catch (error) {
      newStatuses[1] = {
        name: 'CoinGecko Market Data',
        status: 'error',
        message: 'Using fallback data - API unavailable',
        lastChecked: new Date(),
        icon: <AlertCircle className="h-5 w-5" />,
        color: 'text-yellow-400',
      };
    }

    // Check Supabase
    try {
      const { data, error } = await supabase.from('portfolios').select('count').limit(1);
      if (error) throw error;
      newStatuses[2] = {
        name: 'Supabase Database',
        status: 'connected',
        message: 'Database connection active',
        lastChecked: new Date(),
        icon: <CheckCircle className="h-5 w-5" />,
        color: 'text-green-400',
      };
    } catch (error) {
      newStatuses[2] = {
        name: 'Supabase Database',
        status: 'error',
        message: error instanceof Error ? error.message : 'Database connection failed',
        lastChecked: new Date(),
        icon: <XCircle className="h-5 w-5" />,
        color: 'text-red-400',
      };
    }

    setApiStatuses(newStatuses);
  };

  useEffect(() => {
    if (isOpen) {
      checkApiStatus();
    }
  }, [isOpen]);

  const handleRefresh = () => {
    setApiStatuses(prev => prev.map(status => ({
      ...status,
      status: 'checking' as const,
      message: 'Checking connection...',
      icon: <RefreshCw className="h-5 w-5 animate-spin" />,
      color: 'text-yellow-400',
    })));
    checkApiStatus();
  };

  const getOverallStatus = () => {
    const connectedCount = apiStatuses.filter(api => api.status === 'connected').length;
    const errorCount = apiStatuses.filter(api => api.status === 'error').length;
    
    if (connectedCount === apiStatuses.length) {
      return { icon: <Wifi className="h-5 w-5" />, color: 'text-green-400', text: 'All Systems Operational' };
    } else if (errorCount === apiStatuses.length) {
      return { icon: <WifiOff className="h-5 w-5" />, color: 'text-red-400', text: 'Multiple System Issues' };
    } else {
      return { icon: <AlertCircle className="h-5 w-5" />, color: 'text-yellow-400', text: 'Partial System Issues' };
    }
  };

  if (!isOpen) return null;

  const overallStatus = getOverallStatus();

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
      <div className="bg-gray-800 rounded-lg p-6 w-full max-w-2xl mx-4 max-h-[90vh] overflow-y-auto">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center">
            <Settings className="h-6 w-6 text-blue-400 mr-3" />
            <h2 className="text-2xl font-bold text-white">API Status Dashboard</h2>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-white transition-colors"
          >
            <X className="h-6 w-6" />
          </button>
        </div>

        {/* Overall Status */}
        <div className="bg-gray-700 rounded-lg p-4 mb-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center">
              <div className={overallStatus.color}>
                {overallStatus.icon}
              </div>
              <div className="ml-3">
                <h3 className="text-white font-semibold">System Status</h3>
                <p className={`text-sm ${overallStatus.color}`}>{overallStatus.text}</p>
              </div>
            </div>
            <button
              onClick={handleRefresh}
              className="flex items-center px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors text-sm"
            >
              <RefreshCw className="h-4 w-4 mr-2" />
              Refresh All
            </button>
          </div>
        </div>

        {/* Individual API Status */}
        <div className="space-y-4">
          {apiStatuses.map((api, index) => (
            <div key={index} className="bg-gray-700 rounded-lg p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center">
                  <div className={api.color}>
                    {api.icon}
                  </div>
                  <h3 className="text-white font-semibold ml-3">{api.name}</h3>
                </div>
                <span className={`px-2 py-1 rounded text-xs font-medium ${
                  api.status === 'connected' ? 'bg-green-900/30 text-green-400' :
                  api.status === 'error' ? 'bg-red-900/30 text-red-400' :
                  'bg-yellow-900/30 text-yellow-400'
                }`}>
                  {api.status.toUpperCase()}
                </span>
              </div>
              <p className="text-gray-300 text-sm mb-2">{api.message}</p>
              <p className="text-gray-400 text-xs">
                Last checked: {api.lastChecked.toLocaleTimeString()}
              </p>
            </div>
          ))}
        </div>

        {/* API Information */}
        <div className="mt-6 bg-gray-700 rounded-lg p-4">
          <h3 className="text-white font-semibold mb-3">API Information</h3>
          <div className="space-y-2 text-sm text-gray-300">
            <div className="flex justify-between">
              <span>Alpaca Environment:</span>
              <span className="text-blue-400">Paper Trading</span>
            </div>
            <div className="flex justify-between">
              <span>CoinGecko Plan:</span>
              <span className="text-blue-400">Free Tier</span>
            </div>
            <div className="flex justify-between">
              <span>Supabase Project:</span>
              <span className="text-blue-400">Connected</span>
            </div>
          </div>
        </div>

        <div className="mt-6 text-center">
          <button
            onClick={onClose}
            className="px-6 py-2 bg-gray-600 hover:bg-gray-500 text-white rounded-lg transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
};