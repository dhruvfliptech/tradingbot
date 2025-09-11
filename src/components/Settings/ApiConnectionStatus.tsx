import React, { useState, useEffect } from 'react';
import { Wifi, WifiOff, RefreshCw, AlertCircle } from 'lucide-react';
import { API_PROVIDERS, apiKeysService } from '../../services/apiKeysService';
import { coinGeckoService } from '../../services/coinGeckoService';
import { alpacaService } from '../../services/alpacaService';
import { groqService } from '../../services/groqService';
import { supabase } from '../../lib/supabase';

interface ConnectionStatus {
  provider: string;
  status: 'connected' | 'disconnected' | 'checking' | 'error';
  lastChecked?: Date;
  error?: string;
}

export const ApiConnectionStatus: React.FC = () => {
  const [connections, setConnections] = useState<ConnectionStatus[]>([]);
  const [checking, setChecking] = useState(false);

  useEffect(() => {
    checkAllConnections();
  }, []);

  const checkApiConnectivity = async (provider: string): Promise<boolean> => {
    try {
      switch (provider) {
        case 'coingecko':
          // Test with a simple API call
          await coinGeckoService.getCryptoData(['bitcoin']);
          return true;
        
        case 'alpaca':
          // Test Alpaca connection
          await alpacaService.getAccount();
          return true;
        
        case 'groq':
          // Test Groq connection by checking if API key exists
          const groqApiKey = import.meta.env.VITE_GROQ_API_KEY;
          return !!groqApiKey;
        
        case 'supabase':
          // Test Supabase connection with an existing table
          await supabase.from('models').select('count').limit(1);
          return true;
        
        default:
          // For other providers, just check if we have environment variables
          const envKey = `VITE_${provider.toUpperCase()}_API_KEY`;
          return !!import.meta.env[envKey];
      }
    } catch (error) {
      console.warn(`API connectivity check failed for ${provider}:`, error);
      // For certain providers, fall back to environment variable check
      if (['coingecko', 'alpaca', 'groq'].includes(provider)) {
        const envKey = `VITE_${provider.toUpperCase()}_API_KEY`;
        return !!import.meta.env[envKey];
      }
      return false;
    }
  };

  const checkAllConnections = async () => {
    setChecking(true);
    const newConnections: ConnectionStatus[] = [];

    for (const provider of Object.keys(API_PROVIDERS)) {
      const config = API_PROVIDERS[provider as keyof typeof API_PROVIDERS];
      
      newConnections.push({
        provider,
        status: 'checking',
      });
    }

    setConnections(newConnections);

    // Check each provider's connection
    for (let i = 0; i < newConnections.length; i++) {
      const connection = newConnections[i];

      try {
        const isConnected = await checkApiConnectivity(connection.provider);
        
        newConnections[i] = {
          ...connection,
          status: isConnected ? 'connected' : 'disconnected',
          lastChecked: new Date(),
        };
      } catch (error) {
        newConnections[i] = {
          ...connection,
          status: 'error',
          lastChecked: new Date(),
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }

      // Update state after each check to show progress
      setConnections([...newConnections]);
    }

    setChecking(false);
  };

  const checkSingleConnection = async (provider: string) => {
    setConnections(prev => 
      prev.map(conn => 
        conn.provider === provider 
          ? { ...conn, status: 'checking' as const }
          : conn
      )
    );

    try {
      const isConnected = await checkApiConnectivity(provider);
      
      setConnections(prev => 
        prev.map(conn => 
          conn.provider === provider 
            ? { 
                ...conn, 
                status: isConnected ? 'connected' as const : 'disconnected' as const,
                lastChecked: new Date(),
                error: undefined,
              }
            : conn
        )
      );
    } catch (error) {
      setConnections(prev => 
        prev.map(conn => 
          conn.provider === provider 
            ? { 
                ...conn, 
                status: 'error' as const,
                lastChecked: new Date(),
                error: error instanceof Error ? error.message : 'Unknown error',
              }
            : conn
        )
      );
    }
  };

  const getStatusIcon = (status: ConnectionStatus['status']) => {
    switch (status) {
      case 'connected':
        return <Wifi className="h-4 w-4 text-green-500" />;
      case 'disconnected':
        return <WifiOff className="h-4 w-4 text-red-500" />;
      case 'checking':
        return <RefreshCw className="h-4 w-4 text-blue-500 animate-spin" />;
      case 'error':
        return <AlertCircle className="h-4 w-4 text-yellow-500" />;
      default:
        return <WifiOff className="h-4 w-4 text-gray-500" />;
    }
  };

  const getStatusText = (connection: ConnectionStatus) => {
    switch (connection.status) {
      case 'connected':
        return 'Connected';
      case 'disconnected':
        return 'Disconnected';
      case 'checking':
        return 'Checking...';
      case 'error':
        return `Error: ${connection.error}`;
      default:
        return 'Unknown';
    }
  };

  const getStatusColor = (status: ConnectionStatus['status']) => {
    switch (status) {
      case 'connected':
        return 'text-green-400';
      case 'disconnected':
        return 'text-red-400';
      case 'checking':
        return 'text-blue-400';
      case 'error':
        return 'text-yellow-400';
      default:
        return 'text-gray-400';
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h4 className="text-lg font-medium text-white">API Connection Status</h4>
        <div className="flex space-x-2">
          <button
            onClick={async () => {
              try {
                const result = await apiKeysService.testConnection();
                console.log('Connection test result:', result);
                if (result.error) {
                  console.error('Connection test failed:', result.error);
                } else {
                  console.log('✅ Connection test passed');
                }
              } catch (error) {
                console.error('Connection test error:', error);
              }
            }}
            className="px-3 py-1 bg-blue-600 hover:bg-blue-700 text-white rounded text-sm flex items-center space-x-1"
          >
            <span>Test Connection</span>
          </button>
          <button
            onClick={checkAllConnections}
            disabled={checking}
            className="px-3 py-1 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded text-sm flex items-center space-x-1"
          >
            <RefreshCw className={`h-4 w-4 ${checking ? 'animate-spin' : ''}`} />
            <span>Refresh All</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
        {connections.map((connection) => {
          const config = API_PROVIDERS[connection.provider as keyof typeof API_PROVIDERS];
          
          return (
            <div
              key={connection.provider}
              className="bg-gray-800 border border-gray-600 rounded-lg p-3"
            >
              <div className="flex items-center justify-between mb-2">
                <span className="font-medium text-white">{config.name}</span>
                <button
                  onClick={() => checkSingleConnection(connection.provider)}
                  disabled={connection.status === 'checking'}
                  className="p-1 hover:bg-gray-700 rounded"
                >
                  {getStatusIcon(connection.status)}
                </button>
              </div>

              <div className={`text-sm ${getStatusColor(connection.status)}`}>
                {getStatusText(connection)}
              </div>

              {connection.lastChecked && (
                <div className="text-xs text-gray-500 mt-1">
                  Last checked: {connection.lastChecked.toLocaleTimeString()}
                </div>
              )}
            </div>
          );
        })}
      </div>

      {connections.length === 0 && !checking && (
        <div className="text-center py-8">
          <WifiOff className="h-8 w-8 text-gray-400 mx-auto mb-2" />
          <p className="text-gray-400">Click "Refresh All" to check connections</p>
        </div>
      )}
    </div>
  );
};