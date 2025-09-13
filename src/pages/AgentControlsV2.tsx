import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  Play, 
  Pause, 
  AlertCircle, 
  Save, 
  RefreshCw, 
  Activity, 
  MessageSquare, 
  SlidersHorizontal,
  Settings,
  Shield
} from 'lucide-react';
import { tradingAgentV2 } from '../services/tradingAgentV2';
import { agentSettingsService, AgentSettings, DEFAULT_SETTINGS } from '../services/agentSettingsService';
import { auditLogService } from '../services/persistence/auditLogService';
import { AutoTradeSettings } from '../components/Dashboard/AutoTradeSettings';
import { AutoTradeActivity } from '../components/Dashboard/AutoTradeActivity';
import { DashboardChat } from '../components/Assistant/DashboardChat';
import { alpacaService } from '../services/alpacaService';
import { coinGeckoService } from '../services/coinGeckoService';
import { Account, Position, Order, CryptoData } from '../types/trading';

export const AgentControlsV2: React.FC = () => {
  const navigate = useNavigate();
  const [isAgentActive, setIsAgentActive] = useState(false);
  const [pausedThisMonth, setPausedThisMonth] = useState(0);
  const [activeTab, setActiveTab] = useState<'settings' | 'activity' | 'chat'>('settings');
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [cryptoData, setCryptoData] = useState<CryptoData[]>([]);
  const [settings, setSettings] = useState<AgentSettings>(DEFAULT_SETTINGS);
  const [apiStatuses, setApiStatuses] = useState({
    alpaca: 'checking' as 'connected' | 'error' | 'checking',
    coingecko: 'checking' as 'connected' | 'error' | 'checking',
    supabase: 'checking' as 'connected' | 'error' | 'checking',
  });

  useEffect(() => {
    // Check agent status
    setIsAgentActive(tradingAgentV2.isRunning());
    
    // Load settings
    loadSettings();
    loadTradingData();
    
    // Subscribe to agent events
    const unsubscribe = tradingAgentV2.subscribe((event) => {
      if (event.type === 'status') {
        setIsAgentActive(event.active);
      }
      if (event.type === 'order_submitted') {
        loadTradingData(); // Refresh data when trades happen
      }
    });
    
    // Refresh data periodically
    const interval = setInterval(loadTradingData, 30000); // 30 seconds
    
    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, []);

  const loadSettings = async () => {
    try {
      const s = await agentSettingsService.getSettings();
      setSettings(s);
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };
  
  const loadTradingData = async () => {
    try {
      const [accountData, positionsData, ordersData, cryptoData] = await Promise.all([
        alpacaService.getAccount(),
        alpacaService.getPositions(),
        alpacaService.getOrders(),
        coinGeckoService.getCryptoData(['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano'])
      ]);
      setAccount(accountData);
      setPositions(positionsData);
      setOrders(ordersData);
      setCryptoData(cryptoData);
      setApiStatuses({
        alpaca: 'connected',
        coingecko: 'connected',
        supabase: 'connected',
      });
    } catch (error) {
      console.error('Failed to load trading data:', error);
    }
  };
  
  const handleOrderPlaced = async () => {
    await loadTradingData();
  };

  const handleToggleAgent = async () => {
    if (isAgentActive) {
      // Pause logic
      if (pausedThisMonth >= 5) {
        alert('You have reached the maximum number of pauses this month.');
        return;
      }
      
      const reason = prompt('Please provide a reason for pausing the agent:');
      if (!reason) return;
      
      await tradingAgentV2.stop();
      await auditLogService.logAgentControl('pause', reason);
      setPausedThisMonth(prev => prev + 1);
      setIsAgentActive(false);
    } else {
      // Resume logic
      await tradingAgentV2.start();
      await auditLogService.logAgentControl('resume', 'Agent resumed by user');
      setIsAgentActive(true);
    }
  };

  return (
    <div className="min-h-screen bg-gray-900">
      {/* Header */}
      <header className="bg-gray-800 border-b border-gray-700">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center">
              <button
                onClick={() => navigate('/')}
                className="flex items-center text-gray-400 hover:text-white transition-colors"
              >
                <ArrowLeft className="h-5 w-5 mr-2" />
                Back to Dashboard
              </button>
              <div className="ml-8 flex items-center">
                <Settings className="h-6 w-6 text-indigo-400 mr-2" />
                <h1 className="text-xl font-bold text-white">Agent Control Center</h1>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Agent Status */}
              <div className={`px-4 py-2 rounded-full ${isAgentActive ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                {isAgentActive ? 'ACTIVE' : 'PAUSED'}
              </div>
              
              {/* Toggle Button */}
              <button
                onClick={handleToggleAgent}
                className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
                  isAgentActive
                    ? 'bg-red-600 hover:bg-red-700 text-white'
                    : 'bg-green-600 hover:bg-green-700 text-white'
                }`}
              >
                {isAgentActive ? (
                  <><Pause className="h-4 w-4 mr-2" /> Pause Agent</>
                ) : (
                  <><Play className="h-4 w-4 mr-2" /> Start Agent</>
                )}
              </button>
            </div>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Agent Status Card */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">AI Trading Agent Control Center</h2>
              <p className="text-gray-400">Manage your automated trading strategy and monitor activity</p>
            </div>
          </div>
          
          {/* Quick Stats Bar */}
          <div className="grid grid-cols-5 gap-4">
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs">Risk Budget</div>
              <div className="text-white font-medium">${settings.riskBudgetUsd}/trade</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs">Confidence</div>
              <div className="text-white font-medium">{(settings.confidenceThreshold * 100).toFixed(0)}%</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs">Cooldown</div>
              <div className="text-white font-medium">{settings.cooldownMinutes} min</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs">Max Positions</div>
              <div className="text-white font-medium">{positions.length}/{settings.maxOpenPositions}</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs">Monthly Pauses</div>
              <div className="text-white font-medium">{pausedThisMonth}/5 used</div>
            </div>
          </div>
        </div>
        
        {/* Tab Navigation */}
        <div className="bg-gray-800 rounded-t-lg border-b border-gray-700">
          <div className="flex space-x-1 p-1">
            <button
              onClick={() => setActiveTab('settings')}
              className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'settings'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <SlidersHorizontal className="h-4 w-4 mr-2" />
              Risk Settings
            </button>
            <button
              onClick={() => setActiveTab('activity')}
              className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'activity'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <Activity className="h-4 w-4 mr-2" />
              Live Activity
            </button>
            <button
              onClick={() => setActiveTab('chat')}
              className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'chat'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <MessageSquare className="h-4 w-4 mr-2" />
              AI Assistant
            </button>
          </div>
        </div>
        
        {/* Tab Content */}
        <div className="bg-gray-800 rounded-b-lg p-6">
          {activeTab === 'settings' && (
            <div>
              <div className="flex items-center mb-4">
                <Shield className="h-5 w-5 text-indigo-400 mr-2" />
                <h3 className="text-xl font-bold text-white">Risk Management Settings</h3>
              </div>
              <AutoTradeSettings />
              
              {/* Additional Safety Info */}
              <div className="mt-6 bg-blue-900/20 border border-blue-500 rounded-lg p-4">
                <h4 className="text-sm font-medium text-blue-300 mb-2">Safety Features Active</h4>
                <div className="grid grid-cols-2 gap-4 text-sm">
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-2" />
                    <span className="text-gray-300">Position limits enforced</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-2" />
                    <span className="text-gray-300">Risk management enabled</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-2" />
                    <span className="text-gray-300">Audit logging active</span>
                  </div>
                  <div className="flex items-center">
                    <div className="w-2 h-2 bg-green-400 rounded-full mr-2" />
                    <span className="text-gray-300">$50K portfolio baseline</span>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'activity' && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center">
                  <Activity className="h-5 w-5 text-green-400 mr-2" />
                  <h3 className="text-xl font-bold text-white">Live Trading Activity</h3>
                </div>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${isAgentActive ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
                  <span className="text-sm text-gray-400">
                    {isAgentActive ? 'Agent analyzing markets...' : 'Agent paused'}
                  </span>
                </div>
              </div>
              <div className="bg-gray-700 rounded-lg p-4" style={{ minHeight: '500px', maxHeight: '600px', overflowY: 'auto' }}>
                <AutoTradeActivity />
              </div>
              <div className="mt-4 text-sm text-gray-400">
                Activity feed shows real-time decisions, analysis, and trade executions. Refresh automatically every 45 seconds.
              </div>
            </div>
          )}
          
          {activeTab === 'chat' && (
            <div>
              <div className="flex items-center mb-4">
                <MessageSquare className="h-5 w-5 text-blue-400 mr-2" />
                <h3 className="text-xl font-bold text-white">AI Trading Assistant</h3>
              </div>
              <p className="text-gray-400 mb-4">
                Ask questions about trading strategies, market analysis, or get help understanding the agent's decisions.
              </p>
              <div className="bg-gray-700 rounded-lg" style={{ height: '500px' }}>
                <DashboardChat
                  account={account}
                  positions={positions}
                  orders={orders}
                  cryptoData={cryptoData}
                  apiStatuses={apiStatuses}
                  onPostTrade={handleOrderPlaced}
                  embedded={true}
                />
              </div>
            </div>
          )}
        </div>
        
        {/* Quick Actions */}
        <div className="mt-6 grid grid-cols-3 gap-4">
          <button
            onClick={() => navigate('/audit-logs')}
            className="bg-gray-800 hover:bg-gray-700 rounded-lg p-4 text-left transition-colors"
          >
            <div className="text-white font-medium mb-1">View Audit Logs</div>
            <div className="text-sm text-gray-400">See all agent decisions and actions</div>
          </button>
          <button
            onClick={() => navigate('/settings')}
            className="bg-gray-800 hover:bg-gray-700 rounded-lg p-4 text-left transition-colors"
          >
            <div className="text-white font-medium mb-1">Portfolio Settings</div>
            <div className="text-sm text-gray-400">Reset portfolio or export data</div>
          </button>
          <button
            onClick={() => window.open('https://docs.trading-agent.ai', '_blank')}
            className="bg-gray-800 hover:bg-gray-700 rounded-lg p-4 text-left transition-colors"
          >
            <div className="text-white font-medium mb-1">Documentation</div>
            <div className="text-sm text-gray-400">Learn about agent strategies</div>
          </button>
        </div>
      </main>
    </div>
  );
};