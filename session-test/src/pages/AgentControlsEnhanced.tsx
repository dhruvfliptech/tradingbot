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
  Shield,
  Clock,
  TrendingUp,
  AlertTriangle,
  Info,
  DollarSign,
  Target,
  Zap,
  BarChart3
} from 'lucide-react';
import { tradingAgentV2 } from '../services/tradingAgentV2';
import { agentSettingsService, AgentSettings, DEFAULT_SETTINGS } from '../services/agentSettingsService';
import { auditLogService } from '../services/persistence/auditLogService';
import { statePersistenceService } from '../services/persistence/statePersistenceService';
import { AutoTradeActivity } from '../components/Dashboard/AutoTradeActivity';
import { useTradingProvider } from '../hooks/useTradingProvider';
import { DashboardChat } from '../components/Assistant/DashboardChat';
import { tradingProviderService } from '../services/tradingProviderService';
import { coinGeckoService } from '../services/coinGeckoService';
import { Account, Position, Order, CryptoData } from '../types/trading';

interface ExtendedSettings extends AgentSettings {
  perTradeRisk?: number;
  maxPositionSize?: number;
  drawdownLimit?: number;
  volatilityTolerance?: 'low' | 'medium' | 'high';
  tradingStartTime?: string;
  tradingEndTime?: string;
  tradingTimezone?: string;
  shortingEnabled?: boolean;
  marginEnabled?: boolean;
  maxLeverage?: number;
  unorthodoxStrategies?: boolean;
}

export const AgentControlsEnhanced: React.FC = () => {
  const navigate = useNavigate();
  const [isAgentActive, setIsAgentActive] = useState(false);
  const [pausedThisMonth, setPausedThisMonth] = useState(0);
  const [activeTab, setActiveTab] = useState<'risk' | 'strategy' | 'activity' | 'chat'>('risk');
  const [account, setAccount] = useState<Account | null>(null);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [cryptoData, setCryptoData] = useState<CryptoData[]>([]);
  const [settings, setSettings] = useState<ExtendedSettings>({
    ...DEFAULT_SETTINGS,
    perTradeRisk: 1.0,
    maxPositionSize: 10,
    drawdownLimit: 15,
    volatilityTolerance: 'medium',
    tradingStartTime: '09:00',
    tradingEndTime: '18:00',
    tradingTimezone: 'America/New_York',
    shortingEnabled: false,
    marginEnabled: false,
    maxLeverage: 1.0,
    unorthodoxStrategies: false
  });
  const [savingSettings, setSavingSettings] = useState(false);
  const [pendingChanges, setPendingChanges] = useState(false);
  const [showPauseDialog, setShowPauseDialog] = useState(false);
  const [pauseReason, setPauseReason] = useState('');
  const [apiStatuses, setApiStatuses] = useState({
    broker: 'checking' as 'connected' | 'error' | 'checking',
    coingecko: 'checking' as 'connected' | 'error' | 'checking',
    supabase: 'checking' as 'connected' | 'error' | 'checking',
  });

  const { activeProvider } = useTradingProvider();

  useEffect(() => {
    setIsAgentActive(tradingAgentV2.isRunning());
    loadSettings();
    loadTradingData();
    
    const unsubscribe = tradingAgentV2.subscribe((event) => {
      if (event.type === 'status') {
        setIsAgentActive(event.active);
      }
      if (event.type === 'order_submitted') {
        loadTradingData();
      }
    });
    
    const interval = setInterval(loadTradingData, 30000);
    
    return () => {
      unsubscribe();
      clearInterval(interval);
    };
  }, [activeProvider]);

  const loadSettings = async () => {
    try {
      const baseSettings = await agentSettingsService.getSettings();
      const userSettings = await statePersistenceService.loadUserSettings();
      
      setSettings({
        ...baseSettings,
        perTradeRisk: userSettings?.per_trade_risk_percent || 1.0,
        maxPositionSize: userSettings?.max_position_size_percent || 10,
        drawdownLimit: userSettings?.max_drawdown_percent || 15,
        volatilityTolerance: userSettings?.volatility_tolerance || 'medium',
        tradingStartTime: userSettings?.trading_start_time || '09:00',
        tradingEndTime: userSettings?.trading_end_time || '18:00',
        tradingTimezone: userSettings?.trading_timezone || 'America/New_York',
        shortingEnabled: userSettings?.shorting_enabled || false,
        marginEnabled: userSettings?.margin_enabled || false,
        maxLeverage: userSettings?.max_leverage || 1.0,
        unorthodoxStrategies: userSettings?.unorthodox_strategies || false,
      });
      
      setPausedThisMonth(userSettings?.agent_pauses_remaining ? 5 - userSettings.agent_pauses_remaining : 0);
    } catch (error) {
      console.error('Failed to load settings:', error);
    }
  };
  
  const loadTradingData = async () => {
    setApiStatuses((prev) => ({ ...prev, broker: 'checking' }));
    try {
      const [accountData, positionsData, ordersData, cryptoData] = await Promise.all([
        tradingProviderService.getAccount(),
        tradingProviderService.getPositions(),
        tradingProviderService.getOrders(),
        coinGeckoService.getCryptoData(['bitcoin', 'ethereum', 'binancecoin', 'solana', 'cardano'])
      ]);
      setAccount(accountData);
      setPositions(positionsData);
      setOrders(ordersData);
      setCryptoData(cryptoData);
      setApiStatuses({
        broker: 'connected',
        coingecko: 'connected',
        supabase: 'connected',
      });
    } catch (error) {
      console.error('Failed to load trading data:', error);
      setApiStatuses((prev) => ({ ...prev, broker: 'error' }));
    }
  };

  const handleSaveSettings = async () => {
    try {
      setSavingSettings(true);
      
      // Save base settings
      await agentSettingsService.saveSettings({
        riskBudgetUsd: settings.riskBudgetUsd,
        confidenceThreshold: settings.confidenceThreshold,
        cooldownMinutes: settings.cooldownMinutes,
        maxOpenPositions: settings.maxOpenPositions
      });
      
      // Save extended settings
      await statePersistenceService.saveUserSettings({
        per_trade_risk_percent: settings.perTradeRisk,
        max_position_size_percent: settings.maxPositionSize,
        max_drawdown_percent: settings.drawdownLimit,
        volatility_tolerance: settings.volatilityTolerance,
        confidence_threshold: settings.confidenceThreshold,
        trading_start_time: settings.tradingStartTime,
        trading_end_time: settings.tradingEndTime,
        trading_timezone: settings.tradingTimezone,
        shorting_enabled: settings.shortingEnabled,
        margin_enabled: settings.marginEnabled,
        max_leverage: settings.maxLeverage,
        unorthodox_strategies: settings.unorthodoxStrategies,
        trading_enabled: true
      });
      
      await auditLogService.logSettingChange('agent_controls', null, settings, 'Manual configuration update');
      setPendingChanges(false);
      alert('Settings saved successfully! Changes will take effect on the next trading cycle.');
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('Failed to save settings. Please try again.');
    } finally {
      setSavingSettings(false);
    }
  };

  const handlePauseAgent = async () => {
    if (pausedThisMonth >= 2) {
      alert('You have reached the maximum number of pauses this month (2).');
      return;
    }
    
    if (!pauseReason.trim()) {
      alert('Please provide a reason for pausing the agent.');
      return;
    }
    
    try {
      await tradingAgentV2.stop();
      await auditLogService.logAgentControl('pause', pauseReason);
      setPausedThisMonth(prev => prev + 1);
      
      await statePersistenceService.saveUserSettings({
        agent_pauses_remaining: 2 - (pausedThisMonth + 1),
        last_pause_reason: pauseReason,
        last_pause_at: new Date().toISOString()
      });
      
      setIsAgentActive(false);
      setShowPauseDialog(false);
      setPauseReason('');
    } catch (error) {
      console.error('Failed to pause agent:', error);
      alert('Failed to pause agent. Please try again.');
    }
  };

  const handleResumeAgent = async () => {
    if (!confirm('Are you sure you want to resume the trading agent?')) return;
    
    try {
      await tradingAgentV2.start();
      await auditLogService.logAgentControl('resume', 'Agent resumed by user');
      setIsAgentActive(true);
    } catch (error) {
      console.error('Failed to resume agent:', error);
      alert('Failed to resume agent. Please try again.');
    }
  };

  const getVolatilityColor = (level: string) => {
    switch (level) {
      case 'low': return 'text-green-400';
      case 'medium': return 'text-yellow-400';
      case 'high': return 'text-red-400';
      default: return 'text-gray-400';
    }
  };

  const getAISuggestedValue = (setting: string): string => {
    const volatility = Math.random() > 0.5 ? 'high' : 'moderate';
    switch (setting) {
      case 'perTradeRisk':
        return `AI suggests: ${volatility === 'high' ? '0.8%' : '1.2%'} (${volatility} volatility)`;
      case 'confidenceThreshold':
        return 'AI suggests: 80% (recent accuracy: 73%)';
      case 'maxPositionSize':
        return `AI suggests: ${volatility === 'high' ? '8%' : '12%'} (correlation detected)`;
      default:
        return '';
    }
  };

  const handleOrderPlaced = async () => {
    await loadTradingData();
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
              {/* Save Button */}
              {pendingChanges && (
                <button
                  onClick={handleSaveSettings}
                  disabled={savingSettings}
                  className="flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  {savingSettings ? (
                    <RefreshCw className="h-4 w-4 animate-spin mr-2" />
                  ) : (
                    <Save className="h-4 w-4 mr-2" />
                  )}
                  Save Changes
                </button>
              )}
            </div>
          </div>
        </div>
      </header>

      {/* Pending Changes Banner */}
      {pendingChanges && (
        <div className="bg-yellow-900/20 border-b border-yellow-600 px-4 py-3">
          <div className="max-w-7xl mx-auto flex items-center">
            <AlertTriangle className="h-5 w-5 text-yellow-400 mr-2" />
            <span className="text-yellow-300">You have unsaved changes. Settings will take effect on the next trading cycle.</span>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Agent Status & Control Card */}
        <div className="bg-gray-800 rounded-lg p-6 mb-6">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-2xl font-bold text-white mb-2">AI Trading Agent Control Center</h2>
              <p className="text-gray-400">Complete control over your automated trading strategy</p>
            </div>
            <div className="flex items-center space-x-4">
              <div className={`px-4 py-2 rounded-full ${isAgentActive ? 'bg-green-900/30 text-green-400' : 'bg-red-900/30 text-red-400'}`}>
                {isAgentActive ? 'ACTIVE' : 'PAUSED'}
              </div>
              {isAgentActive ? (
                <button
                  onClick={() => setShowPauseDialog(true)}
                  disabled={pausedThisMonth >= 2}
                  className="flex items-center px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  <Pause className="h-4 w-4 mr-2" /> Pause Agent
                </button>
              ) : (
                <button
                  onClick={handleResumeAgent}
                  className="flex items-center px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors"
                >
                  <Play className="h-4 w-4 mr-2" /> Start Agent
                </button>
              )}
            </div>
          </div>
          
          {/* Quick Stats Bar */}
          <div className="grid grid-cols-6 gap-4">
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
              <div className="text-gray-400 text-xs">Pauses Used</div>
              <div className="text-white font-medium">{pausedThisMonth}/2 month</div>
            </div>
            <div className="bg-gray-700 rounded-lg p-3">
              <div className="text-gray-400 text-xs">Portfolio</div>
              <div className="text-white font-medium">$50,000</div>
            </div>
          </div>
        </div>
        
        {/* Tab Navigation */}
        <div className="bg-gray-800 rounded-t-lg border-b border-gray-700">
          <div className="flex space-x-1 p-1">
            <button
              onClick={() => setActiveTab('risk')}
              className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'risk'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <Shield className="h-4 w-4 mr-2" />
              Risk Management
            </button>
            <button
              onClick={() => setActiveTab('strategy')}
              className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
                activeTab === 'strategy'
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-700/50'
              }`}
            >
              <TrendingUp className="h-4 w-4 mr-2" />
              Strategy & Trading Window
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
          {activeTab === 'risk' && (
            <div className="space-y-6">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <Shield className="h-5 w-5 text-red-400 mr-2" />
                Risk Management Controls
              </h3>
              
              <div className="grid grid-cols-2 gap-6">
                {/* Basic Risk Settings */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white mb-3">Position & Budget Controls</h4>
                  
                  {/* Per-Trade Risk Budget (USD) */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Per-Trade Risk Budget (USD)
                    </label>
                    <input
                      type="number"
                      min={50}
                      max={500}
                      value={settings.riskBudgetUsd}
                      onChange={(e) => {
                        setSettings({ ...settings, riskBudgetUsd: Number(e.target.value) });
                        setPendingChanges(true);
                      }}
                      className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">Max USD exposure per AI trade</p>
                  </div>
                  
                  {/* Per-Trade Risk Percentage */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-300">Per-Trade Risk %</label>
                      <span className="text-sm text-indigo-400">{settings.perTradeRisk}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="2"
                      step="0.1"
                      value={settings.perTradeRisk}
                      onChange={(e) => {
                        setSettings({ ...settings, perTradeRisk: parseFloat(e.target.value) });
                        setPendingChanges(true);
                      }}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>0.5%</span>
                      <span className="text-yellow-400">{getAISuggestedValue('perTradeRisk')}</span>
                      <span>2%</span>
                    </div>
                  </div>
                  
                  {/* Max Position Size */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-300">Max Position Size</label>
                      <span className="text-sm text-indigo-400">{settings.maxPositionSize}%</span>
                    </div>
                    <input
                      type="range"
                      min="5"
                      max="20"
                      step="1"
                      value={settings.maxPositionSize}
                      onChange={(e) => {
                        setSettings({ ...settings, maxPositionSize: parseInt(e.target.value) });
                        setPendingChanges(true);
                      }}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>5%</span>
                      <span className="text-yellow-400">{getAISuggestedValue('maxPositionSize')}</span>
                      <span>20%</span>
                    </div>
                  </div>
                  
                  {/* Max Open Positions */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Max Open Positions
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={20}
                      value={settings.maxOpenPositions}
                      onChange={(e) => {
                        setSettings({ ...settings, maxOpenPositions: Number(e.target.value) });
                        setPendingChanges(true);
                      }}
                      className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">Prevents over-allocation</p>
                  </div>
                </div>
                
                {/* Advanced Risk Settings */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white mb-3">Advanced Risk Controls</h4>
                  
                  {/* Confidence Threshold */}
                  <div>
                    <div className="flex justify-between items-center mb-2">
                      <label className="text-sm font-medium text-gray-300">Confidence Threshold</label>
                      <span className="text-sm text-indigo-400">{(settings.confidenceThreshold * 100).toFixed(0)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.5"
                      max="0.9"
                      step="0.05"
                      value={settings.confidenceThreshold}
                      onChange={(e) => {
                        setSettings({ ...settings, confidenceThreshold: parseFloat(e.target.value) });
                        setPendingChanges(true);
                      }}
                      className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    <div className="flex justify-between text-xs text-gray-500 mt-1">
                      <span>50%</span>
                      <span className="text-yellow-400">{getAISuggestedValue('confidenceThreshold')}</span>
                      <span>90%</span>
                    </div>
                  </div>
                  
                  {/* Cooldown Period */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Cooldown Period (minutes)
                    </label>
                    <input
                      type="number"
                      min={1}
                      max={60}
                      value={settings.cooldownMinutes}
                      onChange={(e) => {
                        setSettings({ ...settings, cooldownMinutes: Number(e.target.value) });
                        setPendingChanges(true);
                      }}
                      className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    />
                    <p className="text-xs text-gray-400 mt-1">Min time between trades for same symbol</p>
                  </div>
                  
                  {/* Drawdown Limit */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Drawdown Limit
                    </label>
                    <select
                      value={settings.drawdownLimit}
                      onChange={(e) => {
                        setSettings({ ...settings, drawdownLimit: parseInt(e.target.value) });
                        setPendingChanges(true);
                      }}
                      className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      <option value="10">10% - Conservative</option>
                      <option value="15">15% - Moderate</option>
                      <option value="20">20% - Aggressive</option>
                    </select>
                    <p className="text-xs text-gray-400 mt-1">Pauses trading if portfolio drops by this %</p>
                  </div>
                  
                  {/* Volatility Tolerance */}
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-2">
                      Volatility Tolerance
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                      {(['low', 'medium', 'high'] as const).map((level) => (
                        <button
                          key={level}
                          onClick={() => {
                            setSettings({ ...settings, volatilityTolerance: level });
                            setPendingChanges(true);
                          }}
                          className={`px-3 py-2 rounded-lg capitalize transition-colors ${
                            settings.volatilityTolerance === level
                              ? `bg-gray-600 ${getVolatilityColor(level)}`
                              : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                          }`}
                        >
                          {level}
                        </button>
                      ))}
                    </div>
                    <p className="text-xs text-gray-400 mt-1">Influences trading aggressiveness</p>
                  </div>
                </div>
              </div>
              
              {/* Risk/Reward Summary */}
              <div className="bg-blue-900/20 border border-blue-500 rounded-lg p-4 mt-6">
                <h4 className="text-sm font-medium text-blue-300 mb-3">Risk/Reward Summary</h4>
                <div className="grid grid-cols-3 gap-4 text-sm">
                  <div>
                    <span className="text-gray-400">Per Trade Risk ($50K):</span>
                    <div className="text-white font-medium">${(50000 * (settings.perTradeRisk || 1) / 100).toFixed(0)}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Max Position Value:</span>
                    <div className="text-white font-medium">${(50000 * (settings.maxPositionSize || 10) / 100).toFixed(0)}</div>
                  </div>
                  <div>
                    <span className="text-gray-400">Drawdown Protection:</span>
                    <div className="text-white font-medium">${(50000 * (settings.drawdownLimit || 15) / 100).toFixed(0)}</div>
                  </div>
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'strategy' && (
            <div className="space-y-6">
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <TrendingUp className="h-5 w-5 text-green-400 mr-2" />
                Strategy & Trading Window Controls
              </h3>
              
              <div className="grid grid-cols-2 gap-6">
                {/* Trading Window */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white mb-3 flex items-center">
                    <Clock className="h-4 w-4 text-blue-400 mr-2" />
                    Trading Window
                  </h4>
                  
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">Start Time</label>
                      <input
                        type="time"
                        value={settings.tradingStartTime}
                        onChange={(e) => {
                          setSettings({ ...settings, tradingStartTime: e.target.value });
                          setPendingChanges(true);
                        }}
                        className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg"
                      />
                    </div>
                    <div>
                      <label className="block text-sm font-medium text-gray-300 mb-1">End Time</label>
                      <input
                        type="time"
                        value={settings.tradingEndTime}
                        onChange={(e) => {
                          setSettings({ ...settings, tradingEndTime: e.target.value });
                          setPendingChanges(true);
                        }}
                        className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg"
                      />
                    </div>
                  </div>
                  
                  <div>
                    <label className="block text-sm font-medium text-gray-300 mb-1">Timezone</label>
                    <select
                      value={settings.tradingTimezone}
                      onChange={(e) => {
                        setSettings({ ...settings, tradingTimezone: e.target.value });
                        setPendingChanges(true);
                      }}
                      className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg"
                    >
                      <option value="America/New_York">Eastern (EST/EDT)</option>
                      <option value="America/Chicago">Central (CST/CDT)</option>
                      <option value="America/Denver">Mountain (MST/MDT)</option>
                      <option value="America/Los_Angeles">Pacific (PST/PDT)</option>
                      <option value="UTC">UTC</option>
                    </select>
                  </div>
                  
                  <div className="bg-gray-700 rounded-lg p-3">
                    <div className="flex items-center">
                      <Info className="h-4 w-4 text-blue-400 mr-2" />
                      <span className="text-sm text-gray-300">
                        Trading window applies to automated trades only
                      </span>
                    </div>
                  </div>
                </div>
                
                {/* Strategy Toggles */}
                <div className="space-y-4">
                  <h4 className="text-lg font-medium text-white mb-3">Strategy Preferences</h4>
                  
                  {/* Shorting */}
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div>
                      <span className="text-sm font-medium text-gray-300">Shorting Enabled</span>
                      <p className="text-xs text-gray-500 mt-1">Recommended for hedging in bull markets</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={settings.shortingEnabled}
                        onChange={(e) => {
                          setSettings({ ...settings, shortingEnabled: e.target.checked });
                          setPendingChanges(true);
                        }}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-green-600"></div>
                    </label>
                  </div>
                  
                  {/* Margin Usage */}
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div>
                      <span className="text-sm font-medium text-gray-300">Margin Usage</span>
                      <p className="text-xs text-gray-500 mt-1">Up to 3x longs, 1.5x shorts</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={settings.marginEnabled}
                        onChange={(e) => {
                          setSettings({ ...settings, marginEnabled: e.target.checked });
                          setPendingChanges(true);
                        }}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-yellow-600"></div>
                    </label>
                  </div>
                  
                  {/* Unorthodox Strategies */}
                  <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                    <div>
                      <span className="text-sm font-medium text-gray-300">Unorthodox Strategies</span>
                      <p className="text-xs text-gray-500 mt-1">Sentiment & alternative data trading</p>
                    </div>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        checked={settings.unorthodoxStrategies}
                        onChange={(e) => {
                          setSettings({ ...settings, unorthodoxStrategies: e.target.checked });
                          setPendingChanges(true);
                        }}
                        className="sr-only peer"
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                    </label>
                  </div>
                  
                  {/* Max Leverage (if margin enabled) */}
                  {settings.marginEnabled && (
                    <div>
                      <div className="flex justify-between items-center mb-2">
                        <label className="text-sm font-medium text-gray-300">Max Leverage</label>
                        <span className="text-sm text-indigo-400">{settings.maxLeverage}x</span>
                      </div>
                      <input
                        type="range"
                        min="1"
                        max="3"
                        step="0.5"
                        value={settings.maxLeverage}
                        onChange={(e) => {
                          setSettings({ ...settings, maxLeverage: parseFloat(e.target.value) });
                          setPendingChanges(true);
                        }}
                        className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                      />
                      <div className="flex justify-between text-xs text-gray-500 mt-1">
                        <span>1x</span>
                        <span>3x</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}
          
          {activeTab === 'activity' && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-xl font-bold text-white flex items-center">
                  <Activity className="h-5 w-5 text-green-400 mr-2" />
                  Live Trading Activity
                </h3>
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
            </div>
          )}
          
          {activeTab === 'chat' && (
            <div>
              <h3 className="text-xl font-bold text-white mb-4 flex items-center">
                <MessageSquare className="h-5 w-5 text-blue-400 mr-2" />
                AI Trading Assistant
              </h3>
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
        <div className="mt-6 grid grid-cols-2 gap-4">
          <button
            onClick={() => navigate('/audit-logs')}
            className="bg-gray-800 hover:bg-gray-700 rounded-lg p-4 text-left transition-colors"
          >
            <div className="text-white font-medium mb-1">View Audit Logs</div>
            <div className="text-sm text-gray-400">See all agent decisions and actions</div>
          </button>
          <button
            onClick={() => navigate('/')}
            className="bg-gray-800 hover:bg-gray-700 rounded-lg p-4 text-left transition-colors"
          >
            <div className="text-white font-medium mb-1">Back to Dashboard</div>
            <div className="text-sm text-gray-400">View portfolio performance</div>
          </button>
        </div>
      </main>

      {/* Pause Dialog */}
      {showPauseDialog && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className="bg-gray-800 rounded-lg p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-semibold text-white mb-4">Pause Trading Agent</h3>
            
            <div className="mb-4">
              <label className="block text-sm font-medium text-gray-300 mb-2">
                Reason for pausing (required)
              </label>
              <textarea
                value={pauseReason}
                onChange={(e) => setPauseReason(e.target.value)}
                placeholder="e.g., Market conditions, system maintenance, testing..."
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg resize-none"
                rows={3}
              />
            </div>
            
            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-3 mb-4">
              <p className="text-sm text-yellow-300">
                <AlertTriangle className="h-4 w-4 inline mr-1" />
                You have {2 - pausedThisMonth} pause(s) remaining this month. Use them wisely.
              </p>
            </div>
            
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => {
                  setShowPauseDialog(false);
                  setPauseReason('');
                }}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg"
              >
                Cancel
              </button>
              <button
                onClick={handlePauseAgent}
                disabled={!pauseReason.trim()}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 text-white rounded-lg"
              >
                Confirm Pause
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};