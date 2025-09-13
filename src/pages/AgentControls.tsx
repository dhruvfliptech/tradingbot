import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  Clock, 
  Shield, 
  TrendingUp, 
  AlertTriangle,
  Save,
  RefreshCw,
  Info,
  Pause,
  Play,
  Settings
} from 'lucide-react';
import { statePersistenceService } from '../services/persistence/statePersistenceService';
import { auditLogService } from '../services/persistence/auditLogService';
import { tradingAgentV2 } from '../services/tradingAgentV2';

interface RiskSettings {
  perTradeRisk: number;
  maxPositionSize: number;
  drawdownLimit: number;
  volatilityTolerance: 'low' | 'medium' | 'high';
  confidenceThreshold: number;
  riskRewardMinimum: number;
}

interface StrategySettings {
  shortingEnabled: boolean;
  marginEnabled: boolean;
  maxLeverage: number;
  unorthodoxStrategies: boolean;
}

interface TradingWindow {
  startTime: string;
  endTime: string;
  timezone: string;
  overrideEnabled: boolean;
}

export const AgentControls: React.FC = () => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pendingChanges, setPendingChanges] = useState(false);
  const [agentActive, setAgentActive] = useState(false);
  const [pausesRemaining, setPausesRemaining] = useState(2);
  const [showPauseDialog, setShowPauseDialog] = useState(false);
  const [pauseReason, setPauseReason] = useState('');
  
  // Risk Management State
  const [riskSettings, setRiskSettings] = useState<RiskSettings>({
    perTradeRisk: 1.0,
    maxPositionSize: 10,
    drawdownLimit: 15,
    volatilityTolerance: 'medium',
    confidenceThreshold: 75,
    riskRewardMinimum: 3.0
  });

  // Strategy Settings State
  const [strategySettings, setStrategySettings] = useState<StrategySettings>({
    shortingEnabled: false,
    marginEnabled: false,
    maxLeverage: 1.0,
    unorthodoxStrategies: false
  });

  // Trading Window State
  const [tradingWindow, setTradingWindow] = useState<TradingWindow>({
    startTime: '09:00',
    endTime: '18:00',
    timezone: 'America/New_York',
    overrideEnabled: false
  });

  useEffect(() => {
    loadSettings();
    checkAgentStatus();
  }, []);

  const loadSettings = async () => {
    try {
      setLoading(true);
      const settings = await statePersistenceService.loadUserSettings();
      
      if (settings) {
        setRiskSettings({
          perTradeRisk: settings.per_trade_risk_percent || 1.0,
          maxPositionSize: settings.max_position_size_percent || 10,
          drawdownLimit: settings.max_drawdown_percent || 15,
          volatilityTolerance: settings.volatility_tolerance || 'medium',
          confidenceThreshold: settings.confidence_threshold || 75,
          riskRewardMinimum: settings.risk_reward_minimum || 3.0
        });
        
        setStrategySettings({
          shortingEnabled: settings.shorting_enabled || false,
          marginEnabled: settings.margin_enabled || false,
          maxLeverage: settings.max_leverage || 1.0,
          unorthodoxStrategies: settings.unorthodox_strategies || false
        });
        
        setTradingWindow({
          startTime: settings.trading_start_time || '09:00',
          endTime: settings.trading_end_time || '18:00',
          timezone: settings.trading_timezone || 'America/New_York',
          overrideEnabled: false
        });
        
        setPausesRemaining(settings.agent_pauses_remaining || 2);
      }
    } catch (error) {
      console.error('Failed to load settings:', error);
    } finally {
      setLoading(false);
    }
  };

  const checkAgentStatus = () => {
    setAgentActive(tradingAgentV2.isRunning());
  };

  const handleSaveSettings = async () => {
    try {
      setSaving(true);
      
      const settings = {
        per_trade_risk_percent: riskSettings.perTradeRisk,
        max_position_size_percent: riskSettings.maxPositionSize,
        max_drawdown_percent: riskSettings.drawdownLimit,
        volatility_tolerance: riskSettings.volatilityTolerance,
        confidence_threshold: riskSettings.confidenceThreshold,
        risk_reward_minimum: riskSettings.riskRewardMinimum,
        shorting_enabled: strategySettings.shortingEnabled,
        margin_enabled: strategySettings.marginEnabled,
        max_leverage: strategySettings.maxLeverage,
        unorthodox_strategies: strategySettings.unorthodoxStrategies,
        trading_start_time: tradingWindow.startTime,
        trading_end_time: tradingWindow.endTime,
        trading_timezone: tradingWindow.timezone
      };
      
      await statePersistenceService.saveUserSettings(settings);
      
      // Log settings change
      await auditLogService.logSettingChange(
        'agent_controls',
        null,
        settings,
        'Manual configuration update'
      );
      
      setPendingChanges(false);
      
      // Show success message
      alert('Settings saved successfully! Changes will take effect on the next trading cycle.');
    } catch (error) {
      console.error('Failed to save settings:', error);
      alert('Failed to save settings. Please try again.');
    } finally {
      setSaving(false);
    }
  };

  const handlePauseAgent = async () => {
    if (pausesRemaining <= 0) {
      alert('You have no pauses remaining this month.');
      return;
    }
    
    if (!pauseReason.trim()) {
      alert('Please provide a reason for pausing the agent.');
      return;
    }
    
    try {
      await tradingAgentV2.stop();
      await auditLogService.logAgentControl('pause', pauseReason);
      
      // Update pauses remaining
      setPausesRemaining(prev => prev - 1);
      await statePersistenceService.saveUserSettings({
        agent_pauses_remaining: pausesRemaining - 1,
        last_pause_reason: pauseReason,
        last_pause_at: new Date().toISOString()
      });
      
      setAgentActive(false);
      setShowPauseDialog(false);
      setPauseReason('');
    } catch (error) {
      console.error('Failed to pause agent:', error);
      alert('Failed to pause agent. Please try again.');
    }
  };

  const handleResumeAgent = async () => {
    if (!confirm('Are you sure you want to resume the trading agent?')) {
      return;
    }
    
    try {
      await tradingAgentV2.start();
      setAgentActive(true);
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
    // Placeholder for AI suggestions based on market conditions
    switch (setting) {
      case 'perTradeRisk':
        return 'AI suggests: 0.8% (current volatility is high)';
      case 'confidenceThreshold':
        return 'AI suggests: 80% (recent accuracy: 73%)';
      case 'maxPositionSize':
        return 'AI suggests: 8% (high correlation detected)';
      default:
        return '';
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gray-900 flex items-center justify-center">
        <RefreshCw className="h-8 w-8 animate-spin text-gray-400" />
      </div>
    );
  }

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
                <h1 className="text-xl font-bold text-white">Agent Controls</h1>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              {/* Agent Status */}
              <div className={`flex items-center px-4 py-2 rounded-lg ${agentActive ? 'bg-green-900/30' : 'bg-red-900/30'}`}>
                {agentActive ? (
                  <>
                    <Play className="h-4 w-4 text-green-400 mr-2" />
                    <span className="text-green-400">Agent Active</span>
                  </>
                ) : (
                  <>
                    <Pause className="h-4 w-4 text-red-400 mr-2" />
                    <span className="text-red-400">Agent Paused</span>
                  </>
                )}
              </div>
              
              {/* Save Button */}
              {pendingChanges && (
                <button
                  onClick={handleSaveSettings}
                  disabled={saving}
                  className="flex items-center px-4 py-2 bg-indigo-600 hover:bg-indigo-700 disabled:bg-gray-600 text-white rounded-lg transition-colors"
                >
                  {saving ? (
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
          <div className="max-w-7xl mx-auto flex items-center justify-between">
            <div className="flex items-center">
              <AlertTriangle className="h-5 w-5 text-yellow-400 mr-2" />
              <span className="text-yellow-300">You have unsaved changes. Settings will take effect on the next trading cycle.</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          
          {/* Trading Window Configuration */}
          <section className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <Clock className="h-5 w-5 text-blue-400 mr-2" />
              <h2 className="text-lg font-semibold text-white">Trading Window</h2>
            </div>
            
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">Start Time</label>
                  <input
                    type="time"
                    value={tradingWindow.startTime}
                    onChange={(e) => {
                      setTradingWindow(prev => ({ ...prev, startTime: e.target.value }));
                      setPendingChanges(true);
                    }}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-300 mb-1">End Time</label>
                  <input
                    type="time"
                    value={tradingWindow.endTime}
                    onChange={(e) => {
                      setTradingWindow(prev => ({ ...prev, endTime: e.target.value }));
                      setPendingChanges(true);
                    }}
                    className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                  />
                </div>
              </div>
              
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Timezone</label>
                <select
                  value={tradingWindow.timezone}
                  onChange={(e) => {
                    setTradingWindow(prev => ({ ...prev, timezone: e.target.value }));
                    setPendingChanges(true);
                  }}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="America/New_York">Eastern (EST/EDT)</option>
                  <option value="America/Chicago">Central (CST/CDT)</option>
                  <option value="America/Denver">Mountain (MST/MDT)</option>
                  <option value="America/Los_Angeles">Pacific (PST/PDT)</option>
                  <option value="UTC">UTC</option>
                </select>
              </div>
              
              <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <span className="text-sm text-gray-300">Override for Testing</span>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={tradingWindow.overrideEnabled}
                    onChange={(e) => {
                      setTradingWindow(prev => ({ ...prev, overrideEnabled: e.target.checked }));
                      setPendingChanges(true);
                    }}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-indigo-600"></div>
                </label>
              </div>
            </div>
          </section>

          {/* Risk Management Controls */}
          <section className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <Shield className="h-5 w-5 text-red-400 mr-2" />
              <h2 className="text-lg font-semibold text-white">Risk Management</h2>
            </div>
            
            <div className="space-y-4">
              {/* Per-Trade Risk Budget */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-sm font-medium text-gray-300">Per-Trade Risk Budget</label>
                  <span className="text-sm text-indigo-400">{riskSettings.perTradeRisk}%</span>
                </div>
                <input
                  type="range"
                  min="0.5"
                  max="2"
                  step="0.1"
                  value={riskSettings.perTradeRisk}
                  onChange={(e) => {
                    setRiskSettings(prev => ({ ...prev, perTradeRisk: parseFloat(e.target.value) }));
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
                <div className="flex justify-between items-center mb-1">
                  <label className="text-sm font-medium text-gray-300">Max Position Size</label>
                  <span className="text-sm text-indigo-400">{riskSettings.maxPositionSize}%</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="20"
                  step="1"
                  value={riskSettings.maxPositionSize}
                  onChange={(e) => {
                    setRiskSettings(prev => ({ ...prev, maxPositionSize: parseInt(e.target.value) }));
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

              {/* Drawdown Limit */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-sm font-medium text-gray-300">Drawdown Limit</label>
                  <span className="text-sm text-indigo-400">{riskSettings.drawdownLimit}%</span>
                </div>
                <select
                  value={riskSettings.drawdownLimit}
                  onChange={(e) => {
                    setRiskSettings(prev => ({ ...prev, drawdownLimit: parseInt(e.target.value) }));
                    setPendingChanges(true);
                  }}
                  className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white"
                >
                  <option value="10">10% - Conservative</option>
                  <option value="15">15% - Moderate</option>
                  <option value="20">20% - Aggressive</option>
                </select>
              </div>

              {/* Volatility Tolerance */}
              <div>
                <label className="block text-sm font-medium text-gray-300 mb-1">Volatility Tolerance</label>
                <div className="grid grid-cols-3 gap-2">
                  {(['low', 'medium', 'high'] as const).map((level) => (
                    <button
                      key={level}
                      onClick={() => {
                        setRiskSettings(prev => ({ ...prev, volatilityTolerance: level }));
                        setPendingChanges(true);
                      }}
                      className={`px-3 py-2 rounded-lg capitalize transition-colors ${
                        riskSettings.volatilityTolerance === level
                          ? `bg-gray-600 ${getVolatilityColor(level)}`
                          : 'bg-gray-700 text-gray-400 hover:bg-gray-600'
                      }`}
                    >
                      {level}
                    </button>
                  ))}
                </div>
              </div>

              {/* Confidence Threshold */}
              <div>
                <div className="flex justify-between items-center mb-1">
                  <label className="text-sm font-medium text-gray-300">Confidence Threshold</label>
                  <span className="text-sm text-indigo-400">{riskSettings.confidenceThreshold}%</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="90"
                  step="5"
                  value={riskSettings.confidenceThreshold}
                  onChange={(e) => {
                    setRiskSettings(prev => ({ ...prev, confidenceThreshold: parseInt(e.target.value) }));
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
            </div>
          </section>

          {/* Strategy Controls */}
          <section className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <TrendingUp className="h-5 w-5 text-green-400 mr-2" />
              <h2 className="text-lg font-semibold text-white">Strategy Controls</h2>
            </div>
            
            <div className="space-y-4">
              {/* Shorting */}
              <div className="flex items-center justify-between p-3 bg-gray-700 rounded-lg">
                <div>
                  <span className="text-sm font-medium text-gray-300">Shorting Enabled</span>
                  <p className="text-xs text-gray-500 mt-1">Recommended for hedging</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={strategySettings.shortingEnabled}
                    onChange={(e) => {
                      setStrategySettings(prev => ({ ...prev, shortingEnabled: e.target.checked }));
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
                    checked={strategySettings.marginEnabled}
                    onChange={(e) => {
                      setStrategySettings(prev => ({ ...prev, marginEnabled: e.target.checked }));
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
                  <p className="text-xs text-gray-500 mt-1">Sentiment & alternative data</p>
                </div>
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={strategySettings.unorthodoxStrategies}
                    onChange={(e) => {
                      setStrategySettings(prev => ({ ...prev, unorthodoxStrategies: e.target.checked }));
                      setPendingChanges(true);
                    }}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-purple-600"></div>
                </label>
              </div>

              {strategySettings.marginEnabled && (
                <div>
                  <div className="flex justify-between items-center mb-1">
                    <label className="text-sm font-medium text-gray-300">Max Leverage</label>
                    <span className="text-sm text-indigo-400">{strategySettings.maxLeverage}x</span>
                  </div>
                  <input
                    type="range"
                    min="1"
                    max="3"
                    step="0.5"
                    value={strategySettings.maxLeverage}
                    onChange={(e) => {
                      setStrategySettings(prev => ({ ...prev, maxLeverage: parseFloat(e.target.value) }));
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
          </section>

          {/* Safety Controls */}
          <section className="bg-gray-800 rounded-lg p-6">
            <div className="flex items-center mb-4">
              <AlertTriangle className="h-5 w-5 text-yellow-400 mr-2" />
              <h2 className="text-lg font-semibold text-white">Safety Controls</h2>
            </div>
            
            <div className="space-y-4">
              <div className="bg-gray-700 rounded-lg p-4">
                <div className="flex justify-between items-center mb-3">
                  <span className="text-sm font-medium text-gray-300">Agent Control</span>
                  <span className="text-xs text-gray-500">Pauses remaining: {pausesRemaining}/2 this month</span>
                </div>
                
                {agentActive ? (
                  <button
                    onClick={() => setShowPauseDialog(true)}
                    disabled={pausesRemaining <= 0}
                    className="w-full px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors flex items-center justify-center"
                  >
                    <Pause className="h-4 w-4 mr-2" />
                    Pause Agent
                  </button>
                ) : (
                  <button
                    onClick={handleResumeAgent}
                    className="w-full px-4 py-2 bg-green-600 hover:bg-green-700 text-white rounded-lg transition-colors flex items-center justify-center"
                  >
                    <Play className="h-4 w-4 mr-2" />
                    Resume Agent
                  </button>
                )}
                
                <div className="mt-3 p-3 bg-gray-800 rounded">
                  <p className="text-xs text-gray-400">
                    <Info className="h-3 w-3 inline mr-1" />
                    Pausing the agent stops all automated trading. Use sparingly as it may impact performance.
                  </p>
                </div>
              </div>

              <div className="bg-blue-900/20 border border-blue-500 rounded-lg p-4">
                <h3 className="text-sm font-medium text-blue-300 mb-2">Risk/Reward Settings</h3>
                <div className="space-y-2">
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Minimum Risk/Reward Ratio:</span>
                    <span className="text-white">1:{riskSettings.riskRewardMinimum}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Per Trade Risk ($50K portfolio):</span>
                    <span className="text-white">${(50000 * riskSettings.perTradeRisk / 100).toFixed(0)}</span>
                  </div>
                  <div className="flex justify-between text-sm">
                    <span className="text-gray-400">Expected Gain per Trade:</span>
                    <span className="text-green-400">
                      ${(50000 * riskSettings.perTradeRisk / 100 * riskSettings.riskRewardMinimum).toFixed(0)}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Audit Log Preview */}
        <section className="mt-8 bg-gray-800 rounded-lg p-6">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-lg font-semibold text-white">Recent Agent Decisions</h2>
            <button
              onClick={() => navigate('/audit-logs')}
              className="text-sm text-indigo-400 hover:text-indigo-300"
            >
              View Full Audit Log →
            </button>
          </div>
          
          <div className="text-gray-400 text-sm">
            <p>Audit log viewer will show last 100 decisions with filters...</p>
          </div>
        </section>
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
                className="w-full px-3 py-2 bg-gray-700 border border-gray-600 rounded-lg text-white placeholder-gray-500 resize-none"
                rows={3}
              />
            </div>
            
            <div className="bg-yellow-900/20 border border-yellow-600 rounded-lg p-3 mb-4">
              <p className="text-sm text-yellow-300">
                <AlertTriangle className="h-4 w-4 inline mr-1" />
                You have {pausesRemaining} pause(s) remaining this month. Use them wisely.
              </p>
            </div>
            
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => {
                  setShowPauseDialog(false);
                  setPauseReason('');
                }}
                className="px-4 py-2 bg-gray-700 hover:bg-gray-600 text-white rounded-lg transition-colors"
              >
                Cancel
              </button>
              <button
                onClick={handlePauseAgent}
                disabled={!pauseReason.trim()}
                className="px-4 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg transition-colors"
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