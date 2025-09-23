import React, { useState, useEffect } from 'react';
import { Settings, BarChart3, TrendingUp, Activity, Shield, Brain, Target } from 'lucide-react';
import { agentSettingsService } from '../../services/agentSettingsService';
import { tradingAgentV2 } from '../../services/tradingAgentV2';

interface StrategyConfig {
  enabled: boolean;
  weight: number;
  lastSignal?: string;
  confidence?: number;
  status: 'active' | 'paused' | 'error';
}

interface ValidatorConfig {
  enabled: boolean;
  weight: number;
  threshold?: number;
}

interface StrategyControlSettings {
  strategies: {
    liquidity: StrategyConfig;
    smartMoney: StrategyConfig;
    volumeProfile: StrategyConfig;
    microstructure: StrategyConfig;
  };
  validators: {
    trend: ValidatorConfig;
    volume: ValidatorConfig;
    volatility: ValidatorConfig;
    riskReward: ValidatorConfig;
    sentiment: ValidatorConfig;
    positionSize: ValidatorConfig;
  };
  strategyWeight: number; // 0-100% strategy vs validator balance
}

const DEFAULT_SETTINGS: StrategyControlSettings = {
  strategies: {
    liquidity: { enabled: true, weight: 25, status: 'active' },
    smartMoney: { enabled: true, weight: 25, status: 'active' },
    volumeProfile: { enabled: true, weight: 25, status: 'active' },
    microstructure: { enabled: true, weight: 25, status: 'active' }
  },
  validators: {
    trend: { enabled: true, weight: 20, threshold: 0.7 },
    volume: { enabled: true, weight: 15, threshold: 0.6 },
    volatility: { enabled: true, weight: 15, threshold: 0.65 },
    riskReward: { enabled: true, weight: 20, threshold: 0.75 },
    sentiment: { enabled: true, weight: 15, threshold: 0.6 },
    positionSize: { enabled: true, weight: 15, threshold: 0.8 }
  },
  strategyWeight: 30 // 30% strategies, 70% validators
};

const StrategyControlPanel: React.FC = () => {
  const [settings, setSettings] = useState<StrategyControlSettings>(DEFAULT_SETTINGS);
  const [isLoading, setIsLoading] = useState(true);
  const [isSaving, setIsSaving] = useState(false);
  const [lastSaved, setLastSaved] = useState<Date | null>(null);
  const [strategySignals, setStrategySignals] = useState<Map<string, any>>(new Map());

  useEffect(() => {
    loadSettings();
    // Subscribe to strategy signals
    const unsubscribe = tradingAgentV2.subscribe((event) => {
      if (event.type === 'analysis') {
        setStrategySignals(tradingAgentV2.getStrategySignals());
      }
    });

    return unsubscribe;
  }, []);

  const loadSettings = async () => {
    try {
      const storedSettings = localStorage.getItem('strategy_control_settings');
      if (storedSettings) {
        const parsed = JSON.parse(storedSettings);
        setSettings({ ...DEFAULT_SETTINGS, ...parsed });
      }
      setIsLoading(false);
    } catch (error) {
      console.error('Error loading strategy settings:', error);
      setIsLoading(false);
    }
  };

  const saveSettings = async () => {
    setIsSaving(true);
    try {
      localStorage.setItem('strategy_control_settings', JSON.stringify(settings));
      
      // Sync with agentSettingsService for integration
      await agentSettingsService.saveSettings({
        riskBudgetUsd: 100,
        confidenceThreshold: 0.78,
        cooldownMinutes: 5,
        maxOpenPositions: 10
      });

      setLastSaved(new Date());
    } catch (error) {
      console.error('Error saving settings:', error);
    }
    setIsSaving(false);
  };

  const updateStrategyEnabled = (strategy: keyof typeof settings.strategies, enabled: boolean) => {
    setSettings(prev => ({
      ...prev,
      strategies: {
        ...prev.strategies,
        [strategy]: { ...prev.strategies[strategy], enabled }
      }
    }));
  };

  const updateStrategyWeight = (strategy: keyof typeof settings.strategies, weight: number) => {
    setSettings(prev => ({
      ...prev,
      strategies: {
        ...prev.strategies,
        [strategy]: { ...prev.strategies[strategy], weight }
      }
    }));
  };

  const updateValidatorEnabled = (validator: keyof typeof settings.validators, enabled: boolean) => {
    setSettings(prev => ({
      ...prev,
      validators: {
        ...prev.validators,
        [validator]: { ...prev.validators[validator], enabled }
      }
    }));
  };

  const updateValidatorWeight = (validator: keyof typeof settings.validators, weight: number) => {
    setSettings(prev => ({
      ...prev,
      validators: {
        ...prev.validators,
        [validator]: { ...prev.validators[validator], weight }
      }
    }));
  };

  const getStrategyStatus = (strategyName: string) => {
    const signals = Array.from(strategySignals.values());
    if (signals.length === 0) return 'paused';
    
    const hasActiveSignal = signals.some(signal => 
      signal[strategyName] && signal[strategyName].confidence > 50
    );
    
    return hasActiveSignal ? 'active' : 'paused';
  };

  const getLastSignalInfo = (strategyName: string) => {
    const signals = Array.from(strategySignals.values());
    if (signals.length === 0) return { action: 'None', confidence: 0 };
    
    for (const signal of signals) {
      if (signal[strategyName]) {
        return {
          action: signal[strategyName].action || 'None',
          confidence: signal[strategyName].confidence || 0
        };
      }
    }
    
    return { action: 'None', confidence: 0 };
  };

  const strategyIcons = {
    liquidity: Activity,
    smartMoney: Brain,
    volumeProfile: BarChart3,
    microstructure: TrendingUp
  };

  const validatorIcons = {
    trend: TrendingUp,
    volume: BarChart3,
    volatility: Activity,
    riskReward: Target,
    sentiment: Brain,
    positionSize: Shield
  };

  if (isLoading) {
    return (
      <div className="bg-gray-900 rounded-lg p-6">
        <div className="animate-pulse flex space-x-4">
          <div className="rounded-full bg-gray-700 h-10 w-10"></div>
          <div className="flex-1 space-y-6 py-1">
            <div className="h-2 bg-gray-700 rounded"></div>
            <div className="space-y-3">
              <div className="grid grid-cols-3 gap-4">
                <div className="h-2 bg-gray-700 rounded col-span-2"></div>
                <div className="h-2 bg-gray-700 rounded col-span-1"></div>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-gray-900 rounded-lg border border-gray-700 p-6 space-y-6 h-full overflow-y-auto">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-3">
          <Settings className="w-6 h-6 text-blue-400" />
          <h3 className="text-xl font-semibold text-white">Strategy Control Panel</h3>
        </div>
        <div className="flex items-center space-x-3">
          {lastSaved && (
            <span className="text-sm text-gray-400">
              Last saved: {lastSaved.toLocaleTimeString()}
            </span>
          )}
          <button
            onClick={saveSettings}
            disabled={isSaving}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-blue-800 px-4 py-2 rounded-lg text-white text-sm font-medium transition-colors"
          >
            {isSaving ? 'Saving...' : 'Save Settings'}
          </button>
        </div>
      </div>

      {/* Strategy vs Validator Balance */}
      <div className="bg-gray-800 rounded-lg p-4">
        <div className="flex items-center justify-between mb-4">
          <h4 className="text-lg font-medium text-white">Strategy vs Validator Balance</h4>
          <span className="text-sm text-gray-400">
            {settings.strategyWeight}% Strategies / {100 - settings.strategyWeight}% Validators
          </span>
        </div>
        <input
          type="range"
          min="0"
          max="100"
          value={settings.strategyWeight}
          onChange={(e) => setSettings(prev => ({ ...prev, strategyWeight: parseInt(e.target.value) }))}
          className="w-full h-2 bg-gray-700 rounded-lg appearance-none cursor-pointer slider"
        />
        <div className="flex justify-between text-xs text-gray-400 mt-1">
          <span>100% Validators</span>
          <span>Balanced</span>
          <span>100% Strategies</span>
        </div>
      </div>

      {/* Strategies Section */}
      <div className="space-y-4">
        <h4 className="text-lg font-medium text-white flex items-center">
          <Brain className="w-5 h-5 mr-2 text-purple-400" />
          Trading Strategies
        </h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Object.entries(settings.strategies).map(([key, config]) => {
            const Icon = strategyIcons[key as keyof typeof strategyIcons];
            const status = getStrategyStatus(key);
            const lastSignal = getLastSignalInfo(key);
            
            return (
              <div key={key} className="bg-gray-800 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Icon className="w-5 h-5 text-purple-400" />
                    <span className="font-medium text-white capitalize">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </span>
                  </div>
                  <div className="flex items-center space-x-2">
                    <span className={`text-xs px-2 py-1 rounded-full ${
                      status === 'active' ? 'bg-green-900 text-green-300' :
                      status === 'paused' ? 'bg-yellow-900 text-yellow-300' :
                      'bg-red-900 text-red-300'
                    }`}>
                      {status}
                    </span>
                    <label className="relative inline-flex items-center cursor-pointer">
                      <input
                        type="checkbox"
                        className="sr-only peer"
                        checked={config.enabled}
                        onChange={(e) => updateStrategyEnabled(key as any, e.target.checked)}
                      />
                      <div className="w-11 h-6 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
                    </label>
                  </div>
                </div>
                
                {config.enabled && (
                  <>
                    <div className="space-y-2">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-400">Weight:</span>
                        <span className="text-white">{config.weight}%</span>
                      </div>
                      <input
                        type="range"
                        min="0"
                        max="100"
                        value={config.weight}
                        onChange={(e) => updateStrategyWeight(key as any, parseInt(e.target.value))}
                        className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                      />
                    </div>
                    
                    <div className="grid grid-cols-2 gap-2 text-xs">
                      <div>
                        <span className="text-gray-400">Last Signal:</span>
                        <div className="text-white font-medium">{lastSignal.action}</div>
                      </div>
                      <div>
                        <span className="text-gray-400">Confidence:</span>
                        <div className="text-white font-medium">{lastSignal.confidence.toFixed(1)}%</div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Validators Section */}
      <div className="space-y-4">
        <h4 className="text-lg font-medium text-white flex items-center">
          <Shield className="w-5 h-5 mr-2 text-green-400" />
          Validation Systems
        </h4>
        
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {Object.entries(settings.validators).map(([key, config]) => {
            const Icon = validatorIcons[key as keyof typeof validatorIcons];
            
            return (
              <div key={key} className="bg-gray-800 rounded-lg p-4 space-y-3">
                <div className="flex items-center justify-between">
                  <div className="flex items-center space-x-2">
                    <Icon className="w-4 h-4 text-green-400" />
                    <span className="font-medium text-white capitalize text-sm">
                      {key.replace(/([A-Z])/g, ' $1').trim()}
                    </span>
                  </div>
                  <label className="relative inline-flex items-center cursor-pointer">
                    <input
                      type="checkbox"
                      className="sr-only peer"
                      checked={config.enabled}
                      onChange={(e) => updateValidatorEnabled(key as any, e.target.checked)}
                    />
                    <div className="w-9 h-5 bg-gray-600 peer-focus:outline-none rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-0.5 after:left-[2px] after:bg-white after:rounded-full after:h-4 after:w-4 after:transition-all peer-checked:bg-green-600"></div>
                  </label>
                </div>
                
                {config.enabled && (
                  <div className="space-y-2">
                    <div className="flex justify-between text-xs">
                      <span className="text-gray-400">Weight:</span>
                      <span className="text-white">{config.weight}%</span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="50"
                      value={config.weight}
                      onChange={(e) => updateValidatorWeight(key as any, parseInt(e.target.value))}
                      className="w-full h-1 bg-gray-700 rounded-lg appearance-none cursor-pointer"
                    />
                    {config.threshold && (
                      <div className="flex justify-between text-xs">
                        <span className="text-gray-400">Threshold:</span>
                        <span className="text-white">{(config.threshold * 100).toFixed(0)}%</span>
                      </div>
                    )}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      <style jsx>{`
        .slider::-webkit-slider-thumb {
          appearance: none;
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
          border: 2px solid #1F2937;
        }

        .slider::-moz-range-thumb {
          height: 20px;
          width: 20px;
          border-radius: 50%;
          background: #3B82F6;
          cursor: pointer;
          border: 2px solid #1F2937;
        }

        /* Custom scrollbar styling */
        .overflow-y-auto::-webkit-scrollbar {
          width: 6px;
        }

        .overflow-y-auto::-webkit-scrollbar-track {
          background: #1F2937;
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb {
          background: #4B5563;
          border-radius: 3px;
        }

        .overflow-y-auto::-webkit-scrollbar-thumb:hover {
          background: #6B7280;
        }
      `}</style>
    </div>
  );
};

export { StrategyControlPanel };
export default StrategyControlPanel;