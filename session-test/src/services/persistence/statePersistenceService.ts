import { supabase } from '../../lib/supabase';
import { requireUserId } from '../../lib/supabaseAuth';

export interface PersistedState {
  // Agent State
  agentActive?: boolean;
  agentPaused?: boolean;
  agentPauseReason?: string;
  lastAgentCycle?: string;
  
  // Trading Settings
  confidenceThreshold?: number;
  riskPerTrade?: number;
  maxPositionSize?: number;
  tradingEnabled?: boolean;
  preferredBroker?: 'alpaca' | 'binance';
  tradingMode?: 'demo' | 'live';
  
  // UI State
  dashboardLayout?: any[];
  selectedTimeframe?: string;
  activeWidgets?: string[];
  collapsedPanels?: string[];
  
  // Market Data Cache
  lastPrices?: Record<string, number>;
  lastUpdateTime?: string;
  
  // Session Info
  sessionStartTime?: string;
  lastActivityTime?: string;
}

export interface UserSettings {
  // Trading Window
  trading_start_time?: string;
  trading_end_time?: string;
  trading_timezone?: string;
  trading_enabled?: boolean;
  
  // Risk Management
  per_trade_risk_percent?: number;
  max_position_size_percent?: number;
  max_drawdown_percent?: number;
  volatility_tolerance?: 'low' | 'medium' | 'high';
  confidence_threshold?: number;
  risk_reward_minimum?: number;
  
  // Strategy Settings
  shorting_enabled?: boolean;
  margin_enabled?: boolean;
  max_leverage?: number;
  unorthodox_strategies?: boolean;
  
  // Agent Controls
  agent_pauses_remaining?: number;
  agent_pauses_reset_date?: string;
  last_pause_reason?: string;
  last_pause_at?: string;
  
  // Notifications
  email_notifications?: boolean;
  critical_alerts_only?: boolean;
}

class StatePersistenceService {
  private readonly LOCAL_STORAGE_KEY = 'tradingbot_state';
  private readonly SESSION_STORAGE_KEY = 'tradingbot_session';
  private readonly SETTINGS_TABLE = 'user_settings';
  private state: PersistedState = {};
  private saveDebounceTimer: number | null = null;
  private readonly SAVE_DEBOUNCE_MS = 1000;
  private syncInterval: number | null = null;
  private readonly SYNC_INTERVAL_MS = 30000; // 30 seconds

  constructor() {
    this.initialize();
  }

  /**
   * Initialize the service
   */
  private async initialize() {
    // Load state from localStorage
    this.loadLocalState();
    
    // Start periodic sync with database
    this.startPeriodicSync();
    
    // Listen for storage events (cross-tab sync)
    window.addEventListener('storage', this.handleStorageChange.bind(this));
    
    // Save state before unload
    window.addEventListener('beforeunload', this.saveStateImmediately.bind(this));
    
    // Track activity
    this.updateActivity();
  }

  /**
   * Get current persisted state
   */
  getState(): PersistedState {
    return { ...this.state };
  }

  /**
   * Update state (with debounced save)
   */
  setState(updates: Partial<PersistedState>): void {
    this.state = {
      ...this.state,
      ...updates,
      lastActivityTime: new Date().toISOString()
    };
    
    this.saveStateDebounced();
  }

  /**
   * Get a specific state value
   */
  getValue<K extends keyof PersistedState>(key: K): PersistedState[K] {
    return this.state[key];
  }

  /**
   * Set a specific state value
   */
  setValue<K extends keyof PersistedState>(key: K, value: PersistedState[K]): void {
    this.setState({ [key]: value } as Partial<PersistedState>);
  }

  /**
   * Load user settings from database
   */
  async loadUserSettings(): Promise<UserSettings | null> {
    try {
      const userId = await requireUserId();
      const { data, error } = await supabase
        .from(this.SETTINGS_TABLE)
        .select('*')
        .eq('user_id', userId)
        .maybeSingle();

      if (error) {
        if (error.code === 'PGRST116') {
          // No settings exist, create default
          return await this.createDefaultSettings();
        }
        console.error('Error loading user settings:', error);
        return null;
      }

      if (!data) {
        return await this.createDefaultSettings();
      }

      // Cache important settings in local state
      this.setState({
        confidenceThreshold: data.confidence_threshold,
        riskPerTrade: data.per_trade_risk_percent,
        maxPositionSize: data.max_position_size_percent,
        tradingEnabled: data.trading_enabled
      });

      return data;
    } catch (error) {
      if (error instanceof Error && error.message.includes('not authenticated')) {
        return null;
      }
      console.error('Failed to load user settings:', error);
      return null;
    }
  }

  /**
   * Save user settings to database
   */
  async saveUserSettings(settings: Partial<UserSettings>): Promise<boolean> {
    try {
      const userId = await requireUserId();
      const payload = {
        ...settings,
        user_id: userId,
        updated_at: new Date().toISOString(),
      };

      const { error } = await supabase
        .from(this.SETTINGS_TABLE)
        .upsert(payload, {
          onConflict: 'user_id'
        });

      if (error) {
        console.error('Error saving user settings:', error);
        return false;
      }

      // Update local cache
      if (settings.confidence_threshold !== undefined) {
        this.setValue('confidenceThreshold', settings.confidence_threshold);
      }
      if (settings.per_trade_risk_percent !== undefined) {
        this.setValue('riskPerTrade', settings.per_trade_risk_percent);
      }
      if (settings.max_position_size_percent !== undefined) {
        this.setValue('maxPositionSize', settings.max_position_size_percent);
      }
      if (settings.trading_enabled !== undefined) {
        this.setValue('tradingEnabled', settings.trading_enabled);
      }

      return true;
    } catch (error) {
      console.error('Failed to save user settings:', error);
      return false;
    }
  }

  /**
   * Clear all persisted state
   */
  clearState(): void {
    this.state = {};
    localStorage.removeItem(this.LOCAL_STORAGE_KEY);
    sessionStorage.removeItem(this.SESSION_STORAGE_KEY);
  }

  /**
   * Export state for debugging
   */
  exportState(): string {
    return JSON.stringify({
      localStorage: this.state,
      sessionStorage: this.getSessionState(),
      timestamp: new Date().toISOString()
    }, null, 2);
  }

  /**
   * Import state (for testing/migration)
   */
  importState(stateJson: string): void {
    try {
      const imported = JSON.parse(stateJson);
      if (imported.localStorage) {
        this.state = imported.localStorage;
        this.saveStateImmediately();
      }
    } catch (error) {
      console.error('Failed to import state:', error);
    }
  }

  // Session-specific state (doesn't persist across browser close)
  private getSessionState(): any {
    try {
      const stored = sessionStorage.getItem(this.SESSION_STORAGE_KEY);
      return stored ? JSON.parse(stored) : {};
    } catch {
      return {};
    }
  }

  private setSessionState(state: any): void {
    try {
      sessionStorage.setItem(this.SESSION_STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
      console.error('Failed to save session state:', error);
    }
  }

  // Private helper methods
  private loadLocalState(): void {
    try {
      const stored = localStorage.getItem(this.LOCAL_STORAGE_KEY);
      if (stored) {
        this.state = JSON.parse(stored);
      }
    } catch (error) {
      console.error('Failed to load local state:', error);
      this.state = {};
    }
  }

  private saveStateDebounced(): void {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
    }
    
    this.saveDebounceTimer = window.setTimeout(() => {
      this.saveStateImmediately();
    }, this.SAVE_DEBOUNCE_MS);
  }

  private saveStateImmediately(): void {
    try {
      localStorage.setItem(this.LOCAL_STORAGE_KEY, JSON.stringify(this.state));
    } catch (error) {
      console.error('Failed to save state:', error);
    }
  }

  private handleStorageChange(event: StorageEvent): void {
    if (event.key === this.LOCAL_STORAGE_KEY && event.newValue) {
      try {
        const newState = JSON.parse(event.newValue);
        this.state = newState;
        
        // Emit event for UI updates
        window.dispatchEvent(new CustomEvent('state-sync', {
          detail: newState
        }));
      } catch (error) {
        console.error('Failed to sync state from storage:', error);
      }
    }
  }

  private async createDefaultSettings(): Promise<UserSettings> {
    const defaultSettings: UserSettings = {
      trading_start_time: '09:00:00',
      trading_end_time: '18:00:00',
      trading_timezone: 'America/New_York',
      trading_enabled: true,
      per_trade_risk_percent: 1.0,
      max_position_size_percent: 10.0,
      max_drawdown_percent: 15.0,
      volatility_tolerance: 'medium',
      confidence_threshold: 75.0,
      risk_reward_minimum: 3.0,
      shorting_enabled: false,
      margin_enabled: false,
      max_leverage: 1.0,
      unorthodox_strategies: false,
      agent_pauses_remaining: 2,
      email_notifications: true,
      critical_alerts_only: false
    };

    await this.saveUserSettings(defaultSettings);
    return defaultSettings;
  }

  private startPeriodicSync(): void {
    this.syncInterval = window.setInterval(async () => {
      // Sync critical settings with database
      await this.loadUserSettings();
    }, this.SYNC_INTERVAL_MS);
  }

  private updateActivity(): void {
    this.setState({
      lastActivityTime: new Date().toISOString()
    });
    
    // Update session info
    const sessionState = this.getSessionState();
    if (!sessionState.sessionStartTime) {
      this.setSessionState({
        ...sessionState,
        sessionStartTime: new Date().toISOString()
      });
    }
  }

  /**
   * Clean up
   */
  destroy(): void {
    if (this.saveDebounceTimer) {
      clearTimeout(this.saveDebounceTimer);
    }
    if (this.syncInterval) {
      clearInterval(this.syncInterval);
    }
    this.saveStateImmediately();
  }
}

// Export singleton instance
export const statePersistenceService = new StatePersistenceService();

// Custom hooks for React components
export const usePersistentState = <K extends keyof PersistedState>(
  key: K,
  defaultValue: PersistedState[K]
): [PersistedState[K], (value: PersistedState[K]) => void] => {
  const value = statePersistenceService.getValue(key) ?? defaultValue;
  
  const setValue = (newValue: PersistedState[K]) => {
    statePersistenceService.setValue(key, newValue);
  };
  
  return [value, setValue];
};
