import { supabase } from '../lib/supabase';

export interface AgentSettings {
  riskBudgetUsd: number; // per trade budget in USD
  confidenceThreshold: number; // 0..1
  cooldownMinutes: number; // min time between trades per symbol
  maxOpenPositions: number; // cap on total open positions
}

const DEFAULT_SETTINGS: AgentSettings = {
  riskBudgetUsd: 100, // conservative default
  confidenceThreshold: 0.78,
  cooldownMinutes: 5,
  maxOpenPositions: 10,
};

const keyFor = (userId: string) => `agent_settings_${userId}`;

class AgentSettingsService {
  async getSettings(): Promise<AgentSettings> {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return DEFAULT_SETTINGS;

    // Try Supabase first
    try {
      const { data, error } = await supabase
        .from('agent_settings')
        .select('risk_budget_usd, confidence_threshold, cooldown_minutes, max_open_positions')
        .eq('user_id', user.id)
        .maybeSingle();

      if (error) throw error;

      if (data) {
        return {
          riskBudgetUsd: data.risk_budget_usd ?? DEFAULT_SETTINGS.riskBudgetUsd,
          confidenceThreshold: data.confidence_threshold ?? DEFAULT_SETTINGS.confidenceThreshold,
          cooldownMinutes: data.cooldown_minutes ?? DEFAULT_SETTINGS.cooldownMinutes,
          maxOpenPositions: data.max_open_positions ?? DEFAULT_SETTINGS.maxOpenPositions,
        };
      }
    } catch (err) {
      // Fall back to localStorage
      const cached = localStorage.getItem(keyFor(user.id));
      if (cached) return JSON.parse(cached) as AgentSettings;
    }

    return DEFAULT_SETTINGS;
  }

  async saveSettings(settings: AgentSettings): Promise<void> {
    const { data: { user } } = await supabase.auth.getUser();
    if (!user) return;

    // Persist to Supabase; if table missing or error, store in localStorage
    try {
      const { error } = await supabase
        .from('agent_settings')
        .upsert({
          user_id: user.id,
          risk_budget_usd: settings.riskBudgetUsd,
          confidence_threshold: settings.confidenceThreshold,
          cooldown_minutes: settings.cooldownMinutes,
          max_open_positions: settings.maxOpenPositions,
          updated_at: new Date().toISOString(),
        }, { onConflict: 'user_id' });

      if (error) throw error;
    } catch (err) {
      // Store locally as fallback
      localStorage.setItem(keyFor(user.id), JSON.stringify(settings));
    }
  }
}

export const agentSettingsService = new AgentSettingsService();
export { DEFAULT_SETTINGS };


