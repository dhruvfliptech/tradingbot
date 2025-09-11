import React, { useEffect, useState } from 'react';
import { Save, Sliders } from 'lucide-react';
import { AgentSettings, agentSettingsService, DEFAULT_SETTINGS } from '../../services/agentSettingsService';

export const AutoTradeSettings: React.FC = () => {
  const [settings, setSettings] = useState<AgentSettings>(DEFAULT_SETTINGS);
  const [saving, setSaving] = useState(false);
  const [savedAt, setSavedAt] = useState<Date | null>(null);

  useEffect(() => {
    (async () => {
      const s = await agentSettingsService.getSettings();
      setSettings(s);
    })();
  }, []);

  const save = async () => {
    setSaving(true);
    await agentSettingsService.saveSettings(settings);
    setSavedAt(new Date());
    setSaving(false);
  };

  return (
    <div className="h-full overflow-y-auto bg-gray-800 rounded-lg p-6 pr-2">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-white flex items-center"><Sliders className="h-5 w-5 mr-2"/>Auto-Trade Settings</h2>
        <button onClick={save} disabled={saving} className="flex items-center px-3 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white rounded-lg">
          <Save className="h-4 w-4 mr-2" />
          {saving ? 'Saving…' : 'Save'}
        </button>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        <div>
          <label className="block text-gray-400 text-xs sm:text-sm mb-2">Per-Trade Risk Budget (USD)</label>
          <input
            type="number"
            min={1}
            value={settings.riskBudgetUsd}
            onChange={(e) => setSettings({ ...settings, riskBudgetUsd: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <p className="text-xs text-gray-400 mt-1">Max USD exposure per AI trade.</p>
        </div>

        <div>
          <label className="block text-gray-400 text-xs sm:text-sm mb-2">Confidence Threshold</label>
          <input
            type="number"
            min={0}
            max={1}
            step={0.01}
            value={settings.confidenceThreshold}
            onChange={(e) => setSettings({ ...settings, confidenceThreshold: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <p className="text-xs text-gray-400 mt-1">Minimum model confidence to execute a trade.</p>
        </div>

        <div>
          <label className="block text-gray-400 text-xs sm:text-sm mb-2">Cooldown (minutes)</label>
          <input
            type="number"
            min={1}
            value={settings.cooldownMinutes}
            onChange={(e) => setSettings({ ...settings, cooldownMinutes: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <p className="text-xs text-gray-400 mt-1">Minimum time between trades for the same symbol.</p>
        </div>

        <div>
          <label className="block text-gray-400 text-xs sm:text-sm mb-2">Max Open Positions</label>
          <input
            type="number"
            min={1}
            value={settings.maxOpenPositions}
            onChange={(e) => setSettings({ ...settings, maxOpenPositions: Number(e.target.value) })}
            className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
          />
          <p className="text-xs text-gray-400 mt-1">Prevents over-allocation; trades will pause if cap reached.</p>
        </div>
      </div>

      {savedAt && (
        <div className="text-xs text-gray-400 mt-3">Saved {savedAt.toLocaleTimeString()}</div>
      )}
    </div>
  );
};

export default AutoTradeSettings;


