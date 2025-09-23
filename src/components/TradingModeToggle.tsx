import { useEffect, useState } from 'react';
import { brokerManager, TradingMode } from '../services/brokerManager';

export function TradingModeToggle() {
  const [mode, setMode] = useState<TradingMode>(brokerManager.getTradingMode());
  const [isSwitching, setIsSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const unsubscribe = brokerManager.subscribe((nextMode) => {
      setMode(nextMode);
      setIsSwitching(false);
      setError(null);
    });

    return unsubscribe;
  }, []);

  const handleToggle = async (targetMode: TradingMode) => {
    if (mode === targetMode || isSwitching) {
      return;
    }

    setIsSwitching(true);
    setError(null);

    try {
      await brokerManager.setTradingMode(targetMode);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to switch trading mode.');
      setIsSwitching(false);
    }
  };

  return (
    <div className="flex flex-col items-end gap-2">
      <div className="inline-flex items-center rounded-full bg-gray-800 p-1 text-sm text-gray-300">
        <button
          type="button"
          onClick={() => handleToggle('demo')}
          className={`rounded-full px-3 py-1 transition-colors ${
            mode === 'demo' ? 'bg-blue-600 text-white' : 'hover:bg-gray-700'
          } ${isSwitching ? 'opacity-75' : ''}`}
          disabled={isSwitching}
        >
          Demo
        </button>
        <button
          type="button"
          onClick={() => handleToggle('live')}
          className={`rounded-full px-3 py-1 transition-colors ${
            mode === 'live' ? 'bg-green-600 text-white' : 'hover:bg-gray-700'
          } ${isSwitching ? 'opacity-75' : ''}`}
          disabled={isSwitching}
        >
          Live
        </button>
      </div>
      {error ? (
        <span className="text-xs text-red-400">{error}</span>
      ) : (
        <span className="text-xs text-gray-400">{brokerManager.formatDisplayLabel()}</span>
      )}
    </div>
  );
}
