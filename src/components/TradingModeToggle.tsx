import { useEffect, useState } from 'react';
import { brokerManager, TradingMode } from '../services/brokerManager';
import { tradingAgentV2 } from '../services/tradingAgentV2';
import { Play, Pause, Square } from 'lucide-react';

export function TradingModeToggle() {
  const [mode, setMode] = useState<TradingMode>(brokerManager.getTradingMode());
  const [isSwitching, setIsSwitching] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [isAgentActive, setIsAgentActive] = useState(false);

  useEffect(() => {
    const unsubscribe = brokerManager.subscribe((nextMode) => {
      setMode(nextMode);
      setIsSwitching(false);
      setError(null);
    });

    // Subscribe to trading agent status
    const agentUnsubscribe = tradingAgentV2.subscribe((event) => {
      if (event.type === 'status') {
        setIsAgentActive(event.active);
      }
    });

    return () => {
      unsubscribe();
      agentUnsubscribe();
    };
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

  const handleAgentControl = async (action: 'start' | 'stop') => {
    try {
      if (action === 'start') {
        await tradingAgentV2.start();
      } else {
        await tradingAgentV2.stop();
      }
    } catch (err) {
      console.error(`Failed to ${action} agent:`, err);
      setError(`Failed to ${action} agent`);
    }
  };

  return (
    <div className="flex items-center gap-3">
      {/* Trading Mode Toggle */}
      <div className="inline-flex items-center rounded-full bg-gray-800 p-1 text-sm text-gray-300">
        <button
          type="button"
          onClick={() => handleToggle('demo')}
          className={`rounded-full px-4 py-2 transition-colors ${
            mode === 'demo' ? 'bg-blue-600 text-white' : 'hover:bg-gray-700'
          } ${isSwitching ? 'opacity-75' : ''}`}
          disabled={isSwitching}
        >
          Demo
        </button>
        <button
          type="button"
          onClick={() => handleToggle('live')}
          className={`rounded-full px-4 py-2 transition-colors ${
            mode === 'live' ? 'bg-green-600 text-white' : 'hover:bg-gray-700'
          } ${isSwitching ? 'opacity-75' : ''}`}
          disabled={isSwitching}
        >
          Live
        </button>
      </div>

      {/* Auto-Trading Agent Control */}
      <div className="inline-flex items-center rounded-full bg-gray-800 p-1 text-sm text-gray-300">
        <button
          type="button"
          onClick={() => handleAgentControl('start')}
          className={`rounded-full px-3 py-2 transition-colors ${
            isAgentActive ? 'bg-green-600 text-white' : 'hover:bg-gray-700'
          }`}
          disabled={isAgentActive}
          title="Start Auto-Trading"
        >
          <Play className="h-3 w-3 mr-1 inline" />
          Auto
        </button>
        <button
          type="button"
          onClick={() => handleAgentControl('stop')}
          className={`rounded-full px-3 py-2 transition-colors ${
            !isAgentActive ? 'bg-red-600 text-white' : 'hover:bg-gray-700'
          }`}
          disabled={!isAgentActive}
          title="Stop Auto-Trading"
        >
          <Square className="h-3 w-3 mr-1 inline" />
          Stop
        </button>
      </div>

      {/* Error/Status Message */}
      {error && (
        <span className="text-xs text-red-400 max-w-[200px] truncate">{error}</span>
      )}
    </div>
  );
}
