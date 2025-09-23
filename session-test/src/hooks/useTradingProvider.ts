import { useEffect, useState } from 'react';
import { tradingProviderService, TradingBrokerId } from '../services/tradingProviderService';
import { brokerManager, TradingMode } from '../services/brokerManager';

interface UseTradingProviderResult {
  activeProvider: TradingBrokerId;
  providers: ReturnType<typeof tradingProviderService.getProviderMetadata>;
  setActiveProvider: (provider: TradingBrokerId) => void;
  tradingMode: TradingMode;
}

export function useTradingProvider(): UseTradingProviderResult {
  const [activeProvider, setActiveProvider] = useState<TradingBrokerId>(
    brokerManager.getActiveBrokerId()
  );
  const [tradingMode, setTradingMode] = useState<TradingMode>(brokerManager.getTradingMode());
  const [providers] = useState(tradingProviderService.getProviderMetadata());

  useEffect(() => {
    const unsubscribe = brokerManager.subscribe((mode) => {
      setTradingMode(mode);
      setActiveProvider(mode === 'live' ? 'binance' : 'alpaca');
    });

    return unsubscribe;
  }, []);

  const handleSetProvider = (provider: TradingBrokerId) => {
    const mode: TradingMode = provider === 'binance' ? 'live' : 'demo';
    brokerManager.setTradingMode(mode).catch((error) => {
      console.error('Failed to switch trading mode:', error);
    });
  };

  return {
    activeProvider,
    providers,
    setActiveProvider: handleSetProvider,
    tradingMode,
  };
}
