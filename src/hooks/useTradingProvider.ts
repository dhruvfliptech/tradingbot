import { useEffect, useState } from 'react';
import { tradingProviderService, TradingBrokerId } from '../services/tradingProviderService';

interface UseTradingProviderResult {
  activeProvider: TradingBrokerId;
  providers: ReturnType<typeof tradingProviderService.getProviderMetadata>;
  setActiveProvider: (provider: TradingBrokerId) => void;
}

export function useTradingProvider(): UseTradingProviderResult {
  const [activeProvider, setActiveProvider] = useState<TradingBrokerId>(
    tradingProviderService.getActiveProviderId()
  );
  const [providers] = useState(tradingProviderService.getProviderMetadata());

  useEffect(() => {
    const unsubscribe = tradingProviderService.onProviderChange((provider) => {
      setActiveProvider(provider);
    });
    return unsubscribe;
  }, []);

  const handleSetProvider = (provider: TradingBrokerId) => {
    tradingProviderService.setActiveProvider(provider);
  };

  return {
    activeProvider,
    providers,
    setActiveProvider: handleSetProvider,
  };
}
