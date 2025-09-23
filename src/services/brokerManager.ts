import { tradingProviderService, TradingBrokerId } from './tradingProviderService';
import { TradingBroker } from './brokers/types';
import { apiKeysService } from './apiKeysService';
import { statePersistenceService } from './persistence/statePersistenceService';

export type TradingMode = 'demo' | 'live';

const MODE_TO_BROKER: Record<TradingMode, TradingBrokerId> = {
  demo: 'alpaca',
  live: 'binance'
};

const BROKER_TO_MODE: Record<TradingBrokerId, TradingMode> = {
  alpaca: 'demo',
  binance: 'live'
};

class BrokerManager {
  private subscribers = new Set<(mode: TradingMode) => void>();
  private unsubscribeProviderChange: (() => void) | null = null;

  constructor() {
    const persistedMode = statePersistenceService.getValue('tradingMode');
    const persistedBroker = statePersistenceService.getValue('preferredBroker');
    const envDefault = (import.meta.env.VITE_DEFAULT_BROKER || '').toLowerCase() === 'binance'
      ? 'binance'
      : 'alpaca';

    // Always default to Live (Binance) on first load unless explicitly overridden this session
    let initialBroker: TradingBrokerId = 'binance';

    if (persistedMode && MODE_TO_BROKER[persistedMode]) {
      initialBroker = MODE_TO_BROKER[persistedMode];
    } else if (persistedBroker && BROKER_TO_MODE[persistedBroker]) {
      initialBroker = persistedBroker;
    } else if (envDefault === 'alpaca') {
      initialBroker = 'alpaca';
    }

    try {
      tradingProviderService.setActiveProvider(initialBroker);
    } catch (error) {
      console.warn('Failed to activate persisted broker, falling back to Alpaca:', error);
      initialBroker = 'alpaca';
      tradingProviderService.setActiveProvider(initialBroker);
    }

    this.persistSelection(initialBroker);

    this.unsubscribeProviderChange = tradingProviderService.onProviderChange((provider) => {
      if (!BROKER_TO_MODE[provider]) {
        return;
      }
      this.persistSelection(provider);
      this.notifySubscribers();
    });
  }

  getBroker(): TradingBroker {
    return tradingProviderService.getBroker();
  }

  getActiveBrokerId(): TradingBrokerId {
    return tradingProviderService.getActiveProviderId();
  }

  getTradingMode(): TradingMode {
    return BROKER_TO_MODE[this.getActiveBrokerId()];
  }

  async setTradingMode(mode: TradingMode): Promise<void> {
    const targetBroker = MODE_TO_BROKER[mode];
    if (mode === 'live' && !(await this.isLiveAvailable())) {
      throw new Error('Binance API keys not configured');
    }
    await this.setBroker(targetBroker);
  }

  async setBroker(broker: TradingBrokerId): Promise<void> {
    if (this.getActiveBrokerId() === broker) {
      return;
    }
    tradingProviderService.setActiveProvider(broker);
    this.persistSelection(broker);
    this.notifySubscribers();
  }

  subscribe(listener: (mode: TradingMode) => void): () => void {
    this.subscribers.add(listener);
    return () => {
      this.subscribers.delete(listener);
    };
  }

  async isLiveAvailable(): Promise<boolean> {
    const [apiKey, secretKey] = await Promise.all([
      apiKeysService.getApiKeyWithFallback('binance', 'api_key'),
      apiKeysService.getApiKeyWithFallback('binance', 'secret_key')
    ]);
    return Boolean(apiKey && secretKey);
  }

  formatOrderSymbol(symbol: string): string {
    return this.getBroker().normalizeSymbol(symbol);
  }

  formatDisplayLabel(): string {
    return this.getTradingMode() === 'live' ? 'Live • Binance' : 'Demo • Alpaca';
  }

  destroy(): void {
    if (this.unsubscribeProviderChange) {
      this.unsubscribeProviderChange();
      this.unsubscribeProviderChange = null;
    }
    this.subscribers.clear();
  }

  private persistSelection(broker: TradingBrokerId): void {
    statePersistenceService.setValue('preferredBroker', broker);
    statePersistenceService.setValue('tradingMode', BROKER_TO_MODE[broker]);
  }

  private notifySubscribers(): void {
    const mode = this.getTradingMode();
    for (const listener of this.subscribers) {
      try {
        listener(mode);
      } catch (error) {
        console.error('Broker manager subscriber error:', error);
      }
    }
  }
}

export const brokerManager = new BrokerManager();
