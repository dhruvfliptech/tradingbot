import { Account, CryptoData, Order, Position } from '../types/trading';
import { alpacaBroker } from './brokers/alpacaBroker';
import { binanceBroker } from './brokers/binanceBroker';
import { PlaceOrderParams, TradingBroker, TradingBrokerId } from './brokers/types';

export type { PlaceOrderParams, TradingBrokerId } from './brokers/types';

const STORAGE_KEY = 'trading.active_provider';

class TradingProviderService {
  private providers: Record<TradingBrokerId, TradingBroker> = {
    alpaca: alpacaBroker,
    binance: binanceBroker,
  };

  private activeProvider: TradingBrokerId;
  private listeners = new Set<(provider: TradingBrokerId) => void>();

  constructor() {
    this.activeProvider = this.loadInitialProvider();
  }

  private loadInitialProvider(): TradingBrokerId {
    try {
      if (typeof window !== 'undefined') {
        const stored = window.localStorage.getItem(STORAGE_KEY) as TradingBrokerId | null;
        if (stored && stored in this.providers) {
          return stored;
        }
      }
    } catch (error) {
      console.warn('Unable to read trading provider from storage:', error);
    }
    return 'alpaca';
  }

  private persistProvider(provider: TradingBrokerId): void {
    try {
      if (typeof window !== 'undefined') {
        window.localStorage.setItem(STORAGE_KEY, provider);
      }
    } catch (error) {
      console.warn('Unable to persist trading provider selection:', error);
    }
  }

  private notify(): void {
    for (const listener of this.listeners) {
      try {
        listener(this.activeProvider);
      } catch (error) {
        console.warn('Trading provider listener error:', error);
      }
    }
  }

  onProviderChange(listener: (provider: TradingBrokerId) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  listProviders(): TradingBroker[] {
    return Object.values(this.providers);
  }

  getProviderMetadata(): Array<{ id: TradingBrokerId; label: string; description: string; features: TradingBroker['metadata']['features']; baseCurrency: string; docsUrl?: string }> {
    return this.listProviders().map((broker) => broker.metadata);
  }

  getActiveProviderId(): TradingBrokerId {
    return this.activeProvider;
  }

  setActiveProvider(provider: TradingBrokerId): void {
    if (!this.providers[provider]) {
      throw new Error(`Unknown trading provider: ${provider}`);
    }
    if (this.activeProvider === provider) return;
    this.activeProvider = provider;
    this.persistProvider(provider);
    this.notify();
  }

  getBroker(provider?: TradingBrokerId): TradingBroker {
    const id = provider || this.activeProvider;
    const broker = this.providers[id];
    if (!broker) {
      throw new Error(`Trading broker not found: ${id}`);
    }
    return broker;
  }

  normalizeSymbol(symbol: string, provider?: TradingBrokerId): string {
    return this.getBroker(provider).normalizeSymbol(symbol);
  }

  toBrokerSymbol(symbol: string, provider?: TradingBrokerId): string {
    return this.getBroker(provider).toBrokerSymbol(symbol);
  }

  fromBrokerSymbol(symbol: string, provider?: TradingBrokerId): string {
    return this.getBroker(provider).fromBrokerSymbol(symbol);
  }

  async testConnection(provider?: TradingBrokerId): Promise<boolean> {
    const broker = this.getBroker(provider);
    if (typeof broker.testConnection === 'function') {
      return broker.testConnection();
    }
    try {
      await broker.getAccount();
      return true;
    } catch (error) {
      console.warn(`Connection test for ${broker.metadata.label} failed:`, error);
      return false;
    }
  }

  async getAccount(provider?: TradingBrokerId): Promise<Account> {
    return this.getBroker(provider).getAccount();
  }

  async getPositions(provider?: TradingBrokerId): Promise<Position[]> {
    return this.getBroker(provider).getPositions();
  }

  async getOrders(provider?: TradingBrokerId): Promise<Order[]> {
    return this.getBroker(provider).getOrders();
  }

  async placeOrder(order: PlaceOrderParams, provider?: TradingBrokerId): Promise<Order> {
    return this.getBroker(provider).placeOrder(order);
  }

  async getCryptoData(symbols?: string[], provider?: TradingBrokerId): Promise<CryptoData[]> {
    const broker = this.getBroker(provider);
    if (typeof broker.getCryptoData === 'function') {
      return broker.getCryptoData(symbols);
    }
    return [];
  }
}

export const tradingProviderService = new TradingProviderService();
