export interface BrokerOrderRequest {
  symbol: string;
  quantity: number;
  side: 'buy' | 'sell';
  orderType: 'market' | 'limit';
  limitPrice?: number;
  timeInForce?: 'day' | 'gtc' | 'ioc';
}

export interface BrokerAdapter {
  getAccount(): Promise<any>;
  getPositions(): Promise<any[]>;
  getOrders(): Promise<any[]>;
  placeOrder(order: BrokerOrderRequest): Promise<any>;
  cancelOrder?(symbol: string, orderId: string): Promise<void>;
}

export interface PersistenceAdapter {
  loadState<T = any>(key: string): Promise<T | null>;
  saveState<T = any>(key: string, value: T): Promise<void>;
  appendAuditLog(entry: Record<string, any>): Promise<void>;
  recordTrade(trade: Record<string, any>): Promise<void>;
}

export interface MarketDataAdapter {
  fetchWatchlistPrices(symbols: string[]): Promise<Array<{ symbol: string; price: number }>>;
}

export interface SettingsProvider {
  loadSettings(): Promise<Record<string, any>>;
}

export interface AgentEventEmitter {
  emit(event: string, payload: Record<string, any>): void;
}

export interface AgentContext {
  broker: BrokerAdapter;
  persistence: PersistenceAdapter;
  marketData: MarketDataAdapter;
  settings: SettingsProvider;
  events?: AgentEventEmitter;
  logger?: {
    info: (...args: any[]) => void;
    warn: (...args: any[]) => void;
    error: (...args: any[]) => void;
    debug?: (...args: any[]) => void;
  };
}
