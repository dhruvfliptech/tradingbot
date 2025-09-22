import { Account, CryptoData, Order, Position } from '../../types/trading';

export type TradingBrokerId = 'alpaca' | 'binance';

export interface BrokerFeatures {
  paperTrading: boolean;
  liveTrading: boolean;
  margin?: boolean;
  supportsCrypto: boolean;
  supportsEquities?: boolean;
}

export interface BrokerMetadata {
  id: TradingBrokerId;
  label: string;
  description: string;
  features: BrokerFeatures;
  baseCurrency: string;
  docsUrl?: string;
}

export interface PlaceOrderParams {
  symbol: string;
  qty: string | number;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit';
  time_in_force?: 'day' | 'gtc' | 'ioc';
  limit_price?: number | null;
  client_order_id?: string;
}

export interface TradingBroker {
  readonly id: TradingBrokerId;
  readonly metadata: BrokerMetadata;
  normalizeSymbol(symbol: string): string;
  toBrokerSymbol(symbol: string): string;
  fromBrokerSymbol(symbol: string): string;
  getAccount(): Promise<Account>;
  getPositions(): Promise<Position[]>;
  getOrders(): Promise<Order[]>;
  placeOrder(order: PlaceOrderParams): Promise<Order>;
  cancelOrder?(orderId: string, symbol?: string): Promise<void>;
  testConnection?(): Promise<boolean>;
  getCryptoData?(symbols?: string[]): Promise<CryptoData[]>;
}
