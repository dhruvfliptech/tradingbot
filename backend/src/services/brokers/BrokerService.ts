/**
 * Base Broker Service - Abstract class for all broker implementations
 * Handles secure API key management and common broker operations
 */

import { EventEmitter } from 'events';
import logger from '../../utils/logger';

export interface BrokerAccount {
  id: string;
  balance_usd: number;
  balance_btc: number;
  portfolio_value: number;
  available_balance: number;
  total_trades: number;
}

export interface BrokerPosition {
  symbol: string;
  qty: string | number;
  market_value: number;
  cost_basis: number;
  unrealized_pl: number;
  unrealized_plpc: number;
  side: 'long' | 'short';
  name: string;
  image?: string;
}

export interface BrokerOrder {
  id: string;
  symbol: string;
  qty: string | number;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit';
  status: 'pending' | 'filled' | 'canceled';
  filled_qty?: string | number;
  filled_avg_price?: number;
  submitted_at?: string;
  filled_at?: string | null;
  limit_price?: number | null;
}

export interface PlaceOrderParams {
  symbol: string;
  qty: string | number;
  side: 'buy' | 'sell';
  order_type: 'market' | 'limit';
  limit_price?: number;
  time_in_force?: 'day' | 'gtc' | 'ioc' | 'fok';
  client_order_id?: string;
}

export interface MarketData {
  symbol: string;
  price: number;
  change: number;
  changePercent: number;
  volume: number;
  high: number;
  low: number;
  timestamp: string;
}

export interface BrokerConfig {
  apiKey?: string;
  secretKey?: string;
  baseUrl?: string;
  paperTrading?: boolean;
  recvWindow?: number;
  timeout?: number;
}

export abstract class BrokerService extends EventEmitter {
  protected apiKey: string = '';
  protected secretKey: string = '';
  protected baseUrl: string = '';
  protected isConnected: boolean = false;
  protected lastError?: string;

  constructor(protected config: BrokerConfig = {}) {
    super();
    this.initializeConfig(config);
  }

  /**
   * Initialize broker configuration from environment or provided config
   */
  protected initializeConfig(config: BrokerConfig): void {
    this.apiKey = config.apiKey || process.env[`${this.getBrokerName()}_API_KEY`] || '';
    this.secretKey = config.secretKey || process.env[`${this.getBrokerName()}_SECRET_KEY`] || '';
    this.baseUrl = config.baseUrl || this.getDefaultBaseUrl();

    if (!this.apiKey || !this.secretKey) {
      logger.warn(`${this.getBrokerName()} API keys not configured`);
    }
  }

  /**
   * Get broker name for configuration
   */
  abstract getBrokerName(): string;

  /**
   * Get default base URL for the broker
   */
  abstract getDefaultBaseUrl(): string;

  /**
   * Test connection to broker API
   */
  abstract testConnection(): Promise<boolean>;

  /**
   * Get account information
   */
  abstract getAccount(): Promise<BrokerAccount>;

  /**
   * Get current positions
   */
  abstract getPositions(): Promise<BrokerPosition[]>;

  /**
   * Get orders (open and closed)
   */
  abstract getOrders(status?: 'all' | 'open' | 'closed'): Promise<BrokerOrder[]>;

  /**
   * Place an order
   */
  abstract placeOrder(params: PlaceOrderParams): Promise<BrokerOrder>;

  /**
   * Cancel an order
   */
  abstract cancelOrder(orderId: string): Promise<boolean>;

  /**
   * Get market data for symbols
   */
  abstract getMarketData(symbols: string[]): Promise<MarketData[]>;

  /**
   * Normalize symbol to broker format
   */
  abstract normalizeSymbol(symbol: string): string;

  /**
   * Check if broker is connected
   */
  isConnected(): boolean {
    return this.isConnected;
  }

  /**
   * Get last error message
   */
  getLastError(): string | undefined {
    return this.lastError;
  }

  /**
   * Validate API credentials
   */
  protected validateCredentials(): void {
    if (!this.apiKey || !this.secretKey) {
      throw new Error(`${this.getBrokerName()} API credentials not configured`);
    }
  }

  /**
   * Handle API errors consistently
   */
  protected handleApiError(error: any): void {
    const errorMessage = error?.message || 'Unknown error';
    this.lastError = errorMessage;
    logger.error(`${this.getBrokerName()} API Error:`, error);
    this.emit('error', { broker: this.getBrokerName(), error: errorMessage });
  }

  /**
   * Format quantity based on broker requirements
   */
  protected formatQuantity(quantity: number, decimals: number = 8): string {
    return quantity.toFixed(decimals)
      .replace(/\.0+$/, '')
      .replace(/(\.\d*?)0+$/, '$1')
      .replace(/\.$/, '');
  }

  /**
   * Format price based on broker requirements
   */
  protected formatPrice(price: number, decimals: number = 8): string {
    return this.formatQuantity(price, decimals);
  }
}