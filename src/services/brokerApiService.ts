/**
 * Broker API Service - Frontend client for broker operations
 * Communicates with backend broker services via REST API
 */

import { supabase } from '../lib/supabase';

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
  broker?: 'alpaca' | 'binance';
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

class BrokerApiService {
  private apiUrl: string;

  constructor() {
    this.apiUrl = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001';
  }

  /**
   * Make API call to backend
   */
  private async apiCall<T = any>(
    endpoint: string,
    method: string = 'GET',
    body?: any
  ): Promise<T> {
    try {
      // Get auth token
      const { data: { session } } = await supabase.auth.getSession();
      const token = session?.access_token;

      const options: RequestInit = {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...(token && { 'Authorization': `Bearer ${token}` })
        }
      };

      if (body && method !== 'GET') {
        options.body = JSON.stringify(body);
      }

      const response = await fetch(`${this.apiUrl}/api/v1${endpoint}`, options);
      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || `HTTP ${response.status}: ${response.statusText}`);
      }

      return data.data;
    } catch (error) {
      console.error(`Broker API call failed: ${endpoint}`, error);
      throw error;
    }
  }

  /**
   * Get available brokers
   */
  async getBrokers(): Promise<{
    brokers: Array<{ type: string; connected: boolean; name: string }>;
    active: string;
  }> {
    return this.apiCall('/brokers');
  }

  /**
   * Set active broker
   */
  async setActiveBroker(broker: 'alpaca' | 'binance'): Promise<void> {
    await this.apiCall('/brokers/active', 'POST', { broker });
  }

  /**
   * Test broker connections
   */
  async testConnections(): Promise<{ connections: Record<string, boolean> }> {
    return this.apiCall('/brokers/test');
  }

  /**
   * Get account information
   */
  async getAccount(broker?: 'alpaca' | 'binance'): Promise<{
    account: BrokerAccount;
    broker: string;
  }> {
    const query = broker ? `?broker=${broker}` : '';
    return this.apiCall(`/brokers/account${query}`);
  }

  /**
   * Get aggregated accounts from all brokers
   */
  async getAggregatedAccounts(): Promise<{
    total_portfolio_value: number;
    total_available_balance: number;
    accounts: Array<{ broker: string; account: BrokerAccount }>;
  }> {
    return this.apiCall('/brokers/accounts');
  }

  /**
   * Get positions
   */
  async getPositions(broker?: 'alpaca' | 'binance'): Promise<{
    positions: BrokerPosition[];
    broker: string;
  }> {
    const query = broker ? `?broker=${broker}` : '';
    return this.apiCall(`/brokers/positions${query}`);
  }

  /**
   * Get orders
   */
  async getOrders(options?: {
    broker?: 'alpaca' | 'binance';
    status?: 'all' | 'open' | 'closed';
  }): Promise<{
    orders: BrokerOrder[];
    broker: string;
  }> {
    const params = new URLSearchParams();
    if (options?.broker) params.set('broker', options.broker);
    if (options?.status) params.set('status', options.status);
    const query = params.toString() ? `?${params.toString()}` : '';
    return this.apiCall(`/brokers/orders${query}`);
  }

  /**
   * Place an order
   */
  async placeOrder(params: PlaceOrderParams): Promise<{
    order: BrokerOrder;
    broker: string;
  }> {
    return this.apiCall('/brokers/orders', 'POST', params);
  }

  /**
   * Cancel an order
   */
  async cancelOrder(
    orderId: string,
    broker?: 'alpaca' | 'binance'
  ): Promise<{
    message: string;
    broker: string;
  }> {
    const query = broker ? `?broker=${broker}` : '';
    return this.apiCall(`/brokers/orders/${orderId}${query}`, 'DELETE');
  }

  /**
   * Get market data
   */
  async getMarketData(
    symbols: string[],
    broker?: 'alpaca' | 'binance'
  ): Promise<{
    marketData: MarketData[];
    broker: string;
  }> {
    const params = new URLSearchParams();
    params.set('symbols', symbols.join(','));
    if (broker) params.set('broker', broker);
    return this.apiCall(`/brokers/market-data?${params.toString()}`);
  }

  /**
   * Initialize broker with API credentials
   */
  async initializeBroker(
    broker: 'alpaca' | 'binance',
    apiKey: string,
    secretKey: string,
    baseUrl?: string
  ): Promise<{
    message: string;
    broker: string;
  }> {
    return this.apiCall('/brokers/initialize', 'POST', {
      broker,
      apiKey,
      secretKey,
      baseUrl
    });
  }
}

// Export singleton instance
export const brokerApiService = new BrokerApiService();