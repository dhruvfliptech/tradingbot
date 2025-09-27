/**
 * Secure Binance Broker Implementation
 * Uses backend proxy endpoints - NO API KEYS IN FRONTEND
 */

import { Account, Order, Position } from '../../types/trading';
import { PlaceOrderParams, TradingBroker } from './types';

const API_BASE = import.meta.env.VITE_BACKEND_URL || 'http://localhost:3001';

class BinanceBrokerSecure implements TradingBroker {
  readonly id = 'binance' as const;

  readonly metadata = {
    id: this.id,
    label: 'Binance US',
    description: 'Live crypto spot trading via Binance US exchange',
    features: {
      paperTrading: false,
      liveTrading: true,
      margin: false,
      supportsCrypto: true,
      supportsEquities: false,
    },
    baseCurrency: 'USD',
    docsUrl: 'https://docs.binance.us/#public-rest-api',
  } as const;

  private async fetchWithAuth(endpoint: string, options: RequestInit = {}): Promise<Response> {
    // Get auth token from localStorage (set by Supabase)
    const token = localStorage.getItem('supabase.auth.token');

    const response = await fetch(`${API_BASE}/api/v1${endpoint}`, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...(token && { 'Authorization': `Bearer ${token}` }),
        ...options.headers,
      },
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ error: 'Request failed' }));
      throw new Error(error.error || `Request failed with status ${response.status}`);
    }

    return response;
  }

  async initialize(): Promise<boolean> {
    try {
      // Test connection by fetching server time
      const response = await this.fetchWithAuth('/binance/time');
      const data = await response.json();
      return data.success === true;
    } catch (error) {
      console.error('Failed to initialize Binance broker:', error);
      return false;
    }
  }

  async getAccount(): Promise<Account> {
    try {
      const response = await this.fetchWithAuth('/binance/account');
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch account');
      }

      // Transform backend response to Account format
      const account = result.data;
      return {
        id: 'binance-account',
        brokerType: 'binance',
        accountNumber: account.accountNumber || 'BINANCE-US',
        buyingPower: account.buyingPower || 0,
        cash: account.cash || 0,
        portfolioValue: account.portfolioValue || 0,
        dayTradeCount: account.dayTradeCount || 0,
        patternDayTrader: false,
        tradingBlocked: account.tradingBlocked || false,
        transfersBlocked: account.transfersBlocked || false,
        accountBlocked: account.accountBlocked || false,
        createdAt: account.createdAt || new Date().toISOString(),
        currency: 'USD',
        positions: account.positions || [],
        cryptoBalances: account.cryptoBalances || {}
      };
    } catch (error) {
      console.error('Error fetching account:', error);
      throw error;
    }
  }

  async getPositions(): Promise<Position[]> {
    try {
      const response = await this.fetchWithAuth('/binance/positions');
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch positions');
      }

      return result.data || [];
    } catch (error) {
      console.error('Error fetching positions:', error);
      throw error;
    }
  }

  async placeOrder(params: PlaceOrderParams): Promise<Order> {
    try {
      const response = await this.fetchWithAuth('/binance/order', {
        method: 'POST',
        body: JSON.stringify(params),
      });

      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to place order');
      }

      return result.data;
    } catch (error) {
      console.error('Error placing order:', error);
      throw error;
    }
  }

  async cancelOrder(orderId: string, symbol?: string): Promise<boolean> {
    try {
      if (!symbol) {
        throw new Error('Symbol is required to cancel Binance order');
      }

      const response = await this.fetchWithAuth(`/binance/orders/${orderId}`, {
        method: 'DELETE',
        body: JSON.stringify({ symbol }),
      });

      const result = await response.json();
      return result.success === true;
    } catch (error) {
      console.error('Error canceling order:', error);
      return false;
    }
  }

  async getOrders(): Promise<Order[]> {
    try {
      const response = await this.fetchWithAuth('/binance/orders');
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch orders');
      }

      return result.data || [];
    } catch (error) {
      console.error('Error fetching orders:', error);
      throw error;
    }
  }

  async getOpenOrders(symbol?: string): Promise<Order[]> {
    try {
      const url = symbol
        ? `/binance/orders?symbol=${encodeURIComponent(symbol)}`
        : '/binance/orders';

      const response = await this.fetchWithAuth(url);
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch open orders');
      }

      return result.data || [];
    } catch (error) {
      console.error('Error fetching open orders:', error);
      throw error;
    }
  }

  async getOrderHistory(symbol?: string, limit?: number): Promise<Order[]> {
    try {
      const params = new URLSearchParams();
      if (symbol) params.append('symbol', symbol);
      if (limit) params.append('limit', limit.toString());

      const url = `/binance/orders/history${params.toString() ? '?' + params.toString() : ''}`;
      const response = await this.fetchWithAuth(url);
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch order history');
      }

      return result.data || [];
    } catch (error) {
      console.error('Error fetching order history:', error);
      throw error;
    }
  }

  async getMarketData(symbol: string): Promise<any> {
    try {
      const response = await this.fetchWithAuth(`/binance/market/${encodeURIComponent(symbol)}`);
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch market data');
      }

      return result.data;
    } catch (error) {
      console.error('Error fetching market data:', error);
      throw error;
    }
  }

  async getExchangeInfo(symbol?: string): Promise<any> {
    try {
      const url = symbol
        ? `/binance/exchangeInfo?symbol=${encodeURIComponent(symbol)}`
        : '/binance/exchangeInfo';

      const response = await this.fetchWithAuth(url);
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch exchange info');
      }

      return result.data;
    } catch (error) {
      console.error('Error fetching exchange info:', error);
      throw error;
    }
  }

  async getServerTime(): Promise<number> {
    try {
      const response = await this.fetchWithAuth('/binance/time');
      const result = await response.json();

      if (!result.success) {
        throw new Error(result.error || 'Failed to fetch server time');
      }

      return result.data.serverTime;
    } catch (error) {
      console.error('Error fetching server time:', error);
      throw error;
    }
  }

  // Utility method to check if the broker is available
  async isAvailable(): Promise<boolean> {
    try {
      const response = await this.fetchWithAuth('/binance/time');
      const result = await response.json();
      return result.success === true;
    } catch {
      return false;
    }
  }

  // Method to test authentication
  async testAuth(): Promise<boolean> {
    try {
      const response = await this.fetchWithAuth('/binance/account');
      const result = await response.json();
      return result.success === true;
    } catch {
      return false;
    }
  }
}

export const binanceBrokerSecure = new BinanceBrokerSecure();