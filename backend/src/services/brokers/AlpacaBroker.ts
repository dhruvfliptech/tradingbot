/**
 * Alpaca Broker Service - Backend implementation
 * Handles paper trading for equities & crypto via Alpaca Markets
 */

import fetch from 'node-fetch';
import {
  BrokerService,
  BrokerAccount,
  BrokerPosition,
  BrokerOrder,
  PlaceOrderParams,
  MarketData,
  BrokerConfig
} from './BrokerService';
import logger from '../../utils/logger';

interface AlpacaAccount {
  id: string;
  cash: string;
  portfolio_value: string;
  buying_power: string;
  account_number: string;
  status: string;
}

interface AlpacaPosition {
  symbol: string;
  qty: string;
  market_value: string;
  cost_basis: string;
  unrealized_pl: string;
  unrealized_plpc: string;
  side: string;
}

interface AlpacaOrder {
  id: string;
  symbol: string;
  qty: string;
  side: string;
  order_type: string;
  status: string;
  filled_qty: string;
  filled_avg_price: string;
  submitted_at: string;
  filled_at: string | null;
  limit_price: string | null;
}

export class AlpacaBroker extends BrokerService {
  private cryptoSymbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'DOT', 'LINK', 'UNI', 'ATOM', 'NEAR'];

  constructor(config?: BrokerConfig) {
    super(config);
  }

  getBrokerName(): string {
    return 'ALPACA';
  }

  getDefaultBaseUrl(): string {
    const isPaper = process.env.ALPACA_PAPER_TRADING !== 'false';
    return isPaper
      ? 'https://paper-api.alpaca.markets/v2'
      : 'https://api.alpaca.markets/v2';
  }

  private getHeaders(): Record<string, string> {
    return {
      'APCA-API-KEY-ID': this.apiKey,
      'APCA-API-SECRET-KEY': this.secretKey,
      'Content-Type': 'application/json'
    };
  }

  private async request<T = any>(endpoint: string, options: any = {}): Promise<T> {
    try {
      this.validateCredentials();

      const url = `${this.baseUrl}${endpoint}`;
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.getHeaders(),
          ...options.headers
        }
      });

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`Alpaca API Error (${response.status}): ${errorText}`);
      }

      return await response.json() as T;
    } catch (error) {
      this.handleApiError(error);
      throw error;
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.request('/account');
      this.isConnected = true;
      return true;
    } catch (error) {
      logger.warn('Alpaca connection test failed:', error);
      this.isConnected = false;
      return false;
    }
  }

  async getAccount(): Promise<BrokerAccount> {
    try {
      const account = await this.request<AlpacaAccount>('/account');

      return {
        id: account.id,
        balance_usd: parseFloat(account.cash),
        balance_btc: 0, // Alpaca doesn't provide BTC balance directly
        portfolio_value: parseFloat(account.portfolio_value),
        available_balance: parseFloat(account.buying_power),
        total_trades: 0 // Not provided by Alpaca
      };
    } catch (error) {
      logger.error('Error fetching Alpaca account:', error);
      throw error;
    }
  }

  async getPositions(): Promise<BrokerPosition[]> {
    try {
      const positions = await this.request<AlpacaPosition[]>('/positions');

      return positions.map(position => ({
        symbol: position.symbol,
        qty: position.qty,
        market_value: parseFloat(position.market_value),
        cost_basis: parseFloat(position.cost_basis),
        unrealized_pl: parseFloat(position.unrealized_pl),
        unrealized_plpc: parseFloat(position.unrealized_plpc),
        side: position.side as 'long' | 'short',
        name: position.symbol,
        image: this.getCryptoImage(position.symbol)
      }));
    } catch (error) {
      logger.error('Error fetching Alpaca positions:', error);
      return [];
    }
  }

  async getOrders(status: 'all' | 'open' | 'closed' = 'all'): Promise<BrokerOrder[]> {
    try {
      const params = new URLSearchParams({
        status: status,
        limit: '50'
      });

      const orders = await this.request<AlpacaOrder[]>(`/orders?${params}`);

      return orders.map(order => this.mapOrder(order));
    } catch (error) {
      logger.error('Error fetching Alpaca orders:', error);
      return [];
    }
  }

  async placeOrder(params: PlaceOrderParams): Promise<BrokerOrder> {
    try {
      this.validateCredentials();

      // For crypto orders, Alpaca requires time_in_force to be 'gtc'
      const isCrypto = this.isCryptoSymbol(params.symbol);
      const timeInForce = isCrypto ? 'gtc' : (params.time_in_force || 'gtc');

      const payload = {
        symbol: this.normalizeSymbol(params.symbol),
        qty: typeof params.qty === 'number' ? params.qty.toString() : params.qty,
        side: params.side,
        type: params.order_type,
        time_in_force: timeInForce,
        ...(params.limit_price ? { limit_price: params.limit_price.toString() } : {}),
        ...(params.client_order_id ? { client_order_id: params.client_order_id } : {})
      };

      const order = await this.request<AlpacaOrder>('/orders', {
        method: 'POST',
        body: JSON.stringify(payload)
      });

      return this.mapOrder(order);
    } catch (error) {
      logger.error('Error placing Alpaca order:', error);
      throw error;
    }
  }

  async cancelOrder(orderId: string): Promise<boolean> {
    try {
      await this.request(`/orders/${orderId}`, {
        method: 'DELETE'
      });
      return true;
    } catch (error) {
      logger.error(`Error canceling Alpaca order ${orderId}:`, error);
      return false;
    }
  }

  async getMarketData(symbols: string[]): Promise<MarketData[]> {
    try {
      const normalizedSymbols = symbols.map(s => this.normalizeSymbol(s));
      const symbolsParam = normalizedSymbols.join(',');

      const barsData = await this.request<any>(
        `/crypto/bars?symbols=${symbolsParam}&timeframe=1Min&limit=1`
      );

      const marketData: MarketData[] = [];

      for (const symbol of normalizedSymbols) {
        const bars = barsData.bars[symbol];
        if (bars && bars.length > 0) {
          const latestBar = bars[0];
          const price = latestBar.c;
          const change = latestBar.c - latestBar.o;
          const changePercent = (change / latestBar.o) * 100;

          marketData.push({
            symbol: symbol.replace('USD', ''),
            price,
            change,
            changePercent,
            volume: latestBar.v,
            high: latestBar.h,
            low: latestBar.l,
            timestamp: new Date().toISOString()
          });
        }
      }

      return marketData;
    } catch (error) {
      logger.error('Error fetching Alpaca market data:', error);
      return [];
    }
  }

  normalizeSymbol(symbol: string): string {
    if (!symbol) return symbol;
    let normalized = symbol.trim().toUpperCase();

    // Convert USDT to USD for Alpaca
    if (normalized.endsWith('USDT')) {
      normalized = normalized.replace(/USDT$/, 'USD');
    }

    // Add USD suffix if not present
    if (!normalized.endsWith('USD') && normalized.length <= 6) {
      normalized = `${normalized}USD`;
    }

    return normalized;
  }

  private isCryptoSymbol(symbol: string): boolean {
    const normalizedSymbol = symbol.toUpperCase();
    return this.cryptoSymbols.some(crypto =>
      normalizedSymbol.startsWith(crypto) ||
      normalizedSymbol.includes(crypto)
    ) || normalizedSymbol.endsWith('USD');
  }

  private mapOrder(order: AlpacaOrder): BrokerOrder {
    return {
      id: order.id,
      symbol: order.symbol,
      qty: order.qty,
      side: order.side as 'buy' | 'sell',
      order_type: order.order_type as 'market' | 'limit',
      status: this.mapOrderStatus(order.status),
      filled_qty: order.filled_qty || '0',
      filled_avg_price: parseFloat(order.filled_avg_price || '0'),
      submitted_at: order.submitted_at,
      filled_at: order.filled_at,
      limit_price: order.limit_price ? parseFloat(order.limit_price) : null
    };
  }

  private mapOrderStatus(status: string): 'pending' | 'filled' | 'canceled' {
    switch (status.toLowerCase()) {
      case 'new':
      case 'accepted':
      case 'pending_new':
      case 'partially_filled':
        return 'pending';
      case 'filled':
        return 'filled';
      case 'canceled':
      case 'expired':
      case 'rejected':
        return 'canceled';
      default:
        return 'pending';
    }
  }

  private getCryptoImage(symbol: string): string {
    const key = symbol.toUpperCase().replace('USD', '');
    const images: Record<string, string> = {
      BTC: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
      ETH: 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
      ADA: 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
      SOL: 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
      BNB: 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
      DOGE: 'https://assets.coingecko.com/coins/images/5/large/dogecoin.png',
      MATIC: 'https://assets.coingecko.com/coins/images/4713/large/matic-token-icon.png',
      AVAX: 'https://assets.coingecko.com/coins/images/12559/large/coin-round-red.png',
      DOT: 'https://assets.coingecko.com/coins/images/12171/large/polkadot.png',
      LINK: 'https://assets.coingecko.com/coins/images/877/large/chainlink-new-logo.png',
      UNI: 'https://assets.coingecko.com/coins/images/12504/large/uniswap.png',
      ATOM: 'https://assets.coingecko.com/coins/images/1481/large/cosmos.png',
      NEAR: 'https://assets.coingecko.com/coins/images/10365/large/near.png'
    };
    return images[key] || '';
  }
}