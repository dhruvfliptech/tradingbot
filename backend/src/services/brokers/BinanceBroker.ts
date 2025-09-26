/**
 * Binance Broker Service - Backend implementation
 * Handles live crypto spot trading via Binance US exchange
 */

import fetch from 'node-fetch';
import * as crypto from 'crypto';
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

interface BinanceBalance {
  asset: string;
  free: string;
  locked: string;
}

interface BinanceAccount {
  balances: BinanceBalance[];
  accountType: string;
  updateTime: number;
}

interface BinanceOrderResponse {
  orderId: number;
  clientOrderId: string;
  symbol: string;
  status: string;
  side: 'BUY' | 'SELL';
  type: string;
  origQty: string;
  executedQty: string;
  cummulativeQuoteQty: string;
  price: string;
  fills?: Array<{
    price: string;
    qty: string;
    commission: string;
    commissionAsset: string;
  }>;
  updateTime: number;
  transactTime?: number;
}

interface BinanceTicker {
  symbol: string;
  lastPrice: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
  highPrice: string;
  lowPrice: string;
}

export class BinanceBroker extends BrokerService {
  private recvWindow: number = 5000;
  private serverTimeOffsetMs: number = 0;

  constructor(config?: BrokerConfig) {
    super(config);
    this.recvWindow = config?.recvWindow || Number(process.env.BINANCE_RECV_WINDOW) || 5000;
    // Initialize server time sync
    this.syncServerTime().catch(() => {});
  }

  getBrokerName(): string {
    return 'BINANCE';
  }

  getDefaultBaseUrl(): string {
    // Default to Binance US
    return process.env.BINANCE_BASE_URL || 'https://api.binance.us';
  }

  private async syncServerTime(): Promise<void> {
    try {
      const response = await fetch(`${this.baseUrl}/api/v3/time`);
      if (!response.ok) return;

      const data = await response.json() as any;
      if (data && typeof data.serverTime === 'number') {
        this.serverTimeOffsetMs = data.serverTime - Date.now();
        logger.debug('Binance server time synced, offset:', this.serverTimeOffsetMs);
      }
    } catch (error) {
      logger.warn('Failed to sync Binance server time:', error);
    }
  }

  private generateSignature(queryString: string): string {
    return crypto
      .createHmac('sha256', this.secretKey)
      .update(queryString)
      .digest('hex');
  }

  private async request<T = any>(
    path: string,
    options: {
      method?: 'GET' | 'POST' | 'DELETE';
      params?: Record<string, string | number | undefined | null>;
      signed?: boolean;
      retryOnTimeSkew?: boolean;
    } = {}
  ): Promise<T> {
    try {
      const method = options.method || 'GET';
      const params = new URLSearchParams();

      if (options.params) {
        for (const [key, value] of Object.entries(options.params)) {
          if (value !== undefined && value !== null && value !== '') {
            params.append(key, String(value));
          }
        }
      }

      let headers: Record<string, string> = {};
      let body: string | undefined;
      let url = `${this.baseUrl}${path}`;

      if (options.signed) {
        this.validateCredentials();

        // Add timestamp with server time offset
        const now = Date.now() + this.serverTimeOffsetMs;
        params.set('timestamp', Math.floor(now).toString());
        if (!params.has('recvWindow')) {
          params.set('recvWindow', this.recvWindow.toString());
        }

        // Generate signature
        const queryString = params.toString();
        const signature = this.generateSignature(queryString);
        params.set('signature', signature);

        headers['X-MBX-APIKEY'] = this.apiKey;

        if (method === 'GET' || method === 'DELETE') {
          url += `?${params.toString()}`;
        } else {
          body = params.toString();
          headers['Content-Type'] = 'application/x-www-form-urlencoded';
        }
      } else {
        const queryString = params.toString();
        if (queryString) {
          url += `?${queryString}`;
        }
      }

      const response = await fetch(url, {
        method,
        headers,
        body
      });

      if (!response.ok) {
        let errorBody: any = null;
        try {
          errorBody = await response.json();
        } catch {
          errorBody = { message: await response.text() };
        }

        // Handle time skew error
        if (errorBody?.code === -1021 || String(errorBody?.message).includes('timestamp')) {
          if (options.signed && options.retryOnTimeSkew !== false) {
            await this.syncServerTime();
            // Retry once with updated offset
            return await this.request<T>(path, { ...options, retryOnTimeSkew: false });
          }
        }

        const errorMsg = `Binance API Error ${errorBody?.code || response.status}: ${errorBody?.msg || errorBody?.message}`;
        throw new Error(errorMsg);
      }

      if (response.status === 204) {
        return {} as T;
      }

      return await response.json() as T;
    } catch (error) {
      this.handleApiError(error);
      throw error;
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.request('/api/v3/account', { signed: true });
      this.isConnected = true;
      return true;
    } catch (error) {
      logger.warn('Binance connection test failed:', error);
      this.isConnected = false;
      return false;
    }
  }

  async getAccount(): Promise<BrokerAccount> {
    try {
      const account = await this.request<BinanceAccount>('/api/v3/account', { signed: true });
      const balances = account.balances || [];
      const filtered = balances.filter(b => this.hasBalance(b));

      // Get prices for all assets
      const assets = filtered.map(b => b.asset);
      const priceMap = await this.getPriceMap(assets);

      let portfolioValue = 0;
      let availableBalance = 0;
      let usdCashFree = 0;
      let btcBalance = 0;

      for (const bal of filtered) {
        const totalQty = parseFloat(bal.free) + parseFloat(bal.locked);
        const freeQty = parseFloat(bal.free);
        const usdValue = this.convertToUsd(bal.asset, totalQty, priceMap);
        const usdFree = this.convertToUsd(bal.asset, freeQty, priceMap);

        portfolioValue += usdValue;
        availableBalance += usdFree;

        if (bal.asset === 'BTC') {
          btcBalance = totalQty;
        }
        if (bal.asset === 'USD') {
          usdCashFree = freeQty;
        }
      }

      return {
        id: 'binance-account',
        balance_usd: portfolioValue,
        balance_btc: btcBalance,
        portfolio_value: portfolioValue,
        available_balance: usdCashFree || availableBalance,
        total_trades: 0
      };
    } catch (error) {
      logger.error('Error fetching Binance account:', error);
      throw error;
    }
  }

  async getPositions(): Promise<BrokerPosition[]> {
    try {
      const account = await this.request<BinanceAccount>('/api/v3/account', { signed: true });
      const balances = (account.balances || []).filter(b => this.hasBalance(b));
      const assets = balances.map(b => b.asset);
      const priceMap = await this.getPriceMap(assets);

      return balances.map(bal => {
        const asset = bal.asset;
        const totalQty = parseFloat(bal.free) + parseFloat(bal.locked);
        const price = priceMap[asset] || 0;
        const marketValue = this.convertToUsd(asset, totalQty, priceMap);

        return {
          symbol: `${asset}USD`,
          qty: totalQty,
          market_value: marketValue,
          cost_basis: marketValue, // Binance doesn't provide cost basis
          unrealized_pl: 0,
          unrealized_plpc: 0,
          side: 'long' as const,
          name: asset,
          image: this.getCryptoImage(asset)
        };
      });
    } catch (error) {
      logger.error('Error fetching Binance positions:', error);
      return [];
    }
  }

  async getOrders(status: 'all' | 'open' | 'closed' = 'all'): Promise<BrokerOrder[]> {
    const symbols = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'ADAUSD'];
    const orders: BrokerOrder[] = [];

    for (const symbol of symbols) {
      try {
        const response = await this.request<any[]>('/api/v3/allOrders', {
          signed: true,
          params: { symbol, limit: 20 }
        });

        for (const raw of response) {
          orders.push(this.mapOrder(raw));
        }
      } catch (error) {
        logger.warn(`Failed to fetch Binance orders for ${symbol}:`, error);
      }
    }

    // Filter by status if needed
    let filtered = orders;
    if (status === 'open') {
      filtered = orders.filter(o => o.status === 'pending');
    } else if (status === 'closed') {
      filtered = orders.filter(o => o.status !== 'pending');
    }

    // Sort by time descending
    return filtered.sort((a, b) => {
      const aTime = a.filled_at || a.submitted_at || '';
      const bTime = b.filled_at || b.submitted_at || '';
      return new Date(bTime).getTime() - new Date(aTime).getTime();
    });
  }

  async placeOrder(params: PlaceOrderParams): Promise<BrokerOrder> {
    try {
      this.validateCredentials();

      const symbol = this.normalizeSymbol(params.symbol);
      const qty = typeof params.qty === 'number' ? params.qty : Number(params.qty);

      if (!qty || Number.isNaN(qty) || qty <= 0) {
        throw new Error('Invalid quantity for Binance order');
      }

      const orderParams: Record<string, string | number> = {
        symbol,
        side: params.side.toUpperCase(),
        type: params.order_type === 'limit' ? 'LIMIT' : 'MARKET',
        quantity: this.formatQuantity(qty),
        newOrderRespType: 'FULL'
      };

      if (orderParams.type === 'LIMIT') {
        orderParams.timeInForce = (params.time_in_force || 'GTC').toUpperCase();
        if (params.limit_price) {
          orderParams.price = this.formatPrice(params.limit_price);
        } else {
          throw new Error('Limit orders require a limit_price');
        }
      }

      if (params.client_order_id) {
        orderParams.newClientOrderId = params.client_order_id;
      }

      const response = await this.request<BinanceOrderResponse>('/api/v3/order', {
        method: 'POST',
        signed: true,
        params: orderParams
      });

      return this.mapOrder(response);
    } catch (error) {
      logger.error('Error placing Binance order:', error);
      throw error;
    }
  }

  async cancelOrder(orderId: string): Promise<boolean> {
    try {
      // Note: Binance requires symbol for cancellation
      // In production, you'd need to track order-symbol mapping
      await this.request('/api/v3/order', {
        method: 'DELETE',
        signed: true,
        params: { orderId }
      });
      return true;
    } catch (error) {
      logger.error(`Error canceling Binance order ${orderId}:`, error);
      return false;
    }
  }

  async getMarketData(symbols: string[]): Promise<MarketData[]> {
    try {
      const brokerSymbols = symbols.map(s => this.normalizeSymbol(s));
      const response = await this.request<BinanceTicker[]>('/api/v3/ticker/24hr', {
        params: { symbols: JSON.stringify(brokerSymbols) }
      });

      return response.map(ticker => {
        const symbolCode = ticker.symbol.replace('USD', '');
        return {
          symbol: symbolCode,
          price: parseFloat(ticker.lastPrice),
          change: parseFloat(ticker.priceChange),
          changePercent: parseFloat(ticker.priceChangePercent),
          volume: parseFloat(ticker.volume),
          high: parseFloat(ticker.highPrice),
          low: parseFloat(ticker.lowPrice),
          timestamp: new Date().toISOString()
        };
      });
    } catch (error) {
      logger.error('Error fetching Binance market data:', error);
      return [];
    }
  }

  normalizeSymbol(symbol: string): string {
    if (!symbol) return symbol;
    const normalized = symbol.trim().toUpperCase();
    if (normalized.endsWith('USD') || normalized.endsWith('USDT')) {
      return normalized;
    }
    // Default to USD pairs on Binance US
    return `${normalized}USD`;
  }

  private hasBalance(balance: BinanceBalance): boolean {
    const free = parseFloat(balance.free);
    const locked = parseFloat(balance.locked);
    return (free + locked) > 0.0000001;
  }

  private async getPriceMap(assets: string[]): Promise<Record<string, number>> {
    const uniqueAssets = Array.from(new Set(assets))
      .filter(asset => asset !== 'USDT' && asset !== 'BUSD' && asset !== 'USD');

    if (uniqueAssets.length === 0) {
      return { USDT: 1, BUSD: 1, USD: 1 };
    }

    try {
      const symbols = uniqueAssets.map(asset => `${asset}USD`);
      const response = await this.request<any[]>('/api/v3/ticker/price', {
        params: { symbols: JSON.stringify(symbols) }
      });

      const map: Record<string, number> = {};
      for (const ticker of response) {
        const asset = ticker.symbol.replace(/USD.*$/, '');
        map[asset] = parseFloat(ticker.price);
      }
      map.USDT = 1;
      map.BUSD = 1;
      map.USD = 1;
      return map;
    } catch (error) {
      logger.warn('Failed to fetch Binance price map:', error);
      const map: Record<string, number> = {};
      for (const asset of uniqueAssets) {
        map[asset] = 0;
      }
      map.USDT = 1;
      map.BUSD = 1;
      map.USD = 1;
      return map;
    }
  }

  private convertToUsd(asset: string, quantity: number, priceMap: Record<string, number>): number {
    if (asset === 'USDT' || asset === 'BUSD' || asset === 'USD') {
      return quantity;
    }
    const price = priceMap[asset] || 0;
    return quantity * price;
  }

  private mapOrder(raw: any): BrokerOrder {
    const status = this.mapOrderStatus(raw.status);
    const side = (raw.side || '').toLowerCase() as 'buy' | 'sell';
    const filledQty = raw.executedQty || '0';
    const avgPrice = this.calculateAveragePrice(raw);

    return {
      id: String(raw.orderId || raw.id),
      symbol: raw.symbol,
      qty: raw.origQty || raw.quantity || '0',
      side,
      order_type: this.mapOrderType(raw.type),
      status,
      filled_qty: filledQty,
      filled_avg_price: avgPrice,
      submitted_at: raw.time ? new Date(raw.time).toISOString() : new Date().toISOString(),
      filled_at: raw.updateTime ? new Date(raw.updateTime).toISOString() : null,
      limit_price: raw.price ? parseFloat(raw.price) : null
    };
  }

  private mapOrderStatus(status: string): 'pending' | 'filled' | 'canceled' {
    const normalized = status?.toUpperCase() || '';
    switch (normalized) {
      case 'NEW':
      case 'PARTIALLY_FILLED':
      case 'PENDING_CANCEL':
        return 'pending';
      case 'FILLED':
        return 'filled';
      case 'CANCELED':
      case 'REJECTED':
      case 'EXPIRED':
      default:
        return 'canceled';
    }
  }

  private mapOrderType(type: string): 'market' | 'limit' {
    return type?.toUpperCase() === 'LIMIT' ? 'limit' : 'market';
  }

  private calculateAveragePrice(order: any): number {
    if (order.cummulativeQuoteQty && order.executedQty && parseFloat(order.executedQty) > 0) {
      return parseFloat(order.cummulativeQuoteQty) / parseFloat(order.executedQty);
    }
    if (order.fills && order.fills.length > 0) {
      const totalQuote = order.fills.reduce(
        (sum: number, fill: any) => sum + parseFloat(fill.price) * parseFloat(fill.qty),
        0
      );
      const totalQty = order.fills.reduce(
        (sum: number, fill: any) => sum + parseFloat(fill.qty),
        0
      );
      return totalQty > 0 ? totalQuote / totalQty : 0;
    }
    return order.price ? parseFloat(order.price) : 0;
  }

  private getCryptoImage(asset: string): string {
    const key = asset.toUpperCase();
    const images: Record<string, string> = {
      BTC: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
      ETH: 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
      BNB: 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
      SOL: 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
      ADA: 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
      XRP: 'https://assets.coingecko.com/coins/images/44/large/xrp-symbol-white-128.png',
      DOGE: 'https://assets.coingecko.com/coins/images/5/large/dogecoin.png'
    };
    return images[key] || '';
  }
}