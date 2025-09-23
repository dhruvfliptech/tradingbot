import CryptoJS from 'crypto-js';
import { Account, CryptoData, Order, Position } from '../../types/trading';
import { apiKeysService } from '../apiKeysService';
import { PlaceOrderParams, TradingBroker } from './types';

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
  fills?: Array<{ price: string; qty: string; commission: string; commissionAsset: string }>;
  updateTime: number;
  transactTime?: number;
}

class BinanceBroker implements TradingBroker {
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

  private apiKey: string;
  private secretKey: string;
  private baseUrl: string;
  private recvWindow = Number(import.meta.env.VITE_BINANCE_RECV_WINDOW || 5000);
  private useProxy: boolean;
  private serverTimeOffsetMs: number = 0; // serverTime - Date.now()

  constructor() {
    this.apiKey = import.meta.env.VITE_BINANCE_API_KEY || '';
    this.secretKey = import.meta.env.VITE_BINANCE_SECRET_KEY || '';
    // Force Binance US by default; allow explicit override via VITE_BINANCE_BASE_URL if needed
    this.baseUrl = (import.meta.env.VITE_BINANCE_BASE_URL || 'https://api.binance.us').replace(/\/$/, '');
    
    // Opt-in proxy usage via env flag only (default false)
    this.useProxy = (import.meta.env.VITE_BINANCE_USE_PROXY || '').toString() === 'true';

    void this.initializeApiKeys();
    // Prime time offset in background (best-effort)
    void this.syncServerTime().catch(() => {});
  }

  private async initializeApiKeys(): Promise<void> {
    try {
      const storedApiKey = await apiKeysService.getApiKeyWithFallback('binance', 'api_key');
      const storedSecretKey = await apiKeysService.getApiKeyWithFallback('binance', 'secret_key');

      if (storedApiKey) this.apiKey = storedApiKey;
      if (storedSecretKey) this.secretKey = storedSecretKey;

      if (storedApiKey && storedSecretKey) {
        console.log('✅ Binance API keys loaded from secure storage');
      } else if (this.apiKey && this.secretKey) {
        console.log('📋 Using Binance API keys from environment variables');
      } else {
        console.warn('⚠️ No Binance API keys found in storage or environment');
      }
    } catch (error) {
      console.warn('⚠️ Failed to load Binance API keys from storage:', error);
    }
  }

  private async syncServerTime(): Promise<void> {
    try {
      const res = await fetch(`${this.baseUrl}/api/v3/time`);
      if (!res.ok) return;
      const data = await res.json();
      if (data && typeof data.serverTime === 'number') {
        this.serverTimeOffsetMs = data.serverTime - Date.now();
      }
    } catch {
      // noop
    }
  }

  private normalizeSymbolInternal(symbol: string): string {
    if (!symbol) return symbol;
    const normalized = symbol.trim().toUpperCase();
    if (normalized.endsWith('USD') || normalized.endsWith('USDT')) return normalized;
    // Default to USD on Binance US
    return `${normalized}USD`;
  }

  normalizeSymbol(symbol: string): string {
    return this.fromBrokerSymbol(this.normalizeSymbolInternal(symbol));
  }

  toBrokerSymbol(symbol: string): string {
    const base = symbol.trim().toUpperCase();
    if (base.endsWith('USD') || base.endsWith('USDT')) return base;
    return `${base}USD`;
  }

  fromBrokerSymbol(symbol: string): string {
    if (!symbol) return symbol;
    return symbol;
  }

  private async getCredentials(): Promise<{ apiKey: string; secretKey: string }> {
    const apiKey = await apiKeysService.getApiKeyWithFallback('binance', 'api_key');
    const secretKey = await apiKeysService.getApiKeyWithFallback('binance', 'secret_key');
    return {
      apiKey: apiKey || this.apiKey,
      secretKey: secretKey || this.secretKey,
    };
  }

  private async request<T = any>(path: string, options: {
    method?: 'GET' | 'POST' | 'DELETE';
    params?: Record<string, string | number | undefined | null>;
    signed?: boolean;
    timeoutMs?: number;
    retryOnTimeSkew?: boolean;
  } = {}): Promise<T> {
    const method = options.method || 'GET';
    
    // If using proxy, send the request through Netlify function
    if (this.useProxy) {
      return this.requestViaProxy<T>(path, options);
    }
    
    // Direct API call (for server-side usage)
    return this.requestDirect<T>(path, options);
  }

  private async requestViaProxy<T = any>(path: string, options: {
    method?: 'GET' | 'POST' | 'DELETE';
    params?: Record<string, string | number | undefined | null>;
    signed?: boolean;
    timeoutMs?: number;
  } = {}): Promise<T> {
    const method = options.method || 'GET';
    const params = new URLSearchParams();

    if (options.params) {
      for (const [key, value] of Object.entries(options.params)) {
        if (value === undefined || value === null || value === '') continue;
        params.append(key, String(value));
      }
    }

    // Build proxy URL
    const proxyUrl = `/.netlify/functions/binance-proxy${path}`;
    let url = proxyUrl;
    let body: string | undefined;

    let headers: Record<string, string> = {};

    // For signed requests, send API key in header
    if (options.signed) {
      const { apiKey } = await this.getCredentials();
      if (!apiKey) {
        throw new Error('Binance API key is not configured');
      }
      headers['X-MBX-APIKEY'] = apiKey;
    }

    if (method === 'POST') {
      // For POST requests, send all parameters in the body, not in URL
      const queryString = params.toString();
      if (queryString) {
        body = queryString;
        headers['Content-Type'] = 'application/x-www-form-urlencoded';
      }
    } else {
      // For GET/DELETE requests, send parameters in URL
      const queryString = params.toString();
      if (queryString) {
        url = `${proxyUrl}?${queryString}`;
      }
    }

    const controller = new AbortController();
    const timeout = options.timeoutMs || 10000;
    const timer = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body,
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`Binance API error ${response.status}: ${errorText}`);
        throw new Error(`Binance API error ${response.status}: ${errorText}`);
      }

      if (response.status === 204) {
        return {} as T;
      }

      return await response.json();
    } finally {
      clearTimeout(timer);
    }
  }

  private async requestDirect<T = any>(path: string, options: {
    method?: 'GET' | 'POST' | 'DELETE';
    params?: Record<string, string | number | undefined | null>;
    signed?: boolean;
    timeoutMs?: number;
  } = {}): Promise<T> {
    const method = options.method || 'GET';
    const params = new URLSearchParams();

    if (options.params) {
      for (const [key, value] of Object.entries(options.params)) {
        if (value === undefined || value === null || value === '') continue;
        params.append(key, String(value));
      }
    }

    let headers: Record<string, string> = {};
    let body: string | undefined;
    let url = `${this.baseUrl}${path}`;

    if (options.signed) {
      const { apiKey, secretKey } = await this.getCredentials();
      if (!apiKey || !secretKey) {
        throw new Error('Binance API keys are not configured');
      }

      // Use server time with offset to avoid -1021 errors
      const now = Date.now() + this.serverTimeOffsetMs;
      params.set('timestamp', Math.floor(now).toString());
      if (!params.has('recvWindow')) {
        params.set('recvWindow', this.recvWindow.toString());
      }

      const query = params.toString();
      const signature = CryptoJS.HmacSHA256(query, secretKey).toString(CryptoJS.enc.Hex);
      params.set('signature', signature);
      headers['X-MBX-APIKEY'] = apiKey;

      if (method === 'GET' || method === 'DELETE') {
        url += `?${params.toString()}`;
      } else {
        body = params.toString();
        headers['Content-Type'] = 'application/x-www-form-urlencoded';
      }
    } else {
      const query = params.toString();
      if (query) {
        url += `?${query}`;
      }
      if (method !== 'GET' && method !== 'DELETE' && query) {
        body = query;
        headers['Content-Type'] = 'application/x-www-form-urlencoded';
      }
    }

    const controller = new AbortController();
    const timeout = options.timeoutMs || 10000;
    const timer = setTimeout(() => controller.abort(), timeout);

    try {
      const response = await fetch(url, {
        method,
        headers,
        body,
        signal: controller.signal,
      });

      if (!response.ok) {
        // Try to parse structured error
        let errorBody: any = null;
        try {
          errorBody = await response.json();
        } catch {
          // fallback to text
          errorBody = { message: await response.text() };
        }

        // Handle time skew error (-1021)
        if (errorBody && (errorBody.code === -1021 || String(errorBody.message).includes('timestamp'))) {
          if (options.signed && options.retryOnTimeSkew !== false) {
            await this.syncServerTime();
            // Retry once with updated offset
            return await this.request<T>(path, { ...options, retryOnTimeSkew: false });
          }
        }

        const code = errorBody?.code || response.status;
        const msg = errorBody?.msg || errorBody?.message || 'Unknown error';
        const composed = `Binance API error ${code}: ${msg}`;
        console.error(composed);
        throw new Error(composed);
      }

      if (response.status === 204) {
        return {} as T;
      }

      return await response.json();
    } finally {
      clearTimeout(timer);
    }
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.request('/api/v3/account', { signed: true });
      return true;
    } catch (error) {
      console.warn('Binance connection test failed:', error);
      return false;
    }
  }

  async getAccount(): Promise<Account> {
    try {
      const accountData = await this.request<any>('/api/v3/account', { signed: true });
      const balances = accountData.balances || [];
      const filtered = balances.filter((b: any) => this.hasBalance(b));
      const assets = filtered.map((b: any) => b.asset);
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
          usdCashFree = freeQty; // true USD cash on Binance US
        }
      }

      return {
        id: accountData.accountId ? String(accountData.accountId) : 'binance-account',
        balance_usd: portfolioValue,
        balance_btc: btcBalance,
        portfolio_value: portfolioValue,
        // Show pure USD cash for available balance to match requested semantics
        available_balance: usdCashFree || availableBalance,
        total_trades: accountData.totalTradeCount || 0,
      };
    } catch (error) {
      console.error('Error fetching Binance account:', error);
      return {
        id: 'binance-demo',
        balance_usd: 0,
        balance_btc: 0,
        portfolio_value: 0,
        available_balance: 0,
        total_trades: 0,
      };
    }
  }

  async getPositions(): Promise<Position[]> {
    try {
      const accountData = await this.request<any>('/api/v3/account', { signed: true });
      const balances = (accountData.balances || []).filter((b: any) => this.hasBalance(b));
      const assets = balances.map((b: any) => b.asset);
      const priceMap = await this.getPriceMap(assets);

      return balances.map((bal: any) => {
        const asset = bal.asset;
        const totalQty = parseFloat(bal.free) + parseFloat(bal.locked);
        const price = priceMap[asset] || 0;
        const marketValue = totalQty * price;

        return {
          symbol: this.fromBrokerSymbol(`${asset}USD`),
          qty: totalQty.toString(),
          market_value: marketValue,
          cost_basis: marketValue, // Binance spot does not expose per-position cost basis
          unrealized_pl: 0,
          unrealized_plpc: 0,
          side: 'long',
          name: asset,
          image: this.getCryptoImage(asset),
        };
      });
    } catch (error) {
      console.error('Error fetching Binance positions:', error);
      return [];
    }
  }

  async getOrders(): Promise<Order[]> {
    const symbols = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'ADAUSD'];
    const orders: Order[] = [];

    for (const symbol of symbols) {
      try {
        const response = await this.request<any[]>('/api/v3/allOrders', {
          signed: true,
          params: { symbol, limit: 20 },
        });

        for (const raw of response) {
          orders.push(this.mapOrder(raw));
        }
      } catch (error) {
        console.warn(`Failed to fetch Binance orders for ${symbol}:`, error);
      }
    }

    // Sort by update time descending and remove duplicates by order id
    const unique: Record<number, Order> = {};
    for (const order of orders) {
      unique[Number(order.id)] = order;
    }

    return Object.values(unique).sort((a, b) => {
      const aTime = a.filled_at || a.submitted_at || '';
      const bTime = b.filled_at || b.submitted_at || '';
      return new Date(bTime).getTime() - new Date(aTime).getTime();
    });
  }

  async placeOrder(orderData: PlaceOrderParams): Promise<Order> {
    const symbol = this.toBrokerSymbol(orderData.symbol);
    const qty = typeof orderData.qty === 'number' ? orderData.qty : Number(orderData.qty);

    if (!qty || Number.isNaN(qty) || qty <= 0) {
      throw new Error('Invalid quantity for Binance order');
    }

    const params: Record<string, string | number> = {
      symbol,
      side: orderData.side.toUpperCase(),
      type: orderData.order_type === 'limit' ? 'LIMIT' : 'MARKET',
      quantity: this.formatQuantity(qty),
      newOrderRespType: 'FULL',
    };

    if (params.type === 'LIMIT') {
      params.timeInForce = (orderData.time_in_force || 'GTC').toUpperCase();
      if (orderData.limit_price) {
        params.price = this.formatPrice(orderData.limit_price);
      } else {
        throw new Error('Limit orders require a limit_price');
      }
    }

    if (orderData.client_order_id) {
      params.newClientOrderId = orderData.client_order_id;
    }

    const response = await this.request<BinanceOrderResponse>('/api/v3/order', {
      method: 'POST',
      signed: true,
      params,
    });

    return this.mapOrder(response);
  }

  private hasBalance(balance: { free: string; locked: string; asset: string }): boolean {
    const free = parseFloat(balance.free);
    const locked = parseFloat(balance.locked);
    return (free + locked) > 0.0000001;
  }

  private async getPriceMap(assets: string[]): Promise<Record<string, number>> {
    const uniqueAssets = Array.from(new Set(assets)).filter((asset) => asset !== 'USDT' && asset !== 'BUSD' && asset !== 'USD');
    if (uniqueAssets.length === 0) {
      return {};
    }

    try {
      const symbols = uniqueAssets.map((asset) => `${asset}USD`);
      const response = await this.request<any[]>('/api/v3/ticker/price', {
        params: { symbols: JSON.stringify(symbols) },
      });

      const map: Record<string, number> = {};
      for (const ticker of response) {
        const asset = ticker.symbol.replace(/USDT$/, '');
        map[asset] = parseFloat(ticker.price);
      }
      map.USDT = 1;
      map.BUSD = 1;
      map.USD = 1;
      return map;
    } catch (error) {
      console.warn('Failed to fetch Binance price map:', error);
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

  private mapOrder(raw: any): Order {
    const status = this.mapStatus(raw.status);
    const side = (raw.side || '').toLowerCase() as 'buy' | 'sell';
    const filledQty = raw.executedQty || '0';
    const avgPrice = this.calculateAveragePrice(raw);

    return {
      id: String(raw.orderId || raw.id),
      symbol: this.fromBrokerSymbol(raw.symbol),
      qty: raw.origQty || raw.quantity || '0',
      side,
      order_type: this.mapOrderType(raw.type),
      status,
      filled_qty: filledQty,
      filled_avg_price: avgPrice,
      submitted_at: raw.time ? new Date(raw.time).toISOString() : (raw.transactTime ? new Date(raw.transactTime).toISOString() : new Date().toISOString()),
      filled_at: raw.updateTime ? new Date(raw.updateTime).toISOString() : null,
      limit_price: raw.price ? parseFloat(raw.price) : null,
    };
  }

  private mapStatus(status: string): 'pending' | 'filled' | 'canceled' {
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

  private calculateAveragePrice(order: BinanceOrderResponse): number {
    if (order.cummulativeQuoteQty && order.executedQty && parseFloat(order.executedQty) > 0) {
      return parseFloat(order.cummulativeQuoteQty) / parseFloat(order.executedQty);
    }
    if (order.fills && order.fills.length > 0) {
      const totalQuote = order.fills.reduce((sum, fill) => sum + parseFloat(fill.price) * parseFloat(fill.qty), 0);
      const totalQty = order.fills.reduce((sum, fill) => sum + parseFloat(fill.qty), 0);
      return totalQty > 0 ? totalQuote / totalQty : 0;
    }
    return order.price ? parseFloat(order.price) : 0;
  }

  private formatQuantity(quantity: number): string {
    // Binance supports up to 8 decimal places on most spot pairs
    return quantity.toFixed(8).replace(/\.0+$/, '').replace(/0+$/, '').replace(/\.$/, '');
  }

  private formatPrice(price: number): string {
    return price.toFixed(8).replace(/\.0+$/, '').replace(/0+$/, '').replace(/\.$/, '');
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
      DOGE: 'https://assets.coingecko.com/coins/images/5/large/dogecoin.png',
    };
    return images[key] || '';
  }

  async getCryptoData(symbols: string[] = ['BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD']): Promise<CryptoData[]> {
    try {
      const brokerSymbols = symbols.map((symbol) => this.toBrokerSymbol(symbol));
      const response = await this.request<any[]>('/api/v3/ticker/24hr', {
        params: { symbols: JSON.stringify(brokerSymbols) },
      });

      return response.map((ticker) => {
        const normalizedSymbol = this.fromBrokerSymbol(ticker.symbol);
        const symbolCode = normalizedSymbol.replace('USD', '');
        return {
          id: symbolCode.toLowerCase(),
          symbol: symbolCode,
          name: symbolCode,
          image: this.getCryptoImage(symbolCode),
          price: parseFloat(ticker.lastPrice),
          change: parseFloat(ticker.priceChange),
          changePercent: parseFloat(ticker.priceChangePercent),
          volume: parseFloat(ticker.volume),
          high: parseFloat(ticker.highPrice),
          low: parseFloat(ticker.lowPrice),
          market_cap: 0,
          market_cap_rank: 0,
          price_change_24h: parseFloat(ticker.priceChange),
          price_change_percentage_24h: parseFloat(ticker.priceChangePercent),
          circulating_supply: 0,
          total_supply: 0,
          max_supply: 0,
          ath: 0,
          ath_change_percentage: 0,
          last_updated: new Date().toISOString(),
        };
      });
    } catch (error) {
      console.error('Error fetching Binance crypto data:', error);
      return [];
    }
  }
}

export const binanceBroker = new BinanceBroker();
