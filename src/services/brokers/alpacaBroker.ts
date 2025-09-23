import { Account, CryptoData, Order, Position } from '../../types/trading';
import { apiKeysService } from '../apiKeysService';
import { PlaceOrderParams, TradingBroker } from './types';

class AlpacaBroker implements TradingBroker {
  readonly id = 'alpaca' as const;

  readonly metadata = {
    id: this.id,
    label: 'Alpaca',
    description: 'Paper trading equities & crypto via Alpaca Markets',
    features: {
      paperTrading: true,
      liveTrading: false,
      margin: false,
      supportsCrypto: true,
      supportsEquities: true,
    },
    baseCurrency: 'USD',
    docsUrl: 'https://alpaca.markets/docs/api-references/trading-api/crypto-trading/',
  } as const;

  private apiKey: string;
  private secretKey: string;
  private baseUrl: string;

  constructor() {
    this.apiKey = import.meta.env.VITE_ALPACA_API_KEY || '';
    this.secretKey = import.meta.env.VITE_ALPACA_SECRET_KEY || '';
    const rawBaseUrl = (import.meta.env.VITE_ALPACA_BASE_URL || 'https://paper-api.alpaca.markets/v2').replace(/\/$/, '');
    this.baseUrl = /\/v2$/i.test(rawBaseUrl) ? rawBaseUrl : `${rawBaseUrl}/v2`;

    void this.initializeApiKeys();
  }

  private async initializeApiKeys(): Promise<void> {
    try {
      const storedApiKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'api_key');
      const storedSecretKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'secret_key');

      if (storedApiKey) this.apiKey = storedApiKey;
      if (storedSecretKey) this.secretKey = storedSecretKey;

      if (storedApiKey && storedSecretKey) {
        console.log('✅ Alpaca API keys loaded from secure storage');
      } else if (this.apiKey && this.secretKey) {
        console.log('📋 Using Alpaca API keys from environment variables');
      } else {
        console.warn('⚠️ No Alpaca API keys found in storage or environment');
        // Disable Alpaca integration if keys are missing
        this.apiKey = '';
        this.secretKey = '';
      }
    } catch (error) {
      console.warn('⚠️ Failed to load Alpaca API keys from storage:', error);
    }
  }

  private normalizeSymbolInternal(symbol: string): string {
    if (!symbol) return symbol;
    let normalized = symbol.trim().toUpperCase();
    if (normalized.endsWith('USDT')) normalized = normalized.replace(/USDT$/, 'USD');
    if (!normalized.endsWith('USD') && normalized.length <= 6) normalized = `${normalized}USD`;
    return normalized;
  }

  normalizeSymbol(symbol: string): string {
    return this.normalizeSymbolInternal(symbol);
  }

  private isCryptoSymbol(symbol: string): boolean {
    // Common crypto symbols that Alpaca supports
    const cryptoSymbols = ['BTC', 'ETH', 'ADA', 'SOL', 'DOGE', 'MATIC', 'AVAX', 'DOT', 'LINK', 'UNI', 'ATOM', 'NEAR'];
    const normalizedSymbol = symbol.toUpperCase();
    
    // Check if symbol contains any crypto currency prefix or ends with USD (most Alpaca crypto symbols)
    return cryptoSymbols.some(crypto => 
      normalizedSymbol.startsWith(crypto) || 
      normalizedSymbol.includes(crypto)
    ) || normalizedSymbol.endsWith('USD');
  }

  toBrokerSymbol(symbol: string): string {
    return this.normalizeSymbolInternal(symbol);
  }

  fromBrokerSymbol(symbol: string): string {
    return this.normalizeSymbolInternal(symbol);
  }

  private async getHeaders(): Promise<Record<string, string>> {
    const apiKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'api_key');
    const secretKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'secret_key');

    return {
      'APCA-API-KEY-ID': apiKey || this.apiKey,
      'APCA-API-SECRET-KEY': secretKey || this.secretKey,
      'Content-Type': 'application/json',
    };
  }

  private async request<T = any>(endpoint: string, options: RequestInit = {}): Promise<T> {
    const headers = await this.getHeaders();
    
    // Check if we have valid credentials
    if (!headers['APCA-API-KEY-ID'] || !headers['APCA-API-SECRET-KEY']) {
      throw new Error('Alpaca API credentials not configured. Please add your API key and secret key to the environment variables.');
    }

    const response = await fetch(`${this.baseUrl}${endpoint}`, {
      ...options,
      headers: {
        ...headers,
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorText = await response.text();
      console.error(`Alpaca API error! status: ${response.status}, message: ${errorText}`);
      throw new Error(`Alpaca API Error: ${response.status}`);
    }

    return response.json();
  }

  async testConnection(): Promise<boolean> {
    try {
      await this.request('/account');
      return true;
    } catch (error) {
      console.warn('Alpaca connection test failed:', error);
      return false;
    }
  }

  async getAccount(): Promise<Account> {
    try {
      const accountData = await this.request<any>('/account');
      return {
        id: accountData.id,
        balance_usd: parseFloat(accountData.cash),
        balance_btc: 0,
        portfolio_value: parseFloat(accountData.portfolio_value),
        available_balance: parseFloat(accountData.buying_power),
        total_trades: 0,
      };
    } catch (error) {
      console.error('Error fetching Alpaca account:', error);
      return {
        id: 'alpaca-demo',
        balance_usd: 100000,
        balance_btc: 0,
        portfolio_value: 100000,
        available_balance: 100000,
        total_trades: 0,
      };
    }
  }

  async getPositions(): Promise<Position[]> {
    try {
      const positions = await this.request<any[]>('/positions');
      return positions.map((position) => ({
        symbol: this.fromBrokerSymbol(position.symbol),
        qty: position.qty,
        market_value: parseFloat(position.market_value),
        cost_basis: parseFloat(position.cost_basis),
        unrealized_pl: parseFloat(position.unrealized_pl),
        unrealized_plpc: parseFloat(position.unrealized_plpc),
        side: position.side,
        name: position.symbol,
        image: this.getCryptoImage(position.symbol),
      }));
    } catch (error) {
      console.error('Error fetching Alpaca positions:', error);
      return [];
    }
  }

  async getOrders(): Promise<Order[]> {
    try {
      const orders = await this.request<any[]>('/orders?status=all&limit=50');
      return orders.map((order) => this.mapOrder(order));
    } catch (error) {
      console.error('Error fetching Alpaca orders:', error);
      return [];
    }
  }

  async placeOrder(orderData: PlaceOrderParams): Promise<Order> {
    try {
      // For crypto orders, Alpaca requires time_in_force to be 'gtc' (Good 'Til Canceled)
      // Other values like 'day' are not supported for crypto trading
      const isCryptoSymbol = this.isCryptoSymbol(orderData.symbol);
      const timeInForce = isCryptoSymbol ? 'gtc' : (orderData.time_in_force || 'gtc');
      
      const payload = {
        symbol: this.toBrokerSymbol(orderData.symbol),
        qty: typeof orderData.qty === 'number' ? orderData.qty.toString() : orderData.qty,
        side: orderData.side,
        type: orderData.order_type,
        time_in_force: timeInForce,
        ...(orderData.limit_price ? { limit_price: orderData.limit_price.toString() } : {}),
        ...(orderData.client_order_id ? { client_order_id: orderData.client_order_id } : {}),
      };

      const order = await this.request<any>('/orders', {
        method: 'POST',
        body: JSON.stringify(payload),
      });

      return this.mapOrder(order);
    } catch (error) {
      console.error('Error placing Alpaca order:', error);
      throw error;
    }
  }

  async getCryptoData(symbols: string[] = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD']): Promise<CryptoData[]> {
    try {
      const normalized = symbols.map((symbol) => this.toBrokerSymbol(symbol));
      const symbolsParam = normalized.join(',');
      const barsData = await this.request<any>(`/crypto/bars?symbols=${symbolsParam}&timeframe=1Min&limit=1`);

      const cryptoData: CryptoData[] = [];
      for (const symbol of normalized) {
        const bars = barsData.bars[symbol];
        if (bars && bars.length > 0) {
          const latestBar = bars[0];
          const price = latestBar.c;
          const change = latestBar.c - latestBar.o;
          const changePercent = (change / latestBar.o) * 100;

          cryptoData.push({
            id: symbol.toLowerCase(),
            symbol: this.fromBrokerSymbol(symbol).replace('USD', ''),
            name: this.getCryptoName(symbol),
            image: this.getCryptoImage(symbol),
            price,
            change,
            changePercent,
            volume: latestBar.v,
            high: latestBar.h,
            low: latestBar.l,
            market_cap: 0,
            market_cap_rank: this.getCryptoRank(symbol),
            price_change_24h: change,
            price_change_percentage_24h: changePercent,
            circulating_supply: 0,
            total_supply: 0,
            max_supply: 0,
            ath: 0,
            ath_change_percentage: 0,
            last_updated: new Date().toISOString(),
          });
        }
      }

      return cryptoData;
    } catch (error) {
      console.error('Error fetching Alpaca crypto data:', error);
      return this.getFallbackCryptoData();
    }
  }

  private mapOrder(order: any): Order {
    return {
      id: order.id,
      symbol: this.fromBrokerSymbol(order.symbol),
      qty: order.qty,
      side: order.side,
      order_type: order.order_type,
      status: order.status,
      filled_qty: order.filled_qty || '0',
      filled_avg_price: parseFloat(order.filled_avg_price || '0'),
      submitted_at: order.submitted_at,
      filled_at: order.filled_at,
      limit_price: order.limit_price ? parseFloat(order.limit_price) : null,
    };
  }

  private getCryptoName(symbol: string): string {
    const names: Record<string, string> = {
      BTCUSD: 'Bitcoin',
      ETHUSD: 'Ethereum',
      ADAUSD: 'Cardano',
      SOLUSD: 'Solana',
      BNBUSD: 'BNB',
    };
    return names[this.normalizeSymbolInternal(symbol)] || symbol;
  }

  private getCryptoImage(symbol: string): string {
    const key = this.normalizeSymbolInternal(symbol);
    const images: Record<string, string> = {
      BTCUSD: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
      BTC: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
      ETHUSD: 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
      ETH: 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
      ADAUSD: 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
      ADA: 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
      SOLUSD: 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
      SOL: 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
      BNBUSD: 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
      BNB: 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
    };
    return images[key] || '';
  }

  private getCryptoRank(symbol: string): number {
    const ranks: Record<string, number> = {
      BTCUSD: 1,
      ETHUSD: 2,
      BNBUSD: 4,
      SOLUSD: 5,
      ADAUSD: 8,
    };
    return ranks[this.normalizeSymbolInternal(symbol)] || 999;
  }

  private getFallbackCryptoData(): CryptoData[] {
    return [
      {
        id: 'bitcoin',
        symbol: 'BTC',
        name: 'Bitcoin',
        image: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
        price: 43250,
        change: 1250.5,
        changePercent: 2.98,
        volume: 28500000000,
        high: 44100,
        low: 42800,
        market_cap: 850000000000,
        market_cap_rank: 1,
        price_change_24h: 1250.5,
        price_change_percentage_24h: 2.98,
        circulating_supply: 19600000,
        total_supply: 21000000,
        max_supply: 21000000,
        ath: 69000,
        ath_change_percentage: -37.3,
        last_updated: new Date().toISOString(),
      },
      {
        id: 'ethereum',
        symbol: 'ETH',
        name: 'Ethereum',
        image: 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
        price: 2650,
        change: -45.2,
        changePercent: -1.68,
        volume: 15200000000,
        high: 2720,
        low: 2580,
        market_cap: 320000000000,
        market_cap_rank: 2,
        price_change_24h: -45.2,
        price_change_percentage_24h: -1.68,
        circulating_supply: 120000000,
        total_supply: 120000000,
        max_supply: 0,
        ath: 4878,
        ath_change_percentage: -45.7,
        last_updated: new Date().toISOString(),
      },
    ];
  }
}

export const alpacaBroker = new AlpacaBroker();
