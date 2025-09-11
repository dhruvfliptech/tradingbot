import { Account, Order, Position, CryptoData } from '../types/trading';
import { apiKeysService } from './apiKeysService';

class AlpacaService {
  private apiKey: string;
  private secretKey: string;
  private baseUrl: string;
  private headers: Record<string, string>;

  constructor() {
    this.apiKey = import.meta.env.VITE_ALPACA_API_KEY || '';
    this.secretKey = import.meta.env.VITE_ALPACA_SECRET_KEY || '';
    this.baseUrl = 'https://paper-api.alpaca.markets/v2';
    
    // Initialize with environment variables, will be updated dynamically
    this.headers = {
      'APCA-API-KEY-ID': this.apiKey,
      'APCA-API-SECRET-KEY': this.secretKey,
      'Content-Type': 'application/json',
    };

    this.initializeApiKeys();
  }

  private async initializeApiKeys() {
    try {
      const storedApiKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'api_key');
      const storedSecretKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'secret_key');
      
      if (storedApiKey && storedSecretKey) {
        this.apiKey = storedApiKey;
        this.secretKey = storedSecretKey;
        console.log('✅ Alpaca API keys loaded from stored keys');
      } else if (this.apiKey && this.secretKey) {
        console.log('📋 Using Alpaca API keys from environment variables');
      } else {
        console.warn('⚠️ No Alpaca API keys found in stored keys or environment');
      }
    } catch (error) {
      console.warn('⚠️ Failed to load Alpaca API keys from stored keys, using environment fallback');
    }
  }

  private async getHeaders(): Promise<Record<string, string>> {
    // Always try to get the latest API keys
    const apiKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'api_key');
    const secretKey = await apiKeysService.getApiKeyWithFallback('alpaca', 'secret_key');
    
    return {
      'APCA-API-KEY-ID': apiKey || this.apiKey,
      'APCA-API-SECRET-KEY': secretKey || this.secretKey,
      'Content-Type': 'application/json',
    };
  }

  private async makeRequest(endpoint: string, options: RequestInit = {}): Promise<any> {
    try {
      // Get current headers with latest API keys
      const headers = await this.getHeaders();
      
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

      return await response.json();
    } catch (error) {
      console.error('Alpaca API request error:', error);
      throw error;
    }
  }

  async getAccount(): Promise<Account> {
    try {
      const accountData = await this.makeRequest('/account');
      
      return {
        id: accountData.id,
        balance_usd: parseFloat(accountData.cash),
        balance_btc: 0, // Alpaca doesn't track BTC balance separately
        portfolio_value: parseFloat(accountData.portfolio_value),
        available_balance: parseFloat(accountData.buying_power),
        total_trades: 0, // We'll track this separately
      };
    } catch (error) {
      console.error('Error fetching Alpaca account:', error);
      // Return fallback account data
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
      const positions = await this.makeRequest('/positions');
      
      return positions.map((position: any) => ({
        symbol: position.symbol,
        qty: position.qty,
        market_value: parseFloat(position.market_value),
        cost_basis: parseFloat(position.cost_basis),
        unrealized_pl: parseFloat(position.unrealized_pl),
        unrealized_plpc: parseFloat(position.unrealized_plpc),
        side: position.side,
        name: position.symbol, // Alpaca doesn't provide full names
        image: this.getCryptoImage(position.symbol),
      }));
    } catch (error) {
      console.error('Error fetching Alpaca positions:', error);
      return [];
    }
  }

  async getOrders(): Promise<Order[]> {
    try {
      const orders = await this.makeRequest('/orders?status=all&limit=50');
      
      return orders.map((order: any) => ({
        id: order.id,
        symbol: order.symbol,
        qty: order.qty,
        side: order.side,
        order_type: order.order_type,
        status: order.status,
        filled_qty: order.filled_qty || '0',
        filled_avg_price: parseFloat(order.filled_avg_price || '0'),
        submitted_at: order.submitted_at,
        filled_at: order.filled_at,
        limit_price: order.limit_price ? parseFloat(order.limit_price) : null,
      }));
    } catch (error) {
      console.error('Error fetching Alpaca orders:', error);
      return [];
    }
  }

  async placeOrder(orderData: Partial<Order>): Promise<Order> {
    try {
      const alpacaOrder = {
        symbol: orderData.symbol,
        qty: orderData.qty,
        side: orderData.side,
        type: orderData.order_type,
        time_in_force: 'gtc',
        ...(orderData.limit_price && { limit_price: orderData.limit_price.toString() }),
      };

      const order = await this.makeRequest('/orders', {
        method: 'POST',
        body: JSON.stringify(alpacaOrder),
      });

      return {
        id: order.id,
        symbol: order.symbol,
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
    } catch (error) {
      console.error('Error placing Alpaca order:', error);
      throw error;
    }
  }

  async getCryptoData(symbols: string[] = ['BTCUSD', 'ETHUSD', 'ADAUSD', 'SOLUSD']): Promise<CryptoData[]> {
    try {
      // Get latest crypto bars (1 minute timeframe)
      const symbolsParam = symbols.join(',');
      const barsData = await this.makeRequest(`/crypto/bars?symbols=${symbolsParam}&timeframe=1Min&limit=1`);
      
      const cryptoData: CryptoData[] = [];
      
      for (const symbol of symbols) {
        const bars = barsData.bars[symbol];
        if (bars && bars.length > 0) {
          const latestBar = bars[0];
          const price = latestBar.c; // Close price
          const change = latestBar.c - latestBar.o; // Close - Open
          const changePercent = ((change / latestBar.o) * 100);
          
          cryptoData.push({
            id: symbol.toLowerCase(),
            symbol: symbol.replace('USD', ''),
            name: this.getCryptoName(symbol),
            image: this.getCryptoImage(symbol),
            price: price,
            change: change,
            changePercent: changePercent,
            volume: latestBar.v,
            high: latestBar.h,
            low: latestBar.l,
            market_cap: 0, // Alpaca doesn't provide market cap
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
      // Return fallback data
      return this.getFallbackCryptoData();
    }
  }

  private getCryptoName(symbol: string): string {
    const names: Record<string, string> = {
      'BTCUSD': 'Bitcoin',
      'ETHUSD': 'Ethereum',
      'ADAUSD': 'Cardano',
      'SOLUSD': 'Solana',
      'BNBUSD': 'BNB',
    };
    return names[symbol] || symbol;
  }

  private getCryptoImage(symbol: string): string {
    const images: Record<string, string> = {
      'BTCUSD': 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
      'BTC': 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
      'ETHUSD': 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
      'ETH': 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
      'ADAUSD': 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
      'ADA': 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
      'SOLUSD': 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
      'SOL': 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
      'BNBUSD': 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
      'BNB': 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
    };
    return images[symbol] || '';
  }

  private getCryptoRank(symbol: string): number {
    const ranks: Record<string, number> = {
      'BTCUSD': 1,
      'ETHUSD': 2,
      'BNBUSD': 4,
      'SOLUSD': 5,
      'ADAUSD': 8,
    };
    return ranks[symbol] || 999;
  }

  private getFallbackCryptoData(): CryptoData[] {
    return [
      {
        id: 'bitcoin',
        symbol: 'BTC',
        name: 'Bitcoin',
        image: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
        price: 43250,
        change: 1250.50,
        changePercent: 2.98,
        volume: 28500000000,
        high: 44100,
        low: 42800,
        market_cap: 850000000000,
        market_cap_rank: 1,
        price_change_24h: 1250.50,
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
        change: -45.20,
        changePercent: -1.68,
        volume: 15200000000,
        high: 2720,
        low: 2580,
        market_cap: 320000000000,
        market_cap_rank: 2,
        price_change_24h: -45.20,
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

export const alpacaService = new AlpacaService();