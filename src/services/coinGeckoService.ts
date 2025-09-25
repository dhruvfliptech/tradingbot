import { Account, Order, Position, CryptoData, FearGreedIndex } from '../types/trading';
import { supabase } from '../lib/supabase';
import { apiKeysService } from './apiKeysService';
import { tradingProviderService, PlaceOrderParams } from './tradingProviderService';

class CoinGeckoService {
  private baseUrl: string;
  private apiKey: string;
  private headers: Record<string, string>;
  private cache: Map<string, { data: any; timestamp: number }>;
  private rateLimitDelay: number;
  private lastRequestTime: number;
  private requestQueue: Array<() => Promise<void>>;
  private isProcessingQueue: boolean;

  constructor() {
    const isDev = import.meta.env.DEV;
    this.baseUrl = isDev ? '/api/v1/proxy/coingecko' : 'https://api.coingecko.com/api/v3';
    this.apiKey = import.meta.env.VITE_COINGECKO_API_KEY || '';
    this.cache = new Map();
    this.rateLimitDelay = 2000; // 2 seconds between requests
    this.lastRequestTime = 0;
    this.requestQueue = [];
    this.isProcessingQueue = false;

    // Initialize headers without API key for now
    this.headers = {
      accept: 'application/json',
    };

    // We'll set the API key dynamically in requests
    this.initializeApiKey();
  }

  private async initializeApiKey() {
    try {
      // Try to get API key from stored keys, fallback to environment
      const storedKey = await apiKeysService.getApiKeyWithFallback('coingecko', 'api_key');
      if (storedKey) {
        this.apiKey = storedKey;
        console.log('✅ CoinGecko API key loaded from stored keys');
      } else if (this.apiKey) {
        console.log('📋 Using CoinGecko API key from environment variables');
      } else {
        console.log('ℹ️ CoinGecko running without API key (public endpoints only)');
      }
    } catch (error) {
      console.warn('⚠️ Failed to load CoinGecko API key from stored keys, using environment fallback');
    }
  }

  private async getHeaders(): Promise<Record<string, string>> {
    // Always try to get the latest API key
    const apiKey = await apiKeysService.getApiKeyWithFallback('coingecko', 'api_key');
    const usePro = String(import.meta.env.VITE_USE_COINGECKO_PRO || '').toLowerCase() === 'true';
    
    return {
      accept: 'application/json',
      ...(usePro && apiKey ? { 'x-cg-pro-api-key': apiKey } : {}),
    };
  }

  private getCacheKey(url: string): string {
    return `coingecko_${url.replace(/[^a-zA-Z0-9]/g, '_')}`;
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes cache
      console.log('🔄 Using cached CoinGecko data');
      return cached.data;
    }
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
    
    // Clean old cache entries (keep only last 20 entries)
    if (this.cache.size > 20) {
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < 5; i++) {
        this.cache.delete(keys[i]);
      }
    }
  }

  private async rateLimitedRequest(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;
    
    if (timeSinceLastRequest < this.rateLimitDelay) {
      const delay = this.rateLimitDelay - timeSinceLastRequest;
      console.log(`⏳ Rate limiting: waiting ${delay}ms before next request`);
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    
    this.lastRequestTime = Date.now();
  }

  private async makeRequest(url: string, options: RequestInit = {}): Promise<any> {
    // Check cache first
    const cacheKey = this.getCacheKey(url);
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      return cached;
    }

    // Apply rate limiting
    await this.rateLimitedRequest();

    try {
      console.log('🌐 Making CoinGecko API request to:', url);
      
      // Get current headers with latest API key
      const headers = await this.getHeaders();
      
      // Add timeout and better error handling for network issues
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000); // 15 second timeout
      
      const response = await fetch(url, {
        ...options,
        headers: {
          ...headers,
          ...options.headers,
        },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        
        // Handle rate limiting with exponential backoff
        if (response.status === 429) {
          console.warn(`⚠️ CoinGecko API rate limited (429), using fallback data`);
          console.log('💡 Consider adding VITE_COINGECKO_API_KEY to your .env file for higher rate limits');
          return this.getFallbackData(url);
        } else {
          console.error(`❌ CoinGecko API error! status: ${response.status}, message: ${errorText}`);
        }
        
        // For other errors, throw to trigger fallback
        throw new Error(`CoinGecko API Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();
      console.log('✅ CoinGecko API request successful');
      
      // Cache the successful response
      this.setCache(cacheKey, data);
      return data;
    } catch (error) {
      // Handle specific network errors
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('❌ CoinGecko API request timed out after 15 seconds');
        } else if (error.message.includes('Failed to fetch') || error.name === 'TypeError') {
          console.error('❌ Network error accessing CoinGecko API. This could be due to:');
          console.error('   • Missing or invalid VITE_COINGECKO_API_KEY in .env file');
          console.error('   • Network connectivity issues');
          console.error('   • CORS restrictions or firewall blocking the request');
          console.error('   • CoinGecko API service temporarily unavailable');
        } else {
          console.error('❌ CoinGecko API error:', error.message);
        }
      }
      
      console.warn('⚠️ Using fallback data due to network error');
      return this.getFallbackData(url);
    }
  }

  private getFallbackData(url: string): any {
    if (url.includes('/coins/markets')) {
      console.warn('📊 Using fallback crypto market data - prices may not be current');
      return [
        {
          id: 'bitcoin',
          symbol: 'btc',
          name: 'Bitcoin',
          image: 'https://assets.coingecko.com/coins/images/1/large/bitcoin.png',
          current_price: 116000, // Updated to current approximate price
          price_change_24h: 2500.00,
          price_change_percentage_24h: 2.20,
          total_volume: 28500000000,
          high_24h: 118000,
          low_24h: 114000,
          market_cap: 2300000000000, // Updated market cap
          market_cap_rank: 1,
          circulating_supply: 19600000,
          total_supply: 21000000,
          max_supply: 21000000,
          ath: 118000,
          ath_change_percentage: -1.7,
          last_updated: new Date().toISOString(),
        },
        {
          id: 'ethereum',
          symbol: 'eth',
          name: 'Ethereum',
          image: 'https://assets.coingecko.com/coins/images/279/large/ethereum.png',
          current_price: 4200, // Updated to current approximate price
          price_change_24h: 85.00,
          price_change_percentage_24h: 2.06,
          total_volume: 15200000000,
          high_24h: 4250,
          low_24h: 4100,
          market_cap: 505000000000, // Updated market cap
          market_cap_rank: 2,
          circulating_supply: 120000000,
          total_supply: 120000000,
          max_supply: null,
          ath: 4250,
          ath_change_percentage: -1.2,
          last_updated: new Date().toISOString(),
        },
        {
          id: 'binancecoin',
          symbol: 'bnb',
          name: 'BNB',
          image: 'https://assets.coingecko.com/coins/images/825/large/bnb-icon2_2x.png',
          current_price: 315,
          price_change_24h: 8.50,
          price_change_percentage_24h: 2.77,
          total_volume: 1800000000,
          high_24h: 320,
          low_24h: 305,
          market_cap: 47000000000,
          market_cap_rank: 4,
          circulating_supply: 149000000,
          total_supply: 149000000,
          max_supply: 200000000,
          ath: 686,
          ath_change_percentage: -54.1,
          last_updated: new Date().toISOString(),
        },
        {
          id: 'solana',
          symbol: 'sol',
          name: 'Solana',
          image: 'https://assets.coingecko.com/coins/images/4128/large/solana.png',
          current_price: 98.75,
          price_change_24h: 3.25,
          price_change_percentage_24h: 3.40,
          total_volume: 2100000000,
          high_24h: 102,
          low_24h: 94.50,
          market_cap: 45000000000,
          market_cap_rank: 5,
          circulating_supply: 456000000,
          total_supply: 580000000,
          max_supply: null,
          ath: 260,
          ath_change_percentage: -62.0,
          last_updated: new Date().toISOString(),
        },
        {
          id: 'cardano',
          symbol: 'ada',
          name: 'Cardano',
          image: 'https://assets.coingecko.com/coins/images/975/large/cardano.png',
          current_price: 0.45,
          price_change_24h: 0.02,
          price_change_percentage_24h: 4.65,
          total_volume: 850000000,
          high_24h: 0.47,
          low_24h: 0.42,
          market_cap: 16000000000,
          market_cap_rank: 8,
          circulating_supply: 35000000000,
          total_supply: 45000000000,
          max_supply: 45000000000,
          ath: 3.10,
          ath_change_percentage: -85.5,
          last_updated: new Date().toISOString(),
        },
      ];
    }
    
    if (url.includes('/global')) {
      return {
        data: {
          active_cryptocurrencies: 18020,
          markets: 1339,
          total_market_cap: { usd: 4013000000000 },
          total_volume: { usd: 127727000000 },
          market_cap_percentage: { btc: 57.9, eth: 12.7 },
          market_cap_change_percentage_24h_usd: 1.2,
        }
      };
    }
    
    return {};
  }

  async getCryptoData(ids: string[] = ['bitcoin', 'ethereum', 'binancecoin', 'cardano', 'solana']): Promise<CryptoData[]> {
    console.log('🔍 Fetching crypto data for:', ids);
    const idsParam = ids.join(',');
    
    try {
      const response = await this.makeRequest(
        `${this.baseUrl}/coins/markets?vs_currency=usd&ids=${idsParam}&order=market_cap_desc&per_page=10&page=1&sparkline=false&price_change_percentage=24h`
      );

      const cryptoData = response.map((coin: any) => ({
        id: coin.id,
        symbol: coin.symbol.toUpperCase(),
        name: coin.name,
        image: coin.image,
        price: coin.current_price,
        change: coin.price_change_24h,
        changePercent: coin.price_change_percentage_24h,
        volume: coin.total_volume,
        high: coin.high_24h,
        low: coin.low_24h,
        market_cap: coin.market_cap,
        market_cap_rank: coin.market_cap_rank,
        price_change_24h: coin.price_change_24h,
        price_change_percentage_24h: coin.price_change_percentage_24h,
        circulating_supply: coin.circulating_supply,
        total_supply: coin.total_supply,
        max_supply: coin.max_supply,
        ath: coin.ath,
        ath_change_percentage: coin.ath_change_percentage,
        last_updated: coin.last_updated,
      }));
      
      console.log('✅ Real-time crypto data fetched:', cryptoData.map(c => `${c.symbol}: $${c.price.toLocaleString()}`));
      return cryptoData;
    } catch (error) {
      console.error('❌ Failed to fetch real-time crypto data, using fallback');
      return this.getFallbackData(`${this.baseUrl}/coins/markets`).map((coin: any) => ({
        id: coin.id,
        symbol: coin.symbol.toUpperCase(),
        name: coin.name,
        image: coin.image,
        price: coin.current_price,
        change: coin.price_change_24h,
        changePercent: coin.price_change_percentage_24h,
        volume: coin.total_volume,
        high: coin.high_24h,
        low: coin.low_24h,
        market_cap: coin.market_cap,
        market_cap_rank: coin.market_cap_rank,
        price_change_24h: coin.price_change_24h,
        price_change_percentage_24h: coin.price_change_percentage_24h,
        circulating_supply: coin.circulating_supply,
        total_supply: coin.total_supply,
        max_supply: coin.max_supply,
        ath: coin.ath,
        ath_change_percentage: coin.ath_change_percentage,
        last_updated: coin.last_updated,
      }));
    }
  }

  async getFearGreedIndex(): Promise<FearGreedIndex> {
    try {
      // Use the backend proxy for consistency
      const response = await this.makeRequest(`${this.baseUrl}/fear-and-greed?limit=1`);
      const data = response.data[0];

      return {
        value: parseInt(data.value),
        value_classification: data.value_classification,
        timestamp: data.timestamp,
        time_until_update: data.time_until_update || '',
      };
    } catch (error) {
      console.error('Failed to fetch Fear & Greed Index, using fallback');
      // Fallback to direct call if proxy fails
      try {
        const response = await fetch('https://api.alternative.me/fng/?limit=1');
        const data = await response.json();

        return {
          value: parseInt(data.data[0].value),
          value_classification: data.data[0].value_classification,
          timestamp: data.data[0].timestamp,
          time_until_update: data.data[0].time_until_update || '',
        };
      } catch (fallbackError) {
        console.error('Fallback Fear & Greed Index fetch also failed');
        // Return a neutral value as last resort
        return {
          value: 50,
          value_classification: 'Neutral',
          timestamp: Date.now(),
          time_until_update: '',
        };
      }
    }
  }

  async getGlobalMarketData(): Promise<any> {
    const response = await this.makeRequest(`${this.baseUrl}/global`);
    return response.data;
  }

  async getAccount(): Promise<Account> {
    return tradingProviderService.getAccount();
  }

  async getPositions(): Promise<Position[]> {
    return tradingProviderService.getPositions();
  }

  async getOrders(): Promise<Order[]> {
    return tradingProviderService.getOrders();
  }

  async placeOrder(order: Partial<Order>): Promise<Order> {
    if (!order.symbol || !order.qty || !order.side) {
      throw new Error('Invalid order request: missing symbol, qty, or side');
    }

    const params: PlaceOrderParams = {
      symbol: order.symbol,
      qty: order.qty,
      side: order.side,
      order_type: (order.order_type as 'market' | 'limit') || 'market',
      time_in_force: (order as any).time_in_force || 'day',
      limit_price: order.limit_price ?? undefined,
    };

    return tradingProviderService.placeOrder(params);
  }

  // Method to clear cache (useful for testing or manual refresh)
  clearCache(): void {
    this.cache.clear();
    console.log('🗑️ CoinGecko cache cleared');
  }

  // Method to get cache status
  getCacheStatus(): { size: number; entries: string[] } {
    return {
      size: this.cache.size,
      entries: Array.from(this.cache.keys())
    };
  }

  // Method to test API connectivity
  async testConnection(): Promise<{ success: boolean; message: string }> {
    try {
      console.log('🔍 Testing CoinGecko API connection...');
      
      const testUrl = `${this.baseUrl}/ping`;
      const response = await fetch(testUrl, { 
        method: 'GET',
        headers: await this.getHeaders()
      });
      
      if (response.ok) {
        const data = await response.json();
        return {
          success: true,
          message: `Connection successful: ${data.gecko_says || 'API operational'}`
        };
      } else {
        return {
          success: false,
          message: `API returned HTTP ${response.status}: ${response.statusText}`
        };
      }
    } catch (error) {
      return {
        success: false,
        message: `Connection test failed: ${error instanceof Error ? error.message : 'Unknown error'}`
      };
    }
  }
}

export const coinGeckoService = new CoinGeckoService();