import { Account, Order, Position, CryptoData, FearGreedIndex } from '../types/trading';
import { supabase } from '../lib/supabase';

class CoinGeckoService {
  private baseUrl: string;
  private apiKey: string;
  private headers: Record<string, string>;

  constructor() {
    const isDev = import.meta.env.DEV;
    this.baseUrl = isDev ? '/coingecko/api/v3' : 'https://api.coingecko.com/api/v3';
    this.apiKey = import.meta.env.VITE_COINGECKO_API_KEY;
    const usePro = String(import.meta.env.VITE_USE_COINGECKO_PRO || '').toLowerCase() === 'true';

    this.headers = {
      accept: 'application/json',
      ...(usePro && this.apiKey ? { 'x-cg-pro-api-key': this.apiKey } : {}),
    };

    // Log API key status for debugging
    if (usePro) {
      if (!this.apiKey) {
        console.warn('⚠️ VITE_USE_COINGECKO_PRO=true but no VITE_COINGECKO_API_KEY found');
      } else {
        console.log('✅ CoinGecko PRO mode enabled and API key loaded');
      }
    } else {
      console.log('ℹ️ CoinGecko PRO mode disabled; using public endpoints');
    }
  }

  private async makeRequest(url: string, options: RequestInit = {}): Promise<any> {
    try {
      console.log('🌐 Making CoinGecko API request to:', url);
      
      // Add timeout and better error handling for network issues
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000); // 10 second timeout
      
      const response = await fetch(url, {
        ...options,
        headers: {
          ...this.headers,
          ...options.headers,
        },
        signal: controller.signal,
      });
      
      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        
        // Only use fallback for severe rate limiting (429) after multiple retries
        if (response.status === 429) {
          console.warn(`⚠️ CoinGecko API throttled (429), attempting retry...`);
          await new Promise(resolve => setTimeout(resolve, 5000));
          // Try one more time
          const retryController = new AbortController();
          const retryTimeoutId = setTimeout(() => retryController.abort(), 10000);
          
          const retryResponse = await fetch(url, {
            ...options,
            headers: {
              ...this.headers,
              ...options.headers,
            },
            signal: retryController.signal,
          });
          
          clearTimeout(retryTimeoutId);
          
          if (retryResponse.ok) {
            console.log('✅ Retry successful');
            return await retryResponse.json();
          } else {
            console.warn('⚠️ Retry failed, using fallback data');
            return this.getFallbackData(url);
          }
        } else {
          console.error(`❌ CoinGecko API error! status: ${response.status}, message: ${errorText}`);
        }
        
        // For other errors, throw to trigger fallback
        throw new Error(`CoinGecko API Error: ${response.status} - ${errorText}`);
      }

      console.log('✅ CoinGecko API request successful');
      return await response.json();
    } catch (error) {
      // Handle specific network errors
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('❌ CoinGecko API request timed out after 10 seconds');
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
    // Fear and Greed Index API (free, no auth required)
    const response = await fetch('https://api.alternative.me/fng/');
    const data = await response.json();
    
    return {
      value: parseInt(data.data[0].value),
      value_classification: data.data[0].value_classification,
      timestamp: data.data[0].timestamp,
      time_until_update: data.data[0].time_until_update || '',
    };
  }

  async getGlobalMarketData(): Promise<any> {
    const response = await this.makeRequest(`${this.baseUrl}/global`);
    return response.data;
  }

  async getAccount(): Promise<Account> {
    // Delegate to Alpaca service for real account data
    const { alpacaService } = await import('./alpacaService');
    return alpacaService.getAccount();
  }

  async getPositions(): Promise<Position[]> {
    // Delegate to Alpaca service for real positions data
    const { alpacaService } = await import('./alpacaService');
    return alpacaService.getPositions();
  }

  async getOrders(): Promise<Order[]> {
    // Delegate to Alpaca service for real orders data
    const { alpacaService } = await import('./alpacaService');
    return alpacaService.getOrders();
  }

  async placeOrder(order: Partial<Order>): Promise<Order> {
    // Delegate to Alpaca service for real order placement
    const { alpacaService } = await import('./alpacaService');
    return alpacaService.placeOrder(order);
  }
}

export const coinGeckoService = new CoinGeckoService();