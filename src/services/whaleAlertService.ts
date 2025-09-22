import { apiKeysService } from './apiKeysService';

// Types for WhaleAlert API responses
export interface WhaleTransaction {
  blockchain: string;
  symbol: string;
  id: string;
  transaction_type: 'transfer' | 'mint' | 'burn';
  hash: string;
  from: {
    address: string;
    owner: string;
    owner_type: 'exchange' | 'wallet' | 'unknown';
  };
  to: {
    address: string;
    owner: string;
    owner_type: 'exchange' | 'wallet' | 'unknown';
  };
  timestamp: number;
  amount: number;
  amount_usd: number;
  transaction_count: number;
}

export interface ExchangeFlow {
  id: string;
  exchange: string;
  symbol: string;
  blockchain: string;
  flow_type: 'in' | 'out';
  amount: number;
  amount_usd: number;
  timestamp: number;
  transaction_hash: string;
}

export interface WhaleMovement {
  id: string;
  blockchain: string;
  symbol: string;
  amount: number;
  amount_usd: number;
  from_address: string;
  to_address: string;
  from_owner: string;
  to_owner: string;
  timestamp: number;
  hash: string;
  classification: 'whale_to_exchange' | 'exchange_to_whale' | 'whale_to_whale' | 'unknown';
}

class WhaleAlertService {
  private baseUrl: string;
  private apiKey: string;
  private cache: Map<string, { data: any; timestamp: number }>;
  private rateLimitDelay: number;
  private lastRequestTime: number;

  constructor() {
    this.baseUrl = 'https://api.whale-alert.io/v1';
    this.apiKey = ''; // Initialize empty, will be loaded from apiKeysService
    this.cache = new Map();
    this.rateLimitDelay = 1000; // 1 second between requests
    this.lastRequestTime = 0;
    
    this.initializeApiKey();
  }

  private async initializeApiKey() {
    try {
      const storedKey = await apiKeysService.getApiKeyWithFallback('whalealert', 'api_key');
      if (storedKey) {
        this.apiKey = storedKey;
        console.log('✅ WhaleAlert API key loaded from stored keys');
      } else {
        console.log('📋 Using default WhaleAlert API key');
      }
    } catch (error) {
      console.warn('⚠️ Failed to load WhaleAlert API key from stored keys, using default');
    }
  }

  private async getApiKey(): Promise<string> {
    const apiKey = await apiKeysService.getApiKeyWithFallback('whalealert', 'api_key');
    return apiKey || this.apiKey;
  }

  private getCacheKey(endpoint: string, params: Record<string, any>): string {
    return `whalealert_${endpoint}_${JSON.stringify(params)}`;
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < 300000) { // 5 minutes cache
      return cached.data;
    }
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
    
    // Clean old cache entries (keep only last 50 entries)
    if (this.cache.size > 50) {
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < 10; i++) {
        this.cache.delete(keys[i]);
      }
    }
  }

  private async rateLimitedRequest(): Promise<void> {
    const now = Date.now();
    const timeSinceLastRequest = now - this.lastRequestTime;
    
    if (timeSinceLastRequest < this.rateLimitDelay) {
      const delay = this.rateLimitDelay - timeSinceLastRequest;
      await new Promise(resolve => setTimeout(resolve, delay));
    }
    
    this.lastRequestTime = Date.now();
  }

  private async makeRequest(endpoint: string, params: Record<string, any> = {}): Promise<any> {
    const cacheKey = this.getCacheKey(endpoint, params);
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      console.log('🔄 Using cached WhaleAlert data');
      return cached;
    }

    await this.rateLimitedRequest();

    try {
      console.log(`🌐 Making WhaleAlert API request to: ${endpoint}`);
      
      const apiKey = await this.getApiKey();
      const searchParams = new URLSearchParams({
        ...params,
        api_key: apiKey,
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);

      // Use proxy for CORS issues
      const proxyUrl = `/api/whale-alert-proxy?${searchParams}`;
      const response = await fetch(proxyUrl, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        throw new Error(`WhaleAlert API Error: ${response.status} - ${errorText}`);
      }

      const data = await response.json();

      if (data.result !== 'success') {
        throw new Error(`WhaleAlert API Error: ${data.message || 'Unknown error'}`);
      }

      console.log('✅ WhaleAlert API request successful');
      this.setCache(cacheKey, data);
      return data;
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('❌ WhaleAlert API request timed out');
        } else {
          console.error('❌ WhaleAlert API error:', error.message);
        }
      }
      throw error;
    }
  }

  async getLargeTransactions(minValue: number = 500000): Promise<WhaleTransaction[]> {
    try {
      console.log(`💰 Fetching large transactions above $${minValue.toLocaleString()}`);
      
      // WhaleAlert uses timestamps, get last 24 hours
      const endTime = Math.floor(Date.now() / 1000);
      const startTime = endTime - (24 * 60 * 60); // 24 hours ago

      const response = await this.makeRequest('transactions', {
        start: startTime.toString(),
        end: endTime.toString(),
        min_value: minValue.toString(),
        limit: '100',
      });

      const transactions = response.transactions || [];
      
      const whaleTransactions: WhaleTransaction[] = transactions.map((tx: any) => ({
        blockchain: tx.blockchain || 'unknown',
        symbol: tx.symbol || 'UNKNOWN',
        id: tx.id || tx.hash,
        transaction_type: tx.transaction_type || 'transfer',
        hash: tx.hash || '',
        from: {
          address: tx.from?.address || '',
          owner: tx.from?.owner || 'unknown',
          owner_type: tx.from?.owner_type || 'unknown',
        },
        to: {
          address: tx.to?.address || '',
          owner: tx.to?.owner || 'unknown',
          owner_type: tx.to?.owner_type || 'unknown',
        },
        timestamp: tx.timestamp || Date.now() / 1000,
        amount: tx.amount || 0,
        amount_usd: tx.amount_usd || 0,
        transaction_count: tx.transaction_count || 1,
      }));

      console.log(`✅ Found ${whaleTransactions.length} large transactions`);
      return whaleTransactions;
    } catch (error) {
      console.error('❌ Failed to fetch large transactions:', error);
      
      // Return mock data as fallback
      return this.getMockTransactions(minValue);
    }
  }

  async getExchangeFlows(exchange: string): Promise<ExchangeFlow[]> {
    try {
      console.log(`🏦 Fetching exchange flows for ${exchange}`);
      
      const endTime = Math.floor(Date.now() / 1000);
      const startTime = endTime - (24 * 60 * 60); // 24 hours ago

      // WhaleAlert doesn't have a direct exchange flows endpoint, 
      // so we'll filter transactions involving the exchange
      const response = await this.makeRequest('transactions', {
        start: startTime.toString(),
        end: endTime.toString(),
        min_value: '100000',
        limit: '100',
      });

      const transactions = response.transactions || [];
      
      const exchangeFlows: ExchangeFlow[] = transactions
        .filter((tx: any) => {
          const fromOwner = (tx.from?.owner || '').toLowerCase();
          const toOwner = (tx.to?.owner || '').toLowerCase();
          return fromOwner.includes(exchange.toLowerCase()) || toOwner.includes(exchange.toLowerCase());
        })
        .map((tx: any) => {
          const isOutflow = (tx.from?.owner || '').toLowerCase().includes(exchange.toLowerCase());
          
          return {
            id: tx.id || tx.hash,
            exchange,
            symbol: tx.symbol || 'UNKNOWN',
            blockchain: tx.blockchain || 'ethereum',
            flow_type: isOutflow ? 'out' as const : 'in' as const,
            amount: tx.amount || 0,
            amount_usd: tx.amount_usd || 0,
            timestamp: tx.timestamp || Date.now() / 1000,
            transaction_hash: tx.hash || '',
          };
        });

      console.log(`✅ Found ${exchangeFlows.length} exchange flows for ${exchange}`);
      return exchangeFlows;
    } catch (error) {
      console.error('❌ Failed to fetch exchange flows:', error);
      
      // Return mock data as fallback
      return this.getMockExchangeFlows(exchange);
    }
  }

  async getWhaleMovements(timeframe: '1h' | '24h' | '7d' = '24h'): Promise<WhaleMovement[]> {
    try {
      console.log(`🐋 Fetching whale movements for timeframe: ${timeframe}`);
      
      const endTime = Math.floor(Date.now() / 1000);
      const timeframeMap = {
        '1h': 60 * 60,
        '24h': 24 * 60 * 60,
        '7d': 7 * 24 * 60 * 60,
      };
      const startTime = endTime - timeframeMap[timeframe];

      const response = await this.makeRequest('transactions', {
        start: startTime.toString(),
        end: endTime.toString(),
        min_value: '1000000', // Only very large transactions (1M+)
        limit: '100',
      });

      const transactions = response.transactions || [];
      
      const whaleMovements: WhaleMovement[] = transactions.map((tx: any) => {
        let classification: WhaleMovement['classification'] = 'unknown';
        
        const fromType = tx.from?.owner_type;
        const toType = tx.to?.owner_type;
        
        if (fromType === 'wallet' && toType === 'exchange') {
          classification = 'whale_to_exchange';
        } else if (fromType === 'exchange' && toType === 'wallet') {
          classification = 'exchange_to_whale';
        } else if (fromType === 'wallet' && toType === 'wallet') {
          classification = 'whale_to_whale';
        }
        
        return {
          id: tx.id || tx.hash,
          blockchain: tx.blockchain || 'ethereum',
          symbol: tx.symbol || 'UNKNOWN',
          amount: tx.amount || 0,
          amount_usd: tx.amount_usd || 0,
          from_address: tx.from?.address || '',
          to_address: tx.to?.address || '',
          from_owner: tx.from?.owner || 'unknown',
          to_owner: tx.to?.owner || 'unknown',
          timestamp: tx.timestamp || Date.now() / 1000,
          hash: tx.hash || '',
          classification,
        };
      });

      console.log(`✅ Found ${whaleMovements.length} whale movements`);
      return whaleMovements;
    } catch (error) {
      console.error('❌ Failed to fetch whale movements:', error);
      
      // Return mock data as fallback
      return this.getMockWhaleMovements(timeframe);
    }
  }

  async getStatus(): Promise<{ result: string; message: string }> {
    try {
      const response = await this.makeRequest('status');
      return {
        result: response.result || 'success',
        message: response.message || 'WhaleAlert API is operational',
      };
    } catch (error) {
      console.error('❌ Failed to get WhaleAlert status:', error);
      return {
        result: 'error',
        message: 'Failed to connect to WhaleAlert API',
      };
    }
  }

  // Mock data methods for fallback
  private getMockTransactions(minValue: number): WhaleTransaction[] {
    const mockTransactions: WhaleTransaction[] = [];
    const symbols = ['BTC', 'ETH', 'USDT', 'USDC'];
    const blockchains = ['bitcoin', 'ethereum'];
    
    for (let i = 0; i < 10; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const blockchain = blockchains[Math.floor(Math.random() * blockchains.length)];
      const amount = minValue + Math.random() * minValue * 2;
      
      mockTransactions.push({
        blockchain,
        symbol,
        id: `mock_${i}`,
        transaction_type: 'transfer',
        hash: `0x${Math.random().toString(16).substring(2, 66)}`,
        from: {
          address: `0x${Math.random().toString(16).substring(2, 42)}`,
          owner: 'unknown',
          owner_type: 'wallet',
        },
        to: {
          address: `0x${Math.random().toString(16).substring(2, 42)}`,
          owner: 'binance',
          owner_type: 'exchange',
        },
        timestamp: Date.now() / 1000 - Math.random() * 24 * 60 * 60,
        amount: amount / (symbol === 'BTC' ? 100000 : symbol === 'ETH' ? 4000 : 1),
        amount_usd: amount,
        transaction_count: 1,
      });
    }
    
    return mockTransactions;
  }

  private getMockExchangeFlows(exchange: string): ExchangeFlow[] {
    const mockFlows: ExchangeFlow[] = [];
    const symbols = ['BTC', 'ETH', 'USDT'];
    
    for (let i = 0; i < 5; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const flowType = Math.random() > 0.5 ? 'in' : 'out';
      const amount = 100000 + Math.random() * 500000;
      
      mockFlows.push({
        id: `mock_flow_${i}`,
        exchange,
        symbol,
        blockchain: 'ethereum',
        flow_type: flowType,
        amount: amount / (symbol === 'BTC' ? 100000 : symbol === 'ETH' ? 4000 : 1),
        amount_usd: amount,
        timestamp: Date.now() / 1000 - Math.random() * 24 * 60 * 60,
        transaction_hash: `0x${Math.random().toString(16).substring(2, 66)}`,
      });
    }
    
    return mockFlows;
  }

  private getMockWhaleMovements(timeframe: string): WhaleMovement[] {
    const mockMovements: WhaleMovement[] = [];
    const symbols = ['BTC', 'ETH', 'USDT'];
    const classifications: WhaleMovement['classification'][] = [
      'whale_to_exchange',
      'exchange_to_whale',
      'whale_to_whale',
    ];
    
    for (let i = 0; i < 8; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const classification = classifications[Math.floor(Math.random() * classifications.length)];
      const amount = 1000000 + Math.random() * 5000000;
      
      mockMovements.push({
        id: `mock_movement_${i}`,
        blockchain: 'ethereum',
        symbol,
        amount: amount / (symbol === 'BTC' ? 100000 : symbol === 'ETH' ? 4000 : 1),
        amount_usd: amount,
        from_address: `0x${Math.random().toString(16).substring(2, 42)}`,
        to_address: `0x${Math.random().toString(16).substring(2, 42)}`,
        from_owner: classification.includes('exchange_to') ? 'binance' : 'unknown',
        to_owner: classification.includes('to_exchange') ? 'coinbase' : 'unknown',
        timestamp: Date.now() / 1000 - Math.random() * 24 * 60 * 60,
        hash: `0x${Math.random().toString(16).substring(2, 66)}`,
        classification,
      });
    }
    
    return mockMovements;
  }
}

export const whaleAlertService = new WhaleAlertService();