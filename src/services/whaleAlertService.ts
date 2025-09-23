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
    this.apiKey = 'MmQia8CLAkvFbFJ1eKGb0AR6i2eCf7Ez'; // Default API key provided by user
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
      
      // In development, try direct API call first (with CORS handling)
      if (import.meta.env.DEV) {
        console.log('🔧 Development mode: attempting direct API call');
        try {
          const directUrl = `https://api.whale-alert.io/v1/${endpoint}?${new URLSearchParams({
            ...params,
            api_key: apiKey,
          })}`;
          
          const controller = new AbortController();
          const timeoutId = setTimeout(() => controller.abort(), 15000);

          const response = await fetch(directUrl, {
            signal: controller.signal,
            mode: 'cors',
          });

          clearTimeout(timeoutId);

          if (response.ok) {
            const data = await response.json();
            console.log('✅ Direct WhaleAlert API request successful');
            this.setCache(cacheKey, data);
            return data;
          }
        } catch (directError) {
          console.log('⚠️ Direct API call failed, falling back to mock data:', directError instanceof Error ? directError.message : 'Unknown error');
          throw new Error('Direct API call failed - using fallback data');
        }
      }

      // Production mode: use proxy
      const searchParams = new URLSearchParams({
        ...params,
        endpoint,
        api_key: apiKey,
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);

      const proxyUrl = `/api/whale-alert-proxy?${searchParams}`;
      const response = await fetch(proxyUrl, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        const errorText = await response.text();
        console.error(`❌ WhaleAlert API HTTP Error: ${response.status} - ${errorText}`);
        throw new Error(`WhaleAlert API Error: ${response.status} - ${errorText}`);
      }

      const responseText = await response.text();
      
      // Check if response is HTML (error page or Netlify function not deployed)
      if (responseText.trim().startsWith('<!DOCTYPE') || responseText.trim().startsWith('<html')) {
        console.error('❌ WhaleAlert API returned HTML instead of JSON - proxy function may not be deployed');
        console.log('🔄 Falling back to mock data due to proxy configuration issue');
        throw new Error('WhaleAlert API proxy returned HTML - using fallback data');
      }

      let data;
      try {
        data = JSON.parse(responseText);
      } catch (parseError) {
        console.error('❌ Failed to parse WhaleAlert API response as JSON:', responseText.substring(0, 200));
        console.log('🔄 Falling back to mock data due to JSON parsing error');
        throw new Error('Invalid JSON response from WhaleAlert API - using fallback data');
      }

      // Check if the response indicates an error
      if (data.result && data.result !== 'success') {
        console.error(`❌ WhaleAlert API returned error: ${data.message || 'Unknown error'}`);
        console.log('🔄 Falling back to mock data due to API error');
        throw new Error(`WhaleAlert API Error: ${data.message || 'Unknown error'} - using fallback data`);
      }

      console.log('✅ WhaleAlert API request successful');
      this.setCache(cacheKey, data);
      return data;
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('❌ WhaleAlert API request timed out - using fallback data');
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
      console.log('🔄 Falling back to mock data for large transactions');
      
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
        message: `Failed to connect to WhaleAlert API: ${error instanceof Error ? error.message : 'Unknown error'}`,
      };
    }
  }

  // Method to test the proxy connection
  async testConnection(): Promise<{ success: boolean; message: string; details?: any }> {
    try {
      console.log('🔍 Testing WhaleAlert proxy connection...');
      
      const apiKey = await this.getApiKey();
      console.log(`🔑 Using API key: ${apiKey ? 'configured' : 'not configured'}`);
      
      const searchParams = new URLSearchParams({
        endpoint: 'status',
        api_key: apiKey,
      });

      const proxyUrl = `/api/whale-alert-proxy?${searchParams}`;
      console.log(`🌐 Testing proxy URL: ${proxyUrl}`);
      
      const response = await fetch(proxyUrl, { method: 'GET' });
      
      if (!response.ok) {
        return {
          success: false,
          message: `Proxy returned HTTP ${response.status}: ${response.statusText}`,
          details: { status: response.status, statusText: response.statusText }
        };
      }

      const responseText = await response.text();
      console.log(`📄 Response type: ${response.headers.get('content-type')}`);
      console.log(`📄 Response preview: ${responseText.substring(0, 200)}...`);
      
      if (responseText.trim().startsWith('<!DOCTYPE') || responseText.trim().startsWith('<html')) {
        return {
          success: false,
          message: 'Proxy returned HTML instead of JSON - Netlify function may not be deployed or configured properly',
          details: { 
            contentType: response.headers.get('content-type'),
            responsePreview: responseText.substring(0, 500)
          }
        };
      }

      try {
        const data = JSON.parse(responseText);
        return {
          success: true,
          message: `Connection successful: ${data.message || 'API operational'}`,
          details: data
        };
      } catch (parseError) {
        return {
          success: false,
          message: `Invalid JSON response: ${responseText.substring(0, 100)}...`,
          details: { 
            parseError: parseError instanceof Error ? parseError.message : 'Unknown parse error',
            responsePreview: responseText.substring(0, 200)
          }
        };
      }
    } catch (error) {
      return {
        success: false,
        message: `Connection test failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        details: { error: error instanceof Error ? error.stack : error }
      };
    }
  }

  // Method to test direct API connection (for debugging)
  async testDirectConnection(): Promise<{ success: boolean; message: string; details?: any }> {
    try {
      console.log('🔍 Testing direct WhaleAlert API connection...');
      
      const apiKey = await this.getApiKey();
      if (!apiKey || apiKey === 'default_key') {
        return {
          success: false,
          message: 'No valid API key configured for direct connection test'
        };
      }

      const url = `https://api.whale-alert.io/v1/status?api_key=${apiKey}`;
      console.log(`🌐 Testing direct URL: ${url.replace(apiKey, '***')}`);
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);
      
      const response = await fetch(url, {
        signal: controller.signal,
        mode: 'cors',
      });
      
      clearTimeout(timeoutId);
      
      if (!response.ok) {
        const errorText = await response.text();
        return {
          success: false,
          message: `Direct API returned HTTP ${response.status}: ${response.statusText}`,
          details: { 
            status: response.status, 
            statusText: response.statusText,
            errorText: errorText.substring(0, 200)
          }
        };
      }

      const data = await response.json();
      return {
        success: true,
        message: `Direct connection successful: ${data.message || 'API operational'}`,
        details: data
      };
    } catch (error) {
      return {
        success: false,
        message: `Direct connection test failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
        details: { 
          error: error instanceof Error ? error.message : error,
          isCorsError: error instanceof Error && error.message.includes('CORS'),
          isNetworkError: error instanceof Error && (error.message.includes('fetch') || error.message.includes('network'))
        }
      };
    }
  }

  // Mock data methods for fallback
  private getMockTransactions(minValue: number): WhaleTransaction[] {
    const mockTransactions: WhaleTransaction[] = [];
    const symbols = ['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'ADA'];
    const blockchains = ['bitcoin', 'ethereum', 'binance-smart-chain'];
    const exchanges = ['binance', 'coinbase', 'kraken', 'huobi', 'okx'];
    const owners = ['unknown', 'whale_wallet', 'institutional'];
    
    // Generate 8-12 realistic mock transactions
    const count = 8 + Math.floor(Math.random() * 5);
    
    for (let i = 0; i < count; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)];
      const blockchain = blockchains[Math.floor(Math.random() * blockchains.length)];
      const amount = minValue + Math.random() * minValue * 3;
      const isToExchange = Math.random() > 0.3;
      const isFromExchange = Math.random() > 0.7;
      
      mockTransactions.push({
        blockchain,
        symbol,
        id: `mock_${Date.now()}_${i}`,
        transaction_type: 'transfer',
        hash: `0x${Math.random().toString(16).substring(2, 66)}`,
        from: {
          address: `0x${Math.random().toString(16).substring(2, 42)}`,
          owner: isFromExchange ? exchanges[Math.floor(Math.random() * exchanges.length)] : owners[Math.floor(Math.random() * owners.length)],
          owner_type: isFromExchange ? 'exchange' : 'wallet',
        },
        to: {
          address: `0x${Math.random().toString(16).substring(2, 42)}`,
          owner: isToExchange ? exchanges[Math.floor(Math.random() * exchanges.length)] : owners[Math.floor(Math.random() * owners.length)],
          owner_type: isToExchange ? 'exchange' : 'wallet',
        },
        timestamp: Date.now() / 1000 - Math.random() * 24 * 60 * 60,
        amount: amount / (symbol === 'BTC' ? 100000 : symbol === 'ETH' ? 4000 : symbol === 'BNB' ? 600 : 1),
        amount_usd: amount,
        transaction_count: 1,
      });
    }
    
    console.log(`🔄 Generated ${mockTransactions.length} mock whale transactions`);
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

// Global test function for debugging (available in browser console)
if (typeof window !== 'undefined') {
  (window as any).testWhaleAlert = {
    async testProxy() {
      console.log('🧪 Testing WhaleAlert proxy connection...');
      const result = await whaleAlertService.testConnection();
      console.log('📊 Proxy test result:', result);
      return result;
    },
    async testDirect() {
      console.log('🧪 Testing WhaleAlert direct connection...');
      const result = await whaleAlertService.testDirectConnection();
      console.log('📊 Direct test result:', result);
      return result;
    },
    async testBoth() {
      console.log('🧪 Testing both WhaleAlert connections...');
      const [proxyResult, directResult] = await Promise.all([
        whaleAlertService.testConnection(),
        whaleAlertService.testDirectConnection()
      ]);
      console.log('📊 Proxy test result:', proxyResult);
      console.log('📊 Direct test result:', directResult);
      return { proxy: proxyResult, direct: directResult };
    },
    async getMockData() {
      console.log('🧪 Getting mock WhaleAlert data...');
      const result = await whaleAlertService.getLargeTransactions(500000);
      console.log('📊 Mock data result:', result);
      return result;
    }
  };
  
  console.log('🔧 WhaleAlert test functions available:');
  console.log('  - testWhaleAlert.testProxy() - Test proxy connection');
  console.log('  - testWhaleAlert.testDirect() - Test direct API connection');
  console.log('  - testWhaleAlert.testBoth() - Test both connections');
  console.log('  - testWhaleAlert.getMockData() - Get mock data');
}