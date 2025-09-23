import { apiKeysService } from './apiKeysService';

// Types for Etherscan API responses
export interface EtherscanTransaction {
  hash: string;
  from: string;
  to: string;
  value: string;
  timeStamp: string;
  blockNumber: string;
  gas: string;
  gasPrice: string;
  gasUsed: string;
  input: string;
  isError: string;
}

export interface GasMetrics {
  SafeGasPrice: string;
  StandardGasPrice: string;
  FastGasPrice: string;
  suggestBaseFee: string;
  gasUsedRatio: string;
}

export interface WalletBalance {
  account: string;
  balance: string;
  balanceInEth: number;
}

export interface ActiveAddress {
  address: string;
  transactionCount: number;
  lastActivity: string;
}

class EtherscanService {
  private baseUrl: string;
  private apiKey: string;
  private cache: Map<string, { data: any; timestamp: number }>;
  private rateLimitDelay: number;
  private lastRequestTime: number;

  constructor() {
    this.baseUrl = 'https://api.etherscan.io/api';
    this.apiKey = ''; // Initialize empty, will be loaded from apiKeysService
    this.cache = new Map();
    this.rateLimitDelay = 200; // 200ms between requests (5 requests per second)
    this.lastRequestTime = 0;
    
    this.initializeApiKey();
  }

  private async initializeApiKey() {
    try {
      const storedKey = await apiKeysService.getApiKeyWithFallback('etherscan', 'api_key');
      if (storedKey) {
        this.apiKey = storedKey;
        console.log('✅ Etherscan API key loaded from stored keys');
      } else {
        console.log('📋 Using default Etherscan API key');
      }
    } catch (error) {
      console.warn('⚠️ Failed to load Etherscan API key from stored keys, using default');
    }
  }

  private async getApiKey(): Promise<string> {
    const apiKey = await apiKeysService.getApiKeyWithFallback('etherscan', 'api_key');
    return apiKey || this.apiKey;
  }

  private getCacheKey(endpoint: string, params: Record<string, any>): string {
    return `${endpoint}_${JSON.stringify(params)}`;
  }

  private getFromCache(key: string): any | null {
    const cached = this.cache.get(key);
    if (cached && Date.now() - cached.timestamp < 60000) { // 1 minute cache
      return cached.data;
    }
    return null;
  }

  private setCache(key: string, data: any): void {
    this.cache.set(key, { data, timestamp: Date.now() });
    
    // Clean old cache entries (keep only last 100 entries)
    if (this.cache.size > 100) {
      const keys = Array.from(this.cache.keys());
      for (let i = 0; i < 20; i++) {
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

  private async makeRequest(params: Record<string, any>): Promise<any> {
    const cacheKey = this.getCacheKey('etherscan', params);
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      console.log('🔄 Using cached Etherscan data');
      return cached;
    }

    await this.rateLimitedRequest();

    try {
      console.log('🌐 Making Etherscan API request');
      
      const apiKey = await this.getApiKey();
      
      // If no API key is available, return empty data instead of failing
      if (!apiKey) {
        console.log('⚠️ No Etherscan API key found, returning empty data');
        const emptyData = { status: '1', message: 'OK', result: [] };
        this.setCache(cacheKey, emptyData);
        return emptyData;
      }
      
      const searchParams = new URLSearchParams({
        ...params,
        apikey: apiKey,
      });

      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 10000);

      const response = await fetch(`${this.baseUrl}?${searchParams}`, {
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Etherscan API Error: ${response.status}`);
      }

      const data = await response.json();

      if (data.status === '0' && data.message !== 'No transactions found') {
        // Handle common Etherscan API errors gracefully
        if (data.message === 'NOTOK' || data.message === 'Invalid API Key') {
          console.log('⚠️ Etherscan API key issue, returning empty data');
          const emptyData = { status: '1', message: 'OK', result: [] };
          this.setCache(cacheKey, emptyData);
          return emptyData;
        }
        throw new Error(`Etherscan API Error: ${data.message}`);
      }

      console.log('✅ Etherscan API request successful');
      this.setCache(cacheKey, data);
      return data;
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('❌ Etherscan API request timed out');
        } else {
          console.error('❌ Etherscan API error:', error.message);
        }
      }
      
      // Return empty data instead of throwing to prevent UI crashes
      console.log('🔄 Returning empty data due to Etherscan error');
      const emptyData = { status: '1', message: 'OK', result: [] };
      this.setCache(cacheKey, emptyData);
      return emptyData;
    }
  }

  async getActiveAddresses(timeframe: '1d' | '7d' | '30d' = '1d'): Promise<ActiveAddress[]> {
    try {
      console.log(`🔍 Fetching active addresses for timeframe: ${timeframe}`);
      
      // Calculate timestamp based on timeframe
      const now = Math.floor(Date.now() / 1000);
      const timeframeMap = {
        '1d': 24 * 60 * 60,
        '7d': 7 * 24 * 60 * 60,
        '30d': 30 * 24 * 60 * 60,
      };
      const startTime = now - timeframeMap[timeframe];

      // Get latest block transactions to find active addresses
      const response = await this.makeRequest({
        module: 'proxy',
        action: 'eth_getBlockByNumber',
        tag: 'latest',
        boolean: 'true',
      });

      // Process transactions to extract unique addresses
      const transactions = response.result?.transactions || [];
      const addressActivity = new Map<string, { count: number; lastActivity: string }>();

      for (const tx of transactions.slice(0, 100)) { // Limit to recent transactions
        if (tx.from) {
          const current = addressActivity.get(tx.from) || { count: 0, lastActivity: '0' };
          addressActivity.set(tx.from, {
            count: current.count + 1,
            lastActivity: Math.max(parseInt(current.lastActivity), parseInt(tx.blockNumber || '0')).toString(),
          });
        }
        if (tx.to) {
          const current = addressActivity.get(tx.to) || { count: 0, lastActivity: '0' };
          addressActivity.set(tx.to, {
            count: current.count + 1,
            lastActivity: Math.max(parseInt(current.lastActivity), parseInt(tx.blockNumber || '0')).toString(),
          });
        }
      }

      const activeAddresses: ActiveAddress[] = Array.from(addressActivity.entries())
        .map(([address, activity]) => ({
          address,
          transactionCount: activity.count,
          lastActivity: new Date(parseInt(activity.lastActivity) * 1000).toISOString(),
        }))
        .sort((a, b) => b.transactionCount - a.transactionCount)
        .slice(0, 20);

      console.log(`✅ Found ${activeAddresses.length} active addresses`);
      return activeAddresses;
    } catch (error) {
      console.error('❌ Failed to fetch active addresses:', error);
      return [];
    }
  }

  async getGasMetrics(): Promise<GasMetrics | null> {
    try {
      console.log('⛽ Fetching gas metrics');
      
      const response = await this.makeRequest({
        module: 'gastracker',
        action: 'gasoracle',
      });

      if (response.result) {
        console.log('✅ Gas metrics fetched successfully');
        return {
          SafeGasPrice: response.result.SafeGasPrice || '0',
          StandardGasPrice: response.result.ProposeGasPrice || '0',
          FastGasPrice: response.result.FastGasPrice || '0',
          suggestBaseFee: response.result.suggestBaseFee || '0',
          gasUsedRatio: response.result.gasUsedRatio || '0',
        };
      }

      return null;
    } catch (error) {
      console.error('❌ Failed to fetch gas metrics:', error);
      return null;
    }
  }

  async getLargeTransactions(threshold: number = 1000): Promise<EtherscanTransaction[]> {
    try {
      console.log(`💰 Fetching large transactions above ${threshold} ETH`);
      
      // Get latest block number
      const latestBlockResponse = await this.makeRequest({
        module: 'proxy',
        action: 'eth_blockNumber',
      });

      const latestBlock = parseInt(latestBlockResponse.result, 16);
      const fromBlock = latestBlock - 100; // Check last 100 blocks

      // Get transactions from recent blocks
      const transactions: EtherscanTransaction[] = [];
      const thresholdWei = (threshold * 1e18).toString();

      // Check last few blocks for large transactions
      for (let i = 0; i < 5; i++) {
        try {
          const blockResponse = await this.makeRequest({
            module: 'proxy',
            action: 'eth_getBlockByNumber',
            tag: `0x${(latestBlock - i).toString(16)}`,
            boolean: 'true',
          });

          if (blockResponse.result?.transactions) {
            const blockTxs = blockResponse.result.transactions
              .filter((tx: any) => {
                const value = parseInt(tx.value, 16);
                return value >= parseInt(thresholdWei);
              })
              .map((tx: any) => ({
                hash: tx.hash,
                from: tx.from,
                to: tx.to,
                value: (parseInt(tx.value, 16) / 1e18).toFixed(4),
                timeStamp: (parseInt(blockResponse.result.timestamp, 16)).toString(),
                blockNumber: (parseInt(tx.blockNumber, 16)).toString(),
                gas: (parseInt(tx.gas, 16)).toString(),
                gasPrice: (parseInt(tx.gasPrice, 16)).toString(),
                gasUsed: '0',
                input: tx.input,
                isError: '0',
              }));

            transactions.push(...blockTxs);
          }
        } catch (blockError) {
          console.warn(`⚠️ Failed to fetch block ${latestBlock - i}:`, blockError);
        }
      }

      console.log(`✅ Found ${transactions.length} large transactions`);
      return transactions.slice(0, 50); // Return top 50
    } catch (error) {
      console.error('❌ Failed to fetch large transactions:', error);
      return [];
    }
  }

  async getWalletBalance(address: string): Promise<WalletBalance | null> {
    try {
      console.log(`💼 Fetching wallet balance for ${address.substring(0, 10)}...`);
      
      const response = await this.makeRequest({
        module: 'account',
        action: 'balance',
        address: address,
        tag: 'latest',
      });

      if (response.result) {
        const balanceWei = response.result;
        const balanceEth = parseFloat(balanceWei) / 1e18;

        console.log(`✅ Wallet balance fetched: ${balanceEth.toFixed(4)} ETH`);
        return {
          account: address,
          balance: balanceWei,
          balanceInEth: balanceEth,
        };
      }

      return null;
    } catch (error) {
      console.error('❌ Failed to fetch wallet balance:', error);
      return null;
    }
  }

  async getMultipleWalletBalances(addresses: string[]): Promise<WalletBalance[]> {
    try {
      console.log(`💼 Fetching balances for ${addresses.length} wallets`);
      
      const response = await this.makeRequest({
        module: 'account',
        action: 'balancemulti',
        address: addresses.join(','),
        tag: 'latest',
      });

      if (response.result && Array.isArray(response.result)) {
        const balances = response.result.map((item: any) => ({
          account: item.account,
          balance: item.balance,
          balanceInEth: parseFloat(item.balance) / 1e18,
        }));

        console.log(`✅ Fetched ${balances.length} wallet balances`);
        return balances;
      }

      return [];
    } catch (error) {
      console.error('❌ Failed to fetch multiple wallet balances:', error);
      return [];
    }
  }
}

export const etherscanService = new EtherscanService();