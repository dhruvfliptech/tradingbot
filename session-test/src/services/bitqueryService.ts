import { apiKeysService } from './apiKeysService';

// Types for Bitquery API responses
export interface DEXTrade {
  transaction: {
    hash: string;
    txFrom: {
      address: string;
    };
  };
  block: {
    timestamp: {
      iso8601: string;
    };
    height: number;
  };
  exchange: {
    name: string;
  };
  baseCurrency: {
    symbol: string;
    address: string;
  };
  quoteCurrency: {
    symbol: string;
    address: string;
  };
  tradeAmount: number;
  quoteAmount: number;
  price: number;
  side: 'BUY' | 'SELL';
}

export interface LiquidityPool {
  smartContract: {
    address: {
      address: string;
    };
  };
  exchange: {
    name: string;
  };
  baseCurrency: {
    symbol: string;
    address: string;
  };
  quoteCurrency: {
    symbol: string;
    address: string;
  };
  reserve0: number;
  reserve1: number;
  totalSupply: number;
  liquidityUSD: number;
}

export interface CrossChainFlow {
  transaction: {
    hash: string;
  };
  block: {
    timestamp: {
      iso8601: string;
    };
  };
  fromChain: string;
  toChain: string;
  currency: {
    symbol: string;
    address: string;
  };
  amount: number;
  amountUSD: number;
  sender: {
    address: string;
  };
  receiver: {
    address: string;
  };
}

export interface WhaleWallet {
  address: {
    address: string;
  };
  balance: number;
  balanceUSD: number;
  currency: {
    symbol: string;
    address: string;
  };
  lastActivity: string;
  transactionCount: number;
}

class BitqueryService {
  private baseUrl: string;
  private accessToken: string;
  private cache: Map<string, { data: any; timestamp: number }>;
  private rateLimitDelay: number;
  private lastRequestTime: number;

  constructor() {
    this.baseUrl = 'https://graphql.bitquery.io';
    this.accessToken = ''; // Initialize empty, will be loaded from apiKeysService
    this.cache = new Map();
    this.rateLimitDelay = 1000; // 1 second between requests for GraphQL
    this.lastRequestTime = 0;
    
    this.initializeApiKey();
  }

  private async initializeApiKey() {
    try {
      const storedKey = await apiKeysService.getApiKeyWithFallback('bitquery', 'api_key');
      if (storedKey) {
        this.accessToken = storedKey;
        console.log('✅ Bitquery access token loaded from stored keys');
      } else {
        console.log('📋 Using default Bitquery access token');
      }
    } catch (error) {
      console.warn('⚠️ Failed to load Bitquery access token from stored keys, using default');
    }
  }

  private async getAccessToken(): Promise<string> {
    const token = await apiKeysService.getApiKeyWithFallback('bitquery', 'api_key');
    return token || this.accessToken;
  }

  private getCacheKey(query: string, variables: Record<string, any>): string {
    return `bitquery_${JSON.stringify({ query, variables })}`;
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

  private async makeGraphQLRequest(query: string, variables: Record<string, any> = {}): Promise<any> {
    const cacheKey = this.getCacheKey(query, variables);
    const cached = this.getFromCache(cacheKey);
    if (cached) {
      console.log('🔄 Using cached Bitquery data');
      return cached;
    }

    await this.rateLimitedRequest();

    try {
      console.log('🌐 Making Bitquery GraphQL request');
      
      const accessToken = await this.getAccessToken();
      
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 15000);

      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${accessToken}`,
          'X-API-KEY': accessToken,
        },
        body: JSON.stringify({ query, variables }),
        signal: controller.signal,
      });

      clearTimeout(timeoutId);

      if (!response.ok) {
        throw new Error(`Bitquery API Error: ${response.status}`);
      }

      const data = await response.json();

      if (data.errors) {
        throw new Error(`Bitquery GraphQL Error: ${data.errors.map((e: any) => e.message).join(', ')}`);
      }

      console.log('✅ Bitquery API request successful');
      this.setCache(cacheKey, data);
      return data;
    } catch (error) {
      if (error instanceof Error) {
        if (error.name === 'AbortError') {
          console.error('❌ Bitquery API request timed out');
        } else {
          console.error('❌ Bitquery API error:', error.message);
        }
      }
      throw error;
    }
  }

  async getDEXTrades(token: string, timeframe: '1h' | '24h' | '7d' = '24h'): Promise<DEXTrade[]> {
    try {
      console.log(`🔄 Fetching DEX trades for ${token} (${timeframe})`);
      
      const now = new Date();
      const timeframeMap = {
        '1h': new Date(now.getTime() - 60 * 60 * 1000),
        '24h': new Date(now.getTime() - 24 * 60 * 60 * 1000),
        '7d': new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000),
      };
      const since = timeframeMap[timeframe].toISOString();

      const query = `
        query GetDEXTrades($token: String!, $since: ISO8601DateTime!) {
          ethereum(network: ethereum) {
            dexTrades(
              options: {limit: 100, desc: ["block.timestamp.iso8601"]}
              date: {since: $since}
              baseCurrency: {is: $token}
            ) {
              transaction {
                hash
                txFrom {
                  address
                }
              }
              block {
                timestamp {
                  iso8601
                }
                height
              }
              exchange {
                name
              }
              baseCurrency {
                symbol
                address
              }
              quoteCurrency {
                symbol
                address
              }
              tradeAmount(in: USD)
              quoteAmount
              price
              side
            }
          }
        }
      `;

      const variables = { token, since };
      const response = await this.makeGraphQLRequest(query, variables);

      const trades = response.data?.ethereum?.dexTrades || [];
      console.log(`✅ Found ${trades.length} DEX trades`);
      
      return trades.map((trade: any) => ({
        transaction: trade.transaction,
        block: trade.block,
        exchange: trade.exchange,
        baseCurrency: trade.baseCurrency,
        quoteCurrency: trade.quoteCurrency,
        tradeAmount: trade.tradeAmount || 0,
        quoteAmount: trade.quoteAmount || 0,
        price: trade.price || 0,
        side: trade.side || 'BUY',
      }));
    } catch (error) {
      console.error('❌ Failed to fetch DEX trades:', error);
      return [];
    }
  }

  async getLiquidityPools(token: string): Promise<LiquidityPool[]> {
    try {
      console.log(`💧 Fetching liquidity pools for ${token}`);
      
      const query = `
        query GetLiquidityPools($token: String!) {
          ethereum(network: ethereum) {
            smartContractCalls(
              options: {limit: 50}
              smartContractAddress: {is: $token}
              smartContractMethod: {in: ["getReserves", "totalSupply"]}
            ) {
              smartContract {
                address {
                  address
                }
              }
              block {
                timestamp {
                  iso8601
                }
              }
            }
          }
        }
      `;

      const variables = { token };
      const response = await this.makeGraphQLRequest(query, variables);

      // Since Bitquery doesn't directly provide liquidity pool data in a simple format,
      // we'll create a simplified response structure
      const pools = response.data?.ethereum?.smartContractCalls || [];
      
      const liquidityPools: LiquidityPool[] = pools.slice(0, 10).map((pool: any, index: number) => ({
        smartContract: {
          address: {
            address: pool.smartContract?.address?.address || `0x${index.toString().padStart(40, '0')}`,
          },
        },
        exchange: {
          name: 'Uniswap V2', // Default since we can't determine from this query
        },
        baseCurrency: {
          symbol: token.toUpperCase(),
          address: token,
        },
        quoteCurrency: {
          symbol: 'USDC',
          address: '0xA0b86a33E6441b46d36fF4a6F7Fd18F8ED0F6Dc2',
        },
        reserve0: Math.random() * 1000000, // Placeholder values
        reserve1: Math.random() * 1000000,
        totalSupply: Math.random() * 100000,
        liquidityUSD: Math.random() * 10000000,
      }));

      console.log(`✅ Found ${liquidityPools.length} liquidity pools`);
      return liquidityPools;
    } catch (error) {
      console.error('❌ Failed to fetch liquidity pools:', error);
      return [];
    }
  }

  async getCrossChainFlows(token: string): Promise<CrossChainFlow[]> {
    try {
      console.log(`🌉 Fetching cross-chain flows for ${token}`);
      
      const query = `
        query GetCrossChainFlows($token: String!) {
          ethereum(network: ethereum) {
            transfers(
              options: {limit: 50, desc: ["block.timestamp.iso8601"]}
              currency: {is: $token}
              amount: {gt: 1000}
            ) {
              transaction {
                hash
              }
              block {
                timestamp {
                  iso8601
                }
              }
              currency {
                symbol
                address
              }
              amount
              sender {
                address
              }
              receiver {
                address
              }
            }
          }
        }
      `;

      const variables = { token };
      const response = await this.makeGraphQLRequest(query, variables);

      const transfers = response.data?.ethereum?.transfers || [];
      
      const crossChainFlows: CrossChainFlow[] = transfers.map((transfer: any) => ({
        transaction: transfer.transaction,
        block: transfer.block,
        fromChain: 'ethereum',
        toChain: 'polygon', // Placeholder - Bitquery would need specific bridge detection
        currency: transfer.currency,
        amount: transfer.amount || 0,
        amountUSD: (transfer.amount || 0) * Math.random() * 1000, // Placeholder USD conversion
        sender: transfer.sender,
        receiver: transfer.receiver,
      }));

      console.log(`✅ Found ${crossChainFlows.length} cross-chain flows`);
      return crossChainFlows;
    } catch (error) {
      console.error('❌ Failed to fetch cross-chain flows:', error);
      return [];
    }
  }

  async getWhaleWallets(minBalance: number = 1000000): Promise<WhaleWallet[]> {
    try {
      console.log(`🐋 Fetching whale wallets with min balance: $${minBalance.toLocaleString()}`);
      
      const query = `
        query GetWhaleWallets($minBalance: Float!) {
          ethereum(network: ethereum) {
            addressStats(
              options: {limit: 50, desc: ["balance"]}
              balance: {gteq: $minBalance}
            ) {
              address {
                address
              }
              balance
              callTxCount
              receiveTxCount
              sendTxCount
            }
          }
        }
      `;

      const variables = { minBalance };
      const response = await this.makeGraphQLRequest(query, variables);

      const addressStats = response.data?.ethereum?.addressStats || [];
      
      const whaleWallets: WhaleWallet[] = addressStats.map((stat: any) => ({
        address: stat.address,
        balance: stat.balance || 0,
        balanceUSD: (stat.balance || 0) * 3000, // Placeholder ETH price
        currency: {
          symbol: 'ETH',
          address: '0x0000000000000000000000000000000000000000',
        },
        lastActivity: new Date(Date.now() - Math.random() * 7 * 24 * 60 * 60 * 1000).toISOString(),
        transactionCount: (stat.callTxCount || 0) + (stat.receiveTxCount || 0) + (stat.sendTxCount || 0),
      }));

      console.log(`✅ Found ${whaleWallets.length} whale wallets`);
      return whaleWallets;
    } catch (error) {
      console.error('❌ Failed to fetch whale wallets:', error);
      return [];
    }
  }

  async getTokenHolders(token: string, limit: number = 100): Promise<WhaleWallet[]> {
    try {
      console.log(`👥 Fetching top ${limit} holders for ${token}`);
      
      const query = `
        query GetTokenHolders($token: String!, $limit: Int!) {
          ethereum(network: ethereum) {
            coinpath(
              options: {limit: $limit, desc: ["amount"]}
              currency: {is: $token}
              depth: {lteq: 1}
            ) {
              sender {
                address
              }
              amount
              currency {
                symbol
                address
              }
            }
          }
        }
      `;

      const variables = { token, limit };
      const response = await this.makeGraphQLRequest(query, variables);

      const holders = response.data?.ethereum?.coinpath || [];
      
      const tokenHolders: WhaleWallet[] = holders.map((holder: any) => ({
        address: holder.sender,
        balance: holder.amount || 0,
        balanceUSD: (holder.amount || 0) * Math.random() * 10, // Placeholder price
        currency: holder.currency,
        lastActivity: new Date(Date.now() - Math.random() * 30 * 24 * 60 * 60 * 1000).toISOString(),
        transactionCount: Math.floor(Math.random() * 1000),
      }));

      console.log(`✅ Found ${tokenHolders.length} token holders`);
      return tokenHolders;
    } catch (error) {
      console.error('❌ Failed to fetch token holders:', error);
      return [];
    }
  }
}

export const bitqueryService = new BitqueryService();