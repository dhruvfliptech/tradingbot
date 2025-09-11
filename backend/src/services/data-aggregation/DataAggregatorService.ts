import { EventEmitter } from 'events';
import logger from '../../utils/logger';
import { CacheService } from './CacheService';
import { RateLimiter } from './RateLimiter';
import { EtherscanClient } from './clients/EtherscanClient';
import { BitqueryClient } from './clients/BitqueryClient';
import { CovalentClient } from './clients/CovalentClient';
import { CoinglassClient } from './clients/CoinglassClient';
import { BinanceClient } from './clients/BinanceClient';
import { CryptoQuantClient } from './clients/CryptoQuantClient';
import { DataNormalizer } from './DataNormalizer';

export interface DataAggregatorConfig {
  apis: {
    etherscan: {
      apiKey: string;
      endpoint: string;
      rateLimit: number; // requests per second
    };
    bitquery: {
      apiKey?: string;
      endpoint: string;
      rateLimit: number;
    };
    covalent: {
      apiKey: string;
      endpoint: string;
      rateLimit: number;
    };
    coinglass: {
      endpoint: string;
      rateLimit: number;
    };
    binance: {
      endpoint: string;
      rateLimit: number;
    };
    cryptoquant: {
      apiKey?: string;
      endpoint: string;
      rateLimit: number;
    };
  };
  cache: {
    ttl: {
      onchain: number; // seconds
      funding: number;
      whale: number;
      market: number;
    };
    maxSize: number;
  };
  fallback: {
    maxRetries: number;
    retryDelay: number; // milliseconds
    timeout: number; // milliseconds
  };
}

export interface OnChainData {
  address: string;
  balance: number;
  transactions: Transaction[];
  tokenHoldings: TokenHolding[];
  lastActivity: Date;
  riskScore: number;
}

export interface Transaction {
  hash: string;
  from: string;
  to: string;
  value: number;
  token?: string;
  timestamp: Date;
  gasUsed: number;
  gasPrice: number;
  type: 'transfer' | 'swap' | 'deposit' | 'withdrawal';
}

export interface TokenHolding {
  token: string;
  symbol: string;
  balance: number;
  valueUsd: number;
  percentage: number;
}

export interface FundingRateData {
  symbol: string;
  rate: number;
  nextFundingTime: Date;
  predictedRate?: number;
  historicalAverage: number;
  source: string;
  timestamp: Date;
}

export interface WhaleAlert {
  id: string;
  symbol: string;
  amount: number;
  amountUsd: number;
  from: string;
  to: string;
  txHash: string;
  timestamp: Date;
  type: 'transfer' | 'exchange_inflow' | 'exchange_outflow';
  exchange?: string;
  confidence: number;
}

export interface SmartMoneyFlow {
  address: string;
  label: string;
  action: 'buy' | 'sell' | 'hold';
  amount: number;
  amountUsd: number;
  symbol: string;
  timestamp: Date;
  confidence: number;
  source: string;
}

export interface LiquidationData {
  symbol: string;
  side: 'long' | 'short';
  amount: number;
  amountUsd: number;
  price: number;
  timestamp: Date;
  exchange: string;
}

export interface AggregatedData {
  onchain?: OnChainData[];
  funding?: FundingRateData[];
  whales?: WhaleAlert[];
  smartMoney?: SmartMoneyFlow[];
  liquidations?: LiquidationData[];
  metadata: {
    timestamp: Date;
    sources: string[];
    reliability: number;
  };
}

export class DataAggregatorService extends EventEmitter {
  private config: DataAggregatorConfig;
  private cache: CacheService;
  private rateLimiter: RateLimiter;
  private clients: {
    etherscan: EtherscanClient;
    bitquery: BitqueryClient;
    covalent: CovalentClient;
    coinglass: CoinglassClient;
    binance: BinanceClient;
    cryptoquant: CryptoQuantClient;
  };
  private normalizer: DataNormalizer;
  private isInitialized: boolean = false;

  constructor(config: DataAggregatorConfig) {
    super();
    this.config = config;
    this.cache = new CacheService(config.cache);
    this.rateLimiter = new RateLimiter();
    this.normalizer = new DataNormalizer();

    // Initialize API clients
    this.clients = {
      etherscan: new EtherscanClient(config.apis.etherscan),
      bitquery: new BitqueryClient(config.apis.bitquery),
      covalent: new CovalentClient(config.apis.covalent),
      coinglass: new CoinglassClient(config.apis.coinglass),
      binance: new BinanceClient(config.apis.binance),
      cryptoquant: new CryptoQuantClient(config.apis.cryptoquant)
    };
  }

  async initialize(): Promise<void> {
    try {
      logger.info('Initializing DataAggregatorService...');

      // Initialize cache
      await this.cache.initialize();

      // Setup rate limiters for each API
      Object.entries(this.config.apis).forEach(([name, config]) => {
        this.rateLimiter.addLimit(`${name}`, config.rateLimit, 1000); // per second
      });

      // Test API connections
      await this.testConnections();

      this.isInitialized = true;
      logger.info('DataAggregatorService initialized successfully');

    } catch (error) {
      logger.error('Failed to initialize DataAggregatorService:', error);
      throw error;
    }
  }

  async aggregateData(symbols: string[], options: {
    includeOnchain?: boolean;
    includeFunding?: boolean;
    includeWhales?: boolean;
    includeSmartMoney?: boolean;
    includeLiquidations?: boolean;
    forceRefresh?: boolean;
  } = {}): Promise<AggregatedData> {
    if (!this.isInitialized) {
      throw new Error('DataAggregatorService not initialized');
    }

    try {
      logger.info(`Aggregating data for symbols: ${symbols.join(', ')}`);

      const cacheKey = this.generateCacheKey(symbols, options);
      
      // Check cache first unless force refresh
      if (!options.forceRefresh) {
        const cached = await this.cache.get<AggregatedData>(cacheKey);
        if (cached) {
          logger.debug('Returning cached aggregated data');
          return cached;
        }
      }

      const results: AggregatedData = {
        metadata: {
          timestamp: new Date(),
          sources: [],
          reliability: 0
        }
      };

      const promises: Promise<void>[] = [];

      // Fetch on-chain data
      if (options.includeOnchain !== false) {
        promises.push(this.fetchOnChainData(symbols).then(data => {
          if (data.length > 0) {
            results.onchain = data;
            results.metadata.sources.push('onchain');
          }
        }).catch(error => {
          logger.warn('Failed to fetch on-chain data:', error);
        }));
      }

      // Fetch funding rate data
      if (options.includeFunding !== false) {
        promises.push(this.fetchFundingRates(symbols).then(data => {
          if (data.length > 0) {
            results.funding = data;
            results.metadata.sources.push('funding');
          }
        }).catch(error => {
          logger.warn('Failed to fetch funding rates:', error);
        }));
      }

      // Fetch whale alerts
      if (options.includeWhales !== false) {
        promises.push(this.fetchWhaleAlerts(symbols).then(data => {
          if (data.length > 0) {
            results.whales = data;
            results.metadata.sources.push('whales');
          }
        }).catch(error => {
          logger.warn('Failed to fetch whale alerts:', error);
        }));
      }

      // Fetch smart money flows
      if (options.includeSmartMoney !== false) {
        promises.push(this.fetchSmartMoneyFlows(symbols).then(data => {
          if (data.length > 0) {
            results.smartMoney = data;
            results.metadata.sources.push('smart_money');
          }
        }).catch(error => {
          logger.warn('Failed to fetch smart money flows:', error);
        }));
      }

      // Fetch liquidation data
      if (options.includeLiquidations !== false) {
        promises.push(this.fetchLiquidations(symbols).then(data => {
          if (data.length > 0) {
            results.liquidations = data;
            results.metadata.sources.push('liquidations');
          }
        }).catch(error => {
          logger.warn('Failed to fetch liquidations:', error);
        }));
      }

      // Wait for all data fetching to complete
      await Promise.allSettled(promises);

      // Calculate reliability score
      results.metadata.reliability = this.calculateReliability(results);

      // Cache results
      const ttl = this.determineCacheTtl(options);
      await this.cache.set(cacheKey, results, ttl);

      logger.info(`Data aggregation completed with ${results.metadata.sources.length} sources`);
      
      this.emit('data_aggregated', {
        symbols,
        sources: results.metadata.sources,
        reliability: results.metadata.reliability
      });

      return results;

    } catch (error) {
      logger.error('Error aggregating data:', error);
      throw error;
    }
  }

  private async fetchOnChainData(symbols: string[]): Promise<OnChainData[]> {
    const results: OnChainData[] = [];

    for (const symbol of symbols) {
      try {
        // Get token addresses and whale addresses for the symbol
        const addresses = await this.getRelevantAddresses(symbol);

        for (const address of addresses) {
          await this.rateLimiter.waitForToken('etherscan');
          
          const data = await this.clients.etherscan.getAddressData(address);
          if (data) {
            const normalized = this.normalizer.normalizeOnChainData(data, symbol);
            results.push(normalized);
          }

          // Also try Covalent for multi-chain support
          if (this.config.apis.covalent.apiKey) {
            await this.rateLimiter.waitForToken('covalent');
            
            const covalentData = await this.clients.covalent.getAddressData(address);
            if (covalentData && covalentData.address !== data?.address) {
              const normalized = this.normalizer.normalizeOnChainData(covalentData, symbol);
              results.push(normalized);
            }
          }
        }
      } catch (error) {
        logger.warn(`Failed to fetch on-chain data for ${symbol}:`, error);
      }
    }

    return results;
  }

  private async fetchFundingRates(symbols: string[]): Promise<FundingRateData[]> {
    const results: FundingRateData[] = [];

    for (const symbol of symbols) {
      try {
        // Fetch from Coinglass first (comprehensive)
        await this.rateLimiter.waitForToken('coinglass');
        const coinglassData = await this.clients.coinglass.getFundingRate(symbol);
        if (coinglassData) {
          results.push(this.normalizer.normalizeFundingRate(coinglassData, 'coinglass'));
        }

        // Fetch from Binance as backup/comparison
        await this.rateLimiter.waitForToken('binance');
        const binanceData = await this.clients.binance.getFundingRate(symbol);
        if (binanceData) {
          results.push(this.normalizer.normalizeFundingRate(binanceData, 'binance'));
        }

      } catch (error) {
        logger.warn(`Failed to fetch funding rates for ${symbol}:`, error);
      }
    }

    return results;
  }

  private async fetchWhaleAlerts(symbols: string[]): Promise<WhaleAlert[]> {
    const results: WhaleAlert[] = [];

    for (const symbol of symbols) {
      try {
        // Use Bitquery for DEX whale movements
        await this.rateLimiter.waitForToken('bitquery');
        const bitqueryWhales = await this.clients.bitquery.getWhaleTransfers(symbol);
        results.push(...bitqueryWhales.map(w => this.normalizer.normalizeWhaleAlert(w, 'bitquery')));

        // Use Etherscan for on-chain whale movements
        await this.rateLimiter.waitForToken('etherscan');
        const etherscanWhales = await this.clients.etherscan.getLargeTransactions(symbol);
        results.push(...etherscanWhales.map(w => this.normalizer.normalizeWhaleAlert(w, 'etherscan')));

      } catch (error) {
        logger.warn(`Failed to fetch whale alerts for ${symbol}:`, error);
      }
    }

    // Sort by amount and remove duplicates
    return this.deduplicateWhaleAlerts(results);
  }

  private async fetchSmartMoneyFlows(symbols: string[]): Promise<SmartMoneyFlow[]> {
    const results: SmartMoneyFlow[] = [];

    for (const symbol of symbols) {
      try {
        // Use Bitquery for smart money tracking
        await this.rateLimiter.waitForToken('bitquery');
        const smartMoneyData = await this.clients.bitquery.getSmartMoneyFlows(symbol);
        results.push(...smartMoneyData.map(d => this.normalizer.normalizeSmartMoneyFlow(d)));

        // Use CryptoQuant if available
        if (this.config.apis.cryptoquant.apiKey) {
          await this.rateLimiter.waitForToken('cryptoquant');
          const cryptoQuantData = await this.clients.cryptoquant.getSmartMoneyFlows(symbol);
          results.push(...cryptoQuantData.map(d => this.normalizer.normalizeSmartMoneyFlow(d)));
        }

      } catch (error) {
        logger.warn(`Failed to fetch smart money flows for ${symbol}:`, error);
      }
    }

    return results;
  }

  private async fetchLiquidations(symbols: string[]): Promise<LiquidationData[]> {
    const results: LiquidationData[] = [];

    for (const symbol of symbols) {
      try {
        // Fetch from Coinglass
        await this.rateLimiter.waitForToken('coinglass');
        const liquidationData = await this.clients.coinglass.getLiquidations(symbol);
        results.push(...liquidationData.map(d => this.normalizer.normalizeLiquidation(d)));

      } catch (error) {
        logger.warn(`Failed to fetch liquidations for ${symbol}:`, error);
      }
    }

    return results;
  }

  private async getRelevantAddresses(symbol: string): Promise<string[]> {
    // Cache key for addresses
    const cacheKey = `addresses:${symbol}`;
    const cached = await this.cache.get<string[]>(cacheKey);
    if (cached) return cached;

    try {
      // Get known whale addresses and smart contract addresses for the symbol
      const addresses: string[] = [];

      // Add known exchange addresses, whale addresses, etc.
      // This could be enhanced with a database of known addresses
      const knownAddresses = await this.getKnownAddresses(symbol);
      addresses.push(...knownAddresses);

      // Cache for 1 hour
      await this.cache.set(cacheKey, addresses, 3600);
      
      return addresses;

    } catch (error) {
      logger.error(`Failed to get relevant addresses for ${symbol}:`, error);
      return [];
    }
  }

  private async getKnownAddresses(symbol: string): Promise<string[]> {
    // This would typically come from a database or configuration
    // For now, return some example addresses based on symbol
    const addressMap: Record<string, string[]> = {
      'WBTC': ['0x2260FAC5E5542a773Aa44fBCfeDf7C193bc2C599'],
      'USDC': ['0xA0b86a33E6441c8C0F8aF8f6c38e5Cc9A6c98c18'],
      'USDT': ['0xdAC17F958D2ee523a2206206994597C13D831ec7'],
      'ETH': ['0x0000000000000000000000000000000000000000'] // ETH address
    };

    return addressMap[symbol] || [];
  }

  private deduplicateWhaleAlerts(alerts: WhaleAlert[]): WhaleAlert[] {
    const seen = new Set<string>();
    return alerts.filter(alert => {
      const key = `${alert.txHash}_${alert.timestamp.getTime()}`;
      if (seen.has(key)) return false;
      seen.add(key);
      return true;
    }).sort((a, b) => b.amountUsd - a.amountUsd);
  }

  private calculateReliability(data: AggregatedData): number {
    let score = 0;
    const maxScore = 100;

    // Source diversity (40%)
    const sourcesWeight = 40;
    const maxSources = 6; // We have 6 different data source types
    score += (data.metadata.sources.length / maxSources) * sourcesWeight;

    // Data freshness (30%)
    const freshnessWeight = 30;
    const now = new Date().getTime();
    const dataAge = now - data.metadata.timestamp.getTime();
    const maxAge = 5 * 60 * 1000; // 5 minutes
    const freshnessScore = Math.max(0, (maxAge - dataAge) / maxAge);
    score += freshnessScore * freshnessWeight;

    // Data completeness (30%)
    const completenessWeight = 30;
    let completenessCount = 0;
    if (data.onchain?.length) completenessCount++;
    if (data.funding?.length) completenessCount++;
    if (data.whales?.length) completenessCount++;
    if (data.smartMoney?.length) completenessCount++;
    if (data.liquidations?.length) completenessCount++;

    score += (completenessCount / 5) * completenessWeight;

    return Math.min(maxScore, Math.round(score));
  }

  private determineCacheTtl(options: any): number {
    // Return the shortest TTL based on requested data types
    const ttls = [];
    
    if (options.includeOnchain !== false) ttls.push(this.config.cache.ttl.onchain);
    if (options.includeFunding !== false) ttls.push(this.config.cache.ttl.funding);
    if (options.includeWhales !== false) ttls.push(this.config.cache.ttl.whale);
    if (options.includeSmartMoney !== false || options.includeLiquidations !== false) {
      ttls.push(this.config.cache.ttl.market);
    }

    return ttls.length > 0 ? Math.min(...ttls) : this.config.cache.ttl.market;
  }

  private generateCacheKey(symbols: string[], options: any): string {
    const optionsKey = Object.entries(options)
      .sort()
      .map(([k, v]) => `${k}:${v}`)
      .join('|');
    
    return `aggregated:${symbols.sort().join(',')}:${optionsKey}`;
  }

  private async testConnections(): Promise<void> {
    const tests = [
      { name: 'etherscan', test: () => this.clients.etherscan.testConnection() },
      { name: 'bitquery', test: () => this.clients.bitquery.testConnection() },
      { name: 'covalent', test: () => this.clients.covalent.testConnection() },
      { name: 'coinglass', test: () => this.clients.coinglass.testConnection() },
      { name: 'binance', test: () => this.clients.binance.testConnection() },
      { name: 'cryptoquant', test: () => this.clients.cryptoquant.testConnection() }
    ];

    const results = await Promise.allSettled(
      tests.map(async ({ name, test }) => {
        try {
          await test();
          logger.info(`✓ ${name} API connection successful`);
          return { name, status: 'connected' };
        } catch (error) {
          logger.warn(`✗ ${name} API connection failed:`, error.message);
          return { name, status: 'failed', error: error.message };
        }
      })
    );

    const connected = results.filter(r => r.status === 'fulfilled' && r.value.status === 'connected').length;
    logger.info(`API connections: ${connected}/${tests.length} successful`);

    if (connected === 0) {
      throw new Error('No API connections successful');
    }
  }

  // Integration methods for AdaptiveThreshold and Composer
  async enrichTradingSignal(signal: any): Promise<any> {
    try {
      const aggregatedData = await this.aggregateData([signal.symbol], {
        includeFunding: true,
        includeWhales: true,
        includeLiquidations: true
      });

      const enrichment: any = {
        onchain_activity: 0,
        whale_sentiment: 0,
        funding_bias: 0,
        liquidation_risk: 0
      };

      // Calculate on-chain activity score
      if (aggregatedData.onchain?.length) {
        const recentActivity = aggregatedData.onchain.filter(
          data => new Date().getTime() - data.lastActivity.getTime() < 24 * 60 * 60 * 1000
        );
        enrichment.onchain_activity = Math.min(100, recentActivity.length * 10);
      }

      // Calculate whale sentiment
      if (aggregatedData.whales?.length) {
        const recentWhales = aggregatedData.whales.filter(
          whale => new Date().getTime() - whale.timestamp.getTime() < 4 * 60 * 60 * 1000
        );
        const buyVolume = recentWhales.filter(w => w.type === 'exchange_inflow').reduce((sum, w) => sum + w.amountUsd, 0);
        const sellVolume = recentWhales.filter(w => w.type === 'exchange_outflow').reduce((sum, w) => sum + w.amountUsd, 0);
        
        if (buyVolume + sellVolume > 0) {
          enrichment.whale_sentiment = ((buyVolume - sellVolume) / (buyVolume + sellVolume)) * 100;
        }
      }

      // Calculate funding bias
      if (aggregatedData.funding?.length) {
        const avgFunding = aggregatedData.funding.reduce((sum, f) => sum + f.rate, 0) / aggregatedData.funding.length;
        enrichment.funding_bias = Math.max(-100, Math.min(100, avgFunding * 10000)); // Convert to basis points
      }

      // Calculate liquidation risk
      if (aggregatedData.liquidations?.length) {
        const recentLiqs = aggregatedData.liquidations.filter(
          liq => new Date().getTime() - liq.timestamp.getTime() < 60 * 60 * 1000
        );
        const longLiqs = recentLiqs.filter(l => l.side === 'long').reduce((sum, l) => sum + l.amountUsd, 0);
        const shortLiqs = recentLiqs.filter(l => l.side === 'short').reduce((sum, l) => sum + l.amountUsd, 0);
        
        if (longLiqs + shortLiqs > 0) {
          enrichment.liquidation_risk = Math.min(100, (longLiqs + shortLiqs) / 1000000); // Scale by millions
        }
      }

      return {
        ...signal,
        enrichment,
        data_sources: aggregatedData.metadata.sources,
        reliability: aggregatedData.metadata.reliability
      };

    } catch (error) {
      logger.error('Failed to enrich trading signal:', error);
      return signal; // Return original signal if enrichment fails
    }
  }

  async getMarketContext(symbols: string[]): Promise<{
    overall_sentiment: number;
    risk_level: 'low' | 'medium' | 'high';
    trending_symbols: string[];
    market_regime: 'bull' | 'bear' | 'sideways' | 'volatile';
  }> {
    try {
      const aggregatedData = await this.aggregateData(symbols, {
        includeWhales: true,
        includeFunding: true,
        includeLiquidations: true
      });

      // Calculate overall sentiment
      let sentimentScore = 0;
      let sentimentCount = 0;

      if (aggregatedData.funding?.length) {
        const avgFunding = aggregatedData.funding.reduce((sum, f) => sum + f.rate, 0) / aggregatedData.funding.length;
        sentimentScore += avgFunding > 0 ? -20 : 20; // Negative funding = bullish
        sentimentCount++;
      }

      if (aggregatedData.whales?.length) {
        const inflowValue = aggregatedData.whales.filter(w => w.type === 'exchange_inflow').reduce((sum, w) => sum + w.amountUsd, 0);
        const outflowValue = aggregatedData.whales.filter(w => w.type === 'exchange_outflow').reduce((sum, w) => sum + w.amountUsd, 0);
        
        if (inflowValue + outflowValue > 0) {
          const whaleRatio = (outflowValue - inflowValue) / (inflowValue + outflowValue);
          sentimentScore += whaleRatio * 50;
          sentimentCount++;
        }
      }

      const overallSentiment = sentimentCount > 0 ? sentimentScore / sentimentCount : 0;

      // Determine risk level
      let riskLevel: 'low' | 'medium' | 'high' = 'low';
      if (aggregatedData.liquidations?.length) {
        const totalLiquidations = aggregatedData.liquidations.reduce((sum, l) => sum + l.amountUsd, 0);
        if (totalLiquidations > 10000000) riskLevel = 'high';
        else if (totalLiquidations > 1000000) riskLevel = 'medium';
      }

      // Determine market regime (simplified)
      let marketRegime: 'bull' | 'bear' | 'sideways' | 'volatile' = 'sideways';
      if (Math.abs(overallSentiment) > 30) {
        marketRegime = overallSentiment > 0 ? 'bull' : 'bear';
      }
      if (riskLevel === 'high') marketRegime = 'volatile';

      return {
        overall_sentiment: Math.round(overallSentiment),
        risk_level: riskLevel,
        trending_symbols: symbols, // Could be enhanced with actual trending analysis
        market_regime: marketRegime
      };

    } catch (error) {
      logger.error('Failed to get market context:', error);
      return {
        overall_sentiment: 0,
        risk_level: 'medium',
        trending_symbols: [],
        market_regime: 'sideways'
      };
    }
  }

  async getHealthStatus(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    services: Array<{
      name: string;
      status: 'up' | 'down';
      latency?: number;
      error?: string;
    }>;
    cache: {
      hitRate: number;
      size: number;
    };
    rateLimits: Array<{
      service: string;
      remaining: number;
      resetTime: Date;
    }>;
  }> {
    const services = await Promise.allSettled([
      this.clients.etherscan.testConnection(),
      this.clients.bitquery.testConnection(),
      this.clients.covalent.testConnection(),
      this.clients.coinglass.testConnection(),
      this.clients.binance.testConnection(),
      this.clients.cryptoquant.testConnection()
    ]);

    const serviceNames = ['etherscan', 'bitquery', 'covalent', 'coinglass', 'binance', 'cryptoquant'];
    const serviceStatuses = services.map((result, index) => ({
      name: serviceNames[index],
      status: result.status === 'fulfilled' ? 'up' as const : 'down' as const,
      error: result.status === 'rejected' ? result.reason?.message : undefined
    }));

    const upServices = serviceStatuses.filter(s => s.status === 'up').length;
    const totalServices = serviceStatuses.length;

    let overallStatus: 'healthy' | 'degraded' | 'unhealthy';
    if (upServices === totalServices) overallStatus = 'healthy';
    else if (upServices > totalServices / 2) overallStatus = 'degraded';
    else overallStatus = 'unhealthy';

    return {
      status: overallStatus,
      services: serviceStatuses,
      cache: await this.cache.getStats(),
      rateLimits: this.rateLimiter.getStatus()
    };
  }

  async shutdown(): Promise<void> {
    logger.info('Shutting down DataAggregatorService...');
    
    try {
      await this.cache.shutdown();
      this.removeAllListeners();
      logger.info('DataAggregatorService shutdown complete');
    } catch (error) {
      logger.error('Error during DataAggregatorService shutdown:', error);
    }
  }
}