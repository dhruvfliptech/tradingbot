import { jest, describe, beforeEach, afterEach, it, expect } from '@jest/globals';
import { DataAggregatorService } from '../DataAggregatorService';
import { DataAggregatorConfigManager } from '../config/DataAggregatorConfig';
import { DataAggregatorIntegration } from '../integration/DataAggregatorIntegration';
import { CacheService } from '../CacheService';
import { RateLimiter } from '../RateLimiter';
import { DataNormalizer } from '../DataNormalizer';
import { DataAggregatorErrorHandler } from '../ErrorHandler';

// Mock external dependencies
jest.mock('node-fetch');
jest.mock('../../../utils/logger', () => ({
  default: {
    info: jest.fn(),
    warn: jest.fn(),
    error: jest.fn(),
    debug: jest.fn()
  }
}));

// Mock environment variables
const mockEnv = {
  ETHERSCAN_API_KEY: 'test_etherscan_key',
  COVALENT_API_KEY: 'test_covalent_key',
  BITQUERY_API_KEY: 'test_bitquery_key',
  CRYPTOQUANT_API_KEY: 'test_cryptoquant_key',
  ENABLE_ETHERSCAN: 'true',
  ENABLE_COVALENT: 'true',
  ENABLE_BITQUERY: 'true',
  ENABLE_COINGLASS: 'true',
  ENABLE_BINANCE: 'true',
  ENABLE_CRYPTOQUANT: 'false',
  CACHE_TTL_ONCHAIN: '300',
  CACHE_TTL_FUNDING: '60',
  CACHE_TTL_WHALE: '30',
  CACHE_TTL_MARKET: '60',
  CACHE_MAX_SIZE: '1000',
  MAX_RETRIES: '3',
  RETRY_DELAY: '1000',
  REQUEST_TIMEOUT: '10000'
};

describe('DataAggregatorService', () => {
  let dataAggregator: DataAggregatorService;
  let configManager: DataAggregatorConfigManager;

  beforeEach(() => {
    // Set up environment variables
    Object.entries(mockEnv).forEach(([key, value]) => {
      process.env[key] = value;
    });

    configManager = new DataAggregatorConfigManager();
    const config = configManager.getConfiguration();
    dataAggregator = new DataAggregatorService(config);
  });

  afterEach(() => {
    // Clean up environment variables
    Object.keys(mockEnv).forEach(key => {
      delete process.env[key];
    });
  });

  describe('Configuration Management', () => {
    it('should load configuration from environment variables', () => {
      const config = configManager.getConfiguration();
      
      expect(config.apis.etherscan.apiKey).toBe('test_etherscan_key');
      expect(config.apis.etherscan.enabled).toBe(true);
      expect(config.apis.covalent.apiKey).toBe('test_covalent_key');
      expect(config.apis.cryptoquant.enabled).toBe(false);
      expect(config.cache.ttl.onchain).toBe(300);
      expect(config.fallback.maxRetries).toBe(3);
    });

    it('should validate configuration correctly', () => {
      expect(() => new DataAggregatorConfigManager()).not.toThrow();
    });

    it('should throw error for invalid configuration', () => {
      delete process.env.ETHERSCAN_API_KEY;
      
      expect(() => new DataAggregatorConfigManager()).toThrow();
    });

    it('should provide health check for configuration', async () => {
      const health = await configManager.healthCheck();
      
      expect(health.status).toBeDefined();
      expect(Array.isArray(health.issues)).toBe(true);
      expect(Array.isArray(health.recommendations)).toBe(true);
    });
  });

  describe('Cache Service', () => {
    let cache: CacheService;

    beforeEach(() => {
      const cacheConfig = {
        maxSize: 100,
        ttl: {
          onchain: 300,
          funding: 60,
          whale: 30,
          market: 60
        }
      };
      cache = new CacheService(cacheConfig);
    });

    afterEach(async () => {
      await cache.shutdown();
    });

    it('should initialize cache service', async () => {
      await cache.initialize();
      expect(true).toBe(true); // No errors thrown
    });

    it('should store and retrieve data', async () => {
      await cache.initialize();
      
      const testData = { test: 'data' };
      await cache.set('test_key', testData, 60);
      
      const retrieved = await cache.get('test_key');
      expect(retrieved).toEqual(testData);
    });

    it('should handle TTL expiration', async () => {
      await cache.initialize();
      
      const testData = { test: 'data' };
      await cache.set('test_key', testData, 0.01); // 10ms TTL
      
      // Wait for expiration
      await new Promise(resolve => setTimeout(resolve, 20));
      
      const retrieved = await cache.get('test_key');
      expect(retrieved).toBeNull();
    });

    it('should provide cache statistics', async () => {
      await cache.initialize();
      
      await cache.set('key1', 'data1', 60);
      await cache.get('key1');
      await cache.get('nonexistent');
      
      const stats = await cache.getStats();
      expect(stats.totalHits).toBe(1);
      expect(stats.totalMisses).toBe(1);
      expect(stats.hitRate).toBe(50);
    });
  });

  describe('Rate Limiter', () => {
    let rateLimiter: RateLimiter;

    beforeEach(() => {
      rateLimiter = new RateLimiter();
    });

    it('should add rate limits', () => {
      rateLimiter.addLimit('test_service', 5, 1000);
      
      const status = rateLimiter.getStatus();
      const testServiceStatus = status.find(s => s.service === 'test_service');
      
      expect(testServiceStatus).toBeDefined();
      expect(testServiceStatus!.remaining).toBeGreaterThan(0);
    });

    it('should enforce rate limits', async () => {
      rateLimiter.addLimit('test_service', 1, 1000); // 1 request per second
      
      // First request should succeed immediately
      const start = Date.now();
      await rateLimiter.waitForToken('test_service');
      const firstRequestTime = Date.now() - start;
      
      expect(firstRequestTime).toBeLessThan(100); // Should be immediate
      
      // Second request should be delayed
      const start2 = Date.now();
      await rateLimiter.waitForToken('test_service');
      const secondRequestTime = Date.now() - start2;
      
      expect(secondRequestTime).toBeGreaterThan(900); // Should wait ~1 second
    });

    it('should provide health status', () => {
      rateLimiter.addLimit('test_service', 5, 1000);
      
      const health = rateLimiter.getHealthStatus();
      expect(health.status).toBeDefined();
      expect(Array.isArray(health.issues)).toBe(true);
    });
  });

  describe('Data Normalizer', () => {
    let normalizer: DataNormalizer;

    beforeEach(() => {
      normalizer = new DataNormalizer();
    });

    it('should normalize Etherscan data', () => {
      const rawData = {
        address: '0x742d35Cc7bF9D4C3D9E4d0b8aEf3b6b3f6C3A1b2',
        balance: '1000000000000000000', // 1 ETH in wei
        transactions: [
          {
            hash: '0x123...',
            from: '0xabc...',
            to: '0xdef...',
            value: '500000000000000000',
            timeStamp: '1640995200'
          }
        ],
        tokenHoldings: [],
        lastActivity: new Date('2023-01-01')
      };

      const normalized = normalizer.normalizeOnChainData(rawData, 'ETH');
      
      expect(normalized.address).toBe(rawData.address.toLowerCase());
      expect(normalized.balance).toBe(1); // Converted from wei
      expect(normalized.transactions).toHaveLength(1);
      expect(normalized.riskScore).toBeGreaterThanOrEqual(0);
      expect(normalized.riskScore).toBeLessThanOrEqual(100);
    });

    it('should normalize funding rate data', () => {
      const rawData = {
        symbol: 'BTCUSDT',
        rate: '0.0001',
        nextFundingTime: Date.now() + 8 * 60 * 60 * 1000,
        markPrice: '45000',
        estimatedRate: '0.00009'
      };

      const normalized = normalizer.normalizeFundingRate(rawData, 'binance');
      
      expect(normalized.symbol).toBe('BTC');
      expect(normalized.rate).toBe(0.0001);
      expect(normalized.source).toBe('binance');
      expect(normalized.nextFundingTime).toBeInstanceOf(Date);
    });

    it('should validate normalized data', () => {
      const validOnChainData = {
        address: '0x123...',
        balance: 1.5,
        transactions: [],
        tokenHoldings: [],
        lastActivity: new Date(),
        riskScore: 50
      };

      const isValid = normalizer.validateNormalizedData(validOnChainData, 'onchain');
      expect(isValid).toBe(true);
    });
  });

  describe('Error Handler', () => {
    let errorHandler: DataAggregatorErrorHandler;

    beforeEach(() => {
      errorHandler = new DataAggregatorErrorHandler();
    });

    it('should retry failed operations', async () => {
      let attempts = 0;
      const mockFunction = jest.fn().mockImplementation(() => {
        attempts++;
        if (attempts < 3) {
          throw new Error('Temporary failure');
        }
        return 'success';
      });

      const result = await errorHandler.executeWithFallback(
        mockFunction,
        {
          service: 'test',
          method: 'testMethod',
          maxAttempts: 3
        }
      );

      expect(result).toBe('success');
      expect(attempts).toBe(3);
    });

    it('should handle circuit breaker', async () => {
      const mockFunction = jest.fn().mockRejectedValue(new Error('Service down'));

      // Trigger multiple failures to open circuit breaker
      for (let i = 0; i < 6; i++) {
        try {
          await errorHandler.executeWithFallback(
            mockFunction,
            {
              service: 'test',
              method: 'testMethod',
              maxAttempts: 1
            }
          );
        } catch (error) {
          // Expected to fail
        }
      }

      const health = errorHandler.getHealthStatus();
      const circuitBreakerState = health.circuitBreakers['test'];
      
      expect(circuitBreakerState?.isOpen).toBe(true);
    });

    it('should not retry non-retryable errors', async () => {
      let attempts = 0;
      const mockFunction = jest.fn().mockImplementation(() => {
        attempts++;
        throw new Error('Invalid API key');
      });

      try {
        await errorHandler.executeWithFallback(
          mockFunction,
          {
            service: 'test',
            method: 'testMethod',
            maxAttempts: 3
          }
        );
      } catch (error) {
        expect(error.message).toContain('Invalid API key');
      }

      expect(attempts).toBe(1); // Should not retry
    });
  });

  describe('DataAggregator Integration', () => {
    let integration: DataAggregatorIntegration;

    beforeEach(() => {
      integration = new DataAggregatorIntegration();
    });

    afterEach(async () => {
      if (integration) {
        await integration.shutdown();
      }
    });

    it('should initialize integration', async () => {
      await integration.initialize();
      
      const health = await integration.getHealthStatus();
      expect(health.integration.initialized).toBe(true);
    });

    it('should enrich trading signals', async () => {
      await integration.initialize();
      
      const mockSignal = {
        symbol: 'BTC',
        confidence: 75,
        action: 'BUY',
        indicators: { rsi: 30 },
        changePercent: 5.2,
        volume: 1000000
      };

      const enrichment = await integration.enrichTradingSignal(mockSignal);
      
      expect(enrichment).toBeDefined();
      expect(enrichment.confidence_boost).toBeDefined();
      expect(enrichment.market_regime).toBeDefined();
      expect(Array.isArray(enrichment.data_sources)).toBe(true);
    });

    it('should provide market context', async () => {
      await integration.initialize();
      
      const context = await integration.getMarketContext(['BTC', 'ETH']);
      
      expect(context).toBeDefined();
      expect(context.overall_sentiment).toBeDefined();
      expect(['low', 'medium', 'high']).toContain(context.risk_level);
      expect(['bull', 'bear', 'sideways', 'volatile']).toContain(context.market_regime);
    });

    it('should prepare adaptive threshold input', async () => {
      await integration.initialize();
      
      const mockSignal = {
        symbol: 'BTC',
        confidence: 75,
        action: 'BUY',
        indicators: { rsi: 30 },
        changePercent: 5.2,
        volume: 1000000
      };

      const input = await integration.prepareAdaptiveThresholdInput(mockSignal);
      
      expect(input.signal).toBeDefined();
      expect(input.enrichment).toBeDefined();
      expect(input.market_context).toBeDefined();
      expect(input.signal.symbol).toBe('BTC');
    });
  });

  describe('Integration Tests', () => {
    beforeEach(async () => {
      await dataAggregator.initialize();
    });

    afterEach(async () => {
      await dataAggregator.shutdown();
    });

    it('should aggregate data from multiple sources', async () => {
      // Mock successful API responses
      const mockFetch = jest.fn()
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            status: '1',
            result: '1000000000000000000' // 1 ETH
          })
        })
        .mockResolvedValueOnce({
          ok: true,
          json: async () => ({
            code: 0,
            data: {
              BTC: {
                dataMap: {
                  binance: {
                    rate: '0.0001',
                    nextFundingTime: Date.now() + 8 * 60 * 60 * 1000
                  }
                }
              }
            }
          })
        });

      require('node-fetch').default = mockFetch;

      const result = await dataAggregator.aggregateData(['BTC'], {
        includeOnchain: true,
        includeFunding: true,
        includeWhales: false,
        includeSmartMoney: false,
        includeLiquidations: false
      });

      expect(result).toBeDefined();
      expect(result.metadata.sources.length).toBeGreaterThan(0);
      expect(result.metadata.reliability).toBeGreaterThan(0);
      expect(result.metadata.timestamp).toBeInstanceOf(Date);
    });

    it('should handle API failures gracefully', async () => {
      // Mock API failures
      const mockFetch = jest.fn().mockRejectedValue(new Error('API Error'));
      require('node-fetch').default = mockFetch;

      const result = await dataAggregator.aggregateData(['BTC'], {
        includeOnchain: true,
        includeFunding: true
      });

      // Should return result even with failures (using fallbacks)
      expect(result).toBeDefined();
      expect(result.metadata).toBeDefined();
    });

    it('should respect rate limits', async () => {
      const startTime = Date.now();
      
      // Make multiple requests that should trigger rate limiting
      const promises = Array(3).fill(null).map(() =>
        dataAggregator.aggregateData(['BTC'], { includeFunding: true })
      );

      await Promise.all(promises);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should take some time due to rate limiting
      expect(duration).toBeGreaterThan(100);
    });

    it('should cache results effectively', async () => {
      // Mock API response
      const mockFetch = jest.fn().mockResolvedValue({
        ok: true,
        json: async () => ({ data: 'test' })
      });
      require('node-fetch').default = mockFetch;

      // First request
      await dataAggregator.aggregateData(['BTC'], { includeFunding: true });
      const firstCallCount = mockFetch.mock.calls.length;

      // Second request (should use cache)
      await dataAggregator.aggregateData(['BTC'], { includeFunding: true });
      const secondCallCount = mockFetch.mock.calls.length;

      // Should not make additional API calls due to caching
      expect(secondCallCount).toBe(firstCallCount);
    });
  });

  describe('Performance Tests', () => {
    beforeEach(async () => {
      await dataAggregator.initialize();
    });

    afterEach(async () => {
      await dataAggregator.shutdown();
    });

    it('should handle concurrent requests efficiently', async () => {
      const startTime = Date.now();
      
      // Make 10 concurrent requests
      const promises = Array(10).fill(null).map((_, index) =>
        dataAggregator.aggregateData([`TEST${index}`], { includeFunding: true })
      );

      const results = await Promise.allSettled(promises);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within reasonable time
      expect(duration).toBeLessThan(30000); // 30 seconds
      
      // Most requests should succeed (even if some APIs fail)
      const successful = results.filter(r => r.status === 'fulfilled').length;
      expect(successful).toBeGreaterThan(5);
    });

    it('should maintain performance under load', async () => {
      const times: number[] = [];
      
      // Make 5 sequential requests and measure time
      for (let i = 0; i < 5; i++) {
        const start = Date.now();
        await dataAggregator.aggregateData(['BTC'], { includeFunding: true });
        const end = Date.now();
        times.push(end - start);
      }
      
      // Times should not increase significantly (caching should help)
      const avgTime = times.reduce((sum, time) => sum + time, 0) / times.length;
      expect(avgTime).toBeLessThan(5000); // 5 seconds average
    });
  });

  describe('Health Monitoring', () => {
    beforeEach(async () => {
      await dataAggregator.initialize();
    });

    afterEach(async () => {
      await dataAggregator.shutdown();
    });

    it('should provide health status', async () => {
      const health = await dataAggregator.getHealthStatus();
      
      expect(health.status).toBeDefined();
      expect(['healthy', 'degraded', 'unhealthy']).toContain(health.status);
      expect(Array.isArray(health.services)).toBe(true);
      expect(health.cache).toBeDefined();
      expect(Array.isArray(health.rateLimits)).toBe(true);
    });

    it('should report service statuses', async () => {
      const health = await dataAggregator.getHealthStatus();
      
      health.services.forEach(service => {
        expect(service.name).toBeDefined();
        expect(['up', 'down']).toContain(service.status);
      });
    });

    it('should report cache statistics', async () => {
      const health = await dataAggregator.getHealthStatus();
      
      expect(health.cache.hitRate).toBeGreaterThanOrEqual(0);
      expect(health.cache.hitRate).toBeLessThanOrEqual(100);
      expect(health.cache.size).toBeGreaterThanOrEqual(0);
    });
  });
});

describe('End-to-End Integration', () => {
  let integration: DataAggregatorIntegration;

  beforeEach(() => {
    // Set up test environment
    Object.entries(mockEnv).forEach(([key, value]) => {
      process.env[key] = value;
    });

    integration = new DataAggregatorIntegration();
  });

  afterEach(async () => {
    if (integration) {
      await integration.shutdown();
    }

    // Clean up environment
    Object.keys(mockEnv).forEach(key => {
      delete process.env[key];
    });
  });

  it('should complete full trading signal enrichment workflow', async () => {
    await integration.initialize();

    const mockSignal = {
      symbol: 'BTC',
      confidence: 75,
      action: 'BUY',
      indicators: { rsi: 30, macd: 0.5 },
      changePercent: 5.2,
      volume: 1000000,
      currentPrice: 45000
    };

    // Test signal enrichment
    const enrichment = await integration.enrichTradingSignal(mockSignal);
    expect(enrichment).toBeDefined();

    // Test market context
    const context = await integration.getMarketContext(['BTC']);
    expect(context).toBeDefined();

    // Test adaptive threshold input preparation
    const adaptiveInput = await integration.prepareAdaptiveThresholdInput(mockSignal);
    expect(adaptiveInput.signal).toBeDefined();
    expect(adaptiveInput.enrichment).toBeDefined();
    expect(adaptiveInput.market_context).toBeDefined();

    // Verify enrichment quality
    expect(adaptiveInput.enrichment.confidence_boost).toBeGreaterThanOrEqual(-15);
    expect(adaptiveInput.enrichment.confidence_boost).toBeLessThanOrEqual(15);
    expect(adaptiveInput.enrichment.reliability).toBeGreaterThanOrEqual(0);
  });

  it('should handle backtest data enrichment workflow', async () => {
    await integration.initialize();

    const startDate = new Date('2023-01-01');
    const endDate = new Date('2023-01-02');
    const symbols = ['BTC', 'ETH'];

    const backtestEnrichment = await integration.enrichBacktestData(symbols, startDate, endDate);

    expect(backtestEnrichment).toBeDefined();
    expect(Array.isArray(backtestEnrichment.whale_activity)).toBe(true);
    expect(Array.isArray(backtestEnrichment.funding_rates)).toBe(true);
    expect(Array.isArray(backtestEnrichment.smart_money)).toBe(true);
    expect(Array.isArray(backtestEnrichment.liquidations)).toBe(true);
  });

  it('should maintain service health throughout operation', async () => {
    await integration.initialize();

    // Perform various operations
    await integration.enrichTradingSignal({ symbol: 'BTC', confidence: 75, action: 'BUY' });
    await integration.getMarketContext(['BTC', 'ETH']);
    
    // Check health after operations
    const health = await integration.getHealthStatus();
    
    expect(health.aggregator.status).toBeDefined();
    expect(health.errorHandler.status).toBeDefined();
    expect(health.integration.initialized).toBe(true);
  });
});

// Mock data for testing
const mockEtherscanResponse = {
  status: '1',
  message: 'OK',
  result: {
    address: '0x742d35Cc7bF9D4C3D9E4d0b8aEf3b6b3f6C3A1b2',
    balance: '1000000000000000000',
    transactions: []
  }
};

const mockBinanceFundingResponse = {
  symbol: 'BTCUSDT',
  markPrice: '45000.00',
  indexPrice: '45001.00',
  estimatedSettlePrice: '45000.50',
  lastFundingRate: '0.00010000',
  nextFundingTime: Date.now() + 8 * 60 * 60 * 1000,
  time: Date.now()
};

const mockCoinglassResponse = {
  code: 0,
  msg: 'success',
  data: {
    BTC: {
      dataMap: {
        binance: {
          rate: '0.0001',
          nextFundingTime: Date.now() + 8 * 60 * 60 * 1000,
          markPrice: '45000'
        }
      }
    }
  }
};