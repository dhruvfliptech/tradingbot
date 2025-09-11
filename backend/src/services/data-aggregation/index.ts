// Main service exports
export { DataAggregatorService } from './DataAggregatorService';
export { DataAggregatorIntegration } from './integration/DataAggregatorIntegration';

// Configuration exports
export { DataAggregatorConfigManager } from './config/DataAggregatorConfig';

// Core service exports
export { CacheService } from './CacheService';
export { RateLimiter } from './RateLimiter';
export { DataNormalizer } from './DataNormalizer';
export { DataAggregatorErrorHandler } from './ErrorHandler';

// API client exports
export { EtherscanClient } from './clients/EtherscanClient';
export { BitqueryClient } from './clients/BitqueryClient';
export { CovalentClient } from './clients/CovalentClient';
export { CoinglassClient } from './clients/CoinglassClient';
export { BinanceClient } from './clients/BinanceClient';
export { CryptoQuantClient } from './clients/CryptoQuantClient';

// Type exports
export type {
  OnChainData,
  Transaction,
  TokenHolding,
  FundingRateData,
  WhaleAlert,
  SmartMoneyFlow,
  LiquidationData,
  AggregatedData
} from './DataAggregatorService';

export type {
  TradingSignalEnrichment,
  MarketContext,
  AdaptiveThresholdInput,
  ComposerBacktestEnrichment
} from './integration/DataAggregatorIntegration';

// Configuration types
export type { DataAggregatorEnvironmentConfig } from './config/DataAggregatorConfig';

// Factory function for easy initialization
export async function createDataAggregatorIntegration(): Promise<DataAggregatorIntegration> {
  const integration = new DataAggregatorIntegration();
  await integration.initialize();
  return integration;
}

// Utility function to create a basic configuration
export function createBasicConfig(apiKeys: {
  etherscan: string;
  covalent: string;
  bitquery?: string;
  cryptoquant?: string;
}): any {
  return {
    apis: {
      etherscan: {
        apiKey: apiKeys.etherscan,
        endpoint: 'https://api.etherscan.io/api',
        rateLimit: 5,
        enabled: true
      },
      bitquery: {
        apiKey: apiKeys.bitquery,
        endpoint: 'https://graphql.bitquery.io',
        rateLimit: 2,
        enabled: !!apiKeys.bitquery
      },
      covalent: {
        apiKey: apiKeys.covalent,
        endpoint: 'https://api.covalenthq.com/v1',
        rateLimit: 10,
        enabled: true
      },
      coinglass: {
        endpoint: 'https://fapi.coinglass.com',
        rateLimit: 5,
        enabled: true
      },
      binance: {
        endpoint: 'https://api.binance.com',
        rateLimit: 20,
        enabled: true
      },
      cryptoquant: {
        apiKey: apiKeys.cryptoquant,
        endpoint: 'https://api.cryptoquant.com',
        rateLimit: 1,
        enabled: !!apiKeys.cryptoquant
      }
    },
    cache: {
      ttl: {
        onchain: 300,    // 5 minutes
        funding: 60,     // 1 minute
        whale: 30,       // 30 seconds
        market: 60       // 1 minute
      },
      maxSize: 10000
    },
    fallback: {
      maxRetries: 3,
      retryDelay: 1000,
      timeout: 10000
    }
  };
}