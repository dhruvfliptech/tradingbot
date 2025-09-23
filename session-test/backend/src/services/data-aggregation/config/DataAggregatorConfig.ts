import logger from '../../../utils/logger';

export interface DataAggregatorEnvironmentConfig {
  // API Keys
  ETHERSCAN_API_KEY: string;
  BITQUERY_API_KEY?: string;
  COVALENT_API_KEY: string;
  CRYPTOQUANT_API_KEY?: string;

  // API Endpoints
  ETHERSCAN_ENDPOINT?: string;
  BITQUERY_ENDPOINT?: string;
  COVALENT_ENDPOINT?: string;
  COINGLASS_ENDPOINT?: string;
  BINANCE_ENDPOINT?: string;
  CRYPTOQUANT_ENDPOINT?: string;

  // Rate Limits (requests per second)
  ETHERSCAN_RATE_LIMIT?: string;
  BITQUERY_RATE_LIMIT?: string;
  COVALENT_RATE_LIMIT?: string;
  COINGLASS_RATE_LIMIT?: string;
  BINANCE_RATE_LIMIT?: string;
  CRYPTOQUANT_RATE_LIMIT?: string;

  // Cache Settings
  CACHE_TTL_ONCHAIN?: string;
  CACHE_TTL_FUNDING?: string;
  CACHE_TTL_WHALE?: string;
  CACHE_TTL_MARKET?: string;
  CACHE_MAX_SIZE?: string;

  // Fallback Settings
  MAX_RETRIES?: string;
  RETRY_DELAY?: string;
  REQUEST_TIMEOUT?: string;

  // Feature Flags
  ENABLE_ETHERSCAN?: string;
  ENABLE_BITQUERY?: string;
  ENABLE_COVALENT?: string;
  ENABLE_COINGLASS?: string;
  ENABLE_BINANCE?: string;
  ENABLE_CRYPTOQUANT?: string;

  // Monitoring
  ENABLE_METRICS?: string;
  METRICS_INTERVAL?: string;
  LOG_LEVEL?: string;
}

export class DataAggregatorConfigManager {
  private config: any;
  private envConfig: DataAggregatorEnvironmentConfig;

  constructor() {
    this.envConfig = this.loadEnvironmentConfig();
    this.config = this.buildConfiguration();
    this.validateConfiguration();
  }

  private loadEnvironmentConfig(): DataAggregatorEnvironmentConfig {
    return {
      // API Keys
      ETHERSCAN_API_KEY: process.env.ETHERSCAN_API_KEY || '',
      BITQUERY_API_KEY: process.env.BITQUERY_API_KEY,
      COVALENT_API_KEY: process.env.COVALENT_API_KEY || '',
      CRYPTOQUANT_API_KEY: process.env.CRYPTOQUANT_API_KEY,

      // API Endpoints
      ETHERSCAN_ENDPOINT: process.env.ETHERSCAN_ENDPOINT,
      BITQUERY_ENDPOINT: process.env.BITQUERY_ENDPOINT,
      COVALENT_ENDPOINT: process.env.COVALENT_ENDPOINT,
      COINGLASS_ENDPOINT: process.env.COINGLASS_ENDPOINT,
      BINANCE_ENDPOINT: process.env.BINANCE_ENDPOINT,
      CRYPTOQUANT_ENDPOINT: process.env.CRYPTOQUANT_ENDPOINT,

      // Rate Limits
      ETHERSCAN_RATE_LIMIT: process.env.ETHERSCAN_RATE_LIMIT,
      BITQUERY_RATE_LIMIT: process.env.BITQUERY_RATE_LIMIT,
      COVALENT_RATE_LIMIT: process.env.COVALENT_RATE_LIMIT,
      COINGLASS_RATE_LIMIT: process.env.COINGLASS_RATE_LIMIT,
      BINANCE_RATE_LIMIT: process.env.BINANCE_RATE_LIMIT,
      CRYPTOQUANT_RATE_LIMIT: process.env.CRYPTOQUANT_RATE_LIMIT,

      // Cache Settings
      CACHE_TTL_ONCHAIN: process.env.CACHE_TTL_ONCHAIN,
      CACHE_TTL_FUNDING: process.env.CACHE_TTL_FUNDING,
      CACHE_TTL_WHALE: process.env.CACHE_TTL_WHALE,
      CACHE_TTL_MARKET: process.env.CACHE_TTL_MARKET,
      CACHE_MAX_SIZE: process.env.CACHE_MAX_SIZE,

      // Fallback Settings
      MAX_RETRIES: process.env.MAX_RETRIES,
      RETRY_DELAY: process.env.RETRY_DELAY,
      REQUEST_TIMEOUT: process.env.REQUEST_TIMEOUT,

      // Feature Flags
      ENABLE_ETHERSCAN: process.env.ENABLE_ETHERSCAN,
      ENABLE_BITQUERY: process.env.ENABLE_BITQUERY,
      ENABLE_COVALENT: process.env.ENABLE_COVALENT,
      ENABLE_COINGLASS: process.env.ENABLE_COINGLASS,
      ENABLE_BINANCE: process.env.ENABLE_BINANCE,
      ENABLE_CRYPTOQUANT: process.env.ENABLE_CRYPTOQUANT,

      // Monitoring
      ENABLE_METRICS: process.env.ENABLE_METRICS,
      METRICS_INTERVAL: process.env.METRICS_INTERVAL,
      LOG_LEVEL: process.env.LOG_LEVEL
    };
  }

  private buildConfiguration(): any {
    return {
      apis: {
        etherscan: {
          apiKey: this.envConfig.ETHERSCAN_API_KEY,
          endpoint: this.envConfig.ETHERSCAN_ENDPOINT || 'https://api.etherscan.io/api',
          rateLimit: parseInt(this.envConfig.ETHERSCAN_RATE_LIMIT || '5'), // 5 calls/sec for free tier
          enabled: this.parseBool(this.envConfig.ENABLE_ETHERSCAN, true)
        },
        bitquery: {
          apiKey: this.envConfig.BITQUERY_API_KEY,
          endpoint: this.envConfig.BITQUERY_ENDPOINT || 'https://graphql.bitquery.io',
          rateLimit: parseInt(this.envConfig.BITQUERY_RATE_LIMIT || '2'), // Conservative for free tier
          enabled: this.parseBool(this.envConfig.ENABLE_BITQUERY, true)
        },
        covalent: {
          apiKey: this.envConfig.COVALENT_API_KEY,
          endpoint: this.envConfig.COVALENT_ENDPOINT || 'https://api.covalenthq.com/v1',
          rateLimit: parseInt(this.envConfig.COVALENT_RATE_LIMIT || '10'), // 100k credits/month
          enabled: this.parseBool(this.envConfig.ENABLE_COVALENT, true)
        },
        coinglass: {
          endpoint: this.envConfig.COINGLASS_ENDPOINT || 'https://fapi.coinglass.com',
          rateLimit: parseInt(this.envConfig.COINGLASS_RATE_LIMIT || '5'), // Conservative for free tier
          enabled: this.parseBool(this.envConfig.ENABLE_COINGLASS, true)
        },
        binance: {
          endpoint: this.envConfig.BINANCE_ENDPOINT || 'https://api.binance.com',
          rateLimit: parseInt(this.envConfig.BINANCE_RATE_LIMIT || '20'), // 1200/min = 20/sec
          enabled: this.parseBool(this.envConfig.ENABLE_BINANCE, true)
        },
        cryptoquant: {
          apiKey: this.envConfig.CRYPTOQUANT_API_KEY,
          endpoint: this.envConfig.CRYPTOQUANT_ENDPOINT || 'https://api.cryptoquant.com',
          rateLimit: parseInt(this.envConfig.CRYPTOQUANT_RATE_LIMIT || '1'), // Limited free tier
          enabled: this.parseBool(this.envConfig.ENABLE_CRYPTOQUANT, !!this.envConfig.CRYPTOQUANT_API_KEY)
        }
      },
      cache: {
        ttl: {
          onchain: parseInt(this.envConfig.CACHE_TTL_ONCHAIN || '300'), // 5 minutes
          funding: parseInt(this.envConfig.CACHE_TTL_FUNDING || '60'),  // 1 minute
          whale: parseInt(this.envConfig.CACHE_TTL_WHALE || '30'),      // 30 seconds
          market: parseInt(this.envConfig.CACHE_TTL_MARKET || '60')     // 1 minute
        },
        maxSize: parseInt(this.envConfig.CACHE_MAX_SIZE || '10000')
      },
      fallback: {
        maxRetries: parseInt(this.envConfig.MAX_RETRIES || '3'),
        retryDelay: parseInt(this.envConfig.RETRY_DELAY || '1000'), // 1 second
        timeout: parseInt(this.envConfig.REQUEST_TIMEOUT || '10000') // 10 seconds
      },
      monitoring: {
        enabled: this.parseBool(this.envConfig.ENABLE_METRICS, true),
        interval: parseInt(this.envConfig.METRICS_INTERVAL || '60000'), // 1 minute
        logLevel: this.envConfig.LOG_LEVEL || 'info'
      }
    };
  }

  private validateConfiguration(): void {
    const errors: string[] = [];

    // Check required API keys
    if (!this.config.apis.etherscan.apiKey && this.config.apis.etherscan.enabled) {
      errors.push('ETHERSCAN_API_KEY is required when Etherscan is enabled');
    }

    if (!this.config.apis.covalent.apiKey && this.config.apis.covalent.enabled) {
      errors.push('COVALENT_API_KEY is required when Covalent is enabled');
    }

    // Validate rate limits
    Object.entries(this.config.apis).forEach(([name, apiConfig]: [string, any]) => {
      if (apiConfig.enabled && (apiConfig.rateLimit <= 0 || apiConfig.rateLimit > 100)) {
        errors.push(`Invalid rate limit for ${name}: ${apiConfig.rateLimit}`);
      }
    });

    // Validate cache settings
    if (this.config.cache.maxSize <= 0) {
      errors.push(`Invalid cache max size: ${this.config.cache.maxSize}`);
    }

    Object.entries(this.config.cache.ttl).forEach(([type, ttl]: [string, any]) => {
      if (ttl <= 0 || ttl > 3600) {
        errors.push(`Invalid cache TTL for ${type}: ${ttl} (should be 1-3600 seconds)`);
      }
    });

    // Validate fallback settings
    if (this.config.fallback.maxRetries < 0 || this.config.fallback.maxRetries > 10) {
      errors.push(`Invalid max retries: ${this.config.fallback.maxRetries}`);
    }

    if (this.config.fallback.retryDelay < 100 || this.config.fallback.retryDelay > 30000) {
      errors.push(`Invalid retry delay: ${this.config.fallback.retryDelay}`);
    }

    if (this.config.fallback.timeout < 1000 || this.config.fallback.timeout > 60000) {
      errors.push(`Invalid timeout: ${this.config.fallback.timeout}`);
    }

    // Check that at least one API is enabled
    const enabledApis = Object.values(this.config.apis).filter((api: any) => api.enabled);
    if (enabledApis.length === 0) {
      errors.push('At least one API must be enabled');
    }

    if (errors.length > 0) {
      logger.error('Configuration validation failed:', errors);
      throw new Error(`Configuration validation failed: ${errors.join(', ')}`);
    }

    logger.info('Configuration validated successfully', {
      enabledApis: Object.entries(this.config.apis)
        .filter(([_, config]: [string, any]) => config.enabled)
        .map(([name, _]) => name),
      cacheSettings: this.config.cache,
      fallbackSettings: this.config.fallback
    });
  }

  private parseBool(value: string | undefined, defaultValue: boolean): boolean {
    if (value === undefined) return defaultValue;
    return value.toLowerCase() === 'true' || value === '1';
  }

  getConfiguration(): any {
    return this.config;
  }

  getApiConfig(apiName: string): any {
    return this.config.apis[apiName];
  }

  getCacheConfig(): any {
    return this.config.cache;
  }

  getFallbackConfig(): any {
    return this.config.fallback;
  }

  getMonitoringConfig(): any {
    return this.config.monitoring;
  }

  isApiEnabled(apiName: string): boolean {
    return this.config.apis[apiName]?.enabled || false;
  }

  // Dynamic configuration updates
  updateApiConfig(apiName: string, updates: Partial<any>): void {
    if (!this.config.apis[apiName]) {
      throw new Error(`Unknown API: ${apiName}`);
    }

    this.config.apis[apiName] = {
      ...this.config.apis[apiName],
      ...updates
    };

    logger.info(`Updated configuration for ${apiName}`, updates);
  }

  updateCacheConfig(updates: Partial<any>): void {
    this.config.cache = {
      ...this.config.cache,
      ...updates
    };

    logger.info('Updated cache configuration', updates);
  }

  // Configuration export/import for backup
  exportConfiguration(): string {
    const exportConfig = {
      ...this.config,
      // Remove sensitive data
      apis: Object.fromEntries(
        Object.entries(this.config.apis).map(([name, config]: [string, any]) => [
          name,
          {
            ...config,
            apiKey: config.apiKey ? '***REDACTED***' : undefined
          }
        ])
      )
    };

    return JSON.stringify(exportConfig, null, 2);
  }

  // Health check for configuration
  async healthCheck(): Promise<{
    status: 'healthy' | 'warning' | 'error';
    issues: string[];
    recommendations: string[];
  }> {
    const issues: string[] = [];
    const recommendations: string[] = [];

    // Check API key availability
    const missingKeys = Object.entries(this.config.apis)
      .filter(([_, config]: [string, any]) => config.enabled && !config.apiKey)
      .map(([name, _]) => name);

    if (missingKeys.length > 0) {
      issues.push(`Missing API keys for: ${missingKeys.join(', ')}`);
    }

    // Check rate limits for optimization
    Object.entries(this.config.apis).forEach(([name, config]: [string, any]) => {
      if (config.enabled && config.rateLimit < 5) {
        recommendations.push(`Consider increasing rate limit for ${name} (current: ${config.rateLimit})`);
      }
    });

    // Check cache settings
    if (this.config.cache.maxSize > 50000) {
      recommendations.push('Large cache size may impact memory usage');
    }

    Object.entries(this.config.cache.ttl).forEach(([type, ttl]: [string, any]) => {
      if (ttl > 600) {
        recommendations.push(`Long cache TTL for ${type} (${ttl}s) may reduce data freshness`);
      }
    });

    let status: 'healthy' | 'warning' | 'error' = 'healthy';
    if (issues.length > 0) status = 'error';
    else if (recommendations.length > 0) status = 'warning';

    return { status, issues, recommendations };
  }

  // Environment-specific configurations
  getEnvironmentType(): 'development' | 'staging' | 'production' {
    const env = process.env.NODE_ENV?.toLowerCase() || 'development';
    
    if (env.includes('prod')) return 'production';
    if (env.includes('stag')) return 'staging';
    return 'development';
  }

  getEnvironmentSpecificConfig(): any {
    const environment = this.getEnvironmentType();
    
    const environmentConfigs = {
      development: {
        cache: {
          ttl: {
            onchain: 60,    // Shorter cache for faster development
            funding: 30,
            whale: 15,
            market: 30
          }
        },
        fallback: {
          maxRetries: 1,    // Faster failures in development
          retryDelay: 500
        }
      },
      staging: {
        cache: {
          ttl: {
            onchain: 180,   // Medium cache for testing
            funding: 45,
            whale: 20,
            market: 45
          }
        },
        fallback: {
          maxRetries: 2,
          retryDelay: 750
        }
      },
      production: {
        cache: {
          ttl: {
            onchain: 300,   // Longer cache for production
            funding: 60,
            whale: 30,
            market: 60
          }
        },
        fallback: {
          maxRetries: 3,
          retryDelay: 1000
        }
      }
    };

    return environmentConfigs[environment];
  }

  // Apply environment-specific overrides
  applyEnvironmentOverrides(): void {
    const envConfig = this.getEnvironmentSpecificConfig();
    
    // Deep merge environment config
    this.config = this.deepMerge(this.config, envConfig);
    
    logger.info(`Applied ${this.getEnvironmentType()} environment overrides`);
  }

  private deepMerge(target: any, source: any): any {
    const result = { ...target };
    
    for (const key in source) {
      if (source[key] && typeof source[key] === 'object' && !Array.isArray(source[key])) {
        result[key] = this.deepMerge(result[key] || {}, source[key]);
      } else {
        result[key] = source[key];
      }
    }
    
    return result;
  }

  // Configuration monitoring
  getConfigurationMetrics(): {
    enabledApis: number;
    totalRateLimit: number;
    cacheSize: number;
    averageTtl: number;
  } {
    const enabledApis = Object.values(this.config.apis).filter((api: any) => api.enabled).length;
    const totalRateLimit = Object.values(this.config.apis)
      .filter((api: any) => api.enabled)
      .reduce((sum: number, api: any) => sum + api.rateLimit, 0);
    
    const ttlValues = Object.values(this.config.cache.ttl) as number[];
    const averageTtl = ttlValues.reduce((sum, ttl) => sum + ttl, 0) / ttlValues.length;

    return {
      enabledApis,
      totalRateLimit,
      cacheSize: this.config.cache.maxSize,
      averageTtl: Math.round(averageTtl)
    };
  }
}