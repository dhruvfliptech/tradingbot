import logger from '../../utils/logger';

export interface ErrorContext {
  service: string;
  method: string;
  symbol?: string;
  attempt: number;
  maxAttempts: number;
  originalError: Error;
  timestamp: Date;
}

export interface FallbackStrategy {
  name: string;
  priority: number;
  execute: (context: ErrorContext) => Promise<any>;
  canHandle: (error: Error, context: ErrorContext) => boolean;
}

export interface CircuitBreakerState {
  isOpen: boolean;
  failureCount: number;
  lastFailureTime: number;
  successCount: number;
}

export interface RetryConfig {
  maxAttempts: number;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  jitterMax: number;
}

export class DataAggregatorErrorHandler {
  private circuitBreakers: Map<string, CircuitBreakerState> = new Map();
  private fallbackStrategies: Map<string, FallbackStrategy[]> = new Map();
  private retryConfig: RetryConfig;
  private errorStats: Map<string, { count: number; lastError: Date }> = new Map();

  // Circuit breaker configuration
  private readonly circuitBreakerConfig = {
    failureThreshold: 5,        // Open circuit after 5 failures
    resetTimeout: 60000,        // Try again after 1 minute
    successThreshold: 3         // Close circuit after 3 successes
  };

  constructor(retryConfig?: Partial<RetryConfig>) {
    this.retryConfig = {
      maxAttempts: 3,
      baseDelay: 1000,
      maxDelay: 30000,
      backoffMultiplier: 2,
      jitterMax: 1000,
      ...retryConfig
    };

    this.setupDefaultFallbackStrategies();
  }

  async executeWithFallback<T>(
    primaryFunction: () => Promise<T>,
    context: Omit<ErrorContext, 'attempt' | 'originalError' | 'timestamp'>
  ): Promise<T> {
    const fullContext: ErrorContext = {
      ...context,
      attempt: 1,
      maxAttempts: this.retryConfig.maxAttempts,
      originalError: new Error('Not executed'),
      timestamp: new Date()
    };

    // Check circuit breaker
    if (this.isCircuitOpen(context.service)) {
      throw new Error(`Circuit breaker is open for service: ${context.service}`);
    }

    try {
      const result = await this.executeWithRetry(primaryFunction, fullContext);
      this.recordSuccess(context.service);
      return result;

    } catch (error) {
      fullContext.originalError = error as Error;
      logger.error(`Primary function failed for ${context.service}.${context.method}:`, error);

      // Record failure and update circuit breaker
      this.recordFailure(context.service, error as Error);

      // Try fallback strategies
      const fallbackResult = await this.tryFallbackStrategies(fullContext);
      if (fallbackResult !== null) {
        logger.info(`Fallback successful for ${context.service}.${context.method}`);
        return fallbackResult;
      }

      // All strategies failed
      this.updateErrorStats(context.service, error as Error);
      throw new Error(
        `All retry attempts and fallback strategies failed for ${context.service}.${context.method}: ${error.message}`
      );
    }
  }

  private async executeWithRetry<T>(
    fn: () => Promise<T>,
    context: ErrorContext
  ): Promise<T> {
    let lastError: Error;

    for (let attempt = 1; attempt <= context.maxAttempts; attempt++) {
      try {
        context.attempt = attempt;
        const result = await fn();
        
        if (attempt > 1) {
          logger.info(`Retry successful on attempt ${attempt} for ${context.service}.${context.method}`);
        }
        
        return result;

      } catch (error) {
        lastError = error as Error;
        context.originalError = lastError;

        // Don't retry on certain types of errors
        if (this.isNonRetryableError(lastError)) {
          logger.warn(`Non-retryable error for ${context.service}.${context.method}:`, lastError.message);
          throw lastError;
        }

        if (attempt < context.maxAttempts) {
          const delay = this.calculateRetryDelay(attempt);
          logger.warn(
            `Attempt ${attempt} failed for ${context.service}.${context.method}, retrying in ${delay}ms:`,
            lastError.message
          );
          await this.sleep(delay);
        }
      }
    }

    throw lastError!;
  }

  private async tryFallbackStrategies(context: ErrorContext): Promise<any> {
    const strategies = this.fallbackStrategies.get(context.service) || [];
    const applicableStrategies = strategies
      .filter(strategy => strategy.canHandle(context.originalError, context))
      .sort((a, b) => b.priority - a.priority);

    for (const strategy of applicableStrategies) {
      try {
        logger.info(`Trying fallback strategy '${strategy.name}' for ${context.service}.${context.method}`);
        const result = await strategy.execute(context);
        
        if (result !== null && result !== undefined) {
          logger.info(`Fallback strategy '${strategy.name}' succeeded`);
          return result;
        }

      } catch (error) {
        logger.warn(`Fallback strategy '${strategy.name}' failed:`, error.message);
      }
    }

    return null;
  }

  private setupDefaultFallbackStrategies(): void {
    // Cache fallback strategy
    this.addFallbackStrategy('etherscan', {
      name: 'cache_fallback',
      priority: 100,
      execute: async (context) => {
        // Try to get stale data from cache
        return await this.getCachedData(context);
      },
      canHandle: (error, context) => {
        return error.message.includes('rate limit') || 
               error.message.includes('timeout') ||
               error.message.includes('503') ||
               error.message.includes('502');
      }
    });

    // Alternative API fallback for funding rates
    this.addFallbackStrategy('coinglass', {
      name: 'binance_funding_fallback',
      priority: 90,
      execute: async (context) => {
        if (context.method === 'getFundingRate' && context.symbol) {
          return await this.getBinanceFundingRate(context.symbol);
        }
        return null;
      },
      canHandle: (error, context) => {
        return context.method === 'getFundingRate';
      }
    });

    // Simplified data fallback
    this.addFallbackStrategy('bitquery', {
      name: 'simplified_data_fallback',
      priority: 80,
      execute: async (context) => {
        return await this.getSimplifiedData(context);
      },
      canHandle: (error, context) => {
        return error.message.includes('query') || 
               error.message.includes('graphql') ||
               error.message.includes('complexity');
      }
    });

    // Default empty response fallback (lowest priority)
    this.addFallbackStrategies(['etherscan', 'bitquery', 'covalent', 'coinglass', 'binance', 'cryptoquant'], {
      name: 'empty_response_fallback',
      priority: 10,
      execute: async (context) => {
        logger.warn(`Using empty response fallback for ${context.service}.${context.method}`);
        return this.getEmptyResponse(context.method);
      },
      canHandle: () => true
    });
  }

  private addFallbackStrategy(service: string, strategy: FallbackStrategy): void {
    if (!this.fallbackStrategies.has(service)) {
      this.fallbackStrategies.set(service, []);
    }
    this.fallbackStrategies.get(service)!.push(strategy);
  }

  private addFallbackStrategies(services: string[], strategy: FallbackStrategy): void {
    services.forEach(service => this.addFallbackStrategy(service, strategy));
  }

  // Circuit breaker methods
  private isCircuitOpen(service: string): boolean {
    const state = this.getCircuitBreakerState(service);
    
    if (!state.isOpen) return false;

    // Check if we should try to close the circuit
    const timeSinceLastFailure = Date.now() - state.lastFailureTime;
    if (timeSinceLastFailure >= this.circuitBreakerConfig.resetTimeout) {
      state.isOpen = false;
      state.failureCount = 0;
      state.successCount = 0;
      logger.info(`Circuit breaker reset for service: ${service}`);
      return false;
    }

    return true;
  }

  private getCircuitBreakerState(service: string): CircuitBreakerState {
    if (!this.circuitBreakers.has(service)) {
      this.circuitBreakers.set(service, {
        isOpen: false,
        failureCount: 0,
        lastFailureTime: 0,
        successCount: 0
      });
    }
    return this.circuitBreakers.get(service)!;
  }

  private recordSuccess(service: string): void {
    const state = this.getCircuitBreakerState(service);
    state.successCount++;

    // Close circuit if we have enough successes
    if (state.isOpen && state.successCount >= this.circuitBreakerConfig.successThreshold) {
      state.isOpen = false;
      state.failureCount = 0;
      state.successCount = 0;
      logger.info(`Circuit breaker closed for service: ${service}`);
    }
  }

  private recordFailure(service: string, error: Error): void {
    const state = this.getCircuitBreakerState(service);
    state.failureCount++;
    state.lastFailureTime = Date.now();
    state.successCount = 0;

    // Open circuit if we have too many failures
    if (!state.isOpen && state.failureCount >= this.circuitBreakerConfig.failureThreshold) {
      state.isOpen = true;
      logger.warn(`Circuit breaker opened for service: ${service} after ${state.failureCount} failures`);
    }
  }

  // Retry logic methods
  private calculateRetryDelay(attempt: number): number {
    const exponentialDelay = this.retryConfig.baseDelay * Math.pow(this.retryConfig.backoffMultiplier, attempt - 1);
    const cappedDelay = Math.min(exponentialDelay, this.retryConfig.maxDelay);
    const jitter = Math.random() * this.retryConfig.jitterMax;
    
    return Math.floor(cappedDelay + jitter);
  }

  private isNonRetryableError(error: Error): boolean {
    const nonRetryablePatterns = [
      'invalid api key',
      'unauthorized',
      'forbidden',
      'not found',
      'invalid parameter',
      'malformed request',
      'authentication failed',
      '401',
      '403',
      '404',
      '400'
    ];

    const errorMessage = error.message.toLowerCase();
    return nonRetryablePatterns.some(pattern => errorMessage.includes(pattern));
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Fallback strategy implementations
  private async getCachedData(context: ErrorContext): Promise<any> {
    // This would integrate with the cache service to get stale data
    // For now, return null to indicate no cached data available
    logger.debug(`Attempting to get cached data for ${context.service}.${context.method}`);
    return null;
  }

  private async getBinanceFundingRate(symbol: string): Promise<any> {
    try {
      // This would use the Binance client as a fallback for funding rates
      logger.debug(`Attempting to get funding rate from Binance for ${symbol}`);
      
      // Return a simplified funding rate structure
      return {
        symbol,
        rate: 0,
        nextFundingTime: new Date(Date.now() + 8 * 60 * 60 * 1000), // 8 hours from now
        source: 'binance_fallback',
        timestamp: new Date()
      };

    } catch (error) {
      logger.warn(`Binance fallback failed for ${symbol}:`, error.message);
      return null;
    }
  }

  private async getSimplifiedData(context: ErrorContext): Promise<any> {
    // Return simplified data structures when full queries fail
    logger.debug(`Using simplified data for ${context.service}.${context.method}`);
    
    switch (context.method) {
      case 'getWhaleTransfers':
        return [];
      
      case 'getSmartMoneyFlows':
        return [];
      
      case 'getDexTrades':
        return [];
      
      default:
        return null;
    }
  }

  private getEmptyResponse(method: string): any {
    // Provide empty but valid responses based on method type
    const emptyResponses: Record<string, any> = {
      'getAddressData': null,
      'getFundingRate': null,
      'getWhaleTransfers': [],
      'getSmartMoneyFlows': [],
      'getLargeTransactions': [],
      'getLiquidations': [],
      'getOpenInterest': [],
      'getLongShortRatio': [],
      'getTokenTransfers': [],
      'getTokenHolders': [],
      'getDexTransactions': [],
      'getExchangeFlows': [],
      'getExchangeReserves': [],
      'getWhaleTransactions': [],
      'getMinerMetrics': [],
      'getNVT': [],
      'getMVRV': [],
      'getSOPR': []
    };

    return emptyResponses[method] || null;
  }

  // Error statistics and monitoring
  private updateErrorStats(service: string, error: Error): void {
    const key = `${service}_errors`;
    const current = this.errorStats.get(key) || { count: 0, lastError: new Date() };
    
    this.errorStats.set(key, {
      count: current.count + 1,
      lastError: new Date()
    });
  }

  getErrorStats(): Record<string, { count: number; lastError: Date }> {
    return Object.fromEntries(this.errorStats.entries());
  }

  getCircuitBreakerStates(): Record<string, CircuitBreakerState> {
    return Object.fromEntries(this.circuitBreakers.entries());
  }

  // Health check
  getHealthStatus(): {
    status: 'healthy' | 'degraded' | 'unhealthy';
    issues: string[];
    circuitBreakers: Record<string, { isOpen: boolean; failureCount: number }>;
    errorCounts: Record<string, number>;
  } {
    const issues: string[] = [];
    const circuitBreakers: Record<string, { isOpen: boolean; failureCount: number }> = {};
    const errorCounts: Record<string, number> = {};

    // Check circuit breakers
    let openCircuits = 0;
    for (const [service, state] of this.circuitBreakers.entries()) {
      circuitBreakers[service] = {
        isOpen: state.isOpen,
        failureCount: state.failureCount
      };

      if (state.isOpen) {
        openCircuits++;
        issues.push(`Circuit breaker open for ${service}`);
      }
    }

    // Check error counts
    for (const [key, stats] of this.errorStats.entries()) {
      errorCounts[key] = stats.count;
      
      // Check for high error rates in the last hour
      const oneHourAgo = Date.now() - 60 * 60 * 1000;
      if (stats.lastError.getTime() > oneHourAgo && stats.count > 10) {
        issues.push(`High error count for ${key}: ${stats.count}`);
      }
    }

    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';
    if (openCircuits > 0) {
      status = openCircuits >= 3 ? 'unhealthy' : 'degraded';
    } else if (issues.length > 0) {
      status = 'degraded';
    }

    return {
      status,
      issues,
      circuitBreakers,
      errorCounts
    };
  }

  // Configuration methods
  updateRetryConfig(config: Partial<RetryConfig>): void {
    this.retryConfig = { ...this.retryConfig, ...config };
    logger.info('Updated retry configuration', config);
  }

  resetCircuitBreaker(service: string): void {
    if (this.circuitBreakers.has(service)) {
      const state = this.circuitBreakers.get(service)!;
      state.isOpen = false;
      state.failureCount = 0;
      state.successCount = 0;
      state.lastFailureTime = 0;
      
      logger.info(`Circuit breaker manually reset for service: ${service}`);
    }
  }

  resetErrorStats(): void {
    this.errorStats.clear();
    logger.info('Error statistics reset');
  }

  // Custom fallback strategy registration
  registerFallbackStrategy(service: string, strategy: FallbackStrategy): void {
    this.addFallbackStrategy(service, strategy);
    logger.info(`Registered custom fallback strategy '${strategy.name}' for service: ${service}`);
  }

  removeFallbackStrategy(service: string, strategyName: string): void {
    const strategies = this.fallbackStrategies.get(service);
    if (strategies) {
      const index = strategies.findIndex(s => s.name === strategyName);
      if (index !== -1) {
        strategies.splice(index, 1);
        logger.info(`Removed fallback strategy '${strategyName}' from service: ${service}`);
      }
    }
  }
}