import logger from '../../utils/logger';

export interface RateLimitConfig {
  requests: number;
  windowMs: number;
  burst?: number; // Allow burst requests
}

export interface RateLimitStatus {
  service: string;
  remaining: number;
  resetTime: Date;
  isBlocked: boolean;
}

interface TokenBucket {
  tokens: number;
  maxTokens: number;
  refillRate: number; // tokens per second
  lastRefill: number;
  burst: number;
}

export class RateLimiter {
  private buckets: Map<string, TokenBucket> = new Map();
  private blocked: Map<string, number> = new Map(); // service -> unblock timestamp
  private stats: Map<string, { requests: number; blocked: number }> = new Map();

  constructor() {
    // Start token refill interval
    setInterval(() => {
      this.refillTokens();
    }, 100); // Refill every 100ms for smooth rate limiting
  }

  addLimit(service: string, requestsPerSecond: number, windowMs: number = 1000, burst?: number): void {
    const maxTokens = Math.max(1, Math.floor(requestsPerSecond * (windowMs / 1000)));
    const burstTokens = burst || Math.min(maxTokens * 2, maxTokens + 10);

    this.buckets.set(service, {
      tokens: maxTokens,
      maxTokens,
      refillRate: requestsPerSecond,
      lastRefill: Date.now(),
      burst: burstTokens
    });

    this.stats.set(service, { requests: 0, blocked: 0 });

    logger.debug(`Rate limiter configured for ${service}: ${requestsPerSecond} req/s, burst: ${burstTokens}`);
  }

  async waitForToken(service: string): Promise<void> {
    const bucket = this.buckets.get(service);
    if (!bucket) {
      logger.warn(`No rate limit configured for service: ${service}`);
      return;
    }

    // Check if service is currently blocked
    const blockUntil = this.blocked.get(service);
    if (blockUntil && Date.now() < blockUntil) {
      const waitTime = blockUntil - Date.now();
      logger.debug(`Service ${service} is blocked, waiting ${waitTime}ms`);
      await this.sleep(waitTime);
      this.blocked.delete(service);
    }

    // Try to consume a token
    if (bucket.tokens > 0) {
      bucket.tokens--;
      this.incrementStats(service, 'requests');
      return;
    }

    // No tokens available, calculate wait time
    const tokensNeeded = 1;
    const waitTimeMs = (tokensNeeded / bucket.refillRate) * 1000;
    
    logger.debug(`Rate limit reached for ${service}, waiting ${waitTimeMs}ms`);
    this.incrementStats(service, 'blocked');
    
    await this.sleep(waitTimeMs);
    
    // Try again after waiting
    if (bucket.tokens > 0) {
      bucket.tokens--;
      this.incrementStats(service, 'requests');
    } else {
      // Still no tokens, this shouldn't happen but handle gracefully
      logger.warn(`Token still not available for ${service} after wait`);
    }
  }

  async checkLimit(service: string): Promise<boolean> {
    const bucket = this.buckets.get(service);
    if (!bucket) {
      return true; // No limit configured, allow
    }

    // Check if service is currently blocked
    const blockUntil = this.blocked.get(service);
    if (blockUntil && Date.now() < blockUntil) {
      return false;
    }

    return bucket.tokens > 0;
  }

  getRemainingTokens(service: string): number {
    const bucket = this.buckets.get(service);
    return bucket ? bucket.tokens : 0;
  }

  getResetTime(service: string): Date {
    const bucket = this.buckets.get(service);
    if (!bucket) {
      return new Date();
    }

    const tokensToFull = bucket.maxTokens - bucket.tokens;
    const timeToFill = (tokensToFull / bucket.refillRate) * 1000;
    return new Date(Date.now() + timeToFill);
  }

  blockService(service: string, durationMs: number): void {
    const unblockTime = Date.now() + durationMs;
    this.blocked.set(service, unblockTime);
    logger.warn(`Service ${service} blocked for ${durationMs}ms due to rate limit violation`);
  }

  unblockService(service: string): void {
    this.blocked.delete(service);
    logger.info(`Service ${service} unblocked`);
  }

  getStatus(): RateLimitStatus[] {
    return Array.from(this.buckets.entries()).map(([service, bucket]) => {
      const blockUntil = this.blocked.get(service);
      return {
        service,
        remaining: bucket.tokens,
        resetTime: this.getResetTime(service),
        isBlocked: blockUntil ? Date.now() < blockUntil : false
      };
    });
  }

  getStats(): Record<string, { requests: number; blocked: number; hitRate: number }> {
    const result: Record<string, { requests: number; blocked: number; hitRate: number }> = {};
    
    for (const [service, stats] of this.stats.entries()) {
      const total = stats.requests + stats.blocked;
      const hitRate = total > 0 ? (stats.requests / total) * 100 : 100;
      
      result[service] = {
        requests: stats.requests,
        blocked: stats.blocked,
        hitRate: Math.round(hitRate * 100) / 100
      };
    }
    
    return result;
  }

  reset(service?: string): void {
    if (service) {
      const bucket = this.buckets.get(service);
      if (bucket) {
        bucket.tokens = bucket.maxTokens;
      }
      this.blocked.delete(service);
      this.stats.set(service, { requests: 0, blocked: 0 });
    } else {
      // Reset all services
      for (const [service, bucket] of this.buckets.entries()) {
        bucket.tokens = bucket.maxTokens;
        this.blocked.delete(service);
        this.stats.set(service, { requests: 0, blocked: 0 });
      }
    }
  }

  private refillTokens(): void {
    const now = Date.now();
    
    for (const [service, bucket] of this.buckets.entries()) {
      const timePassed = (now - bucket.lastRefill) / 1000; // seconds
      const tokensToAdd = timePassed * bucket.refillRate;
      
      if (tokensToAdd >= 0.1) { // Only refill if at least 0.1 tokens
        bucket.tokens = Math.min(bucket.maxTokens, bucket.tokens + tokensToAdd);
        bucket.lastRefill = now;
      }
    }
  }

  private incrementStats(service: string, type: 'requests' | 'blocked'): void {
    const stats = this.stats.get(service);
    if (stats) {
      stats[type]++;
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Advanced rate limiting features

  // Burst allowance - temporarily allow more requests
  async useBurstToken(service: string): Promise<boolean> {
    const bucket = this.buckets.get(service);
    if (!bucket) {
      return false;
    }

    if (bucket.tokens > 0) {
      bucket.tokens--;
      return true;
    }

    // Check if burst tokens are available
    if (bucket.burst > bucket.maxTokens && bucket.tokens < 0) {
      const burstUsed = Math.abs(bucket.tokens);
      if (burstUsed < (bucket.burst - bucket.maxTokens)) {
        bucket.tokens--;
        return true;
      }
    }

    return false;
  }

  // Adaptive rate limiting based on API response
  adaptLimit(service: string, responseTime: number, statusCode: number): void {
    const bucket = this.buckets.get(service);
    if (!bucket) return;

    if (statusCode === 429) {
      // Rate limited by API, reduce our limit temporarily
      bucket.refillRate = Math.max(0.1, bucket.refillRate * 0.5);
      this.blockService(service, 60000); // Block for 1 minute
      logger.warn(`Adaptive rate limiting: reduced rate for ${service} due to 429 response`);
    } else if (statusCode >= 500) {
      // Server error, slow down
      bucket.refillRate = Math.max(0.1, bucket.refillRate * 0.8);
      logger.warn(`Adaptive rate limiting: reduced rate for ${service} due to server error`);
    } else if (responseTime > 5000) {
      // Slow response, reduce rate
      bucket.refillRate = Math.max(0.1, bucket.refillRate * 0.9);
      logger.debug(`Adaptive rate limiting: reduced rate for ${service} due to slow response`);
    } else if (statusCode === 200 && responseTime < 1000) {
      // Good response, can potentially increase rate (but be conservative)
      const originalRate = bucket.maxTokens; // Assume this was the original rate
      bucket.refillRate = Math.min(originalRate, bucket.refillRate * 1.01);
    }
  }

  // Priority-based rate limiting
  async waitForPriorityToken(service: string, priority: 'low' | 'normal' | 'high' = 'normal'): Promise<void> {
    const bucket = this.buckets.get(service);
    if (!bucket) {
      return this.waitForToken(service);
    }

    const priorityMultiplier = {
      low: 0.3,
      normal: 1.0,
      high: 2.0
    };

    // High priority requests can use more tokens, low priority use fewer
    const tokensNeeded = priority === 'high' ? 1 : priority === 'low' ? 0.5 : 1;

    if (bucket.tokens >= tokensNeeded) {
      bucket.tokens -= tokensNeeded;
      this.incrementStats(service, 'requests');
      return;
    }

    // Calculate wait time based on priority
    const baseWaitTime = (tokensNeeded / bucket.refillRate) * 1000;
    const waitTime = baseWaitTime / priorityMultiplier[priority];
    
    await this.sleep(waitTime);
    
    if (bucket.tokens >= tokensNeeded) {
      bucket.tokens -= tokensNeeded;
      this.incrementStats(service, 'requests');
    }
  }

  // Health check
  getHealthStatus(): {
    status: 'healthy' | 'degraded' | 'unhealthy';
    issues: string[];
  } {
    const issues: string[] = [];
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';

    const stats = this.getStats();
    let blockedServices = 0;
    let lowHitRateServices = 0;

    for (const [service, serviceStats] of Object.entries(stats)) {
      if (serviceStats.hitRate < 50) {
        lowHitRateServices++;
        issues.push(`Low hit rate for ${service}: ${serviceStats.hitRate}%`);
      }

      const serviceStatus = this.getStatus().find(s => s.service === service);
      if (serviceStatus?.isBlocked) {
        blockedServices++;
        issues.push(`Service ${service} is blocked`);
      }
    }

    if (blockedServices > 0) {
      status = blockedServices > this.buckets.size / 2 ? 'unhealthy' : 'degraded';
    } else if (lowHitRateServices > this.buckets.size / 3) {
      status = 'degraded';
    }

    return { status, issues };
  }
}