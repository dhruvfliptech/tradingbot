interface RateLimitConfig {
  maxRequests: number;
  windowMs: number;
  delayMs?: number;
}

interface RequestRecord {
  timestamps: number[];
  lastRequestTime: number;
}

class RateLimiter {
  private limits: Map<string, RateLimitConfig>;
  private requests: Map<string, RequestRecord>;
  private queues: Map<string, Array<{ resolve: () => void; reject: (error: Error) => void; timestamp: number }>>;

  constructor() {
    this.limits = new Map();
    this.requests = new Map();
    this.queues = new Map();
    this.setupDefaultLimits();
  }

  private setupDefaultLimits(): void {
    // Define rate limits for different API providers
    this.limits.set('etherscan', {
      maxRequests: 5,
      windowMs: 1000, // 5 requests per second
      delayMs: 200,
    });

    this.limits.set('bitquery', {
      maxRequests: 1,
      windowMs: 1000, // 1 request per second for GraphQL
      delayMs: 1000,
    });

    this.limits.set('whalealert', {
      maxRequests: 1,
      windowMs: 1000, // 1 request per second
      delayMs: 1000,
    });

    this.limits.set('coingecko', {
      maxRequests: 10,
      windowMs: 60000, // 10 requests per minute for free tier
      delayMs: 6000,
    });

    this.limits.set('coingecko-pro', {
      maxRequests: 500,
      windowMs: 60000, // 500 requests per minute for pro tier
      delayMs: 120,
    });

    this.limits.set('alpaca', {
      maxRequests: 200,
      windowMs: 60000, // 200 requests per minute
      delayMs: 300,
    });

    this.limits.set('binance', {
      maxRequests: 1200,
      windowMs: 60000, // 1200 weight per minute (Binance default)
      delayMs: 50,
    });

    this.limits.set('groq', {
      maxRequests: 30,
      windowMs: 60000, // 30 requests per minute
      delayMs: 2000,
    });

    this.limits.set('default', {
      maxRequests: 10,
      windowMs: 60000, // Conservative default
      delayMs: 6000,
    });
  }

  setLimit(provider: string, config: RateLimitConfig): void {
    this.limits.set(provider, config);
    console.log(`🎛️ Rate limit set for ${provider}: ${config.maxRequests}/${config.windowMs}ms`);
  }

  async waitForSlot(provider: string): Promise<void> {
    const config = this.limits.get(provider) || this.limits.get('default')!;
    const now = Date.now();
    
    // Initialize request record if not exists
    if (!this.requests.has(provider)) {
      this.requests.set(provider, {
        timestamps: [],
        lastRequestTime: 0,
      });
    }

    const record = this.requests.get(provider)!;

    // Clean old timestamps outside the window
    record.timestamps = record.timestamps.filter(
      timestamp => now - timestamp < config.windowMs
    );

    // Check if we've exceeded the rate limit
    if (record.timestamps.length >= config.maxRequests) {
      const oldestRequest = Math.min(...record.timestamps);
      const waitTime = config.windowMs - (now - oldestRequest);
      
      if (waitTime > 0) {
        console.log(`⏱️ Rate limit reached for ${provider}, waiting ${waitTime}ms`);
        await this.delay(waitTime);
        return this.waitForSlot(provider); // Recursive check after wait
      }
    }

    // Check minimum delay between requests
    if (config.delayMs && record.lastRequestTime > 0) {
      const timeSinceLastRequest = now - record.lastRequestTime;
      if (timeSinceLastRequest < config.delayMs) {
        const delayTime = config.delayMs - timeSinceLastRequest;
        console.log(`⏱️ Minimum delay for ${provider}, waiting ${delayTime}ms`);
        await this.delay(delayTime);
      }
    }

    // Record this request
    record.timestamps.push(Date.now());
    record.lastRequestTime = Date.now();
  }

  async executeWithRateLimit<T>(
    provider: string,
    fn: () => Promise<T>,
    retryCount: number = 3
  ): Promise<T> {
    for (let attempt = 1; attempt <= retryCount; attempt++) {
      try {
        await this.waitForSlot(provider);
        return await fn();
      } catch (error) {
        console.error(`🔄 Attempt ${attempt} failed for ${provider}:`, error);
        
        if (attempt === retryCount) {
          throw error;
        }

        // Exponential backoff for retries
        const backoffTime = Math.pow(2, attempt) * 1000;
        console.log(`⏳ Retrying ${provider} in ${backoffTime}ms...`);
        await this.delay(backoffTime);
      }
    }

    throw new Error(`Max retries exceeded for ${provider}`);
  }

  // Queue system for burst protection
  async queueRequest(provider: string): Promise<void> {
    return new Promise((resolve, reject) => {
      if (!this.queues.has(provider)) {
        this.queues.set(provider, []);
      }

      const queue = this.queues.get(provider)!;
      const timeout = setTimeout(() => {
        reject(new Error(`Request timeout for ${provider}`));
      }, 30000); // 30 second timeout

      queue.push({
        resolve: () => {
          clearTimeout(timeout);
          resolve();
        },
        reject: (error: Error) => {
          clearTimeout(timeout);
          reject(error);
        },
        timestamp: Date.now(),
      });

      this.processQueue(provider);
    });
  }

  private async processQueue(provider: string): Promise<void> {
    const queue = this.queues.get(provider);
    if (!queue || queue.length === 0) return;

    const config = this.limits.get(provider) || this.limits.get('default')!;
    const record = this.requests.get(provider);
    
    if (!record) return;

    const now = Date.now();
    
    // Clean old timestamps
    record.timestamps = record.timestamps.filter(
      timestamp => now - timestamp < config.windowMs
    );

    // Check if we can process the next request
    if (record.timestamps.length < config.maxRequests) {
      const nextRequest = queue.shift();
      if (nextRequest) {
        record.timestamps.push(now);
        record.lastRequestTime = now;
        nextRequest.resolve();
        
        // Schedule next processing with minimum delay
        if (queue.length > 0 && config.delayMs) {
          setTimeout(() => this.processQueue(provider), config.delayMs);
        }
      }
    } else {
      // Wait until oldest request expires
      const oldestRequest = Math.min(...record.timestamps);
      const waitTime = config.windowMs - (now - oldestRequest);
      setTimeout(() => this.processQueue(provider), waitTime);
    }
  }

  getStatus(provider: string): {
    currentRequests: number;
    maxRequests: number;
    windowMs: number;
    nextAvailableSlot: number;
  } {
    const config = this.limits.get(provider) || this.limits.get('default')!;
    const record = this.requests.get(provider);
    
    if (!record) {
      return {
        currentRequests: 0,
        maxRequests: config.maxRequests,
        windowMs: config.windowMs,
        nextAvailableSlot: 0,
      };
    }

    const now = Date.now();
    const validTimestamps = record.timestamps.filter(
      timestamp => now - timestamp < config.windowMs
    );

    let nextAvailableSlot = 0;
    if (validTimestamps.length >= config.maxRequests) {
      const oldestRequest = Math.min(...validTimestamps);
      nextAvailableSlot = config.windowMs - (now - oldestRequest);
    }

    return {
      currentRequests: validTimestamps.length,
      maxRequests: config.maxRequests,
      windowMs: config.windowMs,
      nextAvailableSlot,
    };
  }

  private delay(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Cleanup old data periodically
  cleanup(): void {
    const now = Date.now();
    const maxAge = 5 * 60 * 1000; // 5 minutes

    for (const [provider, record] of this.requests.entries()) {
      record.timestamps = record.timestamps.filter(
        timestamp => now - timestamp < maxAge
      );
      
      if (record.timestamps.length === 0 && now - record.lastRequestTime > maxAge) {
        this.requests.delete(provider);
      }
    }

    // Clean up old queued requests
    for (const [provider, queue] of this.queues.entries()) {
      const validRequests = queue.filter(req => now - req.timestamp < 30000);
      if (validRequests.length !== queue.length) {
        this.queues.set(provider, validRequests);
      }
      if (validRequests.length === 0) {
        this.queues.delete(provider);
      }
    }
  }

  // Start periodic cleanup
  startCleanup(): void {
    setInterval(() => this.cleanup(), 60000); // Cleanup every minute
  }
}

export const rateLimiter = new RateLimiter();

// Start cleanup when module is loaded
rateLimiter.startCleanup();

export default rateLimiter;