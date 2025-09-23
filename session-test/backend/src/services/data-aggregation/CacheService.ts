import logger from '../../utils/logger';

export interface CacheStats {
  hitRate: number;
  size: number;
  maxSize: number;
  totalHits: number;
  totalMisses: number;
}

export interface CacheConfig {
  maxSize: number;
  ttl: {
    onchain: number;
    funding: number;
    whale: number;
    market: number;
  };
}

interface CacheEntry<T> {
  data: T;
  expiresAt: number;
  createdAt: number;
}

export class CacheService {
  private cache: Map<string, CacheEntry<any>> = new Map();
  private config: CacheConfig;
  private stats = {
    hits: 0,
    misses: 0,
    sets: 0,
    evictions: 0
  };
  private cleanupInterval: NodeJS.Timeout | null = null;

  constructor(config: CacheConfig) {
    this.config = config;
  }

  async initialize(): Promise<void> {
    logger.info('Initializing CacheService...');
    
    // Start cleanup interval (every 5 minutes)
    this.cleanupInterval = setInterval(() => {
      this.cleanup();
    }, 5 * 60 * 1000);

    logger.info('CacheService initialized');
  }

  async get<T>(key: string): Promise<T | null> {
    const entry = this.cache.get(key);
    
    if (!entry) {
      this.stats.misses++;
      return null;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      this.stats.misses++;
      return null;
    }

    this.stats.hits++;
    return entry.data as T;
  }

  async set<T>(key: string, data: T, ttlSeconds?: number): Promise<void> {
    const now = Date.now();
    const ttl = ttlSeconds || this.config.ttl.market;
    
    const entry: CacheEntry<T> = {
      data,
      expiresAt: now + (ttl * 1000),
      createdAt: now
    };

    // Check if cache is full and evict if necessary
    if (this.cache.size >= this.config.maxSize && !this.cache.has(key)) {
      this.evictLeastRecentlyUsed();
    }

    this.cache.set(key, entry);
    this.stats.sets++;
  }

  async delete(key: string): Promise<boolean> {
    return this.cache.delete(key);
  }

  async clear(): Promise<void> {
    this.cache.clear();
    this.stats = {
      hits: 0,
      misses: 0,
      sets: 0,
      evictions: 0
    };
  }

  async has(key: string): Promise<boolean> {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return false;
    }

    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  async getKeys(pattern?: string): Promise<string[]> {
    const keys = Array.from(this.cache.keys());
    
    if (!pattern) {
      return keys;
    }

    // Simple pattern matching (supports * wildcard)
    const regex = new RegExp(pattern.replace(/\*/g, '.*'));
    return keys.filter(key => regex.test(key));
  }

  async getStats(): Promise<CacheStats> {
    const totalRequests = this.stats.hits + this.stats.misses;
    const hitRate = totalRequests > 0 ? (this.stats.hits / totalRequests) * 100 : 0;

    return {
      hitRate: Math.round(hitRate * 100) / 100,
      size: this.cache.size,
      maxSize: this.config.maxSize,
      totalHits: this.stats.hits,
      totalMisses: this.stats.misses
    };
  }

  private cleanup(): void {
    const now = Date.now();
    let cleanedCount = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
        cleanedCount++;
      }
    }

    if (cleanedCount > 0) {
      logger.debug(`Cache cleanup: removed ${cleanedCount} expired entries`);
    }
  }

  private evictLeastRecentlyUsed(): void {
    // Find the oldest entry
    let oldestKey: string | null = null;
    let oldestTime = Date.now();

    for (const [key, entry] of this.cache.entries()) {
      if (entry.createdAt < oldestTime) {
        oldestTime = entry.createdAt;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      this.cache.delete(oldestKey);
      this.stats.evictions++;
      logger.debug(`Cache eviction: removed key ${oldestKey}`);
    }
  }

  async shutdown(): Promise<void> {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
      this.cleanupInterval = null;
    }

    this.cache.clear();
    logger.info('CacheService shutdown complete');
  }

  // Advanced cache operations

  async mget<T>(keys: string[]): Promise<Array<T | null>> {
    return Promise.all(keys.map(key => this.get<T>(key)));
  }

  async mset<T>(entries: Array<{ key: string; data: T; ttl?: number }>): Promise<void> {
    await Promise.all(entries.map(entry => 
      this.set(entry.key, entry.data, entry.ttl)
    ));
  }

  async increment(key: string, by: number = 1, ttl?: number): Promise<number> {
    const current = await this.get<number>(key) || 0;
    const newValue = current + by;
    await this.set(key, newValue, ttl);
    return newValue;
  }

  async setIfNotExists<T>(key: string, data: T, ttlSeconds?: number): Promise<boolean> {
    if (await this.has(key)) {
      return false;
    }

    await this.set(key, data, ttlSeconds);
    return true;
  }

  async getOrSet<T>(key: string, factory: () => Promise<T>, ttlSeconds?: number): Promise<T> {
    const cached = await this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    const data = await factory();
    await this.set(key, data, ttlSeconds);
    return data;
  }

  // Cache warming methods
  async warmup(warmupFunction: () => Promise<Array<{ key: string; data: any; ttl?: number }>>): Promise<void> {
    try {
      logger.info('Starting cache warmup...');
      const entries = await warmupFunction();
      await this.mset(entries);
      logger.info(`Cache warmup complete: ${entries.length} entries loaded`);
    } catch (error) {
      logger.error('Cache warmup failed:', error);
    }
  }

  // Memory usage estimation
  getMemoryUsage(): {
    entriesCount: number;
    estimatedSizeBytes: number;
  } {
    let estimatedSize = 0;
    
    for (const [key, entry] of this.cache.entries()) {
      // Rough estimation: key size + JSON size of data + overhead
      estimatedSize += key.length * 2; // UTF-16 encoding
      estimatedSize += JSON.stringify(entry.data).length * 2;
      estimatedSize += 100; // Overhead for entry structure
    }

    return {
      entriesCount: this.cache.size,
      estimatedSizeBytes: estimatedSize
    };
  }

  // Debug methods
  getEntry(key: string): CacheEntry<any> | undefined {
    return this.cache.get(key);
  }

  getAllEntries(): Array<{ key: string; entry: CacheEntry<any> }> {
    return Array.from(this.cache.entries()).map(([key, entry]) => ({ key, entry }));
  }

  // Cache health check
  async healthCheck(): Promise<{
    status: 'healthy' | 'degraded' | 'unhealthy';
    issues: string[];
  }> {
    const issues: string[] = [];
    let status: 'healthy' | 'degraded' | 'unhealthy' = 'healthy';

    const stats = await this.getStats();
    const memoryUsage = this.getMemoryUsage();

    // Check hit rate
    if (stats.hitRate < 30) {
      issues.push(`Low cache hit rate: ${stats.hitRate}%`);
      status = 'degraded';
    }

    // Check cache utilization
    const utilizationPercent = (stats.size / stats.maxSize) * 100;
    if (utilizationPercent > 90) {
      issues.push(`High cache utilization: ${utilizationPercent}%`);
      status = 'degraded';
    }

    // Check memory usage (rough estimate)
    if (memoryUsage.estimatedSizeBytes > 100 * 1024 * 1024) { // 100MB
      issues.push(`High memory usage: ${Math.round(memoryUsage.estimatedSizeBytes / 1024 / 1024)}MB`);
      status = 'degraded';
    }

    if (issues.length > 2) {
      status = 'unhealthy';
    }

    return { status, issues };
  }
}