interface CacheEntry<T> {
  data: T;
  timestamp: number;
  expiry: number;
  dependencies?: string[];
}

interface CacheConfig {
  maxSize: number;
  defaultTTL: number; // Time to live in milliseconds
}

class DataCache {
  private cache: Map<string, CacheEntry<any>>;
  private config: CacheConfig;
  private accessTimes: Map<string, number>; // For LRU eviction
  private cleanupInterval: NodeJS.Timeout;

  constructor(config: Partial<CacheConfig> = {}) {
    this.cache = new Map();
    this.accessTimes = new Map();
    this.config = {
      maxSize: config.maxSize || 100,
      defaultTTL: config.defaultTTL || 5 * 60 * 1000, // 5 minutes
    };

    // Start periodic cleanup
    this.cleanupInterval = setInterval(() => this.cleanup(), 30000); // Every 30 seconds
    
    console.log('🗄️ DataCache initialized:', this.config);
  }

  // Store data in cache
  set<T>(key: string, data: T, ttl?: number, dependencies?: string[]): void {
    const now = Date.now();
    const expiry = now + (ttl || this.config.defaultTTL);

    // Check if cache is at max size and evict LRU entry
    if (this.cache.size >= this.config.maxSize && !this.cache.has(key)) {
      this.evictLRU();
    }

    const entry: CacheEntry<T> = {
      data,
      timestamp: now,
      expiry,
      dependencies,
    };

    this.cache.set(key, entry);
    this.accessTimes.set(key, now);

    console.log(`💾 Cache SET: ${key}, expires in ${((expiry - now) / 1000).toFixed(0)}s`);
  }

  // Retrieve data from cache
  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    if (!entry) {
      console.log(`🔍 Cache MISS: ${key}`);
      return null;
    }

    const now = Date.now();
    
    // Check if entry has expired
    if (now > entry.expiry) {
      console.log(`⏰ Cache EXPIRED: ${key}`);
      this.delete(key);
      return null;
    }

    // Update access time for LRU
    this.accessTimes.set(key, now);
    
    console.log(`✅ Cache HIT: ${key}, age: ${((now - entry.timestamp) / 1000).toFixed(0)}s`);
    return entry.data;
  }

  // Check if key exists and is not expired
  has(key: string): boolean {
    const entry = this.cache.get(key);
    if (!entry) return false;

    if (Date.now() > entry.expiry) {
      this.delete(key);
      return false;
    }

    return true;
  }

  // Delete specific key
  delete(key: string): void {
    this.cache.delete(key);
    this.accessTimes.delete(key);
  }

  // Clear cache by pattern
  clearByPattern(pattern: RegExp): number {
    let deletedCount = 0;
    
    for (const key of this.cache.keys()) {
      if (pattern.test(key)) {
        this.delete(key);
        deletedCount++;
      }
    }

    console.log(`🧹 Cache cleared ${deletedCount} entries matching pattern:`, pattern);
    return deletedCount;
  }

  // Clear cache by dependency
  clearByDependency(dependency: string): number {
    let deletedCount = 0;
    
    for (const [key, entry] of this.cache.entries()) {
      if (entry.dependencies?.includes(dependency)) {
        this.delete(key);
        deletedCount++;
      }
    }

    console.log(`🧹 Cache cleared ${deletedCount} entries with dependency: ${dependency}`);
    return deletedCount;
  }

  // Get or set pattern - useful for async operations
  async getOrSet<T>(
    key: string,
    fetcher: () => Promise<T>,
    ttl?: number,
    dependencies?: string[]
  ): Promise<T> {
    // Check if data exists in cache
    const cached = this.get<T>(key);
    if (cached !== null) {
      return cached;
    }

    // Fetch fresh data
    console.log(`📡 Cache fetching fresh data for: ${key}`);
    try {
      const data = await fetcher();
      this.set(key, data, ttl, dependencies);
      return data;
    } catch (error) {
      console.error(`❌ Cache fetch failed for ${key}:`, error);
      throw error;
    }
  }

  // Batch operations
  getBatch<T>(keys: string[]): Map<string, T | null> {
    const results = new Map<string, T | null>();
    
    for (const key of keys) {
      results.set(key, this.get<T>(key));
    }

    return results;
  }

  setBatch<T>(entries: Array<{ key: string; data: T; ttl?: number; dependencies?: string[] }>): void {
    for (const entry of entries) {
      this.set(entry.key, entry.data, entry.ttl, entry.dependencies);
    }
  }

  // Evict least recently used entry
  private evictLRU(): void {
    let oldestKey: string | undefined;
    let oldestTime = Date.now();

    for (const [key, accessTime] of this.accessTimes.entries()) {
      if (accessTime < oldestTime) {
        oldestTime = accessTime;
        oldestKey = key;
      }
    }

    if (oldestKey) {
      console.log(`🗑️ Cache evicted LRU entry: ${oldestKey}`);
      this.delete(oldestKey);
    }
  }

  // Cleanup expired entries
  private cleanup(): void {
    const now = Date.now();
    let expiredCount = 0;

    for (const [key, entry] of this.cache.entries()) {
      if (now > entry.expiry) {
        this.delete(key);
        expiredCount++;
      }
    }

    if (expiredCount > 0) {
      console.log(`🧹 Cache cleanup removed ${expiredCount} expired entries`);
    }
  }

  // Get cache statistics
  getStats(): {
    size: number;
    maxSize: number;
    entries: Array<{
      key: string;
      age: number;
      ttl: number;
      size: number;
      dependencies?: string[];
    }>;
  } {
    const now = Date.now();
    const entries = Array.from(this.cache.entries()).map(([key, entry]) => ({
      key,
      age: now - entry.timestamp,
      ttl: entry.expiry - now,
      size: JSON.stringify(entry.data).length,
      dependencies: entry.dependencies,
    }));

    return {
      size: this.cache.size,
      maxSize: this.config.maxSize,
      entries: entries.sort((a, b) => b.age - a.age), // Sort by age descending
    };
  }

  // Clear all cache
  clear(): void {
    const size = this.cache.size;
    this.cache.clear();
    this.accessTimes.clear();
    console.log(`🧹 Cache cleared all ${size} entries`);
  }

  // Destroy cache and cleanup
  destroy(): void {
    if (this.cleanupInterval) {
      clearInterval(this.cleanupInterval);
    }
    this.clear();
    console.log('💥 DataCache destroyed');
  }

  // Cache key generators for different data types
  static generateKey(type: string, identifier: string, params?: Record<string, any>): string {
    const baseKey = `${type}:${identifier}`;
    if (!params || Object.keys(params).length === 0) {
      return baseKey;
    }
    
    // Sort params to ensure consistent keys
    const sortedParams = Object.keys(params)
      .sort()
      .map(key => `${key}=${params[key]}`)
      .join('&');
    
    return `${baseKey}?${sortedParams}`;
  }

  // Precomputed key generators for common use cases
  static keys = {
    price: (symbol: string, exchange?: string) => 
      DataCache.generateKey('price', symbol, exchange ? { exchange } : undefined),
    
    gasMetrics: () => 
      DataCache.generateKey('gas', 'metrics'),
    
    whaleTransactions: (minValue: number) => 
      DataCache.generateKey('whale', 'transactions', { minValue }),
    
    dexTrades: (token: string, timeframe: string) => 
      DataCache.generateKey('dex', token, { timeframe }),
    
    liquidityPools: (token: string) => 
      DataCache.generateKey('liquidity', token),
    
    tokenHolders: (token: string, limit: number) => 
      DataCache.generateKey('holders', token, { limit }),
    
    activeAddresses: (timeframe: string) => 
      DataCache.generateKey('addresses', 'active', { timeframe }),
    
    strategySignal: (symbol: string, strategy: string) => 
      DataCache.generateKey('strategy', symbol, { strategy }),
  };
}

// Global cache instance
export const dataCache = new DataCache({
  maxSize: 150, // Increased for trading data
  defaultTTL: 2 * 60 * 1000, // 2 minutes for real-time trading data
});

// Specialized caches for different data types with different TTLs
export const priceCache = new DataCache({
  maxSize: 50,
  defaultTTL: 30 * 1000, // 30 seconds for price data
});

export const staticDataCache = new DataCache({
  maxSize: 100,
  defaultTTL: 30 * 60 * 1000, // 30 minutes for static data (token info, etc.)
});

export const analysisCache = new DataCache({
  maxSize: 200,
  defaultTTL: 5 * 60 * 1000, // 5 minutes for analysis results
});

// Cleanup all caches on page unload
if (typeof window !== 'undefined') {
  window.addEventListener('beforeunload', () => {
    dataCache.destroy();
    priceCache.destroy();
    staticDataCache.destroy();
    analysisCache.destroy();
  });
}

export { DataCache };
export default dataCache;