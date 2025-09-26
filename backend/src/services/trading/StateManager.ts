import Redis from 'ioredis';
import logger from '../../utils/logger';
import { TradingSession } from './TradingEngineService';
import { Order, Position, TradingSettings } from '../../types/trading';

interface MarketRegime {
  type: 'bull' | 'bear' | 'sideways' | 'high_volatility' | 'mean_reverting' | 'momentum';
  confidence: number;
  updatedAt: Date;
}

export class StateManager {
  private redis: Redis;
  private redisPub: Redis;
  private redisSub: Redis;
  private isInitialized: boolean = false;

  // Redis key prefixes
  private readonly KEYS = {
    SESSION: 'trading:session:',
    POSITION: 'trading:position:',
    ORDER: 'trading:order:',
    PENDING_ORDER: 'trading:pending_order:',
    LAST_TRADE: 'trading:last_trade:',
    MARKET_REGIME: 'trading:market_regime',
    SETTINGS: 'trading:settings:',
    QUEUE_ORDERS: 'trading:queue:orders',
    QUEUE_SIGNALS: 'trading:queue:signals',
    METRICS: 'trading:metrics:',
    CACHE_MARKET_DATA: 'cache:market_data:',
    LOCK: 'trading:lock:',
  };

  // TTL values (in seconds)
  private readonly TTL = {
    MARKET_DATA: 60, // 1 minute
    SESSION: 86400, // 24 hours
    METRICS: 3600, // 1 hour
    LOCK: 10, // 10 seconds for distributed locks
  };

  constructor() {
    // Initialize Redis connections
    const redisConfig = {
      host: process.env.REDIS_HOST || 'localhost',
      port: parseInt(process.env.REDIS_PORT || '6379'),
      password: process.env.REDIS_PASSWORD,
      retryStrategy: (times: number) => {
        const delay = Math.min(times * 50, 2000);
        return delay;
      },
    };

    this.redis = new Redis(redisConfig);
    this.redisPub = new Redis(redisConfig);
    this.redisSub = new Redis(redisConfig);

    // Handle connection events
    this.redis.on('connect', () => logger.info('Redis connected'));
    this.redis.on('error', (err) => logger.error('Redis error:', err));
  }

  async initialize(): Promise<void> {
    if (this.isInitialized) return;

    try {
      // Test Redis connection
      await this.redis.ping();

      // Set up pub/sub channels
      await this.setupPubSub();

      // Initialize Redis Streams for order queue
      await this.initializeStreams();

      this.isInitialized = true;
      logger.info('State Manager initialized with Redis');
    } catch (error) {
      logger.error('Failed to initialize State Manager:', error);
      throw error;
    }
  }

  private async setupPubSub(): Promise<void> {
    // Subscribe to channels for real-time updates
    await this.redisSub.subscribe(
      'trading:events',
      'trading:signals',
      'trading:orders',
      'trading:positions'
    );

    this.redisSub.on('message', (channel, message) => {
      this.handlePubSubMessage(channel, message);
    });
  }

  private async initializeStreams(): Promise<void> {
    // Create Redis streams for order and signal queues
    try {
      await this.redis.xgroup('CREATE', this.KEYS.QUEUE_ORDERS, 'trading-group', '$', 'MKSTREAM');
    } catch (err: any) {
      if (!err.message.includes('BUSYGROUP')) {
        throw err;
      }
    }

    try {
      await this.redis.xgroup('CREATE', this.KEYS.QUEUE_SIGNALS, 'trading-group', '$', 'MKSTREAM');
    } catch (err: any) {
      if (!err.message.includes('BUSYGROUP')) {
        throw err;
      }
    }
  }

  // Session Management
  async saveSession(session: TradingSession): Promise<void> {
    const key = this.KEYS.SESSION + session.userId;
    await this.redis.setex(
      key,
      this.TTL.SESSION,
      JSON.stringify(session)
    );
  }

  async getSession(userId: string): Promise<TradingSession | null> {
    const key = this.KEYS.SESSION + userId;
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  async getActiveSessions(): Promise<TradingSession[]> {
    const keys = await this.redis.keys(this.KEYS.SESSION + '*');
    const sessions: TradingSession[] = [];

    for (const key of keys) {
      const data = await this.redis.get(key);
      if (data) {
        const session = JSON.parse(data);
        if (session.status === 'active' || session.status === 'paused') {
          sessions.push(session);
        }
      }
    }

    return sessions;
  }

  // Position Management
  async savePosition(position: Position): Promise<void> {
    const key = this.KEYS.POSITION + position.id;
    await this.redis.set(key, JSON.stringify(position));

    // Also maintain a set of positions per user
    const userKey = this.KEYS.POSITION + 'user:' + position.userId;
    await this.redis.sadd(userKey, position.id);

    // Publish position update
    await this.redisPub.publish('trading:positions', JSON.stringify({
      type: 'position_update',
      position
    }));
  }

  async getPosition(positionId: string): Promise<Position | null> {
    const key = this.KEYS.POSITION + positionId;
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  async getUserPositions(userId: string): Promise<Position[]> {
    const userKey = this.KEYS.POSITION + 'user:' + userId;
    const positionIds = await this.redis.smembers(userKey);
    const positions: Position[] = [];

    for (const id of positionIds) {
      const position = await this.getPosition(id);
      if (position) positions.push(position);
    }

    return positions;
  }

  async getAllPositions(): Promise<Position[]> {
    const keys = await this.redis.keys(this.KEYS.POSITION + 'pos_*');
    const positions: Position[] = [];

    for (const key of keys) {
      const data = await this.redis.get(key);
      if (data) positions.push(JSON.parse(data));
    }

    return positions;
  }

  // Order Management
  async savePendingOrder(order: Order): Promise<void> {
    const key = this.KEYS.PENDING_ORDER + order.id;
    await this.redis.set(key, JSON.stringify(order));

    // Add to order stream for processing
    await this.redis.xadd(
      this.KEYS.QUEUE_ORDERS,
      '*',
      'orderId', order.id,
      'userId', order.userId,
      'symbol', order.symbol,
      'side', order.side,
      'type', order.type,
      'quantity', order.quantity.toString(),
      'timestamp', new Date().toISOString()
    );
  }

  async removePendingOrder(orderId: string): Promise<void> {
    const key = this.KEYS.PENDING_ORDER + orderId;
    await this.redis.del(key);
  }

  async getPendingOrders(): Promise<Order[]> {
    const keys = await this.redis.keys(this.KEYS.PENDING_ORDER + '*');
    const orders: Order[] = [];

    for (const key of keys) {
      const data = await this.redis.get(key);
      if (data) orders.push(JSON.parse(data));
    }

    return orders;
  }

  async saveExecutedOrder(order: Order): Promise<void> {
    const key = this.KEYS.ORDER + order.id;
    await this.redis.set(key, JSON.stringify(order));

    // Publish order update
    await this.redisPub.publish('trading:orders', JSON.stringify({
      type: 'order_executed',
      order
    }));
  }

  async getOrder(orderId: string): Promise<Order | null> {
    const key = this.KEYS.ORDER + orderId;
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  // Trading State Management
  async updateLastTrade(userId: string, symbol: string): Promise<void> {
    const key = this.KEYS.LAST_TRADE + userId;
    await this.redis.hset(key, symbol, Date.now().toString());
  }

  async getLastTrades(userId: string): Promise<Record<string, number>> {
    const key = this.KEYS.LAST_TRADE + userId;
    const trades = await this.redis.hgetall(key);
    const result: Record<string, number> = {};

    for (const [symbol, timestamp] of Object.entries(trades)) {
      result[symbol] = parseInt(timestamp);
    }

    return result;
  }

  // Market Regime
  async saveMarketRegime(regime: MarketRegime): Promise<void> {
    await this.redis.set(this.KEYS.MARKET_REGIME, JSON.stringify(regime));
  }

  async getMarketRegime(): Promise<MarketRegime> {
    const data = await this.redis.get(this.KEYS.MARKET_REGIME);
    return data ? JSON.parse(data) : {
      type: 'sideways',
      confidence: 0.5,
      updatedAt: new Date()
    };
  }

  // Settings Management
  async saveUserSettings(userId: string, settings: TradingSettings): Promise<void> {
    const key = this.KEYS.SETTINGS + userId;
    await this.redis.set(key, JSON.stringify(settings));
  }

  async getUserSettings(userId: string): Promise<TradingSettings | null> {
    const key = this.KEYS.SETTINGS + userId;
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  // Market Data Caching
  async cacheMarketData(symbol: string, data: any): Promise<void> {
    const key = this.KEYS.CACHE_MARKET_DATA + symbol;
    await this.redis.setex(key, this.TTL.MARKET_DATA, JSON.stringify(data));
  }

  async getCachedMarketData(symbol: string): Promise<any | null> {
    const key = this.KEYS.CACHE_MARKET_DATA + symbol;
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  // Metrics and Analytics
  async saveMetrics(userId: string, metrics: any): Promise<void> {
    const key = this.KEYS.METRICS + userId;
    await this.redis.setex(key, this.TTL.METRICS, JSON.stringify(metrics));
  }

  async getMetrics(userId: string): Promise<any | null> {
    const key = this.KEYS.METRICS + userId;
    const data = await this.redis.get(key);
    return data ? JSON.parse(data) : null;
  }

  // Distributed Locking (for preventing race conditions)
  async acquireLock(resource: string, ttl: number = this.TTL.LOCK): Promise<boolean> {
    const key = this.KEYS.LOCK + resource;
    const result = await this.redis.set(key, '1', 'NX', 'EX', ttl);
    return result === 'OK';
  }

  async releaseLock(resource: string): Promise<void> {
    const key = this.KEYS.LOCK + resource;
    await this.redis.del(key);
  }

  // Stream Processing for Order Queue
  async readOrderStream(count: number = 10): Promise<any[]> {
    try {
      const result = await this.redis.xreadgroup(
        'GROUP',
        'trading-group',
        'consumer-1',
        'COUNT',
        count,
        'STREAMS',
        this.KEYS.QUEUE_ORDERS,
        '>'
      );

      if (!result || result.length === 0) return [];

      const orders = [];
      const stream = result[0];
      const messages = stream[1];

      for (const message of messages) {
        const [id, fields] = message;
        const order: any = { streamId: id };

        // Parse field pairs
        for (let i = 0; i < fields.length; i += 2) {
          order[fields[i]] = fields[i + 1];
        }

        orders.push(order);

        // Acknowledge message
        await this.redis.xack(this.KEYS.QUEUE_ORDERS, 'trading-group', id);
      }

      return orders;
    } catch (error) {
      logger.error('Error reading order stream:', error);
      return [];
    }
  }

  // Signal Queue
  async addSignalToQueue(signal: any): Promise<void> {
    await this.redis.xadd(
      this.KEYS.QUEUE_SIGNALS,
      '*',
      'signal', JSON.stringify(signal),
      'timestamp', new Date().toISOString()
    );
  }

  async readSignalStream(count: number = 10): Promise<any[]> {
    try {
      const result = await this.redis.xreadgroup(
        'GROUP',
        'trading-group',
        'consumer-1',
        'COUNT',
        count,
        'STREAMS',
        this.KEYS.QUEUE_SIGNALS,
        '>'
      );

      if (!result || result.length === 0) return [];

      const signals = [];
      const stream = result[0];
      const messages = stream[1];

      for (const message of messages) {
        const [id, fields] = message;
        const signalData = fields[fields.indexOf('signal') + 1];
        signals.push(JSON.parse(signalData));

        // Acknowledge message
        await this.redis.xack(this.KEYS.QUEUE_SIGNALS, 'trading-group', id);
      }

      return signals;
    } catch (error) {
      logger.error('Error reading signal stream:', error);
      return [];
    }
  }

  // Pub/Sub Message Handler
  private handlePubSubMessage(channel: string, message: string): void {
    try {
      const data = JSON.parse(message);
      logger.debug(`Received message on channel ${channel}:`, data);

      // Handle different message types
      switch (channel) {
        case 'trading:events':
          this.handleTradingEvent(data);
          break;
        case 'trading:signals':
          this.handleSignalEvent(data);
          break;
        case 'trading:orders':
          this.handleOrderEvent(data);
          break;
        case 'trading:positions':
          this.handlePositionEvent(data);
          break;
      }
    } catch (error) {
      logger.error(`Error handling pub/sub message on ${channel}:`, error);
    }
  }

  private handleTradingEvent(data: any): void {
    // Process trading events
    logger.debug('Trading event received:', data);
  }

  private handleSignalEvent(data: any): void {
    // Process signal events
    logger.debug('Signal event received:', data);
  }

  private handleOrderEvent(data: any): void {
    // Process order events
    logger.debug('Order event received:', data);
  }

  private handlePositionEvent(data: any): void {
    // Process position events
    logger.debug('Position event received:', data);
  }

  // Publish Events
  async publishEvent(channel: string, event: any): Promise<void> {
    await this.redisPub.publish(channel, JSON.stringify(event));
  }

  // Cleanup
  async cleanup(): Promise<void> {
    await this.redisSub.unsubscribe();
    await this.redis.quit();
    await this.redisPub.quit();
    await this.redisSub.quit();
  }
}

export default StateManager;