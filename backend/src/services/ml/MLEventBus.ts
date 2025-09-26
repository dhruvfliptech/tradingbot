/**
 * ML Event Bus Service
 * Manages event-driven communication between Trading Engine, ML Service, and RL Service
 * Uses Redis Streams for reliable event processing
 */

import { EventEmitter } from 'events';
import Redis from 'ioredis';
import logger from '../../utils/logger';
import {
  MLEventType,
  MLEvent,
  BaseEvent,
  MarketDataUpdateEvent,
  TradeExecutedEvent,
  RLSignalGeneratedEvent,
  ThresholdAdjustedEvent,
  MLEventFactory
} from '../../types/ml-events';

interface EventBusConfig {
  redis: {
    host: string;
    port: number;
    password?: string;
    db?: number;
  };
  streams: {
    maxLen: number; // Maximum events per stream
    blockTimeout: number; // Milliseconds to block when reading
    batchSize: number; // Events to read per batch
  };
  consumerGroup: string;
  consumerId: string;
}

interface StreamInfo {
  stream: string;
  consumerGroup: string;
  lastId: string;
  pending: number;
}

export class MLEventBus extends EventEmitter {
  private redis: Redis;
  private subscriber: Redis;
  private config: EventBusConfig;
  private isRunning: boolean = false;
  private consumers: Map<string, NodeJS.Timeout> = new Map();
  private eventHandlers: Map<MLEventType, Set<(event: MLEvent) => Promise<void>>> = new Map();
  private circuitBreakers: Map<string, CircuitBreaker> = new Map();

  // Stream names for different event categories
  private readonly STREAMS = {
    MARKET: 'ml:stream:market',
    ML_SERVICE: 'ml:stream:ml',
    RL_SERVICE: 'ml:stream:rl',
    FEEDBACK: 'ml:stream:feedback',
    SYSTEM: 'ml:stream:system'
  };

  constructor(config?: Partial<EventBusConfig>) {
    super();

    this.config = {
      redis: {
        host: process.env.REDIS_HOST || 'localhost',
        port: parseInt(process.env.REDIS_PORT || '6379'),
        password: process.env.REDIS_PASSWORD,
        db: parseInt(process.env.REDIS_DB || '0')
      },
      streams: {
        maxLen: 10000,
        blockTimeout: 5000,
        batchSize: 10
      },
      consumerGroup: process.env.ML_CONSUMER_GROUP || 'ml-pipeline',
      consumerId: `${process.env.SERVICE_NAME || 'trading-engine'}-${process.pid}`,
      ...config
    };

    this.redis = new Redis(this.config.redis);
    this.subscriber = new Redis(this.config.redis);

    this.initializeCircuitBreakers();
    this.setupErrorHandlers();
  }

  /**
   * Initialize circuit breakers for each service
   */
  private initializeCircuitBreakers(): void {
    this.circuitBreakers.set('ml-service', new CircuitBreaker('ml-service', {
      failureThreshold: 5,
      resetTimeout: 30000,
      monitoringPeriod: 60000
    }));

    this.circuitBreakers.set('rl-service', new CircuitBreaker('rl-service', {
      failureThreshold: 5,
      resetTimeout: 30000,
      monitoringPeriod: 60000
    }));
  }

  /**
   * Setup error handlers for Redis connections
   */
  private setupErrorHandlers(): void {
    this.redis.on('error', (error) => {
      logger.error('Redis connection error:', error);
      this.emit('error', error);
    });

    this.subscriber.on('error', (error) => {
      logger.error('Redis subscriber error:', error);
      this.emit('error', error);
    });

    this.redis.on('connect', () => {
      logger.info('MLEventBus connected to Redis');
      this.emit('connected');
    });

    this.redis.on('disconnect', () => {
      logger.warn('MLEventBus disconnected from Redis');
      this.emit('disconnected');
    });
  }

  /**
   * Start the event bus and begin consuming streams
   */
  async start(): Promise<void> {
    if (this.isRunning) {
      logger.warn('MLEventBus is already running');
      return;
    }

    try {
      // Create consumer groups if they don't exist
      await this.initializeConsumerGroups();

      // Start consuming from streams
      this.startConsumers();

      this.isRunning = true;
      logger.info('MLEventBus started successfully');

    } catch (error) {
      logger.error('Failed to start MLEventBus:', error);
      throw error;
    }
  }

  /**
   * Initialize consumer groups for all streams
   */
  private async initializeConsumerGroups(): Promise<void> {
    const streams = Object.values(this.STREAMS);

    for (const stream of streams) {
      try {
        // Try to create consumer group
        await this.redis.xgroup(
          'CREATE',
          stream,
          this.config.consumerGroup,
          '$',
          'MKSTREAM'
        );
        logger.info(`Created consumer group ${this.config.consumerGroup} for stream ${stream}`);
      } catch (error: any) {
        // Group might already exist, which is fine
        if (!error.message.includes('BUSYGROUP')) {
          logger.error(`Error creating consumer group for ${stream}:`, error);
        }
      }
    }
  }

  /**
   * Start consuming from all streams
   */
  private startConsumers(): void {
    // Market events consumer
    this.startStreamConsumer(this.STREAMS.MARKET, async (events) => {
      for (const event of events) {
        await this.processMarketEvent(event);
      }
    });

    // ML service events consumer
    this.startStreamConsumer(this.STREAMS.ML_SERVICE, async (events) => {
      for (const event of events) {
        await this.processMLServiceEvent(event);
      }
    });

    // RL service events consumer
    this.startStreamConsumer(this.STREAMS.RL_SERVICE, async (events) => {
      for (const event of events) {
        await this.processRLServiceEvent(event);
      }
    });

    // Feedback events consumer
    this.startStreamConsumer(this.STREAMS.FEEDBACK, async (events) => {
      for (const event of events) {
        await this.processFeedbackEvent(event);
      }
    });
  }

  /**
   * Start consuming from a specific stream
   */
  private startStreamConsumer(
    stream: string,
    handler: (events: MLEvent[]) => Promise<void>
  ): void {
    const consume = async () => {
      while (this.isRunning) {
        try {
          // Read from stream using consumer group
          const result = await this.redis.xreadgroup(
            'GROUP',
            this.config.consumerGroup,
            this.config.consumerId,
            'BLOCK',
            this.config.streams.blockTimeout,
            'COUNT',
            this.config.streams.batchSize,
            'STREAMS',
            stream,
            '>'
          );

          if (result && result.length > 0) {
            const [, messages] = result[0];
            const events = messages.map(([id, fields]) => this.parseEvent(id, fields));

            // Process events
            await handler(events);

            // Acknowledge processed messages
            const messageIds = messages.map(([id]) => id);
            if (messageIds.length > 0) {
              await this.redis.xack(stream, this.config.consumerGroup, ...messageIds);
            }
          }
        } catch (error) {
          logger.error(`Error consuming from stream ${stream}:`, error);
          await this.sleep(1000); // Back off on error
        }
      }
    };

    // Start consuming in background
    consume().catch(error => {
      logger.error(`Fatal error in stream consumer ${stream}:`, error);
    });
  }

  /**
   * Publish an event to the appropriate stream
   */
  async publishEvent(event: MLEvent): Promise<string> {
    const stream = this.getStreamForEvent(event.type);
    const circuitBreaker = this.getCircuitBreakerForEvent(event.type);

    // Check circuit breaker
    if (circuitBreaker && !circuitBreaker.canExecute()) {
      logger.warn(`Circuit breaker open for ${event.type}, event dropped`);
      this.emit('event:dropped', event);
      return '';
    }

    try {
      // Serialize event
      const eventData = this.serializeEvent(event);

      // Add to stream with max length to prevent unbounded growth
      const messageId = await this.redis.xadd(
        stream,
        'MAXLEN',
        '~',
        this.config.streams.maxLen,
        '*',
        ...eventData
      );

      logger.debug(`Published event ${event.id} to stream ${stream}`, {
        type: event.type,
        userId: event.userId,
        messageId
      });

      // Record success
      circuitBreaker?.recordSuccess();

      // Emit for local listeners
      this.emit(`event:published:${event.type}`, event);

      return messageId;

    } catch (error) {
      logger.error(`Failed to publish event ${event.id}:`, error);
      circuitBreaker?.recordFailure();
      throw error;
    }
  }

  /**
   * Subscribe to specific event types
   */
  onEvent(
    eventType: MLEventType,
    handler: (event: MLEvent) => Promise<void>
  ): void {
    if (!this.eventHandlers.has(eventType)) {
      this.eventHandlers.set(eventType, new Set());
    }
    this.eventHandlers.get(eventType)!.add(handler);
  }

  /**
   * Unsubscribe from event type
   */
  offEvent(
    eventType: MLEventType,
    handler: (event: MLEvent) => Promise<void>
  ): void {
    const handlers = this.eventHandlers.get(eventType);
    if (handlers) {
      handlers.delete(handler);
    }
  }

  /**
   * Process market events
   */
  private async processMarketEvent(event: MLEvent): Promise<void> {
    const handlers = this.eventHandlers.get(event.type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          await handler(event);
        } catch (error) {
          logger.error(`Error processing market event ${event.id}:`, error);
        }
      }
    }

    // Emit for Socket.IO or other listeners
    this.emit('market:event', event);
  }

  /**
   * Process ML service events
   */
  private async processMLServiceEvent(event: MLEvent): Promise<void> {
    const handlers = this.eventHandlers.get(event.type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          await handler(event);
        } catch (error) {
          logger.error(`Error processing ML service event ${event.id}:`, error);
        }
      }
    }

    this.emit('ml:event', event);
  }

  /**
   * Process RL service events
   */
  private async processRLServiceEvent(event: MLEvent): Promise<void> {
    const handlers = this.eventHandlers.get(event.type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          await handler(event);
        } catch (error) {
          logger.error(`Error processing RL service event ${event.id}:`, error);
        }
      }
    }

    this.emit('rl:event', event);
  }

  /**
   * Process feedback events
   */
  private async processFeedbackEvent(event: MLEvent): Promise<void> {
    const handlers = this.eventHandlers.get(event.type);
    if (handlers) {
      for (const handler of handlers) {
        try {
          await handler(event);
        } catch (error) {
          logger.error(`Error processing feedback event ${event.id}:`, error);
        }
      }
    }

    this.emit('feedback:event', event);
  }

  /**
   * Get stream name for event type
   */
  private getStreamForEvent(eventType: MLEventType): string {
    if (eventType.startsWith('market.')) return this.STREAMS.MARKET;
    if (eventType.startsWith('ml.')) return this.STREAMS.ML_SERVICE;
    if (eventType.startsWith('rl.')) return this.STREAMS.RL_SERVICE;
    if (eventType.startsWith('feedback.')) return this.STREAMS.FEEDBACK;
    return this.STREAMS.SYSTEM;
  }

  /**
   * Get circuit breaker for event type
   */
  private getCircuitBreakerForEvent(eventType: MLEventType): CircuitBreaker | null {
    if (eventType.startsWith('ml.')) return this.circuitBreakers.get('ml-service') || null;
    if (eventType.startsWith('rl.')) return this.circuitBreakers.get('rl-service') || null;
    return null;
  }

  /**
   * Serialize event for Redis
   */
  private serializeEvent(event: MLEvent): string[] {
    const flat: string[] = [];
    const addToFlat = (obj: any, prefix: string = '') => {
      for (const [key, value] of Object.entries(obj)) {
        const fullKey = prefix ? `${prefix}.${key}` : key;
        if (typeof value === 'object' && value !== null && !(value instanceof Date)) {
          addToFlat(value, fullKey);
        } else {
          flat.push(fullKey, String(value));
        }
      }
    };
    addToFlat(event);
    return flat;
  }

  /**
   * Parse event from Redis
   */
  private parseEvent(id: string, fields: string[]): MLEvent {
    const event: any = { _streamId: id };

    for (let i = 0; i < fields.length; i += 2) {
      const key = fields[i];
      const value = fields[i + 1];

      // Reconstruct nested object
      const keys = key.split('.');
      let current = event;

      for (let j = 0; j < keys.length - 1; j++) {
        if (!current[keys[j]]) {
          current[keys[j]] = {};
        }
        current = current[keys[j]];
      }

      // Parse value
      const lastKey = keys[keys.length - 1];
      if (value === 'true') current[lastKey] = true;
      else if (value === 'false') current[lastKey] = false;
      else if (!isNaN(Number(value))) current[lastKey] = Number(value);
      else if (lastKey.includes('Date') || lastKey === 'timestamp') {
        current[lastKey] = new Date(value);
      }
      else current[lastKey] = value;
    }

    return event as MLEvent;
  }

  /**
   * Get stream information
   */
  async getStreamInfo(): Promise<StreamInfo[]> {
    const info: StreamInfo[] = [];

    for (const [name, stream] of Object.entries(this.STREAMS)) {
      try {
        const streamInfo = await this.redis.xinfo('STREAM', stream);
        const groupInfo = await this.redis.xinfo('GROUPS', stream);

        info.push({
          stream: name,
          consumerGroup: this.config.consumerGroup,
          lastId: streamInfo[9] || '0-0',
          pending: groupInfo[0]?.[7] || 0
        });
      } catch (error) {
        logger.error(`Error getting stream info for ${name}:`, error);
      }
    }

    return info;
  }

  /**
   * Stop the event bus
   */
  async stop(): Promise<void> {
    this.isRunning = false;

    // Clear all consumers
    for (const [, timeout] of this.consumers) {
      clearTimeout(timeout);
    }
    this.consumers.clear();

    // Close Redis connections
    await this.redis.quit();
    await this.subscriber.quit();

    logger.info('MLEventBus stopped');
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

/**
 * Circuit Breaker for service protection
 */
class CircuitBreaker {
  private failures: number = 0;
  private lastFailureTime: number = 0;
  private state: 'closed' | 'open' | 'half-open' = 'closed';

  constructor(
    private name: string,
    private config: {
      failureThreshold: number;
      resetTimeout: number;
      monitoringPeriod: number;
    }
  ) {}

  canExecute(): boolean {
    if (this.state === 'closed') return true;

    if (this.state === 'open') {
      const now = Date.now();
      if (now - this.lastFailureTime > this.config.resetTimeout) {
        this.state = 'half-open';
        return true;
      }
      return false;
    }

    return true; // half-open: allow one request
  }

  recordSuccess(): void {
    if (this.state === 'half-open') {
      this.reset();
    }
  }

  recordFailure(): void {
    this.failures++;
    this.lastFailureTime = Date.now();

    if (this.failures >= this.config.failureThreshold) {
      this.state = 'open';
      logger.warn(`Circuit breaker ${this.name} opened after ${this.failures} failures`);
    }
  }

  private reset(): void {
    this.failures = 0;
    this.state = 'closed';
    logger.info(`Circuit breaker ${this.name} reset to closed state`);
  }
}

// Singleton instance
let eventBusInstance: MLEventBus | null = null;

export function getMLEventBus(): MLEventBus {
  if (!eventBusInstance) {
    eventBusInstance = new MLEventBus();
  }
  return eventBusInstance;
}