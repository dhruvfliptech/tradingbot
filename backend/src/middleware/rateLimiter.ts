/**
 * Rate Limiting Middleware
 * Prevents abuse and ensures fair API usage
 */

import rateLimit from 'express-rate-limit';
import { Request, Response } from 'express';
import logger from '../utils/logger';

/**
 * Create a rate limiter with custom options
 */
const createRateLimiter = (options: {
  windowMs: number;
  max: number;
  message?: string;
  skipSuccessfulRequests?: boolean;
}) => {
  return rateLimit({
    windowMs: options.windowMs,
    max: options.max,
    message: options.message || 'Too many requests, please try again later.',
    standardHeaders: true, // Return rate limit info in `RateLimit-*` headers
    legacyHeaders: false, // Disable `X-RateLimit-*` headers
    skipSuccessfulRequests: options.skipSuccessfulRequests || false,
    handler: (req: Request, res: Response) => {
      logger.warn('Rate limit exceeded:', {
        ip: req.ip,
        path: req.path,
        user: (req as any).user?.id
      });
      res.status(429).json({
        success: false,
        error: options.message || 'Too many requests, please try again later.',
        retryAfter: res.getHeader('Retry-After')
      });
    }
  });
};

/**
 * General API rate limiter
 * 100 requests per 15 minutes
 */
export const apiLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  message: 'API rate limit exceeded. Please wait before making more requests.'
});

/**
 * Strict rate limiter for authentication endpoints
 * 5 requests per 15 minutes
 */
export const authLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 5,
  message: 'Too many authentication attempts. Please try again later.',
  skipSuccessfulRequests: true // Don't count successful auth
});

/**
 * Trading operations rate limiter
 * 30 requests per minute
 */
export const tradingLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: 30,
  message: 'Trading rate limit exceeded. Please slow down your trading activity.'
});

/**
 * Order placement rate limiter
 * 10 orders per minute
 */
export const orderLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: 10,
  message: 'Order placement limit exceeded. Maximum 10 orders per minute.'
});

/**
 * Bot control rate limiter
 * 5 operations per minute
 */
export const botControlLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: 5,
  message: 'Bot control rate limit exceeded. Please wait before making changes.'
});

/**
 * Market data rate limiter
 * 60 requests per minute
 */
export const marketDataLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: 60,
  message: 'Market data rate limit exceeded.'
});

/**
 * WebSocket connection rate limiter
 * 3 connections per minute
 */
export const websocketLimiter = createRateLimiter({
  windowMs: 60 * 1000, // 1 minute
  max: 3,
  message: 'Too many WebSocket connection attempts. Please wait.'
});

/**
 * Dynamic rate limiter based on user tier
 */
export const tierBasedLimiter = (req: Request, res: Response, next: Function) => {
  const userTier = (req as any).user?.tier || 'basic';

  // Define limits per tier
  const tierLimits: Record<string, { windowMs: number; max: number }> = {
    basic: { windowMs: 15 * 60 * 1000, max: 100 },      // 100 requests per 15 min
    premium: { windowMs: 15 * 60 * 1000, max: 500 },    // 500 requests per 15 min
    professional: { windowMs: 15 * 60 * 1000, max: 2000 } // 2000 requests per 15 min
  };

  const limit = tierLimits[userTier] || tierLimits.basic;

  const limiter = createRateLimiter({
    windowMs: limit.windowMs,
    max: limit.max,
    message: `Rate limit exceeded for ${userTier} tier.`
  });

  limiter(req, res, next);
};

/**
 * IP-based rate limiter for public endpoints
 */
export const publicLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 50,
  message: 'Too many requests from this IP. Please try again later.'
});

/**
 * Prevent brute force attacks on sensitive endpoints
 */
export const bruteForceLimiter = createRateLimiter({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 3,
  message: 'Too many failed attempts. Account temporarily locked.',
  skipSuccessfulRequests: true
});

/**
 * Custom rate limiter with Redis store for distributed systems
 */
export const createRedisRateLimiter = (redisClient: any) => {
  return (options: {
    windowMs: number;
    max: number;
    keyPrefix: string;
  }) => {
    return async (req: Request, res: Response, next: Function) => {
      const userId = (req as any).user?.id || req.ip;
      const key = `${options.keyPrefix}:${userId}`;

      try {
        const current = await redisClient.incr(key);

        if (current === 1) {
          await redisClient.expire(key, Math.round(options.windowMs / 1000));
        }

        if (current > options.max) {
          const ttl = await redisClient.ttl(key);
          res.setHeader('Retry-After', ttl);
          res.setHeader('X-RateLimit-Limit', options.max);
          res.setHeader('X-RateLimit-Remaining', Math.max(0, options.max - current));
          res.setHeader('X-RateLimit-Reset', new Date(Date.now() + ttl * 1000).toISOString());

          return res.status(429).json({
            success: false,
            error: 'Rate limit exceeded',
            retryAfter: ttl
          });
        }

        res.setHeader('X-RateLimit-Limit', options.max);
        res.setHeader('X-RateLimit-Remaining', options.max - current);
        next();
      } catch (error) {
        logger.error('Redis rate limiter error:', error);
        // Fall back to allowing the request if Redis fails
        next();
      }
    };
  };
};

/**
 * Sliding window rate limiter for more accurate rate limiting
 */
export class SlidingWindowLimiter {
  private requests: Map<string, number[]> = new Map();
  private windowMs: number;
  private maxRequests: number;

  constructor(windowMs: number, maxRequests: number) {
    this.windowMs = windowMs;
    this.maxRequests = maxRequests;

    // Clean up old entries periodically
    setInterval(() => this.cleanup(), windowMs);
  }

  check(identifier: string): boolean {
    const now = Date.now();
    const userRequests = this.requests.get(identifier) || [];

    // Remove old requests outside the window
    const validRequests = userRequests.filter(
      timestamp => now - timestamp < this.windowMs
    );

    if (validRequests.length >= this.maxRequests) {
      return false; // Rate limit exceeded
    }

    // Add current request
    validRequests.push(now);
    this.requests.set(identifier, validRequests);
    return true;
  }

  private cleanup(): void {
    const now = Date.now();
    for (const [key, timestamps] of this.requests.entries()) {
      const valid = timestamps.filter(t => now - t < this.windowMs);
      if (valid.length === 0) {
        this.requests.delete(key);
      } else {
        this.requests.set(key, valid);
      }
    }
  }
}

// Export a sliding window instance for critical operations
export const criticalOperationsLimiter = new SlidingWindowLimiter(
  60 * 1000, // 1 minute window
  5 // max 5 operations
);