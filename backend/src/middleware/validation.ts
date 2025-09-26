/**
 * Validation Middleware
 * Provides comprehensive input validation for API endpoints
 */

import { Request, Response, NextFunction } from 'express';
import { body, param, query, validationResult } from 'express-validator';
import logger from '../utils/logger';

/**
 * Handle validation errors
 */
export const handleValidationErrors = (req: Request, res: Response, next: NextFunction) => {
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    logger.warn('Validation error:', {
      path: req.path,
      errors: errors.array()
    });

    return res.status(400).json({
      success: false,
      error: 'Validation failed',
      details: errors.array().map(err => ({
        field: err.param,
        message: err.msg
      }))
    });
  }
  next();
};

/**
 * Trading Bot validations
 */
export const validateStartBot = [
  body('watchlist')
    .optional()
    .isArray()
    .withMessage('Watchlist must be an array')
    .custom((value) => {
      if (value && value.length > 20) {
        throw new Error('Watchlist cannot contain more than 20 symbols');
      }
      return true;
    }),
  body('cycleIntervalMs')
    .optional()
    .isInt({ min: 10000, max: 3600000 })
    .withMessage('Cycle interval must be between 10 seconds and 1 hour'),
  body('cooldownMinutes')
    .optional()
    .isInt({ min: 1, max: 60 })
    .withMessage('Cooldown must be between 1 and 60 minutes'),
  body('maxOpenPositions')
    .optional()
    .isInt({ min: 1, max: 50 })
    .withMessage('Max open positions must be between 1 and 50'),
  body('riskBudgetUsd')
    .optional()
    .isFloat({ min: 10, max: 1000000 })
    .withMessage('Risk budget must be between $10 and $1,000,000'),
  body('confidenceThreshold')
    .optional()
    .isFloat({ min: 0.1, max: 1.0 })
    .withMessage('Confidence threshold must be between 0.1 and 1.0'),
  body('settings.stopLossPercent')
    .optional()
    .isFloat({ min: 0.1, max: 50 })
    .withMessage('Stop loss must be between 0.1% and 50%'),
  body('settings.takeProfitPercent')
    .optional()
    .isFloat({ min: 0.1, max: 100 })
    .withMessage('Take profit must be between 0.1% and 100%'),
  handleValidationErrors
];

export const validateStopBot = [
  body('reason')
    .optional()
    .isString()
    .trim()
    .isLength({ max: 500 })
    .withMessage('Reason must be less than 500 characters'),
  handleValidationErrors
];

export const validateUpdateConfig = [
  body('watchlist')
    .optional()
    .isArray()
    .withMessage('Watchlist must be an array'),
  body('cycleIntervalMs')
    .optional()
    .isInt({ min: 10000, max: 3600000 })
    .withMessage('Cycle interval must be between 10 seconds and 1 hour'),
  body('confidenceThreshold')
    .optional()
    .isFloat({ min: 0.1, max: 1.0 })
    .withMessage('Confidence threshold must be between 0.1 and 1.0'),
  handleValidationErrors
];

/**
 * Order validations
 */
export const validatePlaceOrder = [
  body('symbol')
    .notEmpty()
    .withMessage('Symbol is required')
    .isString()
    .trim()
    .toUpperCase()
    .matches(/^[A-Z]{2,10}(USD|USDT)?$/)
    .withMessage('Invalid symbol format'),
  body('qty')
    .notEmpty()
    .withMessage('Quantity is required')
    .isFloat({ min: 0.00001, max: 1000000 })
    .withMessage('Quantity must be between 0.00001 and 1,000,000'),
  body('side')
    .notEmpty()
    .withMessage('Side is required')
    .isIn(['buy', 'sell', 'BUY', 'SELL'])
    .withMessage('Side must be buy or sell')
    .customSanitizer((value) => value.toLowerCase()),
  body('order_type')
    .optional()
    .isIn(['market', 'limit'])
    .withMessage('Order type must be market or limit'),
  body('limit_price')
    .optional()
    .isFloat({ min: 0.01, max: 10000000 })
    .withMessage('Limit price must be between 0.01 and 10,000,000')
    .custom((value, { req }) => {
      if (req.body.order_type === 'limit' && !value) {
        throw new Error('Limit price is required for limit orders');
      }
      return true;
    }),
  body('time_in_force')
    .optional()
    .isIn(['day', 'gtc', 'ioc', 'fok'])
    .withMessage('Invalid time in force value'),
  body('client_order_id')
    .optional()
    .isString()
    .isLength({ max: 100 })
    .withMessage('Client order ID must be less than 100 characters'),
  handleValidationErrors
];

export const validateCancelOrder = [
  param('orderId')
    .notEmpty()
    .withMessage('Order ID is required')
    .isString(),
  handleValidationErrors
];

/**
 * Broker validations
 */
export const validateSetActiveBroker = [
  body('broker')
    .notEmpty()
    .withMessage('Broker is required')
    .isIn(['alpaca', 'binance'])
    .withMessage('Broker must be alpaca or binance'),
  handleValidationErrors
];

export const validateInitializeBroker = [
  body('broker')
    .notEmpty()
    .withMessage('Broker is required')
    .isIn(['alpaca', 'binance'])
    .withMessage('Broker must be alpaca or binance'),
  body('apiKey')
    .notEmpty()
    .withMessage('API key is required')
    .isString()
    .isLength({ min: 10, max: 200 })
    .withMessage('Invalid API key format'),
  body('secretKey')
    .notEmpty()
    .withMessage('Secret key is required')
    .isString()
    .isLength({ min: 10, max: 200 })
    .withMessage('Invalid secret key format'),
  body('baseUrl')
    .optional()
    .isURL({ protocols: ['http', 'https'], require_protocol: true })
    .withMessage('Invalid base URL format'),
  handleValidationErrors
];

/**
 * Market data validations
 */
export const validateGetMarketData = [
  query('symbols')
    .notEmpty()
    .withMessage('Symbols parameter is required')
    .isString()
    .custom((value) => {
      const symbols = value.split(',');
      if (symbols.length > 50) {
        throw new Error('Cannot request more than 50 symbols at once');
      }
      return true;
    }),
  query('broker')
    .optional()
    .isIn(['alpaca', 'binance'])
    .withMessage('Broker must be alpaca or binance'),
  handleValidationErrors
];

/**
 * Performance validations
 */
export const validateGetPerformance = [
  query('period')
    .optional()
    .isIn(['hourly', 'daily', 'weekly', 'monthly'])
    .withMessage('Invalid period value'),
  handleValidationErrors
];

/**
 * Sanitize user input to prevent XSS
 */
export const sanitizeInput = (req: Request, res: Response, next: NextFunction) => {
  // Recursively sanitize all string values in body, query, and params
  const sanitizeObject = (obj: any): any => {
    if (typeof obj === 'string') {
      // Remove HTML tags and script content
      return obj.replace(/<script\b[^<]*(?:(?!<\/script>)<[^<]*)*<\/script>/gi, '')
                .replace(/<[^>]+>/g, '')
                .trim();
    } else if (Array.isArray(obj)) {
      return obj.map(sanitizeObject);
    } else if (obj && typeof obj === 'object') {
      const sanitized: any = {};
      for (const key in obj) {
        sanitized[key] = sanitizeObject(obj[key]);
      }
      return sanitized;
    }
    return obj;
  };

  if (req.body) req.body = sanitizeObject(req.body);
  if (req.query) req.query = sanitizeObject(req.query);
  if (req.params) req.params = sanitizeObject(req.params);

  next();
};

/**
 * Validate amount limits based on user tier
 */
export const validateAmountLimits = (req: Request, res: Response, next: NextFunction) => {
  const { qty, side } = req.body;
  const userTier = (req as any).user?.tier || 'basic';

  // Define limits per tier
  const limits: Record<string, number> = {
    basic: 10000,
    premium: 50000,
    professional: 500000
  };

  const maxAmount = limits[userTier] || limits.basic;

  // For now, just check quantity (in production, multiply by price)
  if (qty && Number(qty) > maxAmount) {
    return res.status(403).json({
      success: false,
      error: `Order quantity exceeds limit for ${userTier} tier (max: ${maxAmount})`
    });
  }

  next();
};

/**
 * Validate symbol whitelist
 */
export const validateSymbolWhitelist = (req: Request, res: Response, next: NextFunction) => {
  const { symbol } = req.body;

  // Define allowed symbols (can be loaded from database)
  const allowedSymbols = [
    'BTC', 'ETH', 'BNB', 'SOL', 'ADA', 'DOT', 'AVAX', 'MATIC', 'LINK', 'UNI',
    'BTCUSD', 'ETHUSD', 'BNBUSD', 'SOLUSD', 'ADAUSD',
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'ADAUSDT'
  ];

  if (symbol) {
    const baseSymbol = symbol.replace(/USD[T]?$/, '');
    const isAllowed = allowedSymbols.some(allowed =>
      symbol === allowed || baseSymbol === allowed
    );

    if (!isAllowed) {
      return res.status(403).json({
        success: false,
        error: `Symbol ${symbol} is not allowed for trading`
      });
    }
  }

  next();
};