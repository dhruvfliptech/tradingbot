/**
 * Authentication Middleware
 * Handles JWT verification and user authentication
 */

import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import logger from '../utils/logger';
import { secretsManager } from '../config/secrets';

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    email: string;
    tier?: string;
    permissions?: string[];
  };
}

/**
 * Verify JWT token and attach user to request
 */
export const authenticate = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader) {
      res.status(401).json({
        success: false,
        error: 'No authorization token provided'
      });
      return;
    }

    const token = authHeader.startsWith('Bearer ')
      ? authHeader.slice(7)
      : authHeader;

    if (!token) {
      res.status(401).json({
        success: false,
        error: 'Invalid authorization format'
      });
      return;
    }

    // Get JWT secret from secrets manager
    const jwtConfig = secretsManager.getJWTConfig();
    const secret = jwtConfig?.secret || process.env.JWT_SECRET || 'default-secret';

    // Verify token
    const decoded = jwt.verify(token, secret) as any;

    // Extract user information
    req.user = {
      id: decoded.userId || decoded.sub || decoded.id,
      email: decoded.email,
      tier: decoded.tier || 'basic',
      permissions: decoded.permissions || []
    };

    // Log authentication
    logger.debug('User authenticated:', {
      userId: req.user.id,
      path: req.path
    });

    next();
  } catch (error) {
    if (error instanceof jwt.TokenExpiredError) {
      res.status(401).json({
        success: false,
        error: 'Token has expired'
      });
    } else if (error instanceof jwt.JsonWebTokenError) {
      res.status(401).json({
        success: false,
        error: 'Invalid token'
      });
    } else {
      logger.error('Authentication error:', error);
      res.status(500).json({
        success: false,
        error: 'Authentication failed'
      });
    }
  }
};

/**
 * Optional authentication - doesn't fail if no token
 */
export const optionalAuth = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;

    if (!authHeader) {
      // No token, continue without user
      req.user = undefined;
      return next();
    }

    const token = authHeader.startsWith('Bearer ')
      ? authHeader.slice(7)
      : authHeader;

    if (token) {
      const jwtConfig = secretsManager.getJWTConfig();
      const secret = jwtConfig?.secret || process.env.JWT_SECRET || 'default-secret';

      try {
        const decoded = jwt.verify(token, secret) as any;
        req.user = {
          id: decoded.userId || decoded.sub || decoded.id,
          email: decoded.email,
          tier: decoded.tier || 'basic',
          permissions: decoded.permissions || []
        };
      } catch {
        // Invalid token, continue without user
        req.user = undefined;
      }
    }

    next();
  } catch (error) {
    // Continue without authentication
    req.user = undefined;
    next();
  }
};

/**
 * Require specific user tier
 */
export const requireTier = (minTier: 'basic' | 'premium' | 'professional') => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }

    const tierLevels = {
      basic: 1,
      premium: 2,
      professional: 3
    };

    const userTierLevel = tierLevels[req.user.tier as keyof typeof tierLevels] || 1;
    const requiredTierLevel = tierLevels[minTier];

    if (userTierLevel < requiredTierLevel) {
      return res.status(403).json({
        success: false,
        error: `This feature requires ${minTier} tier or higher`
      });
    }

    next();
  };
};

/**
 * Require specific permissions
 */
export const requirePermissions = (...permissions: string[]) => {
  return (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
    if (!req.user) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }

    const userPermissions = req.user.permissions || [];
    const hasAllPermissions = permissions.every(p => userPermissions.includes(p));

    if (!hasAllPermissions) {
      logger.warn('Permission denied:', {
        userId: req.user.id,
        required: permissions,
        actual: userPermissions
      });

      return res.status(403).json({
        success: false,
        error: 'Insufficient permissions'
      });
    }

    next();
  };
};

/**
 * Verify API key for service-to-service communication
 */
export const verifyApiKey = (req: Request, res: Response, next: NextFunction) => {
  const apiKey = req.headers['x-api-key'] as string;

  if (!apiKey) {
    return res.status(401).json({
      success: false,
      error: 'API key required'
    });
  }

  // In production, verify against stored API keys
  const validApiKeys = process.env.VALID_API_KEYS?.split(',') || [];

  if (!validApiKeys.includes(apiKey)) {
    logger.warn('Invalid API key attempt:', {
      ip: req.ip,
      path: req.path
    });

    return res.status(401).json({
      success: false,
      error: 'Invalid API key'
    });
  }

  next();
};

/**
 * Demo mode check - restrict certain operations in demo
 */
export const restrictInDemo = (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  const isDemoUser = req.user?.email?.includes('demo') || req.user?.id === 'demo-user';

  if (isDemoUser) {
    return res.status(403).json({
      success: false,
      error: 'This operation is not available in demo mode'
    });
  }

  next();
};

/**
 * Log all authenticated requests
 */
export const auditLog = (req: AuthenticatedRequest, res: Response, next: NextFunction) => {
  if (req.user) {
    logger.info('Authenticated request:', {
      userId: req.user.id,
      method: req.method,
      path: req.path,
      ip: req.ip,
      userAgent: req.headers['user-agent']
    });
  }
  next();
};