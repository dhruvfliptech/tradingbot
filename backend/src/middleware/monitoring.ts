/**
 * Monitoring Middleware
 * Tracks API metrics and performance
 */

import { Request, Response, NextFunction } from 'express';
import monitoringService from '../services/monitoring/UnifiedMonitoringService';

/**
 * Track API request metrics
 */
export const trackApiMetrics = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();

  // Track request
  monitoringService.recordMetric('api.request.total', 1, 'counter', {
    method: req.method,
    path: req.path
  });

  // Override res.end to capture response metrics
  const originalEnd = res.end;
  res.end = function(...args: any[]) {
    const responseTime = Date.now() - startTime;

    // Record metrics
    monitoringService.recordApiMetrics({
      endpoint: req.path,
      method: req.method,
      statusCode: res.statusCode,
      responseTime,
      userId: (req as any).user?.id
    });

    // Call original end
    return originalEnd.apply(res, args);
  };

  next();
};

/**
 * Track trading operations
 */
export const trackTradingMetrics = (operation: string) => {
  return (req: Request, res: Response, next: NextFunction) => {
    monitoringService.recordMetric(`trading.operation.${operation}`, 1, 'counter', {
      userId: (req as any).user?.id
    });
    next();
  };
};

/**
 * Health check for specific services
 */
export const healthCheck = (service: string) => {
  return async (req: Request, res: Response, next: NextFunction) => {
    const startTime = Date.now();

    try {
      // Perform health check (customize per service)
      monitoringService.recordHealthCheck(
        service,
        'healthy',
        Date.now() - startTime
      );
      next();
    } catch (error) {
      monitoringService.recordHealthCheck(
        service,
        'unhealthy',
        Date.now() - startTime,
        { error: (error as Error).message }
      );
      next();
    }
  };
};