import { Request, Response } from 'express';
import monitoringService from '../services/monitoring/UnifiedMonitoringService';
import logger from '../utils/logger';

export class MonitoringController {
  /**
   * Get dashboard data
   */
  async getDashboard(req: Request, res: Response): Promise<void> {
    try {
      const dashboard = monitoringService.getDashboardData();

      res.json({
        success: true,
        data: dashboard,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get dashboard data:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch dashboard data'
      });
    }
  }

  /**
   * Get metrics
   */
  async getMetrics(req: Request, res: Response): Promise<void> {
    try {
      const { name, period } = req.query;

      const metrics = monitoringService.getMetrics(
        name as string,
        period ? parseInt(period as string) : undefined
      );

      res.json({
        success: true,
        data: metrics,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get metrics:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch metrics'
      });
    }
  }

  /**
   * Get alerts
   */
  async getAlerts(req: Request, res: Response): Promise<void> {
    try {
      const { severity, limit = 100 } = req.query;

      const alerts = monitoringService.getAlerts(
        severity as any,
        parseInt(limit as string)
      );

      res.json({
        success: true,
        data: alerts,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get alerts:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch alerts'
      });
    }
  }

  /**
   * Get health status
   */
  async getHealth(req: Request, res: Response): Promise<void> {
    try {
      const health = monitoringService.getHealthStatus();

      const statusCode = health.overall === 'healthy' ? 200 :
                        health.overall === 'degraded' ? 200 : 503;

      res.status(statusCode).json({
        success: true,
        data: health,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get health status:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch health status'
      });
    }
  }

  /**
   * Create a test alert
   */
  async createTestAlert(req: Request, res: Response): Promise<void> {
    try {
      const { severity = 'info', title, message } = req.body;

      monitoringService.createAlert(
        severity,
        title || 'Test Alert',
        message || 'This is a test alert',
        'manual',
        { user: (req as any).user?.id }
      );

      res.json({
        success: true,
        message: 'Test alert created',
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to create test alert:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to create test alert'
      });
    }
  }
}