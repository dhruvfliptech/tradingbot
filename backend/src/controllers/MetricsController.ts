import { Request, Response } from 'express';
import { PerformanceMetricsService } from '../services/metrics/PerformanceMetricsService';
import logger from '../utils/logger';
import { AuthenticatedRequest } from '../middleware/auth';

export class MetricsController {
  private metricsService: PerformanceMetricsService | null = null;

  setMetricsService(service: PerformanceMetricsService): void {
    this.metricsService = service;
  }

  /**
   * Get overall performance metrics
   */
  async getPerformanceMetrics(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { period = '30d', userId } = req.query;
      const targetUserId = userId as string || req.user?.id;

      if (!targetUserId) {
        res.status(400).json({
          success: false,
          error: 'User ID is required'
        });
        return;
      }

      if (!this.metricsService) {
        res.status(503).json({
          success: false,
          error: 'Metrics service not available'
        });
        return;
      }

      const metrics = await this.metricsService.getPerformanceMetrics(
        targetUserId,
        period as string
      );

      res.json({
        success: true,
        data: metrics,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get performance metrics:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch performance metrics'
      });
    }
  }

  /**
   * Get Sharpe ratio calculation
   */
  async getSharpeRatio(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { period = '30d', riskFreeRate = 0.02 } = req.query;
      const userId = req.user?.id;

      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
        return;
      }

      if (!this.metricsService) {
        res.status(503).json({
          success: false,
          error: 'Metrics service not available'
        });
        return;
      }

      const sharpeRatio = await this.metricsService.calculateSharpeRatio(
        userId,
        period as string,
        Number(riskFreeRate)
      );

      res.json({
        success: true,
        data: {
          sharpeRatio,
          period,
          riskFreeRate: Number(riskFreeRate),
          interpretation: this.interpretSharpeRatio(sharpeRatio)
        },
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to calculate Sharpe ratio:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to calculate Sharpe ratio'
      });
    }
  }

  /**
   * Get drawdown analysis
   */
  async getDrawdownAnalysis(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { period = '30d' } = req.query;
      const userId = req.user?.id;

      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
        return;
      }

      if (!this.metricsService) {
        res.status(503).json({
          success: false,
          error: 'Metrics service not available'
        });
        return;
      }

      const drawdown = await this.metricsService.getDrawdownAnalysis(
        userId,
        period as string
      );

      res.json({
        success: true,
        data: drawdown,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get drawdown analysis:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch drawdown analysis'
      });
    }
  }

  /**
   * Get P&L analysis
   */
  async getPnLAnalysis(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { period = '30d', groupBy = 'day' } = req.query;
      const userId = req.user?.id;

      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
        return;
      }

      if (!this.metricsService) {
        res.status(503).json({
          success: false,
          error: 'Metrics service not available'
        });
        return;
      }

      const pnl = await this.metricsService.getPnLAnalysis(
        userId,
        period as string,
        groupBy as 'day' | 'week' | 'month'
      );

      res.json({
        success: true,
        data: pnl,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get P&L analysis:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch P&L analysis'
      });
    }
  }

  /**
   * Get win rate analysis
   */
  async getWinRateAnalysis(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const { period = '30d', bySymbol = false } = req.query;
      const userId = req.user?.id;

      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
        return;
      }

      if (!this.metricsService) {
        res.status(503).json({
          success: false,
          error: 'Metrics service not available'
        });
        return;
      }

      const winRate = await this.metricsService.getWinRateAnalysis(
        userId,
        period as string,
        bySymbol === 'true'
      );

      res.json({
        success: true,
        data: winRate,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get win rate analysis:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch win rate analysis'
      });
    }
  }

  /**
   * Get risk metrics
   */
  async getRiskMetrics(req: AuthenticatedRequest, res: Response): Promise<void> {
    try {
      const userId = req.user?.id;

      if (!userId) {
        res.status(401).json({
          success: false,
          error: 'Authentication required'
        });
        return;
      }

      if (!this.metricsService) {
        res.status(503).json({
          success: false,
          error: 'Metrics service not available'
        });
        return;
      }

      const riskMetrics = await this.metricsService.getRiskMetrics(userId);

      res.json({
        success: true,
        data: riskMetrics,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get risk metrics:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch risk metrics'
      });
    }
  }

  /**
   * Helper to interpret Sharpe ratio
   */
  private interpretSharpeRatio(ratio: number): string {
    if (ratio < 0) return 'Negative - Strategy is losing money';
    if (ratio < 0.5) return 'Poor - Low risk-adjusted returns';
    if (ratio < 1) return 'Suboptimal - Below average risk-adjusted returns';
    if (ratio < 1.5) return 'Good - Decent risk-adjusted returns';
    if (ratio < 2) return 'Very Good - Strong risk-adjusted returns';
    if (ratio < 3) return 'Excellent - Outstanding risk-adjusted returns';
    return 'Exceptional - World-class risk-adjusted returns';
  }
}