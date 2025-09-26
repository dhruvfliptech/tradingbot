import { Request, Response } from 'express';
import { DataAggregatorService } from '../services/data-aggregation/DataAggregatorService';
import logger from '../utils/logger';

export class DataAggregatorController {
  private dataAggregator: DataAggregatorService | null = null;

  setDataAggregator(aggregator: DataAggregatorService): void {
    this.dataAggregator = aggregator;
  }

  /**
   * Get aggregated market data from multiple sources
   */
  async getAggregatedMarketData(req: Request, res: Response): Promise<void> {
    try {
      const { symbols } = req.query;

      if (!symbols || typeof symbols !== 'string') {
        res.status(400).json({
          success: false,
          error: 'Symbols parameter is required'
        });
        return;
      }

      if (!this.dataAggregator) {
        res.status(503).json({
          success: false,
          error: 'Data aggregator service not available'
        });
        return;
      }

      const symbolList = symbols.split(',');
      const aggregatedData = await this.dataAggregator.getAggregatedMarketData(symbolList);

      res.json({
        success: true,
        data: aggregatedData,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get aggregated market data:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch aggregated market data'
      });
    }
  }

  /**
   * Get funding rates from multiple exchanges
   */
  async getFundingRates(req: Request, res: Response): Promise<void> {
    try {
      const { symbol } = req.query;

      if (!this.dataAggregator) {
        res.status(503).json({
          success: false,
          error: 'Data aggregator service not available'
        });
        return;
      }

      const fundingRates = await this.dataAggregator.getFundingRates(
        symbol as string || 'BTCUSDT'
      );

      res.json({
        success: true,
        data: fundingRates,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get funding rates:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch funding rates'
      });
    }
  }

  /**
   * Get whale alerts and large transactions
   */
  async getWhaleAlerts(req: Request, res: Response): Promise<void> {
    try {
      const { limit = 10, minValue = 1000000 } = req.query;

      if (!this.dataAggregator) {
        res.status(503).json({
          success: false,
          error: 'Data aggregator service not available'
        });
        return;
      }

      const alerts = await this.dataAggregator.getWhaleAlerts({
        limit: Number(limit),
        minValueUsd: Number(minValue)
      });

      res.json({
        success: true,
        data: alerts,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get whale alerts:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch whale alerts'
      });
    }
  }

  /**
   * Get on-chain data for specific addresses
   */
  async getOnChainData(req: Request, res: Response): Promise<void> {
    try {
      const { address, chain = 'ethereum' } = req.query;

      if (!address || typeof address !== 'string') {
        res.status(400).json({
          success: false,
          error: 'Address parameter is required'
        });
        return;
      }

      if (!this.dataAggregator) {
        res.status(503).json({
          success: false,
          error: 'Data aggregator service not available'
        });
        return;
      }

      const onChainData = await this.dataAggregator.getOnChainData(
        address,
        chain as string
      );

      res.json({
        success: true,
        data: onChainData,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get on-chain data:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch on-chain data'
      });
    }
  }

  /**
   * Get sentiment analysis from news and social media
   */
  async getSentimentAnalysis(req: Request, res: Response): Promise<void> {
    try {
      const { symbol, sources = 'all' } = req.query;

      if (!symbol || typeof symbol !== 'string') {
        res.status(400).json({
          success: false,
          error: 'Symbol parameter is required'
        });
        return;
      }

      if (!this.dataAggregator) {
        res.status(503).json({
          success: false,
          error: 'Data aggregator service not available'
        });
        return;
      }

      const sentiment = await this.dataAggregator.getSentimentData(
        symbol,
        sources as string
      );

      res.json({
        success: true,
        data: sentiment,
        timestamp: new Date()
      });
    } catch (error) {
      logger.error('Failed to get sentiment analysis:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to fetch sentiment analysis'
      });
    }
  }
}