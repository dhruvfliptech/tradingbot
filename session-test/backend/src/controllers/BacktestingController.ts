import { Request, Response, NextFunction } from 'express';
import { ComposerService, BacktestConfig, StrategyDefinition } from '../services/composer/ComposerService';
import { StrategyValidationService } from '../services/validation/StrategyValidationService';
import { AdaptiveThreshold } from '../ml-service/adaptive_threshold';
import { DatabaseService } from '../services/database/DatabaseService';
import logger from '../utils/logger';

export class BacktestingController {
  private composerService: ComposerService;
  private validationService: StrategyValidationService;
  private databaseService: DatabaseService;

  constructor() {
    this.composerService = new ComposerService();
    this.validationService = new StrategyValidationService(this.composerService);
    this.databaseService = new DatabaseService();
    this.initializeServices();
  }

  private async initializeServices(): Promise<void> {
    try {
      await this.composerService.connect();
      logger.info('Composer MCP service connected');
    } catch (error) {
      logger.error('Failed to connect to Composer MCP service:', error);
    }
  }

  /**
   * POST /api/v1/backtesting/strategies
   * Create a new strategy definition
   */
  async createStrategy(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const strategyData = req.body;

      // Validate strategy definition with Composer MCP
      const validation = await this.composerService.validateStrategy(strategyData);
      
      if (!validation.isValid) {
        return res.status(400).json({
          success: false,
          message: 'Strategy validation failed',
          errors: validation.errors,
          warnings: validation.warnings
        });
      }

      // Save strategy to database
      const strategy = await this.databaseService.createStrategy(userId, {
        name: strategyData.name,
        description: strategyData.description,
        strategyType: strategyData.strategy_type || 'custom',
        version: strategyData.version || '1.0.0',
        parameters: strategyData.parameters,
        entryRules: strategyData.entry_rules,
        exitRules: strategyData.exit_rules,
        riskManagement: strategyData.risk_management,
        isValidated: validation.isValid,
        validationScore: validation.isValid ? 0.8 : 0
      });

      res.status(201).json({
        success: true,
        data: {
          id: strategy.id,
          name: strategy.name,
          validation: validation
        }
      });

    } catch (error) {
      logger.error('Error creating strategy:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/backtesting/strategies
   * Get user's strategy definitions
   */
  async getStrategies(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { active_only = false, validated_only = false } = req.query;

      const strategies = await this.databaseService.getUserStrategies(userId, {
        activeOnly: active_only === 'true',
        validatedOnly: validated_only === 'true'
      });

      res.json({
        success: true,
        data: strategies
      });

    } catch (error) {
      logger.error('Error fetching strategies:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/backtesting/strategies/:strategyId
   * Get specific strategy definition
   */
  async getStrategy(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { strategyId } = req.params;

      const strategy = await this.databaseService.getStrategy(userId, strategyId);
      
      if (!strategy) {
        return res.status(404).json({
          success: false,
          message: 'Strategy not found'
        });
      }

      res.json({
        success: true,
        data: strategy
      });

    } catch (error) {
      logger.error('Error fetching strategy:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/backtesting/strategies/:strategyId/validate
   * Validate a strategy with comprehensive testing
   */
  async validateStrategy(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { strategyId } = req.params;
      const { symbols, period } = req.body;

      const strategy = await this.databaseService.getStrategy(userId, strategyId);
      
      if (!strategy) {
        return res.status(404).json({
          success: false,
          message: 'Strategy not found'
        });
      }

      // Run comprehensive validation
      const report = await this.validationService.runComprehensiveValidation(
        strategy as StrategyDefinition,
        symbols || ['BTC/USD', 'ETH/USD'],
        period || {
          start: new Date(Date.now() - 90 * 24 * 60 * 60 * 1000).toISOString(),
          end: new Date().toISOString()
        }
      );

      // Save validation results
      await this.databaseService.saveValidationResults(userId, strategyId, report);

      // Update strategy validation status
      await this.databaseService.updateStrategyValidation(
        userId,
        strategyId,
        report.status === 'passed',
        report.overallScore / 100
      );

      res.json({
        success: true,
        data: report
      });

    } catch (error) {
      logger.error('Error validating strategy:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/backtesting/run
   * Run a backtest
   */
  async runBacktest(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const config: BacktestConfig = req.body;

      // Validate strategy exists
      const strategy = await this.databaseService.getStrategy(userId, config.strategyId);
      if (!strategy) {
        return res.status(404).json({
          success: false,
          message: 'Strategy not found'
        });
      }

      // Save backtest configuration
      const backtestConfig = await this.databaseService.createBacktestConfig(userId, {
        strategyId: config.strategyId,
        name: `Backtest ${new Date().toISOString()}`,
        symbols: config.symbols,
        startDate: new Date(config.startDate),
        endDate: new Date(config.endDate),
        initialCapital: config.initialCapital,
        strategyParameters: config.parameters,
        riskSettings: config.riskSettings
      });

      // Run backtest with Composer MCP
      const externalBacktestId = await this.composerService.runBacktest(config);

      // Create backtest result record
      const backtestResult = await this.databaseService.createBacktestResult(userId, {
        configId: backtestConfig.id,
        strategyId: config.strategyId,
        externalBacktestId,
        status: 'running'
      });

      // Set up event listeners for backtest progress
      this.composerService.on('backtest_progress', (data) => {
        if (data.backtestId === externalBacktestId) {
          this.databaseService.updateBacktestProgress(
            backtestResult.id,
            data.progress,
            data.performance
          );
        }
      });

      this.composerService.on('backtest_completed', async (result) => {
        if (result.id === externalBacktestId) {
          await this.handleBacktestCompletion(userId, backtestResult.id, result);
        }
      });

      this.composerService.on('backtest_error', async (result) => {
        if (result.id === externalBacktestId) {
          await this.databaseService.updateBacktestResult(backtestResult.id, {
            status: 'failed',
            errorMessage: result.error,
            completedAt: new Date()
          });
        }
      });

      res.status(202).json({
        success: true,
        data: {
          backtestId: backtestResult.id,
          externalId: externalBacktestId,
          status: 'running',
          message: 'Backtest started successfully'
        }
      });

    } catch (error) {
      logger.error('Error starting backtest:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/backtesting/results
   * Get backtest results for user
   */
  async getBacktestResults(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { 
        strategy_id, 
        status, 
        limit = 20, 
        offset = 0,
        sort_by = 'created_at',
        sort_order = 'desc'
      } = req.query;

      const results = await this.databaseService.getBacktestResults(userId, {
        strategyId: strategy_id as string,
        status: status as string,
        limit: Number(limit),
        offset: Number(offset),
        sortBy: sort_by as string,
        sortOrder: sort_order as 'asc' | 'desc'
      });

      res.json({
        success: true,
        data: results
      });

    } catch (error) {
      logger.error('Error fetching backtest results:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/backtesting/results/:backtestId
   * Get specific backtest result with detailed metrics
   */
  async getBacktestResult(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { backtestId } = req.params;
      const { include_trades = false } = req.query;

      const result = await this.databaseService.getBacktestResult(
        userId, 
        backtestId,
        { includeTrades: include_trades === 'true' }
      );

      if (!result) {
        return res.status(404).json({
          success: false,
          message: 'Backtest result not found'
        });
      }

      res.json({
        success: true,
        data: result
      });

    } catch (error) {
      logger.error('Error fetching backtest result:', error);
      next(error);
    }
  }

  /**
   * DELETE /api/v1/backtesting/results/:backtestId
   * Cancel a running backtest
   */
  async cancelBacktest(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { backtestId } = req.params;

      const result = await this.databaseService.getBacktestResult(userId, backtestId);
      
      if (!result) {
        return res.status(404).json({
          success: false,
          message: 'Backtest not found'
        });
      }

      if (result.status !== 'running') {
        return res.status(400).json({
          success: false,
          message: 'Can only cancel running backtests'
        });
      }

      // Cancel with Composer MCP
      if (result.externalBacktestId) {
        await this.composerService.cancelBacktest(result.externalBacktestId);
      }

      // Update database
      await this.databaseService.updateBacktestResult(backtestId, {
        status: 'cancelled',
        completedAt: new Date()
      });

      res.json({
        success: true,
        message: 'Backtest cancelled successfully'
      });

    } catch (error) {
      logger.error('Error cancelling backtest:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/backtesting/optimize
   * Run strategy optimization
   */
  async optimizeStrategy(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const {
        strategy_id,
        config,
        optimization_params
      } = req.body;

      const strategy = await this.databaseService.getStrategy(userId, strategy_id);
      if (!strategy) {
        return res.status(404).json({
          success: false,
          message: 'Strategy not found'
        });
      }

      // Run optimization with Composer MCP
      const optimizationResult = await this.composerService.optimizeStrategy(
        strategy_id,
        config,
        optimization_params
      );

      // Save optimization results
      const optimization = await this.databaseService.createOptimization(userId, {
        strategyId: strategy_id,
        externalOptimizationId: optimizationResult.optimizationId,
        optimizationMethod: optimization_params.method,
        objectiveFunction: optimization_params.objectiveFunction,
        parametersToOptimize: optimization_params.parameters,
        iterations: optimization_params.iterations,
        bestParameters: optimizationResult.bestParameters,
        bestPerformance: optimizationResult.bestPerformance,
        optimizationResults: optimizationResult.results,
        status: 'completed'
      });

      res.json({
        success: true,
        data: {
          optimizationId: optimization.id,
          bestParameters: optimizationResult.bestParameters,
          bestPerformance: optimizationResult.bestPerformance,
          totalIterations: optimizationResult.results.length
        }
      });

    } catch (error) {
      logger.error('Error optimizing strategy:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/backtesting/historical-data
   * Get historical market data via Composer MCP
   */
  async getHistoricalData(req: Request, res: Response, next: NextFunction) {
    try {
      const {
        symbols,
        start_date,
        end_date,
        timeframe = '1h'
      } = req.query;

      if (!symbols || !start_date || !end_date) {
        return res.status(400).json({
          success: false,
          message: 'symbols, start_date, and end_date are required'
        });
      }

      const symbolsArray = Array.isArray(symbols) ? symbols : [symbols];
      
      const data = await this.composerService.getHistoricalData(
        symbolsArray as string[],
        start_date as string,
        end_date as string,
        timeframe as any
      );

      res.json({
        success: true,
        data: data
      });

    } catch (error) {
      logger.error('Error fetching historical data:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/backtesting/market-regimes
   * Get market regime analysis
   */
  async getMarketRegimes(req: Request, res: Response, next: NextFunction) {
    try {
      const {
        symbols,
        start_date,
        end_date
      } = req.query;

      if (!symbols || !start_date || !end_date) {
        return res.status(400).json({
          success: false,
          message: 'symbols, start_date, and end_date are required'
        });
      }

      const symbolsArray = Array.isArray(symbols) ? symbols : [symbols];
      
      const regimes = await this.composerService.getMarketRegimes(
        symbolsArray as string[],
        start_date as string,
        end_date as string
      );

      res.json({
        success: true,
        data: regimes
      });

    } catch (error) {
      logger.error('Error fetching market regimes:', error);
      next(error);
    }
  }

  /**
   * POST /api/v1/backtesting/adaptive-threshold/train
   * Train adaptive threshold with backtest data
   */
  async trainAdaptiveThreshold(req: Request, res: Response, next: NextFunction) {
    try {
      const userId = req.user?.id;
      const { strategy_id, backtest_results } = req.body;

      if (!backtest_results || backtest_results.length === 0) {
        return res.status(400).json({
          success: false,
          message: 'Backtest results are required for training'
        });
      }

      // Get strategy
      const strategy = await this.databaseService.getStrategy(userId, strategy_id);
      if (!strategy) {
        return res.status(404).json({
          success: false,
          message: 'Strategy not found'
        });
      }

      // Prepare training data from backtest results
      const trainingData = await this.prepareAdaptiveThresholdTrainingData(
        userId,
        strategy_id,
        backtest_results
      );

      // Save training data
      await this.databaseService.saveAdaptiveThresholdTraining(userId, trainingData);

      // Initialize/update adaptive threshold for this user and strategy
      const adaptiveThreshold = new AdaptiveThreshold(userId, strategy_id);
      const thresholdUpdates = await adaptiveThreshold.adapt_thresholds();

      res.json({
        success: true,
        data: {
          trainingDataPoints: trainingData.length,
          thresholdUpdates,
          currentThresholds: adaptiveThreshold.get_current_thresholds()
        }
      });

    } catch (error) {
      logger.error('Error training adaptive threshold:', error);
      next(error);
    }
  }

  /**
   * GET /api/v1/backtesting/available-strategies
   * Get available strategy templates from Composer MCP
   */
  async getAvailableStrategies(req: Request, res: Response, next: NextFunction) {
    try {
      const strategies = await this.composerService.getAvailableStrategies();

      res.json({
        success: true,
        data: strategies
      });

    } catch (error) {
      logger.error('Error fetching available strategies:', error);
      next(error);
    }
  }

  // Private helper methods

  private async handleBacktestCompletion(userId: string, backtestId: string, result: any): Promise<void> {
    try {
      // Update backtest result with performance metrics
      await this.databaseService.updateBacktestResult(backtestId, {
        status: 'completed',
        totalReturn: result.performance.totalReturn,
        sharpeRatio: result.performance.sharpeRatio,
        sortinoRatio: result.performance.sortino,
        calmarRatio: result.performance.calmar,
        maxDrawdown: result.performance.maxDrawdown,
        winRate: result.performance.winRate,
        profitFactor: result.performance.profitFactor,
        volatility: result.performance.volatility,
        beta: result.riskAnalysis?.beta,
        alpha: result.riskAnalysis?.alpha,
        totalTrades: result.metrics.totalTrades,
        winningTrades: result.metrics.winningTrades,
        losingTrades: result.metrics.losingTrades,
        avgWin: result.metrics.avgWin,
        avgLoss: result.metrics.avgLoss,
        largestWin: result.metrics.largestWin,
        largestLoss: result.metrics.largestLoss,
        consecutiveWins: result.metrics.consecutiveWins,
        consecutiveLosses: result.metrics.consecutiveLosses,
        totalCommissions: result.metrics.totalCommissions,
        var95: result.riskAnalysis?.var95,
        var99: result.riskAnalysis?.var99,
        expectedShortfall: result.riskAnalysis?.expectedShortfall,
        trackingError: result.riskAnalysis?.trackingError,
        informationRatio: result.riskAnalysis?.informationRatio,
        downsideDeviation: result.riskAnalysis?.downsideDeviation,
        detailedMetrics: result,
        completedAt: new Date()
      });

      // Save individual trades
      if (result.trades && result.trades.length > 0) {
        await this.databaseService.saveBacktestTrades(userId, backtestId, result.trades);
      }

      logger.info(`Backtest ${backtestId} completed successfully`);

    } catch (error) {
      logger.error('Error handling backtest completion:', error);
    }
  }

  private async prepareAdaptiveThresholdTrainingData(
    userId: string,
    strategyId: string,
    backtestResults: any[]
  ): Promise<any[]> {
    const trainingData = [];

    for (const result of backtestResults) {
      // Extract features and targets for training
      const dataPoint = {
        userId,
        strategyId,
        backtestResultId: result.id,
        trainingPeriodStart: new Date(result.started_at),
        trainingPeriodEnd: new Date(result.completed_at),
        marketConditions: {
          volatility: result.volatility || 0,
          volume: result.avg_volume || 0,
          trend: result.trend_direction || 'sideways'
        },
        strategyParameters: result.strategy_parameters || {},
        thresholdValues: result.threshold_values || {},
        performanceScore: this.calculatePerformanceScore(result),
        sharpeRatio: result.sharpe_ratio || 0,
        maxDrawdown: result.max_drawdown || 0,
        winRate: result.win_rate || 0,
        dataQualityScore: 0.8, // Default quality score
        regimeType: result.regime_type || 'unknown',
        volatilityBucket: this.getVolatilityBucket(result.volatility || 0)
      };

      trainingData.push(dataPoint);
    }

    return trainingData;
  }

  private calculatePerformanceScore(result: any): number {
    // Weighted performance score (0-100)
    const sharpe = result.sharpe_ratio || 0;
    const winRate = result.win_rate || 0;
    const maxDrawdown = result.max_drawdown || 1;
    const profitFactor = result.profit_factor || 0;

    const score = (
      Math.min(sharpe / 2, 1) * 30 +
      winRate * 25 +
      Math.max(0, 1 - maxDrawdown) * 25 +
      Math.min(profitFactor / 3, 1) * 20
    );

    return Math.max(0, Math.min(100, score));
  }

  private getVolatilityBucket(volatility: number): string {
    if (volatility < 0.15) return 'low';
    if (volatility < 0.3) return 'medium';
    return 'high';
  }
}