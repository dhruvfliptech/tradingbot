import { EventEmitter } from 'events';
import { ComposerService, BacktestConfig, BacktestResult, StrategyDefinition } from '../composer/ComposerService';
import { DatabaseService } from '../database/DatabaseService';
import logger from '../../utils/logger';

export interface AdaptiveTrainingConfig {
  userId: string;
  strategyId?: string;
  symbols: string[];
  trainingPeriods: Array<{
    start: string;
    end: string;
    regime?: 'bull' | 'bear' | 'sideways' | 'volatile';
  }>;
  validationPeriods: Array<{
    start: string;
    end: string;
  }>;
  thresholdParameters: string[];
  optimizationObjective: 'sharpe' | 'return' | 'profit_factor' | 'sortino';
  maxIterations: number;
}

export interface AdaptiveTrainingResult {
  trainingId: string;
  bestThresholds: Record<string, number>;
  performanceImprovement: number;
  validationMetrics: {
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    profitFactor: number;
  };
  regimeSpecificThresholds: Record<string, Record<string, number>>;
  generalizationScore: number;
  robustnessScore: number;
}

export interface ThresholdOptimizationRun {
  parameters: Record<string, number>;
  backtestResult: BacktestResult;
  performanceScore: number;
  regimeType?: string;
}

export class ComposerAdaptiveIntegration extends EventEmitter {
  private composerService: ComposerService;
  private databaseService: DatabaseService;
  private activeTrainingJobs: Map<string, AdaptiveTrainingConfig> = new Map();

  constructor(composerService: ComposerService) {
    super();
    this.composerService = composerService;
    this.databaseService = new DatabaseService();
  }

  /**
   * Run comprehensive adaptive threshold training using Composer MCP backtesting
   */
  async runAdaptiveThresholdTraining(config: AdaptiveTrainingConfig): Promise<AdaptiveTrainingResult> {
    try {
      const trainingId = `at_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      this.activeTrainingJobs.set(trainingId, config);

      logger.info(`Starting adaptive threshold training ${trainingId} for user ${config.userId}`);

      // Step 1: Get or create base strategy
      let strategy: StrategyDefinition;
      if (config.strategyId) {
        const existingStrategy = await this.databaseService.getStrategy(config.userId, config.strategyId);
        if (!existingStrategy) {
          throw new Error('Strategy not found');
        }
        strategy = existingStrategy;
      } else {
        // Create a default adaptive threshold strategy
        strategy = await this.createAdaptiveThresholdStrategy(config.userId);
      }

      // Step 2: Run training across multiple periods and market regimes
      const trainingResults = await this.runMultiPeriodTraining(trainingId, strategy, config);

      // Step 3: Optimize thresholds based on training results
      const optimizedThresholds = await this.optimizeThresholds(trainingResults, config);

      // Step 4: Validate on out-of-sample periods
      const validationResults = await this.validateThresholds(
        strategy,
        optimizedThresholds,
        config.validationPeriods,
        config
      );

      // Step 5: Calculate regime-specific thresholds
      const regimeThresholds = await this.calculateRegimeSpecificThresholds(trainingResults);

      // Step 6: Compute final scores
      const result: AdaptiveTrainingResult = {
        trainingId,
        bestThresholds: optimizedThresholds,
        performanceImprovement: this.calculatePerformanceImprovement(trainingResults),
        validationMetrics: validationResults,
        regimeSpecificThresholds: regimeThresholds,
        generalizationScore: this.calculateGeneralizationScore(trainingResults, validationResults),
        robustnessScore: this.calculateRobustnessScore(trainingResults)
      };

      // Step 7: Save training results and update adaptive threshold system
      await this.saveTrainingResults(config.userId, trainingId, result, trainingResults);

      this.activeTrainingJobs.delete(trainingId);
      
      logger.info(`Adaptive threshold training ${trainingId} completed successfully`);
      this.emit('training_completed', result);

      return result;

    } catch (error) {
      logger.error('Error in adaptive threshold training:', error);
      throw error;
    }
  }

  /**
   * Create a default strategy for adaptive threshold training
   */
  private async createAdaptiveThresholdStrategy(userId: string): Promise<StrategyDefinition> {
    const strategy: StrategyDefinition = {
      id: `adaptive_strategy_${Date.now()}`,
      name: 'Adaptive Threshold Strategy',
      description: 'Dynamic trading strategy with adaptive thresholds',
      parameters: [
        { name: 'rsi_threshold', type: 'number', defaultValue: 70, min: 50, max: 90, description: 'RSI overbought threshold' },
        { name: 'confidence_threshold', type: 'number', defaultValue: 0.65, min: 0.2, max: 1.0, description: 'Minimum confidence for trades' },
        { name: 'momentum_threshold', type: 'number', defaultValue: 2.0, min: 0.5, max: 5.0, description: 'Minimum momentum for signals' },
        { name: 'volume_threshold', type: 'number', defaultValue: 1000000, min: 100000, max: 10000000, description: 'Minimum volume requirement' },
        { name: 'macd_threshold', type: 'number', defaultValue: 0.0, min: -0.5, max: 0.5, description: 'MACD signal threshold' }
      ],
      entryRules: [
        'RSI < rsi_threshold AND confidence > confidence_threshold',
        'momentum > momentum_threshold',
        'volume > volume_threshold',
        'MACD_signal > macd_threshold'
      ],
      exitRules: [
        'RSI > 80 OR confidence < 0.4',
        'stop_loss_hit OR take_profit_hit',
        'momentum < momentum_threshold * 0.5'
      ],
      riskManagement: [
        { type: 'stop_loss', parameter: 'stop_loss_percent', value: 0.05, condition: 'less_than' },
        { type: 'take_profit', parameter: 'take_profit_percent', value: 0.15, condition: 'greater_than' },
        { type: 'position_size', parameter: 'max_position_percent', value: 0.1, condition: 'less_than' }
      ],
      version: '1.0.0'
    };

    // Save strategy to database
    const savedStrategy = await this.databaseService.createStrategy(userId, {
      name: strategy.name,
      description: strategy.description,
      strategyType: 'ml_based',
      version: strategy.version,
      parameters: strategy.parameters,
      entryRules: strategy.entryRules,
      exitRules: strategy.exitRules,
      riskManagement: strategy.riskManagement
    });

    return { ...strategy, id: savedStrategy.id };
  }

  /**
   * Run training across multiple periods and market regimes
   */
  private async runMultiPeriodTraining(
    trainingId: string,
    strategy: StrategyDefinition,
    config: AdaptiveTrainingConfig
  ): Promise<ThresholdOptimizationRun[]> {
    const results: ThresholdOptimizationRun[] = [];
    
    for (const period of config.trainingPeriods) {
      logger.info(`Training period: ${period.start} to ${period.end}`);
      
      // Get market regime for this period
      const regimes = await this.composerService.getMarketRegimes(
        config.symbols,
        period.start,
        period.end
      );
      
      const dominantRegime = this.identifyDominantRegime(regimes.regimes);
      
      // Generate parameter combinations for optimization
      const parameterCombinations = this.generateParameterCombinations(
        strategy.parameters,
        config.thresholdParameters,
        20 // Number of combinations per period
      );

      // Run backtests for each parameter combination
      for (const params of parameterCombinations) {
        try {
          const backtestConfig: BacktestConfig = {
            strategyId: strategy.id,
            symbols: config.symbols,
            startDate: period.start,
            endDate: period.end,
            initialCapital: 10000,
            parameters: params,
            riskSettings: {
              maxPositionSize: 0.1,
              stopLoss: 0.05,
              takeProfit: 0.15
            }
          };

          const backtestId = await this.composerService.runBacktest(backtestConfig);
          const result = await this.waitForBacktestCompletion(backtestId);

          if (result && result.status === 'completed') {
            const performanceScore = this.calculateAdaptivePerformanceScore(
              result,
              config.optimizationObjective
            );

            results.push({
              parameters: params,
              backtestResult: result,
              performanceScore,
              regimeType: dominantRegime
            });

            this.emit('training_progress', {
              trainingId,
              completed: results.length,
              total: config.trainingPeriods.length * parameterCombinations.length,
              currentResult: {
                period: `${period.start} to ${period.end}`,
                parameters: params,
                score: performanceScore
              }
            });
          }

        } catch (error) {
          logger.error(`Error in backtest for parameters ${JSON.stringify(params)}:`, error);
        }
      }
    }

    return results;
  }

  /**
   * Optimize thresholds based on training results
   */
  private async optimizeThresholds(
    trainingResults: ThresholdOptimizationRun[],
    config: AdaptiveTrainingConfig
  ): Promise<Record<string, number>> {
    // Group results by regime type
    const regimeResults = this.groupResultsByRegime(trainingResults);
    
    // Find best parameters overall and per regime
    const bestOverall = trainingResults.reduce((best, current) => 
      current.performanceScore > best.performanceScore ? current : best
    );

    const optimizedThresholds: Record<string, number> = {};

    // Start with best overall parameters
    for (const param of config.thresholdParameters) {
      optimizedThresholds[param] = bestOverall.parameters[param];
    }

    // Refine based on regime-specific performance
    for (const [regime, results] of Object.entries(regimeResults)) {
      if (results.length < 3) continue; // Skip regimes with insufficient data

      const regimeBest = results.reduce((best, current) => 
        current.performanceScore > best.performanceScore ? current : best
      );

      // Weight regime-specific optimization based on regime frequency
      const regimeWeight = results.length / trainingResults.length;
      
      for (const param of config.thresholdParameters) {
        const currentValue = optimizedThresholds[param];
        const regimeValue = regimeBest.parameters[param];
        
        // Weighted average between overall best and regime best
        optimizedThresholds[param] = currentValue * (1 - regimeWeight * 0.3) + 
                                   regimeValue * (regimeWeight * 0.3);
      }
    }

    return optimizedThresholds;
  }

  /**
   * Validate optimized thresholds on out-of-sample data
   */
  private async validateThresholds(
    strategy: StrategyDefinition,
    thresholds: Record<string, number>,
    validationPeriods: Array<{ start: string; end: string }>,
    config: AdaptiveTrainingConfig
  ): Promise<{
    sharpeRatio: number;
    maxDrawdown: number;
    winRate: number;
    profitFactor: number;
  }> {
    const validationResults: BacktestResult[] = [];

    for (const period of validationPeriods) {
      const backtestConfig: BacktestConfig = {
        strategyId: strategy.id,
        symbols: config.symbols,
        startDate: period.start,
        endDate: period.end,
        initialCapital: 10000,
        parameters: thresholds,
        riskSettings: {
          maxPositionSize: 0.1,
          stopLoss: 0.05,
          takeProfit: 0.15
        }
      };

      const backtestId = await this.composerService.runBacktest(backtestConfig);
      const result = await this.waitForBacktestCompletion(backtestId);

      if (result && result.status === 'completed') {
        validationResults.push(result);
      }
    }

    if (validationResults.length === 0) {
      throw new Error('No successful validation backtests');
    }

    // Calculate average validation metrics
    const avgSharpe = validationResults.reduce((sum, r) => sum + r.performance.sharpeRatio, 0) / validationResults.length;
    const maxDrawdown = Math.max(...validationResults.map(r => r.performance.maxDrawdown));
    const avgWinRate = validationResults.reduce((sum, r) => sum + r.performance.winRate, 0) / validationResults.length;
    const avgProfitFactor = validationResults.reduce((sum, r) => sum + r.performance.profitFactor, 0) / validationResults.length;

    return {
      sharpeRatio: avgSharpe,
      maxDrawdown,
      winRate: avgWinRate,
      profitFactor: avgProfitFactor
    };
  }

  /**
   * Calculate regime-specific threshold recommendations
   */
  private async calculateRegimeSpecificThresholds(
    trainingResults: ThresholdOptimizationRun[]
  ): Promise<Record<string, Record<string, number>>> {
    const regimeResults = this.groupResultsByRegime(trainingResults);
    const regimeThresholds: Record<string, Record<string, number>> = {};

    for (const [regime, results] of Object.entries(regimeResults)) {
      if (results.length < 3) continue;

      const best = results.reduce((best, current) => 
        current.performanceScore > best.performanceScore ? current : best
      );

      regimeThresholds[regime] = best.parameters;
    }

    return regimeThresholds;
  }

  // Helper methods

  private identifyDominantRegime(regimes: any[]): string {
    if (regimes.length === 0) return 'unknown';
    
    // Find the regime with the longest duration
    const regimeDurations = regimes.reduce((acc, regime) => {
      const duration = new Date(regime.period.end).getTime() - new Date(regime.period.start).getTime();
      acc[regime.type] = (acc[regime.type] || 0) + duration;
      return acc;
    }, {} as Record<string, number>);

    return Object.entries(regimeDurations).reduce((a, b) => a[1] > b[1] ? a : b)[0];
  }

  private generateParameterCombinations(
    strategyParams: any[],
    thresholdParams: string[],
    numCombinations: number
  ): Record<string, number>[] {
    const combinations: Record<string, number>[] = [];
    
    // Create base parameter set
    const baseParams = strategyParams.reduce((acc, param) => {
      acc[param.name] = param.defaultValue;
      return acc;
    }, {} as Record<string, number>);

    // Generate random combinations within parameter bounds
    for (let i = 0; i < numCombinations; i++) {
      const params = { ...baseParams };
      
      for (const paramName of thresholdParams) {
        const param = strategyParams.find(p => p.name === paramName);
        if (param && param.min !== undefined && param.max !== undefined) {
          // Generate random value within bounds
          params[paramName] = param.min + Math.random() * (param.max - param.min);
        }
      }
      
      combinations.push(params);
    }

    return combinations;
  }

  private groupResultsByRegime(results: ThresholdOptimizationRun[]): Record<string, ThresholdOptimizationRun[]> {
    return results.reduce((groups, result) => {
      const regime = result.regimeType || 'unknown';
      if (!groups[regime]) groups[regime] = [];
      groups[regime].push(result);
      return groups;
    }, {} as Record<string, ThresholdOptimizationRun[]>);
  }

  private calculateAdaptivePerformanceScore(
    result: BacktestResult,
    objective: string
  ): number {
    switch (objective) {
      case 'sharpe':
        return result.performance.sharpeRatio;
      case 'return':
        return result.performance.totalReturn;
      case 'profit_factor':
        return result.performance.profitFactor;
      case 'sortino':
        return result.performance.sortino;
      default:
        // Composite score
        return (
          result.performance.sharpeRatio * 0.4 +
          result.performance.totalReturn * 0.3 +
          result.performance.winRate * 0.2 +
          (1 - result.performance.maxDrawdown) * 0.1
        );
    }
  }

  private calculatePerformanceImprovement(results: ThresholdOptimizationRun[]): number {
    if (results.length < 2) return 0;
    
    const scores = results.map(r => r.performanceScore);
    const best = Math.max(...scores);
    const worst = Math.min(...scores);
    
    return worst > 0 ? (best - worst) / worst : 0;
  }

  private calculateGeneralizationScore(
    trainingResults: ThresholdOptimizationRun[],
    validationMetrics: any
  ): number {
    if (trainingResults.length === 0) return 0;
    
    const avgTrainingSharpe = trainingResults.reduce((sum, r) => 
      sum + r.backtestResult.performance.sharpeRatio, 0) / trainingResults.length;
    
    const validationSharpe = validationMetrics.sharpeRatio;
    
    // Score based on how well validation performance matches training
    const ratio = validationSharpe / avgTrainingSharpe;
    return Math.max(0, Math.min(100, ratio * 100));
  }

  private calculateRobustnessScore(results: ThresholdOptimizationRun[]): number {
    if (results.length < 5) return 50; // Default score for insufficient data
    
    const scores = results.map(r => r.performanceScore);
    const mean = scores.reduce((sum, s) => sum + s, 0) / scores.length;
    const variance = scores.reduce((sum, s) => sum + Math.pow(s - mean, 2), 0) / scores.length;
    const stdDev = Math.sqrt(variance);
    
    // Lower standard deviation = higher robustness
    const coefficientOfVariation = stdDev / Math.abs(mean);
    return Math.max(0, Math.min(100, (1 - coefficientOfVariation) * 100));
  }

  private async waitForBacktestCompletion(backtestId: string, timeout: number = 300000): Promise<BacktestResult | null> {
    return new Promise((resolve) => {
      const startTime = Date.now();
      
      const checkStatus = async () => {
        try {
          const result = await this.composerService.getBacktestStatus(backtestId);
          
          if (result && (result.status === 'completed' || result.status === 'failed')) {
            resolve(result);
            return;
          }

          if (Date.now() - startTime > timeout) {
            logger.warn(`Backtest ${backtestId} timed out`);
            resolve(null);
            return;
          }

          setTimeout(checkStatus, 5000); // Check every 5 seconds
        } catch (error) {
          logger.error('Error checking backtest status:', error);
          resolve(null);
        }
      };

      checkStatus();
    });
  }

  private async saveTrainingResults(
    userId: string,
    trainingId: string,
    result: AdaptiveTrainingResult,
    trainingRuns: ThresholdOptimizationRun[]
  ): Promise<void> {
    try {
      // Save training metadata
      const trainingData = trainingRuns.map(run => ({
        userId,
        strategyId: run.backtestResult.strategyId,
        backtestResultId: run.backtestResult.id,
        trainingPeriodStart: new Date(run.backtestResult.startedAt),
        trainingPeriodEnd: new Date(run.backtestResult.completedAt || run.backtestResult.startedAt),
        marketConditions: {
          volatility: run.backtestResult.performance.volatility,
          regimeType: run.regimeType
        },
        strategyParameters: run.parameters,
        thresholdValues: result.bestThresholds,
        performanceScore: run.performanceScore,
        sharpeRatio: run.backtestResult.performance.sharpeRatio,
        maxDrawdown: run.backtestResult.performance.maxDrawdown,
        winRate: run.backtestResult.performance.winRate,
        dataQualityScore: 0.8,
        regimeType: run.regimeType || 'unknown',
        volatilityBucket: this.getVolatilityBucket(run.backtestResult.performance.volatility)
      }));

      await this.databaseService.saveAdaptiveThresholdTraining(userId, trainingData);

      logger.info(`Saved ${trainingData.length} training data points for user ${userId}`);

    } catch (error) {
      logger.error('Error saving training results:', error);
      throw error;
    }
  }

  private getVolatilityBucket(volatility: number): string {
    if (volatility < 0.15) return 'low';
    if (volatility < 0.3) return 'medium';
    return 'high';
  }

  /**
   * Get current training job status
   */
  getTrainingJobStatus(trainingId: string): AdaptiveTrainingConfig | null {
    return this.activeTrainingJobs.get(trainingId) || null;
  }

  /**
   * Cancel a running training job
   */
  async cancelTrainingJob(trainingId: string): Promise<void> {
    const config = this.activeTrainingJobs.get(trainingId);
    if (config) {
      this.activeTrainingJobs.delete(trainingId);
      this.emit('training_cancelled', { trainingId });
      logger.info(`Training job ${trainingId} cancelled`);
    }
  }
}