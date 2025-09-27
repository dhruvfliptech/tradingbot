import { EventEmitter } from 'events';
import { ComposerService, StrategyDefinition, BacktestConfig, BacktestResult } from '../composer/ComposerService';
// import { AnalyticsService } from '../analytics/AnalyticsService'; // TODO: Create AnalyticsService
import logger from '../../utils/logger';

export interface ValidationRule {
  name: string;
  type: 'syntax' | 'logic' | 'performance' | 'risk' | 'market_fit';
  severity: 'error' | 'warning' | 'info';
  description: string;
  validator: (strategy: StrategyDefinition, context?: ValidationContext) => ValidationResult;
}

export interface ValidationContext {
  historicalData?: any[];
  marketConditions?: {
    volatility: number;
    trend: 'bull' | 'bear' | 'sideways';
    liquidity: number;
  };
  userProfile?: {
    riskTolerance: 'low' | 'medium' | 'high';
    experience: 'beginner' | 'intermediate' | 'advanced';
    capital: number;
  };
  backtestResults?: BacktestResult[];
}

export interface ValidationResult {
  passed: boolean;
  score: number; // 0-100
  message: string;
  details?: any;
  suggestions?: string[];
}

export interface StrategyValidationReport {
  strategyId: string;
  overallScore: number;
  status: 'passed' | 'failed' | 'warning';
  validatedAt: Date;
  errors: ValidationIssue[];
  warnings: ValidationIssue[];
  recommendations: string[];
  performanceValidation?: PerformanceValidationResult;
  riskValidation?: RiskValidationResult;
  marketFitValidation?: MarketFitValidationResult;
}

export interface ValidationIssue {
  rule: string;
  severity: 'error' | 'warning' | 'info';
  message: string;
  location?: string;
  suggestion?: string;
}

export interface PerformanceValidationResult {
  minSharpeRequired: number;
  actualSharpe: number;
  minWinRateRequired: number;
  actualWinRate: number;
  maxDrawdownAllowed: number;
  actualMaxDrawdown: number;
  profitFactorThreshold: number;
  actualProfitFactor: number;
  backtestPeriods: string[];
  consistencyScore: number;
}

export interface RiskValidationResult {
  maxPositionSizeCheck: boolean;
  stopLossValidation: boolean;
  diversificationCheck: boolean;
  correlationAnalysis: {
    maxCorrelation: number;
    averageCorrelation: number;
    problematicPairs: string[];
  };
  volatilityExposure: {
    averageVolatility: number;
    maxVolatility: number;
    volatilityBudget: number;
  };
}

export interface MarketFitValidationResult {
  regimePerformance: {
    bull: { score: number; trades: number };
    bear: { score: number; trades: number };
    sideways: { score: number; trades: number };
    volatile: { score: number; trades: number };
  };
  adaptabilityScore: number;
  robustnessScore: number;
  liquidityRequirements: {
    minVolume: number;
    avgSpread: number;
    marketImpact: number;
  };
}

export class StrategyValidationService extends EventEmitter {
  private composerService: ComposerService;
  private analyticsService: AnalyticsService;
  private validationRules: Map<string, ValidationRule> = new Map();

  constructor(composerService: ComposerService) {
    super();
    this.composerService = composerService;
    // this.analyticsService = new AnalyticsService(); // TODO: Create AnalyticsService
    this.initializeValidationRules();
  }

  private initializeValidationRules(): void {
    // Syntax Validation Rules
    this.addValidationRule({
      name: 'parameter_completeness',
      type: 'syntax',
      severity: 'error',
      description: 'Validates that all required parameters are defined',
      validator: this.validateParameterCompleteness.bind(this)
    });

    this.addValidationRule({
      name: 'entry_exit_rules',
      type: 'logic',
      severity: 'error',
      description: 'Validates that entry and exit rules are properly defined',
      validator: this.validateEntryExitRules.bind(this)
    });

    this.addValidationRule({
      name: 'risk_management',
      type: 'risk',
      severity: 'error',
      description: 'Validates risk management rules',
      validator: this.validateRiskManagement.bind(this)
    });

    // Performance Validation Rules
    this.addValidationRule({
      name: 'backtest_performance',
      type: 'performance',
      severity: 'warning',
      description: 'Validates strategy performance metrics',
      validator: this.validateBacktestPerformance.bind(this)
    });

    this.addValidationRule({
      name: 'overfitting_check',
      type: 'performance',
      severity: 'warning',
      description: 'Checks for signs of overfitting',
      validator: this.validateOverfitting.bind(this)
    });

    // Market Fit Validation Rules
    this.addValidationRule({
      name: 'regime_adaptability',
      type: 'market_fit',
      severity: 'info',
      description: 'Validates performance across different market regimes',
      validator: this.validateRegimeAdaptability.bind(this)
    });

    this.addValidationRule({
      name: 'liquidity_requirements',
      type: 'market_fit',
      severity: 'warning',
      description: 'Validates strategy liquidity requirements',
      validator: this.validateLiquidityRequirements.bind(this)
    });
  }

  private addValidationRule(rule: ValidationRule): void {
    this.validationRules.set(rule.name, rule);
  }

  async validateStrategy(
    strategy: StrategyDefinition,
    context?: ValidationContext
  ): Promise<StrategyValidationReport> {
    try {
      logger.info(`Starting validation for strategy ${strategy.id}`);

      const report: StrategyValidationReport = {
        strategyId: strategy.id,
        overallScore: 0,
        status: 'passed',
        validatedAt: new Date(),
        errors: [],
        warnings: [],
        recommendations: []
      };

      let totalScore = 0;
      let ruleCount = 0;

      // Run all validation rules
      for (const [ruleName, rule] of this.validationRules) {
        try {
          const result = await rule.validator(strategy, context);
          ruleCount++;
          totalScore += result.score;

          if (!result.passed) {
            const issue: ValidationIssue = {
              rule: ruleName,
              severity: rule.severity,
              message: result.message,
              suggestion: result.suggestions?.[0]
            };

            if (rule.severity === 'error') {
              report.errors.push(issue);
            } else if (rule.severity === 'warning') {
              report.warnings.push(issue);
            }
          }

          if (result.suggestions) {
            report.recommendations.push(...result.suggestions);
          }

        } catch (error) {
          logger.error(`Error running validation rule ${ruleName}:`, error);
          report.errors.push({
            rule: ruleName,
            severity: 'error',
            message: `Validation rule failed: ${error.message}`
          });
        }
      }

      // Calculate overall score
      report.overallScore = ruleCount > 0 ? totalScore / ruleCount : 0;

      // Determine status
      if (report.errors.length > 0) {
        report.status = 'failed';
      } else if (report.warnings.length > 0) {
        report.status = 'warning';
      } else {
        report.status = 'passed';
      }

      // Run specialized validations if context is available
      if (context?.backtestResults && context.backtestResults.length > 0) {
        report.performanceValidation = await this.validatePerformance(
          strategy,
          context.backtestResults
        );
        
        report.riskValidation = await this.validateRisk(
          strategy,
          context.backtestResults
        );
        
        report.marketFitValidation = await this.validateMarketFit(
          strategy,
          context.backtestResults
        );
      }

      logger.info(`Strategy validation completed. Score: ${report.overallScore}, Status: ${report.status}`);
      return report;

    } catch (error) {
      logger.error('Error validating strategy:', error);
      throw error;
    }
  }

  private validateParameterCompleteness(strategy: StrategyDefinition): ValidationResult {
    const requiredFields = ['name', 'description', 'parameters', 'entryRules', 'exitRules'];
    const missing = requiredFields.filter(field => !strategy[field as keyof StrategyDefinition]);

    if (missing.length > 0) {
      return {
        passed: false,
        score: 0,
        message: `Missing required fields: ${missing.join(', ')}`,
        suggestions: [`Add the missing fields: ${missing.join(', ')}`]
      };
    }

    // Check parameter definitions
    const parameterIssues = [];
    for (const param of strategy.parameters) {
      if (!param.name || !param.type || param.defaultValue === undefined) {
        parameterIssues.push(`Parameter "${param.name}" is incomplete`);
      }
    }

    if (parameterIssues.length > 0) {
      return {
        passed: false,
        score: 50,
        message: `Parameter definition issues: ${parameterIssues.join(', ')}`,
        suggestions: ['Ensure all parameters have name, type, and defaultValue']
      };
    }

    return {
      passed: true,
      score: 100,
      message: 'All required parameters are complete'
    };
  }

  private validateEntryExitRules(strategy: StrategyDefinition): ValidationResult {
    if (strategy.entryRules.length === 0) {
      return {
        passed: false,
        score: 0,
        message: 'No entry rules defined',
        suggestions: ['Add at least one entry rule for signal generation']
      };
    }

    if (strategy.exitRules.length === 0) {
      return {
        passed: false,
        score: 0,
        message: 'No exit rules defined',
        suggestions: ['Add exit rules for position management']
      };
    }

    // Check for balanced entry/exit logic
    const hasStopLoss = strategy.exitRules.some(rule => 
      rule.toLowerCase().includes('stop') || rule.toLowerCase().includes('loss')
    );
    
    const hasTakeProfit = strategy.exitRules.some(rule => 
      rule.toLowerCase().includes('profit') || rule.toLowerCase().includes('target')
    );

    if (!hasStopLoss) {
      return {
        passed: false,
        score: 60,
        message: 'No stop loss rule detected',
        suggestions: ['Add stop loss rules for risk management']
      };
    }

    if (!hasTakeProfit) {
      return {
        passed: true,
        score: 80,
        message: 'No take profit rule detected',
        suggestions: ['Consider adding take profit rules for better profit management']
      };
    }

    return {
      passed: true,
      score: 100,
      message: 'Entry and exit rules are properly defined'
    };
  }

  private validateRiskManagement(strategy: StrategyDefinition): ValidationResult {
    const riskRules = strategy.riskManagement;
    const issues = [];
    let score = 100;

    // Check for position sizing
    const hasPositionSizing = riskRules.some((rule: any) => 
      rule.type === 'position_size' || rule.parameter?.includes('position')
    );
    
    if (!hasPositionSizing) {
      issues.push('No position sizing rules defined');
      score -= 30;
    }

    // Check for stop loss
    const hasStopLoss = riskRules.some((rule: any) => 
      rule.type === 'stop_loss'
    );
    
    if (!hasStopLoss) {
      issues.push('No stop loss rules defined');
      score -= 40;
    }

    // Check for exposure limits
    const hasExposureLimit = riskRules.some((rule: any) => 
      rule.type === 'exposure' || rule.type === 'correlation'
    );
    
    if (!hasExposureLimit) {
      issues.push('No exposure or correlation limits defined');
      score -= 30;
    }

    if (issues.length > 0) {
      return {
        passed: score >= 60,
        score: Math.max(0, score),
        message: `Risk management issues: ${issues.join(', ')}`,
        suggestions: [
          'Add position sizing rules',
          'Define stop loss mechanisms',
          'Set exposure and correlation limits'
        ]
      };
    }

    return {
      passed: true,
      score: 100,
      message: 'Risk management rules are comprehensive'
    };
  }

  private validateBacktestPerformance(
    strategy: StrategyDefinition,
    context?: ValidationContext
  ): ValidationResult {
    if (!context?.backtestResults || context.backtestResults.length === 0) {
      return {
        passed: false,
        score: 0,
        message: 'No backtest results available for performance validation',
        suggestions: ['Run backtests before validating performance']
      };
    }

    const results = context.backtestResults;
    const avgSharpe = results.reduce((sum, r) => sum + r.performance.sharpeRatio, 0) / results.length;
    const avgWinRate = results.reduce((sum, r) => sum + r.performance.winRate, 0) / results.length;
    const maxDrawdown = Math.max(...results.map(r => r.performance.maxDrawdown));

    let score = 100;
    const issues = [];

    // Sharpe ratio check
    if (avgSharpe < 1.0) {
      score -= 30;
      issues.push(`Low Sharpe ratio: ${avgSharpe.toFixed(2)}`);
    }

    // Win rate check
    if (avgWinRate < 0.4) {
      score -= 20;
      issues.push(`Low win rate: ${(avgWinRate * 100).toFixed(1)}%`);
    }

    // Drawdown check
    if (maxDrawdown > 0.2) {
      score -= 25;
      issues.push(`High maximum drawdown: ${(maxDrawdown * 100).toFixed(1)}%`);
    }

    // Profit factor check
    const avgProfitFactor = results.reduce((sum, r) => sum + r.performance.profitFactor, 0) / results.length;
    if (avgProfitFactor < 1.2) {
      score -= 25;
      issues.push(`Low profit factor: ${avgProfitFactor.toFixed(2)}`);
    }

    if (issues.length > 0) {
      return {
        passed: score >= 60,
        score: Math.max(0, score),
        message: `Performance issues: ${issues.join(', ')}`,
        suggestions: [
          'Consider adjusting strategy parameters',
          'Review entry and exit criteria',
          'Optimize risk management rules'
        ]
      };
    }

    return {
      passed: true,
      score: 100,
      message: 'Strategy performance meets requirements'
    };
  }

  private validateOverfitting(
    strategy: StrategyDefinition,
    context?: ValidationContext
  ): ValidationResult {
    if (!context?.backtestResults || context.backtestResults.length < 2) {
      return {
        passed: true,
        score: 50,
        message: 'Insufficient data for overfitting analysis',
        suggestions: ['Run multiple backtest periods for better validation']
      };
    }

    const results = context.backtestResults;
    const sharpeRatios = results.map(r => r.performance.sharpeRatio);
    const returns = results.map(r => r.performance.totalReturn);

    // Calculate consistency metrics
    const sharpeStd = this.calculateStandardDeviation(sharpeRatios);
    const returnStd = this.calculateStandardDeviation(returns);
    
    const sharpeMean = sharpeRatios.reduce((sum, val) => sum + val, 0) / sharpeRatios.length;
    const returnMean = returns.reduce((sum, val) => sum + val, 0) / returns.length;

    // High coefficient of variation indicates potential overfitting
    const sharpeCv = sharpeMean !== 0 ? sharpeStd / Math.abs(sharpeMean) : Infinity;
    const returnCv = returnMean !== 0 ? returnStd / Math.abs(returnMean) : Infinity;

    let score = 100;
    const issues = [];

    if (sharpeCv > 0.5) {
      score -= 40;
      issues.push('High variance in Sharpe ratios across periods');
    }

    if (returnCv > 0.8) {
      score -= 30;
      issues.push('High variance in returns across periods');
    }

    // Check for declining performance over time
    const recentResults = results.slice(-Math.ceil(results.length / 2));
    const earlyResults = results.slice(0, Math.floor(results.length / 2));
    
    const recentAvgSharpe = recentResults.reduce((sum, r) => sum + r.performance.sharpeRatio, 0) / recentResults.length;
    const earlyAvgSharpe = earlyResults.reduce((sum, r) => sum + r.performance.sharpeRatio, 0) / earlyResults.length;

    if (recentAvgSharpe < earlyAvgSharpe * 0.7) {
      score -= 30;
      issues.push('Declining performance in recent periods');
    }

    if (issues.length > 0) {
      return {
        passed: score >= 60,
        score: Math.max(0, score),
        message: `Potential overfitting detected: ${issues.join(', ')}`,
        suggestions: [
          'Test strategy on out-of-sample data',
          'Simplify strategy parameters',
          'Use walk-forward analysis',
          'Implement robust cross-validation'
        ]
      };
    }

    return {
      passed: true,
      score: score,
      message: 'No significant overfitting detected'
    };
  }

  private validateRegimeAdaptability(
    strategy: StrategyDefinition,
    context?: ValidationContext
  ): ValidationResult {
    // This would require regime analysis from ComposerService
    // For now, return a placeholder implementation
    return {
      passed: true,
      score: 75,
      message: 'Regime adaptability analysis requires extended backtesting',
      suggestions: ['Run backtests across different market conditions']
    };
  }

  private validateLiquidityRequirements(
    strategy: StrategyDefinition,
    context?: ValidationContext
  ): ValidationResult {
    // Analyze strategy's liquidity requirements based on trading frequency
    const parameters = strategy.parameters;
    const hasHighFrequency = parameters.some(p => 
      p.name.toLowerCase().includes('frequency') && 
      typeof p.defaultValue === 'number' && 
      p.defaultValue > 10
    );

    if (hasHighFrequency) {
      return {
        passed: true,
        score: 70,
        message: 'High-frequency strategy detected',
        suggestions: [
          'Ensure adequate liquidity in target markets',
          'Consider market impact in position sizing',
          'Monitor bid-ask spreads'
        ]
      };
    }

    return {
      passed: true,
      score: 90,
      message: 'Liquidity requirements appear reasonable'
    };
  }

  private async validatePerformance(
    strategy: StrategyDefinition,
    backtestResults: BacktestResult[]
  ): Promise<PerformanceValidationResult> {
    const avgSharpe = backtestResults.reduce((sum, r) => sum + r.performance.sharpeRatio, 0) / backtestResults.length;
    const avgWinRate = backtestResults.reduce((sum, r) => sum + r.performance.winRate, 0) / backtestResults.length;
    const maxDrawdown = Math.max(...backtestResults.map(r => r.performance.maxDrawdown));
    const avgProfitFactor = backtestResults.reduce((sum, r) => sum + r.performance.profitFactor, 0) / backtestResults.length;

    // Calculate consistency score
    const sharpeRatios = backtestResults.map(r => r.performance.sharpeRatio);
    const consistencyScore = 100 - (this.calculateStandardDeviation(sharpeRatios) / Math.abs(avgSharpe)) * 50;

    return {
      minSharpeRequired: 1.0,
      actualSharpe: avgSharpe,
      minWinRateRequired: 0.4,
      actualWinRate: avgWinRate,
      maxDrawdownAllowed: 0.2,
      actualMaxDrawdown: maxDrawdown,
      profitFactorThreshold: 1.2,
      actualProfitFactor: avgProfitFactor,
      backtestPeriods: backtestResults.map(r => `${r.startedAt.toISOString()}`),
      consistencyScore: Math.max(0, Math.min(100, consistencyScore))
    };
  }

  private async validateRisk(
    strategy: StrategyDefinition,
    backtestResults: BacktestResult[]
  ): Promise<RiskValidationResult> {
    // Analyze risk characteristics from backtest results
    const maxPositionSizeCheck = strategy.riskManagement.some((rule: any) => rule.type === 'position_size');
    const stopLossValidation = strategy.riskManagement.some((rule: any) => rule.type === 'stop_loss');
    
    // Calculate volatility metrics
    const returns = backtestResults.flatMap(r => r.trades.map(t => t.pnlPercent || 0));
    const avgVolatility = this.calculateStandardDeviation(returns);
    const maxVolatility = Math.max(...returns.map(Math.abs));

    return {
      maxPositionSizeCheck,
      stopLossValidation,
      diversificationCheck: true, // Placeholder
      correlationAnalysis: {
        maxCorrelation: 0.8, // Placeholder
        averageCorrelation: 0.3, // Placeholder
        problematicPairs: []
      },
      volatilityExposure: {
        averageVolatility: avgVolatility,
        maxVolatility: maxVolatility,
        volatilityBudget: 0.15 // 15% volatility budget
      }
    };
  }

  private async validateMarketFit(
    strategy: StrategyDefinition,
    backtestResults: BacktestResult[]
  ): Promise<MarketFitValidationResult> {
    // This would require market regime analysis
    // For now, return placeholder data
    return {
      regimePerformance: {
        bull: { score: 75, trades: 50 },
        bear: { score: 60, trades: 30 },
        sideways: { score: 80, trades: 40 },
        volatile: { score: 55, trades: 25 }
      },
      adaptabilityScore: 70,
      robustnessScore: 75,
      liquidityRequirements: {
        minVolume: 1000000,
        avgSpread: 0.001,
        marketImpact: 0.0005
      }
    };
  }

  private calculateStandardDeviation(values: number[]): number {
    if (values.length === 0) return 0;
    
    const mean = values.reduce((sum, val) => sum + val, 0) / values.length;
    const squaredDiffs = values.map(val => Math.pow(val - mean, 2));
    const avgSquaredDiff = squaredDiffs.reduce((sum, val) => sum + val, 0) / values.length;
    
    return Math.sqrt(avgSquaredDiff);
  }

  async runComprehensiveValidation(
    strategy: StrategyDefinition,
    symbols: string[],
    period: { start: string; end: string }
  ): Promise<StrategyValidationReport> {
    try {
      logger.info(`Running comprehensive validation for strategy ${strategy.id}`);

      // Step 1: Run initial syntax and logic validation
      let report = await this.validateStrategy(strategy);

      if (report.status === 'failed') {
        logger.warn('Strategy failed initial validation, skipping performance tests');
        return report;
      }

      // Step 2: Run backtests for performance validation
      const backtestConfig: BacktestConfig = {
        strategyId: strategy.id,
        symbols,
        startDate: period.start,
        endDate: period.end,
        initialCapital: 10000,
        parameters: strategy.parameters.reduce((acc, param) => {
          acc[param.name] = param.defaultValue;
          return acc;
        }, {} as Record<string, any>),
        riskSettings: {
          maxPositionSize: 0.1,
          stopLoss: 0.05,
          takeProfit: 0.15
        }
      };

      const backtestId = await this.composerService.runBacktest(backtestConfig);

      // Wait for backtest completion
      const backtestResult = await this.waitForBacktestCompletion(backtestId);
      
      if (backtestResult && backtestResult.status === 'completed') {
        // Step 3: Re-run validation with backtest results
        const context: ValidationContext = {
          backtestResults: [backtestResult]
        };

        report = await this.validateStrategy(strategy, context);
      }

      return report;

    } catch (error) {
      logger.error('Error in comprehensive validation:', error);
      throw error;
    }
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
}