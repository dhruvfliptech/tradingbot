import { createClient } from '@supabase/supabase-js';
import logger from '../../utils/logger';

export interface StrategyData {
  name: string;
  description: string;
  strategyType: string;
  version: string;
  parameters: any;
  entryRules: string[];
  exitRules: string[];
  riskManagement: any;
  isValidated?: boolean;
  validationScore?: number;
}

export interface BacktestConfigData {
  strategyId: string;
  name: string;
  symbols: string[];
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  timeframe?: string;
  commissionRate?: number;
  slippageRate?: number;
  strategyParameters: any;
  riskSettings: any;
}

export interface BacktestResultData {
  configId: string;
  strategyId: string;
  externalBacktestId: string;
  status: string;
  progress?: number;
  totalReturn?: number;
  sharpeRatio?: number;
  sortinoRatio?: number;
  calmarRatio?: number;
  maxDrawdown?: number;
  winRate?: number;
  profitFactor?: number;
  volatility?: number;
  beta?: number;
  alpha?: number;
  totalTrades?: number;
  winningTrades?: number;
  losingTrades?: number;
  avgWin?: number;
  avgLoss?: number;
  largestWin?: number;
  largestLoss?: number;
  consecutiveWins?: number;
  consecutiveLosses?: number;
  avgTradeDuration?: string;
  totalCommissions?: number;
  var95?: number;
  var99?: number;
  expectedShortfall?: number;
  trackingError?: number;
  informationRatio?: number;
  downsideDeviation?: number;
  detailedMetrics?: any;
  equityCurve?: any[];
  drawdownPeriods?: any[];
  monthlyReturns?: any;
  errorMessage?: string;
  startedAt?: Date;
  completedAt?: Date;
}

export interface OptimizationData {
  strategyId: string;
  configId?: string;
  externalOptimizationId: string;
  optimizationMethod: string;
  objectiveFunction: string;
  parametersToOptimize: string[];
  iterations: number;
  status: string;
  progress?: number;
  bestParameters?: any;
  bestPerformance?: number;
  bestMetrics?: any;
  optimizationResults?: any[];
  parameterImportance?: any;
  errorMessage?: string;
  startedAt?: Date;
  completedAt?: Date;
}

export class DatabaseService {
  private supabase: any;

  constructor() {
    this.supabase = createClient(
      process.env.SUPABASE_URL!,
      process.env.SUPABASE_SERVICE_ROLE_KEY!
    );
  }

  // Strategy Management

  async createStrategy(userId: string, data: StrategyData): Promise<any> {
    try {
      const { data: strategy, error } = await this.supabase
        .from('strategy_definitions')
        .insert({
          user_id: userId,
          name: data.name,
          description: data.description,
          strategy_type: data.strategyType,
          version: data.version,
          parameters: data.parameters,
          entry_rules: data.entryRules,
          exit_rules: data.exitRules,
          risk_management: data.riskManagement,
          is_validated: data.isValidated || false,
          validation_score: data.validationScore || null
        })
        .select()
        .single();

      if (error) throw error;
      return strategy;
    } catch (error) {
      logger.error('Error creating strategy:', error);
      throw error;
    }
  }

  async getUserStrategies(userId: string, options: {
    activeOnly?: boolean;
    validatedOnly?: boolean;
  } = {}): Promise<any[]> {
    try {
      let query = this.supabase
        .from('strategy_definitions')
        .select('*')
        .eq('user_id', userId)
        .order('created_at', { ascending: false });

      if (options.activeOnly) {
        query = query.eq('is_active', true);
      }

      if (options.validatedOnly) {
        query = query.eq('is_validated', true);
      }

      const { data, error } = await query;
      if (error) throw error;
      return data || [];
    } catch (error) {
      logger.error('Error fetching user strategies:', error);
      throw error;
    }
  }

  async getStrategy(userId: string, strategyId: string): Promise<any | null> {
    try {
      const { data, error } = await this.supabase
        .from('strategy_definitions')
        .select('*')
        .eq('user_id', userId)
        .eq('id', strategyId)
        .single();

      if (error && error.code !== 'PGRST116') throw error;
      return data;
    } catch (error) {
      logger.error('Error fetching strategy:', error);
      throw error;
    }
  }

  async updateStrategyValidation(
    userId: string,
    strategyId: string,
    isValidated: boolean,
    validationScore: number
  ): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('strategy_definitions')
        .update({
          is_validated: isValidated,
          validation_score: validationScore,
          updated_at: new Date().toISOString()
        })
        .eq('user_id', userId)
        .eq('id', strategyId);

      if (error) throw error;
    } catch (error) {
      logger.error('Error updating strategy validation:', error);
      throw error;
    }
  }

  // Backtest Configuration

  async createBacktestConfig(userId: string, data: BacktestConfigData): Promise<any> {
    try {
      const { data: config, error } = await this.supabase
        .from('backtest_configs')
        .insert({
          user_id: userId,
          strategy_id: data.strategyId,
          name: data.name,
          symbols: data.symbols,
          start_date: data.startDate.toISOString(),
          end_date: data.endDate.toISOString(),
          initial_capital: data.initialCapital,
          timeframe: data.timeframe || '1h',
          commission_rate: data.commissionRate || 0.001,
          slippage_rate: data.slippageRate || 0.0005,
          strategy_parameters: data.strategyParameters,
          risk_settings: data.riskSettings
        })
        .select()
        .single();

      if (error) throw error;
      return config;
    } catch (error) {
      logger.error('Error creating backtest config:', error);
      throw error;
    }
  }

  // Backtest Results

  async createBacktestResult(userId: string, data: Partial<BacktestResultData>): Promise<any> {
    try {
      const { data: result, error } = await this.supabase
        .from('backtest_results')
        .insert({
          user_id: userId,
          config_id: data.configId,
          strategy_id: data.strategyId,
          external_backtest_id: data.externalBacktestId,
          status: data.status || 'pending',
          started_at: data.startedAt?.toISOString() || new Date().toISOString()
        })
        .select()
        .single();

      if (error) throw error;
      return result;
    } catch (error) {
      logger.error('Error creating backtest result:', error);
      throw error;
    }
  }

  async updateBacktestProgress(
    backtestId: string,
    progress: number,
    performance?: any
  ): Promise<void> {
    try {
      const updateData: any = {
        progress,
        updated_at: new Date().toISOString()
      };

      if (performance) {
        updateData.total_return = performance.totalReturn;
        updateData.sharpe_ratio = performance.sharpeRatio;
        updateData.max_drawdown = performance.maxDrawdown;
        updateData.win_rate = performance.winRate;
      }

      const { error } = await this.supabase
        .from('backtest_results')
        .update(updateData)
        .eq('id', backtestId);

      if (error) throw error;
    } catch (error) {
      logger.error('Error updating backtest progress:', error);
      throw error;
    }
  }

  async updateBacktestResult(backtestId: string, data: Partial<BacktestResultData>): Promise<void> {
    try {
      const updateData: any = {
        ...data,
        updated_at: new Date().toISOString()
      };

      // Convert Date objects to ISO strings
      if (data.startedAt) updateData.started_at = data.startedAt.toISOString();
      if (data.completedAt) updateData.completed_at = data.completedAt.toISOString();

      // Map camelCase to snake_case for database fields
      if (data.totalReturn !== undefined) updateData.total_return = data.totalReturn;
      if (data.sharpeRatio !== undefined) updateData.sharpe_ratio = data.sharpeRatio;
      if (data.sortinoRatio !== undefined) updateData.sortino_ratio = data.sortinoRatio;
      if (data.calmarRatio !== undefined) updateData.calmar_ratio = data.calmarRatio;
      if (data.maxDrawdown !== undefined) updateData.max_drawdown = data.maxDrawdown;
      if (data.winRate !== undefined) updateData.win_rate = data.winRate;
      if (data.profitFactor !== undefined) updateData.profit_factor = data.profitFactor;
      if (data.totalTrades !== undefined) updateData.total_trades = data.totalTrades;
      if (data.winningTrades !== undefined) updateData.winning_trades = data.winningTrades;
      if (data.losingTrades !== undefined) updateData.losing_trades = data.losingTrades;
      if (data.avgWin !== undefined) updateData.avg_win = data.avgWin;
      if (data.avgLoss !== undefined) updateData.avg_loss = data.avgLoss;
      if (data.largestWin !== undefined) updateData.largest_win = data.largestWin;
      if (data.largestLoss !== undefined) updateData.largest_loss = data.largestLoss;
      if (data.consecutiveWins !== undefined) updateData.consecutive_wins = data.consecutiveWins;
      if (data.consecutiveLosses !== undefined) updateData.consecutive_losses = data.consecutiveLosses;
      if (data.avgTradeDuration !== undefined) updateData.avg_trade_duration = data.avgTradeDuration;
      if (data.totalCommissions !== undefined) updateData.total_commissions = data.totalCommissions;
      if (data.var95 !== undefined) updateData.var_95 = data.var95;
      if (data.var99 !== undefined) updateData.var_99 = data.var99;
      if (data.expectedShortfall !== undefined) updateData.expected_shortfall = data.expectedShortfall;
      if (data.trackingError !== undefined) updateData.tracking_error = data.trackingError;
      if (data.informationRatio !== undefined) updateData.information_ratio = data.informationRatio;
      if (data.downsideDeviation !== undefined) updateData.downside_deviation = data.downsideDeviation;
      if (data.detailedMetrics !== undefined) updateData.detailed_metrics = data.detailedMetrics;
      if (data.equityCurve !== undefined) updateData.equity_curve = data.equityCurve;
      if (data.drawdownPeriods !== undefined) updateData.drawdown_periods = data.drawdownPeriods;
      if (data.monthlyReturns !== undefined) updateData.monthly_returns = data.monthlyReturns;
      if (data.errorMessage !== undefined) updateData.error_message = data.errorMessage;

      const { error } = await this.supabase
        .from('backtest_results')
        .update(updateData)
        .eq('id', backtestId);

      if (error) throw error;
    } catch (error) {
      logger.error('Error updating backtest result:', error);
      throw error;
    }
  }

  async getBacktestResults(userId: string, options: {
    strategyId?: string;
    status?: string;
    limit?: number;
    offset?: number;
    sortBy?: string;
    sortOrder?: 'asc' | 'desc';
  } = {}): Promise<any[]> {
    try {
      let query = this.supabase
        .from('backtest_results')
        .select(`
          *,
          strategy_definitions(name, description),
          backtest_configs(name, symbols, start_date, end_date)
        `)
        .eq('user_id', userId);

      if (options.strategyId) {
        query = query.eq('strategy_id', options.strategyId);
      }

      if (options.status) {
        query = query.eq('status', options.status);
      }

      if (options.limit) {
        query = query.limit(options.limit);
      }

      if (options.offset) {
        query = query.range(options.offset, (options.offset + (options.limit || 20)) - 1);
      }

      const sortField = options.sortBy || 'created_at';
      const sortOrder = options.sortOrder || 'desc';
      query = query.order(sortField, { ascending: sortOrder === 'asc' });

      const { data, error } = await query;
      if (error) throw error;
      return data || [];
    } catch (error) {
      logger.error('Error fetching backtest results:', error);
      throw error;
    }
  }

  async getBacktestResult(
    userId: string,
    backtestId: string,
    options: { includeTrades?: boolean } = {}
  ): Promise<any | null> {
    try {
      let query = this.supabase
        .from('backtest_results')
        .select(`
          *,
          strategy_definitions(name, description, parameters),
          backtest_configs(*)
        `)
        .eq('user_id', userId)
        .eq('id', backtestId)
        .single();

      const { data: result, error } = await query;
      if (error && error.code !== 'PGRST116') throw error;

      if (result && options.includeTrades) {
        const { data: trades, error: tradesError } = await this.supabase
          .from('backtest_trades')
          .select('*')
          .eq('backtest_result_id', backtestId)
          .order('entry_time', { ascending: true });

        if (tradesError) throw tradesError;
        result.trades = trades || [];
      }

      return result;
    } catch (error) {
      logger.error('Error fetching backtest result:', error);
      throw error;
    }
  }

  // Backtest Trades

  async saveBacktestTrades(userId: string, backtestResultId: string, trades: any[]): Promise<void> {
    try {
      const tradeData = trades.map((trade, index) => ({
        backtest_result_id: backtestResultId,
        user_id: userId,
        trade_number: index + 1,
        symbol: trade.symbol,
        side: trade.side,
        entry_price: trade.entryPrice,
        exit_price: trade.exitPrice,
        quantity: trade.quantity,
        entry_time: trade.entryTime,
        exit_time: trade.exitTime,
        pnl: trade.pnl,
        pnl_percent: trade.pnlPercent,
        commission: trade.commission || 0,
        slippage: trade.slippage || 0,
        hold_duration: trade.holdDuration,
        entry_reason: trade.reason,
        exit_reason: trade.exitReason,
        confidence: trade.confidence,
        indicators: trade.indicators || {}
      }));

      const { error } = await this.supabase
        .from('backtest_trades')
        .insert(tradeData);

      if (error) throw error;
    } catch (error) {
      logger.error('Error saving backtest trades:', error);
      throw error;
    }
  }

  // Strategy Optimization

  async createOptimization(userId: string, data: OptimizationData): Promise<any> {
    try {
      const { data: optimization, error } = await this.supabase
        .from('strategy_optimizations')
        .insert({
          user_id: userId,
          strategy_id: data.strategyId,
          config_id: data.configId,
          external_optimization_id: data.externalOptimizationId,
          optimization_method: data.optimizationMethod,
          objective_function: data.objectiveFunction,
          parameters_to_optimize: data.parametersToOptimize,
          iterations: data.iterations,
          status: data.status,
          progress: data.progress,
          best_parameters: data.bestParameters,
          best_performance: data.bestPerformance,
          best_metrics: data.bestMetrics,
          optimization_results: data.optimizationResults,
          parameter_importance: data.parameterImportance,
          error_message: data.errorMessage,
          started_at: data.startedAt?.toISOString() || new Date().toISOString(),
          completed_at: data.completedAt?.toISOString()
        })
        .select()
        .single();

      if (error) throw error;
      return optimization;
    } catch (error) {
      logger.error('Error creating optimization:', error);
      throw error;
    }
  }

  // Validation Results

  async saveValidationResults(userId: string, strategyId: string, report: any): Promise<void> {
    try {
      const { error } = await this.supabase
        .from('strategy_validations')
        .insert({
          strategy_id: strategyId,
          user_id: userId,
          validation_type: 'comprehensive',
          status: report.status,
          score: report.overallScore / 100,
          errors: report.errors,
          warnings: report.warnings,
          recommendations: report.recommendations,
          min_sharpe_required: report.performanceValidation?.minSharpeRequired,
          actual_sharpe: report.performanceValidation?.actualSharpe,
          min_win_rate_required: report.performanceValidation?.minWinRateRequired,
          actual_win_rate: report.performanceValidation?.actualWinRate,
          max_drawdown_allowed: report.performanceValidation?.maxDrawdownAllowed,
          actual_max_drawdown: report.performanceValidation?.actualMaxDrawdown,
          expires_at: new Date(Date.now() + 30 * 24 * 60 * 60 * 1000).toISOString() // 30 days
        });

      if (error) throw error;
    } catch (error) {
      logger.error('Error saving validation results:', error);
      throw error;
    }
  }

  // Adaptive Threshold Training

  async saveAdaptiveThresholdTraining(userId: string, trainingData: any[]): Promise<void> {
    try {
      const insertData = trainingData.map(data => ({
        user_id: userId,
        strategy_id: data.strategyId,
        backtest_result_id: data.backtestResultId,
        training_period_start: data.trainingPeriodStart.toISOString(),
        training_period_end: data.trainingPeriodEnd.toISOString(),
        market_conditions: data.marketConditions,
        strategy_parameters: data.strategyParameters,
        threshold_values: data.thresholdValues,
        performance_score: data.performanceScore,
        sharpe_ratio: data.sharpeRatio,
        max_drawdown: data.maxDrawdown,
        win_rate: data.winRate,
        data_quality_score: data.dataQualityScore,
        regime_type: data.regimeType,
        volatility_bucket: data.volatilityBucket
      }));

      const { error } = await this.supabase
        .from('adaptive_threshold_training')
        .insert(insertData);

      if (error) throw error;
    } catch (error) {
      logger.error('Error saving adaptive threshold training data:', error);
      throw error;
    }
  }
}