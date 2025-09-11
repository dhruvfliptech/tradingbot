import { EventEmitter } from 'events';
import logger from '../../../utils/logger';
import { DataAggregatorService, AggregatedData } from '../DataAggregatorService';
import { DataAggregatorConfigManager } from '../config/DataAggregatorConfig';
import { DataAggregatorErrorHandler } from '../ErrorHandler';

export interface TradingSignalEnrichment {
  onchain_activity: number;
  whale_sentiment: number;
  funding_bias: number;
  liquidation_risk: number;
  smart_money_flow: number;
  market_regime: 'bull' | 'bear' | 'sideways' | 'volatile';
  confidence_boost: number;
  data_sources: string[];
  reliability: number;
}

export interface MarketContext {
  overall_sentiment: number;
  risk_level: 'low' | 'medium' | 'high';
  trending_symbols: string[];
  market_regime: 'bull' | 'bear' | 'sideways' | 'volatile';
  funding_rate_bias: number;
  whale_activity_score: number;
  liquidation_pressure: number;
  smart_money_sentiment: number;
}

export interface AdaptiveThresholdInput {
  signal: {
    symbol: string;
    confidence: number;
    rsi?: number;
    change_percent?: number;
    volume?: number;
    action: string;
  };
  enrichment?: TradingSignalEnrichment;
  market_context?: MarketContext;
}

export interface ComposerBacktestEnrichment {
  whale_activity: Array<{
    timestamp: Date;
    net_flow: number;
    large_transactions: number;
  }>;
  funding_rates: Array<{
    timestamp: Date;
    rate: number;
    bias: number;
  }>;
  smart_money: Array<{
    timestamp: Date;
    sentiment: number;
    flow_score: number;
  }>;
  liquidations: Array<{
    timestamp: Date;
    long_ratio: number;
    short_ratio: number;
    pressure_score: number;
  }>;
}

export class DataAggregatorIntegration extends EventEmitter {
  private dataAggregator: DataAggregatorService;
  private configManager: DataAggregatorConfigManager;
  private errorHandler: DataAggregatorErrorHandler;
  private isInitialized: boolean = false;
  private enrichmentCache: Map<string, { data: any; timestamp: number }> = new Map();
  private readonly ENRICHMENT_CACHE_TTL = 60000; // 1 minute

  constructor() {
    super();
    this.configManager = new DataAggregatorConfigManager();
    this.errorHandler = new DataAggregatorErrorHandler();
  }

  async initialize(): Promise<void> {
    try {
      logger.info('Initializing DataAggregator integration...');

      const config = this.configManager.getConfiguration();
      this.dataAggregator = new DataAggregatorService(config);
      
      await this.dataAggregator.initialize();
      
      // Set up event listeners
      this.setupEventListeners();
      
      this.isInitialized = true;
      logger.info('DataAggregator integration initialized successfully');

    } catch (error) {
      logger.error('Failed to initialize DataAggregator integration:', error);
      throw error;
    }
  }

  // Integration with AdaptiveThreshold ML Service
  async enrichTradingSignal(signal: any): Promise<TradingSignalEnrichment> {
    if (!this.isInitialized) {
      throw new Error('DataAggregator integration not initialized');
    }

    const cacheKey = `signal_${signal.symbol}_${signal.confidence}_${Math.floor(Date.now() / this.ENRICHMENT_CACHE_TTL)}`;
    const cached = this.enrichmentCache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < this.ENRICHMENT_CACHE_TTL) {
      return cached.data;
    }

    try {
      const enrichment = await this.errorHandler.executeWithFallback(
        async () => {
          const aggregatedData = await this.dataAggregator.aggregateData([signal.symbol], {
            includeFunding: true,
            includeWhales: true,
            includeLiquidations: true,
            includeSmartMoney: true
          });

          return this.calculateEnrichment(signal, aggregatedData);
        },
        {
          service: 'data-aggregator',
          method: 'enrichTradingSignal',
          symbol: signal.symbol,
          maxAttempts: 3
        }
      );

      // Cache the result
      this.enrichmentCache.set(cacheKey, {
        data: enrichment,
        timestamp: Date.now()
      });

      // Emit enrichment event
      this.emit('signal_enriched', {
        symbol: signal.symbol,
        originalConfidence: signal.confidence,
        enrichedConfidence: signal.confidence + enrichment.confidence_boost,
        enrichment
      });

      return enrichment;

    } catch (error) {
      logger.error(`Failed to enrich trading signal for ${signal.symbol}:`, error);
      
      // Return default enrichment on failure
      return this.getDefaultEnrichment();
    }
  }

  // Integration with Composer Service for backtesting
  async enrichBacktestData(symbols: string[], startDate: Date, endDate: Date): Promise<ComposerBacktestEnrichment> {
    if (!this.isInitialized) {
      throw new Error('DataAggregator integration not initialized');
    }

    try {
      return await this.errorHandler.executeWithFallback(
        async () => {
          // Get historical data for the backtest period
          const enrichmentData: ComposerBacktestEnrichment = {
            whale_activity: [],
            funding_rates: [],
            smart_money: [],
            liquidations: []
          };

          // Generate time series data for the backtest period
          const timePoints = this.generateTimePoints(startDate, endDate, 'hour');

          for (const timePoint of timePoints) {
            // Get data for each time point
            const aggregatedData = await this.dataAggregator.aggregateData(symbols, {
              includeFunding: true,
              includeWhales: true,
              includeLiquidations: true,
              includeSmartMoney: true
            });

            // Process whale activity
            if (aggregatedData.whales?.length) {
              const whaleMetrics = this.calculateWhaleMetrics(aggregatedData.whales, timePoint);
              enrichmentData.whale_activity.push(whaleMetrics);
            }

            // Process funding rates
            if (aggregatedData.funding?.length) {
              const fundingMetrics = this.calculateFundingMetrics(aggregatedData.funding, timePoint);
              enrichmentData.funding_rates.push(fundingMetrics);
            }

            // Process smart money flows
            if (aggregatedData.smartMoney?.length) {
              const smartMoneyMetrics = this.calculateSmartMoneyMetrics(aggregatedData.smartMoney, timePoint);
              enrichmentData.smart_money.push(smartMoneyMetrics);
            }

            // Process liquidations
            if (aggregatedData.liquidations?.length) {
              const liquidationMetrics = this.calculateLiquidationMetrics(aggregatedData.liquidations, timePoint);
              enrichmentData.liquidations.push(liquidationMetrics);
            }
          }

          return enrichmentData;
        },
        {
          service: 'data-aggregator',
          method: 'enrichBacktestData',
          maxAttempts: 2
        }
      );

    } catch (error) {
      logger.error('Failed to enrich backtest data:', error);
      
      // Return empty enrichment data on failure
      return {
        whale_activity: [],
        funding_rates: [],
        smart_money: [],
        liquidations: []
      };
    }
  }

  async getMarketContext(symbols: string[]): Promise<MarketContext> {
    if (!this.isInitialized) {
      throw new Error('DataAggregator integration not initialized');
    }

    const cacheKey = `market_context_${symbols.sort().join(',')}_${Math.floor(Date.now() / 60000)}`;
    const cached = this.enrichmentCache.get(cacheKey);
    
    if (cached && Date.now() - cached.timestamp < 60000) {
      return cached.data;
    }

    try {
      const marketContext = await this.errorHandler.executeWithFallback(
        async () => {
          const aggregatedData = await this.dataAggregator.aggregateData(symbols, {
            includeFunding: true,
            includeWhales: true,
            includeLiquidations: true,
            includeSmartMoney: true
          });

          return this.calculateMarketContext(aggregatedData, symbols);
        },
        {
          service: 'data-aggregator',
          method: 'getMarketContext',
          maxAttempts: 3
        }
      );

      // Cache the result
      this.enrichmentCache.set(cacheKey, {
        data: marketContext,
        timestamp: Date.now()
      });

      // Emit market context event
      this.emit('market_context_updated', marketContext);

      return marketContext;

    } catch (error) {
      logger.error('Failed to get market context:', error);
      
      // Return default market context on failure
      return this.getDefaultMarketContext();
    }
  }

  // Integration helpers for AdaptiveThreshold
  async prepareAdaptiveThresholdInput(signal: any): Promise<AdaptiveThresholdInput> {
    const enrichment = await this.enrichTradingSignal(signal);
    const marketContext = await this.getMarketContext([signal.symbol]);

    return {
      signal: {
        symbol: signal.symbol,
        confidence: signal.confidence,
        rsi: signal.indicators?.rsi,
        change_percent: signal.changePercent,
        volume: signal.volume,
        action: signal.action
      },
      enrichment,
      market_context: marketContext
    };
  }

  // Event handling and monitoring
  private setupEventListeners(): void {
    this.dataAggregator.on('data_aggregated', (data) => {
      this.emit('data_aggregated', data);
    });

    // Monitor health and emit alerts
    setInterval(() => {
      this.checkHealthAndEmitAlerts();
    }, 60000); // Every minute
  }

  private async checkHealthAndEmitAlerts(): Promise<void> {
    try {
      const health = await this.dataAggregator.getHealthStatus();
      
      if (health.status === 'unhealthy') {
        this.emit('health_alert', {
          level: 'critical',
          message: 'DataAggregator service is unhealthy',
          services: health.services.filter(s => s.status === 'down'),
          timestamp: new Date()
        });
      } else if (health.status === 'degraded') {
        this.emit('health_alert', {
          level: 'warning',
          message: 'DataAggregator service is degraded',
          services: health.services.filter(s => s.status === 'down'),
          timestamp: new Date()
        });
      }

      // Check cache hit rate
      if (health.cache.hitRate < 30) {
        this.emit('performance_alert', {
          level: 'warning',
          message: `Low cache hit rate: ${health.cache.hitRate}%`,
          metric: 'cache_hit_rate',
          value: health.cache.hitRate,
          timestamp: new Date()
        });
      }

    } catch (error) {
      logger.error('Health check failed:', error);
    }
  }

  // Calculation methods
  private calculateEnrichment(signal: any, aggregatedData: AggregatedData): TradingSignalEnrichment {
    const enrichment: TradingSignalEnrichment = {
      onchain_activity: 0,
      whale_sentiment: 0,
      funding_bias: 0,
      liquidation_risk: 0,
      smart_money_flow: 0,
      market_regime: 'sideways',
      confidence_boost: 0,
      data_sources: aggregatedData.metadata.sources,
      reliability: aggregatedData.metadata.reliability
    };

    // Calculate on-chain activity score
    if (aggregatedData.onchain?.length) {
      const recentActivity = aggregatedData.onchain.filter(
        data => new Date().getTime() - data.lastActivity.getTime() < 24 * 60 * 60 * 1000
      );
      enrichment.onchain_activity = Math.min(100, recentActivity.length * 20);
    }

    // Calculate whale sentiment
    if (aggregatedData.whales?.length) {
      const recentWhales = aggregatedData.whales.filter(
        whale => new Date().getTime() - whale.timestamp.getTime() < 4 * 60 * 60 * 1000
      );
      
      const inflowValue = recentWhales.filter(w => w.type === 'exchange_inflow').reduce((sum, w) => sum + w.amountUsd, 0);
      const outflowValue = recentWhales.filter(w => w.type === 'exchange_outflow').reduce((sum, w) => sum + w.amountUsd, 0);
      
      if (inflowValue + outflowValue > 0) {
        enrichment.whale_sentiment = ((outflowValue - inflowValue) / (inflowValue + outflowValue)) * 100;
      }
    }

    // Calculate funding bias
    if (aggregatedData.funding?.length) {
      const avgFunding = aggregatedData.funding.reduce((sum, f) => sum + f.rate, 0) / aggregatedData.funding.length;
      enrichment.funding_bias = Math.max(-100, Math.min(100, avgFunding * 10000));
    }

    // Calculate liquidation risk
    if (aggregatedData.liquidations?.length) {
      const recentLiqs = aggregatedData.liquidations.filter(
        liq => new Date().getTime() - liq.timestamp.getTime() < 60 * 60 * 1000
      );
      const totalLiquidations = recentLiqs.reduce((sum, l) => sum + l.amountUsd, 0);
      enrichment.liquidation_risk = Math.min(100, totalLiquidations / 1000000);
    }

    // Calculate smart money flow
    if (aggregatedData.smartMoney?.length) {
      const recentFlows = aggregatedData.smartMoney.filter(
        flow => new Date().getTime() - flow.timestamp.getTime() < 2 * 60 * 60 * 1000
      );
      
      const buyFlow = recentFlows.filter(f => f.action === 'buy').reduce((sum, f) => sum + f.amountUsd, 0);
      const sellFlow = recentFlows.filter(f => f.action === 'sell').reduce((sum, f) => sum + f.amountUsd, 0);
      
      if (buyFlow + sellFlow > 0) {
        enrichment.smart_money_flow = ((buyFlow - sellFlow) / (buyFlow + sellFlow)) * 100;
      }
    }

    // Determine market regime
    enrichment.market_regime = this.determineMarketRegime(enrichment);

    // Calculate confidence boost
    enrichment.confidence_boost = this.calculateConfidenceBoost(signal, enrichment);

    return enrichment;
  }

  private calculateMarketContext(aggregatedData: AggregatedData, symbols: string[]): MarketContext {
    const context: MarketContext = {
      overall_sentiment: 0,
      risk_level: 'medium',
      trending_symbols: symbols,
      market_regime: 'sideways',
      funding_rate_bias: 0,
      whale_activity_score: 0,
      liquidation_pressure: 0,
      smart_money_sentiment: 0
    };

    // Calculate funding rate bias
    if (aggregatedData.funding?.length) {
      const avgFunding = aggregatedData.funding.reduce((sum, f) => sum + f.rate, 0) / aggregatedData.funding.length;
      context.funding_rate_bias = avgFunding;
    }

    // Calculate whale activity score
    if (aggregatedData.whales?.length) {
      const totalWhaleValue = aggregatedData.whales.reduce((sum, w) => sum + w.amountUsd, 0);
      context.whale_activity_score = Math.min(100, totalWhaleValue / 10000000); // Scale by $10M
    }

    // Calculate liquidation pressure
    if (aggregatedData.liquidations?.length) {
      const totalLiquidations = aggregatedData.liquidations.reduce((sum, l) => sum + l.amountUsd, 0);
      context.liquidation_pressure = Math.min(100, totalLiquidations / 1000000); // Scale by $1M
    }

    // Calculate smart money sentiment
    if (aggregatedData.smartMoney?.length) {
      const buyFlow = aggregatedData.smartMoney.filter(f => f.action === 'buy').reduce((sum, f) => sum + f.amountUsd, 0);
      const sellFlow = aggregatedData.smartMoney.filter(f => f.action === 'sell').reduce((sum, f) => sum + f.amountUsd, 0);
      
      if (buyFlow + sellFlow > 0) {
        context.smart_money_sentiment = ((buyFlow - sellFlow) / (buyFlow + sellFlow)) * 100;
      }
    }

    // Calculate overall sentiment
    context.overall_sentiment = (
      context.funding_rate_bias * -100 + // Negative funding is bullish
      context.whale_activity_score * 0.5 +
      context.smart_money_sentiment * 0.8 -
      context.liquidation_pressure * 0.3
    ) / 4;

    // Determine risk level
    if (context.liquidation_pressure > 60 || Math.abs(context.funding_rate_bias) > 0.01) {
      context.risk_level = 'high';
    } else if (context.liquidation_pressure > 30 || Math.abs(context.funding_rate_bias) > 0.005) {
      context.risk_level = 'medium';
    } else {
      context.risk_level = 'low';
    }

    // Determine market regime
    if (Math.abs(context.overall_sentiment) > 40) {
      context.market_regime = context.overall_sentiment > 0 ? 'bull' : 'bear';
    } else if (context.liquidation_pressure > 50) {
      context.market_regime = 'volatile';
    } else {
      context.market_regime = 'sideways';
    }

    return context;
  }

  private determineMarketRegime(enrichment: TradingSignalEnrichment): 'bull' | 'bear' | 'sideways' | 'volatile' {
    const sentiment = (enrichment.whale_sentiment + enrichment.smart_money_flow - enrichment.funding_bias) / 3;
    
    if (enrichment.liquidation_risk > 50) return 'volatile';
    if (sentiment > 30) return 'bull';
    if (sentiment < -30) return 'bear';
    return 'sideways';
  }

  private calculateConfidenceBoost(signal: any, enrichment: TradingSignalEnrichment): number {
    let boost = 0;

    // Positive boost for aligned signals
    if (signal.action === 'BUY') {
      if (enrichment.whale_sentiment > 20) boost += 5;
      if (enrichment.smart_money_flow > 20) boost += 8;
      if (enrichment.funding_bias < -20) boost += 3; // Negative funding is bullish
    } else if (signal.action === 'SELL') {
      if (enrichment.whale_sentiment < -20) boost += 5;
      if (enrichment.smart_money_flow < -20) boost += 8;
      if (enrichment.funding_bias > 20) boost += 3; // Positive funding is bearish
    }

    // Risk adjustments
    if (enrichment.liquidation_risk > 60) boost -= 10;
    if (enrichment.reliability < 50) boost -= 5;

    // Data quality boost
    if (enrichment.data_sources.length >= 4) boost += 3;
    if (enrichment.reliability > 80) boost += 5;

    return Math.max(-15, Math.min(15, boost));
  }

  // Helper methods for backtest enrichment
  private generateTimePoints(startDate: Date, endDate: Date, interval: 'hour' | 'day'): Date[] {
    const points: Date[] = [];
    const intervalMs = interval === 'hour' ? 60 * 60 * 1000 : 24 * 60 * 60 * 1000;
    
    let current = new Date(startDate);
    while (current <= endDate) {
      points.push(new Date(current));
      current = new Date(current.getTime() + intervalMs);
    }
    
    return points;
  }

  private calculateWhaleMetrics(whales: any[], timePoint: Date): any {
    const recentWhales = whales.filter(w => 
      Math.abs(w.timestamp.getTime() - timePoint.getTime()) < 60 * 60 * 1000
    );

    const inflowValue = recentWhales.filter(w => w.type === 'exchange_inflow').reduce((sum, w) => sum + w.amountUsd, 0);
    const outflowValue = recentWhales.filter(w => w.type === 'exchange_outflow').reduce((sum, w) => sum + w.amountUsd, 0);

    return {
      timestamp: timePoint,
      net_flow: outflowValue - inflowValue,
      large_transactions: recentWhales.length
    };
  }

  private calculateFundingMetrics(funding: any[], timePoint: Date): any {
    const avgRate = funding.reduce((sum, f) => sum + f.rate, 0) / funding.length;
    
    return {
      timestamp: timePoint,
      rate: avgRate,
      bias: avgRate > 0 ? 'bearish' : 'bullish'
    };
  }

  private calculateSmartMoneyMetrics(smartMoney: any[], timePoint: Date): any {
    const recentFlows = smartMoney.filter(f => 
      Math.abs(f.timestamp.getTime() - timePoint.getTime()) < 60 * 60 * 1000
    );

    const buyFlow = recentFlows.filter(f => f.action === 'buy').reduce((sum, f) => sum + f.amountUsd, 0);
    const sellFlow = recentFlows.filter(f => f.action === 'sell').reduce((sum, f) => sum + f.amountUsd, 0);
    
    const sentiment = buyFlow + sellFlow > 0 ? ((buyFlow - sellFlow) / (buyFlow + sellFlow)) * 100 : 0;

    return {
      timestamp: timePoint,
      sentiment,
      flow_score: Math.min(100, (buyFlow + sellFlow) / 1000000)
    };
  }

  private calculateLiquidationMetrics(liquidations: any[], timePoint: Date): any {
    const recentLiqs = liquidations.filter(l => 
      Math.abs(l.timestamp.getTime() - timePoint.getTime()) < 60 * 60 * 1000
    );

    const longLiqs = recentLiqs.filter(l => l.side === 'long').reduce((sum, l) => sum + l.amountUsd, 0);
    const shortLiqs = recentLiqs.filter(l => l.side === 'short').reduce((sum, l) => sum + l.amountUsd, 0);
    const totalLiqs = longLiqs + shortLiqs;

    return {
      timestamp: timePoint,
      long_ratio: totalLiqs > 0 ? (longLiqs / totalLiqs) * 100 : 50,
      short_ratio: totalLiqs > 0 ? (shortLiqs / totalLiqs) * 100 : 50,
      pressure_score: Math.min(100, totalLiqs / 1000000)
    };
  }

  // Default fallback methods
  private getDefaultEnrichment(): TradingSignalEnrichment {
    return {
      onchain_activity: 0,
      whale_sentiment: 0,
      funding_bias: 0,
      liquidation_risk: 0,
      smart_money_flow: 0,
      market_regime: 'sideways',
      confidence_boost: 0,
      data_sources: [],
      reliability: 0
    };
  }

  private getDefaultMarketContext(): MarketContext {
    return {
      overall_sentiment: 0,
      risk_level: 'medium',
      trending_symbols: [],
      market_regime: 'sideways',
      funding_rate_bias: 0,
      whale_activity_score: 0,
      liquidation_pressure: 0,
      smart_money_sentiment: 0
    };
  }

  // Public methods for service management
  async getHealthStatus(): Promise<any> {
    if (!this.isInitialized) {
      return { status: 'not_initialized' };
    }

    const aggregatorHealth = await this.dataAggregator.getHealthStatus();
    const errorHandlerHealth = this.errorHandler.getHealthStatus();
    
    return {
      aggregator: aggregatorHealth,
      errorHandler: errorHandlerHealth,
      integration: {
        initialized: this.isInitialized,
        cacheSize: this.enrichmentCache.size
      }
    };
  }

  async shutdown(): Promise<void> {
    logger.info('Shutting down DataAggregator integration...');
    
    if (this.dataAggregator) {
      await this.dataAggregator.shutdown();
    }
    
    this.enrichmentCache.clear();
    this.removeAllListeners();
    this.isInitialized = false;
    
    logger.info('DataAggregator integration shutdown complete');
  }
}