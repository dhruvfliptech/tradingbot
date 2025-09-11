import { describe, it, expect, beforeAll, afterAll, beforeEach } from '@jest/globals';
import request from 'supertest';
import { ComposerService } from '../../services/composer/ComposerService';
import { StrategyValidationService } from '../../services/validation/StrategyValidationService';
import { ComposerAdaptiveIntegration } from '../../services/integration/ComposerAdaptiveIntegration';
import { PerformanceMetricsService } from '../../services/metrics/PerformanceMetricsService';
import { DatabaseService } from '../../services/database/DatabaseService';
import { createTestApp } from '../utils/testApp';

describe('Backtesting Integration Tests', () => {
  let app: any;
  let composerService: ComposerService;
  let validationService: StrategyValidationService;
  let adaptiveIntegration: ComposerAdaptiveIntegration;
  let metricsService: PerformanceMetricsService;
  let databaseService: DatabaseService;
  let testUserId: string;
  let authToken: string;

  beforeAll(async () => {
    // Initialize test environment
    app = await createTestApp();
    composerService = new ComposerService(process.env.TEST_COMPOSER_MCP_URL);
    validationService = new StrategyValidationService(composerService);
    adaptiveIntegration = new ComposerAdaptiveIntegration(composerService);
    metricsService = new PerformanceMetricsService();
    databaseService = new DatabaseService();

    // Create test user and get auth token
    const authResponse = await request(app)
      .post('/api/v1/auth/register')
      .send({
        email: 'test@example.com',
        password: 'testpass123',
        username: 'testuser'
      });

    testUserId = authResponse.body.data.user.id;
    authToken = authResponse.body.data.token;

    // Connect to Composer MCP (if available)
    try {
      await composerService.connect();
    } catch (error) {
      console.warn('Composer MCP not available for testing, using mock responses');
    }
  });

  afterAll(async () => {
    // Cleanup test data
    await cleanupTestData();
    composerService.disconnect();
  });

  beforeEach(async () => {
    // Reset any test state before each test
  });

  describe('Strategy Management', () => {
    it('should create a new strategy definition', async () => {
      const strategyData = {
        name: 'Test Momentum Strategy',
        description: 'A test strategy for momentum trading',
        strategy_type: 'momentum',
        version: '1.0.0',
        parameters: [
          {
            name: 'rsi_threshold',
            type: 'number',
            defaultValue: 70,
            min: 50,
            max: 90,
            description: 'RSI overbought threshold'
          },
          {
            name: 'confidence_threshold',
            type: 'number',
            defaultValue: 0.75,
            min: 0.5,
            max: 0.95,
            description: 'Minimum confidence for trades'
          }
        ],
        entry_rules: [
          'RSI < rsi_threshold',
          'confidence > confidence_threshold'
        ],
        exit_rules: [
          'RSI > 80',
          'stop_loss_hit'
        ],
        risk_management: [
          {
            type: 'stop_loss',
            parameter: 'stop_loss_percent',
            value: 0.05,
            condition: 'less_than'
          }
        ]
      };

      const response = await request(app)
        .post('/api/v1/backtesting/strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .send(strategyData)
        .expect(201);

      expect(response.body.success).toBe(true);
      expect(response.body.data.id).toBeDefined();
      expect(response.body.data.name).toBe(strategyData.name);
      expect(response.body.data.validation).toBeDefined();
    });

    it('should retrieve user strategies', async () => {
      const response = await request(app)
        .get('/api/v1/backtesting/strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(Array.isArray(response.body.data)).toBe(true);
      expect(response.body.data.length).toBeGreaterThan(0);
    });

    it('should validate a strategy comprehensively', async () => {
      // First create a strategy
      const strategyResponse = await request(app)
        .post('/api/v1/backtesting/strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Test Validation Strategy',
          description: 'Strategy for validation testing',
          strategy_type: 'momentum',
          parameters: [
            {
              name: 'rsi_threshold',
              type: 'number',
              defaultValue: 70,
              min: 50,
              max: 90,
              description: 'RSI threshold'
            }
          ],
          entry_rules: ['RSI < rsi_threshold'],
          exit_rules: ['RSI > 80'],
          risk_management: [
            {
              type: 'stop_loss',
              parameter: 'stop_loss_percent',
              value: 0.05,
              condition: 'less_than'
            }
          ]
        });

      const strategyId = strategyResponse.body.data.id;

      const validationResponse = await request(app)
        .post(`/api/v1/backtesting/strategies/${strategyId}/validate`)
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          symbols: ['BTC/USD', 'ETH/USD'],
          period: {
            start: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
            end: new Date().toISOString()
          }
        })
        .expect(200);

      expect(validationResponse.body.success).toBe(true);
      expect(validationResponse.body.data.strategyId).toBe(strategyId);
      expect(validationResponse.body.data.overallScore).toBeDefined();
      expect(validationResponse.body.data.status).toMatch(/passed|failed|warning/);
    });
  });

  describe('Backtesting Execution', () => {
    let strategyId: string;

    beforeEach(async () => {
      // Create a test strategy for backtesting
      const strategyResponse = await request(app)
        .post('/api/v1/backtesting/strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Backtest Strategy',
          description: 'Strategy for backtest testing',
          strategy_type: 'momentum',
          parameters: [
            {
              name: 'rsi_threshold',
              type: 'number',
              defaultValue: 70,
              min: 50,
              max: 90,
              description: 'RSI threshold'
            }
          ],
          entry_rules: ['RSI < rsi_threshold'],
          exit_rules: ['RSI > 80'],
          risk_management: []
        });

      strategyId = strategyResponse.body.data.id;
    });

    it('should run a backtest successfully', async () => {
      const backtestConfig = {
        strategyId,
        symbols: ['BTC/USD'],
        startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
        endDate: new Date().toISOString(),
        initialCapital: 10000,
        parameters: {
          rsi_threshold: 70
        },
        riskSettings: {
          maxPositionSize: 0.1,
          stopLoss: 0.05,
          takeProfit: 0.15
        }
      };

      const response = await request(app)
        .post('/api/v1/backtesting/run')
        .set('Authorization', `Bearer ${authToken}`)
        .send(backtestConfig)
        .expect(202);

      expect(response.body.success).toBe(true);
      expect(response.body.data.backtestId).toBeDefined();
      expect(response.body.data.externalId).toBeDefined();
      expect(response.body.data.status).toBe('running');
    });

    it('should retrieve backtest results', async () => {
      const response = await request(app)
        .get('/api/v1/backtesting/results')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(Array.isArray(response.body.data)).toBe(true);
    });

    it('should get specific backtest result with trades', async () => {
      // First run a backtest
      const backtestResponse = await request(app)
        .post('/api/v1/backtesting/run')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          strategyId,
          symbols: ['BTC/USD'],
          startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          endDate: new Date().toISOString(),
          initialCapital: 10000,
          parameters: { rsi_threshold: 70 },
          riskSettings: { maxPositionSize: 0.1, stopLoss: 0.05, takeProfit: 0.15 }
        });

      const backtestId = backtestResponse.body.data.backtestId;

      // Wait a moment for the backtest to be created
      await new Promise(resolve => setTimeout(resolve, 1000));

      const response = await request(app)
        .get(`/api/v1/backtesting/results/${backtestId}?include_trades=true`)
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data.id).toBe(backtestId);
    });
  });

  describe('Strategy Optimization', () => {
    let strategyId: string;

    beforeEach(async () => {
      const strategyResponse = await request(app)
        .post('/api/v1/backtesting/strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Optimization Strategy',
          description: 'Strategy for optimization testing',
          strategy_type: 'momentum',
          parameters: [
            {
              name: 'rsi_threshold',
              type: 'number',
              defaultValue: 70,
              min: 50,
              max: 90,
              description: 'RSI threshold'
            },
            {
              name: 'confidence_threshold',
              type: 'number',
              defaultValue: 0.75,
              min: 0.5,
              max: 0.95,
              description: 'Confidence threshold'
            }
          ],
          entry_rules: ['RSI < rsi_threshold'],
          exit_rules: ['RSI > 80'],
          risk_management: []
        });

      strategyId = strategyResponse.body.data.id;
    });

    it('should optimize strategy parameters', async () => {
      const optimizationConfig = {
        strategy_id: strategyId,
        config: {
          symbols: ['BTC/USD'],
          startDate: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          endDate: new Date().toISOString(),
          initialCapital: 10000
        },
        optimization_params: {
          method: 'grid',
          objectiveFunction: 'sharpe',
          parameters: ['rsi_threshold', 'confidence_threshold'],
          iterations: 10
        }
      };

      const response = await request(app)
        .post('/api/v1/backtesting/optimize')
        .set('Authorization', `Bearer ${authToken}`)
        .send(optimizationConfig)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data.optimizationId).toBeDefined();
      expect(response.body.data.bestParameters).toBeDefined();
      expect(response.body.data.bestPerformance).toBeDefined();
    });
  });

  describe('Adaptive Threshold Integration', () => {
    let strategyId: string;

    beforeEach(async () => {
      const strategyResponse = await request(app)
        .post('/api/v1/backtesting/strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          name: 'Adaptive Strategy',
          description: 'Strategy for adaptive threshold testing',
          strategy_type: 'ml_based',
          parameters: [
            {
              name: 'rsi_threshold',
              type: 'number',
              defaultValue: 70,
              min: 50,
              max: 90,
              description: 'RSI threshold'
            }
          ],
          entry_rules: ['RSI < rsi_threshold'],
          exit_rules: ['RSI > 80'],
          risk_management: []
        });

      strategyId = strategyResponse.body.data.id;
    });

    it('should train adaptive thresholds from backtest data', async () => {
      // First create some mock backtest results
      const mockBacktestResults = [
        {
          id: 'mock_result_1',
          sharpe_ratio: 1.5,
          max_drawdown: 0.1,
          win_rate: 0.6,
          profit_factor: 1.8,
          volatility: 0.2,
          strategy_parameters: { rsi_threshold: 70 },
          threshold_values: { rsi_threshold: 70 }
        }
      ];

      const response = await request(app)
        .post('/api/v1/backtesting/adaptive-threshold/train')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          strategy_id: strategyId,
          backtest_results: mockBacktestResults
        })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data.trainingDataPoints).toBeGreaterThan(0);
      expect(response.body.data.currentThresholds).toBeDefined();
    });
  });

  describe('Historical Data and Market Regimes', () => {
    it('should fetch historical data via Composer MCP', async () => {
      const response = await request(app)
        .get('/api/v1/backtesting/historical-data')
        .set('Authorization', `Bearer ${authToken}`)
        .query({
          symbols: ['BTC/USD'],
          start_date: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          end_date: new Date().toISOString(),
          timeframe: '1h'
        })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(Array.isArray(response.body.data)).toBe(true);
    });

    it('should analyze market regimes', async () => {
      const response = await request(app)
        .get('/api/v1/backtesting/market-regimes')
        .set('Authorization', `Bearer ${authToken}`)
        .query({
          symbols: ['BTC/USD'],
          start_date: new Date(Date.now() - 30 * 24 * 60 * 60 * 1000).toISOString(),
          end_date: new Date().toISOString()
        })
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(response.body.data.regimes).toBeDefined();
      expect(Array.isArray(response.body.data.regimes)).toBe(true);
    });

    it('should get available strategy templates', async () => {
      const response = await request(app)
        .get('/api/v1/backtesting/available-strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .expect(200);

      expect(response.body.success).toBe(true);
      expect(Array.isArray(response.body.data)).toBe(true);
    });
  });

  describe('Performance Metrics', () => {
    it('should calculate comprehensive performance metrics', async () => {
      // Create mock backtest result with trades
      const mockBacktestResult = {
        id: 'test_backtest',
        strategyId: 'test_strategy',
        trades: [
          {
            id: '1',
            symbol: 'BTC/USD',
            side: 'buy',
            quantity: 0.1,
            entryPrice: 50000,
            exitPrice: 52000,
            entryTime: new Date('2023-01-01T10:00:00Z'),
            exitTime: new Date('2023-01-01T14:00:00Z'),
            pnl: 200,
            pnlPercent: 4,
            commission: 10,
            reason: 'RSI oversold',
            confidence: 0.8
          },
          {
            id: '2',
            symbol: 'BTC/USD',
            side: 'sell',
            quantity: 0.1,
            entryPrice: 52000,
            exitPrice: 51000,
            entryTime: new Date('2023-01-02T10:00:00Z'),
            exitTime: new Date('2023-01-02T12:00:00Z'),
            pnl: -100,
            pnlPercent: -1.9,
            commission: 10,
            reason: 'Stop loss hit',
            confidence: 0.6
          }
        ],
        performance: {
          totalReturn: 0.02,
          sharpeRatio: 1.2,
          maxDrawdown: 0.05,
          winRate: 0.5,
          profitFactor: 2.0,
          sortino: 1.5,
          calmar: 0.4,
          volatility: 0.15
        },
        metrics: {
          totalTrades: 2,
          winningTrades: 1,
          losingTrades: 1,
          avgWin: 200,
          avgLoss: 100,
          largestWin: 200,
          largestLoss: -100,
          consecutiveWins: 1,
          consecutiveLosses: 1,
          avgTradeDuration: 4,
          totalCommissions: 20
        },
        riskAnalysis: {
          var95: -0.03,
          var99: -0.05,
          expectedShortfall: -0.04,
          beta: 1.1,
          alpha: 0.02,
          trackingError: 0.08,
          informationRatio: 0.25,
          downsideDeviation: 0.12
        },
        status: 'completed',
        startedAt: new Date('2023-01-01T00:00:00Z'),
        completedAt: new Date('2023-01-03T00:00:00Z')
      };

      const metrics = await metricsService.calculatePerformanceMetrics(mockBacktestResult as any);

      expect(metrics.totalTrades).toBe(2);
      expect(metrics.winRate).toBe(0.5);
      expect(metrics.profitFactor).toBeGreaterThan(0);
      expect(metrics.sharpeRatio).toBeDefined();
      expect(metrics.maxDrawdown).toBeDefined();
      expect(metrics.monthlyReturns).toBeDefined();
    });

    it('should generate comprehensive performance report', async () => {
      const mockBacktestResult = {
        trades: [
          {
            id: '1',
            symbol: 'BTC/USD',
            side: 'buy',
            quantity: 0.1,
            entryPrice: 50000,
            exitPrice: 52000,
            entryTime: new Date('2023-01-01T10:00:00Z'),
            exitTime: new Date('2023-01-01T14:00:00Z'),
            pnl: 200,
            pnlPercent: 4,
            commission: 10,
            reason: 'RSI oversold',
            confidence: 0.8
          }
        ]
      };

      const report = await metricsService.generatePerformanceReport(mockBacktestResult as any);

      expect(report.summary).toBeDefined();
      expect(report.equityCurve).toBeDefined();
      expect(report.drawdownPeriods).toBeDefined();
      expect(report.tradeAnalysis).toBeDefined();
      expect(report.riskAnalysis).toBeDefined();
      expect(Array.isArray(report.equityCurve)).toBe(true);
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid strategy creation', async () => {
      const invalidStrategy = {
        name: '', // Empty name should cause validation error
        description: 'Invalid strategy',
        parameters: []
      };

      const response = await request(app)
        .post('/api/v1/backtesting/strategies')
        .set('Authorization', `Bearer ${authToken}`)
        .send(invalidStrategy)
        .expect(400);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toContain('validation failed');
    });

    it('should handle backtest with non-existent strategy', async () => {
      const response = await request(app)
        .post('/api/v1/backtesting/run')
        .set('Authorization', `Bearer ${authToken}`)
        .send({
          strategyId: 'non-existent-strategy-id',
          symbols: ['BTC/USD'],
          startDate: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString(),
          endDate: new Date().toISOString(),
          initialCapital: 10000,
          parameters: {},
          riskSettings: {}
        })
        .expect(404);

      expect(response.body.success).toBe(false);
      expect(response.body.message).toContain('Strategy not found');
    });

    it('should handle unauthorized requests', async () => {
      const response = await request(app)
        .get('/api/v1/backtesting/strategies')
        .expect(401);

      expect(response.body.success).toBe(false);
    });
  });

  // Helper function to cleanup test data
  async function cleanupTestData() {
    try {
      // Clean up test user data
      if (testUserId) {
        // Delete user strategies, backtests, etc.
        // This would typically be done through the database service
        console.log('Cleaning up test data...');
      }
    } catch (error) {
      console.error('Error cleaning up test data:', error);
    }
  }
});