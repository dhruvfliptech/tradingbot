# Composer MCP Integration for Backtesting

This document outlines the integration of Composer MCP (Model Context Protocol) for advanced backtesting capabilities in our trading bot backend.

## Overview

The Composer MCP integration provides:
- Historical data management (saves 40+ hours of development)
- Advanced backtesting engine
- Strategy validation pipeline  
- Performance metrics extraction
- Market regime analysis
- Adaptive threshold training

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API    │    │  Composer MCP   │
│                 │    │                  │    │                 │
│ Strategy Editor │◄──►│ BacktestingCtrl  │◄──►│ Backtest Engine │
│ Results View    │    │ ValidationSvc    │    │ Historical Data │
│ Performance     │    │ MetricsSvc       │    │ Market Analysis │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                               │
                               ▼
                       ┌──────────────────┐
                       │  AdaptiveML      │
                       │  Service         │
                       │                  │
                       │ Threshold Adapt  │
                       │ Strategy Learning│
                       └──────────────────┘
```

## Core Components

### 1. ComposerService (`/services/composer/ComposerService.ts`)

**Purpose**: WebSocket and HTTP client for Composer MCP server

**Key Features**:
- WebSocket connection management with auto-reconnect
- Backtest execution and monitoring
- Historical data retrieval
- Strategy validation
- Market regime analysis
- Parameter optimization

**API Methods**:
```typescript
// Backtest Management
async runBacktest(config: BacktestConfig): Promise<string>
async cancelBacktest(backtestId: string): Promise<void>
async getBacktestStatus(backtestId: string): Promise<BacktestResult>

// Historical Data
async getHistoricalData(symbols: string[], startDate: string, endDate: string): Promise<HistoricalData[]>

// Strategy Operations
async validateStrategy(strategy: StrategyDefinition): Promise<ValidationResult>
async optimizeStrategy(strategyId: string, config: BacktestConfig, params: OptimizationParams): Promise<OptimizationResult>

// Market Analysis
async getMarketRegimes(symbols: string[], startDate: string, endDate: string): Promise<RegimeAnalysis>
```

### 2. StrategyValidationService (`/services/validation/StrategyValidationService.ts`)

**Purpose**: Comprehensive strategy validation pipeline

**Validation Rules**:
- **Syntax Validation**: Parameter completeness, rule definitions
- **Logic Validation**: Entry/exit rule consistency, risk management
- **Performance Validation**: Backtest performance metrics
- **Overfitting Detection**: Cross-validation and consistency checks
- **Market Fit Analysis**: Performance across different market regimes

**Validation Process**:
1. Syntax and logic checks
2. Run backtests for performance validation
3. Analyze results for overfitting
4. Generate comprehensive validation report

### 3. ComposerAdaptiveIntegration (`/services/integration/ComposerAdaptiveIntegration.ts`)

**Purpose**: Integrates Composer MCP with AdaptiveThreshold ML system

**Training Process**:
1. **Multi-Period Training**: Run backtests across different time periods
2. **Regime-Specific Analysis**: Identify optimal thresholds for bull/bear/sideways markets
3. **Parameter Optimization**: Use genetic/grid/Bayesian optimization
4. **Validation**: Test on out-of-sample data
5. **Threshold Adaptation**: Update ML model with best parameters

**Key Methods**:
```typescript
async runAdaptiveThresholdTraining(config: AdaptiveTrainingConfig): Promise<AdaptiveTrainingResult>
```

### 4. PerformanceMetricsService (`/services/metrics/PerformanceMetricsService.ts`)

**Purpose**: Extract and calculate comprehensive performance metrics

**Metrics Calculated**:
- **Return Metrics**: Total return, annualized return, CAGR
- **Risk Metrics**: Sharpe ratio, Sortino ratio, Calmar ratio, max drawdown
- **Trade Metrics**: Win rate, profit factor, average win/loss
- **Advanced Metrics**: VaR, expected shortfall, omega ratio, kurtosis
- **Time Analysis**: Monthly/yearly returns, drawdown periods
- **Regime Performance**: Performance by market condition

### 5. BacktestingController (`/controllers/BacktestingController.ts`)

**Purpose**: RESTful API endpoints for backtesting operations

**Key Endpoints**:
```
POST   /api/v1/backtesting/strategies              # Create strategy
GET    /api/v1/backtesting/strategies              # List strategies
POST   /api/v1/backtesting/strategies/:id/validate # Validate strategy
POST   /api/v1/backtesting/run                     # Run backtest
GET    /api/v1/backtesting/results                 # Get results
POST   /api/v1/backtesting/optimize                # Optimize strategy
POST   /api/v1/backtesting/adaptive-threshold/train # Train adaptive thresholds
```

## Database Schema

### Core Tables

**strategy_definitions**: Strategy definitions with parameters and rules
**backtest_configs**: Backtest configuration settings
**backtest_results**: Performance metrics and results
**backtest_trades**: Individual trade records
**strategy_optimizations**: Parameter optimization results
**adaptive_threshold_training**: ML training data

### Performance Tables

**historical_data_cache**: Cached market data with compression
**market_regimes**: Market condition analysis
**strategy_regime_performance**: Performance by market regime
**strategy_validations**: Validation results and scores

## Usage Examples

### 1. Create and Validate Strategy

```javascript
// Create strategy
const strategy = await fetch('/api/v1/backtesting/strategies', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    name: 'RSI Momentum Strategy',
    description: 'Momentum strategy using RSI and volume',
    strategy_type: 'momentum',
    parameters: [
      {
        name: 'rsi_threshold',
        type: 'number',
        defaultValue: 70,
        min: 50,
        max: 90,
        description: 'RSI overbought threshold'
      }
    ],
    entry_rules: ['RSI < rsi_threshold', 'volume > volume_threshold'],
    exit_rules: ['RSI > 80', 'stop_loss_hit'],
    risk_management: [
      { type: 'stop_loss', parameter: 'stop_loss_percent', value: 0.05 }
    ]
  })
});

// Validate strategy
const validation = await fetch(`/api/v1/backtesting/strategies/${strategyId}/validate`, {
  method: 'POST',
  body: JSON.stringify({
    symbols: ['BTC/USD', 'ETH/USD'],
    period: {
      start: '2023-01-01T00:00:00Z',
      end: '2023-12-31T23:59:59Z'
    }
  })
});
```

### 2. Run Backtest

```javascript
const backtest = await fetch('/api/v1/backtesting/run', {
  method: 'POST',
  body: JSON.stringify({
    strategyId: 'strategy-uuid',
    symbols: ['BTC/USD', 'ETH/USD'],
    startDate: '2023-01-01T00:00:00Z',
    endDate: '2023-12-31T23:59:59Z',
    initialCapital: 10000,
    parameters: {
      rsi_threshold: 70,
      confidence_threshold: 0.75
    },
    riskSettings: {
      maxPositionSize: 0.1,
      stopLoss: 0.05,
      takeProfit: 0.15
    }
  })
});

// Monitor progress
const result = await fetch(`/api/v1/backtesting/results/${backtestId}?include_trades=true`);
```

### 3. Train Adaptive Thresholds

```javascript
const training = await fetch('/api/v1/backtesting/adaptive-threshold/train', {
  method: 'POST',
  body: JSON.stringify({
    strategy_id: 'strategy-uuid',
    backtest_results: [
      // Array of backtest result objects
    ]
  })
});
```

## Environment Configuration

### Required Environment Variables

```bash
# Composer MCP Configuration
COMPOSER_MCP_URL=https://ai.composer.trade/mcp

# Database
DATABASE_URL=postgresql://user:pass@host:5432/tradingbot
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key

# ML Service
ML_SERVICE_URL=http://ml-service:5000
```

### Docker Compose Updates

```yaml
services:
  backend:
    environment:
      - COMPOSER_MCP_URL=${COMPOSER_MCP_URL:-https://ai.composer.trade/mcp}
      - ML_SERVICE_URL=http://ml-service:5000
    networks:
      - trading-network

  ml-service:
    environment:
      - COMPOSER_MCP_URL=${COMPOSER_MCP_URL:-https://ai.composer.trade/mcp}
    networks:
      - trading-network

networks:
  trading-network:
    driver: bridge
```

## Performance Benefits

### Historical Data Management
- **Before**: 40+ hours to implement OHLCV data collection, cleaning, and storage
- **After**: Direct access to clean, validated historical data via Composer MCP
- **Savings**: Immediate access to multi-exchange, multi-timeframe data

### Strategy Validation
- **Before**: Manual backtesting with limited validation
- **After**: Comprehensive validation pipeline with overfitting detection
- **Benefits**: Higher confidence in strategy performance

### Adaptive Learning
- **Before**: Static trading thresholds
- **After**: Dynamic thresholds that adapt to market conditions
- **Benefits**: Improved performance across different market regimes

### Performance Analytics
- **Before**: Basic P&L tracking
- **After**: Institutional-grade performance analytics
- **Benefits**: Deep insights into strategy behavior and risk characteristics

## Integration Testing

Run the comprehensive test suite:

```bash
# Install dependencies
npm install

# Run integration tests
npm run test:integration

# Run specific test suite
npm run test -- --testNamePattern="Backtesting Integration"
```

### Test Coverage

- Strategy creation and validation
- Backtest execution and monitoring
- Performance metrics calculation
- Adaptive threshold training
- Error handling and edge cases
- WebSocket connection management

## Monitoring and Observability

### Key Metrics to Monitor

- **Composer MCP Connection**: WebSocket connectivity and response times
- **Backtest Performance**: Execution time, success rate, error frequency
- **Database Performance**: Query latency for large result sets
- **ML Training**: Training completion rate, accuracy improvements

### Logging

All services include structured logging with correlation IDs:

```typescript
logger.info('Starting backtest', {
  backtestId: 'bt_123',
  strategyId: 'strategy_456',
  symbols: ['BTC/USD'],
  timeRange: '2023-01-01 to 2023-12-31'
});
```

## Security Considerations

### Data Protection
- All backtest data is scoped to individual users via RLS
- API keys for Composer MCP stored as environment variables
- WebSocket connections use secure protocols (WSS)

### Rate Limiting
- Composer MCP requests are rate-limited to prevent abuse
- Backtest execution is queued to manage system resources
- Historical data requests include caching to reduce external calls

## Troubleshooting

### Common Issues

1. **Composer MCP Connection Failures**
   - Check COMPOSER_MCP_URL environment variable
   - Verify network connectivity
   - Check WebSocket connection logs

2. **Backtest Timeouts**
   - Reduce date range for initial tests
   - Check Composer MCP service status
   - Verify strategy parameter validity

3. **Performance Issues**
   - Enable database query logging
   - Monitor memory usage during large backtests
   - Check InfluxDB performance for time-series data

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=debug
export DEBUG_COMPOSER_MCP=true
```

## Future Enhancements

### Planned Features

1. **Real-time Strategy Monitoring**: Live performance tracking during paper trading
2. **Multi-Strategy Portfolios**: Backtest multiple strategies with allocation
3. **Advanced Optimization**: Monte Carlo simulation and walk-forward analysis
4. **Risk Management**: Portfolio-level risk constraints and correlation limits
5. **Custom Indicators**: User-defined technical indicators integration

### API Versioning

The API is designed for backward compatibility with versioned endpoints:
- `/api/v1/backtesting/*` - Current stable API
- `/api/v2/backtesting/*` - Future enhanced API with breaking changes

---

For additional support or questions about the Composer MCP integration, please refer to the main project documentation or contact the development team.