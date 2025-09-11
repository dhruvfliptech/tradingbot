# Data Aggregation Service

A comprehensive data aggregation system that integrates multiple free APIs for on-chain analytics, funding rates, and whale alerts to enrich trading signals and provide market context.

## Overview

The Data Aggregation Service provides:

- **Unified API Integration**: Seamlessly integrates 6 different data sources
- **Smart Rate Limiting**: Maximizes free tier usage with intelligent rate limiting
- **Intelligent Caching**: Reduces API calls and improves performance
- **Data Normalization**: Consistent data formats across all sources
- **Error Handling**: Robust fallback strategies and circuit breakers
- **Real-time Enrichment**: Enhances trading signals with multi-source data
- **Market Context**: Provides comprehensive market analysis

## Supported APIs

### 1. Etherscan API
- **Purpose**: Ethereum on-chain data, transactions, token balances
- **Free Tier**: 5 calls/second
- **Data**: Whale transactions, smart contract interactions, token movements

### 2. Bitquery GraphQL
- **Purpose**: DEX trades, smart money flows, multi-chain analytics  
- **Free Tier**: Limited queries per day
- **Data**: DEX transactions, whale movements, smart money tracking

### 3. Covalent API
- **Purpose**: Multi-chain portfolio data, token balances
- **Free Tier**: 100k credits/month
- **Data**: Cross-chain balances, DeFi positions, NFT holdings

### 4. Coinglass API
- **Purpose**: Derivatives data, funding rates, liquidations
- **Free Tier**: Limited calls
- **Data**: Funding rates, open interest, liquidation data

### 5. Binance Public API
- **Purpose**: Market data, funding rates, trading metrics
- **Free Tier**: 1200 requests/minute (no auth required)
- **Data**: Real-time prices, funding rates, order book data

### 6. CryptoQuant API
- **Purpose**: On-chain metrics, exchange flows
- **Free Tier**: Very limited
- **Data**: Exchange flows, miner metrics, network indicators

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                 DataAggregatorService                       │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Rate Limiter  │  │  Cache Service  │  │ Error Handler│ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
├─────────────────────────────────────────────────────────────┤
│           ┌─────────────────────────────────────┐           │
│           │        Data Normalizer              │           │
│           └─────────────────────────────────────┘           │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌───────────┐ │
│  │ Etherscan  │ │  Bitquery  │ │  Covalent  │ │ Coinglass │ │
│  │   Client   │ │   Client   │ │   Client   │ │  Client   │ │
│  └────────────┘ └────────────┘ └────────────┘ └───────────┘ │
│  ┌────────────┐ ┌────────────┐                             │
│  │  Binance   │ │CryptoQuant │                             │
│  │   Client   │ │   Client   │                             │
│  └────────────┘ └────────────┘                             │
└─────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Integration Layer                              │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────┐  ┌─────────────────────────────────┐ │
│  │ AdaptiveThreshold   │  │      Composer Service          │ │
│  │    Integration      │  │      Integration               │ │
│  └─────────────────────┘  └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Environment Setup

Copy the environment template:
```bash
cp .env.data-aggregator.example .env
```

Configure your API keys:
```env
# Required API Keys
ETHERSCAN_API_KEY=your_etherscan_api_key
COVALENT_API_KEY=your_covalent_api_key

# Optional API Keys  
BITQUERY_API_KEY=your_bitquery_api_key
CRYPTOQUANT_API_KEY=your_cryptoquant_api_key
```

### 2. Basic Usage

```typescript
import { createDataAggregatorIntegration } from './services/data-aggregation';

// Initialize the service
const dataAggregator = await createDataAggregatorIntegration();

// Enrich a trading signal
const signal = {
  symbol: 'BTC',
  confidence: 75,
  action: 'BUY',
  indicators: { rsi: 30 }
};

const enrichment = await dataAggregator.enrichTradingSignal(signal);
console.log('Enriched signal:', enrichment);

// Get market context
const context = await dataAggregator.getMarketContext(['BTC', 'ETH']);
console.log('Market context:', context);
```

### 3. Integration with Trading Service

```typescript
// In your TradingService
import { DataAggregatorIntegration } from './services/data-aggregation';

class TradingService {
  private dataAggregator: DataAggregatorIntegration;

  async initialize() {
    this.dataAggregator = new DataAggregatorIntegration();
    await this.dataAggregator.initialize();
  }

  async processSignal(signal: any) {
    // Enrich signal with multi-source data
    const enrichment = await this.dataAggregator.enrichTradingSignal(signal);
    
    // Apply enrichment to signal confidence
    const enrichedConfidence = signal.confidence + enrichment.confidence_boost;
    
    // Use enriched signal for trading decisions
    return {
      ...signal,
      confidence: enrichedConfidence,
      enrichment
    };
  }
}
```

## Features

### Multi-Source Data Aggregation

```typescript
const aggregatedData = await dataAggregator.aggregateData(['BTC'], {
  includeOnchain: true,      // Etherscan, Covalent data
  includeFunding: true,      // Coinglass, Binance funding rates
  includeWhales: true,       // Large transaction detection
  includeSmartMoney: true,   // Smart money flow analysis
  includeLiquidations: true  // Liquidation pressure data
});
```

### Trading Signal Enrichment

```typescript
const enrichment = await dataAggregator.enrichTradingSignal(signal);

// Enrichment includes:
// - onchain_activity: Recent on-chain activity score
// - whale_sentiment: Whale flow direction
// - funding_bias: Funding rate bias indicator  
// - liquidation_risk: Current liquidation pressure
// - smart_money_flow: Smart money sentiment
// - confidence_boost: Suggested confidence adjustment
```

### Market Context Analysis

```typescript
const context = await dataAggregator.getMarketContext(['BTC', 'ETH']);

// Context includes:
// - overall_sentiment: Aggregated market sentiment
// - risk_level: Current market risk assessment
// - market_regime: Bull/bear/sideways/volatile
// - funding_rate_bias: Cross-exchange funding bias
// - whale_activity_score: Recent whale activity level
```

### Intelligent Caching

The service implements intelligent caching with different TTLs for different data types:

- **On-chain data**: 5 minutes (slow-changing)
- **Funding rates**: 1 minute (hourly updates)
- **Whale alerts**: 30 seconds (real-time priority)
- **Market data**: 1 minute (frequent updates)

### Rate Limiting

Smart rate limiting maximizes free tier usage:

- **Etherscan**: 5 calls/second
- **Bitquery**: 2 calls/second (conservative)
- **Covalent**: 10 calls/second
- **Coinglass**: 5 calls/second
- **Binance**: 20 calls/second
- **CryptoQuant**: 1 call/second

### Error Handling & Fallbacks

Robust error handling with multiple fallback strategies:

1. **Retry Logic**: Exponential backoff with jitter
2. **Circuit Breakers**: Auto-disable failing services
3. **Fallback APIs**: Alternative data sources
4. **Cache Fallbacks**: Stale data when APIs fail
5. **Default Responses**: Safe defaults for critical paths

## Integration Points

### AdaptiveThreshold ML Service

```typescript
// Prepare enriched input for ML service
const adaptiveInput = await dataAggregator.prepareAdaptiveThresholdInput(signal);

// Send to ML service for threshold adjustment
const mlResponse = await fetch(`${ML_SERVICE_URL}/api/v1/evaluate/${userId}`, {
  method: 'POST',
  body: JSON.stringify(adaptiveInput)
});
```

### Composer Backtesting Service

```typescript
// Enrich backtest with historical data
const enrichment = await dataAggregator.enrichBacktestData(
  ['BTC', 'ETH'], 
  startDate, 
  endDate
);

// Use enrichment in backtest analysis
const backtestConfig = {
  ...baseConfig,
  enrichment
};
```

## Configuration

### Environment Variables

All configuration is done through environment variables:

```env
# API Keys
ETHERSCAN_API_KEY=required
COVALENT_API_KEY=required
BITQUERY_API_KEY=optional
CRYPTOQUANT_API_KEY=optional

# Rate Limits (requests per second)
ETHERSCAN_RATE_LIMIT=5
COVALENT_RATE_LIMIT=10
BINANCE_RATE_LIMIT=20

# Cache Settings (seconds)
CACHE_TTL_ONCHAIN=300
CACHE_TTL_FUNDING=60
CACHE_TTL_WHALE=30
CACHE_MAX_SIZE=10000

# Feature Toggles
ENABLE_ETHERSCAN=true
ENABLE_COVALENT=true
ENABLE_BITQUERY=true
```

### Programmatic Configuration

```typescript
import { DataAggregatorConfigManager } from './config/DataAggregatorConfig';

const configManager = new DataAggregatorConfigManager();

// Update API configuration
configManager.updateApiConfig('etherscan', {
  rateLimit: 10,  // Upgrade to pro plan
  enabled: true
});

// Update cache settings
configManager.updateCacheConfig({
  maxSize: 50000,
  ttl: {
    onchain: 600  // 10 minutes
  }
});
```

## Monitoring & Health

### Health Checks

```typescript
const health = await dataAggregator.getHealthStatus();

console.log(health);
// {
//   status: 'healthy',
//   services: [
//     { name: 'etherscan', status: 'up', latency: 150 },
//     { name: 'binance', status: 'up', latency: 80 }
//   ],
//   cache: {
//     hitRate: 85.2,
//     size: 2840
//   },
//   rateLimits: [
//     { service: 'etherscan', remaining: 4, resetTime: '...' }
//   ]
// }
```

### Event Monitoring

```typescript
dataAggregator.on('data_aggregated', (data) => {
  console.log(`Aggregated data from ${data.sources.length} sources`);
});

dataAggregator.on('health_alert', (alert) => {
  if (alert.level === 'critical') {
    // Send notification
  }
});

dataAggregator.on('performance_alert', (alert) => {
  console.log(`Performance issue: ${alert.message}`);
});
```

## Free Tier Optimization

### Maximizing API Usage

The service is optimized for free tier usage:

1. **Intelligent Caching**: Aggressive caching reduces API calls
2. **Rate Limiting**: Respects free tier limits
3. **Fallback Strategies**: Uses alternative APIs when limits hit
4. **Data Prioritization**: Focuses on high-value data points
5. **Batch Processing**: Combines multiple requests when possible

### Cost Monitoring

```typescript
// Monitor API usage
const stats = await dataAggregator.getHealthStatus();

stats.services.forEach(service => {
  console.log(`${service.name}: ${service.requestsToday} requests today`);
});

// Cache efficiency
console.log(`Cache hit rate: ${stats.cache.hitRate}%`);
```

## Testing

Run the comprehensive test suite:

```bash
npm test data-aggregation
```

The test suite includes:

- **Unit Tests**: Individual component testing
- **Integration Tests**: Multi-component workflows  
- **Performance Tests**: Load and concurrency testing
- **Error Handling Tests**: Failure scenario validation
- **End-to-End Tests**: Complete workflow testing

## Production Deployment

### Recommended Production Settings

```env
# Production environment
NODE_ENV=production

# Upgraded API plans
ETHERSCAN_RATE_LIMIT=20     # Pro plan
COVALENT_RATE_LIMIT=50      # Higher tier
ENABLE_CRYPTOQUANT=true     # With paid plan

# Production cache settings
CACHE_MAX_SIZE=100000
CACHE_TTL_ONCHAIN=300

# Enhanced monitoring
ENABLE_METRICS=true
METRICS_INTERVAL=30000
LOG_LEVEL=info
```

### Production Checklist

- [ ] Upgrade to paid API plans for higher limits
- [ ] Set up Redis for distributed caching
- [ ] Configure monitoring and alerting
- [ ] Implement database storage for historical data
- [ ] Set up API key rotation
- [ ] Configure load balancing for high availability
- [ ] Set up backup data sources
- [ ] Implement comprehensive logging

## Troubleshooting

### Common Issues

**API Rate Limit Exceeded**
```
Error: Rate limit exceeded for etherscan
Solution: Reduce rate limit or upgrade API plan
```

**Circuit Breaker Open**
```
Error: Circuit breaker is open for service: bitquery
Solution: Wait for auto-reset or manually reset circuit breaker
```

**Cache Performance Issues**
```
Warning: Low cache hit rate: 25%
Solution: Increase cache TTL or cache size
```

### Debug Mode

Enable debug logging:
```env
LOG_LEVEL=debug
```

Monitor service health:
```typescript
const health = await dataAggregator.getHealthStatus();
console.log('Service health:', health);
```

## Contributing

1. Follow the existing code structure
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Test with free tier API limits
5. Ensure backward compatibility

## License

This service is part of the trading bot project and follows the same license terms.