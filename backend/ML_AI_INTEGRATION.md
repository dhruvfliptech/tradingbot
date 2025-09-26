# ML/AI Integration Architecture

## Overview
The ML/AI integration uses an **Event-Driven Pipeline** architecture that connects the Trading Engine with ML (Adaptive Threshold) and RL (Reinforcement Learning) services through Redis Streams.

## Architecture Diagram
```
┌─────────────────┐     Redis Streams      ┌─────────────────┐
│                 │  ──────────────────►    │                 │
│  Trading Engine │                         │   ML Service    │
│                 │  ◄──────────────────    │ (Adaptive)      │
└─────────────────┘         Events          └─────────────────┘
         │                                            │
         │                                            │
         ▼                                            ▼
┌─────────────────┐                        ┌─────────────────┐
│  Event Bus      │◄───────────────────────│  RL Service     │
│  (Redis)        │                        │ (Decision)      │
└─────────────────┘                        └─────────────────┘
```

## Event Flow

### 1. Market Data Flow
```
Trading Engine → [MARKET_DATA_UPDATE] → ML Service & RL Service
```
- Trading engine publishes real-time market data
- ML service adjusts thresholds based on market conditions
- RL service generates trading signals

### 2. Signal Generation Flow
```
RL Service → [RL_SIGNAL_GENERATED] → Trading Engine
ML Service → [THRESHOLD_ADJUSTED] → Trading Engine
```
- RL service publishes trading signals with confidence scores
- ML service publishes threshold adjustments
- Trading engine combines signals for execution

### 3. Feedback Loop
```
Trading Engine → [POSITION_CLOSED] → ML & RL Services
Trading Engine → [PREDICTION_OUTCOME] → ML & RL Services
```
- Closed positions trigger learning updates
- Prediction outcomes improve model accuracy

## Components

### 1. MLEventBus (`src/services/ml/MLEventBus.ts`)
- Manages Redis Streams connections
- Handles event publishing and consumption
- Implements circuit breakers for fault tolerance
- Features:
  - Auto-reconnection
  - Event batching
  - Dead letter queues
  - Performance monitoring

### 2. TradingEngineMLIntegration (`src/services/ml/TradingEngineMLIntegration.ts`)
- Integrates Trading Engine with ML pipeline
- Publishes market events
- Subscribes to ML/RL signals
- Manages prediction tracking for feedback

### 3. ML Service Event Integration (`ml-service/event_integration.py`)
- Subscribes to market and feedback events
- Publishes threshold adjustments
- Manages adaptive learning
- Features:
  - Performance-based adaptation
  - Risk limit adjustments
  - Conservative mode on poor performance

### 4. RL Service Event Integration (`rl-service/integration/event_integration.py`)
- Subscribes to market data
- Generates trading signals
- Processes feedback for learning
- Features:
  - State buffering
  - Observation preparation
  - Confidence scoring

## Event Types

### Market Events (Published by Trading Engine)
- `market.data.update` - Real-time price and indicator data
- `trade.executed` - Trade execution details
- `position.opened` - New position information
- `position.closed` - Closed position with P&L
- `performance.update` - Periodic performance metrics

### ML Service Events
- `ml.threshold.adjusted` - Threshold parameter changes
- `ml.risk.limit.update` - Risk management adjustments
- `ml.prediction.ready` - ML model predictions

### RL Service Events
- `rl.signal.generated` - Trading signals with confidence
- `rl.confidence.update` - Model confidence changes
- `rl.action.recommended` - Specific trade recommendations

### Feedback Events
- `feedback.prediction.outcome` - Actual vs predicted results
- `feedback.performance` - Overall performance metrics

## Configuration

### Redis Streams Configuration
```typescript
{
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD,
    db: parseInt(process.env.REDIS_DB || '0')
  },
  streams: {
    maxLen: 10000,        // Max events per stream
    blockTimeout: 5000,   // Read timeout in ms
    batchSize: 10        // Events per batch
  },
  consumerGroup: 'ml-pipeline',
  consumerId: 'trading-engine-{pid}'
}
```

### ML Service Configuration
```python
{
  'adaptation': {
    'min_trades_for_adaptation': 10,
    'adaptation_interval': 3600,  # 1 hour
    'confidence_threshold': 0.6
  },
  'risk': {
    'max_position_size': 0.02,
    'stop_loss_multiplier': 1.5
  }
}
```

### RL Service Configuration
```python
{
  'prediction': {
    'confidence_threshold': 0.6,
    'timeout_seconds': 2.0,
    'max_concurrent': 50
  },
  'exploration': {
    'initial_rate': 0.1,
    'min_rate': 0.01,
    'decay_rate': 0.995
  }
}
```

## Setup Instructions

### 1. Start Redis
```bash
# Using Docker
docker run -d --name redis-ml \
  -p 6379:6379 \
  redis:7-alpine redis-server --appendonly yes

# Or using local Redis
redis-server --appendonly yes --maxmemory 2gb
```

### 2. Start ML Service
```bash
cd backend/ml-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python app.py
```

### 3. Start RL Service
```bash
cd backend/rl-service
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python integration/decision_server.py
```

### 4. Enable ML Integration in Trading Engine
```typescript
// In TradingEngineService.ts
import { getTradingEngineMLIntegration } from '../ml/TradingEngineMLIntegration';

class TradingEngineService {
  private mlIntegration: TradingEngineMLIntegration;

  async initialize() {
    // ... existing initialization

    // Start ML integration
    this.mlIntegration = getTradingEngineMLIntegration();

    // Subscribe to ML/RL signals
    this.mlIntegration.on('mlSignalGenerated', (signal) => {
      this.processMLSignal(signal);
    });

    this.mlIntegration.on('rlSignalGenerated', (signal) => {
      this.processRLSignal(signal);
    });
  }

  async executeTradingCycle(userId: string) {
    // ... existing cycle logic

    // Publish market data to ML pipeline
    await this.mlIntegration.publishMarketData(
      userId,
      marketData,
      indicators
    );

    // ... rest of cycle
  }
}
```

## Monitoring

### Key Metrics
1. **Event Processing**
   - Events published/consumed per minute
   - Event processing latency
   - Queue depths

2. **ML Performance**
   - Threshold adaptation frequency
   - Prediction accuracy
   - Risk limit adjustments

3. **RL Performance**
   - Signal generation rate
   - Confidence scores
   - Action distribution

4. **System Health**
   - Redis memory usage
   - Circuit breaker states
   - Service availability

### Monitoring Endpoints
```bash
# Trading Engine ML Status
GET http://localhost:3001/api/v1/ml/status

# ML Service Health
GET http://localhost:5001/health

# RL Service Status
GET http://localhost:8000/api/status
```

## Troubleshooting

### Common Issues

1. **Events Not Being Processed**
   - Check Redis connection: `redis-cli ping`
   - Verify consumer groups: `redis-cli xinfo groups ml:stream:market`
   - Check service logs for errors

2. **High Latency**
   - Monitor Redis memory: `redis-cli info memory`
   - Check stream lengths: `redis-cli xlen ml:stream:market`
   - Verify network connectivity

3. **Circuit Breaker Open**
   - Check service health endpoints
   - Review error logs
   - Wait for automatic reset (30 seconds)

### Debug Commands
```bash
# View pending messages
redis-cli xpending ml:stream:market ml-pipeline

# Read stream messages
redis-cli xread count 10 streams ml:stream:market 0

# Monitor events in real-time
redis-cli --csv psubscribe 'ml:stream:*'

# Check consumer group info
redis-cli xinfo consumers ml:stream:market ml-pipeline
```

## Performance Optimization

### 1. Batching
- Process multiple events together
- Reduces network overhead
- Improves throughput

### 2. Caching
- Cache ML predictions for 60 seconds
- Store recent market states in memory
- Reduce redundant calculations

### 3. Async Processing
- Non-blocking event publishing
- Parallel signal processing
- Background feedback loops

### 4. Resource Limits
- Max 10,000 events per stream
- Auto-trim old events
- Connection pooling for Redis

## Security Considerations

1. **Event Validation**
   - Validate event schemas
   - Sanitize user inputs
   - Check authorization

2. **Rate Limiting**
   - Limit events per user
   - Throttle signal generation
   - Prevent spam

3. **Data Privacy**
   - Encrypt sensitive data
   - Separate user data streams
   - Audit event access

## Future Enhancements

1. **Multi-Model Ensemble**
   - Add more ML models
   - Weighted voting system
   - A/B testing framework

2. **Advanced RL Features**
   - Multi-agent collaboration
   - Transfer learning
   - Meta-learning

3. **Stream Processing**
   - Apache Kafka integration
   - Real-time analytics
   - Event sourcing

4. **Monitoring**
   - Grafana dashboards
   - Prometheus metrics
   - Alert automation

## Testing

### Integration Test
```bash
# Run integration tests
cd backend
npm run test:ml-integration

# Python tests
cd ml-service
pytest test_event_integration.py

cd ../rl-service
pytest integration/test_event_integration.py
```

### Load Testing
```bash
# Simulate high event load
node scripts/ml-load-test.js --events=1000 --duration=60
```

## Conclusion

The event-driven ML/AI integration provides:
- **Scalability**: Handle thousands of events per second
- **Reliability**: Circuit breakers and fallbacks
- **Flexibility**: Easy to add new models
- **Performance**: Async processing with minimal latency
- **Learning**: Continuous improvement through feedback

This architecture ensures that ML/AI models can influence trading decisions in real-time while maintaining system stability and performance.