# AdaptiveThreshold ML Service

A comprehensive machine learning service that implements adaptive threshold management for crypto trading bots. This service provides real-time parameter adjustment based on performance feedback, helping optimize trading strategies dynamically.

## 🚀 Features

### Core Functionality
- **Adaptive Threshold Management**: Dynamic adjustment of trading parameters based on performance
- **Multi-User Support**: Per-user and per-symbol threshold customization
- **Performance Tracking**: Comprehensive performance metrics and analytics
- **Real-time Integration**: Seamless integration with existing trading systems
- **Gradient-based Learning**: Smart parameter updates using performance gradients

### Advanced Features
- **Safeguards**: Protection against extreme parameter changes
- **Explainable AI**: Clear reasoning for every adaptation decision
- **Performance Analytics**: Detailed performance tracking and trending
- **Health Monitoring**: Comprehensive service health checks and alerting
- **Caching**: Redis-backed caching for improved performance
- **Rate Limiting**: Protection against API abuse

## 📋 Requirements

- Python 3.11+
- PostgreSQL 12+
- Redis 6+ (optional, for caching)
- Docker & Docker Compose (for containerized deployment)

## 🛠 Installation

### Development Setup

1. **Clone and Navigate**
   ```bash
   cd /path/to/tradingbot/backend/ml-service
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Run Database Migrations**
   ```bash
   # Ensure your database is running and accessible
   python -c "from adaptive_threshold import *; print('Database connection test passed')"
   ```

6. **Start Development Server**
   ```bash
   python app.py
   ```

### Docker Deployment

1. **Quick Start**
   ```bash
   ./deploy.sh deploy development
   ```

2. **Production Deployment**
   ```bash
   # Edit .env with production values
   ./deploy.sh deploy production
   ```

3. **View Status**
   ```bash
   ./deploy.sh status
   ```

## 📊 API Reference

### Core Endpoints

#### Get Thresholds
```http
GET /api/v1/thresholds/{user_id}?symbol=BTCUSD
```

Response:
```json
{
  "success": true,
  "data": {
    "user_id": "user123",
    "symbol": "BTCUSD",
    "thresholds": {
      "rsi_threshold": 72.5,
      "confidence_threshold": 0.78,
      "momentum_threshold": 2.1
    },
    "last_adaptation": "2023-12-01T10:30:00Z"
  }
}
```

#### Evaluate Trading Signal
```http
POST /api/v1/evaluate/{user_id}
```

Request:
```json
{
  "signal": {
    "symbol": "BTCUSD",
    "confidence": 0.85,
    "rsi": 65,
    "change_percent": 3.5,
    "volume": 1500000000,
    "action": "BUY"
  }
}
```

Response:
```json
{
  "success": true,
  "data": {
    "user_id": "user123",
    "symbol": "BTCUSD",
    "should_trade": true,
    "current_thresholds": {...},
    "evaluated_at": "2023-12-01T10:35:00Z"
  }
}
```

#### Trigger Adaptation
```http
POST /api/v1/thresholds/{user_id}/adapt
```

Request:
```json
{
  "symbol": "BTCUSD"  // optional
}
```

Response:
```json
{
  "success": true,
  "data": {
    "user_id": "user123",
    "symbol": "BTCUSD",
    "updates": [
      {
        "parameter": "rsi_threshold",
        "old_value": 70.0,
        "new_value": 72.5,
        "reason": "Performance improved",
        "confidence": 0.85
      }
    ],
    "adapted_at": "2023-12-01T10:40:00Z"
  }
}
```

### Monitoring Endpoints

#### Health Check
```http
GET /health
```

#### Performance Metrics
```http
GET /api/v1/performance/{user_id}?symbol=BTCUSD&days_back=30
```

#### Service Statistics
```http
GET /api/v1/stats
```

## 🔧 Configuration

### Environment Variables

```bash
# Basic Configuration
ENVIRONMENT=development          # development, staging, production
DEBUG=true
HOST=0.0.0.0
PORT=5000
WORKERS=4

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_bot
DB_POOL_SIZE=10

# Security
ADMIN_API_KEY=your-secret-admin-key
FLASK_SECRET_KEY=your-flask-secret-key

# Adaptation Parameters
ADAPTATION_LEARNING_RATE=0.01
ADAPTATION_PERFORMANCE_WINDOW=100
ADAPTATION_MIN_TRADES=10

# Monitoring
ENABLE_METRICS=true
LOG_LEVEL=INFO
SENTRY_DSN=your-sentry-dsn      # Optional
SLACK_WEBHOOK_URL=your-webhook  # Optional

# Caching
ENABLE_REDIS_CACHE=true
REDIS_URL=redis://localhost:6379/0
```

### Configuration File

Create `ml_service_config.yaml` (see `ml_service_config_example.yaml`):

```yaml
environment: development
debug: true
database:
  url: postgresql://localhost:5432/trading_bot
  pool_size: 10
adaptation:
  default_learning_rate: 0.01
  performance_window: 100
  # ... more settings
```

## 🏗 Architecture

### Components

1. **AdaptiveThreshold**: Core adaptation logic
2. **PerformanceTracker**: Metrics collection and analysis
3. **Integration Layer**: Trading service integration
4. **Monitoring System**: Health checks and alerting
5. **Configuration Manager**: Dynamic configuration management

### Data Flow

```mermaid
graph TB
    A[Trading Service] --> B[Signal Evaluation]
    B --> C[AdaptiveThreshold]
    C --> D[Performance Analysis]
    D --> E[Parameter Adaptation]
    E --> F[Threshold Updates]
    F --> G[Performance Tracking]
    G --> H[Monitoring & Alerts]
```

### Database Schema

Key tables:
- `adaptive_thresholds`: Current threshold values per user/symbol
- `trades`: Historical trade data for performance calculation
- `trading_signals`: Generated signals and their evaluation
- Performance metrics are stored in local SQLite for efficiency

## 📈 Performance Optimization

### Caching Strategy
- Redis for frequently accessed thresholds
- Local SQLite for metrics aggregation
- In-memory caching for recent calculations

### Monitoring & Alerting
- Real-time performance tracking
- Automatic alerting on failures
- Comprehensive health checks
- Prometheus metrics export

### Safeguards
- Maximum 10% parameter change per adaptation cycle
- Minimum 0.5% change threshold to prevent noise
- 48-hour cooldown after poor performance
- Parameter bounds enforcement

## 🧪 Testing

### Run Tests
```bash
# Unit tests
python -m pytest tests/ -v

# With coverage
python -m pytest tests/ --cov=./ --cov-report=html

# Performance tests
python -m pytest tests/test_performance.py --benchmark-only
```

### Integration Tests
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
python -m pytest tests/integration/ -v

# Cleanup
docker-compose -f docker-compose.test.yml down
```

## 🚀 Deployment

### Development
```bash
./deploy.sh deploy development
```

### Production
```bash
# Configure production environment
cp .env.example .env
# Edit .env with production values

# Deploy
./deploy.sh deploy production

# Monitor
./deploy.sh logs
./deploy.sh status
```

### Scaling
For high-traffic environments:

1. **Horizontal Scaling**:
   ```bash
   docker-compose up --scale ml-service=3
   ```

2. **Load Balancer**: Use nginx or traefik for load balancing

3. **Database Optimization**: 
   - Read replicas for performance metrics
   - Connection pooling
   - Query optimization

## 🔍 Monitoring & Troubleshooting

### Logs
```bash
# View logs
./deploy.sh logs

# Real-time logs
docker-compose logs -f ml-service
```

### Health Checks
```bash
# Service health
curl http://localhost:5000/health

# Integration health
curl http://localhost:5000/api/v1/stats
```

### Common Issues

1. **Database Connection Issues**
   - Check DATABASE_URL
   - Verify database is accessible
   - Check connection pool settings

2. **High Memory Usage**
   - Monitor performance metrics cache
   - Adjust MEMORY_CACHE_SIZE
   - Enable Redis caching

3. **Adaptation Not Triggering**
   - Check minimum trades threshold
   - Verify performance data availability
   - Review adaptation frequency settings

### Performance Monitoring

- **Prometheus Metrics**: Available at `:8000/metrics`
- **Grafana Dashboards**: Included in monitoring profile
- **Application Metrics**: Built-in performance tracking

## 🤝 Integration Guide

### With Existing Trading Service

1. **Add HTTP Client**:
   ```typescript
   const mlServiceUrl = process.env.ML_SERVICE_URL || 'http://ml-service:5000';
   
   async function shouldTrade(userId: string, signal: any): Promise<boolean> {
     const response = await fetch(`${mlServiceUrl}/api/v1/evaluate/${userId}`, {
       method: 'POST',
       headers: { 'Content-Type': 'application/json' },
       body: JSON.stringify({ signal })
     });
     
     const result = await response.json();
     return result.data?.should_trade ?? false;
   }
   ```

2. **Update Trading Loop**:
   ```typescript
   // In your existing trading cycle
   const shouldExecute = await shouldTrade(userId, signalData);
   if (shouldExecute) {
     // Execute trade
     await executeOrder(order);
   }
   ```

3. **Performance Updates**:
   ```typescript
   // After trade completion
   await updatePerformance(userId, symbol, tradeResult);
   ```

## 📚 API Integration Examples

### Python Client
```python
import aiohttp
import asyncio

class MLServiceClient:
    def __init__(self, base_url: str):
        self.base_url = base_url
    
    async def evaluate_signal(self, user_id: str, signal: dict) -> bool:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/v1/evaluate/{user_id}",
                json={"signal": signal}
            ) as response:
                result = await response.json()
                return result["data"]["should_trade"]

# Usage
client = MLServiceClient("http://ml-service:5000")
should_trade = await client.evaluate_signal("user123", signal_data)
```

### JavaScript/Node.js Client
```javascript
class MLServiceClient {
  constructor(baseUrl) {
    this.baseUrl = baseUrl;
  }
  
  async evaluateSignal(userId, signal) {
    const response = await fetch(`${this.baseUrl}/api/v1/evaluate/${userId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ signal })
    });
    
    const result = await response.json();
    return result.data.should_trade;
  }
}

// Usage
const client = new MLServiceClient('http://ml-service:5000');
const shouldTrade = await client.evaluateSignal('user123', signalData);
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and questions:
1. Check the troubleshooting section
2. Review logs: `./deploy.sh logs`
3. Check health status: `./deploy.sh status`
4. Open an issue in the repository

## 🔄 Version History

### v1.0.0
- Initial release
- Core adaptive threshold functionality
- Performance tracking system
- Integration layer
- Comprehensive monitoring

### Roadmap
- [ ] Advanced ML algorithms (neural networks, reinforcement learning)
- [ ] Multi-timeframe analysis
- [ ] Cross-asset correlation analysis
- [ ] Real-time streaming data support
- [ ] Advanced risk management integration