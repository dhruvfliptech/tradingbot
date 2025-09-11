# Strategy Integration Layer

The Strategy Integration Layer provides a comprehensive framework for connecting institutional trading strategies with the reinforcement learning (RL) system. It enables seamless feature extraction, signal processing, and real-time coordination between multiple strategies and the RL environment.

## 🎯 Key Features

- **30+ New Features**: Aggregates features from all institutional strategies into the RL state space
- **<10ms Feature Latency**: High-performance feature processing with sub-10ms latency
- **A/B Testing Support**: Built-in framework for testing strategy combinations
- **Explainable Signals**: Provides detailed reasoning for signal combinations
- **Dynamic Management**: Enable/disable strategies and adjust weights in real-time
- **Comprehensive Monitoring**: Real-time performance tracking and alerting

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                Strategy Integration Manager                  │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐ │
│  │ Feature         │  │ Signal          │  │ Performance     │ │
│  │ Aggregator      │  │ Processor       │  │ Tracker         │ │
│  └─────────────────┘  └─────────────────┘  └─────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              RL Connector                                   │ │
│  └─────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────┐
│                    RL Environment                           │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │ State       │  │ Trading     │  │ Reward      │         │
│  │ Processor   │  │ Agent       │  │ Calculator  │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
```

## 📁 Components

### 1. Strategy Integration Manager (`strategy_manager.py`)

The main orchestrator that coordinates all components and manages strategy lifecycle.

**Key Features:**
- Dynamic strategy loading and management
- Real-time execution coordination
- Performance monitoring
- Error handling and recovery

**Usage:**
```python
from backend.strategies.integration import StrategyIntegrationManager

manager = StrategyIntegrationManager()
await manager.start()

# Get aggregated features
features = manager.get_aggregated_features()

# Get processed signals
signals = manager.get_processed_signals()

# Manage strategies dynamically
manager.enable_strategy('whale_tracker')
manager.update_strategy_weight('volume_profile', 1.5)
```

### 2. Feature Aggregator (`feature_aggregator.py`)

Combines and processes features from multiple strategies into optimized vectors for RL training.

**Key Features:**
- Multi-strategy feature aggregation
- Real-time normalization and scaling
- Feature selection and importance scoring
- Correlation analysis and redundancy removal

**Usage:**
```python
from backend.strategies.integration import FeatureAggregator

aggregator = FeatureAggregator({
    'max_features': 50,
    'normalization_method': 'robust',
    'enable_feature_selection': True
})

await aggregator.start()

# Add features from strategies
await aggregator.add_features({
    'whale_sentiment': 0.75,
    'volume_poc': 30000,
    'order_flow': 0.3
})

# Get aggregated result
final_features = aggregator.get_aggregated_features()
```

### 3. Signal Processor (`signal_processor.py`)

Processes and prioritizes trading signals from multiple strategies with conflict resolution.

**Key Features:**
- Multi-strategy signal aggregation
- Conflict detection and resolution
- Confidence-weighted combining
- Explainable signal reasoning

**Usage:**
```python
from backend.strategies.integration import SignalProcessor

processor = SignalProcessor({
    'conflict_resolution_method': 'weighted_average',
    'consensus_threshold': 0.7
})

await processor.start()

# Add signals from strategies
await processor.add_signals({
    'whale_buy_signal': {
        'type': 'buy',
        'confidence': 0.8,
        'reasoning': 'Large whale accumulation'
    },
    'volume_sell_signal': {
        'type': 'sell', 
        'confidence': 0.6,
        'reasoning': 'Volume divergence detected'
    }
})

# Get processed result
latest_signal = processor.get_latest_signal()
```

### 4. RL Connector (`rl_connector.py`)

Direct interface between the integration system and RL environment for real-time communication.

**Key Features:**
- Real-time feature streaming
- Action interpretation and routing
- Reward feedback processing
- WebSocket server for external connections

**Usage:**
```python
from backend.strategies.integration import RLConnector

connector = RLConnector({
    'action_mode': 'HYBRID',
    'enable_websocket_server': True
})

await connector.start()

# Update features for RL
await connector.update_features({
    'aggregated_features': feature_vector
})

# Request action from RL
action = await connector.request_action()

# Send reward feedback
await connector.send_reward(reward_value, step_id)
```

### 5. Performance Tracker (`performance_tracker.py`)

Comprehensive performance monitoring and analytics with A/B testing support.

**Key Features:**
- Real-time strategy performance monitoring
- Alert system for performance issues
- A/B testing framework
- Historical analysis and reporting

**Usage:**
```python
from backend.strategies.integration import PerformanceTracker, PerformanceMetric

tracker = PerformanceTracker({
    'enable_alerting': True,
    'enable_ab_testing': True
})

await tracker.start()

# Update strategy performance
await tracker.update_strategy_performance('whale_tracker', {
    'total_signals': 100,
    'successful_signals': 75,
    'avg_latency': 0.008
})

# Start A/B test
test_id = await tracker.start_ab_test(
    'strategy_a', 'strategy_b', PerformanceMetric.ACCURACY
)
```

## ⚙️ Configuration

### Strategy Configuration

```python
config = {
    'strategies': {
        'whale_tracker': {
            'module_path': 'backend.strategies.institutional.whale_tracker',
            'class_name': 'WhaleTracker',
            'priority': 'HIGH',
            'weight': 2.0,
            'enabled': True,
            'feature_names': ['whale_sentiment', 'large_transfers'],
            'signal_names': ['whale_buy_signal', 'whale_sell_signal']
        },
        'volume_profile': {
            'module_path': 'backend.strategies.institutional.volume_profile',
            'class_name': 'VolumeProfile',
            'priority': 'HIGH',
            'weight': 1.8,
            'enabled': True,
            'feature_names': ['poc_distance', 'volume_imbalance'],
            'signal_names': ['volume_breakout']
        }
        # ... more strategies
    }
}
```

### Feature Aggregation Configuration

```python
feature_config = {
    'max_features': 50,
    'normalization_method': 'robust',
    'feature_selection_method': 'mutual_info',
    'correlation_threshold': 0.95,
    'enable_dimensionality_reduction': True,
    'pca_variance_ratio': 0.95
}
```

### Signal Processing Configuration

```python
signal_config = {
    'conflict_resolution_method': 'weighted_average',
    'consensus_threshold': 0.6,
    'min_strategies_for_consensus': 2,
    'confidence_threshold': 0.5,
    'enable_signal_validation': True
}
```

## 🚀 Quick Start

### 1. Basic Setup

```python
import asyncio
from backend.strategies.integration import setup_integration_layer

async def main():
    # Quick setup with default configuration
    manager = await setup_integration_layer()
    
    # Let it run for a while
    await asyncio.sleep(30)
    
    # Get system status
    metrics = manager.get_system_metrics()
    print(f"Active strategies: {metrics['active_strategies']}")
    print(f"Total features: {metrics['total_features']}")
    
    await manager.stop()

asyncio.run(main())
```

### 2. Custom Configuration

```python
import asyncio
from backend.strategies.integration import StrategyIntegrationManager

async def main():
    config = {
        'max_workers': 10,
        'execution_interval': 0.5,
        'feature_config': {
            'max_features': 30,
            'normalization_method': 'robust'
        },
        'signal_config': {
            'consensus_threshold': 0.7
        }
    }
    
    manager = StrategyIntegrationManager(config)
    await manager.start()
    
    # Your trading logic here
    
    await manager.stop()

asyncio.run(main())
```

### 3. Adding Custom Strategies

```python
class MyCustomStrategy:
    def __init__(self, config=None):
        self.config = config or {}
    
    async def extract_features(self, market_data):
        return {
            'my_feature_1': 0.75,
            'my_feature_2': 1.2
        }
    
    async def generate_signals(self, market_data):
        return {
            'my_signal': {
                'type': 'buy',
                'confidence': 0.8,
                'reasoning': 'Custom analysis result'
            }
        }

# Register with manager
manager.strategy_instances['my_strategy'] = MyCustomStrategy()
```

## 📊 Performance Metrics

The integration layer provides comprehensive performance metrics:

### Strategy Metrics
- **Signal Accuracy**: Percentage of successful signals
- **Latency**: Average feature extraction and signal processing time
- **Uptime**: Strategy availability ratio
- **Sharpe Ratio**: Risk-adjusted returns
- **Win Rate**: Percentage of profitable trades
- **Max Drawdown**: Maximum portfolio decline

### System Metrics
- **Throughput**: Messages processed per second
- **Error Rate**: Percentage of failed operations
- **Memory Usage**: System memory consumption
- **CPU Usage**: Processor utilization
- **Feature Latency**: Time to process features (target: <10ms)

## 🧪 A/B Testing

Built-in A/B testing framework for strategy optimization:

```python
# Start A/B test
test_id = await tracker.start_ab_test(
    strategy_a='whale_tracker',
    strategy_b='volume_profile', 
    metric=PerformanceMetric.SHARPE_RATIO
)

# Run strategies for testing period
await asyncio.sleep(3600)  # 1 hour

# Get results
result = await tracker.end_ab_test(test_id)
print(f"Winner: {result.winner}")
print(f"Statistical significance: {result.statistical_significance}")
```

## 🔧 Monitoring and Alerts

### Performance Thresholds

```python
thresholds = {
    'strategy_accuracy_min': 0.55,
    'strategy_latency_max': 0.01,  # 10ms
    'system_error_rate_max': 0.01,  # 1%
    'memory_usage_max': 0.8  # 80%
}
```

### Alert Handling

```python
async def alert_handler(alert):
    print(f"ALERT: {alert.message}")
    if alert.severity == AlertSeverity.CRITICAL:
        # Take corrective action
        pass

tracker.add_alert_callback(alert_handler)
```

## 🔗 Integration with RL Environment

The integration layer seamlessly connects with the RL environment:

### Feature Flow
1. **Extraction**: Strategies extract features from market data
2. **Aggregation**: Features are normalized and combined
3. **Selection**: Important features are selected for RL input
4. **Injection**: Features are injected into RL state space

### Signal Flow
1. **Generation**: Strategies generate trading signals
2. **Processing**: Signals are validated and conflicts resolved
3. **Aggregation**: Final signal is computed with confidence weights
4. **Routing**: Signal is sent to RL environment as advisory input

### Reward Feedback
1. **Collection**: Trading results are collected
2. **Calculation**: Rewards are computed with multiple components
3. **Attribution**: Performance is attributed to contributing strategies
4. **Learning**: Feedback improves strategy weights and selection

## 📈 Expected Performance Improvements

With the integration layer, expect:

- **30+ Additional Features**: Enhanced RL state representation
- **Improved Signal Quality**: Multi-strategy consensus reduces noise
- **Reduced Latency**: Optimized processing pipeline (<10ms)
- **Better Risk Management**: Multi-dimensional risk assessment
- **Adaptive Weights**: Dynamic strategy importance based on performance
- **Explainable Decisions**: Clear reasoning for all trading decisions

## 🛠️ Development and Testing

### Running Tests

```bash
# Run integration tests
python -m pytest backend/strategies/integration/tests/

# Run specific component tests
python -m pytest backend/strategies/integration/tests/test_feature_aggregator.py

# Run performance benchmarks
python backend/strategies/integration/benchmark.py
```

### Example Usage

```bash
# Run comprehensive example
python backend/strategies/integration/example_usage.py

# Run specific demonstrations
python -c "
import asyncio
from backend.strategies.integration.example_usage import demo_basic_integration
asyncio.run(demo_basic_integration())
"
```

## 🔒 Security and Reliability

- **Input Validation**: All features and signals are validated
- **Error Handling**: Robust error handling with automatic recovery
- **Circuit Breakers**: Prevent cascade failures
- **Rate Limiting**: Protect against resource exhaustion
- **Monitoring**: Comprehensive health checks and alerts

## 📚 API Reference

### StrategyIntegrationManager

```python
class StrategyIntegrationManager:
    async def start()
    async def stop()
    def get_aggregated_features() -> Dict[str, float]
    def get_processed_signals() -> Dict[str, Any]
    def enable_strategy(name: str) -> bool
    def disable_strategy(name: str) -> bool
    def update_strategy_weight(name: str, weight: float) -> bool
    def get_system_metrics() -> Dict[str, Any]
```

### FeatureAggregator

```python
class FeatureAggregator:
    async def start()
    async def stop()
    async def add_features(features: Dict[str, float], target: float = None)
    def get_aggregated_features() -> Dict[str, float]
    def get_feature_vector() -> np.ndarray
    def get_feature_importance_ranking() -> List[Tuple[str, float]]
```

### SignalProcessor

```python
class SignalProcessor:
    async def start()
    async def stop()
    async def add_signals(signals: Dict[str, Any])
    def get_latest_signal() -> Optional[AggregatedSignal]
    def get_processed_signals() -> List[AggregatedSignal]
    def update_strategy_weight(name: str, weight: float)
```

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Add comprehensive tests for new features
3. Update documentation for API changes
4. Ensure performance requirements are met (<10ms latency)
5. Add proper error handling and logging

## 📄 License

This integration layer is part of the trading bot system and follows the same licensing terms.

---

*For more detailed examples and advanced usage patterns, see the `example_usage.py` file in this directory.*