# Institutional Strategies Test Suite

Comprehensive testing framework for institutional trading strategies with performance validation, benchmarking, and statistical analysis.

## Overview

This test suite validates all institutional strategies and their integration layer against strict performance requirements:

- **Liquidity detection accuracy >80%**
- **Smart money signal accuracy >65%**
- **Volume profile computation <50ms**
- **Correlation updates <100ms**
- **Overall 10-15% performance improvement**
- **Integration latency <10ms**

## Test Structure

```
tests/
├── __init__.py                         # Test configuration and imports
├── test_utils.py                       # Common utilities and fixtures
├── test_institutional_strategies.py    # Complete integration tests
├── test_liquidity_hunting.py          # Liquidity strategy tests
├── test_smart_money.py                # Smart money divergence tests
├── test_volume_profile.py             # Volume profile tests
├── test_correlation.py                # Correlation engine tests
├── test_integration.py                # Integration layer tests
├── benchmark_results.py               # Performance benchmarks
└── README.md                          # This file
```

## Running Tests

### Prerequisites

```bash
pip install pytest pytest-asyncio numpy pandas scipy scikit-learn matplotlib seaborn
```

### Run All Tests

```bash
# From the tradingbot root directory
python -m pytest backend/strategies/tests/ -v --tb=short

# With coverage
python -m pytest backend/strategies/tests/ --cov=backend.strategies --cov-report=html

# Parallel execution
python -m pytest backend/strategies/tests/ -n auto
```

### Run Specific Test Categories

```bash
# Integration tests only
python -m pytest backend/strategies/tests/test_institutional_strategies.py -v

# Individual strategy tests
python -m pytest backend/strategies/tests/test_liquidity_hunting.py -v
python -m pytest backend/strategies/tests/test_smart_money.py -v
python -m pytest backend/strategies/tests/test_volume_profile.py -v
python -m pytest backend/strategies/tests/test_correlation.py -v

# Integration layer tests
python -m pytest backend/strategies/tests/test_integration.py -v

# Performance benchmarks
python -c "from backend.strategies.tests.benchmark_results import run_benchmarks; run_benchmarks()"
```

### Run with Specific Markers

```bash
# Performance tests only
python -m pytest backend/strategies/tests/ -m "performance" -v

# Async tests only
python -m pytest backend/strategies/tests/ -m "asyncio" -v
```

## Test Categories

### 1. Unit Tests (`test_*_strategy.py`)

Individual strategy component testing:
- Initialization and configuration
- Feature extraction accuracy
- Signal generation performance
- Edge case handling
- Error recovery

### 2. Integration Tests (`test_institutional_strategies.py`)

Cross-strategy integration validation:
- Strategy coordination
- Feature aggregation
- Signal processing
- Performance requirements
- Statistical significance

### 3. System Tests (`test_integration.py`)

Integration layer validation:
- Strategy manager functionality
- Performance tracking
- Alert system
- A/B testing framework
- End-to-end flows

### 4. Performance Benchmarks (`benchmark_results.py`)

Comprehensive performance analysis:
- Latency benchmarking
- Accuracy validation
- Throughput testing
- Scalability analysis
- Statistical significance

## Key Test Features

### Mock Data Generation

Realistic market data simulation:
```python
from test_utils import MockDataGenerator

generator = MockDataGenerator(seed=42)
price_data = generator.generate_price_series(1000)
volume_data = generator.generate_volume_series(price_data)
order_book = generator.generate_order_book(30000)
trades = generator.generate_trades(30000, 50000, 20)
```

### Performance Benchmarking

Automated performance validation:
```python
from test_utils import PerformanceBenchmark

benchmark = PerformanceBenchmark()
results = benchmark.benchmark_function(strategy.extract_features, market_data, iterations=100)
assert results['mean'] < 0.050  # 50ms requirement
```

### Statistical Testing

Statistical significance validation:
```python
from test_utils import assert_statistical_significance

t_stat, p_value = stats.ttest_ind(baseline_results, enhanced_results)
assert_statistical_significance(p_value, alpha=0.05, test_name="Performance Improvement")
```

### Async Testing

Async strategy testing support:
```python
@pytest.mark.asyncio
async def test_async_detection():
    result = await async_test_helper(
        detector.detect_divergence('BTC/USD', price_data),
        timeout=5.0
    )
    assert result is not None
```

## Performance Requirements Validation

### Accuracy Requirements

| Strategy | Requirement | Test Method |
|----------|-------------|-------------|
| Liquidity Hunting | >80% | Pattern recognition validation |
| Smart Money | >65% | Divergence detection accuracy |
| Volume Profile | >80% | POC and value area accuracy |
| Correlation | >85% | Matrix calculation validation |

### Latency Requirements

| Component | Requirement | Test Method |
|-----------|-------------|-------------|
| Volume Profile | <50ms | Computation time benchmarking |
| Correlation | <100ms | Update time measurement |
| Integration | <10ms | End-to-end latency testing |
| Strategy Execution | <50ms | Individual strategy timing |

### Improvement Requirements

- **Overall Performance**: 10-15% improvement over baseline
- **Statistical Significance**: p-value < 0.05
- **Consistency**: >90% of tests pass requirements

## Benchmark Reports

The benchmark system generates comprehensive reports:

### JSON Results
```json
{
  "suite_info": {
    "name": "Institutional Strategies Benchmark",
    "overall_score": 85.5,
    "passed_tests": 42,
    "total_tests": 49
  },
  "results": [...],
  "summary": {...}
}
```

### Visual Reports
- Performance overview charts
- Latency comparison graphs  
- Accuracy benchmark plots
- Improvement heatmaps

### Text Summary
```
INSTITUTIONAL STRATEGIES BENCHMARK RESULTS
========================================
Overall Score: 87.8% (43/49 passed)
Duration: 156.3 seconds

Results by Category:
  Strategy Performance: 12/14 (85.7%)
  Performance: 18/20 (90.0%)
  Statistical Analysis: 8/10 (80.0%)
  Integration: 5/5 (100.0%)

Key Requirements Status:
  ✅ PASS Liquidity Detection Accuracy: >80%
  ✅ PASS Smart Money Signal Accuracy: >65%
  ✅ PASS Volume Profile Computation: <50ms
  ✅ PASS Correlation Updates: <100ms
  ❌ FAIL Integration Latency: <10ms
  ✅ PASS Overall Performance Improvement: >10%
```

## Configuration

Test configuration in `__init__.py`:
```python
TEST_CONFIG = {
    'performance_thresholds': {
        'liquidity_detection_accuracy': 0.80,
        'smart_money_signal_accuracy': 0.65,
        'volume_profile_computation_time': 0.050,
        'correlation_update_time': 0.100,
        'integration_latency': 0.010,
        'overall_performance_improvement': 0.10
    },
    'test_data_size': 1000,
    'statistical_significance_level': 0.05,
    'benchmark_iterations': 100,
    'mock_data_seed': 42
}
```

## Continuous Integration

For CI/CD integration:

```yaml
# .github/workflows/test-strategies.yml
name: Strategy Tests
on: [push, pull_request]
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: 3.9
      - name: Install dependencies
        run: pip install -r requirements-test.txt
      - name: Run tests
        run: python -m pytest backend/strategies/tests/ --tb=short --durations=10
      - name: Run benchmarks
        run: python -c "from backend.strategies.tests.benchmark_results import run_benchmarks; run_benchmarks()"
```

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all strategy modules are properly installed
2. **Async Test Failures**: Check event loop configuration
3. **Performance Test Failures**: Verify system resources and reduce test iterations
4. **Mock Data Issues**: Check random seed consistency

### Debug Mode

Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
pytest.main(["-v", "--log-cli-level=DEBUG", "test_file.py"])
```

## Contributing

When adding new tests:

1. Follow the existing naming convention
2. Include both positive and negative test cases
3. Add performance benchmarks for new features
4. Update requirements validation
5. Ensure async compatibility where needed

## Performance Monitoring

The test suite includes performance regression detection:
- Baseline performance tracking
- Automated performance comparison
- Alert generation for regressions
- Historical performance trends

This ensures the institutional strategies maintain their performance characteristics over time and meet all specified requirements.