"""
Test Suite for Institutional Trading Strategies
==============================================

Comprehensive testing framework for all institutional strategies and integration components.

Test Structure:
- Unit tests for individual strategy components
- Integration tests for strategy coordination
- Performance benchmarks and validation
- Statistical significance testing
- Backtesting validation
- Stress tests for edge cases
"""

import sys
import os
from pathlib import Path

# Add parent directories to path for imports
test_dir = Path(__file__).parent
strategies_dir = test_dir.parent
backend_dir = strategies_dir.parent
sys.path.insert(0, str(backend_dir))
sys.path.insert(0, str(strategies_dir))

# Test configuration
TEST_CONFIG = {
    'performance_thresholds': {
        'liquidity_detection_accuracy': 0.80,
        'smart_money_signal_accuracy': 0.65,
        'volume_profile_computation_time': 0.050,  # 50ms
        'correlation_update_time': 0.100,  # 100ms
        'integration_latency': 0.010,  # 10ms
        'overall_performance_improvement': 0.10  # 10%
    },
    'test_data_size': 1000,
    'statistical_significance_level': 0.05,
    'benchmark_iterations': 100,
    'stress_test_duration': 300,  # 5 minutes
    'mock_data_seed': 42
}

# Test utilities will be imported by test modules
from .test_utils import *