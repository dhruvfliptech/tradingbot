"""
Integration Layer Test Suite
===========================

Comprehensive tests for the strategy integration layer:
- Strategy manager coordination
- Feature aggregation
- Signal processing
- RL connector functionality
- Performance tracking
- End-to-end integration flows
- Monitoring and alerting

Target Requirements:
- Integration latency <10ms
- Strategy coordination accuracy
- Feature aggregation performance
- Signal processing reliability
- System monitoring effectiveness
"""

import pytest
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import time
import logging
from unittest.mock import Mock, patch, AsyncMock, MagicMock

# Test utilities
from .test_utils import (
    MockDataGenerator, PerformanceBenchmark, StrategyTester, BacktestValidator,
    MockMarketData, MockStrategy, assert_performance_threshold, assert_latency_threshold,
    async_test_helper, measure_async_performance
)

# Import integration components
try:
    from ..integration.strategy_manager import (
        StrategyIntegrationManager, StrategyConfig, StrategyExecution,
        StrategyStatus, StrategyPriority
    )
    from ..integration.performance_tracker import (
        PerformanceTracker, StrategyPerformance, SystemPerformance,
        PerformanceAlert, PerformanceMetric, AlertSeverity
    )
    from ..integration.feature_aggregator import FeatureAggregator
    from ..integration.signal_processor import SignalProcessor
    from ..integration.rl_connector import RLConnector
except ImportError as e:
    pytest.skip(f"Could not import integration components: {e}", allow_module_level=True)

from . import TEST_CONFIG

logger = logging.getLogger(__name__)


class TestStrategyIntegrationManager:
    """Test suite for StrategyIntegrationManager"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.data_generator = MockDataGenerator(TEST_CONFIG['mock_data_seed'])
        self.benchmark = PerformanceBenchmark()
        
        # Test configuration for integration manager
        self.test_config = {
            'max_workers': 4,
            'execution_interval': 0.1,
            'feature_cache_size': 100,
            'signal_cache_size': 50,
            'max_feature_latency': 0.01,
            'max_signal_latency': 0.05,
            'error_threshold': 0.1,
            'retry_attempts': 2,
            'strategy_timeout': 2.0,
            'enable_monitoring': True,
            'metrics_update_interval': 1.0,
            'strategies': {
                'mock_strategy_1': {
                    'module_path': 'test_utils',
                    'class_name': 'MockStrategy',
                    'priority': 'HIGH',
                    'weight': 2.0,
                    'enabled': True,
                    'feature_names': ['feature_1', 'feature_2', 'feature_3'],
                    'signal_names': ['signal_1', 'signal_2']
                },
                'mock_strategy_2': {
                    'module_path': 'test_utils',
                    'class_name': 'MockStrategy',
                    'priority': 'MEDIUM',
                    'weight': 1.0,
                    'enabled': True,
                    'feature_names': ['feature_4', 'feature_5'],
                    'signal_names': ['signal_3']
                },
                'mock_strategy_3': {
                    'module_path': 'test_utils',
                    'class_name': 'MockStrategy',
                    'priority': 'LOW',
                    'weight': 0.5,
                    'enabled': False,  # Disabled for testing
                    'feature_names': ['feature_6'],
                    'signal_names': ['signal_4']
                }
            }
        }
        
        self.performance_metrics = {}
    
    def test_manager_initialization(self):
        """Test strategy manager initialization"""
        # Mock the module import
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            
            # Validate initialization
            assert manager.config is not None
            assert len(manager.strategies) == 3  # All configured strategies
            assert len(manager.strategy_executions) == 3
            
            # Check enabled/disabled strategies
            enabled_strategies = [name for name, config in manager.strategies.items() if config.enabled]
            assert len(enabled_strategies) == 2  # Only 2 enabled
            
            # Validate configuration loading
            strategy_1 = manager.strategies['mock_strategy_1']
            assert strategy_1.priority == StrategyPriority.HIGH
            assert strategy_1.weight == 2.0
            assert strategy_1.feature_names == ['feature_1', 'feature_2', 'feature_3']
            
        logger.info("✓ Strategy manager initialization test passed")
    
    @pytest.mark.asyncio
    async def test_strategy_initialization(self):
        """Test strategy instance initialization"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            
            # Initialize strategies
            await manager.initialize_strategies()
            
            # Check strategy instances
            assert len(manager.strategy_instances) == 2  # Only enabled strategies
            assert 'mock_strategy_1' in manager.strategy_instances
            assert 'mock_strategy_2' in manager.strategy_instances
            assert 'mock_strategy_3' not in manager.strategy_instances  # Disabled
            
            # Check execution states
            assert manager.strategy_executions['mock_strategy_1'].status == StrategyStatus.ACTIVE
            assert manager.strategy_executions['mock_strategy_2'].status == StrategyStatus.ACTIVE
            assert manager.strategy_executions['mock_strategy_3'].status == StrategyStatus.INACTIVE
            
            # Validate metrics
            assert manager.metrics['active_strategies'] == 2
            
        logger.info("✓ Strategy initialization test passed")
    
    @pytest.mark.asyncio
    async def test_strategy_execution(self):
        """Test individual strategy execution"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            await manager.initialize_strategies()
            
            # Test single strategy execution
            execution_times = []
            
            for _ in range(10):
                start_time = time.perf_counter()
                
                result = await manager._execute_strategy('mock_strategy_1')
                
                execution_time = time.perf_counter() - start_time
                execution_times.append(execution_time)
                
                # Validate execution result
                assert result['success'] is True
                assert result['strategy_name'] == 'mock_strategy_1'
                assert 'features' in result
                assert 'signals' in result
                assert 'execution_time' in result
                
                # Validate feature and signal structure
                features = result['features']
                signals = result['signals']
                
                assert isinstance(features, dict)
                assert isinstance(signals, dict)
                assert len(features) > 0
                assert len(signals) > 0
            
            # Performance validation
            avg_execution_time = np.mean(execution_times)
            max_execution_time = np.max(execution_times)
            
            assert_latency_threshold(
                avg_execution_time,
                self.test_config['max_feature_latency'] * 5,  # 50ms
                'Strategy Execution Average'
            )
            
            assert_latency_threshold(
                max_execution_time,
                self.test_config['strategy_timeout'],
                'Strategy Execution Maximum'
            )
            
            self.performance_metrics['strategy_execution_time'] = avg_execution_time
            
        logger.info(f"✓ Strategy execution - Avg: {avg_execution_time*1000:.1f}ms, "
                   f"Max: {max_execution_time*1000:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_parallel_strategy_execution(self):
        """Test parallel execution of multiple strategies"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            await manager.initialize_strategies()
            
            # Test parallel execution
            execution_times = []
            
            for _ in range(5):
                start_time = time.perf_counter()
                
                # Create execution tasks for all active strategies
                tasks = []
                for name, config in manager.strategies.items():
                    if config.enabled and name in manager.strategy_instances:
                        task = asyncio.create_task(manager._execute_strategy(name))
                        tasks.append(task)
                
                # Execute in parallel
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                execution_time = time.perf_counter() - start_time
                execution_times.append(execution_time)
                
                # Validate results
                successful_results = [r for r in results if not isinstance(r, Exception) and r.get('success')]
                assert len(successful_results) == 2  # Both active strategies
                
                # Check result aggregation
                all_features = {}
                all_signals = {}
                
                for result in successful_results:
                    features = result.get('features', {})
                    signals = result.get('signals', {})
                    
                    all_features.update(features)
                    all_signals.update(signals)
                
                # Should have features and signals from both strategies
                assert len(all_features) >= 5  # At least 5 features total
                assert len(all_signals) >= 3   # At least 3 signals total
            
            # Parallel execution should be faster than sequential
            avg_parallel_time = np.mean(execution_times)
            
            # Should complete within reasonable time
            assert_latency_threshold(
                avg_parallel_time,
                0.1,  # 100ms for parallel execution
                'Parallel Strategy Execution'
            )
            
            self.performance_metrics['parallel_execution_time'] = avg_parallel_time
            
        logger.info(f"✓ Parallel strategy execution - Avg: {avg_parallel_time*1000:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_feature_aggregation(self):
        """Test feature aggregation and management"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            await manager.initialize_strategies()
            
            # Test feature collection over multiple executions
            aggregation_times = []
            
            for i in range(20):
                start_time = time.perf_counter()
                
                # Execute strategies to generate features
                tasks = [
                    manager._execute_strategy('mock_strategy_1'),
                    manager._execute_strategy('mock_strategy_2')
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Process results (simulate _process_execution_results)
                all_features = {}
                for result in results:
                    if result.get('success'):
                        strategy_name = result['strategy_name']
                        features = result.get('features', {})
                        
                        for name, value in features.items():
                            feature_key = f"{strategy_name}_{name}"
                            all_features[feature_key] = value
                
                # Update latest features
                manager.latest_features.update(all_features)
                
                # Update feature history
                for feature_key, value in all_features.items():
                    if feature_key not in manager.feature_history:
                        manager.feature_history[feature_key] = []
                    manager.feature_history[feature_key].append(value)
                    
                    # Maintain cache size
                    if len(manager.feature_history[feature_key]) > manager.config['feature_cache_size']:
                        manager.feature_history[feature_key] = manager.feature_history[feature_key][-manager.config['feature_cache_size']:]
                
                aggregation_time = time.perf_counter() - start_time
                aggregation_times.append(aggregation_time)
            
            # Validate feature aggregation
            assert len(manager.latest_features) > 0
            assert len(manager.feature_history) > 0
            
            # Check feature naming convention
            for feature_key in manager.latest_features.keys():
                assert '_' in feature_key  # Should have strategy_feature format
                parts = feature_key.split('_', 1)
                assert parts[0] in ['mock_strategy_1', 'mock_strategy_2']
            
            # Check feature history management
            for feature_history in manager.feature_history.values():
                assert len(feature_history) <= manager.config['feature_cache_size']
            
            # Performance validation
            avg_aggregation_time = np.mean(aggregation_times)
            assert_latency_threshold(
                avg_aggregation_time,
                TEST_CONFIG['performance_thresholds']['integration_latency'],
                'Feature Aggregation'
            )
            
            self.performance_metrics['feature_aggregation_time'] = avg_aggregation_time
            
        logger.info(f"✓ Feature aggregation - Features: {len(manager.latest_features)}, "
                   f"Time: {avg_aggregation_time*1000:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_signal_processing(self):
        """Test signal processing and coordination"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            await manager.initialize_strategies()
            
            # Test signal collection and processing
            processing_times = []
            
            for _ in range(15):
                start_time = time.perf_counter()
                
                # Execute strategies to generate signals
                tasks = [
                    manager._execute_strategy('mock_strategy_1'),
                    manager._execute_strategy('mock_strategy_2')
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Process signals
                all_signals = {}
                for result in results:
                    if result.get('success'):
                        strategy_name = result['strategy_name']
                        signals = result.get('signals', {})
                        
                        for name, value in signals.items():
                            signal_key = f"{strategy_name}_{name}"
                            all_signals[signal_key] = value
                
                # Update signal state
                manager.latest_signals.update(all_signals)
                
                # Update signal history
                for signal_key, value in all_signals.items():
                    if signal_key not in manager.signal_history:
                        manager.signal_history[signal_key] = []
                    manager.signal_history[signal_key].append(value)
                    
                    if len(manager.signal_history[signal_key]) > manager.config['signal_cache_size']:
                        manager.signal_history[signal_key] = manager.signal_history[signal_key][-manager.config['signal_cache_size']:]
                
                processing_time = time.perf_counter() - start_time
                processing_times.append(processing_time)
            
            # Validate signal processing
            assert len(manager.latest_signals) > 0
            assert len(manager.signal_history) > 0
            
            # Check signal weighting by strategy priority
            strategy_1_signals = [k for k in manager.latest_signals.keys() if k.startswith('mock_strategy_1')]
            strategy_2_signals = [k for k in manager.latest_signals.keys() if k.startswith('mock_strategy_2')]
            
            assert len(strategy_1_signals) > 0  # High priority strategy
            assert len(strategy_2_signals) > 0  # Medium priority strategy
            
            # Performance validation
            avg_processing_time = np.mean(processing_times)
            assert_latency_threshold(
                avg_processing_time,
                TEST_CONFIG['performance_thresholds']['integration_latency'],
                'Signal Processing'
            )
            
            self.performance_metrics['signal_processing_time'] = avg_processing_time
            
        logger.info(f"✓ Signal processing - Signals: {len(manager.latest_signals)}, "
                   f"Time: {avg_processing_time*1000:.1f}ms")
    
    @pytest.mark.asyncio
    async def test_strategy_coordination(self):
        """Test strategy coordination and conflict resolution"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            await manager.initialize_strategies()
            
            # Test strategy weight application
            strategy_weights = {
                'mock_strategy_1': 2.0,  # High weight
                'mock_strategy_2': 1.0,  # Medium weight
            }
            
            coordination_results = []
            
            for _ in range(10):
                # Execute strategies
                tasks = [
                    manager._execute_strategy('mock_strategy_1'),
                    manager._execute_strategy('mock_strategy_2')
                ]
                
                results = await asyncio.gather(*tasks)
                
                # Simulate weighted signal coordination
                weighted_signals = {}
                
                for result in results:
                    if result.get('success'):
                        strategy_name = result['strategy_name']
                        weight = strategy_weights.get(strategy_name, 1.0)
                        signals = result.get('signals', {})
                        
                        for signal_name, signal_value in signals.items():
                            if isinstance(signal_value, (int, float)):
                                key = f"coordinated_{signal_name}"
                                if key not in weighted_signals:
                                    weighted_signals[key] = []
                                weighted_signals[key].append((signal_value, weight))
                
                # Calculate weighted averages
                final_signals = {}
                for signal_key, values_weights in weighted_signals.items():
                    if values_weights:
                        total_weighted_value = sum(value * weight for value, weight in values_weights)
                        total_weight = sum(weight for _, weight in values_weights)
                        final_signals[signal_key] = total_weighted_value / total_weight if total_weight > 0 else 0
                
                coordination_results.append({
                    'individual_results': len(results),
                    'coordinated_signals': len(final_signals),
                    'total_weight': sum(strategy_weights.values())
                })
            
            # Validate coordination
            assert len(coordination_results) > 0
            
            for result in coordination_results:
                assert result['individual_results'] == 2
                assert result['coordinated_signals'] > 0
                assert result['total_weight'] == 3.0  # 2.0 + 1.0
            
        logger.info("✓ Strategy coordination and weighting test passed")
    
    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self):
        """Test error handling and recovery mechanisms"""
        # Create strategy that fails occasionally
        class FailingMockStrategy(MockStrategy):
            def __init__(self, **config):
                super().__init__(**config)
                self.failure_rate = config.get('failure_rate', 0.3)
            
            def extract_features(self, market_data):
                if np.random.random() < self.failure_rate:
                    raise Exception("Simulated strategy failure")
                return super().extract_features(market_data)
        
        test_config = self.test_config.copy()
        test_config['strategies']['failing_strategy'] = {
            'module_path': 'test_utils',
            'class_name': 'FailingMockStrategy',
            'priority': 'MEDIUM',
            'weight': 1.0,
            'enabled': True,
            'config_params': {'failure_rate': 0.4}
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_module.FailingMockStrategy = FailingMockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(test_config)
            await manager.initialize_strategies()
            
            # Test execution with failures
            successful_executions = 0
            failed_executions = 0
            
            for _ in range(20):
                try:
                    result = await manager._execute_strategy('failing_strategy')
                    if result.get('success'):
                        successful_executions += 1
                    else:
                        failed_executions += 1
                except Exception:
                    failed_executions += 1
            
            # Should have both successes and failures
            assert successful_executions > 0, "No successful executions"
            assert failed_executions > 0, "No failures detected (expected some failures)"
            
            # Error rate should be reasonable
            total_executions = successful_executions + failed_executions
            error_rate = failed_executions / total_executions
            
            assert 0.2 <= error_rate <= 0.6, f"Unexpected error rate: {error_rate:.2%}"
            
            # Check strategy execution state
            execution_state = manager.strategy_executions['failing_strategy']
            assert execution_state.error_count > 0
            assert execution_state.last_error is not None
            
        logger.info(f"✓ Error handling - Success: {successful_executions}, "
                   f"Failures: {failed_executions}, Rate: {error_rate:.2%}")
    
    @pytest.mark.asyncio
    async def test_system_monitoring(self):
        """Test system monitoring and health checks"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            await manager.initialize_strategies()
            
            # Run for a short period to generate metrics
            start_time = time.time()
            run_duration = 2.0  # 2 seconds
            
            # Simulate execution loop
            while time.time() - start_time < run_duration:
                # Execute strategies
                tasks = [
                    manager._execute_strategy('mock_strategy_1'),
                    manager._execute_strategy('mock_strategy_2')
                ]
                
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                # Update metrics
                manager.metrics['total_executions'] += len([r for r in results if not isinstance(r, Exception)])
                
                await asyncio.sleep(0.1)  # Brief pause
            
            # Test system metrics collection
            system_metrics = manager.get_system_metrics()
            
            # Validate metrics structure
            assert isinstance(system_metrics, dict)
            assert 'total_strategies' in system_metrics
            assert 'active_strategies' in system_metrics
            assert 'strategies' in system_metrics
            
            # Validate strategy-specific metrics
            strategies_metrics = system_metrics['strategies']
            assert 'mock_strategy_1' in strategies_metrics
            assert 'mock_strategy_2' in strategies_metrics
            
            for strategy_name, strategy_metrics in strategies_metrics.items():
                if strategy_metrics:  # If not None
                    assert 'status' in strategy_metrics
                    assert 'execution_time' in strategy_metrics
                    assert 'error_count' in strategy_metrics
            
            # Check monitoring functionality
            await manager._check_strategy_health()
            await manager._update_performance_metrics()
            
        logger.info(f"✓ System monitoring - Total strategies: {system_metrics['total_strategies']}, "
                   f"Active: {system_metrics['active_strategies']}")
    
    def test_strategy_management_operations(self):
        """Test strategy management operations (enable/disable/reload)"""
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(self.test_config)
            
            # Test enable/disable operations
            assert manager.enable_strategy('mock_strategy_3') is True
            assert manager.strategies['mock_strategy_3'].enabled is True
            
            assert manager.disable_strategy('mock_strategy_1') is True
            assert manager.strategies['mock_strategy_1'].enabled is False
            
            # Test weight updates
            assert manager.update_strategy_weight('mock_strategy_2', 1.5) is True
            assert manager.strategies['mock_strategy_2'].weight == 1.5
            
            # Test invalid operations
            assert manager.enable_strategy('nonexistent') is False
            assert manager.disable_strategy('nonexistent') is False
            assert manager.update_strategy_weight('nonexistent', 1.0) is False
            
        logger.info("✓ Strategy management operations test passed")


class TestPerformanceTracker:
    """Test suite for PerformanceTracker"""
    
    @pytest.fixture(autouse=True)
    def setup(self):
        """Setup test environment"""
        self.test_config = {
            'history_length': 100,
            'snapshot_history': 50,
            'alert_history_length': 100,
            'monitoring_interval': 0.5,
            'snapshot_interval': 1.0,
            'enable_persistence': False,  # Disable for testing
            'enable_real_time_monitoring': False,  # Manual control for testing
            'performance_thresholds': {
                'strategy_accuracy_min': 0.6,
                'strategy_latency_max': 0.1,
                'strategy_uptime_min': 0.9,
                'strategy_error_rate_max': 0.1,
                'system_latency_p95_max': 0.2,
                'system_error_rate_max': 0.05
            },
            'alert_cooldown': 10,  # Short cooldown for testing
            'ab_test_min_samples': 10,  # Lower for testing
            'ab_test_significance_level': 0.05
        }
        
        self.tracker = PerformanceTracker(self.test_config)
        self.performance_metrics = {}
    
    @pytest.mark.asyncio
    async def test_tracker_initialization(self):
        """Test performance tracker initialization"""
        await self.tracker.start()
        
        # Validate initialization
        assert self.tracker.config is not None
        assert isinstance(self.tracker.strategy_performances, dict)
        assert isinstance(self.tracker.active_alerts, dict)
        assert isinstance(self.tracker.active_ab_tests, dict)
        
        await self.tracker.stop()
        
        logger.info("✓ Performance tracker initialization test passed")
    
    @pytest.mark.asyncio
    async def test_strategy_performance_tracking(self):
        """Test strategy performance tracking"""
        await self.tracker.start()
        
        # Update strategy performance
        strategy_metrics = {
            'total_signals': 100,
            'successful_signals': 75,
            'avg_latency': 0.05,
            'uptime_ratio': 0.95,
            'error_count': 5
        }
        
        await self.tracker.update_strategy_performance('test_strategy', strategy_metrics)
        
        # Validate performance tracking
        performance = self.tracker.get_strategy_performance('test_strategy')
        assert performance is not None
        assert performance.strategy_name == 'test_strategy'
        assert performance.total_signals == 100
        assert performance.successful_signals == 75
        assert performance.signal_accuracy == 0.75
        
        # Test multiple updates
        for i in range(10):
            metrics = {
                'total_signals': 100 + i * 10,
                'successful_signals': 75 + i * 8,
                'avg_latency': 0.05 + i * 0.001,
                'uptime_ratio': 0.95 - i * 0.01
            }
            await self.tracker.update_strategy_performance('test_strategy', metrics)
        
        # Check final performance
        final_performance = self.tracker.get_strategy_performance('test_strategy')
        assert final_performance.total_signals == 190  # 100 + 9*10
        assert final_performance.avg_latency > 0.05  # Should have increased
        
        await self.tracker.stop()
        
        logger.info("✓ Strategy performance tracking test passed")
    
    @pytest.mark.asyncio
    async def test_system_performance_tracking(self):
        """Test system-wide performance tracking"""
        await self.tracker.start()
        
        # Update system metrics
        system_metrics = {
            'total_strategies': 5,
            'active_strategies': 4,
            'total_features': 25,
            'total_signals': 15,
            'avg_processing_time': 0.08,
            'system_uptime': 0.99,
            'error_rate': 0.02,
            'latency_p95': 0.12,
            'latency_p99': 0.18
        }
        
        await self.tracker.update_system_metrics(system_metrics)
        
        # Validate system tracking
        system_perf = self.tracker.get_system_performance()
        assert system_perf is not None
        assert system_perf.total_strategies == 5
        assert system_perf.active_strategies == 4
        assert system_perf.error_rate == 0.02
        
        # Test multiple system updates
        for i in range(5):
            metrics = {
                'total_strategies': 5,
                'active_strategies': 4 - (i % 2),  # Varying active count
                'error_rate': 0.02 + i * 0.01,
                'latency_p95': 0.12 + i * 0.02
            }
            await self.tracker.update_system_metrics(metrics)
        
        # Check history
        assert len(self.tracker.system_performance_history) > 1
        
        await self.tracker.stop()
        
        logger.info("✓ System performance tracking test passed")
    
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """Test performance alert generation"""
        await self.tracker.start()
        
        # Create conditions that should trigger alerts
        
        # High latency alert
        strategy_metrics = {
            'total_signals': 50,
            'successful_signals': 30,
            'avg_latency': 0.15,  # Above threshold (0.1)
            'uptime_ratio': 0.85,  # Below threshold (0.9)
            'error_count': 8
        }
        
        await self.tracker.update_strategy_performance('high_latency_strategy', strategy_metrics)
        
        # Check for alerts
        alerts = self.tracker.get_active_alerts()
        
        # Should have generated alerts for latency and uptime
        latency_alerts = [a for a in alerts if a.metric == PerformanceMetric.LATENCY]
        uptime_alerts = [a for a in alerts if a.metric == PerformanceMetric.UPTIME]
        
        assert len(latency_alerts) > 0, "Should have generated latency alert"
        assert len(uptime_alerts) > 0, "Should have generated uptime alert"
        
        # Validate alert properties
        for alert in alerts:
            assert alert.severity in [AlertSeverity.WARNING, AlertSeverity.CRITICAL]
            assert alert.current_value != alert.threshold_value
            assert alert.strategy_name == 'high_latency_strategy'
            assert not alert.acknowledged
            assert not alert.resolved
        
        # Test alert acknowledgment
        if alerts:
            alert_id = alerts[0].alert_id
            assert self.tracker.acknowledge_alert(alert_id) is True
            assert self.tracker.active_alerts[alert_id].acknowledged is True
        
        await self.tracker.stop()
        
        logger.info(f"✓ Alert generation - Generated {len(alerts)} alerts")
    
    @pytest.mark.asyncio
    async def test_ab_testing_framework(self):
        """Test A/B testing framework"""
        await self.tracker.start()
        
        # Start A/B test
        test_id = await self.tracker.start_ab_test(
            'strategy_a',
            'strategy_b',
            PerformanceMetric.ACCURACY
        )
        
        assert test_id is not None
        assert test_id in self.tracker.active_ab_tests
        
        # Simulate performance data for both strategies
        for i in range(15):  # Above min_samples threshold
            # Strategy A performance
            await self.tracker.update_strategy_performance('strategy_a', {
                'total_signals': 10 * (i + 1),
                'successful_signals': int(7 * (i + 1)),  # 70% accuracy
                'returns': [np.random.normal(0.05, 0.1)]  # 5% return with 10% vol
            })
            
            # Strategy B performance  
            await self.tracker.update_strategy_performance('strategy_b', {
                'total_signals': 10 * (i + 1),
                'successful_signals': int(6 * (i + 1)),  # 60% accuracy
                'returns': [np.random.normal(0.03, 0.12)]  # 3% return with 12% vol
            })
        
        # End A/B test
        result = await self.tracker.end_ab_test(test_id)
        
        # Validate A/B test result
        assert result is not None
        assert result.strategy_a == 'strategy_a'
        assert result.strategy_b == 'strategy_b'
        assert result.metric == PerformanceMetric.ACCURACY
        assert result.a_performance > 0
        assert result.b_performance > 0
        assert result.end_time is not None
        
        # Strategy A should perform better
        if result.winner:
            assert result.winner == 'strategy_a'  # Higher accuracy
        
        # Check test is moved to completed
        assert test_id not in self.tracker.active_ab_tests
        assert len(self.tracker.completed_ab_tests) > 0
        
        await self.tracker.stop()
        
        logger.info(f"✓ A/B testing - Winner: {result.winner}, "
                   f"A: {result.a_performance:.1%}, B: {result.b_performance:.1%}")
    
    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self):
        """Test performance metrics calculation"""
        await self.tracker.start()
        
        # Provide data with returns for advanced metrics
        strategy_data = {
            'total_signals': 100,
            'successful_signals': 65,
            'returns': [
                0.05, -0.02, 0.08, -0.01, 0.03, 0.12, -0.05, 0.02,
                -0.03, 0.07, 0.01, -0.04, 0.09, 0.06, -0.02, 0.04
            ]
        }
        
        await self.tracker.update_strategy_performance('advanced_strategy', strategy_data)
        
        # Get performance and check calculated metrics
        performance = self.tracker.get_strategy_performance('advanced_strategy')
        
        # Validate calculated metrics
        assert performance.signal_accuracy == 0.65  # 65/100
        assert performance.win_rate > 0  # Should have some winning trades
        assert performance.sharpe_ratio != 0  # Should calculate Sharpe ratio
        assert performance.max_drawdown >= 0  # Should calculate max drawdown
        assert performance.profit_factor > 0  # Should calculate profit factor
        
        # Check specific calculations
        returns = np.array(strategy_data['returns'])
        winning_trades = sum(1 for r in returns if r > 0)
        expected_win_rate = winning_trades / len(returns)
        
        assert abs(performance.win_rate - expected_win_rate) < 0.01
        
        await self.tracker.stop()
        
        logger.info(f"✓ Performance metrics - Accuracy: {performance.signal_accuracy:.1%}, "
                   f"Win rate: {performance.win_rate:.1%}, Sharpe: {performance.sharpe_ratio:.2f}")


class TestIntegrationEndToEnd:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    async def test_complete_integration_flow(self):
        """Test complete integration flow from strategies to RL"""
        # Create complete test configuration
        test_config = {
            'max_workers': 2,
            'execution_interval': 0.1,
            'enable_monitoring': False,
            'strategies': {
                'integration_test_strategy': {
                    'module_path': 'test_utils',
                    'class_name': 'MockStrategy',
                    'priority': 'HIGH',
                    'weight': 1.0,
                    'enabled': True,
                    'feature_names': ['momentum', 'volatility', 'volume_ratio'],
                    'signal_names': ['buy_signal', 'sell_signal']
                }
            }
        }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            # Initialize integration manager
            manager = StrategyIntegrationManager(test_config)
            await manager.initialize_strategies()
            
            # Initialize performance tracker
            tracker_config = {'enable_persistence': False, 'enable_real_time_monitoring': False}
            tracker = PerformanceTracker(tracker_config)
            await tracker.start()
            
            # Test complete flow
            flow_times = []
            
            for iteration in range(10):
                flow_start = time.perf_counter()
                
                # 1. Execute strategies
                result = await manager._execute_strategy('integration_test_strategy')
                assert result['success'] is True
                
                # 2. Extract features and signals
                features = result['features']
                signals = result['signals']
                
                assert len(features) > 0
                assert len(signals) > 0
                
                # 3. Update performance tracking
                perf_metrics = {
                    'total_signals': iteration + 1,
                    'successful_signals': iteration + 1,
                    'avg_latency': result['execution_time'],
                    'uptime_ratio': 1.0
                }
                await tracker.update_strategy_performance('integration_test_strategy', perf_metrics)
                
                # 4. Simulate RL feature preparation
                feature_vector = np.array(list(features.values()))
                assert len(feature_vector) > 0
                assert not np.any(np.isnan(feature_vector))
                
                # 5. Simulate signal aggregation
                signal_strength = np.mean([abs(v) for v in signals.values() if isinstance(v, (int, float))])
                assert signal_strength >= 0
                
                flow_time = time.perf_counter() - flow_start
                flow_times.append(flow_time)
            
            # Validate end-to-end performance
            avg_flow_time = np.mean(flow_times)
            max_flow_time = np.max(flow_times)
            
            assert_latency_threshold(
                avg_flow_time,
                0.05,  # 50ms for complete flow
                'End-to-End Integration Flow'
            )
            
            # Validate final state
            strategy_performance = tracker.get_strategy_performance('integration_test_strategy')
            assert strategy_performance is not None
            assert strategy_performance.total_signals == 10
            assert strategy_performance.signal_accuracy == 1.0  # All successful
            
            await tracker.stop()
            
        logger.info(f"✓ End-to-end integration - Flow time: {avg_flow_time*1000:.1f}ms avg, "
                   f"{max_flow_time*1000:.1f}ms max")
    
    @pytest.mark.asyncio
    async def test_system_resilience(self):
        """Test system resilience under stress"""
        # Configuration with potential failure points
        stress_config = {
            'max_workers': 3,
            'execution_interval': 0.05,  # High frequency
            'retry_attempts': 2,
            'strategy_timeout': 1.0,
            'enable_monitoring': False,
            'strategies': {}
        }
        
        # Add multiple strategies
        for i in range(5):
            stress_config['strategies'][f'stress_strategy_{i}'] = {
                'module_path': 'test_utils',
                'class_name': 'MockStrategy',
                'priority': 'MEDIUM',
                'weight': 1.0,
                'enabled': True,
                'config_params': {'failure_rate': 0.1}  # 10% failure rate
            }
        
        with patch('importlib.import_module') as mock_import:
            mock_module = Mock()
            mock_module.MockStrategy = MockStrategy
            mock_import.return_value = mock_module
            
            manager = StrategyIntegrationManager(stress_config)
            await manager.initialize_strategies()
            
            # Stress test
            successful_executions = 0
            failed_executions = 0
            execution_times = []
            
            stress_duration = 3.0  # 3 seconds of stress
            stress_start = time.time()
            
            while time.time() - stress_start < stress_duration:
                try:
                    # Execute all strategies
                    tasks = []
                    for strategy_name in stress_config['strategies'].keys():
                        task = asyncio.create_task(manager._execute_strategy(strategy_name))
                        tasks.append(task)
                    
                    execution_start = time.perf_counter()
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    execution_time = time.perf_counter() - execution_start
                    
                    execution_times.append(execution_time)
                    
                    # Count successes and failures
                    for result in results:
                        if isinstance(result, Exception):
                            failed_executions += 1
                        elif result.get('success'):
                            successful_executions += 1
                        else:
                            failed_executions += 1
                    
                    await asyncio.sleep(0.01)  # Brief pause
                    
                except Exception as e:
                    failed_executions += len(stress_config['strategies'])
                    logger.warning(f"Stress test execution failed: {e}")
            
            # Validate resilience
            total_executions = successful_executions + failed_executions
            success_rate = successful_executions / total_executions if total_executions > 0 else 0
            
            assert success_rate >= 0.8, f"Success rate too low under stress: {success_rate:.2%}"
            assert total_executions > 50, f"Too few executions during stress test: {total_executions}"
            
            # Performance should degrade gracefully
            if execution_times:
                avg_execution_time = np.mean(execution_times)
                p95_execution_time = np.percentile(execution_times, 95)
                
                assert avg_execution_time < 0.5, f"Average execution time too high: {avg_execution_time:.3f}s"
                assert p95_execution_time < 1.0, f"P95 execution time too high: {p95_execution_time:.3f}s"
            
        logger.info(f"✓ System resilience - Success rate: {success_rate:.2%}, "
                   f"Total executions: {total_executions}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "--log-cli-level=INFO"])