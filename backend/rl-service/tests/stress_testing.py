"""
Stress Testing Suite for RL Trading System
Tests edge cases, failure scenarios, and system resilience:
- Market crash scenarios
- Data feed interruptions
- High-frequency trading stress
- Memory and computational limits
- Network failures and recovery
- Concurrent user load testing
"""

import pytest
import numpy as np
import pandas as pd
import logging
import time
import threading
import queue
import gc
import psutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
from concurrent.futures import ThreadPoolExecutor, as_completed
import sys
import os
import asyncio
import requests
from contextlib import contextmanager

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.trading_env import TradingEnvironment
from environment.portfolio_manager import PortfolioManager
from agents.ppo_agent import PPOAgent
from agents.ensemble_agent import EnsembleAgent
from integration.rl_service import RLService
from integration.data_connector import DataConnector
from integration.trading_bridge import TradingBridge
from rl_config import RLConfig, ActionType

logger = logging.getLogger(__name__)


class StressTestMetrics:
    """Container for stress test metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.peak_memory_mb = 0
        self.avg_response_time = 0
        self.error_count = 0
        self.success_count = 0
        self.total_requests = 0
        self.cpu_usage_peak = 0
        self.recovery_time = 0
    
    def record_request(self, success: bool, response_time: float):
        """Record a request result"""
        self.total_requests += 1
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
        
        # Update average response time
        self.avg_response_time = (
            (self.avg_response_time * (self.total_requests - 1) + response_time) / self.total_requests
        )
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        
        self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
        self.cpu_usage_peak = max(self.cpu_usage_peak, cpu_percent)
    
    def get_summary(self) -> dict:
        """Get stress test summary"""
        duration = time.time() - self.start_time
        success_rate = self.success_count / self.total_requests if self.total_requests > 0 else 0
        
        return {
            'duration_seconds': duration,
            'total_requests': self.total_requests,
            'success_rate': success_rate,
            'error_rate': self.error_count / self.total_requests if self.total_requests > 0 else 0,
            'avg_response_time_ms': self.avg_response_time * 1000,
            'peak_memory_mb': self.peak_memory_mb,
            'peak_cpu_percent': self.cpu_usage_peak,
            'recovery_time_seconds': self.recovery_time
        }


class TestMarketCrashScenarios:
    """Test system behavior during extreme market conditions"""
    
    @pytest.fixture
    def crash_config(self):
        """Configuration for crash scenario testing"""
        config = RLConfig()
        config.env.initial_balance = 100000.0
        config.reward.max_drawdown_penalty = 0.2  # Higher penalty for drawdowns
        return config
    
    def generate_crash_scenario(self, crash_type: str, duration_hours: int = 24) -> pd.DataFrame:
        """Generate market crash scenario data"""
        
        # Base timeline
        dates = pd.date_range(
            start=datetime.now() - timedelta(hours=duration_hours*2),
            end=datetime.now(),
            freq='1H'
        )
        
        np.random.seed(42)  # Reproducible crashes
        n_periods = len(dates)
        
        if crash_type == "flash_crash":
            # Sudden 30% drop followed by partial recovery
            returns = np.random.normal(0.0001, 0.01, n_periods)
            crash_point = n_periods // 3
            returns[crash_point] = -0.30  # 30% flash crash
            returns[crash_point+1:crash_point+5] = np.random.normal(0.05, 0.02, 4)  # Partial recovery
            
        elif crash_type == "gradual_bear":
            # Gradual 50% decline over period
            trend = np.linspace(0, -0.50, n_periods)
            returns = np.random.normal(-0.002, 0.015, n_periods) + np.diff(np.concatenate([[0], trend]))
            
        elif crash_type == "volatility_spike":
            # Extreme volatility without clear direction
            volatility = np.linspace(0.01, 0.10, n_periods)
            returns = np.random.normal(0, volatility)
            
        elif crash_type == "black_swan":
            # Multiple sudden shocks
            returns = np.random.normal(0.0001, 0.01, n_periods)
            shock_points = [n_periods//4, n_periods//2, 3*n_periods//4]
            shock_magnitudes = [-0.15, -0.20, -0.10]
            for point, magnitude in zip(shock_points, shock_magnitudes):
                returns[point] = magnitude
                
        else:  # normal market
            returns = np.random.normal(0.0001, 0.02, n_periods)
        
        # Generate prices from returns
        initial_price = 50000
        prices = initial_price * np.exp(np.cumsum(returns))
        
        # Create OHLCV data with higher spreads during stress
        stress_factor = np.abs(returns) * 10 + 1  # Higher spreads during volatile periods
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003 * stress_factor, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003 * stress_factor, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.3 * stress_factor, n_periods)  # Higher volume during stress
        }, index=dates)
        
        # Ensure OHLC consistency
        data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
        data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
        
        return data
    
    def test_flash_crash_resilience(self, crash_config):
        """Test system resilience during flash crash"""
        
        crash_data = self.generate_crash_scenario("flash_crash", duration_hours=48)
        
        # Initialize system
        env = TradingEnvironment(config=crash_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': crash_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(crash_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=crash_config
        )
        
        # Run simulation through crash
        metrics = StressTestMetrics()
        obs, _ = env.reset()
        initial_equity = env.portfolio_manager.get_total_equity()
        
        portfolio_history = [initial_equity]
        action_history = []
        
        for step in range(len(crash_data) - 1):
            start_time = time.time()
            
            try:
                # Get agent decision
                action = agent.predict(obs, deterministic=True)[0]
                action_history.append(action)
                
                # Execute step
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Track portfolio
                current_equity = env.portfolio_manager.get_total_equity()
                portfolio_history.append(current_equity)
                
                metrics.record_request(True, time.time() - start_time)
                metrics.update_system_metrics()
                
                if terminated or truncated:
                    obs, _ = env.reset()
                
            except Exception as e:
                logger.error(f"Error during flash crash simulation: {e}")
                metrics.record_request(False, time.time() - start_time)
        
        # Analyze crash performance
        final_equity = portfolio_history[-1]
        max_equity = max(portfolio_history)
        min_equity = min(portfolio_history)
        
        max_drawdown = (max_equity - min_equity) / max_equity
        total_return = (final_equity - initial_equity) / initial_equity
        
        logger.info(f"Flash crash results: Return={total_return:.3f}, Max DD={max_drawdown:.3f}")
        
        # Validate crash resilience
        assert max_drawdown < 0.5, f"Excessive drawdown during flash crash: {max_drawdown:.3f}"
        assert total_return > -0.4, f"Excessive losses during flash crash: {total_return:.3f}"
        assert metrics.success_count / metrics.total_requests > 0.95, "Too many system failures during crash"
        
        # Check that agent adapted behavior during crash
        pre_crash_actions = action_history[:len(action_history)//3]
        post_crash_actions = action_history[len(action_history)//3:]
        
        # Should have more conservative actions after crash
        pre_crash_aggressive = sum(1 for a in pre_crash_actions if a in [ActionType.BUY_80.value, ActionType.BUY_100.value])
        post_crash_aggressive = sum(1 for a in post_crash_actions if a in [ActionType.BUY_80.value, ActionType.BUY_100.value])
        
        if len(pre_crash_actions) > 0 and len(post_crash_actions) > 0:
            pre_crash_aggression_rate = pre_crash_aggressive / len(pre_crash_actions)
            post_crash_aggression_rate = post_crash_aggressive / len(post_crash_actions)
            
            # Should be less aggressive after crash
            assert post_crash_aggression_rate <= pre_crash_aggression_rate * 1.5, \
                "Agent did not adapt to become more conservative after crash"
        
        logger.info("Flash crash resilience test passed")
    
    def test_gradual_bear_market_adaptation(self, crash_config):
        """Test adaptation during gradual bear market"""
        
        bear_data = self.generate_crash_scenario("gradual_bear", duration_hours=168)  # 1 week
        
        env = TradingEnvironment(config=crash_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': bear_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(bear_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=crash_config
        )
        
        # Run simulation
        obs, _ = env.reset()
        initial_equity = env.portfolio_manager.get_total_equity()
        
        portfolio_values = [initial_equity]
        position_sizes = []
        
        for step in range(len(bear_data) - 1):
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            portfolio_values.append(env.portfolio_manager.get_total_equity())
            
            # Track position sizing
            total_position_value = sum(
                pos.quantity * pos.current_price 
                for pos in env.portfolio_manager.positions.values()
            )
            position_size = total_position_value / env.portfolio_manager.get_total_equity()
            position_sizes.append(position_size)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        # Analyze bear market performance
        final_equity = portfolio_values[-1]
        total_return = (final_equity - initial_equity) / initial_equity
        
        # Should limit losses better than buy-and-hold
        market_return = (bear_data['close'].iloc[-1] - bear_data['close'].iloc[0]) / bear_data['close'].iloc[0]
        
        logger.info(f"Bear market: RL return={total_return:.3f}, Market return={market_return:.3f}")
        
        # Should outperform market during bear conditions
        assert total_return > market_return, \
            f"RL system ({total_return:.3f}) underperformed market ({market_return:.3f}) in bear market"
        
        # Should reduce position sizes during declining market
        early_positions = position_sizes[:len(position_sizes)//3]
        late_positions = position_sizes[2*len(position_sizes)//3:]
        
        if early_positions and late_positions:
            avg_early_position = np.mean(early_positions)
            avg_late_position = np.mean(late_positions)
            
            # Should reduce exposure in bear market
            assert avg_late_position <= avg_early_position * 1.2, \
                "System did not reduce exposure during bear market"
        
        logger.info("Gradual bear market adaptation test passed")
    
    def test_extreme_volatility_handling(self, crash_config):
        """Test handling of extreme volatility spikes"""
        
        volatile_data = self.generate_crash_scenario("volatility_spike", duration_hours=72)
        
        env = TradingEnvironment(config=crash_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': volatile_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(volatile_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=crash_config
        )
        
        # Run simulation with volatility monitoring
        obs, _ = env.reset()
        initial_equity = env.portfolio_manager.get_total_equity()
        
        metrics = StressTestMetrics()
        portfolio_volatility = []
        response_times = []
        
        previous_equity = initial_equity
        
        for step in range(len(volatile_data) - 1):
            start_time = time.time()
            
            try:
                action = agent.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                current_equity = env.portfolio_manager.get_total_equity()
                equity_change = (current_equity - previous_equity) / previous_equity
                portfolio_volatility.append(abs(equity_change))
                previous_equity = current_equity
                
                response_time = time.time() - start_time
                response_times.append(response_time)
                metrics.record_request(True, response_time)
                metrics.update_system_metrics()
                
                if terminated or truncated:
                    obs, _ = env.reset()
                
            except Exception as e:
                logger.error(f"Error during volatility test: {e}")
                metrics.record_request(False, time.time() - start_time)
        
        # Analyze volatility handling
        avg_portfolio_vol = np.mean(portfolio_volatility) if portfolio_volatility else 0
        max_response_time = max(response_times) if response_times else 0
        avg_response_time = np.mean(response_times) if response_times else 0
        
        logger.info(f"Volatility test: Avg portfolio vol={avg_portfolio_vol:.4f}, "
                   f"Max response time={max_response_time:.3f}s")
        
        # Performance criteria during extreme volatility
        assert avg_portfolio_vol < 0.1, f"Portfolio too volatile: {avg_portfolio_vol:.4f}"
        assert max_response_time < 1.0, f"Response time too slow during volatility: {max_response_time:.3f}s"
        assert metrics.success_count / metrics.total_requests > 0.90, \
            "Too many failures during volatility"
        
        # System should remain stable
        final_equity = env.portfolio_manager.get_total_equity()
        total_return = (final_equity - initial_equity) / initial_equity
        assert total_return > -0.3, f"Excessive losses during volatility: {total_return:.3f}"
        
        logger.info("Extreme volatility handling test passed")


class TestDataFeedStressTests:
    """Test system behavior under data feed stress conditions"""
    
    @pytest.fixture
    def data_stress_config(self):
        """Configuration for data stress testing"""
        config = RLConfig()
        config.data.enable_data_cache = True
        config.data.cache_duration_hours = 1
        return config
    
    def test_data_feed_interruption(self, data_stress_config):
        """Test behavior when data feeds are interrupted"""
        
        # Create mock data connector with intermittent failures
        connector = DataConnector(config=data_stress_config)
        
        # Track success/failure rates
        request_count = 0
        success_count = 0
        fallback_count = 0
        
        def mock_fetch_with_failures(*args, **kwargs):
            nonlocal request_count, success_count, fallback_count
            request_count += 1
            
            # Simulate 20% failure rate
            if np.random.random() < 0.2:
                if request_count % 3 == 0:
                    # Complete failure
                    raise requests.exceptions.ConnectionError("Data feed unavailable")
                else:
                    # Timeout
                    raise requests.exceptions.Timeout("Data feed timeout")
            else:
                success_count += 1
                return {
                    'timestamp': datetime.now(),
                    'price_data': {'close': 50000 + np.random.normal(0, 1000)},
                    'sentiment_data': {'sentiment_score': np.random.normal(0, 0.1)},
                    'alternative_data': {'funding_rate': np.random.normal(0.0001, 0.00005)}
                }
        
        # Patch data fetching methods
        with patch.object(connector, 'fetch_complete_dataset', side_effect=mock_fetch_with_failures):
            
            # Simulate 100 data requests
            for i in range(100):
                try:
                    data = connector.fetch_complete_dataset('BTC/USD')
                    
                    # Validate data structure even during stress
                    assert 'price_data' in data
                    assert 'timestamp' in data
                    
                except Exception as e:
                    # Should gracefully handle failures with fallback
                    fallback_data = connector.get_fallback_data('BTC/USD')
                    fallback_count += 1
                    
                    assert fallback_data is not None, "No fallback data available"
                    assert 'price_data' in fallback_data
        
        # Analyze data feed stress results
        success_rate = success_count / request_count
        fallback_rate = fallback_count / request_count
        
        logger.info(f"Data feed stress: {success_rate:.1%} success, {fallback_rate:.1%} fallback")
        
        # Should handle failures gracefully
        assert success_rate >= 0.7, f"Data feed success rate too low: {success_rate:.1%}"
        assert fallback_rate <= 0.3, f"Too many fallbacks needed: {fallback_rate:.1%}"
        assert success_count + fallback_count >= request_count * 0.95, \
            "Total data availability too low"
        
        logger.info("Data feed interruption test passed")
    
    def test_high_frequency_data_requests(self, data_stress_config):
        """Test system under high-frequency data request load"""
        
        connector = DataConnector(config=data_stress_config)
        metrics = StressTestMetrics()
        
        # Mock fast data source
        def mock_fast_fetch(*args, **kwargs):
            # Simulate processing time
            time.sleep(0.001 + np.random.exponential(0.002))  # 1-3ms average
            return {
                'timestamp': datetime.now(),
                'price_data': {'close': 50000 + np.random.normal(0, 100)},
                'volume': np.random.lognormal(15, 0.1)
            }
        
        with patch.object(connector, 'fetch_price_data', side_effect=mock_fast_fetch):
            
            # High-frequency requests (10 requests per second for 30 seconds)
            start_time = time.time()
            target_duration = 30  # seconds
            target_rps = 10  # requests per second
            
            request_times = []
            
            while time.time() - start_time < target_duration:
                request_start = time.time()
                
                try:
                    data = connector.fetch_price_data('BTC/USD', datetime.now() - timedelta(hours=1), datetime.now())
                    
                    request_time = time.time() - request_start
                    request_times.append(request_time)
                    metrics.record_request(True, request_time)
                    
                    # Validate data
                    assert 'price_data' in data
                    
                except Exception as e:
                    logger.warning(f"High-frequency request failed: {e}")
                    metrics.record_request(False, time.time() - request_start)
                
                metrics.update_system_metrics()
                
                # Control request rate
                time.sleep(1.0 / target_rps)
        
        # Analyze high-frequency performance
        summary = metrics.get_summary()
        
        logger.info(f"High-frequency test: {summary['success_rate']:.1%} success, "
                   f"{summary['avg_response_time_ms']:.1f}ms avg response")
        
        # Performance criteria
        assert summary['success_rate'] >= 0.95, f"Success rate too low: {summary['success_rate']:.1%}"
        assert summary['avg_response_time_ms'] < 50, f"Response time too slow: {summary['avg_response_time_ms']:.1f}ms"
        assert summary['peak_memory_mb'] < 500, f"Memory usage too high: {summary['peak_memory_mb']:.1f}MB"
        
        # Request rate should be consistent
        if request_times:
            response_time_std = np.std(request_times)
            assert response_time_std < 0.1, f"Response time variability too high: {response_time_std:.3f}s"
        
        logger.info("High-frequency data requests test passed")
    
    def test_malformed_data_handling(self, data_stress_config):
        """Test handling of malformed or corrupted data"""
        
        connector = DataConnector(config=data_stress_config)
        
        # Define various malformed data scenarios
        malformed_scenarios = [
            {},  # Empty data
            {'price_data': None},  # Null price data
            {'price_data': {'close': 'invalid'}},  # Invalid price type
            {'price_data': {'close': float('inf')}},  # Infinite price
            {'price_data': {'close': float('nan')}},  # NaN price
            {'price_data': {'close': -50000}},  # Negative price
            {'timestamp': 'invalid_timestamp'},  # Invalid timestamp
            {'price_data': {'close': 50000}, 'volume': -1000000},  # Negative volume
        ]
        
        validation_success_count = 0
        fallback_success_count = 0
        
        for i, malformed_data in enumerate(malformed_scenarios):
            
            def mock_malformed_fetch(*args, **kwargs):
                return malformed_data
            
            with patch.object(connector, 'fetch_complete_dataset', side_effect=mock_malformed_fetch):
                try:
                    # Should either validate and fix data or use fallback
                    validated_data = connector.get_validated_data('BTC/USD')
                    
                    # Validate that returned data is clean
                    assert validated_data is not None
                    assert 'price_data' in validated_data
                    
                    if 'close' in validated_data['price_data']:
                        price = validated_data['price_data']['close']
                        assert isinstance(price, (int, float))
                        assert not np.isnan(price) and not np.isinf(price)
                        assert price > 0
                    
                    validation_success_count += 1
                    
                except Exception as e:
                    # Should fall back to cached or default data
                    try:
                        fallback_data = connector.get_fallback_data('BTC/USD')
                        assert fallback_data is not None
                        fallback_success_count += 1
                        
                    except Exception as fallback_error:
                        logger.error(f"Both validation and fallback failed for scenario {i}: {fallback_error}")
        
        total_scenarios = len(malformed_scenarios)
        total_success_rate = (validation_success_count + fallback_success_count) / total_scenarios
        
        logger.info(f"Malformed data handling: {validation_success_count}/{total_scenarios} validated, "
                   f"{fallback_success_count}/{total_scenarios} fallback")
        
        # Should handle all malformed data scenarios
        assert total_success_rate >= 0.9, f"Poor malformed data handling: {total_success_rate:.1%}"
        
        logger.info("Malformed data handling test passed")


class TestSystemResourceStressTests:
    """Test system behavior under resource constraints"""
    
    @pytest.fixture
    def resource_config(self):
        """Configuration for resource stress testing"""
        config = RLConfig()
        config.env.episode_length = 1000
        return config
    
    def test_memory_stress_limits(self, resource_config):
        """Test system behavior under memory constraints"""
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Create multiple environments to stress memory
        environments = []
        agents = []
        
        try:
            # Create environments until memory limit
            for i in range(20):  # Limit for testing
                env = TradingEnvironment(config=resource_config, mode='test')
                agent = PPOAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    config=resource_config
                )
                
                environments.append(env)
                agents.append(agent)
                
                current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                memory_increase = current_memory - initial_memory
                
                # Stop if memory usage gets too high
                if memory_increase > 1000:  # 1GB limit
                    break
                
                # Test basic functionality
                obs, _ = env.reset()
                action = agent.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                assert not np.any(np.isnan(obs)), f"NaN in observation at environment {i}"
                assert isinstance(reward, (int, float)), f"Invalid reward type at environment {i}"
        
        finally:
            # Cleanup
            for env in environments:
                env.close()
            
            del environments, agents
            gc.collect()
        
        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_cleaned = initial_memory + 50  # Allow 50MB overhead
        
        logger.info(f"Memory test: Initial={initial_memory:.1f}MB, Peak={final_memory:.1f}MB")
        
        # Memory should be properly cleaned up
        assert final_memory < memory_cleaned, \
            f"Memory leak detected: {final_memory:.1f}MB vs expected {memory_cleaned:.1f}MB"
        
        logger.info("Memory stress limits test passed")
    
    def test_cpu_intensive_operations(self, resource_config):
        """Test CPU-intensive operations and performance"""
        
        # Create ensemble agent (more CPU intensive)
        env = TradingEnvironment(config=resource_config, mode='test')
        ensemble_agent = EnsembleAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=resource_config
        )
        
        # Generate complex market data
        dates = pd.date_range(start=datetime.now() - timedelta(hours=100), 
                             end=datetime.now(), freq='1H')
        complex_data = pd.DataFrame({
            'open': 50000 + np.random.normal(0, 1000, len(dates)),
            'high': 50000 + np.random.normal(500, 1000, len(dates)),
            'low': 50000 + np.random.normal(-500, 1000, len(dates)),
            'close': 50000 + np.random.normal(0, 1000, len(dates)),
            'volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        env.market_simulator.market_data = {'BTC/USD': complex_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(complex_data) - 1
        
        # Run CPU-intensive simulation
        start_time = time.time()
        obs, _ = env.reset()
        
        cpu_times = []
        response_times = []
        
        for step in range(100):  # Limited for testing
            step_start = time.time()
            
            # CPU-intensive prediction
            action = ensemble_agent.predict(obs, deterministic=False)[0]  # Non-deterministic is more intensive
            obs, reward, terminated, truncated, info = env.step(action)
            
            step_time = time.time() - step_start
            response_times.append(step_time)
            
            # Monitor CPU usage
            cpu_percent = psutil.Process().cpu_percent()
            cpu_times.append(cpu_percent)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        total_time = time.time() - start_time
        avg_step_time = np.mean(response_times)
        max_step_time = max(response_times)
        avg_cpu = np.mean(cpu_times)
        
        logger.info(f"CPU test: Avg step time={avg_step_time:.3f}s, Max={max_step_time:.3f}s, "
                   f"Avg CPU={avg_cpu:.1f}%")
        
        # Performance criteria
        assert avg_step_time < 0.5, f"Average step time too slow: {avg_step_time:.3f}s"
        assert max_step_time < 2.0, f"Maximum step time too slow: {max_step_time:.3f}s"
        assert total_time < 200, f"Total simulation time too long: {total_time:.1f}s"
        
        # CPU usage should be reasonable
        assert avg_cpu < 90, f"CPU usage too high: {avg_cpu:.1f}%"
        
        logger.info("CPU intensive operations test passed")
    
    @contextmanager
    def limit_memory(self, limit_mb: int):
        """Context manager to limit memory usage (simulation)"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024
        
        def check_memory():
            current_memory = process.memory_info().rss / 1024 / 1024
            if current_memory - initial_memory > limit_mb:
                raise MemoryError(f"Memory limit exceeded: {current_memory:.1f}MB")
        
        try:
            yield check_memory
        finally:
            gc.collect()
    
    def test_memory_limited_environment(self, resource_config):
        """Test system behavior with memory constraints"""
        
        memory_limit_mb = 200  # 200MB limit
        
        with self.limit_memory(memory_limit_mb) as memory_checker:
            env = TradingEnvironment(config=resource_config, mode='test')
            agent = PPOAgent(
                observation_space=env.observation_space,
                action_space=env.action_space,
                config=resource_config
            )
            
            obs, _ = env.reset()
            
            # Run simulation with memory monitoring
            for step in range(200):
                try:
                    # Check memory before intensive operations
                    memory_checker()
                    
                    action = agent.predict(obs, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if step % 50 == 0:
                        # Force garbage collection periodically
                        gc.collect()
                        memory_checker()
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                
                except MemoryError as e:
                    logger.warning(f"Memory limit reached at step {step}: {e}")
                    # Should gracefully handle memory constraints
                    gc.collect()
                    break
        
        logger.info("Memory limited environment test passed")


class TestConcurrentUserLoadTests:
    """Test system behavior under concurrent user loads"""
    
    @pytest.fixture
    def load_config(self):
        """Configuration for load testing"""
        config = RLConfig()
        config.env.episode_length = 100  # Shorter episodes for load testing
        return config
    
    def simulate_user_session(self, user_id: int, duration_seconds: int, results_queue: queue.Queue):
        """Simulate a single user session"""
        session_metrics = StressTestMetrics()
        
        try:
            config = RLConfig()
            config.env.episode_length = 50  # Fast episodes
            
            env = TradingEnvironment(config=config, mode='test')
            agent = PPOAgent(
                observation_space=env.observation_space,
                action_space=env.action_space,
                config=config
            )
            
            obs, _ = env.reset()
            start_time = time.time()
            
            while time.time() - start_time < duration_seconds:
                request_start = time.time()
                
                try:
                    action = agent.predict(obs, deterministic=True)[0]
                    obs, reward, terminated, truncated, info = env.step(action)
                    
                    if terminated or truncated:
                        obs, _ = env.reset()
                    
                    session_metrics.record_request(True, time.time() - request_start)
                    session_metrics.update_system_metrics()
                    
                except Exception as e:
                    session_metrics.record_request(False, time.time() - request_start)
                    logger.warning(f"User {user_id} error: {e}")
                
                # Small delay to prevent overwhelming
                time.sleep(0.01)
            
            results_queue.put({
                'user_id': user_id,
                'status': 'completed',
                'metrics': session_metrics.get_summary()
            })
            
        except Exception as e:
            results_queue.put({
                'user_id': user_id,
                'status': 'failed',
                'error': str(e)
            })
        
        finally:
            env.close()
    
    def test_concurrent_user_load(self, load_config):
        """Test system under concurrent user load"""
        
        num_users = 10
        session_duration = 30  # seconds
        results_queue = queue.Queue()
        
        # Start concurrent user sessions
        threads = []
        for user_id in range(num_users):
            thread = threading.Thread(
                target=self.simulate_user_session,
                args=(user_id, session_duration, results_queue)
            )
            threads.append(thread)
            thread.start()
        
        # Monitor system resources during load test
        system_metrics = []
        monitor_start = time.time()
        
        while any(t.is_alive() for t in threads):
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            cpu_percent = process.cpu_percent(interval=1)
            
            system_metrics.append({
                'timestamp': time.time() - monitor_start,
                'memory_mb': memory_mb,
                'cpu_percent': cpu_percent
            })
            
            time.sleep(2)
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=60)
        
        # Collect results
        user_results = []
        while not results_queue.empty():
            user_results.append(results_queue.get())
        
        # Analyze load test results
        completed_users = [r for r in user_results if r['status'] == 'completed']
        failed_users = [r for r in user_results if r['status'] == 'failed']
        
        completion_rate = len(completed_users) / num_users
        
        if completed_users:
            avg_success_rate = np.mean([r['metrics']['success_rate'] for r in completed_users])
            avg_response_time = np.mean([r['metrics']['avg_response_time_ms'] for r in completed_users])
        else:
            avg_success_rate = 0
            avg_response_time = float('inf')
        
        # System resource analysis
        if system_metrics:
            peak_memory = max(m['memory_mb'] for m in system_metrics)
            peak_cpu = max(m['cpu_percent'] for m in system_metrics)
            avg_memory = np.mean([m['memory_mb'] for m in system_metrics])
            avg_cpu = np.mean([m['cpu_percent'] for m in system_metrics])
        else:
            peak_memory = peak_cpu = avg_memory = avg_cpu = 0
        
        logger.info(f"Load test: {completion_rate:.1%} completion, {avg_success_rate:.1%} success rate, "
                   f"{avg_response_time:.1f}ms avg response")
        logger.info(f"Resources: Peak memory={peak_memory:.1f}MB, Peak CPU={peak_cpu:.1f}%")
        
        # Performance criteria
        assert completion_rate >= 0.8, f"User completion rate too low: {completion_rate:.1%}"
        assert avg_success_rate >= 0.9, f"Average success rate too low: {avg_success_rate:.1%}"
        assert avg_response_time < 1000, f"Average response time too slow: {avg_response_time:.1f}ms"
        
        # Resource usage should be reasonable
        assert peak_memory < 2000, f"Peak memory usage too high: {peak_memory:.1f}MB"
        assert peak_cpu < 95, f"Peak CPU usage too high: {peak_cpu:.1f}%"
        
        logger.info("Concurrent user load test passed")
    
    def test_burst_load_handling(self, load_config):
        """Test system handling of sudden burst loads"""
        
        # Simulate burst load: many requests in short time
        burst_size = 50
        burst_duration = 5  # seconds
        
        results_queue = queue.Queue()
        
        def burst_request(request_id):
            """Single burst request"""
            try:
                start_time = time.time()
                
                config = RLConfig()
                env = TradingEnvironment(config=config, mode='test')
                agent = PPOAgent(
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    config=config
                )
                
                obs, _ = env.reset()
                action = agent.predict(obs, deterministic=True)[0]
                obs, reward, terminated, truncated, info = env.step(action)
                
                response_time = time.time() - start_time
                
                results_queue.put({
                    'request_id': request_id,
                    'status': 'success',
                    'response_time': response_time
                })
                
            except Exception as e:
                results_queue.put({
                    'request_id': request_id,
                    'status': 'error',
                    'error': str(e)
                })
            
            finally:
                env.close()
        
        # Launch burst requests
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(burst_request, i) for i in range(burst_size)]
            
            # Monitor completion
            completed = 0
            for future in as_completed(futures, timeout=30):
                completed += 1
                if completed % 10 == 0:
                    logger.info(f"Burst progress: {completed}/{burst_size}")
        
        # Collect results
        burst_results = []
        while not results_queue.empty():
            burst_results.append(results_queue.get())
        
        # Analyze burst performance
        successful_requests = [r for r in burst_results if r['status'] == 'success']
        failed_requests = [r for r in burst_results if r['status'] == 'error']
        
        success_rate = len(successful_requests) / len(burst_results) if burst_results else 0
        
        if successful_requests:
            response_times = [r['response_time'] for r in successful_requests]
            avg_response_time = np.mean(response_times)
            max_response_time = max(response_times)
        else:
            avg_response_time = max_response_time = float('inf')
        
        logger.info(f"Burst test: {success_rate:.1%} success, "
                   f"{avg_response_time:.3f}s avg response, {max_response_time:.3f}s max")
        
        # Burst handling criteria
        assert success_rate >= 0.8, f"Burst success rate too low: {success_rate:.1%}"
        assert avg_response_time < 2.0, f"Average burst response time too slow: {avg_response_time:.3f}s"
        assert max_response_time < 5.0, f"Maximum burst response time too slow: {max_response_time:.3f}s"
        
        logger.info("Burst load handling test passed")


if __name__ == "__main__":
    # Run stress tests
    pytest.main([__file__, "-v", "--tb=short", "-s"])