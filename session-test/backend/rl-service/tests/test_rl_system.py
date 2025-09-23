"""
End-to-End RL Trading System Tests
Comprehensive testing of the complete RL trading system including agents, environment, and integration
"""

import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import asyncio
import json
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.ppo_agent import PPOAgent
from agents.ensemble_agent import EnsembleAgent
from environment.trading_env import TradingEnvironment
from environment.portfolio_manager import PortfolioManager, OrderSide, OrderType
from integration.rl_service import RLService
from integration.decision_server import DecisionServer
from integration.trading_bridge import TradingBridge
from rewards.multi_objective_reward import MultiObjectiveReward
from pretrain.pretrain_pipeline import PretrainPipeline
from rl_config import RLConfig, RewardStrategy, ActionType

logger = logging.getLogger(__name__)


class TestRLSystemEndToEnd:
    """End-to-end system tests for the complete RL trading system"""

    @pytest.fixture
    def rl_config(self):
        """Create RL configuration for testing"""
        config = RLConfig()
        config.env.episode_length = 100
        config.env.initial_balance = 10000.0
        config.model.total_timesteps = 1000
        config.data.start_date = '2024-01-01'
        config.data.end_date = '2024-01-31'
        return config

    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
        np.random.seed(42)
        
        # Generate realistic price data
        returns = np.random.normal(0.0001, 0.02, len(dates))
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, len(dates))
        }, index=dates)
        
        return data

    @pytest.fixture
    def trading_environment(self, rl_config, mock_market_data):
        """Create trading environment with mock data"""
        env = TradingEnvironment(config=rl_config, mode='test')
        # Inject mock data
        env.market_simulator.market_data = {'BTC/USD': mock_market_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(mock_market_data) - 1
        return env

    def test_complete_system_workflow(self, trading_environment, rl_config):
        """Test complete RL system workflow from training to inference"""
        
        # 1. Initialize system components
        rl_service = RLService(config=rl_config)
        agent = PPOAgent(
            observation_space=trading_environment.observation_space,
            action_space=trading_environment.action_space,
            config=rl_config
        )
        
        # 2. Test training workflow
        logger.info("Testing training workflow...")
        initial_obs, _ = trading_environment.reset()
        
        # Simulate a short training episode
        episode_rewards = []
        for step in range(50):
            action = agent.predict(initial_obs, deterministic=False)[0]
            obs, reward, terminated, truncated, info = trading_environment.step(action)
            episode_rewards.append(reward)
            
            if terminated or truncated:
                break
            
            initial_obs = obs
        
        assert len(episode_rewards) > 10, "Training episode too short"
        assert isinstance(episode_rewards[0], (int, float)), "Invalid reward type"
        
        # 3. Test inference workflow
        logger.info("Testing inference workflow...")
        obs, _ = trading_environment.reset()
        action, _ = agent.predict(obs, deterministic=True)
        
        assert action in range(len(ActionType)), f"Invalid action: {action}"
        
        # 4. Test portfolio state consistency
        portfolio_state = trading_environment.portfolio_manager.get_portfolio_state()
        assert 'cash_balance' in portfolio_state
        assert 'positions' in portfolio_state
        assert portfolio_state['total_equity'] > 0
        
        # 5. Test metrics collection
        metrics = trading_environment.get_metrics()
        assert 'total_episodes' in metrics
        assert 'portfolio_metrics' in metrics
        
        logger.info("Complete system workflow test passed")

    def test_rl_service_integration(self, rl_config):
        """Test RLService integration and API functionality"""
        
        rl_service = RLService(config=rl_config)
        
        # Test service initialization
        assert rl_service.config == rl_config
        assert rl_service.agent is not None
        assert rl_service.environment is not None
        
        # Test training endpoint simulation
        training_result = rl_service.train_agent(timesteps=100)
        assert 'status' in training_result
        assert 'metrics' in training_result
        
        # Test prediction endpoint simulation
        market_data = {
            'timestamp': datetime.now().isoformat(),
            'price_data': {'open': 50000, 'high': 51000, 'low': 49000, 'close': 50500, 'volume': 1000000},
            'sentiment_data': {'fear_greed_index': 50, 'sentiment_score': 0.0},
            'alternative_data': {'funding_rate': 0.0001}
        }
        
        prediction = rl_service.predict_action(market_data)
        assert 'action' in prediction
        assert 'confidence' in prediction
        assert 'reasoning' in prediction
        
        # Test model saving/loading
        model_path = rl_service.save_model()
        assert os.path.exists(model_path)
        
        # Create new service and load model
        new_service = RLService(config=rl_config)
        load_success = new_service.load_model(model_path)
        assert load_success
        
        logger.info("RL service integration test passed")

    def test_decision_server_workflow(self, rl_config):
        """Test DecisionServer workflow and signal generation"""
        
        decision_server = DecisionServer(config=rl_config)
        
        # Mock market data input
        market_data = {
            'timestamp': datetime.now(),
            'symbol': 'BTC/USD',
            'price_data': {
                'open': 50000, 'high': 51000, 'low': 49000, 
                'close': 50500, 'volume': 1000000
            },
            'technical_indicators': {
                'rsi_14': 65.0, 'macd': 100.0, 'sma_20': 49800.0
            },
            'sentiment_data': {
                'fear_greed_index': 55, 'sentiment_score': 0.1
            }
        }
        
        # Test signal generation
        signal = decision_server.generate_signal(market_data)
        
        assert 'action' in signal
        assert 'confidence' in signal
        assert 'reasoning' in signal
        assert 'timestamp' in signal
        assert signal['action'] in [a.name for a in ActionType]
        assert 0 <= signal['confidence'] <= 1
        
        # Test batch signal processing
        batch_data = [market_data for _ in range(5)]
        batch_signals = decision_server.process_batch(batch_data)
        
        assert len(batch_signals) == 5
        assert all('action' in s for s in batch_signals)
        
        logger.info("Decision server workflow test passed")

    def test_trading_bridge_execution(self, rl_config):
        """Test TradingBridge order execution and management"""
        
        # Mock external trading API
        mock_api = Mock()
        mock_api.place_order.return_value = {'order_id': 'test_order_123', 'status': 'filled'}
        mock_api.get_order.return_value = {'order_id': 'test_order_123', 'status': 'filled', 'filled_qty': 0.1}
        mock_api.get_balance.return_value = {'cash': 10000.0, 'BTC': 0.0}
        
        trading_bridge = TradingBridge(config=rl_config, trading_api=mock_api)
        
        # Test order placement
        signal = {
            'action': 'BUY_20',
            'confidence': 0.8,
            'symbol': 'BTC/USD',
            'reasoning': 'Strong bullish signal'
        }
        
        order_result = trading_bridge.execute_signal(signal)
        
        assert 'order_id' in order_result
        assert 'status' in order_result
        assert order_result['status'] == 'filled'
        
        # Test order monitoring
        mock_api.place_order.assert_called_once()
        
        # Test risk management integration
        risk_check = trading_bridge.validate_signal(signal)
        assert isinstance(risk_check, bool)
        
        logger.info("Trading bridge execution test passed")

    def test_multi_objective_reward_system(self, trading_environment):
        """Test multi-objective reward calculation"""
        
        # Initialize multi-objective reward calculator
        reward_calculator = MultiObjectiveReward()
        
        # Simulate trading scenario
        obs, _ = trading_environment.reset()
        
        # Execute series of trades
        actions = [ActionType.BUY_20.value, ActionType.HOLD.value, ActionType.SELL_20.value]
        rewards = []
        
        for action in actions:
            obs, reward, terminated, truncated, info = trading_environment.step(action)
            rewards.append(reward)
            
            if terminated or truncated:
                break
        
        # Test reward components
        portfolio_state = trading_environment.portfolio_manager.get_portfolio_state()
        
        # Calculate individual reward components
        return_component = reward_calculator.calculate_return_reward(portfolio_state)
        risk_component = reward_calculator.calculate_risk_reward(portfolio_state)
        transaction_component = reward_calculator.calculate_transaction_cost_reward(portfolio_state)
        
        assert isinstance(return_component, (int, float))
        assert isinstance(risk_component, (int, float))
        assert isinstance(transaction_component, (int, float))
        
        # Test combined reward
        combined_reward = reward_calculator.calculate_combined_reward(portfolio_state)
        assert isinstance(combined_reward, (int, float))
        
        logger.info("Multi-objective reward system test passed")

    def test_ensemble_agent_coordination(self, trading_environment, rl_config):
        """Test ensemble agent coordination and decision making"""
        
        # Create ensemble agent with multiple sub-agents
        ensemble_agent = EnsembleAgent(
            observation_space=trading_environment.observation_space,
            action_space=trading_environment.action_space,
            config=rl_config
        )
        
        # Test individual agent predictions
        obs, _ = trading_environment.reset()
        
        individual_predictions = ensemble_agent.get_individual_predictions(obs)
        assert len(individual_predictions) == len(ensemble_agent.agents)
        
        # Test ensemble prediction
        ensemble_action, confidence = ensemble_agent.predict(obs, deterministic=True)
        assert ensemble_action in range(len(ActionType))
        assert 0 <= confidence <= 1
        
        # Test prediction consistency
        predictions = []
        for _ in range(10):
            action, _ = ensemble_agent.predict(obs, deterministic=True)
            predictions.append(action)
        
        # Deterministic predictions should be consistent
        assert len(set(predictions)) <= 2, "Deterministic predictions too variable"
        
        # Test agent voting mechanism
        votes = ensemble_agent.get_agent_votes(obs)
        assert len(votes) == len(ensemble_agent.agents)
        assert all(vote in range(len(ActionType)) for vote in votes)
        
        logger.info("Ensemble agent coordination test passed")

    def test_pretrain_pipeline_integration(self, rl_config):
        """Test pretraining pipeline integration with RL system"""
        
        # Mock Composer data
        mock_composer_data = {
            'patterns': [
                {'pattern_type': 'bullish_reversal', 'confidence': 0.8, 'context': 'oversold_rsi'},
                {'pattern_type': 'bearish_divergence', 'confidence': 0.7, 'context': 'overbought_macd'}
            ],
            'market_conditions': {
                'regime': 'trending',
                'volatility': 'medium',
                'volume_profile': 'high'
            }
        }
        
        pretrain_pipeline = PretrainPipeline(config=rl_config)
        
        # Test pattern extraction
        extracted_patterns = pretrain_pipeline.extract_patterns(mock_composer_data)
        assert len(extracted_patterns) > 0
        assert all('pattern_type' in p for p in extracted_patterns)
        
        # Test knowledge transfer preparation
        transfer_data = pretrain_pipeline.prepare_transfer_learning(extracted_patterns)
        assert 'features' in transfer_data
        assert 'targets' in transfer_data
        
        # Test integration with RL agent
        agent = PPOAgent(
            observation_space=trading_environment.observation_space,
            action_space=trading_environment.action_space,
            config=rl_config
        )
        
        # Simulate knowledge transfer
        transfer_success = pretrain_pipeline.apply_transfer_learning(agent, transfer_data)
        assert isinstance(transfer_success, bool)
        
        logger.info("Pretrain pipeline integration test passed")

    def test_system_recovery_and_failover(self, rl_config):
        """Test system recovery and failover mechanisms"""
        
        rl_service = RLService(config=rl_config)
        
        # Test graceful degradation when RL agent fails
        with patch.object(rl_service.agent, 'predict', side_effect=Exception("Agent failure")):
            market_data = {
                'timestamp': datetime.now().isoformat(),
                'price_data': {'close': 50000, 'volume': 1000000}
            }
            
            # Should fallback to safe default action
            prediction = rl_service.predict_action(market_data, use_fallback=True)
            assert prediction['action'] == 'HOLD'  # Safe fallback
            assert 'error' in prediction
        
        # Test service restart capability
        original_agent = rl_service.agent
        rl_service.restart_service()
        assert rl_service.agent is not None
        assert rl_service.agent != original_agent  # New agent instance
        
        # Test data pipeline resilience
        with patch('requests.get', side_effect=Exception("Data source failure")):
            # Should handle data source failures gracefully
            result = rl_service.health_check()
            assert 'data_source_status' in result
            assert result['data_source_status'] == 'degraded'
        
        logger.info("System recovery and failover test passed")

    def test_memory_and_performance_constraints(self, trading_environment, rl_config):
        """Test system performance under memory and computational constraints"""
        
        import psutil
        import time
        
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run intensive operations
        obs, _ = trading_environment.reset()
        start_time = time.time()
        
        # Simulate high-frequency trading scenario
        for i in range(100):
            action = trading_environment.action_space.sample()
            obs, reward, terminated, truncated, info = trading_environment.step(action)
            
            if terminated or truncated:
                obs, _ = trading_environment.reset()
        
        end_time = time.time()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Performance assertions
        execution_time = end_time - start_time
        memory_increase = final_memory - initial_memory
        
        assert execution_time < 30.0, f"Execution too slow: {execution_time:.2f}s"
        assert memory_increase < 100.0, f"Memory leak detected: {memory_increase:.2f}MB"
        
        # Test memory cleanup
        trading_environment.close()
        del trading_environment
        
        logger.info(f"Performance test passed: {execution_time:.2f}s, {memory_increase:.2f}MB")

    def test_concurrent_user_handling(self, rl_config):
        """Test system ability to handle multiple concurrent users"""
        
        import threading
        import queue
        
        results_queue = queue.Queue()
        
        def simulate_user_session(user_id):
            """Simulate a user trading session"""
            try:
                rl_service = RLService(config=rl_config)
                
                market_data = {
                    'timestamp': datetime.now().isoformat(),
                    'price_data': {'close': 50000, 'volume': 1000000},
                    'user_id': user_id
                }
                
                # Simulate trading decisions
                for _ in range(10):
                    prediction = rl_service.predict_action(market_data)
                    assert 'action' in prediction
                
                results_queue.put(f"User {user_id}: Success")
                
            except Exception as e:
                results_queue.put(f"User {user_id}: Error - {str(e)}")
        
        # Create multiple concurrent user sessions
        threads = []
        num_users = 5
        
        for user_id in range(num_users):
            thread = threading.Thread(target=simulate_user_session, args=(user_id,))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join(timeout=30)
        
        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())
        
        assert len(results) == num_users, "Not all user sessions completed"
        success_count = sum(1 for r in results if "Success" in r)
        assert success_count >= num_users * 0.8, "Too many user session failures"
        
        logger.info(f"Concurrent user handling test passed: {success_count}/{num_users} successful")

    def test_system_configuration_validation(self, rl_config):
        """Test system configuration validation and error handling"""
        
        # Test valid configuration
        valid_config = RLConfig()
        rl_service = RLService(config=valid_config)
        assert rl_service.config == valid_config
        
        # Test invalid configurations
        invalid_configs = [
            # Invalid lookback window
            {'observation': {'lookback_window': 0}},
            # Invalid episode length
            {'env': {'episode_length': -1}},
            # Invalid learning rate
            {'model': {'learning_rate': -0.1}},
            # Invalid position size
            {'env': {'max_position_size': 2.0}}
        ]
        
        for invalid_config in invalid_configs:
            with pytest.raises((ValueError, AssertionError)):
                config = RLConfig()
                # Apply invalid configuration
                for key, value in invalid_config.items():
                    if hasattr(config, key):
                        for subkey, subvalue in value.items():
                            setattr(getattr(config, key), subkey, subvalue)
                config._validate_config()
        
        logger.info("System configuration validation test passed")


class TestSystemMetricsAndMonitoring:
    """Test system metrics collection and monitoring"""

    def test_performance_metrics_collection(self, trading_environment):
        """Test comprehensive performance metrics collection"""
        
        # Run trading simulation
        obs, _ = trading_environment.reset()
        
        for _ in range(50):
            action = trading_environment.action_space.sample()
            obs, reward, terminated, truncated, info = trading_environment.step(action)
            
            if terminated or truncated:
                break
        
        # Collect system metrics
        metrics = trading_environment.get_metrics()
        
        # Validate core metrics
        required_metrics = [
            'total_episodes', 'total_steps', 'current_episode',
            'portfolio_metrics', 'performance_metrics'
        ]
        
        for metric in required_metrics:
            assert metric in metrics, f"Missing required metric: {metric}"
        
        # Validate portfolio metrics
        portfolio_metrics = metrics['portfolio_metrics']
        required_portfolio_metrics = [
            'total_equity', 'cash_balance', 'unrealized_pnl',
            'realized_pnl', 'total_return'
        ]
        
        for metric in required_portfolio_metrics:
            assert metric in portfolio_metrics, f"Missing portfolio metric: {metric}"
        
        logger.info("Performance metrics collection test passed")

    def test_real_time_monitoring_integration(self, rl_config):
        """Test real-time monitoring and alerting integration"""
        
        from integration.monitoring import MonitoringService
        
        monitoring = MonitoringService(config=rl_config)
        
        # Test metric ingestion
        test_metrics = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': 10500.0,
            'total_return': 0.05,
            'sharpe_ratio': 1.2,
            'drawdown': 0.02
        }
        
        monitoring.ingest_metrics(test_metrics)
        
        # Test alert triggering
        # Simulate performance degradation
        degraded_metrics = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': 8500.0,  # 15% drawdown
            'total_return': -0.15,
            'sharpe_ratio': -0.5,
            'drawdown': 0.15
        }
        
        alerts = monitoring.check_alerts(degraded_metrics)
        assert len(alerts) > 0, "No alerts triggered for performance degradation"
        
        # Validate alert content
        for alert in alerts:
            assert 'severity' in alert
            assert 'message' in alert
            assert 'timestamp' in alert
        
        logger.info("Real-time monitoring integration test passed")


if __name__ == "__main__":
    # Run tests with detailed output
    pytest.main([__file__, "-v", "--tb=short", "--capture=no"])