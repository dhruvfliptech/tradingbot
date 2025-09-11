"""
Comprehensive Test Suite for RL Trading Environment
Tests all components: TradingEnvironment, StateProcessor, PortfolioManager, MarketSimulator
"""

import pytest
import numpy as np
import pandas as pd
import gymnasium as gym
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import logging
import sys
import os

# Add the parent directory to the path so we can import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.trading_env import TradingEnvironment, RewardCalculator
from environment.state_processor import StateProcessor, TechnicalIndicatorEngine, MarketState
from environment.portfolio_manager import PortfolioManager, OrderSide, OrderType, Position
from environment.market_simulator import MarketSimulator, DataMode, MarketTick
from rl_config import RLConfig, RewardStrategy, ActionType

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestRLConfig:
    """Test RL configuration"""
    
    def test_config_creation(self):
        """Test configuration object creation"""
        config = RLConfig()
        
        assert config.environment == 'development'
        assert config.observation.lookback_window > 0
        assert config.action.action_space_size > 0
        assert config.env.initial_balance > 0
        assert config.reward.strategy in RewardStrategy
    
    def test_config_validation(self):
        """Test configuration validation"""
        config = RLConfig()
        
        # Should not raise an exception
        config._validate_config()
        
        # Test invalid configuration
        config.observation.lookback_window = 0
        with pytest.raises(ValueError):
            config._validate_config()


class TestStateProcessor:
    """Test StateProcessor functionality"""
    
    @pytest.fixture
    def sample_price_data(self):
        """Create sample price data for testing"""
        dates = pd.date_range(start='2024-01-01', end='2024-06-30', freq='H')
        n_periods = len(dates)
        
        # Generate synthetic price data
        np.random.seed(42)
        returns = np.random.normal(0.0001, 0.02, n_periods)
        prices = 50000 * np.exp(np.cumsum(returns))
        
        data = pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.005, n_periods))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.005, n_periods))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.5, n_periods)
        }, index=dates)
        
        return data
    
    def test_technical_indicator_engine(self, sample_price_data):
        """Test technical indicator computation"""
        engine = TechnicalIndicatorEngine()
        
        features = engine.compute_features(sample_price_data)
        
        assert not features.empty
        assert 'rsi_14' in features.columns
        assert 'sma_20' in features.columns
        assert 'macd' in features.columns
        
        # Check that features are numeric
        assert features.select_dtypes(include=[np.number]).shape[1] == features.shape[1]
    
    def test_state_processor_fitting(self, sample_price_data):
        """Test state processor fitting"""
        processor = StateProcessor(lookback_window=50)
        
        # Should fit without errors
        processor.fit(sample_price_data)
        
        assert processor.is_fitted
        assert len(processor.feature_names) > 0
    
    def test_state_processor_transform(self, sample_price_data):
        """Test state processor transformation"""
        processor = StateProcessor(lookback_window=50)
        processor.fit(sample_price_data)
        
        # Test transformation
        current_data = {
            'timestamp': datetime.now(),
            'price_data': {
                'open': 50000, 'high': 51000, 'low': 49000, 'close': 50500, 'volume': 1000000
            },
            'sentiment_data': {'fear_greed_index': 50, 'sentiment_score': 0.0},
            'alternative_data': {'funding_rate': 0.0001},
            'portfolio_state': {'cash_balance': 10000, 'portfolio_value': 10000}
        }
        
        market_state = processor.transform(current_data)
        
        assert isinstance(market_state, MarketState)
        assert len(market_state.normalized_features) > 0
        assert not np.any(np.isnan(market_state.normalized_features))


class TestPortfolioManager:
    """Test PortfolioManager functionality"""
    
    @pytest.fixture
    def portfolio_manager(self):
        """Create portfolio manager for testing"""
        return PortfolioManager(
            initial_balance=10000.0,
            commission_rate=0.001,
            slippage_rate=0.0005
        )
    
    def test_initial_state(self, portfolio_manager):
        """Test initial portfolio state"""
        assert portfolio_manager.cash_balance == 10000.0
        assert portfolio_manager.get_total_equity() == 10000.0
        assert len(portfolio_manager.positions) == 0
        assert len(portfolio_manager.active_orders) == 0
    
    def test_place_order(self, portfolio_manager):
        """Test order placement"""
        order = portfolio_manager.place_order(
            symbol='BTC/USD',
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET
        )
        
        assert order is not None
        assert order.symbol == 'BTC/USD'
        assert order.side == OrderSide.BUY
        assert order.quantity == 0.1
        assert len(portfolio_manager.active_orders) == 1
    
    def test_order_execution(self, portfolio_manager):
        """Test order execution"""
        # Place buy order
        order = portfolio_manager.place_order(
            symbol='BTC/USD',
            side=OrderSide.BUY,
            quantity=0.1,
            order_type=OrderType.MARKET
        )
        
        # Execute order
        execution_price = 50000.0
        success = portfolio_manager.execute_order(order, execution_price)
        
        assert success
        assert order.is_filled
        assert len(portfolio_manager.active_orders) == 0
        assert len(portfolio_manager.completed_trades) == 1
        
        # Check position
        position = portfolio_manager.get_position('BTC/USD')
        assert position.quantity == 0.1
        assert position.average_price == execution_price
        
        # Check cash balance
        expected_cash = 10000.0 - (0.1 * execution_price * (1 + 0.001))  # Include commission
        assert abs(portfolio_manager.cash_balance - expected_cash) < 0.01
    
    def test_position_tracking(self, portfolio_manager):
        """Test position tracking and P&L calculation"""
        # Buy position
        buy_order = portfolio_manager.place_order('BTC/USD', OrderSide.BUY, 0.1, OrderType.MARKET)
        portfolio_manager.execute_order(buy_order, 50000.0)
        
        # Update market price
        portfolio_manager.update_market_prices({'BTC/USD': 52000.0})
        
        position = portfolio_manager.get_position('BTC/USD')
        expected_unrealized_pnl = 0.1 * (52000.0 - 50000.0)
        
        assert abs(position.unrealized_pnl - expected_unrealized_pnl) < 0.01
        
        # Sell half position
        sell_order = portfolio_manager.place_order('BTC/USD', OrderSide.SELL, 0.05, OrderType.MARKET)
        portfolio_manager.execute_order(sell_order, 52000.0)
        
        # Check realized P&L
        assert portfolio_manager.get_realized_pnl() > 0
    
    def test_performance_metrics(self, portfolio_manager):
        """Test performance metrics calculation"""
        # Execute some trades
        buy_order = portfolio_manager.place_order('BTC/USD', OrderSide.BUY, 0.1, OrderType.MARKET)
        portfolio_manager.execute_order(buy_order, 50000.0)
        
        portfolio_manager.update_market_prices({'BTC/USD': 52000.0})
        
        sell_order = portfolio_manager.place_order('BTC/USD', OrderSide.SELL, 0.1, OrderType.MARKET)
        portfolio_manager.execute_order(sell_order, 52000.0)
        
        metrics = portfolio_manager.get_performance_metrics()
        
        assert 'total_return' in metrics
        assert 'total_trades' in metrics
        assert 'winning_trades' in metrics
        assert metrics['total_trades'] == 2
        assert metrics['winning_trades'] == 1


class TestMarketSimulator:
    """Test MarketSimulator functionality"""
    
    @pytest.fixture
    def market_simulator(self):
        """Create market simulator for testing"""
        return MarketSimulator(
            mode=DataMode.HISTORICAL,
            symbols=['BTC/USD', 'ETH/USD'],
            timeframe='1h'
        )
    
    def test_data_loading(self, market_simulator):
        """Test market data loading"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        
        success = market_simulator.load_data(start_date, end_date)
        
        assert success
        assert len(market_simulator.market_data) > 0
        assert 'BTC/USD' in market_simulator.market_data
        assert not market_simulator.market_data['BTC/USD'].empty
    
    def test_market_step(self, market_simulator):
        """Test market simulation step"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        market_simulator.load_data(start_date, end_date)
        
        ticks, conditions = market_simulator.step()
        
        assert len(ticks) > 0
        assert 'BTC/USD' in ticks
        assert isinstance(ticks['BTC/USD'], MarketTick)
        assert conditions.volatility >= 0
    
    def test_execution_price(self, market_simulator):
        """Test execution price calculation"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        market_simulator.load_data(start_date, end_date)
        
        # Step once to get current prices
        market_simulator.step()
        
        buy_price = market_simulator.get_execution_price('BTC/USD', True, 0.1)
        sell_price = market_simulator.get_execution_price('BTC/USD', False, 0.1)
        
        assert buy_price > 0
        assert sell_price > 0
        assert buy_price >= sell_price  # Buy price should be higher due to spread
    
    def test_market_reset(self, market_simulator):
        """Test market reset functionality"""
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        market_simulator.load_data(start_date, end_date)
        
        # Step a few times
        for _ in range(5):
            market_simulator.step()
        
        initial_step = market_simulator.current_step
        assert initial_step == 5
        
        # Reset
        market_simulator.reset(0)
        assert market_simulator.current_step == 0


class TestTradingEnvironment:
    """Test main TradingEnvironment class"""
    
    @pytest.fixture
    def trading_env(self):
        """Create trading environment for testing"""
        config = RLConfig()
        config.env.episode_length = 100  # Short episodes for testing
        config.env.initial_balance = 10000.0
        
        env = TradingEnvironment(config=config, mode='test')
        return env
    
    def test_environment_creation(self, trading_env):
        """Test environment creation"""
        assert isinstance(trading_env.action_space, gym.spaces.Discrete)
        assert isinstance(trading_env.observation_space, gym.spaces.Box)
        assert trading_env.action_space.n == len(ActionType)
    
    def test_environment_reset(self, trading_env):
        """Test environment reset"""
        # Load some data first
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        trading_env.load_data(start_date, end_date)
        
        obs, info = trading_env.reset()
        
        assert obs.shape == trading_env.observation_space.shape
        assert isinstance(info, dict)
        assert 'episode' in info
        assert 'portfolio_value' in info
        assert trading_env.current_step == 0
    
    def test_environment_step(self, trading_env):
        """Test environment step"""
        # Load data and reset
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        trading_env.load_data(start_date, end_date)
        
        obs, info = trading_env.reset()
        
        # Take a step
        action = ActionType.HOLD.value
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        assert obs.shape == trading_env.observation_space.shape
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        assert trading_env.current_step == 1
    
    def test_action_execution(self, trading_env):
        """Test action execution"""
        # Load data and reset
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        trading_env.load_data(start_date, end_date)
        
        obs, info = trading_env.reset()
        
        initial_cash = trading_env.portfolio_manager.cash_balance
        
        # Execute buy action
        action = ActionType.BUY_20.value
        obs, reward, terminated, truncated, info = trading_env.step(action)
        
        # Cash should have decreased (assuming order was executed)
        if info.get('order_executed', False):
            assert trading_env.portfolio_manager.cash_balance < initial_cash
    
    def test_episode_completion(self, trading_env):
        """Test complete episode"""
        # Load data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        trading_env.load_data(start_date, end_date)
        
        obs, info = trading_env.reset()
        
        total_reward = 0
        steps = 0
        
        while steps < 50:  # Limit steps for testing
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            
            total_reward += reward
            steps += 1
            
            if terminated or truncated:
                break
        
        assert steps > 0
        assert isinstance(total_reward, (int, float))
    
    def test_metrics_collection(self, trading_env):
        """Test metrics collection"""
        # Load data and run a short episode
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        trading_env.load_data(start_date, end_date)
        
        obs, info = trading_env.reset()
        
        for _ in range(10):
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            if terminated or truncated:
                break
        
        metrics = trading_env.get_metrics()
        
        assert 'total_episodes' in metrics
        assert 'total_steps' in metrics
        assert 'current_episode' in metrics
        assert 'portfolio_metrics' in metrics


class TestRewardCalculator:
    """Test reward calculation strategies"""
    
    @pytest.fixture
    def portfolio_manager(self):
        """Create portfolio manager for testing"""
        return PortfolioManager(initial_balance=10000.0)
    
    @pytest.fixture
    def reward_calculator(self):
        """Create reward calculator for testing"""
        from rl_config import RewardConfig
        config = RewardConfig()
        return RewardCalculator(RewardStrategy.RISK_ADJUSTED, config)
    
    def test_simple_return_reward(self, reward_calculator, portfolio_manager):
        """Test simple return reward calculation"""
        reward_calculator.strategy = RewardStrategy.SIMPLE_RETURN
        
        # Create mock state
        from environment.trading_env import EnvironmentState
        from environment.state_processor import MarketState
        from environment.market_simulator import MarketConditions, MarketRegime
        
        mock_state = EnvironmentState(
            market_state=MarketState(
                timestamp=datetime.now(),
                price_data={}, technical_indicators={}, sentiment_data={},
                portfolio_state={}, alternative_data={},
                raw_features=np.array([]), normalized_features=np.array([])
            ),
            portfolio_state=portfolio_manager.get_portfolio_state(),
            market_conditions=MarketConditions(
                regime=MarketRegime.SIDEWAYS, volatility=0.02, trend_strength=0.0,
                volume_profile=1.0, bid_ask_spread=0.001, market_impact_factor=0.001
            ),
            step_number=1, episode_number=1, timestamp=datetime.now()
        )
        
        # First call should return 0 (no previous equity)
        reward1 = reward_calculator.calculate_reward(mock_state, portfolio_manager, mock_state.market_conditions)
        assert reward1 == 0.0
        
        # Simulate profit
        portfolio_manager.cash_balance = 11000.0  # 10% gain
        
        reward2 = reward_calculator.calculate_reward(mock_state, portfolio_manager, mock_state.market_conditions)
        assert reward2 > 0


class TestIntegration:
    """Integration tests for the complete system"""
    
    def test_full_environment_workflow(self):
        """Test complete environment workflow"""
        # Create environment
        config = RLConfig()
        config.env.episode_length = 50
        config.env.initial_balance = 10000.0
        
        env = TradingEnvironment(config=config, mode='test')
        
        # Load data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 31)
        assert env.load_data(start_date, end_date)
        
        # Run multiple episodes
        for episode in range(3):
            obs, info = env.reset()
            episode_reward = 0
            
            for step in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                
                if terminated or truncated:
                    break
            
            logger.info(f"Episode {episode + 1}: Total reward = {episode_reward:.2f}")
        
        # Get final metrics
        metrics = env.get_metrics()
        assert metrics['total_episodes'] >= 3
        
        env.close()
    
    def test_environment_with_gymnasium_api(self):
        """Test environment compatibility with Gymnasium API"""
        config = RLConfig()
        config.env.episode_length = 20
        
        env = TradingEnvironment(config=config)
        
        # Load data
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 1, 15)
        env.load_data(start_date, end_date)
        
        # Test gymnasium API compliance
        assert hasattr(env, 'action_space')
        assert hasattr(env, 'observation_space')
        assert hasattr(env, 'reset')
        assert hasattr(env, 'step')
        assert hasattr(env, 'render')
        assert hasattr(env, 'close')
        
        # Test spaces
        assert isinstance(env.action_space, gym.spaces.Space)
        assert isinstance(env.observation_space, gym.spaces.Space)
        
        # Test reset
        obs, info = env.reset()
        assert env.observation_space.contains(obs)
        assert isinstance(info, dict)
        
        # Test step
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert env.observation_space.contains(obs)
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        env.close()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])