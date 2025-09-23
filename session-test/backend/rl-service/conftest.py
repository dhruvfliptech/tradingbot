"""
Global pytest configuration and fixtures for RL Trading System tests
"""

import pytest
import numpy as np
import pandas as pd
import logging
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
from pathlib import Path
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rl_config import RLConfig
from environment.trading_env import TradingEnvironment
from environment.portfolio_manager import PortfolioManager
from agents.ppo_agent import PPOAgent


# Configure logging for tests
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose logging from external libraries during tests
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


@pytest.fixture(scope="session")
def test_config():
    """Test configuration for RL system"""
    config = RLConfig()
    
    # Override for testing
    config.environment = 'testing'
    config.debug = True
    config.env.episode_length = 100  # Shorter episodes for faster tests
    config.env.initial_balance = 10000.0
    config.model.total_timesteps = 1000  # Reduced for faster training
    config.data.start_date = '2024-01-01'
    config.data.end_date = '2024-01-31'
    config.observation.lookback_window = 20  # Reduced for faster processing
    
    # Test-specific overrides
    config.reward.reward_scaling = 1.0  # Simpler scaling for tests
    config.env.enable_market_impact = False  # Disable for deterministic tests
    
    return config


@pytest.fixture(scope="session")
def test_data_dir():
    """Create and provide temporary directory for test data"""
    temp_dir = tempfile.mkdtemp(prefix="rl_test_")
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_market_data():
    """Generate sample market data for testing"""
    np.random.seed(42)  # Reproducible data
    
    # Generate 30 days of hourly data
    dates = pd.date_range(start='2024-01-01', end='2024-01-31', freq='1H')
    n_periods = len(dates)
    
    # Generate realistic price movements
    returns = np.random.normal(0.0001, 0.02, n_periods)
    initial_price = 50000
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Create OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_periods))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.3, n_periods)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


@pytest.fixture
def mock_trading_environment(test_config, sample_market_data):
    """Create mock trading environment for testing"""
    env = TradingEnvironment(config=test_config, mode='test')
    
    # Inject sample data
    env.market_simulator.market_data = {'BTC/USD': sample_market_data}
    env.market_simulator.current_step = 0
    env.market_simulator.max_steps = len(sample_market_data) - 1
    
    return env


@pytest.fixture
def mock_portfolio_manager(test_config):
    """Create mock portfolio manager for testing"""
    return PortfolioManager(
        initial_balance=test_config.env.initial_balance,
        commission_rate=0.001,
        slippage_rate=0.0005
    )


@pytest.fixture
def mock_ppo_agent(test_config, mock_trading_environment):
    """Create mock PPO agent for testing"""
    return PPOAgent(
        observation_space=mock_trading_environment.observation_space,
        action_space=mock_trading_environment.action_space,
        config=test_config
    )


@pytest.fixture
def mock_external_apis():
    """Mock external API responses for testing"""
    mocks = {}
    
    # Mock data provider APIs
    mocks['alpaca_response'] = {
        'bars': [{
            'c': 50000, 'h': 51000, 'l': 49000, 'o': 49500,
            'v': 1000000, 't': '2024-01-15T12:00:00Z'
        }]
    }
    
    mocks['groq_response'] = {
        'sentiment_score': 0.1,
        'fear_greed_index': 55,
        'confidence': 0.8
    }
    
    mocks['coinglass_response'] = {
        'funding_rate': 0.0001,
        'open_interest': 25000000000,
        'long_short_ratio': 1.2
    }
    
    return mocks


@pytest.fixture
def mock_database_connection():
    """Mock database connection for testing"""
    with patch('sqlalchemy.create_engine') as mock_engine:
        mock_conn = Mock()
        mock_engine.return_value.connect.return_value.__enter__.return_value = mock_conn
        mock_conn.execute.return_value.fetchall.return_value = []
        mock_conn.execute.return_value.fetchone.return_value = None
        yield mock_conn


@pytest.fixture(scope="function", autouse=True)
def reset_random_seeds():
    """Reset random seeds before each test for reproducibility"""
    np.random.seed(42)
    import random
    random.seed(42)


@pytest.fixture
def performance_test_data():
    """Generate data specifically for performance testing"""
    np.random.seed(42)
    
    # Generate 6 months of data for longer-term performance testing
    dates = pd.date_range(start='2023-07-01', end='2024-01-01', freq='1H')
    n_periods = len(dates)
    
    # More complex market dynamics for performance testing
    regime_changes = np.random.choice([0, 1], n_periods, p=[0.995, 0.005])
    volatility_multipliers = np.where(regime_changes, 2.0, 1.0)
    
    returns = np.random.normal(0.0002, 0.025 * volatility_multipliers, n_periods)
    
    # Add autocorrelation for more realistic price movements
    for i in range(1, len(returns)):
        returns[i] += 0.03 * returns[i-1]
    
    # Generate prices
    initial_price = 35000
    prices = initial_price * np.exp(np.cumsum(returns))
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.004, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.004, n_periods))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.4, n_periods)
    }, index=dates)
    
    # Ensure OHLC consistency
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


@pytest.fixture
def stress_test_scenarios():
    """Generate various stress test scenarios"""
    scenarios = {}
    
    # Flash crash scenario
    dates = pd.date_range(start='2024-01-01', end='2024-01-02', freq='1H')
    returns = np.random.normal(0.0001, 0.01, len(dates))
    crash_point = len(dates) // 2
    returns[crash_point] = -0.20  # 20% flash crash
    
    prices = 50000 * np.exp(np.cumsum(returns))
    scenarios['flash_crash'] = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.005, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.005, len(dates)))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.3, len(dates))
    }, index=dates)
    
    # High volatility scenario
    dates = pd.date_range(start='2024-01-01', end='2024-01-03', freq='1H')
    volatile_returns = np.random.normal(0, 0.05, len(dates))  # 5% volatility
    prices = 50000 * np.exp(np.cumsum(volatile_returns))
    
    scenarios['high_volatility'] = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.002, len(dates))),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.008, len(dates)))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.008, len(dates)))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.5, len(dates))
    }, index=dates)
    
    return scenarios


# Test markers for categorization
def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line("markers", "unit: Unit tests for individual components")
    config.addinivalue_line("markers", "integration: Integration tests between services")
    config.addinivalue_line("markers", "performance: Performance validation tests")
    config.addinivalue_line("markers", "stress: Stress testing and failure scenarios")
    config.addinivalue_line("markers", "benchmark: Benchmark comparison tests")
    config.addinivalue_line("markers", "slow: Tests that take a long time to run")
    config.addinivalue_line("markers", "critical: Critical path tests")
    config.addinivalue_line("markers", "regression: Regression tests")
    config.addinivalue_line("markers", "smoke: Quick smoke tests")


# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add automatic markers"""
    for item in items:
        # Auto-mark tests based on file names
        if "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        elif "stress_testing" in item.nodeid:
            item.add_marker(pytest.mark.stress)
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        elif "benchmark_comparison" in item.nodeid:
            item.add_marker(pytest.mark.benchmark)
        elif "test_environment" in item.nodeid or "test_rl_system" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        
        # Mark slow tests
        if any(keyword in item.nodeid for keyword in ["backtest", "monte_carlo", "walk_forward"]):
            item.add_marker(pytest.mark.slow)
        
        # Mark critical tests
        if any(keyword in item.name for keyword in ["sow", "compliance", "target"]):
            item.add_marker(pytest.mark.critical)


# Test session fixtures
@pytest.fixture(scope="session", autouse=True)
def test_session_setup():
    """Setup for the entire test session"""
    print("\n" + "="*50)
    print("RL Trading System Test Suite Starting")
    print("="*50)
    
    # Create test directories
    test_dirs = ['test_reports', 'coverage_reports', 'test_data']
    for dir_name in test_dirs:
        Path(dir_name).mkdir(exist_ok=True)
    
    yield
    
    print("\n" + "="*50)
    print("RL Trading System Test Suite Completed")
    print("="*50)


@pytest.fixture(autouse=True)
def test_logging():
    """Setup logging for individual tests"""
    # Capture test-specific logging
    import logging
    logger = logging.getLogger("test_logger")
    logger.setLevel(logging.INFO)
    
    yield logger


# Utility functions for tests
class TestUtils:
    """Utility functions for testing"""
    
    @staticmethod
    def create_mock_market_data(days: int = 30, volatility: float = 0.02) -> pd.DataFrame:
        """Create mock market data with specified parameters"""
        dates = pd.date_range(start='2024-01-01', periods=days*24, freq='1H')
        returns = np.random.normal(0.0001, volatility, len(dates))
        prices = 50000 * np.exp(np.cumsum(returns))
        
        return pd.DataFrame({
            'open': prices * (1 + np.random.normal(0, 0.001, len(dates))),
            'high': prices * (1 + np.abs(np.random.normal(0, 0.003, len(dates)))),
            'low': prices * (1 - np.abs(np.random.normal(0, 0.003, len(dates)))),
            'close': prices,
            'volume': np.random.lognormal(15, 0.3, len(dates))
        }, index=dates)
    
    @staticmethod
    def calculate_performance_metrics(returns: list) -> dict:
        """Calculate performance metrics from returns"""
        returns_array = np.array(returns)
        
        return {
            'total_return': np.sum(returns_array),
            'sharpe_ratio': np.mean(returns_array) / np.std(returns_array) * np.sqrt(252) if np.std(returns_array) > 0 else 0,
            'max_drawdown': TestUtils._calculate_max_drawdown(returns_array),
            'volatility': np.std(returns_array) * np.sqrt(252)
        }
    
    @staticmethod
    def _calculate_max_drawdown(returns: np.ndarray) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumsum(returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        return abs(np.min(drawdown)) if len(drawdown) > 0 else 0


@pytest.fixture
def test_utils():
    """Provide test utilities"""
    return TestUtils