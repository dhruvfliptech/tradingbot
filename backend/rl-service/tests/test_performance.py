"""
Performance Validation Tests for RL Trading System
Validates that the RL system meets SOW requirements:
- 3-5% weekly returns
- Sharpe ratio > 1.5
- Maximum drawdown < 15%
- Win rate > 60%
- 15-20% outperformance vs AdaptiveThreshold baseline
"""

import pytest
import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple
import statistics
import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.trading_env import TradingEnvironment
from environment.portfolio_manager import PortfolioManager
from agents.ppo_agent import PPOAgent
from agents.ensemble_agent import EnsembleAgent
from integration.rl_service import RLService
from rl_config import RLConfig, RewardStrategy, ActionType

logger = logging.getLogger(__name__)


class SOWTargets:
    """Statement of Work performance targets"""
    MIN_WEEKLY_RETURN = 0.03  # 3%
    MAX_WEEKLY_RETURN = 0.05  # 5%
    MIN_SHARPE_RATIO = 1.5
    MAX_DRAWDOWN = 0.15  # 15%
    MIN_WIN_RATE = 0.60  # 60%
    MIN_OUTPERFORMANCE_VS_BASELINE = 0.15  # 15%
    MAX_OUTPERFORMANCE_VS_BASELINE = 0.20  # 20%


@pytest.fixture
def performance_config():
    """Create configuration optimized for performance testing"""
    config = RLConfig()
    config.env.episode_length = 2000  # Longer episodes for performance measurement
    config.env.initial_balance = 100000.0  # Larger balance for realistic testing
    config.model.total_timesteps = 50000  # More training for better performance
    config.reward.strategy = RewardStrategy.RISK_ADJUSTED
    config.data.start_date = '2023-01-01'
    config.data.end_date = '2024-06-30'  # 18 months of data
    return config


@pytest.fixture
def extended_market_data():
    """Create extended market data for performance testing"""
    # Generate 18 months of hourly data
    dates = pd.date_range(start='2023-01-01', end='2024-06-30', freq='1H')
    np.random.seed(42)  # Reproducible results
    
    # Generate more realistic market data with trends and volatility clustering
    n_periods = len(dates)
    
    # Base return with trend and volatility clustering
    base_return = 0.0001  # Slight positive trend
    volatility = 0.02
    
    # Add regime changes and volatility clustering
    regime_changes = np.random.choice([0, 1], n_periods, p=[0.99, 0.01])
    volatility_multiplier = np.where(regime_changes, 2.0, 1.0)
    
    returns = np.random.normal(base_return, volatility * volatility_multiplier, n_periods)
    
    # Add some autocorrelation to make it more realistic
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i-1]
    
    # Generate prices from returns
    initial_price = 45000  # Starting BTC price
    prices = initial_price * np.exp(np.cumsum(returns))
    
    # Generate OHLCV data
    data = pd.DataFrame({
        'open': prices * (1 + np.random.normal(0, 0.001, n_periods)),
        'high': prices * (1 + np.abs(np.random.normal(0, 0.003, n_periods))),
        'low': prices * (1 - np.abs(np.random.normal(0, 0.003, n_periods))),
        'close': prices,
        'volume': np.random.lognormal(15, 0.3, n_periods)  # More stable volume
    }, index=dates)
    
    # Ensure high >= low and realistic OHLC relationships
    data['high'] = np.maximum(data['high'], data[['open', 'close']].max(axis=1))
    data['low'] = np.minimum(data['low'], data[['open', 'close']].min(axis=1))
    
    return data


class TestPerformanceValidation:
    """Test performance against SOW targets"""

    def test_weekly_return_targets(self, performance_config, extended_market_data):
        """Test that weekly returns meet 3-5% target"""
        
        # Create and run RL system
        env = TradingEnvironment(config=performance_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': extended_market_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(extended_market_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=performance_config
        )
        
        # Run backtesting simulation
        weekly_returns = self._run_performance_simulation(env, agent, weeks=20)
        
        # Calculate statistics
        avg_weekly_return = np.mean(weekly_returns)
        median_weekly_return = np.median(weekly_returns)
        std_weekly_return = np.std(weekly_returns)
        
        # Performance validation
        logger.info(f"Weekly returns - Avg: {avg_weekly_return:.3f}, Median: {median_weekly_return:.3f}, Std: {std_weekly_return:.3f}")
        
        # Test targets
        assert avg_weekly_return >= SOWTargets.MIN_WEEKLY_RETURN, \
            f"Average weekly return {avg_weekly_return:.3f} below target {SOWTargets.MIN_WEEKLY_RETURN}"
        
        # Allow some flexibility for median (should be at least 80% of minimum target)
        assert median_weekly_return >= SOWTargets.MIN_WEEKLY_RETURN * 0.8, \
            f"Median weekly return {median_weekly_return:.3f} significantly below target"
        
        # Check that returns are not excessively high (unrealistic)
        assert avg_weekly_return <= SOWTargets.MAX_WEEKLY_RETURN * 1.5, \
            f"Average weekly return {avg_weekly_return:.3f} unrealistically high"
        
        # At least 60% of weeks should meet minimum target
        weeks_meeting_target = sum(1 for r in weekly_returns if r >= SOWTargets.MIN_WEEKLY_RETURN)
        target_percentage = weeks_meeting_target / len(weekly_returns)
        
        assert target_percentage >= 0.6, \
            f"Only {target_percentage:.1%} of weeks meet return target (need 60%)"
        
        logger.info(f"Weekly return validation passed: {target_percentage:.1%} of weeks meet target")

    def test_sharpe_ratio_target(self, performance_config, extended_market_data):
        """Test that Sharpe ratio exceeds 1.5"""
        
        env = TradingEnvironment(config=performance_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': extended_market_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(extended_market_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=performance_config
        )
        
        # Run simulation and collect daily returns
        daily_returns = self._collect_daily_returns(env, agent, days=90)
        
        # Calculate Sharpe ratio
        risk_free_rate = 0.02 / 365  # 2% annual risk-free rate, daily
        excess_returns = [r - risk_free_rate for r in daily_returns]
        
        if len(excess_returns) > 0 and np.std(excess_returns) > 0:
            sharpe_ratio = np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(365)  # Annualized
        else:
            sharpe_ratio = 0
        
        logger.info(f"Calculated Sharpe ratio: {sharpe_ratio:.2f}")
        
        # Test target
        assert sharpe_ratio >= SOWTargets.MIN_SHARPE_RATIO, \
            f"Sharpe ratio {sharpe_ratio:.2f} below target {SOWTargets.MIN_SHARPE_RATIO}"
        
        # Additional validation: Check consistency
        # Rolling 30-day Sharpe ratios should also be reasonable
        if len(daily_returns) >= 30:
            rolling_sharpes = []
            for i in range(30, len(daily_returns)):
                window_returns = daily_returns[i-30:i]
                window_excess = [r - risk_free_rate for r in window_returns]
                if np.std(window_excess) > 0:
                    rolling_sharpe = np.mean(window_excess) / np.std(window_excess) * np.sqrt(365)
                    rolling_sharpes.append(rolling_sharpe)
            
            if rolling_sharpes:
                avg_rolling_sharpe = np.mean(rolling_sharpes)
                assert avg_rolling_sharpe >= SOWTargets.MIN_SHARPE_RATIO * 0.8, \
                    f"Average rolling Sharpe {avg_rolling_sharpe:.2f} inconsistent with overall performance"
        
        logger.info("Sharpe ratio validation passed")

    def test_maximum_drawdown_constraint(self, performance_config, extended_market_data):
        """Test that maximum drawdown stays below 15%"""
        
        env = TradingEnvironment(config=performance_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': extended_market_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(extended_market_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=performance_config
        )
        
        # Run simulation and track portfolio value
        portfolio_values = self._track_portfolio_values(env, agent, steps=1000)
        
        # Calculate drawdowns
        drawdowns = self._calculate_drawdowns(portfolio_values)
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        logger.info(f"Maximum drawdown: {max_drawdown:.3f} ({max_drawdown*100:.1f}%)")
        
        # Test constraint
        assert max_drawdown <= SOWTargets.MAX_DRAWDOWN, \
            f"Maximum drawdown {max_drawdown:.3f} exceeds limit {SOWTargets.MAX_DRAWDOWN}"
        
        # Additional validations
        # Check that drawdowns are not too frequent
        large_drawdowns = [d for d in drawdowns if d > SOWTargets.MAX_DRAWDOWN * 0.5]
        assert len(large_drawdowns) / len(drawdowns) < 0.1, \
            f"Too many large drawdowns: {len(large_drawdowns)} out of {len(drawdowns)}"
        
        # Check average drawdown is reasonable
        avg_drawdown = np.mean(drawdowns) if drawdowns else 0
        assert avg_drawdown <= SOWTargets.MAX_DRAWDOWN * 0.3, \
            f"Average drawdown {avg_drawdown:.3f} too high"
        
        logger.info("Maximum drawdown validation passed")

    def test_win_rate_target(self, performance_config, extended_market_data):
        """Test that win rate exceeds 60%"""
        
        env = TradingEnvironment(config=performance_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': extended_market_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(extended_market_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=performance_config
        )
        
        # Run simulation and collect trade results
        trade_results = self._collect_trade_results(env, agent, min_trades=100)
        
        # Calculate win rate
        winning_trades = sum(1 for result in trade_results if result > 0)
        total_trades = len(trade_results)
        win_rate = winning_trades / total_trades if total_trades > 0 else 0
        
        logger.info(f"Win rate: {win_rate:.3f} ({win_rate*100:.1f}%) from {total_trades} trades")
        
        # Test target
        assert win_rate >= SOWTargets.MIN_WIN_RATE, \
            f"Win rate {win_rate:.3f} below target {SOWTargets.MIN_WIN_RATE}"
        
        # Additional validations
        assert total_trades >= 50, f"Not enough trades for reliable win rate calculation: {total_trades}"
        
        # Check that winning trades are significantly profitable
        if winning_trades > 0:
            winning_trade_results = [r for r in trade_results if r > 0]
            avg_winning_trade = np.mean(winning_trade_results)
            assert avg_winning_trade > 0.01, \
                f"Average winning trade {avg_winning_trade:.3f} too small"
        
        # Check that losing trades are not too large
        losing_trades = [r for r in trade_results if r < 0]
        if losing_trades:
            avg_losing_trade = np.mean(losing_trades)
            assert avg_losing_trade > -0.05, \
                f"Average losing trade {avg_losing_trade:.3f} too large"
        
        logger.info("Win rate validation passed")

    def test_consistency_across_market_conditions(self, performance_config, extended_market_data):
        """Test performance consistency across different market conditions"""
        
        # Identify different market regimes in the data
        data = extended_market_data
        returns = data['close'].pct_change().dropna()
        
        # Define market regimes based on volatility and trend
        rolling_vol = returns.rolling(window=168).std()  # Weekly volatility
        rolling_trend = returns.rolling(window=168).mean()  # Weekly trend
        
        # Categorize market conditions
        high_vol_periods = rolling_vol > rolling_vol.quantile(0.75)
        low_vol_periods = rolling_vol < rolling_vol.quantile(0.25)
        bull_periods = rolling_trend > rolling_trend.quantile(0.75)
        bear_periods = rolling_trend < rolling_trend.quantile(0.25)
        
        # Test performance in different conditions
        conditions = {
            'high_volatility': high_vol_periods,
            'low_volatility': low_vol_periods,
            'bull_market': bull_periods,
            'bear_market': bear_periods
        }
        
        env = TradingEnvironment(config=performance_config, mode='test')
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=performance_config
        )
        
        condition_performance = {}
        
        for condition_name, condition_mask in conditions.items():
            if condition_mask.sum() > 100:  # Enough data points
                # Extract data for this condition
                condition_data = data[condition_mask]
                
                if len(condition_data) > 100:
                    # Run simulation on this subset
                    env.market_simulator.market_data = {'BTC/USD': condition_data}
                    env.market_simulator.current_step = 0
                    env.market_simulator.max_steps = len(condition_data) - 1
                    
                    performance = self._measure_condition_performance(env, agent)
                    condition_performance[condition_name] = performance
        
        logger.info(f"Performance across conditions: {condition_performance}")
        
        # Validate performance consistency
        for condition, perf in condition_performance.items():
            # Each condition should maintain minimum performance standards
            assert perf.get('sharpe_ratio', 0) >= SOWTargets.MIN_SHARPE_RATIO * 0.7, \
                f"Sharpe ratio in {condition} too low: {perf.get('sharpe_ratio', 0):.2f}"
            
            assert perf.get('max_drawdown', 1) <= SOWTargets.MAX_DRAWDOWN * 1.2, \
                f"Drawdown in {condition} too high: {perf.get('max_drawdown', 1):.3f}"
        
        logger.info("Market condition consistency validation passed")

    def test_risk_adjusted_performance_metrics(self, performance_config, extended_market_data):
        """Test comprehensive risk-adjusted performance metrics"""
        
        env = TradingEnvironment(config=performance_config, mode='test')
        env.market_simulator.market_data = {'BTC/USD': extended_market_data}
        env.market_simulator.current_step = 0
        env.market_simulator.max_steps = len(extended_market_data) - 1
        
        agent = PPOAgent(
            observation_space=env.observation_space,
            action_space=env.action_space,
            config=performance_config
        )
        
        # Collect comprehensive performance data
        daily_returns = self._collect_daily_returns(env, agent, days=120)
        
        # Calculate multiple risk metrics
        metrics = self._calculate_risk_metrics(daily_returns)
        
        logger.info(f"Risk metrics: {metrics}")
        
        # Validate risk metrics
        assert metrics['sharpe_ratio'] >= SOWTargets.MIN_SHARPE_RATIO
        assert metrics['sortino_ratio'] >= SOWTargets.MIN_SHARPE_RATIO * 0.8  # Sortino should be similar
        assert metrics['max_drawdown'] <= SOWTargets.MAX_DRAWDOWN
        assert metrics['calmar_ratio'] >= 0.1  # Minimum Calmar ratio
        assert metrics['var_95'] <= 0.05  # 95% VaR should be reasonable
        
        # Information ratio should indicate skill
        assert metrics.get('information_ratio', 0) >= 0.5, \
            "Information ratio indicates insufficient alpha generation"
        
        logger.info("Risk-adjusted performance validation passed")

    # Helper methods

    def _run_performance_simulation(self, env, agent, weeks: int) -> List[float]:
        """Run simulation and return weekly returns"""
        weekly_returns = []
        
        obs, _ = env.reset()
        initial_equity = env.portfolio_manager.get_total_equity()
        week_start_equity = initial_equity
        steps_in_week = 0
        
        for step in range(weeks * 168):  # 168 hours per week
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            steps_in_week += 1
            
            # Calculate weekly return every 168 steps (1 week)
            if steps_in_week >= 168 or terminated or truncated:
                current_equity = env.portfolio_manager.get_total_equity()
                weekly_return = (current_equity - week_start_equity) / week_start_equity
                weekly_returns.append(weekly_return)
                
                week_start_equity = current_equity
                steps_in_week = 0
                
                if terminated or truncated:
                    obs, _ = env.reset()
        
        return weekly_returns

    def _collect_daily_returns(self, env, agent, days: int) -> List[float]:
        """Collect daily returns from simulation"""
        daily_returns = []
        
        obs, _ = env.reset()
        daily_start_equity = env.portfolio_manager.get_total_equity()
        steps_in_day = 0
        
        for step in range(days * 24):  # 24 hours per day
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            steps_in_day += 1
            
            # Calculate daily return every 24 steps
            if steps_in_day >= 24 or terminated or truncated:
                current_equity = env.portfolio_manager.get_total_equity()
                daily_return = (current_equity - daily_start_equity) / daily_start_equity
                daily_returns.append(daily_return)
                
                daily_start_equity = current_equity
                steps_in_day = 0
                
                if terminated or truncated:
                    obs, _ = env.reset()
                    daily_start_equity = env.portfolio_manager.get_total_equity()
        
        return daily_returns

    def _track_portfolio_values(self, env, agent, steps: int) -> List[float]:
        """Track portfolio values over time"""
        portfolio_values = []
        
        obs, _ = env.reset()
        
        for step in range(steps):
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            portfolio_value = env.portfolio_manager.get_total_equity()
            portfolio_values.append(portfolio_value)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        return portfolio_values

    def _calculate_drawdowns(self, portfolio_values: List[float]) -> List[float]:
        """Calculate drawdowns from portfolio values"""
        if not portfolio_values:
            return []
        
        peak = portfolio_values[0]
        drawdowns = []
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            drawdowns.append(drawdown)
        
        return drawdowns

    def _collect_trade_results(self, env, agent, min_trades: int) -> List[float]:
        """Collect individual trade results"""
        trade_results = []
        
        obs, _ = env.reset()
        
        while len(trade_results) < min_trades:
            action = agent.predict(obs, deterministic=True)[0]
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Check if a trade was completed
            if 'trade_result' in info:
                trade_results.append(info['trade_result'])
            
            # Approximate trade completion based on portfolio changes
            if hasattr(env.portfolio_manager, 'completed_trades'):
                if len(env.portfolio_manager.completed_trades) > len(trade_results):
                    recent_trades = env.portfolio_manager.completed_trades[len(trade_results):]
                    for trade in recent_trades:
                        trade_pnl = trade.pnl / trade.entry_value if trade.entry_value > 0 else 0
                        trade_results.append(trade_pnl)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        return trade_results[:min_trades]

    def _measure_condition_performance(self, env, agent) -> Dict:
        """Measure performance in specific market condition"""
        daily_returns = self._collect_daily_returns(env, agent, days=30)
        portfolio_values = self._track_portfolio_values(env, agent, steps=500)
        
        if not daily_returns or not portfolio_values:
            return {}
        
        # Calculate key metrics
        total_return = sum(daily_returns)
        volatility = np.std(daily_returns) if len(daily_returns) > 1 else 0
        sharpe_ratio = (np.mean(daily_returns) - 0.02/365) / volatility * np.sqrt(365) if volatility > 0 else 0
        drawdowns = self._calculate_drawdowns(portfolio_values)
        max_drawdown = max(drawdowns) if drawdowns else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility
        }

    def _calculate_risk_metrics(self, returns: List[float]) -> Dict:
        """Calculate comprehensive risk metrics"""
        if not returns or len(returns) < 10:
            return {}
        
        returns_array = np.array(returns)
        risk_free_rate = 0.02 / 365  # Daily risk-free rate
        
        # Basic metrics
        total_return = np.sum(returns_array)
        avg_return = np.mean(returns_array)
        volatility = np.std(returns_array)
        
        # Risk-adjusted metrics
        sharpe_ratio = (avg_return - risk_free_rate) / volatility * np.sqrt(365) if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = [r for r in returns if r < 0]
        downside_deviation = np.std(downside_returns) if downside_returns else volatility
        sortino_ratio = (avg_return - risk_free_rate) / downside_deviation * np.sqrt(365) if downside_deviation > 0 else 0
        
        # Maximum drawdown and Calmar ratio
        cumulative_returns = np.cumsum(returns_array)
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = cumulative_returns - running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        max_drawdown = abs(max_drawdown)
        
        calmar_ratio = (total_return * 365 / len(returns)) / max_drawdown if max_drawdown > 0 else 0
        
        # Value at Risk (95%)
        var_95 = np.percentile(returns_array, 5) if len(returns_array) > 20 else 0
        
        # Information ratio (assuming benchmark return of 0)
        information_ratio = avg_return / volatility * np.sqrt(365) if volatility > 0 else 0
        
        return {
            'total_return': total_return,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar_ratio,
            'var_95': abs(var_95),
            'information_ratio': information_ratio,
            'volatility': volatility
        }


if __name__ == "__main__":
    # Run performance validation tests
    pytest.main([__file__, "-v", "--tb=short", "-k", "test_"])