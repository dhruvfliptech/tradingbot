"""
Example Usage of Multi-Objective Reward System

This script demonstrates how to integrate and use the sophisticated
multi-objective reward function in the RL trading environment.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import reward system components
from reward_calculator import RewardCalculator, RewardCalculatorConfig
from reward_components import RewardComponentConfig, MarketRegime
from multi_objective_reward import MultiObjectiveConfig, OptimizationStrategy
from reward_optimizer import create_optimizer, OptimizerConfig
from reward_analysis import create_analyzer, AnalysisConfig


class RewardSystemDemo:
    """
    Demonstration of the multi-objective reward system
    """
    
    def __init__(self):
        """Initialize reward system with SOW-aligned configuration"""
        
        # Configure components for SOW targets
        component_config = RewardComponentConfig(
            weekly_return_target=0.04,  # 4% weekly (middle of 3-5% range)
            target_sharpe_ratio=1.5,
            max_drawdown_limit=0.15,
            drawdown_warning_level=0.10,
            target_win_rate=0.60,
            base_transaction_cost=0.001,
            slippage_factor=0.0005
        )
        
        # Configure multi-objective optimization
        multi_objective_config = MultiObjectiveConfig(
            strategy=OptimizationStrategy.ADAPTIVE,
            base_weights={
                'profit': 0.25,
                'risk_adjusted': 0.30,  # Higher weight for Sharpe
                'drawdown': 0.20,
                'consistency': 0.15,
                'transaction_cost': 0.05,
                'exploration': 0.05
            },
            constraints={
                'drawdown': (-float('inf'), -0.15),
                'risk_adjusted': (1.5, float('inf')),
                'consistency': (0.6, float('inf')),
                'profit': (0.03, 0.05)
            },
            adaptation_rate=0.01,
            normalize_objectives=True
        )
        
        # Configure main calculator
        calculator_config = RewardCalculatorConfig(
            component_config=component_config,
            multi_objective_config=multi_objective_config,
            enable_reward_shaping=True,
            enable_curriculum=True,
            performance_targets={
                'weekly_return': 0.04,
                'sharpe_ratio': 1.5,
                'max_drawdown': 0.15,
                'win_rate': 0.60,
                'profit_factor': 1.5
            }
        )
        
        # Initialize reward calculator
        self.reward_calculator = RewardCalculator(calculator_config)
        
        # Initialize analyzer for debugging
        self.analyzer = create_analyzer(AnalysisConfig(
            window_size=100,
            save_plots=True,
            plot_directory="./reward_analysis"
        ))
        
        # Initialize optimizer
        self.optimizer = create_optimizer(
            OptimizerConfig(
                method='hybrid',
                primary_metric='sharpe_ratio',
                n_iterations=50
            )
        )
        
        logger.info("Reward system initialized with SOW-aligned configuration")
    
    def simulate_trading_episode(self, n_steps: int = 1000):
        """
        Simulate a trading episode with the reward system
        
        Args:
            n_steps: Number of steps in episode
        """
        logger.info(f"Starting episode simulation with {n_steps} steps")
        
        # Initial state
        initial_balance = 10000
        current_state = {
            'portfolio_value': initial_balance,
            'initial_equity': initial_balance,
            'cash_balance': initial_balance,
            'position_size': 0,
            'unrealized_pnl': 0,
            'realized_pnl': 0,
            'current_drawdown': 0,
            'returns_history': [],
            'win_rate': 0.5,
            'total_trades': 0,
            'market_regime': 'SIDEWAYS',
            'volatility': 0.02,
            'trend': 0,
            'max_steps': n_steps
        }
        
        # Track episode metrics
        episode_rewards = []
        portfolio_values = []
        
        # Reset for new episode
        self.reward_calculator.reset_episode()
        
        for step in range(n_steps):
            # Simulate market movement
            price_change = np.random.normal(0.0005, 0.02)  # ~0.05% mean, 2% std
            
            # Update state (simplified simulation)
            previous_state = current_state.copy()
            
            # Simulate portfolio changes
            current_state['portfolio_value'] *= (1 + price_change)
            portfolio_values.append(current_state['portfolio_value'])
            
            # Update returns history
            if len(current_state['returns_history']) > 0:
                daily_return = (current_state['portfolio_value'] - previous_state['portfolio_value']) / \
                             previous_state['portfolio_value']
                current_state['returns_history'].append(daily_return)
                if len(current_state['returns_history']) > 100:
                    current_state['returns_history'].pop(0)
            else:
                current_state['returns_history'] = [0]
            
            # Calculate drawdown
            peak_value = max(portfolio_values)
            current_state['current_drawdown'] = (peak_value - current_state['portfolio_value']) / peak_value
            
            # Simulate trades occasionally
            if np.random.random() < 0.1:  # 10% chance of trade
                trade_result = np.random.normal(0, 0.01)  # Trade P&L
                current_state['realized_pnl'] += trade_result * 1000
                current_state['total_trades'] += 1
                current_state['last_trade_result'] = trade_result
                
                # Update win rate
                if current_state['total_trades'] > 0:
                    wins = sum(1 for _ in range(current_state['total_trades']) 
                             if np.random.random() < 0.6)  # Simulate 60% win rate
                    current_state['win_rate'] = wins / current_state['total_trades']
            
            # Simulate market regime changes
            if step % 100 == 0:
                regimes = ['BULL', 'BEAR', 'SIDEWAYS', 'HIGH_VOLATILITY']
                current_state['market_regime'] = np.random.choice(regimes)
                current_state['volatility'] = np.random.uniform(0.01, 0.05)
            
            # Prepare action info
            action_info = {
                'action': np.random.randint(0, 11),  # Random action
                'type': np.random.choice(['buy', 'sell', 'hold']),
                'size': np.random.uniform(0, 1),
                'trade_volume': np.random.uniform(100, 1000),
                'is_trade': np.random.random() < 0.3,
                'episode_step': step
            }
            
            # Calculate reward
            reward, info = self.reward_calculator.calculate(
                current_state,
                previous_state,
                action_info
            )
            
            episode_rewards.append(reward)
            
            # Record for analysis
            self.analyzer.record(
                reward,
                info.get('objectives', {}),
                current_state,
                action_info['action']
            )
            
            # Log progress
            if step % 100 == 0:
                avg_reward = np.mean(episode_rewards[-100:]) if episode_rewards else 0
                logger.info(f"Step {step}: Avg Reward = {avg_reward:.4f}, "
                          f"Portfolio = ${current_state['portfolio_value']:.2f}, "
                          f"Drawdown = {current_state['current_drawdown']:.2%}")
        
        # Episode summary
        total_return = (current_state['portfolio_value'] - initial_balance) / initial_balance
        avg_reward = np.mean(episode_rewards)
        
        logger.info(f"\nEpisode Summary:")
        logger.info(f"  Total Return: {total_return:.2%}")
        logger.info(f"  Average Reward: {avg_reward:.4f}")
        logger.info(f"  Final Portfolio: ${current_state['portfolio_value']:.2f}")
        logger.info(f"  Max Drawdown: {max(d for d in [s.get('current_drawdown', 0) 
                                                       for s in self.analyzer.state_history]):.2%}")
        
        return episode_rewards, portfolio_values
    
    def optimize_weights(self, n_episodes: int = 10):
        """
        Optimize reward weights using simulated episodes
        
        Args:
            n_episodes: Number of episodes for evaluation
        """
        logger.info("Starting weight optimization")
        
        def evaluation_function(weights: Dict[str, float]) -> float:
            """Evaluate weight configuration"""
            # Update calculator with new weights
            self.reward_calculator.multi_objective.current_weights = weights
            
            # Run multiple episodes
            performances = []
            for _ in range(n_episodes):
                rewards, values = self.simulate_trading_episode(100)  # Short episodes for speed
                
                # Calculate performance metrics
                total_return = (values[-1] - values[0]) / values[0] if values else 0
                sharpe = np.mean(rewards) / (np.std(rewards) + 1e-10) * np.sqrt(252)
                
                # Combined performance metric
                performance = sharpe * 0.5 + total_return * 10 + np.mean(rewards)
                performances.append(performance)
            
            return np.mean(performances)
        
        # Optimize weights
        optimized_weights = self.optimizer.optimize(
            evaluation_function,
            current_regime=MarketRegime.SIDEWAYS
        )
        
        logger.info(f"Optimized weights: {optimized_weights}")
        return optimized_weights
    
    def analyze_performance(self):
        """Generate and display performance analysis"""
        logger.info("Generating performance analysis")
        
        # Generate analysis report
        report = self.analyzer.generate_report()
        
        # Display key metrics
        logger.info("\n=== Performance Analysis ===")
        
        if 'target_alignment' in report:
            logger.info("\nTarget Alignment:")
            for metric, data in report['target_alignment'].items():
                status = "✓" if data.get('achieved', data.get('within_limit', False)) else "✗"
                logger.info(f"  {metric}: {data.get('current', 0):.3f} {status}")
        
        if 'insights' in report:
            logger.info("\nKey Insights:")
            for insight in report['insights']:
                logger.info(f"  - {insight}")
        
        if 'recommendations' in report:
            logger.info("\nRecommendations:")
            for rec in report['recommendations']:
                logger.info(f"  - {rec}")
        
        # Generate plots
        logger.info("\nGenerating analysis plots...")
        self.analyzer.plot_reward_breakdown()
        self.analyzer.plot_correlation_heatmap()
        self.analyzer.plot_performance_alignment()
        self.analyzer.plot_regime_comparison()
        
        # Save report
        self.analyzer.save_report("reward_analysis_report.json")
        
        return report
    
    def demonstrate_market_adaptation(self):
        """Demonstrate reward adaptation to different market regimes"""
        logger.info("\n=== Market Regime Adaptation Demo ===")
        
        regimes = [MarketRegime.BULL, MarketRegime.BEAR, 
                  MarketRegime.HIGH_VOLATILITY, MarketRegime.SIDEWAYS]
        
        for regime in regimes:
            logger.info(f"\nTesting {regime.value} market:")
            
            # Create state for this regime
            state = {
                'portfolio_value': 10000,
                'initial_equity': 10000,
                'returns_history': [np.random.normal(0.001, 0.02) for _ in range(30)],
                'current_drawdown': 0.05,
                'win_rate': 0.6,
                'market_regime': regime.value,
                'volatility': 0.03 if regime == MarketRegime.HIGH_VOLATILITY else 0.02
            }
            
            action_info = {
                'type': 'buy',
                'size': 0.5,
                'trade_volume': 500,
                'is_trade': True
            }
            
            # Calculate reward
            reward, info = self.reward_calculator.calculate(state, None, action_info)
            
            logger.info(f"  Reward: {reward:.4f}")
            logger.info(f"  Components: {info.get('objectives', {})}")


def main():
    """Main demonstration function"""
    logger.info("=" * 60)
    logger.info("Multi-Objective Reward System Demonstration")
    logger.info("=" * 60)
    
    # Initialize demo
    demo = RewardSystemDemo()
    
    # 1. Simulate trading episode
    logger.info("\n1. Simulating Trading Episode")
    logger.info("-" * 40)
    rewards, values = demo.simulate_trading_episode(n_steps=500)
    
    # 2. Optimize weights
    logger.info("\n2. Optimizing Reward Weights")
    logger.info("-" * 40)
    # optimized_weights = demo.optimize_weights(n_episodes=5)  # Commented for speed
    
    # 3. Analyze performance
    logger.info("\n3. Analyzing Performance")
    logger.info("-" * 40)
    report = demo.analyze_performance()
    
    # 4. Demonstrate market adaptation
    logger.info("\n4. Market Regime Adaptation")
    logger.info("-" * 40)
    demo.demonstrate_market_adaptation()
    
    logger.info("\n" + "=" * 60)
    logger.info("Demonstration Complete")
    logger.info("=" * 60)
    
    # Display final metrics
    metrics = demo.reward_calculator.get_metrics()
    logger.info("\nFinal System Metrics:")
    logger.info(f"  Total Steps: {metrics.get('total_steps', 0)}")
    logger.info(f"  Episodes: {metrics.get('total_episodes', 0)}")
    logger.info(f"  Curriculum Stage: {metrics.get('curriculum_stage', 0)}")
    logger.info(f"  Sub-goals Achieved: {metrics.get('sub_goals_achieved', 0)}")
    
    return demo, report


if __name__ == "__main__":
    demo, report = main()