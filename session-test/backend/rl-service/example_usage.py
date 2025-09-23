"""
Example Usage of RL Trading Environment
Demonstrates how to use the Gymnasium-compatible trading environment for training and testing
"""

import numpy as np
import pandas as pd
import logging
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import List, Dict, Any

# Import our environment components
from rl_config import RLConfig, get_rl_config
from environment import TradingEnvironment

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_environment(mode: str = 'train') -> TradingEnvironment:
    """Create and configure trading environment"""
    config = get_rl_config()
    
    # Customize config for demo
    config.env.episode_length = 500
    config.env.initial_balance = 10000.0
    config.env.trading_pairs = ['BTC/USD', 'ETH/USD']
    config.env.primary_pair = 'BTC/USD'
    
    # Create environment
    env = TradingEnvironment(config=config, mode=mode)
    
    return env


def load_data(env: TradingEnvironment) -> bool:
    """Load market data for the environment"""
    logger.info("Loading market data...")
    
    # Use recent data for demo
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    success = env.load_data(start_date, end_date)
    
    if success:
        logger.info("Market data loaded successfully")
        logger.info(f"Available steps: {env.market_simulator.get_available_steps()}")
    else:
        logger.error("Failed to load market data")
    
    return success


def random_agent_demo(env: TradingEnvironment, num_episodes: int = 5) -> List[Dict[str, Any]]:
    """Demonstrate random trading agent"""
    logger.info(f"Running random agent demo for {num_episodes} episodes...")
    
    episode_results = []
    
    for episode in range(num_episodes):
        logger.info(f"Starting episode {episode + 1}")
        
        obs, info = env.reset()
        episode_reward = 0.0
        episode_actions = []
        portfolio_values = [info['portfolio_value']]
        
        done = False
        step = 0
        
        while not done and step < env.episode_length:
            # Random action
            action = env.action_space.sample()
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_actions.append(action)
            portfolio_values.append(info['portfolio_value'])
            
            done = terminated or truncated
            step += 1
            
            # Log every 100 steps
            if step % 100 == 0:
                logger.info(f"  Step {step}: Portfolio=${info['portfolio_value']:,.2f}, Reward={reward:.4f}")
        
        # Episode summary
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        total_return = (final_value - initial_value) / initial_value
        
        episode_result = {
            'episode': episode + 1,
            'steps': step,
            'total_reward': episode_reward,
            'avg_reward': episode_reward / step if step > 0 else 0,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'portfolio_values': portfolio_values,
            'actions': episode_actions
        }
        
        episode_results.append(episode_result)
        
        logger.info(f"Episode {episode + 1} completed:")
        logger.info(f"  Steps: {step}")
        logger.info(f"  Total reward: {episode_reward:.2f}")
        logger.info(f"  Return: {total_return:.2%}")
        logger.info(f"  Final portfolio: ${final_value:,.2f}")
        
        # Get performance metrics
        metrics = env.portfolio_manager.get_performance_metrics()
        if metrics:
            logger.info(f"  Sharpe ratio: {metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"  Max drawdown: {metrics.get('max_drawdown_pct', 0):.1f}%")
    
    return episode_results


def simple_strategy_demo(env: TradingEnvironment, num_episodes: int = 3) -> List[Dict[str, Any]]:
    """Demonstrate simple strategy (buy low, sell high based on RSI-like logic)"""
    logger.info(f"Running simple strategy demo for {num_episodes} episodes...")
    
    from rl_config import ActionType
    
    episode_results = []
    
    for episode in range(num_episodes):
        logger.info(f"Starting simple strategy episode {episode + 1}")
        
        obs, info = env.reset()
        episode_reward = 0.0
        episode_actions = []
        portfolio_values = [info['portfolio_value']]
        
        # Simple state tracking
        last_action = ActionType.HOLD.value
        position_size = 0.0
        
        done = False
        step = 0
        
        while not done and step < env.episode_length:
            # Simple strategy logic
            current_position = env.portfolio_manager.get_position(env.primary_symbol)
            current_allocation = env.portfolio_manager.get_position_allocation(env.primary_symbol)
            
            # Get some features from observation (this is simplified)
            # In practice, you'd extract meaningful features
            obs_mean = np.mean(obs) if len(obs) > 0 else 0
            obs_std = np.std(obs) if len(obs) > 0 else 0
            
            # Simple logic: buy when obs_mean is low, sell when high
            action = ActionType.HOLD.value
            
            if obs_mean < -0.5 and current_allocation < 0.8:
                # Market seems oversold, buy
                action = ActionType.BUY_40.value
            elif obs_mean > 0.5 and current_allocation > 0.2:
                # Market seems overbought, sell
                action = ActionType.SELL_40.value
            elif step % 50 == 0 and current_allocation < 0.1:
                # Periodic small buy to stay invested
                action = ActionType.BUY_20.value
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            
            episode_reward += reward
            episode_actions.append(action)
            portfolio_values.append(info['portfolio_value'])
            
            done = terminated or truncated
            step += 1
            
            # Log every 100 steps
            if step % 100 == 0:
                logger.info(f"  Step {step}: Action={action}, Portfolio=${info['portfolio_value']:,.2f}, "
                          f"Allocation={current_allocation:.1%}")
        
        # Episode summary
        final_value = portfolio_values[-1]
        initial_value = portfolio_values[0]
        total_return = (final_value - initial_value) / initial_value
        
        episode_result = {
            'episode': episode + 1,
            'strategy': 'simple',
            'steps': step,
            'total_reward': episode_reward,
            'avg_reward': episode_reward / step if step > 0 else 0,
            'initial_value': initial_value,
            'final_value': final_value,
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'portfolio_values': portfolio_values,
            'actions': episode_actions
        }
        
        episode_results.append(episode_result)
        
        logger.info(f"Simple strategy episode {episode + 1} completed:")
        logger.info(f"  Return: {total_return:.2%}")
        logger.info(f"  Final portfolio: ${final_value:,.2f}")
    
    return episode_results


def analyze_results(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze episode results"""
    if not results:
        return {}
    
    returns = [r['total_return'] for r in results]
    rewards = [r['total_reward'] for r in results]
    
    analysis = {
        'num_episodes': len(results),
        'avg_return': np.mean(returns),
        'std_return': np.std(returns),
        'min_return': np.min(returns),
        'max_return': np.max(returns),
        'avg_reward': np.mean(rewards),
        'std_reward': np.std(rewards),
        'positive_episodes': sum(1 for r in returns if r > 0),
        'negative_episodes': sum(1 for r in returns if r < 0),
        'win_rate': sum(1 for r in returns if r > 0) / len(returns)
    }
    
    return analysis


def plot_results(results: List[Dict[str, Any]], title: str = "Trading Results"):
    """Plot trading results"""
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title)
        
        # Portfolio value evolution
        for i, result in enumerate(results):
            axes[0, 0].plot(result['portfolio_values'], label=f"Episode {result['episode']}")
        axes[0, 0].set_title("Portfolio Value Evolution")
        axes[0, 0].set_xlabel("Steps")
        axes[0, 0].set_ylabel("Portfolio Value ($)")
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Returns distribution
        returns = [r['total_return_pct'] for r in results]
        axes[0, 1].hist(returns, bins=10, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title("Returns Distribution")
        axes[0, 1].set_xlabel("Return (%)")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].grid(True)
        
        # Cumulative rewards
        for i, result in enumerate(results):
            cumulative_rewards = np.cumsum([0] + result.get('actions', []))  # Simplified
            axes[1, 0].plot(cumulative_rewards, label=f"Episode {result['episode']}")
        axes[1, 0].set_title("Action Distribution")
        axes[1, 0].set_xlabel("Steps")
        axes[1, 0].set_ylabel("Cumulative Action Sum")
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Summary statistics
        analysis = analyze_results(results)
        stats_text = f"""
        Episodes: {analysis['num_episodes']}
        Avg Return: {analysis['avg_return']:.2%}
        Std Return: {analysis['std_return']:.2%}
        Win Rate: {analysis['win_rate']:.1%}
        Min Return: {analysis['min_return']:.2%}
        Max Return: {analysis['max_return']:.2%}
        """
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
        axes[1, 1].set_title("Summary Statistics")
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        logger.warning("Matplotlib not available, skipping plots")
    except Exception as e:
        logger.error(f"Error plotting results: {e}")


def main():
    """Main demo function"""
    logger.info("Starting RL Trading Environment Demo")
    
    try:
        # Create environment
        env = create_environment(mode='test')
        logger.info(f"Environment created with action space: {env.action_space}")
        logger.info(f"Observation space shape: {env.observation_space.shape}")
        
        # Load data
        if not load_data(env):
            logger.error("Failed to load data, exiting")
            return
        
        # Test environment basic functionality
        logger.info("Testing basic environment functionality...")
        obs, info = env.reset()
        logger.info(f"Reset successful, initial portfolio: ${info['portfolio_value']:,.2f}")
        
        # Take a few steps
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            logger.info(f"Step {i+1}: Action={action}, Reward={reward:.4f}, Portfolio=${info['portfolio_value']:,.2f}")
        
        logger.info("Basic functionality test completed successfully!")
        
        # Run random agent demo
        logger.info("\n" + "="*50)
        random_results = random_agent_demo(env, num_episodes=3)
        
        # Run simple strategy demo
        logger.info("\n" + "="*50)
        strategy_results = simple_strategy_demo(env, num_episodes=2)
        
        # Analyze results
        logger.info("\n" + "="*50)
        logger.info("ANALYSIS RESULTS")
        logger.info("="*50)
        
        random_analysis = analyze_results(random_results)
        strategy_analysis = analyze_results(strategy_results)
        
        logger.info("Random Agent Results:")
        for key, value in random_analysis.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        logger.info("\nSimple Strategy Results:")
        for key, value in strategy_analysis.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.4f}")
            else:
                logger.info(f"  {key}: {value}")
        
        # Plot results
        plot_results(random_results, "Random Agent Results")
        plot_results(strategy_results, "Simple Strategy Results")
        
        # Environment metrics
        logger.info("\nEnvironment Metrics:")
        env_metrics = env.get_metrics()
        for key, value in env_metrics.items():
            if key != 'portfolio_metrics':  # Skip nested dict
                logger.info(f"  {key}: {value}")
        
        # Close environment
        env.close()
        
        logger.info("\nDemo completed successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()