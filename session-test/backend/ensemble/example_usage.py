"""
Multi-Agent Ensemble System - Example Usage

This script demonstrates how to use the complete multi-agent ensemble system
for cryptocurrency trading. It shows initialization, training, prediction,
and performance tracking across different market conditions.

The example includes:
1. System initialization with specialized agents
2. Training individual agents
3. Making predictions with ensemble coordination
4. Performance tracking and analysis
5. Market regime adaptation
6. Risk management and reporting

Run this script to see the ensemble system in action with simulated data.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import logging
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../rl-service'))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import ensemble system
from . import (
    EnsembleSystem, create_default_ensemble_system, 
    AgentType, MarketRegime, DecisionStrategy
)

# Import RL components (simulated for example)
try:
    from rl_config import get_rl_config
    from environment.trading_env import TradingEnvironment
except ImportError:
    logger.warning("RL service components not available. Using mock environment.")
    
    # Mock environment for demonstration
    class MockTradingEnvironment:
        def __init__(self, config=None, mode='train'):
            self.config = config
            self.mode = mode
            self.observation_space = type('space', (), {'shape': (50,)})()
            self.action_space = type('space', (), {'shape': (1,)})()
            self.current_step = 0
            self.max_steps = 1000
            
        def reset(self):
            self.current_step = 0
            return np.random.random(50)
        
        def step(self, action):
            self.current_step += 1
            observation = np.random.random(50)
            reward = np.random.normal(0.001, 0.02)  # Simulated return
            done = self.current_step >= self.max_steps
            info = {'portfolio_value': 10000 * (1 + self.current_step * 0.001)}
            return observation, reward, done, info
        
        def load_data(self, start_date, end_date):
            return True
    
    TradingEnvironment = MockTradingEnvironment
    
    def get_rl_config():
        return type('config', (), {})()


def generate_market_data(step: int, regime: MarketRegime = None) -> dict:
    """Generate realistic market data for testing"""
    
    # Base market data
    base_price = 50000  # BTC base price
    
    # Regime-specific patterns
    if regime == MarketRegime.BULL:
        trend = 0.002  # Upward trend
        volatility = 0.015
    elif regime == MarketRegime.BEAR:
        trend = -0.002  # Downward trend
        volatility = 0.025
    elif regime == MarketRegime.HIGH_VOLATILITY:
        trend = 0.0
        volatility = 0.035
    elif regime == MarketRegime.SIDEWAYS:
        trend = 0.0
        volatility = 0.008
    else:
        trend = np.random.normal(0, 0.001)
        volatility = 0.02
    
    # Generate price history
    price_changes = np.random.normal(trend, volatility, 100)
    price_history = base_price * np.cumprod(1 + price_changes)
    
    # Generate volume history
    volume_history = np.random.exponential(1000, 100)
    
    # Current market state
    current_price = price_history[-1]
    prev_price = price_history[-2] if len(price_history) > 1 else current_price
    
    return {
        'price_history': price_history,
        'volume_history': volume_history,
        'current_price': current_price,
        'price_change': (current_price - prev_price) / prev_price,
        'volatility': volatility,
        'trend_strength': abs(trend) / volatility if volatility > 0 else 0,
        'momentum': np.mean(np.diff(price_history[-10:])) / prev_price,
        'portfolio_value': 10000 + step * 50,  # Simulated portfolio growth
        'current_drawdown': max(0, np.random.exponential(0.02)),
        'portfolio_return': step * 0.001,  # Cumulative return
        'sharpe_ratio': np.random.normal(1.0, 0.3),
        'win_rate': 0.55 + np.random.normal(0, 0.1)
    }


def simulate_market_regimes(num_steps: int) -> list:
    """Simulate changing market regimes over time"""
    
    regimes = []
    current_regime = MarketRegime.SIDEWAYS
    regime_duration = 0
    
    for step in range(num_steps):
        # Change regime periodically
        if regime_duration > 30 + np.random.exponential(20):
            # Transition to new regime
            regime_probs = {
                MarketRegime.BULL: 0.25,
                MarketRegime.BEAR: 0.2,
                MarketRegime.SIDEWAYS: 0.3,
                MarketRegime.HIGH_VOLATILITY: 0.15,
                MarketRegime.MEAN_REVERTING: 0.1
            }
            
            current_regime = np.random.choice(
                list(regime_probs.keys()),
                p=list(regime_probs.values())
            )
            regime_duration = 0
        
        regimes.append(current_regime)
        regime_duration += 1
    
    return regimes


def run_ensemble_example():
    """Run complete ensemble system example"""
    
    print("=" * 80)
    print("MULTI-AGENT ENSEMBLE TRADING SYSTEM - EXAMPLE")
    print("=" * 80)
    
    # 1. Initialize the ensemble system
    print("\n1. Initializing Ensemble System...")
    ensemble = create_default_ensemble_system()
    
    # Create mock environment
    env = TradingEnvironment(mode='train')
    
    # Initialize ensemble
    ensemble.initialize(env)
    
    # Get system summary
    summary = ensemble.get_system_summary()
    print(f"   ✓ Initialized with {summary['system_info']['num_agents']} specialized agents")
    
    for agent_name, agent_info in summary['agents'].items():
        print(f"     - {agent_name}: {agent_info['name']}")
    
    # 2. Train the agents (simplified for example)
    print("\n2. Training Ensemble Agents...")
    print("   (Using reduced timesteps for demonstration)")
    
    # In practice, you would use much larger timestep counts
    training_results = ensemble.train_all_agents(total_timesteps=1000)
    
    print("   ✓ Training completed")
    for agent_type, result in training_results.items():
        if 'error' not in result:
            print(f"     - {agent_type}: Successfully trained")
        else:
            print(f"     - {agent_type}: Training failed")
    
    # 3. Simulate trading across different market regimes
    print("\n3. Simulating Trading Across Market Regimes...")
    
    num_trading_steps = 200
    regimes = simulate_market_regimes(num_trading_steps)
    
    trading_results = []
    ensemble_decisions = []
    
    # Reset environment
    observation = env.reset()
    
    for step in range(num_trading_steps):
        current_regime = regimes[step]
        
        # Generate market data for this step
        market_data = generate_market_data(step, current_regime)
        
        # Get ensemble prediction
        action, decision = ensemble.predict(observation, market_data, deterministic=True)
        
        # Execute action in environment
        next_observation, reward, done, info = env.step(action)
        
        # Simulate individual agent returns (for attribution analysis)
        individual_returns = {}
        for agent_type in ensemble.orchestrator.agents.keys():
            # Different agents perform differently in different regimes
            base_return = reward
            
            if current_regime == MarketRegime.BULL and agent_type == AgentType.AGGRESSIVE:
                agent_return = base_return + np.random.normal(0.001, 0.005)
            elif current_regime == MarketRegime.BEAR and agent_type == AgentType.CONSERVATIVE:
                agent_return = base_return + np.random.normal(0.0005, 0.003)
            elif current_regime == MarketRegime.SIDEWAYS and agent_type == AgentType.CONTRARIAN:
                agent_return = base_return + np.random.normal(0.0008, 0.004)
            else:
                agent_return = base_return + np.random.normal(0, 0.004)
            
            individual_returns[agent_type] = agent_return
        
        # Update ensemble performance
        ensemble.update_performance(reward, decision, individual_returns)
        
        # Record results
        trading_results.append({
            'step': step,
            'regime': current_regime.value,
            'ensemble_return': reward,
            'action': action[0] if len(action) > 0 else 0,
            'strategy_used': decision.strategy_used.value,
            'regime_confidence': decision.regime_confidence,
            'active_agents': decision.explanation.get('active_agents', [])
        })
        
        ensemble_decisions.append(decision)
        observation = next_observation
        
        # Log progress periodically
        if step % 50 == 0:
            metrics = ensemble.get_comprehensive_metrics()
            ensemble_perf = metrics['ensemble_performance']
            current_return = ensemble_perf.get('total_return', 0) * 100
            print(f"   Step {step:3d}: Regime={current_regime.value:15s} | "
                  f"Return={current_return:6.2f}% | Strategy={decision.strategy_used.value}")
        
        if done:
            observation = env.reset()
    
    print("   ✓ Trading simulation completed")
    
    # 4. Analyze performance
    print("\n4. Performance Analysis...")
    
    # Generate comprehensive performance report
    performance_report = ensemble.generate_performance_report()
    
    # Display key metrics
    ensemble_metrics = performance_report['ensemble_performance']
    print(f"   Ensemble Performance:")
    print(f"     - Total Return: {ensemble_metrics['total_return']:.2%}")
    print(f"     - Sharpe Ratio: {ensemble_metrics['sharpe_ratio']:.3f}")
    print(f"     - Max Drawdown: {ensemble_metrics['max_drawdown']:.2%}")
    print(f"     - Win Rate: {ensemble_metrics['win_rate']:.2%}")
    print(f"     - Volatility: {ensemble_metrics['volatility']:.2%}")
    print(f"     - Number of Trades: {ensemble_metrics['num_trades']}")
    
    # Agent comparison
    agent_comparison = performance_report['agent_comparison']
    print(f"\n   Individual Agent Performance:")
    
    agent_rankings = []
    for agent_name, metrics in agent_comparison.items():
        if agent_name != 'ensemble' and metrics.get('num_trades', 0) > 0:
            agent_rankings.append((agent_name, metrics.get('sharpe_ratio', 0)))
    
    agent_rankings.sort(key=lambda x: x[1], reverse=True)
    
    for i, (agent_name, sharpe) in enumerate(agent_rankings):
        metrics = agent_comparison[agent_name]
        print(f"     {i+1}. {agent_name:12s}: Return={metrics['total_return']:6.2%} | "
              f"Sharpe={sharpe:5.2f} | Trades={metrics['num_trades']}")
    
    # Regime performance analysis
    if 'regime_analysis' in performance_report and performance_report['regime_analysis']:
        print(f"\n   Performance by Market Regime:")
        regime_analysis = performance_report['regime_analysis']
        
        for regime_name, regime_metrics in regime_analysis.items():
            print(f"     {regime_name:15s}: Return={regime_metrics['mean_return']:7.4f} | "
                  f"Count={regime_metrics['count']:3d} | Win Rate={regime_metrics['win_rate']:.2%}")
    
    # Risk assessment
    if 'risk_assessment' in performance_report:
        risk_assessment = performance_report['risk_assessment']
        print(f"\n   Risk Assessment: {risk_assessment['risk_level']}")
        
        if risk_assessment['risk_warnings']:
            print("   Risk Warnings:")
            for warning in risk_assessment['risk_warnings']:
                print(f"     ⚠️  {warning}")
    
    # 5. System insights
    print("\n5. System Insights...")
    
    # Bandit statistics
    comprehensive_metrics = ensemble.get_comprehensive_metrics()
    bandit_stats = comprehensive_metrics.get('bandit_statistics', {})
    
    if bandit_stats.get('agent_selection_counts'):
        print("   Agent Selection Frequency:")
        selection_counts = bandit_stats['agent_selection_counts']
        total_selections = sum(selection_counts.values())
        
        for agent_name, count in sorted(selection_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_selections) * 100 if total_selections > 0 else 0
            print(f"     {agent_name:12s}: {count:3d} selections ({percentage:5.1f}%)")
    
    # Strategy usage
    strategy_usage = {}
    for result in trading_results:
        strategy = result['strategy_used']
        strategy_usage[strategy] = strategy_usage.get(strategy, 0) + 1
    
    print("   Decision Strategy Usage:")
    total_decisions = len(trading_results)
    for strategy, count in sorted(strategy_usage.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_decisions) * 100 if total_decisions > 0 else 0
        print(f"     {strategy:20s}: {count:3d} times ({percentage:5.1f}%)")
    
    # Regime distribution
    regime_distribution = {}
    for result in trading_results:
        regime = result['regime']
        regime_distribution[regime] = regime_distribution.get(regime, 0) + 1
    
    print("   Market Regime Distribution:")
    for regime, count in sorted(regime_distribution.items(), key=lambda x: x[1], reverse=True):
        percentage = (count / total_decisions) * 100 if total_decisions > 0 else 0
        print(f"     {regime:15s}: {count:3d} steps ({percentage:5.1f}%)")
    
    # 6. Save results
    print("\n6. Saving Results...")
    
    # Save system state
    base_path = "/tmp/ensemble_example_state"
    ensemble.save_system_state(base_path)
    print(f"   ✓ System state saved to {base_path}_*")
    
    # Save trading results to CSV
    results_df = pd.DataFrame(trading_results)
    results_path = "/tmp/ensemble_trading_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"   ✓ Trading results saved to {results_path}")
    
    # Save performance report
    report_path = "/tmp/ensemble_performance_report.json"
    import json
    with open(report_path, 'w') as f:
        json.dump(performance_report, f, indent=2, default=str)
    print(f"   ✓ Performance report saved to {report_path}")
    
    print("\n" + "=" * 80)
    print("ENSEMBLE SYSTEM EXAMPLE COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    # Return results for further analysis
    return {
        'ensemble': ensemble,
        'trading_results': trading_results,
        'performance_report': performance_report,
        'comprehensive_metrics': comprehensive_metrics
    }


def demonstrate_regime_adaptation():
    """Demonstrate how the ensemble adapts to different market regimes"""
    
    print("\n" + "=" * 60)
    print("REGIME ADAPTATION DEMONSTRATION")
    print("=" * 60)
    
    # Create ensemble
    ensemble = create_default_ensemble_system()
    env = TradingEnvironment(mode='train')
    ensemble.initialize(env)
    
    # Test different regimes
    test_regimes = [
        MarketRegime.BULL,
        MarketRegime.BEAR,
        MarketRegime.HIGH_VOLATILITY,
        MarketRegime.SIDEWAYS,
        MarketRegime.MEAN_REVERTING
    ]
    
    observation = np.random.random(50)
    
    for regime in test_regimes:
        print(f"\nTesting {regime.value.upper()} market regime:")
        
        # Generate regime-specific market data
        market_data = generate_market_data(0, regime)
        
        # Get ensemble prediction
        action, decision = ensemble.predict(observation, market_data)
        
        print(f"  Detected Regime: {decision.regime.value} (confidence: {decision.regime_confidence:.2f})")
        print(f"  Strategy Used: {decision.strategy_used.value}")
        print(f"  Risk Level: {decision.risk_level.value if decision.risk_level else 'adaptive'}")
        print(f"  Active Agents: {', '.join(decision.explanation.get('active_agents', []))}")
        print(f"  Final Action: {action[0]:.4f}")
        
        # Show agent weights
        if decision.agent_weights:
            print("  Agent Weights:")
            for agent_name, weight in decision.agent_weights.items():
                print(f"    {agent_name:12s}: {weight:.3f}")


if __name__ == "__main__":
    print("Starting Multi-Agent Ensemble System Example...")
    
    try:
        # Run the main example
        results = run_ensemble_example()
        
        # Run regime adaptation demonstration
        demonstrate_regime_adaptation()
        
        print("\n✅ Example completed successfully!")
        print("\nKey Files Generated:")
        print("  - /tmp/ensemble_example_state_* (system state files)")
        print("  - /tmp/ensemble_trading_results.csv (trading results)")
        print("  - /tmp/ensemble_performance_report.json (performance report)")
        
    except Exception as e:
        logger.error(f"Example failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise