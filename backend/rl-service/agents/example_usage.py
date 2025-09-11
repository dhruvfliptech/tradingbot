"""
Example Usage of PPO Trading Agents

This script demonstrates how to use the complete PPO agent system including:
- Training individual PPO agents
- Training ensemble agents for different market conditions
- Model versioning and A/B testing
- Explainability and decision analysis
- Production deployment features
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent))

from environment.trading_env import TradingEnvironment
from rl_config import get_rl_config
from agents.ppo_agent import PPOAgent, PPOConfig, create_ppo_agent
from agents.trainer import PPOTrainer, TrainingConfig, create_trainer
from agents.ensemble_agent import EnsembleAgent, EnsembleConfig, create_ensemble_agent
from agents.explainer import AgentExplainer, create_explainer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_sample_environment():
    """Create and setup a sample trading environment"""
    logger.info("Creating sample trading environment...")
    
    # Get RL configuration
    rl_config = get_rl_config()
    
    # Create environment
    env = TradingEnvironment(config=rl_config, mode='train')
    
    # Load sample data
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    if env.load_data(start_date, end_date):
        logger.info("Environment created and data loaded successfully")
        return env
    else:
        raise RuntimeError("Failed to load data for environment")


def example_single_agent_training():
    """Example: Train a single PPO agent"""
    logger.info("=== Example 1: Single PPO Agent Training ===")
    
    # Create environment
    env = create_sample_environment()
    
    # Create PPO configuration
    ppo_config = PPOConfig(
        policy_type='attention',  # Use attention-based policy
        learning_rate=3e-4,
        total_timesteps=50000,
        n_steps=2048,
        batch_size=64,
        enable_versioning=True,
        enable_ab_testing=True,
        model_name="btc_trading_agent_v1"
    )
    
    # Create agent
    agent = PPOAgent(env, ppo_config)
    
    # Train agent
    logger.info("Starting training...")
    training_summary = agent.train(total_timesteps=50000)
    
    logger.info("Training completed!")
    logger.info(f"Best mean reward: {training_summary.get('best_mean_reward', 0):.2f}")
    
    # Evaluate agent
    eval_metrics = agent.evaluate(env, n_episodes=10)
    logger.info(f"Evaluation metrics:")
    logger.info(f"  Mean reward: {eval_metrics['mean_reward']:.2f}")
    logger.info(f"  Sharpe ratio: {eval_metrics['sharpe_ratio']:.2f}")
    logger.info(f"  Max drawdown: {eval_metrics['max_drawdown']:.2%}")
    
    # Check if agent meets deployment criteria
    if agent.meets_deployment_criteria():
        logger.info("✅ Agent meets deployment criteria!")
    else:
        logger.info("❌ Agent needs improvement before deployment")
    
    return agent, eval_metrics


def example_advanced_training_pipeline():
    """Example: Advanced training with hyperparameter optimization"""
    logger.info("=== Example 2: Advanced Training Pipeline ===")
    
    def create_env():
        return create_sample_environment()
    
    # Create training configuration
    training_config = TrainingConfig(
        total_timesteps=100000,
        enable_early_stopping=True,
        enable_tensorboard=True,
        enable_hyperopt=False,  # Disable for example (takes time)
        patience=10,
        checkpoint_frequency=10000,
        save_training_logs=True
    )
    
    # Create trainer
    trainer = PPOTrainer(create_env, training_config)
    
    # Create PPO configuration
    ppo_config = PPOConfig(
        policy_type='risk_aware',  # Use risk-aware policy
        learning_rate=3e-4,
        total_timesteps=100000,
        enable_versioning=True
    )
    
    # Train with advanced pipeline
    logger.info("Starting advanced training pipeline...")
    results = trainer.train(ppo_config)
    
    logger.info("Advanced training completed!")
    logger.info(f"Final Sharpe ratio: {results['final_metrics'].get('sharpe_ratio', 0):.2f}")
    
    # Performance comparison
    if results.get('comparison_results'):
        logger.info("Performance comparison with baseline:")
        improvements = results['comparison_results'].get('improvements', {})
        for metric, improvement in improvements.items():
            logger.info(f"  {metric}: {improvement:+.2%}")
    
    return results


def example_ensemble_training():
    """Example: Train ensemble of specialized agents"""
    logger.info("=== Example 3: Ensemble Agent Training ===")
    
    # Create environment
    env = create_sample_environment()
    
    # Create historical data for regime detection
    dates = pd.date_range(datetime(2024, 1, 1), datetime(2024, 6, 30), freq='1H')
    historical_data = pd.DataFrame({
        'timestamp': dates,
        'close': 50000 + np.cumsum(np.random.randn(len(dates)) * 100),
        'volume': np.random.randint(1000, 10000, len(dates))
    })
    historical_data['high'] = historical_data['close'] * 1.01
    historical_data['low'] = historical_data['close'] * 0.99
    
    # Create ensemble configuration
    ensemble_config = EnsembleConfig(
        n_agents=4,
        agent_types=['bull', 'bear', 'sideways', 'high_volatility'],
        weighting_method='confidence',
        train_individual_agents=True
    )
    
    # Create ensemble agent
    ensemble = EnsembleAgent(env, ensemble_config)
    
    # Create base PPO configuration for specialists
    base_config = PPOConfig(
        learning_rate=3e-4,
        total_timesteps=30000,  # Reduced for example
        enable_versioning=True
    )
    
    # Train ensemble
    logger.info("Training ensemble of specialized agents...")
    training_results = ensemble.train_ensemble(
        historical_data, base_config, total_timesteps=30000
    )
    
    logger.info("Ensemble training completed!")
    for agent_type, results in training_results.items():
        if 'eval_metrics' in results:
            sharpe = results['eval_metrics'].get('sharpe_ratio', 0)
            logger.info(f"  {agent_type}: Sharpe ratio = {sharpe:.2f}")
    
    # Evaluate ensemble
    logger.info("Evaluating ensemble performance...")
    ensemble_metrics = ensemble.evaluate_ensemble(
        env, n_episodes=5, market_data=historical_data
    )
    
    logger.info(f"Ensemble Sharpe ratio: {ensemble_metrics['sharpe_ratio']:.2f}")
    logger.info(f"Regime switches per episode: {ensemble_metrics['regime_switches_per_episode']:.1f}")
    
    return ensemble, ensemble_metrics


def example_explainability_analysis(agent, env):
    """Example: Explainability and decision analysis"""
    logger.info("=== Example 4: Explainability Analysis ===")
    
    # Create feature names (simplified for example)
    feature_names = [
        'close_price', 'volume', 'rsi_14', 'macd', 'sma_20', 'sma_50',
        'volatility', 'returns', 'portfolio_value', 'cash_balance',
        'position_size', 'unrealized_pnl', 'fear_greed_index', 'sentiment_score'
    ]
    
    # Pad or truncate to match observation space
    obs_size = env.observation_space.shape[0]
    if len(feature_names) < obs_size:
        feature_names.extend([f'feature_{i}' for i in range(len(feature_names), obs_size)])
    else:
        feature_names = feature_names[:obs_size]
    
    # Create explainer
    explainer = AgentExplainer(agent, feature_names, env)
    
    # Get sample observation
    obs, _ = env.reset()
    
    # Generate decision explanation
    logger.info("Generating decision explanation...")
    explanation = explainer.explain_decision(obs, include_counterfactuals=True)
    
    logger.info(f"Decision: {explanation.action_name}")
    logger.info(f"Confidence: {explanation.confidence:.2%}")
    logger.info(f"Explanation: {explanation.explanation_text}")
    
    # Show top feature contributions
    top_features = sorted(
        explanation.feature_contributions.items(), 
        key=lambda x: x[1], reverse=True
    )[:5]
    
    logger.info("Top contributing features:")
    for feature, contribution in top_features:
        logger.info(f"  {feature}: {contribution:.4f}")
    
    # Analyze feature importance
    logger.info("Analyzing feature importance...")
    importance_results = explainer.analyze_feature_importance(
        n_episodes=3, methods=['permutation', 'attention']
    )
    
    for method, importance in importance_results.items():
        top_important = importance.get_top_features(5)
        logger.info(f"Top 5 important features ({method}):")
        for feature, score in top_important:
            logger.info(f"  {feature}: {score:.4f}")
    
    # Generate comprehensive report
    report_path = "/tmp/agent_explanation_report.json"
    report = explainer.generate_explanation_report(obs, save_path=report_path)
    logger.info(f"Comprehensive explanation report saved to {report_path}")
    
    return explanation, importance_results


def example_production_deployment(agent):
    """Example: Production deployment features"""
    logger.info("=== Example 5: Production Deployment ===")
    
    if not agent.version_manager:
        logger.warning("Agent versioning not enabled, skipping deployment example")
        return
    
    # List available versions
    versions = agent.version_manager.list_versions()
    logger.info(f"Available model versions: {len(versions)}")
    
    for version_info in versions[:3]:  # Show first 3
        logger.info(f"  Version {version_info['version']}: {version_info['status']}")
    
    # Get latest version for deployment
    if versions:
        latest_version = versions[0]['version']
        
        # Deploy with gradual rollout
        logger.info(f"Deploying version {latest_version} with 10% traffic...")
        success = agent.deploy_model(latest_version, rollout_percentage=0.1)
        
        if success:
            logger.info("✅ Model deployed successfully!")
            
            # Check performance comparison
            comparison = agent.get_performance_comparison()
            if comparison:
                logger.info("Performance vs baseline:")
                for metric, improvement in comparison.items():
                    logger.info(f"  {metric}: {improvement:+.2%}")
        else:
            logger.info("❌ Model deployment failed")


def example_a_b_testing_workflow():
    """Example: A/B testing workflow"""
    logger.info("=== Example 6: A/B Testing Workflow ===")
    
    # Create two agents with different configurations
    env = create_sample_environment()
    
    # Agent A: Standard configuration
    config_a = PPOConfig(
        policy_type='standard',
        learning_rate=3e-4,
        model_name="agent_a_standard",
        enable_ab_testing=True
    )
    agent_a = PPOAgent(env, config_a)
    
    # Agent B: Attention-based configuration
    config_b = PPOConfig(
        policy_type='attention',
        learning_rate=5e-4,
        model_name="agent_b_attention",
        enable_ab_testing=True
    )
    agent_b = PPOAgent(env, config_b)
    
    # Train both agents (reduced timesteps for example)
    logger.info("Training Agent A (standard)...")
    agent_a.train(total_timesteps=20000)
    
    logger.info("Training Agent B (attention)...")
    agent_b.train(total_timesteps=20000)
    
    # Evaluate both agents
    metrics_a = agent_a.evaluate(env, n_episodes=5)
    metrics_b = agent_b.evaluate(env, n_episodes=5)
    
    # Compare performance
    logger.info("A/B Testing Results:")
    logger.info(f"Agent A - Sharpe: {metrics_a['sharpe_ratio']:.2f}, Return: {metrics_a['mean_reward']:.2f}")
    logger.info(f"Agent B - Sharpe: {metrics_b['sharpe_ratio']:.2f}, Return: {metrics_b['mean_reward']:.2f}")
    
    # Determine winner
    if metrics_b['sharpe_ratio'] > metrics_a['sharpe_ratio']:
        logger.info("🏆 Agent B (attention) wins the A/B test!")
        winner = agent_b
    else:
        logger.info("🏆 Agent A (standard) wins the A/B test!")
        winner = agent_a
    
    # Check deployment criteria for winner
    if winner.meets_deployment_criteria():
        logger.info("✅ Winner meets deployment criteria!")
    else:
        logger.info("❌ Winner needs improvement before deployment")
    
    return winner


def main():
    """Run all examples"""
    logger.info("🚀 Starting PPO Trading Agent Examples")
    
    try:
        # Example 1: Basic agent training
        agent, metrics = example_single_agent_training()
        
        # Example 2: Advanced training pipeline
        # training_results = example_advanced_training_pipeline()
        
        # Example 3: Ensemble training
        # ensemble, ensemble_metrics = example_ensemble_training()
        
        # Example 4: Explainability analysis
        env = create_sample_environment()
        explanation, importance = example_explainability_analysis(agent, env)
        
        # Example 5: Production deployment
        example_production_deployment(agent)
        
        # Example 6: A/B testing
        # winner = example_a_b_testing_workflow()
        
        logger.info("🎉 All examples completed successfully!")
        
        # Summary
        logger.info("\n=== Summary ===")
        logger.info(f"Single agent Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"Decision explanation generated: {explanation.action_name}")
        logger.info(f"Feature importance methods tested: {len(importance)}")
        
    except Exception as e:
        logger.error(f"Error running examples: {e}")
        raise


if __name__ == "__main__":
    main()