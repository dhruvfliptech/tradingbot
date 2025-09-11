"""
PPO Agent for Crypto Trading

This module implements a sophisticated PPO (Proximal Policy Optimization) agent
specifically designed for cryptocurrency trading with:
- Custom policy networks with attention mechanisms
- Risk-aware action selection
- Model versioning and A/B testing support
- Production-ready inference and deployment features
"""

import os
import json
import pickle
import logging
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import gymnasium as gym
from gymnasium import spaces

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed

from .policy_network import (
    TradingPolicyNetwork, AttentionPolicy, RiskAwarePolicy, 
    NetworkConfig, create_policy_network
)

logger = logging.getLogger(__name__)


@dataclass
class PPOConfig:
    """Configuration for PPO agent"""
    # Algorithm parameters
    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None
    normalize_advantage: bool = True
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    use_sde: bool = False
    sde_sample_freq: int = -1
    target_kl: Optional[float] = None
    
    # Policy network configuration
    policy_type: str = 'attention'  # 'standard', 'attention', 'risk_aware'
    policy_kwargs: Dict[str, Any] = None
    
    # Training configuration
    total_timesteps: int = 1000000
    eval_freq: int = 10000
    n_eval_episodes: int = 10
    eval_deterministic: bool = True
    
    # Model management
    model_version: str = "1.0.0"
    model_name: str = "ppo_trading_agent"
    save_path: str = "/tmp/rl_models"
    checkpoint_freq: int = 50000
    
    # Production features
    enable_versioning: bool = True
    enable_ab_testing: bool = True
    rollout_percentage: float = 0.1  # For gradual rollouts
    performance_threshold: float = 0.15  # 15% improvement threshold
    
    # Risk management
    max_drawdown_threshold: float = 0.2  # 20% max drawdown
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    enable_risk_constraints: bool = True
    
    def __post_init__(self):
        if self.policy_kwargs is None:
            self.policy_kwargs = {}
        
        # Ensure save path exists
        Path(self.save_path).mkdir(parents=True, exist_ok=True)


class ModelVersion:
    """Model versioning system for production deployment"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.versions_file = self.base_path / "versions.json"
        self.versions = self._load_versions()
    
    def _load_versions(self) -> Dict[str, Any]:
        """Load version registry"""
        if self.versions_file.exists():
            with open(self.versions_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_versions(self):
        """Save version registry"""
        with open(self.versions_file, 'w') as f:
            json.dump(self.versions, f, indent=2, default=str)
    
    def register_version(self, version: str, model_path: str, 
                        metadata: Dict[str, Any]) -> str:
        """Register a new model version"""
        version_info = {
            'version': version,
            'model_path': model_path,
            'created_at': datetime.now().isoformat(),
            'metadata': metadata,
            'status': 'created'  # created, testing, deployed, deprecated
        }
        
        self.versions[version] = version_info
        self._save_versions()
        
        logger.info(f"Registered model version {version}")
        return version
    
    def get_version_info(self, version: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific version"""
        return self.versions.get(version)
    
    def list_versions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all versions, optionally filtered by status"""
        versions = list(self.versions.values())
        if status:
            versions = [v for v in versions if v.get('status') == status]
        return sorted(versions, key=lambda x: x['created_at'], reverse=True)
    
    def set_version_status(self, version: str, status: str):
        """Update version status"""
        if version in self.versions:
            self.versions[version]['status'] = status
            self._save_versions()
            logger.info(f"Set version {version} status to {status}")
    
    def get_latest_deployed(self) -> Optional[str]:
        """Get the latest deployed version"""
        deployed = [v for v in self.versions.values() if v.get('status') == 'deployed']
        if deployed:
            return max(deployed, key=lambda x: x['created_at'])['version']
        return None


class PerformanceTracker:
    """Track and compare model performance for A/B testing"""
    
    def __init__(self, baseline_metrics: Optional[Dict[str, float]] = None):
        self.baseline_metrics = baseline_metrics or {}
        self.current_metrics = {}
        self.performance_history = []
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update current performance metrics"""
        self.current_metrics = metrics.copy()
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics.copy()
        })
    
    def compare_to_baseline(self) -> Dict[str, float]:
        """Compare current performance to baseline"""
        if not self.baseline_metrics:
            return {}
        
        comparison = {}
        for metric, current_value in self.current_metrics.items():
            if metric in self.baseline_metrics:
                baseline_value = self.baseline_metrics[metric]
                if baseline_value != 0:
                    improvement = (current_value - baseline_value) / abs(baseline_value)
                    comparison[metric] = improvement
        
        return comparison
    
    def meets_deployment_criteria(self, threshold: float = 0.15) -> bool:
        """Check if model meets deployment criteria"""
        comparison = self.compare_to_baseline()
        
        # Check if key metrics show improvement
        key_metrics = ['total_return', 'sharpe_ratio', 'max_drawdown']
        improvements = []
        
        for metric in key_metrics:
            if metric in comparison:
                improvements.append(comparison[metric])
        
        if not improvements:
            return False
        
        # For max_drawdown, improvement means lower (better) drawdown
        if 'max_drawdown' in comparison:
            comparison['max_drawdown'] *= -1  # Flip sign for drawdown
        
        # Check if average improvement exceeds threshold
        avg_improvement = np.mean(improvements)
        return avg_improvement >= threshold


class PPOTradingCallback(BaseCallback):
    """Custom callback for PPO training with enhanced monitoring"""
    
    def __init__(self, eval_env: VecEnv, eval_freq: int = 10000,
                 n_eval_episodes: int = 10, save_path: str = "/tmp/rl_models",
                 performance_tracker: Optional[PerformanceTracker] = None,
                 verbose: int = 1):
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.save_path = Path(save_path)
        self.performance_tracker = performance_tracker
        
        self.best_mean_reward = -np.inf
        self.evaluations = []
        
    def _on_step(self) -> bool:
        """Called at each training step"""
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self._evaluate_model()
        
        return True
    
    def _evaluate_model(self):
        """Evaluate model performance"""
        try:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=True,
                return_episode_rewards=True
            )
            
            mean_reward = np.mean(episode_rewards)
            std_reward = np.std(episode_rewards)
            mean_length = np.mean(episode_lengths)
            
            # Calculate additional metrics
            sharpe_ratio = mean_reward / std_reward if std_reward > 0 else 0
            
            evaluation = {
                'timesteps': self.n_calls,
                'mean_reward': mean_reward,
                'std_reward': std_reward,
                'mean_length': mean_length,
                'sharpe_ratio': sharpe_ratio,
                'timestamp': datetime.now().isoformat()
            }
            
            self.evaluations.append(evaluation)
            
            # Update performance tracker
            if self.performance_tracker:
                metrics = {
                    'total_return': mean_reward,
                    'sharpe_ratio': sharpe_ratio,
                    'volatility': std_reward
                }
                self.performance_tracker.update_metrics(metrics)
            
            # Save best model
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                best_model_path = self.save_path / f"best_model_{self.n_calls}.zip"
                self.model.save(best_model_path)
                
                if self.verbose >= 1:
                    logger.info(f"New best model saved with reward {mean_reward:.2f}")
            
            if self.verbose >= 1:
                logger.info(f"Evaluation at {self.n_calls} steps:")
                logger.info(f"  Mean reward: {mean_reward:.2f} ± {std_reward:.2f}")
                logger.info(f"  Sharpe ratio: {sharpe_ratio:.2f}")
                logger.info(f"  Mean length: {mean_length:.1f}")
                
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")


class PPOAgent:
    """
    Production-ready PPO agent for cryptocurrency trading
    
    Features:
    - Custom policy networks with attention and risk awareness
    - Model versioning and A/B testing
    - Performance tracking and comparison
    - Gradual rollout capabilities
    - Risk constraints and monitoring
    """
    
    def __init__(self, env: gym.Env, config: Optional[PPOConfig] = None):
        self.env = env
        self.config = config or PPOConfig()
        self.model = None
        self.version_manager = None
        self.performance_tracker = None
        
        if self.config.enable_versioning:
            self.version_manager = ModelVersion(self.config.save_path)
        
        if self.config.enable_ab_testing:
            self.performance_tracker = PerformanceTracker()
        
        self._setup_logging()
        self._initialize_model()
    
    def _setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, os.getenv('LOG_LEVEL', 'INFO').upper())
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def _initialize_model(self):
        """Initialize PPO model with custom policy"""
        try:
            # Create policy kwargs
            policy_kwargs = self.config.policy_kwargs.copy()
            
            # Add network configuration
            if 'config' not in policy_kwargs:
                policy_kwargs['config'] = NetworkConfig(
                    hidden_dims=[256, 256],
                    attention_heads=8,
                    dropout_rate=0.1
                )
            
            # Select policy class
            policy_class = {
                'standard': TradingPolicyNetwork,
                'attention': AttentionPolicy,
                'risk_aware': RiskAwarePolicy
            }.get(self.config.policy_type, AttentionPolicy)
            
            # Create model
            self.model = PPO(
                policy=policy_class,
                env=self.env,
                learning_rate=self.config.learning_rate,
                n_steps=self.config.n_steps,
                batch_size=self.config.batch_size,
                n_epochs=self.config.n_epochs,
                gamma=self.config.gamma,
                gae_lambda=self.config.gae_lambda,
                clip_range=self.config.clip_range,
                clip_range_vf=self.config.clip_range_vf,
                normalize_advantage=self.config.normalize_advantage,
                ent_coef=self.config.ent_coef,
                vf_coef=self.config.vf_coef,
                max_grad_norm=self.config.max_grad_norm,
                use_sde=self.config.use_sde,
                sde_sample_freq=self.config.sde_sample_freq,
                target_kl=self.config.target_kl,
                policy_kwargs=policy_kwargs,
                verbose=1
            )
            
            logger.info(f"PPO model initialized with {self.config.policy_type} policy")
            
        except Exception as e:
            logger.error(f"Error initializing PPO model: {e}")
            raise
    
    def train(self, total_timesteps: Optional[int] = None,
              eval_env: Optional[gym.Env] = None,
              callback_list: Optional[List[BaseCallback]] = None) -> Dict[str, Any]:
        """
        Train the PPO agent
        
        Args:
            total_timesteps: Total training timesteps
            eval_env: Environment for evaluation
            callback_list: Additional callbacks
            
        Returns:
            Training summary
        """
        if not self.model:
            raise ValueError("Model not initialized")
        
        timesteps = total_timesteps or self.config.total_timesteps
        
        # Setup evaluation environment
        if eval_env:
            eval_vec_env = DummyVecEnv([lambda: eval_env])
        else:
            eval_vec_env = DummyVecEnv([lambda: self.env])
        
        # Setup callbacks
        callbacks = callback_list or []
        
        # Add custom callback
        ppo_callback = PPOTradingCallback(
            eval_env=eval_vec_env,
            eval_freq=self.config.eval_freq,
            n_eval_episodes=self.config.n_eval_episodes,
            save_path=self.config.save_path,
            performance_tracker=self.performance_tracker
        )
        callbacks.append(ppo_callback)
        
        try:
            logger.info(f"Starting training for {timesteps:,} timesteps")
            start_time = datetime.now()
            
            # Train model
            self.model.learn(
                total_timesteps=timesteps,
                callback=callbacks,
                reset_num_timesteps=False
            )
            
            training_time = datetime.now() - start_time
            
            # Save final model
            final_model_path = self._save_model(
                suffix=f"final_{timesteps}",
                metadata={
                    'total_timesteps': timesteps,
                    'training_time_seconds': training_time.total_seconds(),
                    'config': asdict(self.config)
                }
            )
            
            # Training summary
            summary = {
                'total_timesteps': timesteps,
                'training_time': training_time.total_seconds(),
                'model_path': str(final_model_path),
                'evaluations': ppo_callback.evaluations,
                'best_mean_reward': ppo_callback.best_mean_reward
            }
            
            logger.info(f"Training completed in {training_time}")
            logger.info(f"Best mean reward: {ppo_callback.best_mean_reward:.2f}")
            
            return summary
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def predict(self, observation: np.ndarray, 
                deterministic: bool = True) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
        """
        Predict action for given observation
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic policy
            
        Returns:
            Tuple of (action, additional_info)
        """
        if not self.model:
            raise ValueError("Model not initialized or loaded")
        
        try:
            action, state = self.model.predict(observation, deterministic=deterministic)
            
            # Get additional info if available
            info = {}
            if hasattr(self.model.policy, 'get_attention_weights'):
                attention_weights = self.model.policy.get_attention_weights()
                if attention_weights is not None:
                    info['attention_weights'] = attention_weights.detach().cpu().numpy()
            
            if hasattr(self.model.policy, 'get_risk_metrics'):
                obs_tensor = torch.FloatTensor(observation).unsqueeze(0)
                risk_metrics = self.model.policy.get_risk_metrics(obs_tensor)
                info['risk_metrics'] = {
                    k: v.detach().cpu().numpy() for k, v in risk_metrics.items()
                }
            
            return action, info
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise
    
    def evaluate(self, eval_env: gym.Env, n_episodes: int = 10,
                deterministic: bool = True) -> Dict[str, float]:
        """
        Evaluate agent performance
        
        Args:
            eval_env: Evaluation environment
            n_episodes: Number of evaluation episodes
            deterministic: Whether to use deterministic policy
            
        Returns:
            Evaluation metrics
        """
        if not self.model:
            raise ValueError("Model not initialized or loaded")
        
        try:
            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                eval_env,
                n_eval_episodes=n_episodes,
                deterministic=deterministic,
                return_episode_rewards=True
            )
            
            # Calculate metrics
            metrics = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards),
                'min_reward': np.min(episode_rewards),
                'max_reward': np.max(episode_rewards),
                'mean_length': np.mean(episode_lengths),
                'total_episodes': len(episode_rewards)
            }
            
            # Calculate Sharpe ratio
            if metrics['std_reward'] > 0:
                excess_return = metrics['mean_reward'] - self.config.risk_free_rate / 252
                metrics['sharpe_ratio'] = excess_return / metrics['std_reward']
            else:
                metrics['sharpe_ratio'] = 0.0
            
            # Calculate max drawdown (simplified)
            cumulative_returns = np.cumsum(episode_rewards)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdowns = (cumulative_returns - running_max) / np.maximum(running_max, 1)
            metrics['max_drawdown'] = np.min(drawdowns)
            
            logger.info(f"Evaluation completed over {n_episodes} episodes:")
            logger.info(f"  Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
            logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
            logger.info(f"  Max drawdown: {metrics['max_drawdown']:.2%}")
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            raise
    
    def _save_model(self, suffix: str = "", metadata: Optional[Dict[str, Any]] = None) -> Path:
        """Save model with versioning"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"{self.config.model_name}_{timestamp}{suffix}"
        model_path = Path(self.config.save_path) / f"{model_name}.zip"
        
        # Save model
        self.model.save(model_path)
        
        # Save metadata
        if metadata:
            metadata_path = model_path.with_suffix('.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        # Register version if versioning enabled
        if self.version_manager:
            version = f"{self.config.model_version}_{timestamp}"
            self.version_manager.register_version(
                version=version,
                model_path=str(model_path),
                metadata=metadata or {}
            )
        
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def load_model(self, model_path: str):
        """Load model from path"""
        try:
            self.model = PPO.load(model_path, env=self.env)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def deploy_model(self, version: str, rollout_percentage: float = None) -> bool:
        """
        Deploy model version for production use
        
        Args:
            version: Model version to deploy
            rollout_percentage: Percentage of traffic to route to new model
            
        Returns:
            True if deployment successful
        """
        if not self.version_manager:
            logger.error("Model versioning not enabled")
            return False
        
        version_info = self.version_manager.get_version_info(version)
        if not version_info:
            logger.error(f"Version {version} not found")
            return False
        
        try:
            # Load model
            self.load_model(version_info['model_path'])
            
            # Update version status
            self.version_manager.set_version_status(version, 'deployed')
            
            # Setup gradual rollout if specified
            rollout_pct = rollout_percentage or self.config.rollout_percentage
            
            logger.info(f"Model version {version} deployed with {rollout_pct:.1%} traffic")
            return True
            
        except Exception as e:
            logger.error(f"Error deploying model: {e}")
            return False
    
    def get_performance_comparison(self) -> Optional[Dict[str, float]]:
        """Get performance comparison with baseline"""
        if not self.performance_tracker:
            return None
        
        return self.performance_tracker.compare_to_baseline()
    
    def meets_deployment_criteria(self) -> bool:
        """Check if current model meets deployment criteria"""
        if not self.performance_tracker:
            return True
        
        return self.performance_tracker.meets_deployment_criteria(
            self.config.performance_threshold
        )


def create_ppo_agent(env: gym.Env, config_dict: Optional[Dict[str, Any]] = None) -> PPOAgent:
    """
    Factory function to create PPO agent with configuration
    
    Args:
        env: Trading environment
        config_dict: Configuration dictionary
        
    Returns:
        Configured PPO agent
    """
    config = PPOConfig(**config_dict) if config_dict else PPOConfig()
    return PPOAgent(env, config)


if __name__ == "__main__":
    # Example usage
    from environment.trading_env import TradingEnvironment
    from rl_config import get_rl_config
    
    # Create environment
    rl_config = get_rl_config()
    env = TradingEnvironment(config=rl_config, mode='train')
    
    # Load data
    from datetime import datetime
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    if env.load_data(start_date, end_date):
        # Create PPO agent
        ppo_config = PPOConfig(
            policy_type='attention',
            total_timesteps=50000,
            eval_freq=5000,
            enable_versioning=True,
            enable_ab_testing=True
        )
        
        agent = PPOAgent(env, ppo_config)
        
        # Train agent
        training_summary = agent.train()
        print(f"Training completed: {training_summary}")
        
        # Evaluate agent
        eval_metrics = agent.evaluate(env, n_episodes=5)
        print(f"Evaluation metrics: {eval_metrics}")
        
        # Check deployment criteria
        if agent.meets_deployment_criteria():
            print("Agent meets deployment criteria!")
        else:
            print("Agent needs more training or improvement")
    
    else:
        print("Failed to load data for training")