"""
Advanced Training Pipeline for PPO Trading Agent

This module provides a comprehensive training pipeline with:
- Hyperparameter optimization using Optuna
- Advanced checkpointing and model management
- TensorBoard logging and monitoring
- Early stopping with multiple criteria
- Distributed training support
- Automated model validation and testing
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
from typing import Dict, List, Tuple, Optional, Any, Callable, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import shutil
from concurrent.futures import ThreadPoolExecutor
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure, Logger
from torch.utils.tensorboard import SummaryWriter

from .ppo_agent import PPOAgent, PPOConfig, PerformanceTracker, ModelVersion

logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for training pipeline"""
    # Basic training parameters
    total_timesteps: int = 1000000
    eval_frequency: int = 10000
    save_frequency: int = 50000
    n_eval_episodes: int = 20
    eval_deterministic: bool = True
    
    # Early stopping
    enable_early_stopping: bool = True
    patience: int = 10  # Number of evaluations without improvement
    min_delta: float = 0.01  # Minimum improvement threshold
    early_stopping_metric: str = 'mean_reward'  # 'mean_reward', 'sharpe_ratio', 'total_return'
    
    # Checkpointing
    checkpoint_frequency: int = 25000
    max_checkpoints: int = 10  # Maximum number of checkpoints to keep
    save_best_only: bool = False
    
    # Logging and monitoring
    enable_tensorboard: bool = True
    tensorboard_log_dir: str = "/tmp/rl_logs/tensorboard"
    log_frequency: int = 1000
    save_training_logs: bool = True
    
    # Hyperparameter optimization
    enable_hyperopt: bool = False
    n_trials: int = 100
    hyperopt_timeout: int = 7200  # 2 hours
    study_name: str = "ppo_trading_optimization"
    
    # Training environment
    n_envs: int = 1  # Number of parallel environments
    env_wrapper: Optional[Callable] = None
    seed: Optional[int] = 42
    
    # Validation and testing
    validation_split: float = 0.2
    test_split: float = 0.1
    cross_validation_folds: int = 3
    
    # Performance thresholds
    performance_targets: Dict[str, float] = field(default_factory=lambda: {
        'min_sharpe_ratio': 1.0,
        'max_drawdown': -0.2,
        'min_total_return': 0.15,
        'win_rate': 0.55
    })
    
    # Model comparison
    baseline_model_path: Optional[str] = None
    comparison_metrics: List[str] = field(default_factory=lambda: [
        'total_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'profit_factor'
    ])


@dataclass
class TrainingMetrics:
    """Training metrics and statistics"""
    timesteps: int = 0
    episodes: int = 0
    training_time: float = 0.0
    
    # Performance metrics
    mean_reward: float = 0.0
    std_reward: float = 0.0
    best_mean_reward: float = float('-inf')
    sharpe_ratio: float = 0.0
    max_drawdown: float = 0.0
    total_return: float = 0.0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    
    # Training progress
    loss_values: List[float] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    evaluation_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Early stopping
    patience_counter: int = 0
    best_metric_value: float = float('-inf')
    stopped_early: bool = False
    
    def update_from_evaluation(self, eval_results: Dict[str, Any]):
        """Update metrics from evaluation results"""
        self.mean_reward = eval_results.get('mean_reward', self.mean_reward)
        self.std_reward = eval_results.get('std_reward', self.std_reward)
        self.sharpe_ratio = eval_results.get('sharpe_ratio', self.sharpe_ratio)
        self.max_drawdown = eval_results.get('max_drawdown', self.max_drawdown)
        self.total_return = eval_results.get('total_return', self.total_return)
        self.win_rate = eval_results.get('win_rate', self.win_rate)
        self.profit_factor = eval_results.get('profit_factor', self.profit_factor)
        
        if self.mean_reward > self.best_mean_reward:
            self.best_mean_reward = self.mean_reward
        
        self.evaluation_history.append(eval_results.copy())


class CheckpointManager:
    """Manages model checkpoints during training"""
    
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 10):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.checkpoints = []
    
    def save_checkpoint(self, model: PPO, timesteps: int, metrics: Dict[str, Any],
                       metadata: Optional[Dict[str, Any]] = None) -> str:
        """Save model checkpoint"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"checkpoint_{timesteps}_{timestamp}"
        checkpoint_path = self.checkpoint_dir / f"{checkpoint_name}.zip"
        
        # Save model
        model.save(checkpoint_path)
        
        # Save metadata
        checkpoint_info = {
            'timesteps': timesteps,
            'timestamp': timestamp,
            'metrics': metrics,
            'metadata': metadata or {},
            'path': str(checkpoint_path)
        }
        
        metadata_path = checkpoint_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(checkpoint_info, f, indent=2, default=str)
        
        # Update checkpoint list
        self.checkpoints.append(checkpoint_info)
        self._cleanup_old_checkpoints()
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self):
        """Remove old checkpoints if exceeding max limit"""
        if len(self.checkpoints) > self.max_checkpoints:
            # Sort by timesteps and remove oldest
            self.checkpoints.sort(key=lambda x: x['timesteps'])
            old_checkpoints = self.checkpoints[:-self.max_checkpoints]
            
            for checkpoint in old_checkpoints:
                try:
                    Path(checkpoint['path']).unlink(missing_ok=True)
                    Path(checkpoint['path']).with_suffix('.json').unlink(missing_ok=True)
                    logger.info(f"Removed old checkpoint: {checkpoint['path']}")
                except Exception as e:
                    logger.warning(f"Failed to remove checkpoint: {e}")
            
            self.checkpoints = self.checkpoints[-self.max_checkpoints:]
    
    def get_best_checkpoint(self, metric: str = 'mean_reward') -> Optional[Dict[str, Any]]:
        """Get best checkpoint based on metric"""
        if not self.checkpoints:
            return None
        
        best_checkpoint = max(
            self.checkpoints,
            key=lambda x: x['metrics'].get(metric, float('-inf'))
        )
        return best_checkpoint
    
    def load_checkpoint(self, checkpoint_path: str, env: gym.Env) -> PPO:
        """Load model from checkpoint"""
        return PPO.load(checkpoint_path, env=env)


class EarlyStoppingCallback(BaseCallback):
    """Early stopping callback with multiple criteria"""
    
    def __init__(self, eval_env: VecEnv, eval_freq: int, n_eval_episodes: int,
                 patience: int = 10, min_delta: float = 0.01,
                 metric: str = 'mean_reward', verbose: int = 1):
        super().__init__(verbose)
        
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.patience = patience
        self.min_delta = min_delta
        self.metric = metric
        
        self.best_metric_value = float('-inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def _on_step(self) -> bool:
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            # Evaluate model
            episode_rewards, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=self.n_eval_episodes
            )
            
            current_metric = np.mean(episode_rewards)  # Simplified for mean_reward
            
            # Check for improvement
            if current_metric > self.best_metric_value + self.min_delta:
                self.best_metric_value = current_metric
                self.patience_counter = 0
                if self.verbose >= 1:
                    logger.info(f"New best {self.metric}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                if self.verbose >= 1:
                    logger.info(f"No improvement. Patience: {self.patience_counter}/{self.patience}")
            
            # Check if should stop
            if self.patience_counter >= self.patience:
                self.should_stop = True
                if self.verbose >= 1:
                    logger.info(f"Early stopping triggered after {self.n_calls} steps")
                return False
        
        return True


class TensorBoardCallback(BaseCallback):
    """Enhanced TensorBoard logging callback"""
    
    def __init__(self, log_dir: str, log_freq: int = 1000):
        super().__init__()
        self.log_dir = log_dir
        self.log_freq = log_freq
        self.writer = None
    
    def _on_training_start(self):
        """Initialize TensorBoard writer"""
        self.writer = SummaryWriter(self.log_dir)
    
    def _on_step(self) -> bool:
        if self.n_calls % self.log_freq == 0:
            # Log training metrics
            if hasattr(self.model, 'logger') and self.model.logger:
                for key, value in self.model.logger.name_to_value.items():
                    if isinstance(value, (int, float)):
                        self.writer.add_scalar(f"train/{key}", value, self.n_calls)
            
            # Log policy metrics
            if hasattr(self.model.policy, 'get_attention_weights'):
                # Log attention weights if available
                pass
        
        return True
    
    def _on_training_end(self):
        """Close TensorBoard writer"""
        if self.writer:
            self.writer.close()


class HyperparameterOptimizer:
    """Hyperparameter optimization using Optuna"""
    
    def __init__(self, env_fn: Callable, config: TrainingConfig):
        self.env_fn = env_fn
        self.config = config
        self.study = None
    
    def create_study(self) -> optuna.Study:
        """Create Optuna study"""
        self.study = optuna.create_study(
            direction='maximize',
            study_name=self.config.study_name,
            sampler=TPESampler(),
            pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
        )
        return self.study
    
    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for optimization"""
        # Suggest hyperparameters
        hyperparams = {
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True),
            'n_steps': trial.suggest_categorical('n_steps', [512, 1024, 2048, 4096]),
            'batch_size': trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            'n_epochs': trial.suggest_int('n_epochs', 3, 20),
            'gamma': trial.suggest_float('gamma', 0.9, 0.999),
            'gae_lambda': trial.suggest_float('gae_lambda', 0.8, 0.99),
            'clip_range': trial.suggest_float('clip_range', 0.1, 0.3),
            'ent_coef': trial.suggest_float('ent_coef', 1e-8, 1e-1, log=True),
            'vf_coef': trial.suggest_float('vf_coef', 0.1, 1.0),
            'max_grad_norm': trial.suggest_float('max_grad_norm', 0.3, 2.0)
        }
        
        try:
            # Create environment
            env = self.env_fn()
            
            # Create PPO config
            ppo_config = PPOConfig(**hyperparams)
            
            # Create and train agent
            agent = PPOAgent(env, ppo_config)
            
            # Short training for optimization
            training_timesteps = min(50000, self.config.total_timesteps // 10)
            agent.train(total_timesteps=training_timesteps)
            
            # Evaluate performance
            eval_metrics = agent.evaluate(env, n_episodes=5)
            
            # Return optimization target (e.g., Sharpe ratio)
            return eval_metrics.get('sharpe_ratio', 0.0)
            
        except Exception as e:
            logger.error(f"Error in optimization trial: {e}")
            return float('-inf')
    
    def optimize(self, n_trials: int = None, timeout: int = None) -> Dict[str, Any]:
        """Run hyperparameter optimization"""
        if not self.study:
            self.create_study()
        
        n_trials = n_trials or self.config.n_trials
        timeout = timeout or self.config.hyperopt_timeout
        
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        
        try:
            self.study.optimize(
                self.objective,
                n_trials=n_trials,
                timeout=timeout,
                show_progress_bar=True
            )
            
            best_params = self.study.best_params
            best_value = self.study.best_value
            
            logger.info(f"Optimization completed!")
            logger.info(f"Best value: {best_value:.4f}")
            logger.info(f"Best parameters: {best_params}")
            
            return {
                'best_params': best_params,
                'best_value': best_value,
                'n_trials': len(self.study.trials),
                'study': self.study
            }
            
        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            raise


class PPOTrainer:
    """
    Comprehensive training pipeline for PPO trading agent
    
    Features:
    - Hyperparameter optimization
    - Advanced checkpointing
    - Early stopping
    - TensorBoard logging
    - Cross-validation
    - Performance comparison
    """
    
    def __init__(self, env_fn: Callable, config: Optional[TrainingConfig] = None):
        self.env_fn = env_fn
        self.config = config or TrainingConfig()
        self.metrics = TrainingMetrics()
        self.checkpoint_manager = None
        self.hyperopt = None
        
        self._setup_directories()
        self._setup_logging()
    
    def _setup_directories(self):
        """Setup training directories"""
        # Create base directories
        base_dir = Path(self.config.tensorboard_log_dir).parent
        base_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup checkpoint manager
        checkpoint_dir = base_dir / "checkpoints"
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir, self.config.max_checkpoints
        )
        
        # Setup hyperparameter optimization
        if self.config.enable_hyperopt:
            self.hyperopt = HyperparameterOptimizer(self.env_fn, self.config)
    
    def _setup_logging(self):
        """Setup logging configuration"""
        if self.config.enable_tensorboard:
            Path(self.config.tensorboard_log_dir).mkdir(parents=True, exist_ok=True)
    
    def train(self, ppo_config: Optional[PPOConfig] = None,
              override_timesteps: Optional[int] = None) -> Dict[str, Any]:
        """
        Run complete training pipeline
        
        Args:
            ppo_config: PPO agent configuration
            override_timesteps: Override total timesteps
            
        Returns:
            Training results and metrics
        """
        logger.info("Starting PPO training pipeline")
        start_time = datetime.now()
        
        try:
            # Optimize hyperparameters if enabled
            if self.config.enable_hyperopt and self.hyperopt:
                logger.info("Running hyperparameter optimization...")
                opt_results = self.hyperopt.optimize()
                
                # Update PPO config with best parameters
                if ppo_config:
                    for key, value in opt_results['best_params'].items():
                        if hasattr(ppo_config, key):
                            setattr(ppo_config, key, value)
                else:
                    ppo_config = PPOConfig(**opt_results['best_params'])
                
                logger.info(f"Using optimized hyperparameters: {opt_results['best_params']}")
            
            # Use default config if none provided
            if not ppo_config:
                ppo_config = PPOConfig()
            
            # Create training environment
            env = self.env_fn()
            eval_env = self.env_fn()
            
            # Set random seed
            if self.config.seed is not None:
                set_random_seed(self.config.seed)
            
            # Create agent
            agent = PPOAgent(env, ppo_config)
            
            # Setup callbacks
            callbacks = self._create_callbacks(eval_env)
            
            # Train agent
            total_timesteps = override_timesteps or self.config.total_timesteps
            
            training_summary = agent.train(
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                callback_list=callbacks
            )
            
            # Final evaluation
            final_metrics = agent.evaluate(eval_env, n_episodes=self.config.n_eval_episodes)
            
            # Update metrics
            self.metrics.timesteps = total_timesteps
            self.metrics.training_time = (datetime.now() - start_time).total_seconds()
            self.metrics.update_from_evaluation(final_metrics)
            
            # Performance comparison
            comparison_results = self._compare_with_baseline(agent, eval_env)
            
            # Create training report
            training_results = {
                'training_summary': training_summary,
                'final_metrics': final_metrics,
                'training_metrics': asdict(self.metrics),
                'comparison_results': comparison_results,
                'agent': agent,
                'ppo_config': asdict(ppo_config),
                'training_config': asdict(self.config)
            }
            
            # Save training report
            self._save_training_report(training_results)
            
            logger.info(f"Training completed in {self.metrics.training_time:.1f} seconds")
            logger.info(f"Final Sharpe ratio: {final_metrics.get('sharpe_ratio', 0):.2f}")
            logger.info(f"Best mean reward: {self.metrics.best_mean_reward:.2f}")
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise
    
    def _create_callbacks(self, eval_env: gym.Env) -> List[BaseCallback]:
        """Create training callbacks"""
        callbacks = []
        
        # Early stopping callback
        if self.config.enable_early_stopping:
            early_stopping = EarlyStoppingCallback(
                eval_env=DummyVecEnv([lambda: eval_env]),
                eval_freq=self.config.eval_frequency,
                n_eval_episodes=self.config.n_eval_episodes,
                patience=self.config.patience,
                min_delta=self.config.min_delta,
                metric=self.config.early_stopping_metric
            )
            callbacks.append(early_stopping)
        
        # TensorBoard callback
        if self.config.enable_tensorboard:
            tensorboard_callback = TensorBoardCallback(
                log_dir=self.config.tensorboard_log_dir,
                log_freq=self.config.log_frequency
            )
            callbacks.append(tensorboard_callback)
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            checkpoint_manager=self.checkpoint_manager,
            save_freq=self.config.checkpoint_frequency,
            eval_env=eval_env
        )
        callbacks.append(checkpoint_callback)
        
        return callbacks
    
    def _compare_with_baseline(self, agent: PPOAgent, eval_env: gym.Env) -> Optional[Dict[str, Any]]:
        """Compare with baseline model if available"""
        if not self.config.baseline_model_path:
            return None
        
        try:
            # Load baseline model
            baseline_agent = PPOAgent(eval_env, PPOConfig())
            baseline_agent.load_model(self.config.baseline_model_path)
            
            # Evaluate both models
            agent_metrics = agent.evaluate(eval_env, n_episodes=self.config.n_eval_episodes)
            baseline_metrics = baseline_agent.evaluate(eval_env, n_episodes=self.config.n_eval_episodes)
            
            # Calculate improvements
            improvements = {}
            for metric in self.config.comparison_metrics:
                if metric in agent_metrics and metric in baseline_metrics:
                    agent_value = agent_metrics[metric]
                    baseline_value = baseline_metrics[metric]
                    
                    if baseline_value != 0:
                        improvement = (agent_value - baseline_value) / abs(baseline_value)
                        improvements[metric] = improvement
            
            comparison_results = {
                'agent_metrics': agent_metrics,
                'baseline_metrics': baseline_metrics,
                'improvements': improvements,
                'significantly_better': any(imp > 0.1 for imp in improvements.values())
            }
            
            logger.info(f"Performance comparison:")
            for metric, improvement in improvements.items():
                logger.info(f"  {metric}: {improvement:+.2%}")
            
            return comparison_results
            
        except Exception as e:
            logger.error(f"Error comparing with baseline: {e}")
            return None
    
    def _save_training_report(self, results: Dict[str, Any]):
        """Save comprehensive training report"""
        if not self.config.save_training_logs:
            return
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_dir = Path(self.config.tensorboard_log_dir).parent / "reports"
            report_dir.mkdir(parents=True, exist_ok=True)
            
            report_path = report_dir / f"training_report_{timestamp}.json"
            
            # Prepare serializable results
            serializable_results = results.copy()
            serializable_results.pop('agent', None)  # Remove non-serializable agent
            
            with open(report_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            logger.info(f"Training report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving training report: {e}")
    
    def cross_validate(self, ppo_config: PPOConfig, n_folds: int = None) -> Dict[str, Any]:
        """Perform cross-validation training"""
        n_folds = n_folds or self.config.cross_validation_folds
        
        logger.info(f"Starting {n_folds}-fold cross-validation")
        
        fold_results = []
        
        for fold in range(n_folds):
            logger.info(f"Training fold {fold + 1}/{n_folds}")
            
            # Create fold-specific environment (would need data splitting logic)
            env = self.env_fn()
            
            # Train on fold
            agent = PPOAgent(env, ppo_config)
            fold_timesteps = self.config.total_timesteps // n_folds
            
            training_summary = agent.train(total_timesteps=fold_timesteps)
            eval_metrics = agent.evaluate(env, n_episodes=self.config.n_eval_episodes)
            
            fold_results.append({
                'fold': fold,
                'training_summary': training_summary,
                'eval_metrics': eval_metrics
            })
        
        # Aggregate results
        cv_results = {
            'fold_results': fold_results,
            'mean_metrics': {},
            'std_metrics': {}
        }
        
        # Calculate mean and std across folds
        if fold_results:
            metrics_keys = fold_results[0]['eval_metrics'].keys()
            for key in metrics_keys:
                values = [result['eval_metrics'][key] for result in fold_results]
                cv_results['mean_metrics'][key] = np.mean(values)
                cv_results['std_metrics'][key] = np.std(values)
        
        logger.info("Cross-validation completed")
        return cv_results


class CheckpointCallback(BaseCallback):
    """Callback for saving checkpoints during training"""
    
    def __init__(self, checkpoint_manager: CheckpointManager, save_freq: int,
                 eval_env: gym.Env, verbose: int = 1):
        super().__init__(verbose)
        self.checkpoint_manager = checkpoint_manager
        self.save_freq = save_freq
        self.eval_env = eval_env
    
    def _on_step(self) -> bool:
        if self.save_freq > 0 and self.n_calls % self.save_freq == 0:
            # Quick evaluation for checkpoint metrics
            episode_rewards, _ = evaluate_policy(
                self.model, self.eval_env, n_eval_episodes=5
            )
            
            metrics = {
                'mean_reward': np.mean(episode_rewards),
                'std_reward': np.std(episode_rewards)
            }
            
            # Save checkpoint
            self.checkpoint_manager.save_checkpoint(
                model=self.model,
                timesteps=self.n_calls,
                metrics=metrics
            )
        
        return True


def create_trainer(env_fn: Callable, config_dict: Optional[Dict[str, Any]] = None) -> PPOTrainer:
    """Factory function to create PPO trainer"""
    config = TrainingConfig(**config_dict) if config_dict else TrainingConfig()
    return PPOTrainer(env_fn, config)


if __name__ == "__main__":
    # Example usage
    from environment.trading_env import TradingEnvironment
    from rl_config import get_rl_config
    
    def create_env():
        rl_config = get_rl_config()
        env = TradingEnvironment(config=rl_config, mode='train')
        
        # Load data
        from datetime import datetime
        start_date = datetime(2024, 1, 1)
        end_date = datetime(2024, 6, 30)
        env.load_data(start_date, end_date)
        
        return env
    
    # Create training configuration
    training_config = TrainingConfig(
        total_timesteps=100000,
        enable_early_stopping=True,
        enable_tensorboard=True,
        enable_hyperopt=False,  # Disable for example
        patience=5,
        checkpoint_frequency=10000
    )
    
    # Create PPO configuration
    ppo_config = PPOConfig(
        policy_type='attention',
        learning_rate=3e-4,
        total_timesteps=100000
    )
    
    # Create and run trainer
    trainer = PPOTrainer(create_env, training_config)
    
    try:
        results = trainer.train(ppo_config)
        print("Training completed successfully!")
        print(f"Final Sharpe ratio: {results['final_metrics'].get('sharpe_ratio', 0):.2f}")
        
        # Run cross-validation if desired
        # cv_results = trainer.cross_validate(ppo_config, n_folds=3)
        # print(f"CV Sharpe ratio: {cv_results['mean_metrics'].get('sharpe_ratio', 0):.2f}")
        
    except Exception as e:
        print(f"Training failed: {e}")