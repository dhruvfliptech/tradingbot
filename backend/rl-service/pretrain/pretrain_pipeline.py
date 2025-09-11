"""
Pre-training Pipeline for RL Trading Agent

This module orchestrates the complete pre-training pipeline that leverages
Composer MCP's strategies to give RL agents a strong baseline through
supervised learning and transfer learning.
"""

import asyncio
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import json
import pickle
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path

from composer_extractor import ComposerExtractor, StrategyPattern
from pattern_analyzer import PatternAnalyzer, PatternAnalysis
from transfer_learning import TransferLearningManager
from validation import PretrainValidator

# Import RL components
import sys
sys.path.append('..')
from agents.policy_network import PolicyNetwork
from agents.ppo_agent import PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PretrainingConfig:
    """Configuration for pre-training pipeline"""
    # Data extraction
    max_strategies: int = 100
    min_confidence: float = 0.6
    min_performance_threshold: float = 0.1
    
    # Pattern analysis
    n_clusters: int = 8
    clustering_method: str = 'kmeans'
    
    # Training parameters
    batch_size: int = 256
    learning_rate: float = 1e-4
    num_epochs: int = 100
    validation_split: float = 0.2
    
    # Model parameters
    state_dim: int = 64
    action_dim: int = 3  # 0: hold, 1: buy, 2: sell
    hidden_dims: List[int] = None
    
    # Transfer learning
    transfer_strategy: str = 'fine_tuning'  # 'fine_tuning', 'feature_extraction', 'progressive'
    freeze_layers: List[str] = None
    
    # Validation
    validation_episodes: int = 100
    validation_envs: List[str] = None
    
    # Storage
    output_dir: str = "/tmp/rl_pretrain"
    model_save_path: str = None
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [512, 256, 128]
        if self.freeze_layers is None:
            self.freeze_layers = []
        if self.validation_envs is None:
            self.validation_envs = ['BTC-USD', 'ETH-USD', 'SPY']
        if self.model_save_path is None:
            self.model_save_path = os.path.join(self.output_dir, 'pretrained_model.pt')

class PretrainingDataset(Dataset):
    """PyTorch dataset for pre-training patterns"""
    
    def __init__(self, patterns: List[StrategyPattern], config: PretrainingConfig):
        self.patterns = patterns
        self.config = config
        
        # Prepare data
        self.states, self.actions, self.rewards, self.weights = self._prepare_data()
        
    def _prepare_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare training data from patterns"""
        states = []
        actions = []
        rewards = []
        weights = []
        
        for pattern in self.patterns:
            # Prepare state
            state = self._prepare_state(pattern)
            states.append(state)
            
            # Prepare action (convert to one-hot or keep as index)
            actions.append(pattern.action_taken)
            
            # Prepare reward
            rewards.append(pattern.reward)
            
            # Prepare weight (based on confidence and performance)
            weight = self._calculate_sample_weight(pattern)
            weights.append(weight)
        
        return (
            torch.FloatTensor(states),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(weights)
        )
    
    def _prepare_state(self, pattern: StrategyPattern) -> np.ndarray:
        """Prepare state vector from pattern"""
        # Base state features
        state_features = pattern.state_features
        
        # Pad or truncate to target dimension
        if len(state_features) < self.config.state_dim:
            padded_state = np.zeros(self.config.state_dim)
            padded_state[:len(state_features)] = state_features
            state_features = padded_state
        else:
            state_features = state_features[:self.config.state_dim]
        
        # Add contextual features
        context_features = self._extract_context_features(pattern)
        
        # Combine and normalize
        combined_state = np.concatenate([state_features, context_features])
        
        # Final padding/truncation to exact state_dim
        if len(combined_state) < self.config.state_dim:
            final_state = np.zeros(self.config.state_dim)
            final_state[:len(combined_state)] = combined_state
        else:
            final_state = combined_state[:self.config.state_dim]
        
        return final_state
    
    def _extract_context_features(self, pattern: StrategyPattern) -> np.ndarray:
        """Extract contextual features from pattern"""
        context = []
        
        # Pattern type encoding
        pattern_types = ['momentum', 'mean_reversion', 'breakout', 'trend_following']
        pattern_encoding = [1.0 if pattern.pattern_type == pt else 0.0 for pt in pattern_types]
        context.extend(pattern_encoding)
        
        # Market regime encoding
        regimes = ['bull', 'bear', 'sideways', 'volatile']
        regime_encoding = [1.0 if pattern.market_regime == regime else 0.0 for regime in regimes]
        context.extend(regime_encoding)
        
        # Confidence and performance
        context.extend([
            pattern.confidence,
            pattern.reward,
            pattern.performance_metrics.get('sharpeRatio', 0.0),
            pattern.performance_metrics.get('winRate', 0.5)
        ])
        
        return np.array(context)
    
    def _calculate_sample_weight(self, pattern: StrategyPattern) -> float:
        """Calculate importance weight for pattern"""
        # Base weight from confidence
        confidence_weight = pattern.confidence
        
        # Performance weight
        reward_weight = np.clip(pattern.reward + 1.0, 0.1, 2.0)
        
        # Strategy performance weight
        sharpe_ratio = pattern.performance_metrics.get('sharpeRatio', 0.0)
        strategy_weight = np.clip(sharpe_ratio + 1.0, 0.1, 2.0)
        
        # Combine weights
        total_weight = confidence_weight * reward_weight * strategy_weight
        
        return np.clip(total_weight, 0.1, 5.0)
    
    def __len__(self) -> int:
        return len(self.patterns)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            self.states[idx],
            self.actions[idx],
            self.rewards[idx],
            self.weights[idx]
        )

class SupervisedPretrainer:
    """Supervised pre-trainer for RL policy networks"""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize model
        self.model = self._create_model()
        self.model.to(self.device)
        
        # Loss functions
        self.policy_criterion = nn.CrossEntropyLoss(reduction='none')
        self.value_criterion = nn.MSELoss(reduction='none')
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=1e-5
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=10
        )
        
        # Training history
        self.training_history = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
    
    def _create_model(self) -> PolicyNetwork:
        """Create policy network for pre-training"""
        return PolicyNetwork(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims
        )
    
    def train(self, 
              train_dataset: PretrainingDataset,
              val_dataset: Optional[PretrainingDataset] = None) -> Dict[str, Any]:
        """Train the model with supervised learning"""
        logger.info("Starting supervised pre-training...")
        
        # Create data loaders
        train_loader = self._create_data_loader(train_dataset, shuffle=True)
        val_loader = self._create_data_loader(val_dataset, shuffle=False) if val_dataset else None
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader) if val_loader else {}
            
            # Update learning rate
            if val_metrics:
                self.scheduler.step(val_metrics['loss'])
            
            # Record history
            self._record_epoch(epoch, train_metrics, val_metrics)
            
            # Early stopping
            if val_metrics and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            
            # Logging
            if epoch % 10 == 0:
                logger.info(f"Epoch {epoch}: Train Loss: {train_metrics['loss']:.4f}, "
                          f"Train Acc: {train_metrics['accuracy']:.4f}")
                if val_metrics:
                    logger.info(f"Val Loss: {val_metrics['loss']:.4f}, "
                              f"Val Acc: {val_metrics['accuracy']:.4f}")
        
        logger.info("Pre-training completed")
        return self.training_history
    
    def _create_data_loader(self, dataset: PretrainingDataset, shuffle: bool) -> DataLoader:
        """Create data loader with weighted sampling"""
        if shuffle and dataset is not None:
            # Create weighted sampler based on pattern importance
            weights = dataset.weights.numpy()
            sampler = WeightedRandomSampler(
                weights=weights,
                num_samples=len(dataset),
                replacement=True
            )
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                sampler=sampler,
                num_workers=4
            )
        else:
            return DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                num_workers=4
            )
    
    def _train_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        for batch_idx, (states, actions, rewards, weights) in enumerate(data_loader):
            states = states.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            weights = weights.to(self.device)
            
            # Forward pass
            action_logits, value_preds = self.model(states)
            
            # Calculate losses
            policy_loss = self.policy_criterion(action_logits, actions)
            value_loss = self.value_criterion(value_preds.squeeze(), rewards)
            
            # Apply sample weights
            weighted_policy_loss = (policy_loss * weights).mean()
            weighted_value_loss = (value_loss * weights).mean()
            
            # Combined loss
            total_loss_batch = weighted_policy_loss + 0.5 * weighted_value_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Metrics
            total_loss += total_loss_batch.item()
            predicted = torch.argmax(action_logits, dim=1)
            total_correct += (predicted == actions).sum().item()
            total_samples += states.size(0)
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _validate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch"""
        self.model.eval()
        
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for states, actions, rewards, weights in data_loader:
                states = states.to(self.device)
                actions = actions.to(self.device)
                rewards = rewards.to(self.device)
                weights = weights.to(self.device)
                
                # Forward pass
                action_logits, value_preds = self.model(states)
                
                # Calculate losses
                policy_loss = self.policy_criterion(action_logits, actions)
                value_loss = self.value_criterion(value_preds.squeeze(), rewards)
                
                # Apply sample weights
                weighted_policy_loss = (policy_loss * weights).mean()
                weighted_value_loss = (value_loss * weights).mean()
                
                # Combined loss
                total_loss_batch = weighted_policy_loss + 0.5 * weighted_value_loss
                
                # Metrics
                total_loss += total_loss_batch.item()
                predicted = torch.argmax(action_logits, dim=1)
                total_correct += (predicted == actions).sum().item()
                total_samples += states.size(0)
        
        return {
            'loss': total_loss / len(data_loader),
            'accuracy': total_correct / total_samples
        }
    
    def _record_epoch(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Record epoch metrics"""
        self.training_history['epoch'].append(epoch)
        self.training_history['train_loss'].append(train_metrics['loss'])
        self.training_history['train_acc'].append(train_metrics['accuracy'])
        
        if val_metrics:
            self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['val_acc'].append(val_metrics['accuracy'])
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(self.config.output_dir, filename)
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, checkpoint_path)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")

class PretrainingPipeline:
    """Main pre-training pipeline orchestrator"""
    
    def __init__(self, config: PretrainingConfig):
        self.config = config
        
        # Initialize components
        self.extractor = ComposerExtractor()
        self.analyzer = PatternAnalyzer()
        self.transfer_manager = TransferLearningManager()
        self.validator = PretrainValidator()
        self.pretrainer = SupervisedPretrainer(config)
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    async def run_pipeline(self) -> Dict[str, Any]:
        """Run the complete pre-training pipeline"""
        logger.info("Starting RL pre-training pipeline...")
        
        pipeline_results = {
            'extraction_results': {},
            'analysis_results': {},
            'training_results': {},
            'transfer_results': {},
            'validation_results': {}
        }
        
        try:
            # Phase 1: Extract patterns from Composer strategies
            logger.info("Phase 1: Extracting patterns from Composer strategies...")
            extraction_results = await self._extract_patterns()
            pipeline_results['extraction_results'] = extraction_results
            
            # Phase 2: Analyze and categorize patterns
            logger.info("Phase 2: Analyzing patterns...")
            patterns = self.extractor.load_patterns(min_confidence=self.config.min_confidence)
            analysis_results = self._analyze_patterns(patterns)
            pipeline_results['analysis_results'] = analysis_results
            
            # Phase 3: Prepare training data
            logger.info("Phase 3: Preparing training data...")
            train_dataset, val_dataset = self._prepare_training_data(patterns, analysis_results)
            
            # Phase 4: Supervised pre-training
            logger.info("Phase 4: Supervised pre-training...")
            training_results = self.pretrainer.train(train_dataset, val_dataset)
            pipeline_results['training_results'] = training_results
            
            # Phase 5: Transfer learning to RL agent
            logger.info("Phase 5: Transfer learning...")
            transfer_results = await self._transfer_to_rl_agent()
            pipeline_results['transfer_results'] = transfer_results
            
            # Phase 6: Validation
            logger.info("Phase 6: Validation...")
            validation_results = await self._validate_pretrained_agent()
            pipeline_results['validation_results'] = validation_results
            
            # Save final results
            self._save_pipeline_results(pipeline_results)
            
            logger.info("Pre-training pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
        
        return pipeline_results
    
    async def _extract_patterns(self) -> Dict[str, Any]:
        """Extract patterns from Composer strategies"""
        async with self.extractor:
            pattern_count = await self.extractor.run_extraction_pipeline(
                max_strategies=self.config.max_strategies,
                min_performance_threshold=self.config.min_performance_threshold
            )
        
        return {
            'total_patterns_extracted': pattern_count,
            'extraction_completed': True
        }
    
    def _analyze_patterns(self, patterns: List[StrategyPattern]) -> PatternAnalysis:
        """Analyze extracted patterns"""
        analysis = self.analyzer.analyze_patterns(patterns)
        
        # Save analysis
        analysis_path = os.path.join(self.config.output_dir, 'pattern_analysis.json')
        self.analyzer.export_analysis(analysis, analysis_path)
        
        # Create visualizations
        viz_path = os.path.join(self.config.output_dir, 'pattern_analysis.png')
        self.analyzer.visualize_patterns(analysis, viz_path)
        
        return analysis
    
    def _prepare_training_data(self, 
                             patterns: List[StrategyPattern],
                             analysis: PatternAnalysis) -> Tuple[PretrainingDataset, PretrainingDataset]:
        """Prepare training and validation datasets"""
        # Filter high-quality patterns
        high_quality_patterns = [
            p for p in patterns 
            if p.confidence >= self.config.min_confidence and
               abs(p.reward) > 0.01  # Filter out near-zero rewards
        ]
        
        logger.info(f"Using {len(high_quality_patterns)} high-quality patterns for training")
        
        # Split into train/validation
        np.random.seed(42)
        indices = np.random.permutation(len(high_quality_patterns))
        split_idx = int(len(high_quality_patterns) * (1 - self.config.validation_split))
        
        train_patterns = [high_quality_patterns[i] for i in indices[:split_idx]]
        val_patterns = [high_quality_patterns[i] for i in indices[split_idx:]]
        
        # Create datasets
        train_dataset = PretrainingDataset(train_patterns, self.config)
        val_dataset = PretrainingDataset(val_patterns, self.config) if val_patterns else None
        
        logger.info(f"Training set: {len(train_patterns)} patterns")
        logger.info(f"Validation set: {len(val_patterns) if val_patterns else 0} patterns")
        
        return train_dataset, val_dataset
    
    async def _transfer_to_rl_agent(self) -> Dict[str, Any]:
        """Transfer pre-trained knowledge to RL agent"""
        # Load pre-trained model
        pretrained_model_path = os.path.join(self.config.output_dir, 'best_model.pt')
        
        if not os.path.exists(pretrained_model_path):
            logger.warning("No pre-trained model found, using random initialization")
            return {'transfer_completed': False, 'reason': 'No pre-trained model'}
        
        # Create RL agent
        rl_agent = PPOAgent(
            state_dim=self.config.state_dim,
            action_dim=self.config.action_dim,
            hidden_dims=self.config.hidden_dims
        )
        
        # Transfer knowledge
        transfer_results = self.transfer_manager.transfer_knowledge(
            source_model_path=pretrained_model_path,
            target_agent=rl_agent,
            transfer_strategy=self.config.transfer_strategy,
            freeze_layers=self.config.freeze_layers
        )
        
        # Save transferred agent
        transferred_agent_path = os.path.join(self.config.output_dir, 'transferred_rl_agent.pt')
        torch.save(rl_agent.state_dict(), transferred_agent_path)
        
        return transfer_results
    
    async def _validate_pretrained_agent(self) -> Dict[str, Any]:
        """Validate pre-trained agent performance"""
        # Load transferred agent
        transferred_agent_path = os.path.join(self.config.output_dir, 'transferred_rl_agent.pt')
        
        if not os.path.exists(transferred_agent_path):
            logger.warning("No transferred agent found for validation")
            return {'validation_completed': False, 'reason': 'No transferred agent'}
        
        # Run validation
        validation_results = await self.validator.validate_pretrained_performance(
            agent_path=transferred_agent_path,
            validation_episodes=self.config.validation_episodes,
            validation_envs=self.config.validation_envs
        )
        
        return validation_results
    
    def _save_pipeline_results(self, results: Dict[str, Any]):
        """Save complete pipeline results"""
        results_path = os.path.join(self.config.output_dir, 'pipeline_results.json')
        
        # Make results JSON serializable
        serializable_results = {}
        for key, value in results.items():
            if hasattr(value, '__dict__'):
                # Convert dataclass to dict
                serializable_results[key] = value.__dict__ if hasattr(value, '__dict__') else str(value)
            else:
                serializable_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Pipeline results saved to {results_path}")

# Example usage and main execution
async def main():
    """Example usage of the pre-training pipeline"""
    # Configuration
    config = PretrainingConfig(
        max_strategies=50,
        min_confidence=0.7,
        num_epochs=50,
        batch_size=128,
        learning_rate=1e-4,
        validation_split=0.2,
        output_dir="/tmp/rl_pretrain_demo"
    )
    
    # Create and run pipeline
    pipeline = PretrainingPipeline(config)
    results = await pipeline.run_pipeline()
    
    # Print summary
    print("Pre-training Pipeline Results:")
    print(f"Patterns extracted: {results['extraction_results'].get('total_patterns_extracted', 0)}")
    print(f"Training completed: {results['training_results'].get('epoch', [])[-1] if results['training_results'].get('epoch') else 'N/A'} epochs")
    print(f"Transfer learning: {'Success' if results['transfer_results'].get('transfer_completed', False) else 'Failed'}")
    print(f"Validation: {'Success' if results['validation_results'].get('validation_completed', False) else 'Failed'}")

if __name__ == "__main__":
    asyncio.run(main())