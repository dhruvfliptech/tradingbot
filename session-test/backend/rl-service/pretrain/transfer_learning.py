"""
Transfer Learning Manager for RL Pre-training Pipeline

This module handles the transfer of learned knowledge from supervised pre-training
to the RL agent while preserving adaptability and enabling continued learning.
"""

import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from copy import deepcopy
import os

# Import RL components
import sys
sys.path.append('..')
from agents.policy_network import PolicyNetwork
from agents.ppo_agent import PPOAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransferConfig:
    """Configuration for transfer learning"""
    strategy: str = 'fine_tuning'  # 'fine_tuning', 'feature_extraction', 'progressive'
    freeze_layers: List[str] = None
    learning_rate_scale: float = 0.1  # Scale factor for transferred layers
    adaptation_epochs: int = 10
    progressive_unfreeze_schedule: List[int] = None  # Epochs at which to unfreeze layers
    knowledge_distillation_weight: float = 0.3
    regularization_weight: float = 0.01
    
    def __post_init__(self):
        if self.freeze_layers is None:
            self.freeze_layers = []
        if self.progressive_unfreeze_schedule is None:
            self.progressive_unfreeze_schedule = [5, 10, 15]

class KnowledgeDistillationLoss(nn.Module):
    """Knowledge distillation loss for transfer learning"""
    
    def __init__(self, temperature: float = 3.0, alpha: float = 0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, 
                student_logits: torch.Tensor,
                teacher_logits: torch.Tensor,
                true_labels: torch.Tensor) -> torch.Tensor:
        """
        Compute knowledge distillation loss
        
        Args:
            student_logits: Output logits from student (RL agent)
            teacher_logits: Output logits from teacher (pretrained model)
            true_labels: True action labels
            
        Returns:
            Combined distillation loss
        """
        # Soft targets from teacher
        teacher_probs = torch.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = torch.log_softmax(student_logits / self.temperature, dim=1)
        
        # Knowledge distillation loss
        kd_loss = self.kl_div(student_log_probs, teacher_probs) * (self.temperature ** 2)
        
        # Hard target loss
        ce_loss = self.ce_loss(student_logits, true_labels)
        
        # Combine losses
        total_loss = self.alpha * kd_loss + (1 - self.alpha) * ce_loss
        
        return total_loss

class FeatureAlignmentLoss(nn.Module):
    """Feature alignment loss for matching intermediate representations"""
    
    def __init__(self, alignment_layers: List[str]):
        super().__init__()
        self.alignment_layers = alignment_layers
        self.mse_loss = nn.MSELoss()
    
    def forward(self, 
                student_features: Dict[str, torch.Tensor],
                teacher_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Compute feature alignment loss between student and teacher
        
        Args:
            student_features: Dictionary of intermediate features from student
            teacher_features: Dictionary of intermediate features from teacher
            
        Returns:
            Feature alignment loss
        """
        total_loss = 0.0
        num_layers = 0
        
        for layer_name in self.alignment_layers:
            if layer_name in student_features and layer_name in teacher_features:
                student_feat = student_features[layer_name]
                teacher_feat = teacher_features[layer_name]
                
                # Handle dimension mismatch
                if student_feat.shape != teacher_feat.shape:
                    # Use adaptive pooling or projection
                    min_dim = min(student_feat.size(-1), teacher_feat.size(-1))
                    student_feat = student_feat[..., :min_dim]
                    teacher_feat = teacher_feat[..., :min_dim]
                
                layer_loss = self.mse_loss(student_feat, teacher_feat.detach())
                total_loss += layer_loss
                num_layers += 1
        
        return total_loss / max(num_layers, 1)

class AdaptiveLearningRateScheduler:
    """Adaptive learning rate scheduler for transfer learning"""
    
    def __init__(self, 
                 base_lr: float,
                 transfer_scale: float = 0.1,
                 adaptation_schedule: Optional[Dict[str, float]] = None):
        self.base_lr = base_lr
        self.transfer_scale = transfer_scale
        self.adaptation_schedule = adaptation_schedule or {}
        
    def get_layer_lr(self, layer_name: str, is_transferred: bool) -> float:
        """Get learning rate for specific layer"""
        if is_transferred:
            # Transferred layers get reduced learning rate
            scale = self.adaptation_schedule.get(layer_name, self.transfer_scale)
            return self.base_lr * scale
        else:
            # New layers get full learning rate
            return self.base_lr

class TransferLearningManager:
    """Manages transfer learning from pretrained models to RL agents"""
    
    def __init__(self, config: Optional[TransferConfig] = None):
        self.config = config or TransferConfig()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize loss functions
        self.kd_loss = KnowledgeDistillationLoss()
        self.feature_alignment_loss = FeatureAlignmentLoss(['hidden1', 'hidden2'])
        
        # Learning rate scheduler
        self.lr_scheduler = AdaptiveLearningRateScheduler(
            base_lr=1e-4,
            transfer_scale=self.config.learning_rate_scale
        )
    
    def transfer_knowledge(self,
                          source_model_path: str,
                          target_agent: PPOAgent,
                          transfer_strategy: Optional[str] = None,
                          freeze_layers: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Transfer knowledge from pretrained model to RL agent
        
        Args:
            source_model_path: Path to pretrained model
            target_agent: RL agent to transfer knowledge to
            transfer_strategy: Transfer learning strategy
            freeze_layers: Layers to freeze during transfer
            
        Returns:
            Transfer results and statistics
        """
        logger.info("Starting knowledge transfer...")
        
        transfer_strategy = transfer_strategy or self.config.strategy
        freeze_layers = freeze_layers or self.config.freeze_layers
        
        # Load pretrained model
        source_model = self._load_pretrained_model(source_model_path)
        if source_model is None:
            return {'transfer_completed': False, 'reason': 'Failed to load source model'}
        
        # Perform transfer based on strategy
        if transfer_strategy == 'fine_tuning':
            results = self._fine_tuning_transfer(source_model, target_agent, freeze_layers)
        elif transfer_strategy == 'feature_extraction':
            results = self._feature_extraction_transfer(source_model, target_agent, freeze_layers)
        elif transfer_strategy == 'progressive':
            results = self._progressive_transfer(source_model, target_agent)
        else:
            logger.error(f"Unknown transfer strategy: {transfer_strategy}")
            return {'transfer_completed': False, 'reason': 'Unknown strategy'}
        
        logger.info("Knowledge transfer completed")
        return results
    
    def _load_pretrained_model(self, model_path: str) -> Optional[nn.Module]:
        """Load pretrained model from checkpoint"""
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Extract model state dict
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Create model architecture (assuming PolicyNetwork)
            # Note: In practice, you'd want to store architecture info in checkpoint
            model = PolicyNetwork(
                state_dim=64,  # Should be loaded from config
                action_dim=3,
                hidden_dims=[512, 256, 128]
            )
            
            model.load_state_dict(state_dict)
            model.to(self.device)
            model.eval()
            
            logger.info(f"Loaded pretrained model from {model_path}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to load pretrained model: {e}")
            return None
    
    def _fine_tuning_transfer(self, 
                             source_model: nn.Module,
                             target_agent: PPOAgent,
                             freeze_layers: List[str]) -> Dict[str, Any]:
        """Perform fine-tuning transfer learning"""
        logger.info("Performing fine-tuning transfer...")
        
        # Copy weights from source to target
        transfer_stats = self._copy_compatible_weights(source_model, target_agent.policy_net)
        
        # Freeze specified layers
        frozen_params = self._freeze_layers(target_agent.policy_net, freeze_layers)
        
        # Setup differential learning rates
        self._setup_differential_learning_rates(target_agent, transfer_stats['transferred_layers'])
        
        results = {
            'transfer_completed': True,
            'strategy': 'fine_tuning',
            'transferred_layers': transfer_stats['transferred_layers'],
            'frozen_layers': freeze_layers,
            'frozen_parameters': frozen_params,
            'total_parameters': sum(p.numel() for p in target_agent.policy_net.parameters()),
            'trainable_parameters': sum(p.numel() for p in target_agent.policy_net.parameters() if p.requires_grad)
        }
        
        logger.info(f"Fine-tuning transfer completed: {transfer_stats['layers_transferred']} layers transferred")
        return results
    
    def _feature_extraction_transfer(self,
                                   source_model: nn.Module,
                                   target_agent: PPOAgent,
                                   freeze_layers: List[str]) -> Dict[str, Any]:
        """Perform feature extraction transfer learning"""
        logger.info("Performing feature extraction transfer...")
        
        # Copy weights and freeze most layers
        transfer_stats = self._copy_compatible_weights(source_model, target_agent.policy_net)
        
        # Freeze all transferred layers for feature extraction
        transferred_layers = transfer_stats['transferred_layers']
        all_freeze_layers = list(set(freeze_layers + transferred_layers))
        
        frozen_params = self._freeze_layers(target_agent.policy_net, all_freeze_layers)
        
        # Only train the final layers
        trainable_layers = self._get_trainable_layers(target_agent.policy_net, all_freeze_layers)
        
        results = {
            'transfer_completed': True,
            'strategy': 'feature_extraction',
            'transferred_layers': transferred_layers,
            'frozen_layers': all_freeze_layers,
            'trainable_layers': trainable_layers,
            'frozen_parameters': frozen_params,
            'total_parameters': sum(p.numel() for p in target_agent.policy_net.parameters()),
            'trainable_parameters': sum(p.numel() for p in target_agent.policy_net.parameters() if p.requires_grad)
        }
        
        logger.info(f"Feature extraction transfer completed: {len(trainable_layers)} layers remain trainable")
        return results
    
    def _progressive_transfer(self,
                            source_model: nn.Module,
                            target_agent: PPOAgent) -> Dict[str, Any]:
        """Perform progressive transfer learning with gradual unfreezing"""
        logger.info("Performing progressive transfer...")
        
        # Initial transfer with all layers frozen
        transfer_stats = self._copy_compatible_weights(source_model, target_agent.policy_net)
        
        # Initially freeze all transferred layers
        all_layers = list(target_agent.policy_net.named_parameters())
        layer_names = [name for name, _ in all_layers]
        
        frozen_params = self._freeze_layers(target_agent.policy_net, layer_names)
        
        # Setup progressive unfreezing schedule
        unfreeze_schedule = self._create_progressive_schedule(layer_names)
        
        results = {
            'transfer_completed': True,
            'strategy': 'progressive',
            'transferred_layers': transfer_stats['transferred_layers'],
            'initial_frozen_layers': layer_names,
            'unfreeze_schedule': unfreeze_schedule,
            'frozen_parameters': frozen_params,
            'total_parameters': sum(p.numel() for p in target_agent.policy_net.parameters())
        }
        
        logger.info("Progressive transfer initialized")
        return results
    
    def _copy_compatible_weights(self, 
                               source_model: nn.Module,
                               target_model: nn.Module) -> Dict[str, Any]:
        """Copy compatible weights between models"""
        source_dict = source_model.state_dict()
        target_dict = target_model.state_dict()
        
        transferred_layers = []
        skipped_layers = []
        
        for name, param in source_dict.items():
            if name in target_dict:
                target_param = target_dict[name]
                
                # Check shape compatibility
                if param.shape == target_param.shape:
                    target_dict[name] = param.clone()
                    transferred_layers.append(name)
                    logger.debug(f"Transferred layer: {name} {param.shape}")
                else:
                    # Handle dimension mismatch
                    if self._can_adapt_dimensions(param, target_param):
                        adapted_param = self._adapt_parameter_dimensions(param, target_param)
                        target_dict[name] = adapted_param
                        transferred_layers.append(name)
                        logger.debug(f"Adapted and transferred layer: {name} {param.shape} -> {adapted_param.shape}")
                    else:
                        skipped_layers.append(name)
                        logger.debug(f"Skipped incompatible layer: {name} {param.shape} vs {target_param.shape}")
            else:
                skipped_layers.append(name)
                logger.debug(f"Skipped missing layer: {name}")
        
        # Load the updated state dict
        target_model.load_state_dict(target_dict)
        
        return {
            'layers_transferred': len(transferred_layers),
            'layers_skipped': len(skipped_layers),
            'transferred_layers': transferred_layers,
            'skipped_layers': skipped_layers
        }
    
    def _can_adapt_dimensions(self, source_param: torch.Tensor, target_param: torch.Tensor) -> bool:
        """Check if parameter dimensions can be adapted"""
        # Simple adaptation rules
        if source_param.dim() != target_param.dim():
            return False
        
        # Allow adaptation if one dimension can be truncated/padded
        source_shape = source_param.shape
        target_shape = target_param.shape
        
        # Check if shapes differ in only one dimension
        diff_dims = sum(1 for s, t in zip(source_shape, target_shape) if s != t)
        return diff_dims <= 1
    
    def _adapt_parameter_dimensions(self, 
                                  source_param: torch.Tensor,
                                  target_param: torch.Tensor) -> torch.Tensor:
        """Adapt parameter dimensions to match target"""
        source_shape = source_param.shape
        target_shape = target_param.shape
        
        adapted_param = source_param.clone()
        
        for dim, (source_size, target_size) in enumerate(zip(source_shape, target_shape)):
            if source_size != target_size:
                if source_size > target_size:
                    # Truncate
                    indices = torch.arange(target_size)
                    adapted_param = torch.index_select(adapted_param, dim, indices)
                else:
                    # Pad with zeros
                    pad_size = target_size - source_size
                    pad_shape = list(adapted_param.shape)
                    pad_shape[dim] = pad_size
                    
                    padding = torch.zeros(*pad_shape, dtype=adapted_param.dtype, device=adapted_param.device)
                    adapted_param = torch.cat([adapted_param, padding], dim=dim)
        
        return adapted_param
    
    def _freeze_layers(self, model: nn.Module, layer_names: List[str]) -> int:
        """Freeze specified layers in the model"""
        frozen_params = 0
        
        for name, param in model.named_parameters():
            should_freeze = any(layer_name in name for layer_name in layer_names)
            if should_freeze:
                param.requires_grad = False
                frozen_params += param.numel()
                logger.debug(f"Frozen layer: {name}")
        
        logger.info(f"Frozen {frozen_params} parameters")
        return frozen_params
    
    def _get_trainable_layers(self, model: nn.Module, frozen_layers: List[str]) -> List[str]:
        """Get list of trainable layer names"""
        trainable_layers = []
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                trainable_layers.append(name)
        
        return trainable_layers
    
    def _setup_differential_learning_rates(self, 
                                         target_agent: PPOAgent,
                                         transferred_layers: List[str]):
        """Setup differential learning rates for transferred vs new layers"""
        # Group parameters by transfer status
        transferred_params = []
        new_params = []
        
        for name, param in target_agent.policy_net.named_parameters():
            if param.requires_grad:
                if any(layer in name for layer in transferred_layers):
                    transferred_params.append(param)
                else:
                    new_params.append(param)
        
        # Create parameter groups with different learning rates
        param_groups = []
        
        if transferred_params:
            param_groups.append({
                'params': transferred_params,
                'lr': target_agent.learning_rate * self.config.learning_rate_scale,
                'name': 'transferred_layers'
            })
        
        if new_params:
            param_groups.append({
                'params': new_params,
                'lr': target_agent.learning_rate,
                'name': 'new_layers'
            })
        
        # Update optimizer with new parameter groups
        if param_groups:
            target_agent.optimizer = torch.optim.Adam(param_groups)
            logger.info(f"Setup differential learning rates: {len(param_groups)} groups")
    
    def _create_progressive_schedule(self, layer_names: List[str]) -> Dict[int, List[str]]:
        """Create progressive unfreezing schedule"""
        schedule = {}
        
        # Reverse order: unfreeze from output to input
        layers_per_epoch = max(1, len(layer_names) // len(self.config.progressive_unfreeze_schedule))
        
        for i, epoch in enumerate(self.config.progressive_unfreeze_schedule):
            start_idx = i * layers_per_epoch
            end_idx = min((i + 1) * layers_per_epoch, len(layer_names))
            
            layers_to_unfreeze = layer_names[start_idx:end_idx]
            schedule[epoch] = layers_to_unfreeze
        
        return schedule
    
    def unfreeze_layers_progressively(self, 
                                    model: nn.Module,
                                    current_epoch: int,
                                    schedule: Dict[int, List[str]]) -> List[str]:
        """Unfreeze layers according to progressive schedule"""
        unfrozen_layers = []
        
        if current_epoch in schedule:
            layers_to_unfreeze = schedule[current_epoch]
            
            for name, param in model.named_parameters():
                if any(layer in name for layer in layers_to_unfreeze):
                    param.requires_grad = True
                    unfrozen_layers.append(name)
            
            logger.info(f"Unfroze {len(unfrozen_layers)} layers at epoch {current_epoch}")
        
        return unfrozen_layers
    
    def compute_transfer_loss(self,
                            student_model: nn.Module,
                            teacher_model: nn.Module,
                            states: torch.Tensor,
                            actions: torch.Tensor,
                            use_knowledge_distillation: bool = True,
                            use_feature_alignment: bool = True) -> Dict[str, torch.Tensor]:
        """Compute transfer learning losses"""
        losses = {}
        
        # Forward pass through both models
        teacher_model.eval()
        with torch.no_grad():
            teacher_logits, teacher_values = teacher_model(states)
        
        student_logits, student_values = student_model(states)
        
        # Knowledge distillation loss
        if use_knowledge_distillation:
            kd_loss = self.kd_loss(student_logits, teacher_logits, actions)
            losses['knowledge_distillation'] = kd_loss * self.config.knowledge_distillation_weight
        
        # Feature alignment loss (if intermediate features available)
        if use_feature_alignment:
            # This would require models to return intermediate features
            # Simplified version here
            feature_loss = torch.tensor(0.0, device=states.device)
            losses['feature_alignment'] = feature_loss
        
        # Regularization loss to prevent catastrophic forgetting
        reg_loss = self._compute_regularization_loss(student_model)
        losses['regularization'] = reg_loss * self.config.regularization_weight
        
        return losses
    
    def _compute_regularization_loss(self, model: nn.Module) -> torch.Tensor:
        """Compute regularization loss to prevent catastrophic forgetting"""
        # L2 regularization on model parameters
        l2_reg = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for param in model.parameters():
            if param.requires_grad:
                l2_reg += torch.norm(param, p=2)
        
        return l2_reg
    
    def save_transfer_checkpoint(self,
                               model: nn.Module,
                               optimizer: torch.optim.Optimizer,
                               epoch: int,
                               transfer_config: TransferConfig,
                               save_path: str):
        """Save transfer learning checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'transfer_config': transfer_config,
            'timestamp': torch.tensor(torch.utils.data.get_worker_info().id if torch.utils.data.get_worker_info() else 0)
        }
        
        torch.save(checkpoint, save_path)
        logger.info(f"Transfer checkpoint saved to {save_path}")
    
    def evaluate_transfer_quality(self,
                                source_model: nn.Module,
                                target_model: nn.Module,
                                test_data: torch.utils.data.DataLoader) -> Dict[str, float]:
        """Evaluate quality of knowledge transfer"""
        source_model.eval()
        target_model.eval()
        
        metrics = {
            'agreement_rate': 0.0,
            'prediction_similarity': 0.0,
            'feature_similarity': 0.0,
            'performance_retention': 0.0
        }
        
        total_samples = 0
        total_agreement = 0
        total_similarity = 0.0
        
        with torch.no_grad():
            for states, actions, _, _ in test_data:
                states = states.to(self.device)
                actions = actions.to(self.device)
                
                # Get predictions from both models
                source_logits, _ = source_model(states)
                target_logits, _ = target_model(states)
                
                source_preds = torch.argmax(source_logits, dim=1)
                target_preds = torch.argmax(target_logits, dim=1)
                
                # Agreement rate
                agreement = (source_preds == target_preds).float().mean()
                total_agreement += agreement.item() * states.size(0)
                
                # Prediction similarity (cosine similarity)
                source_probs = torch.softmax(source_logits, dim=1)
                target_probs = torch.softmax(target_logits, dim=1)
                
                cosine_sim = torch.cosine_similarity(source_probs, target_probs, dim=1).mean()
                total_similarity += cosine_sim.item() * states.size(0)
                
                total_samples += states.size(0)
        
        metrics['agreement_rate'] = total_agreement / total_samples
        metrics['prediction_similarity'] = total_similarity / total_samples
        
        logger.info(f"Transfer quality - Agreement: {metrics['agreement_rate']:.3f}, "
                   f"Similarity: {metrics['prediction_similarity']:.3f}")
        
        return metrics

# Example usage
def main():
    """Example usage of TransferLearningManager"""
    # Create transfer manager
    config = TransferConfig(
        strategy='fine_tuning',
        freeze_layers=['layer1', 'layer2'],
        learning_rate_scale=0.1
    )
    
    transfer_manager = TransferLearningManager(config)
    
    # Create dummy RL agent
    from agents.ppo_agent import PPOAgent
    
    rl_agent = PPOAgent(
        state_dim=64,
        action_dim=3,
        hidden_dims=[512, 256, 128]
    )
    
    # Simulate transfer (would use real pretrained model path)
    pretrained_model_path = "/path/to/pretrained_model.pt"
    
    if os.path.exists(pretrained_model_path):
        results = transfer_manager.transfer_knowledge(
            source_model_path=pretrained_model_path,
            target_agent=rl_agent,
            transfer_strategy='fine_tuning'
        )
        
        print("Transfer Results:")
        print(f"Completed: {results['transfer_completed']}")
        print(f"Transferred layers: {results.get('transferred_layers', [])}")
        print(f"Frozen parameters: {results.get('frozen_parameters', 0)}")
    else:
        print("Pretrained model not found - using random initialization")

if __name__ == "__main__":
    main()