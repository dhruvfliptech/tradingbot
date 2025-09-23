"""
Custom Policy Network Architectures for Trading

This module implements custom neural network architectures specifically designed for trading:
- Attention mechanisms for feature importance
- Risk-aware policy networks
- Multi-head attention for temporal patterns
- Custom feature extractors for trading signals
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from gymnasium import spaces

logger = logging.getLogger(__name__)


@dataclass
class NetworkConfig:
    """Configuration for policy networks"""
    hidden_dims: List[int]
    attention_heads: int = 8
    attention_dim: int = 64
    dropout_rate: float = 0.1
    activation: str = 'relu'
    use_layer_norm: bool = True
    use_residual: bool = True
    risk_aversion: float = 0.1
    uncertainty_estimation: bool = True


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism for sequence processing"""
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention mechanism
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, d_model = x.size()
        
        # Generate queries, keys, values
        Q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        
        # Compute attention
        attention_output, attention_weights = self._attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        # Final linear layer
        output = self.w_o(attention_output)
        
        # Residual connection and layer norm
        output = self.layer_norm(x + self.dropout(output))
        
        return output, attention_weights
    
    def _attention(self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                   mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute scaled dot-product attention"""
        d_k = Q.size(-1)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, V)
        
        return output, attention_weights


class PositionalEncoding(nn.Module):
    """Positional encoding for sequence data"""
    
    def __init__(self, d_model: int, max_seq_length: int = 1000):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input"""
        return x + self.pe[:, :x.size(1)]


class TradingFeaturesExtractor(BaseFeaturesExtractor):
    """Custom feature extractor for trading observations"""
    
    def __init__(self, observation_space: spaces.Space, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        
        # Calculate input dimension
        if isinstance(observation_space, spaces.Box):
            input_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")
        
        # Feature extraction layers
        self.feature_net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(features_dim)
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features from observations"""
        features = self.feature_net(observations)
        return self.layer_norm(features)


class AttentionFeaturesExtractor(BaseFeaturesExtractor):
    """Attention-based feature extractor for temporal patterns"""
    
    def __init__(self, observation_space: spaces.Space, features_dim: int = 256,
                 attention_heads: int = 8, sequence_length: int = 50):
        super().__init__(observation_space, features_dim)
        
        if isinstance(observation_space, spaces.Box):
            input_dim = observation_space.shape[0]
        else:
            raise ValueError(f"Unsupported observation space: {observation_space}")
        
        self.sequence_length = sequence_length
        self.feature_dim = input_dim // sequence_length if input_dim % sequence_length == 0 else input_dim
        
        # Input projection
        self.input_projection = nn.Linear(self.feature_dim, features_dim)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(features_dim, sequence_length)
        
        # Multi-head attention
        self.attention = MultiHeadAttention(features_dim, attention_heads)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(features_dim, features_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(features_dim, features_dim)
        )
        
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract features using attention mechanism"""
        batch_size = observations.shape[0]
        
        # Reshape observations for sequence processing
        if observations.shape[1] % self.sequence_length == 0:
            seq_features = observations.view(batch_size, self.sequence_length, -1)
        else:
            # Pad or truncate to sequence length
            seq_features = observations.view(batch_size, 1, -1)
            seq_features = seq_features.repeat(1, self.sequence_length, 1)
        
        # Project to feature dimension
        projected = self.input_projection(seq_features)
        
        # Add positional encoding
        encoded = self.pos_encoding(projected)
        
        # Apply attention
        attended, attention_weights = self.attention(encoded)
        
        # Global average pooling
        pooled = torch.mean(attended, dim=1)
        
        # Output projection
        output = self.output_projection(pooled)
        
        return output


class RiskAwareNetwork(nn.Module):
    """Risk-aware neural network with uncertainty estimation"""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int],
                 risk_aversion: float = 0.1, dropout_rate: float = 0.1):
        super().__init__()
        
        self.risk_aversion = risk_aversion
        self.dropout_rate = dropout_rate
        
        # Main network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout_rate)
            ])
            prev_dim = hidden_dim
        
        self.main_net = nn.Sequential(*layers)
        
        # Output heads
        self.mean_head = nn.Linear(prev_dim, output_dim)
        self.variance_head = nn.Linear(prev_dim, output_dim)
        
        # Risk penalty network
        self.risk_net = nn.Sequential(
            nn.Linear(prev_dim, hidden_dims[-1] // 2),
            nn.ReLU(),
            nn.Linear(hidden_dims[-1] // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass with uncertainty estimation"""
        features = self.main_net(x)
        
        # Mean and variance predictions
        mean = self.mean_head(features)
        log_variance = self.variance_head(features)
        variance = torch.exp(log_variance)
        
        # Risk penalty
        risk_penalty = self.risk_net(features)
        
        # Risk-adjusted output
        risk_adjusted_mean = mean - self.risk_aversion * variance * risk_penalty
        
        return {
            'mean': mean,
            'risk_adjusted_mean': risk_adjusted_mean,
            'variance': variance,
            'risk_penalty': risk_penalty,
            'features': features
        }


class TradingPolicyNetwork(ActorCriticPolicy):
    """Custom policy network for trading with attention and risk awareness"""
    
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 lr_schedule, net_arch: Optional[List[Union[int, Dict[str, List[int]]]]] = None,
                 activation_fn: type = nn.ReLU, ortho_init: bool = True,
                 use_sde: bool = False, log_std_init: float = 0.0,
                 full_std: bool = True, sde_net_arch: Optional[List[int]] = None,
                 use_expln: bool = False, squash_output: bool = False,
                 features_extractor_class = AttentionFeaturesExtractor,
                 features_extractor_kwargs: Optional[Dict[str, Any]] = None,
                 normalize_images: bool = True, optimizer_class = torch.optim.Adam,
                 optimizer_kwargs: Optional[Dict[str, Any]] = None,
                 config: Optional[NetworkConfig] = None):
        
        self.config = config or NetworkConfig(
            hidden_dims=[256, 256],
            attention_heads=8,
            attention_dim=64
        )
        
        # Set default features extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = {
                'features_dim': 256,
                'attention_heads': self.config.attention_heads
            }
        
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            ortho_init=ortho_init,
            use_sde=use_sde,
            log_std_init=log_std_init,
            full_std=full_std,
            sde_net_arch=sde_net_arch,
            use_expln=use_expln,
            squash_output=squash_output,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            normalize_images=normalize_images,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs
        )
        
        # Store attention weights for explainability
        self.attention_weights = None
        
    def extract_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract features and store attention weights"""
        features = self.features_extractor(obs)
        
        # Store attention weights if available
        if hasattr(self.features_extractor, 'attention'):
            # This would need to be implemented in the features extractor
            pass
            
        return features
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed attention weights for explainability"""
        return self.attention_weights


class AttentionPolicy(TradingPolicyNetwork):
    """Policy network with enhanced attention mechanisms"""
    
    def __init__(self, *args, **kwargs):
        # Use attention-based feature extractor
        kwargs['features_extractor_class'] = AttentionFeaturesExtractor
        super().__init__(*args, **kwargs)


class RiskAwarePolicy(TradingPolicyNetwork):
    """Policy network with integrated risk awareness"""
    
    def __init__(self, *args, **kwargs):
        config = kwargs.get('config') or NetworkConfig(
            hidden_dims=[256, 256],
            risk_aversion=0.1,
            uncertainty_estimation=True
        )
        kwargs['config'] = config
        
        super().__init__(*args, **kwargs)
        
        # Replace value network with risk-aware version
        features_dim = self.features_extractor.features_dim
        self.risk_aware_value_net = RiskAwareNetwork(
            input_dim=features_dim,
            output_dim=1,
            hidden_dims=config.hidden_dims,
            risk_aversion=config.risk_aversion
        )
        
    def predict_values(self, obs: torch.Tensor) -> torch.Tensor:
        """Predict values with risk adjustment"""
        features = self.extract_features(obs)
        risk_outputs = self.risk_aware_value_net(features)
        return risk_outputs['risk_adjusted_mean']
    
    def get_risk_metrics(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Get risk metrics for the current observation"""
        features = self.extract_features(obs)
        return self.risk_aware_value_net(features)


class EnsemblePolicyNetwork(nn.Module):
    """Ensemble of policy networks for different market conditions"""
    
    def __init__(self, observation_space: spaces.Space, action_space: spaces.Space,
                 num_policies: int = 3, **policy_kwargs):
        super().__init__()
        
        self.num_policies = num_policies
        self.observation_space = observation_space
        self.action_space = action_space
        
        # Create multiple policy networks
        self.policies = nn.ModuleList([
            TradingPolicyNetwork(observation_space, action_space, **policy_kwargs)
            for _ in range(num_policies)
        ])
        
        # Market regime classifier
        features_dim = policy_kwargs.get('features_extractor_kwargs', {}).get('features_dim', 256)
        self.regime_classifier = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_policies),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, obs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass through ensemble"""
        # Extract features using first policy's feature extractor
        features = self.policies[0].extract_features(obs)
        
        # Classify market regime
        regime_probs = self.regime_classifier(features)
        
        # Get outputs from all policies
        policy_outputs = []
        for policy in self.policies:
            action_dist = policy.get_distribution(obs)
            value = policy.predict_values(obs)
            policy_outputs.append({
                'action_logits': action_dist.logits if hasattr(action_dist, 'logits') else None,
                'action_probs': action_dist.probs if hasattr(action_dist, 'probs') else None,
                'value': value
            })
        
        # Weighted combination based on regime probabilities
        combined_values = torch.stack([out['value'] for out in policy_outputs], dim=-1)
        final_value = torch.sum(combined_values * regime_probs.unsqueeze(1), dim=-1)
        
        return {
            'regime_probs': regime_probs,
            'policy_outputs': policy_outputs,
            'final_value': final_value,
            'features': features
        }
    
    def select_action(self, obs: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """Select action using regime-aware ensemble"""
        outputs = self.forward(obs)
        
        # Select most likely regime
        regime_idx = torch.argmax(outputs['regime_probs'], dim=-1)
        
        # Use corresponding policy
        selected_policy = self.policies[regime_idx.item()]
        action, _ = selected_policy.predict(obs, deterministic=False)
        
        return action, regime_idx.item()


def create_policy_network(network_type: str, observation_space: spaces.Space,
                         action_space: spaces.Space, **kwargs) -> TradingPolicyNetwork:
    """Factory function to create policy networks"""
    
    network_types = {
        'standard': TradingPolicyNetwork,
        'attention': AttentionPolicy,
        'risk_aware': RiskAwarePolicy
    }
    
    if network_type not in network_types:
        raise ValueError(f"Unknown network type: {network_type}")
    
    network_class = network_types[network_type]
    
    # Default learning rate schedule
    if 'lr_schedule' not in kwargs:
        kwargs['lr_schedule'] = lambda x: 3e-4
    
    return network_class(observation_space, action_space, **kwargs)


if __name__ == "__main__":
    # Example usage
    import gymnasium as gym
    
    # Create dummy environment for testing
    obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(100,), dtype=np.float32)
    action_space = spaces.Discrete(11)
    
    # Test different policy networks
    policies = [
        create_policy_network('standard', obs_space, action_space),
        create_policy_network('attention', obs_space, action_space),
        create_policy_network('risk_aware', obs_space, action_space)
    ]
    
    # Test forward pass
    obs = torch.randn(32, 100)
    
    for i, policy in enumerate(policies):
        print(f"\nTesting policy {i + 1}:")
        features = policy.extract_features(obs)
        print(f"Features shape: {features.shape}")
        
        if hasattr(policy, 'get_risk_metrics'):
            risk_metrics = policy.get_risk_metrics(obs)
            print(f"Risk metrics keys: {risk_metrics.keys()}")
    
    print("\nPolicy network testing completed!")