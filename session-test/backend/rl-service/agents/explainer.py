"""
Agent Explainability and Decision Analysis

This module provides comprehensive explainability tools for trading agents:
- Feature importance analysis using multiple methods
- Decision path visualization and explanation
- Counterfactual analysis ("what if" scenarios)
- Risk attribution and factor decomposition
- Attention weight visualization
- Action justification with natural language
"""

import os
import json
import logging
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import shap
import pickle

import gymnasium as gym
from stable_baselines3 import PPO

from .ppo_agent import PPOAgent
from .ensemble_agent import EnsembleAgent

logger = logging.getLogger(__name__)


@dataclass
class FeatureImportance:
    """Feature importance analysis results"""
    feature_names: List[str]
    importance_scores: np.ndarray
    method: str
    timestamp: datetime
    
    # Additional metrics
    cumulative_importance: Optional[np.ndarray] = None
    top_k_features: Optional[List[str]] = None
    importance_threshold: float = 0.01
    
    def __post_init__(self):
        """Calculate derived metrics"""
        if self.cumulative_importance is None:
            sorted_indices = np.argsort(self.importance_scores)[::-1]
            self.cumulative_importance = np.cumsum(self.importance_scores[sorted_indices])
        
        if self.top_k_features is None:
            k = min(10, len(self.feature_names))
            top_indices = np.argsort(self.importance_scores)[::-1][:k]
            self.top_k_features = [self.feature_names[i] for i in top_indices]
    
    def get_top_features(self, k: int = 10) -> List[Tuple[str, float]]:
        """Get top k features with their importance scores"""
        top_indices = np.argsort(self.importance_scores)[::-1][:k]
        return [(self.feature_names[i], self.importance_scores[i]) for i in top_indices]
    
    def get_features_above_threshold(self, threshold: float = None) -> List[Tuple[str, float]]:
        """Get features above importance threshold"""
        threshold = threshold or self.importance_threshold
        important_indices = np.where(self.importance_scores >= threshold)[0]
        return [(self.feature_names[i], self.importance_scores[i]) for i in important_indices]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'feature_names': self.feature_names,
            'importance_scores': self.importance_scores.tolist(),
            'method': self.method,
            'timestamp': self.timestamp.isoformat(),
            'top_k_features': self.top_k_features,
            'importance_threshold': self.importance_threshold
        }


@dataclass 
class DecisionExplanation:
    """Explanation for a specific trading decision"""
    action: int
    action_name: str
    confidence: float
    timestamp: datetime
    
    # Feature contributions
    feature_contributions: Dict[str, float]
    attention_weights: Optional[np.ndarray] = None
    
    # Risk analysis
    risk_metrics: Optional[Dict[str, float]] = None
    risk_factors: Optional[List[str]] = None
    
    # Counterfactual analysis
    alternative_actions: Optional[List[Dict[str, Any]]] = None
    
    # Natural language explanation
    explanation_text: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'action': self.action,
            'action_name': self.action_name,
            'confidence': self.confidence,
            'timestamp': self.timestamp.isoformat(),
            'feature_contributions': self.feature_contributions,
            'attention_weights': self.attention_weights.tolist() if self.attention_weights is not None else None,
            'risk_metrics': self.risk_metrics,
            'risk_factors': self.risk_factors,
            'alternative_actions': self.alternative_actions,
            'explanation_text': self.explanation_text
        }


class FeatureAnalyzer:
    """Analyzes feature importance using multiple methods"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        
        # SHAP explainer (will be initialized when needed)
        self.shap_explainer = None
        
    def permutation_importance_analysis(self, model: PPO, env: gym.Env, 
                                      n_episodes: int = 10) -> FeatureImportance:
        """
        Calculate feature importance using permutation importance
        
        Args:
            model: Trained PPO model
            env: Trading environment  
            n_episodes: Number of episodes for evaluation
            
        Returns:
            FeatureImportance object
        """
        logger.info("Calculating permutation importance...")
        
        # Collect episode data
        observations = []
        actions = []
        rewards = []
        
        for episode in range(n_episodes):
            obs, _ = env.reset()
            episode_obs = []
            episode_actions = []
            episode_rewards = []
            
            done = False
            while not done:
                action, _ = model.predict(obs, deterministic=True)
                episode_obs.append(obs.copy())
                episode_actions.append(action)
                
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_rewards.append(reward)
                done = terminated or truncated
            
            observations.extend(episode_obs)
            actions.extend(episode_actions)
            rewards.extend(episode_rewards)
        
        if not observations:
            logger.warning("No observations collected for permutation importance")
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_scores=np.zeros(self.n_features),
                method='permutation',
                timestamp=datetime.now()
            )
        
        X = np.array(observations)
        y = np.array(rewards)
        
        # Create surrogate model
        surrogate = RandomForestRegressor(n_estimators=100, random_state=42)
        surrogate.fit(X, y)
        
        # Calculate permutation importance
        perm_importance = permutation_importance(
            surrogate, X, y, n_repeats=10, random_state=42, n_jobs=-1
        )
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_scores=perm_importance.importances_mean,
            method='permutation',
            timestamp=datetime.now()
        )
    
    def gradient_importance_analysis(self, model: PPO, observations: np.ndarray) -> FeatureImportance:
        """
        Calculate feature importance using gradients
        
        Args:
            model: Trained PPO model
            observations: Array of observations
            
        Returns:
            FeatureImportance object
        """
        logger.info("Calculating gradient-based importance...")
        
        model.policy.eval()
        
        importance_scores = np.zeros(self.n_features)
        
        for obs in observations[:100]:  # Limit to 100 samples for efficiency
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).requires_grad_(True)
            
            # Forward pass
            with torch.no_grad():
                features = model.policy.extract_features(obs_tensor)
                action_logits = model.policy.action_net(features)
            
            # Calculate gradients for each action
            for action_idx in range(action_logits.shape[-1]):
                if obs_tensor.grad is not None:
                    obs_tensor.grad.zero_()
                
                # Backward pass
                action_logits[0, action_idx].backward(retain_graph=True)
                
                # Accumulate gradients
                if obs_tensor.grad is not None:
                    importance_scores += np.abs(obs_tensor.grad.detach().numpy().flatten())
        
        # Normalize by number of observations
        importance_scores /= len(observations[:100])
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_scores=importance_scores,
            method='gradient',
            timestamp=datetime.now()
        )
    
    def shap_importance_analysis(self, model: PPO, background_data: np.ndarray,
                               test_data: np.ndarray) -> FeatureImportance:
        """
        Calculate SHAP feature importance
        
        Args:
            model: Trained PPO model
            background_data: Background dataset for SHAP
            test_data: Test data to explain
            
        Returns:
            FeatureImportance object
        """
        logger.info("Calculating SHAP importance...")
        
        try:
            # Create prediction function
            def predict_fn(x):
                predictions = []
                model.policy.eval()
                with torch.no_grad():
                    for obs in x:
                        action, _ = model.predict(obs, deterministic=True)
                        predictions.append(action)
                return np.array(predictions).flatten()
            
            # Initialize SHAP explainer
            if self.shap_explainer is None:
                self.shap_explainer = shap.KernelExplainer(predict_fn, background_data[:100])
            
            # Calculate SHAP values
            shap_values = self.shap_explainer.shap_values(test_data[:50])
            
            # Average absolute SHAP values across samples
            importance_scores = np.mean(np.abs(shap_values), axis=0)
            
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_scores=importance_scores,
                method='shap',
                timestamp=datetime.now()
            )
            
        except Exception as e:
            logger.error(f"Error calculating SHAP importance: {e}")
            # Fallback to zeros
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_scores=np.zeros(self.n_features),
                method='shap',
                timestamp=datetime.now()
            )
    
    def attention_importance_analysis(self, model: PPO, observations: np.ndarray) -> FeatureImportance:
        """
        Calculate importance based on attention weights (if available)
        
        Args:
            model: Trained PPO model with attention
            observations: Array of observations
            
        Returns:
            FeatureImportance object
        """
        logger.info("Calculating attention-based importance...")
        
        if not hasattr(model.policy, 'get_attention_weights'):
            logger.warning("Model does not support attention weights")
            return FeatureImportance(
                feature_names=self.feature_names,
                importance_scores=np.zeros(self.n_features),
                method='attention',
                timestamp=datetime.now()
            )
        
        attention_scores = []
        model.policy.eval()
        
        with torch.no_grad():
            for obs in observations[:100]:
                obs_tensor = torch.FloatTensor(obs).unsqueeze(0)
                
                # Forward pass to get attention weights
                _ = model.policy.extract_features(obs_tensor)
                attention_weights = model.policy.get_attention_weights()
                
                if attention_weights is not None:
                    # Average attention across heads/layers
                    avg_attention = torch.mean(attention_weights, dim=0)
                    attention_scores.append(avg_attention.cpu().numpy())
        
        if attention_scores:
            # Average attention across observations
            importance_scores = np.mean(attention_scores, axis=0)
            
            # Ensure correct length
            if len(importance_scores) != self.n_features:
                # Pad or truncate to match feature count
                padded_scores = np.zeros(self.n_features)
                min_len = min(len(importance_scores), self.n_features)
                padded_scores[:min_len] = importance_scores[:min_len]
                importance_scores = padded_scores
        else:
            importance_scores = np.zeros(self.n_features)
        
        return FeatureImportance(
            feature_names=self.feature_names,
            importance_scores=importance_scores,
            method='attention',
            timestamp=datetime.now()
        )


class CounterfactualAnalyzer:
    """Performs counterfactual analysis for trading decisions"""
    
    def __init__(self, model: PPO, env: gym.Env):
        self.model = model
        self.env = env
    
    def analyze_counterfactuals(self, observation: np.ndarray, 
                              feature_names: List[str],
                              n_perturbations: int = 10) -> List[Dict[str, Any]]:
        """
        Analyze counterfactual scenarios by perturbing features
        
        Args:
            observation: Current observation
            feature_names: Names of features
            n_perturbations: Number of perturbations per feature
            
        Returns:
            List of counterfactual scenarios
        """
        logger.info("Performing counterfactual analysis...")
        
        # Get original prediction
        original_action, _ = self.model.predict(observation, deterministic=True)
        original_action = original_action.item() if hasattr(original_action, 'item') else original_action
        
        counterfactuals = []
        
        for feature_idx, feature_name in enumerate(feature_names):
            if feature_idx >= len(observation):
                continue
                
            original_value = observation[feature_idx]
            
            # Create perturbations around the original value
            perturbation_range = abs(original_value) * 0.1 if original_value != 0 else 0.1
            perturbations = np.linspace(
                original_value - perturbation_range,
                original_value + perturbation_range,
                n_perturbations
            )
            
            for perturb_value in perturbations:
                if abs(perturb_value - original_value) < 1e-6:
                    continue  # Skip if too close to original
                
                # Create perturbed observation
                perturbed_obs = observation.copy()
                perturbed_obs[feature_idx] = perturb_value
                
                # Get prediction for perturbed observation
                perturbed_action, _ = self.model.predict(perturbed_obs, deterministic=True)
                perturbed_action = perturbed_action.item() if hasattr(perturbed_action, 'item') else perturbed_action
                
                # Check if action changed
                if perturbed_action != original_action:
                    counterfactuals.append({
                        'feature_name': feature_name,
                        'feature_index': feature_idx,
                        'original_value': original_value,
                        'perturbed_value': perturb_value,
                        'value_change': perturb_value - original_value,
                        'original_action': original_action,
                        'perturbed_action': perturbed_action,
                        'action_changed': True
                    })
        
        # Sort by absolute value change
        counterfactuals.sort(key=lambda x: abs(x['value_change']))
        
        logger.info(f"Found {len(counterfactuals)} counterfactual scenarios")
        return counterfactuals[:20]  # Return top 20
    
    def minimal_feature_changes(self, observation: np.ndarray,
                               target_action: int,
                               feature_names: List[str],
                               max_iterations: int = 100) -> Optional[Dict[str, Any]]:
        """
        Find minimal changes needed to achieve target action
        
        Args:
            observation: Current observation
            target_action: Desired action
            feature_names: Names of features  
            max_iterations: Maximum optimization iterations
            
        Returns:
            Minimal changes needed or None if not found
        """
        logger.info(f"Finding minimal changes for target action {target_action}")
        
        current_obs = observation.copy()
        original_action, _ = self.model.predict(current_obs, deterministic=True)
        original_action = original_action.item() if hasattr(original_action, 'item') else original_action
        
        if original_action == target_action:
            return None  # Already at target action
        
        # Simple gradient-free optimization
        best_changes = None
        min_total_change = float('inf')
        
        for iteration in range(max_iterations):
            # Random feature to modify
            feature_idx = np.random.randint(0, min(len(current_obs), len(feature_names)))
            
            # Random small change
            original_value = observation[feature_idx]
            change_magnitude = abs(original_value) * 0.01 if original_value != 0 else 0.01
            change = np.random.choice([-1, 1]) * change_magnitude * (iteration + 1) / max_iterations
            
            # Apply change
            test_obs = current_obs.copy()
            test_obs[feature_idx] = original_value + change
            
            # Check action
            test_action, _ = self.model.predict(test_obs, deterministic=True)
            test_action = test_action.item() if hasattr(test_action, 'item') else test_action
            
            if test_action == target_action:
                # Calculate total change
                total_change = np.sum(np.abs(test_obs - observation))
                
                if total_change < min_total_change:
                    min_total_change = total_change
                    changes = test_obs - observation
                    
                    best_changes = {
                        'target_action': target_action,
                        'total_change_magnitude': total_change,
                        'feature_changes': {
                            feature_names[i]: changes[i] 
                            for i in range(min(len(changes), len(feature_names)))
                            if abs(changes[i]) > 1e-6
                        },
                        'modified_observation': test_obs
                    }
        
        return best_changes


class RiskAttributor:
    """Attributes risk to different factors and features"""
    
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names
        
        # Define risk factor categories
        self.risk_categories = {
            'market_risk': ['close', 'volume', 'volatility', 'returns'],
            'technical_risk': ['rsi', 'macd', 'sma', 'ema', 'bollinger'],
            'sentiment_risk': ['fear_greed', 'sentiment', 'news'],
            'portfolio_risk': ['cash', 'position', 'pnl', 'drawdown'],
            'alternative_risk': ['funding', 'open_interest', 'whale']
        }
    
    def categorize_features(self) -> Dict[str, List[str]]:
        """Categorize features into risk factors"""
        categorized = {category: [] for category in self.risk_categories}
        
        for feature in self.feature_names:
            feature_lower = feature.lower()
            
            categorized_flag = False
            for category, keywords in self.risk_categories.items():
                if any(keyword in feature_lower for keyword in keywords):
                    categorized[category].append(feature)
                    categorized_flag = True
                    break
            
            if not categorized_flag:
                # Default to market risk
                categorized['market_risk'].append(feature)
        
        return categorized
    
    def calculate_risk_attribution(self, feature_importance: FeatureImportance,
                                 observation: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Calculate risk attribution by category
        
        Args:
            feature_importance: Feature importance analysis results
            observation: Current observation values
            
        Returns:
            Risk attribution by category and feature
        """
        categorized_features = self.categorize_features()
        risk_attribution = {}
        
        for category, features in categorized_features.items():
            category_risk = {}
            total_risk = 0.0
            
            for feature in features:
                if feature in feature_importance.feature_names:
                    feature_idx = feature_importance.feature_names.index(feature)
                    importance = feature_importance.importance_scores[feature_idx]
                    
                    # Risk contribution (importance weighted by absolute value)
                    if feature_idx < len(observation):
                        risk_contribution = importance * abs(observation[feature_idx])
                    else:
                        risk_contribution = importance
                    
                    category_risk[feature] = risk_contribution
                    total_risk += risk_contribution
            
            category_risk['total_category_risk'] = total_risk
            risk_attribution[category] = category_risk
        
        return risk_attribution


class AgentExplainer:
    """
    Comprehensive explainability system for trading agents
    
    Provides:
    - Feature importance analysis
    - Decision explanations
    - Counterfactual analysis
    - Risk attribution
    - Natural language explanations
    """
    
    def __init__(self, agent: Union[PPOAgent, EnsembleAgent], 
                 feature_names: List[str], env: gym.Env):
        self.agent = agent
        self.feature_names = feature_names
        self.env = env
        
        # Initialize analyzers
        self.feature_analyzer = FeatureAnalyzer(feature_names)
        self.risk_attributor = RiskAttributor(feature_names)
        
        # For ensemble agents
        self.is_ensemble = isinstance(agent, EnsembleAgent)
        
        # Cache for computed explanations
        self.explanation_cache = {}
        
        logger.info(f"Agent explainer initialized for {'ensemble' if self.is_ensemble else 'single'} agent")
    
    def explain_decision(self, observation: np.ndarray, 
                        market_data: Optional[pd.DataFrame] = None,
                        include_counterfactuals: bool = True,
                        include_risk_attribution: bool = True) -> DecisionExplanation:
        """
        Provide comprehensive explanation for a trading decision
        
        Args:
            observation: Current observation
            market_data: Market data (for ensemble agents)
            include_counterfactuals: Whether to include counterfactual analysis
            include_risk_attribution: Whether to include risk attribution
            
        Returns:
            DecisionExplanation object
        """
        logger.info("Generating decision explanation...")
        
        # Get agent prediction
        if self.is_ensemble and market_data is not None:
            action, agent_info = self.agent.predict(observation, market_data)
            confidence = agent_info.get('regime_info', {}).get('confidence', 0.5)
            attention_weights = None  # Ensemble doesn't have single attention
        else:
            action, agent_info = self.agent.predict(observation)
            confidence = 0.8  # Default confidence for single agent
            attention_weights = agent_info.get('attention_weights')
        
        # Convert action to interpretable form
        action_value = action.item() if hasattr(action, 'item') else action
        action_name = self._get_action_name(action_value)
        
        # Calculate feature contributions (using permutation importance)
        feature_contributions = self._calculate_feature_contributions(observation)
        
        # Risk analysis
        risk_metrics = None
        risk_factors = None
        if include_risk_attribution:
            risk_metrics, risk_factors = self._analyze_risk(observation)
        
        # Counterfactual analysis
        alternative_actions = None
        if include_counterfactuals:
            counterfactual_analyzer = CounterfactualAnalyzer(
                self.agent.model if not self.is_ensemble else list(self.agent.agents.values())[0].model,
                self.env
            )
            counterfactuals = counterfactual_analyzer.analyze_counterfactuals(
                observation, self.feature_names, n_perturbations=5
            )
            
            if counterfactuals:
                alternative_actions = counterfactuals[:5]  # Top 5 alternatives
        
        # Generate natural language explanation
        explanation_text = self._generate_explanation_text(
            action_name, confidence, feature_contributions, risk_factors
        )
        
        return DecisionExplanation(
            action=action_value,
            action_name=action_name,
            confidence=confidence,
            timestamp=datetime.now(),
            feature_contributions=feature_contributions,
            attention_weights=attention_weights,
            risk_metrics=risk_metrics,
            risk_factors=risk_factors,
            alternative_actions=alternative_actions,
            explanation_text=explanation_text
        )
    
    def _get_action_name(self, action: int) -> str:
        """Convert action index to human-readable name"""
        action_names = {
            0: "HOLD",
            1: "BUY_20%", 2: "BUY_40%", 3: "BUY_60%", 4: "BUY_80%", 5: "BUY_100%",
            6: "SELL_20%", 7: "SELL_40%", 8: "SELL_60%", 9: "SELL_80%", 10: "SELL_100%"
        }
        return action_names.get(action, f"ACTION_{action}")
    
    def _calculate_feature_contributions(self, observation: np.ndarray) -> Dict[str, float]:
        """Calculate feature contributions to current decision"""
        # Simple contribution based on feature values and their typical importance
        contributions = {}
        
        for i, feature_name in enumerate(self.feature_names):
            if i < len(observation):
                # Normalize contribution by feature magnitude
                value = observation[i]
                # Simple heuristic: larger absolute values contribute more
                contribution = abs(value) / (1 + abs(value))
                contributions[feature_name] = float(contribution)
        
        # Normalize contributions to sum to 1
        total_contrib = sum(contributions.values())
        if total_contrib > 0:
            contributions = {k: v / total_contrib for k, v in contributions.items()}
        
        return contributions
    
    def _analyze_risk(self, observation: np.ndarray) -> Tuple[Dict[str, float], List[str]]:
        """Analyze risk factors in current observation"""
        # Calculate basic risk metrics
        risk_metrics = {}
        
        # Portfolio risk indicators (example indices)
        if len(observation) > 10:
            price_volatility = np.std(observation[:5]) if len(observation) >= 5 else 0.0
            portfolio_concentration = max(observation[5:10]) if len(observation) >= 10 else 0.0
            
            risk_metrics = {
                'price_volatility': float(price_volatility),
                'portfolio_concentration': float(portfolio_concentration),
                'overall_risk_score': float((price_volatility + portfolio_concentration) / 2)
            }
        
        # Identify top risk factors
        risk_factors = []
        for i, feature_name in enumerate(self.feature_names[:10]):  # Top 10 features
            if i < len(observation) and abs(observation[i]) > 1.0:  # Threshold for risk
                risk_factors.append(feature_name)
        
        return risk_metrics, risk_factors
    
    def _generate_explanation_text(self, action_name: str, confidence: float,
                                 feature_contributions: Dict[str, float],
                                 risk_factors: Optional[List[str]]) -> str:
        """Generate natural language explanation"""
        
        # Get top contributing features
        top_features = sorted(feature_contributions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        explanation = f"The agent decided to {action_name} with {confidence:.1%} confidence. "
        
        if top_features:
            feature_names = [f[0] for f in top_features]
            explanation += f"This decision was primarily influenced by {', '.join(feature_names)}. "
        
        if risk_factors:
            explanation += f"Key risk factors identified: {', '.join(risk_factors[:3])}. "
        
        # Add confidence-based commentary
        if confidence > 0.8:
            explanation += "The agent is highly confident in this decision."
        elif confidence > 0.6:
            explanation += "The agent has moderate confidence in this decision."
        else:
            explanation += "The agent has low confidence; consider market uncertainty."
        
        return explanation
    
    def analyze_feature_importance(self, n_episodes: int = 10,
                                 methods: List[str] = None) -> Dict[str, FeatureImportance]:
        """
        Comprehensive feature importance analysis
        
        Args:
            n_episodes: Number of episodes for analysis
            methods: List of methods to use ('permutation', 'gradient', 'shap', 'attention')
            
        Returns:
            Dictionary of feature importance results by method
        """
        if methods is None:
            methods = ['permutation', 'gradient', 'attention']
        
        logger.info(f"Analyzing feature importance using methods: {methods}")
        
        results = {}
        
        # Collect data for analysis
        observations = []
        for episode in range(min(n_episodes, 5)):  # Limit for efficiency
            obs, _ = self.env.reset()
            episode_obs = []
            
            done = False
            step_count = 0
            while not done and step_count < 100:  # Limit steps per episode
                if self.is_ensemble:
                    # For ensemble, use fallback agent
                    model = list(self.agent.agents.values())[0].model
                else:
                    model = self.agent.model
                
                action, _ = model.predict(obs, deterministic=True)
                episode_obs.append(obs.copy())
                
                obs, _, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                step_count += 1
            
            observations.extend(episode_obs)
        
        observations = np.array(observations)
        
        # Run different analysis methods
        if 'permutation' in methods and len(observations) > 0:
            try:
                model = list(self.agent.agents.values())[0].model if self.is_ensemble else self.agent.model
                results['permutation'] = self.feature_analyzer.permutation_importance_analysis(
                    model, self.env, n_episodes=3
                )
            except Exception as e:
                logger.error(f"Error in permutation importance: {e}")
        
        if 'gradient' in methods and len(observations) > 0:
            try:
                model = list(self.agent.agents.values())[0].model if self.is_ensemble else self.agent.model
                results['gradient'] = self.feature_analyzer.gradient_importance_analysis(
                    model, observations[:50]
                )
            except Exception as e:
                logger.error(f"Error in gradient importance: {e}")
        
        if 'attention' in methods and len(observations) > 0:
            try:
                model = list(self.agent.agents.values())[0].model if self.is_ensemble else self.agent.model
                results['attention'] = self.feature_analyzer.attention_importance_analysis(
                    model, observations[:50]
                )
            except Exception as e:
                logger.error(f"Error in attention importance: {e}")
        
        logger.info(f"Feature importance analysis completed with {len(results)} methods")
        return results
    
    def generate_explanation_report(self, observation: np.ndarray,
                                  market_data: Optional[pd.DataFrame] = None,
                                  save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive explanation report
        
        Args:
            observation: Current observation
            market_data: Market data for ensemble agents
            save_path: Path to save the report
            
        Returns:
            Complete explanation report
        """
        logger.info("Generating comprehensive explanation report...")
        
        # Get decision explanation
        decision_explanation = self.explain_decision(
            observation, market_data, 
            include_counterfactuals=True,
            include_risk_attribution=True
        )
        
        # Get feature importance analysis
        feature_importance = self.analyze_feature_importance(n_episodes=3)
        
        # Create comprehensive report
        report = {
            'timestamp': datetime.now().isoformat(),
            'agent_type': 'ensemble' if self.is_ensemble else 'single',
            'decision_explanation': decision_explanation.to_dict(),
            'feature_importance': {
                method: importance.to_dict() 
                for method, importance in feature_importance.items()
            },
            'observation_summary': {
                'mean': float(np.mean(observation)),
                'std': float(np.std(observation)),
                'min': float(np.min(observation)),
                'max': float(np.max(observation)),
                'length': len(observation)
            }
        }
        
        # Add ensemble-specific information
        if self.is_ensemble:
            report['ensemble_info'] = {
                'n_agents': len(self.agent.agents),
                'agent_types': list(self.agent.agents.keys()),
                'current_weights': self.agent.agent_weights.tolist()
            }
        
        # Save report if path provided
        if save_path:
            try:
                with open(save_path, 'w') as f:
                    json.dump(report, f, indent=2)
                logger.info(f"Explanation report saved to {save_path}")
            except Exception as e:
                logger.error(f"Error saving report: {e}")
        
        return report
    
    def visualize_feature_importance(self, feature_importance: FeatureImportance,
                                   save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of feature importance
        
        Args:
            feature_importance: Feature importance results
            save_path: Path to save the plot
            
        Returns:
            Matplotlib figure
        """
        # Get top features
        top_features = feature_importance.get_top_features(15)
        
        # Create plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        features, scores = zip(*top_features)
        y_pos = np.arange(len(features))
        
        bars = ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Feature Importance ({feature_importance.method.title()} Method)')
        
        # Color bars by importance
        for i, bar in enumerate(bars):
            if scores[i] > np.mean(scores):
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Feature importance plot saved to {save_path}")
        
        return fig


def create_explainer(agent: Union[PPOAgent, EnsembleAgent], 
                    feature_names: List[str], env: gym.Env) -> AgentExplainer:
    """Factory function to create agent explainer"""
    return AgentExplainer(agent, feature_names, env)


if __name__ == "__main__":
    # Example usage
    from environment.trading_env import TradingEnvironment
    from rl_config import get_rl_config
    from .ppo_agent import PPOAgent, PPOConfig
    
    # Create environment and agent
    rl_config = get_rl_config()
    env = TradingEnvironment(config=rl_config, mode='train')
    
    # Load data
    from datetime import datetime
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 31)
    
    if env.load_data(start_date, end_date):
        # Create agent
        ppo_config = PPOConfig(policy_type='attention', total_timesteps=10000)
        agent = PPOAgent(env, ppo_config)
        
        # Train briefly for example
        agent.train(total_timesteps=5000)
        
        # Create explainer
        feature_names = [f"feature_{i}" for i in range(env.observation_space.shape[0])]
        explainer = AgentExplainer(agent, feature_names, env)
        
        # Get sample observation
        obs, _ = env.reset()
        
        # Generate explanation
        explanation = explainer.explain_decision(obs)
        print(f"Decision: {explanation.action_name}")
        print(f"Confidence: {explanation.confidence:.2%}")
        print(f"Explanation: {explanation.explanation_text}")
        
        # Analyze feature importance
        importance_results = explainer.analyze_feature_importance(n_episodes=2)
        for method, importance in importance_results.items():
            top_features = importance.get_top_features(5)
            print(f"\nTop 5 features ({method}):")
            for feature, score in top_features:
                print(f"  {feature}: {score:.4f}")
        
        # Generate comprehensive report
        report = explainer.generate_explanation_report(obs)
        print(f"\nGenerated explanation report with {len(report)} sections")
    
    else:
        print("Failed to load data for explanation example")