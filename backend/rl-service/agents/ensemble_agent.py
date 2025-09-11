"""
Ensemble Agent for Multi-Market Conditions

This module implements an ensemble system that combines multiple specialized agents
for different market conditions:
- Bull market specialist
- Bear market specialist  
- Sideways/consolidation specialist
- High volatility specialist
- Market regime detection and automatic switching
- Dynamic weight allocation based on confidence
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
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from .ppo_agent import PPOAgent, PPOConfig
from .policy_network import TradingPolicyNetwork, AttentionPolicy, RiskAwarePolicy
from environment.market_simulator import MarketRegime, MarketConditions

logger = logging.getLogger(__name__)


@dataclass
class MarketRegimeFeatures:
    """Features for market regime classification"""
    # Price-based features
    returns_mean: float = 0.0
    returns_std: float = 0.0
    returns_skew: float = 0.0
    returns_kurt: float = 0.0
    
    # Trend features
    trend_strength: float = 0.0
    trend_consistency: float = 0.0
    sma_slope_short: float = 0.0
    sma_slope_long: float = 0.0
    
    # Volatility features
    realized_volatility: float = 0.0
    volatility_of_volatility: float = 0.0
    garch_volatility: float = 0.0
    
    # Volume features
    volume_trend: float = 0.0
    volume_volatility: float = 0.0
    price_volume_correlation: float = 0.0
    
    # Technical indicators
    rsi: float = 50.0
    macd_signal: float = 0.0
    bollinger_position: float = 0.5
    atr_normalized: float = 0.0
    
    # Market microstructure
    bid_ask_spread: float = 0.0
    order_flow_imbalance: float = 0.0
    market_impact: float = 0.0
    
    def to_array(self) -> np.ndarray:
        """Convert to numpy array for ML models"""
        return np.array([
            self.returns_mean, self.returns_std, self.returns_skew, self.returns_kurt,
            self.trend_strength, self.trend_consistency, self.sma_slope_short, self.sma_slope_long,
            self.realized_volatility, self.volatility_of_volatility, self.garch_volatility,
            self.volume_trend, self.volume_volatility, self.price_volume_correlation,
            self.rsi, self.macd_signal, self.bollinger_position, self.atr_normalized,
            self.bid_ask_spread, self.order_flow_imbalance, self.market_impact
        ])
    
    @classmethod
    def from_market_data(cls, data: pd.DataFrame, window: int = 50) -> 'MarketRegimeFeatures':
        """Extract regime features from market data"""
        if len(data) < window:
            return cls()  # Return default features
        
        recent_data = data.tail(window)
        returns = recent_data['close'].pct_change().dropna()
        
        if len(returns) < 10:
            return cls()
        
        # Calculate features
        features = cls()
        
        # Price-based features
        features.returns_mean = returns.mean()
        features.returns_std = returns.std()
        features.returns_skew = returns.skew() if len(returns) > 3 else 0.0
        features.returns_kurt = returns.kurtosis() if len(returns) > 4 else 0.0
        
        # Trend features
        prices = recent_data['close'].values
        if len(prices) >= 20:
            sma_short = pd.Series(prices).rolling(10).mean()
            sma_long = pd.Series(prices).rolling(20).mean()
            
            features.sma_slope_short = np.polyfit(range(len(sma_short.dropna())), 
                                                sma_short.dropna(), 1)[0] if len(sma_short.dropna()) > 1 else 0.0
            features.sma_slope_long = np.polyfit(range(len(sma_long.dropna())), 
                                               sma_long.dropna(), 1)[0] if len(sma_long.dropna()) > 1 else 0.0
            
            # Trend strength (correlation with linear trend)
            if len(prices) > 1:
                trend_corr = np.corrcoef(range(len(prices)), prices)[0, 1]
                features.trend_strength = abs(trend_corr) if not np.isnan(trend_corr) else 0.0
        
        # Volatility features
        features.realized_volatility = returns.std() * np.sqrt(252)  # Annualized
        
        if len(returns) >= 10:
            rolling_vol = returns.rolling(10).std()
            features.volatility_of_volatility = rolling_vol.std() if len(rolling_vol.dropna()) > 1 else 0.0
        
        # Volume features (if available)
        if 'volume' in recent_data.columns:
            volume_returns = recent_data['volume'].pct_change().dropna()
            if len(volume_returns) > 1:
                features.volume_trend = volume_returns.mean()
                features.volume_volatility = volume_returns.std()
                
                # Price-volume correlation
                if len(returns) == len(volume_returns):
                    pv_corr = np.corrcoef(returns, volume_returns)[0, 1]
                    features.price_volume_correlation = pv_corr if not np.isnan(pv_corr) else 0.0
        
        # Technical indicators (simplified calculations)
        if len(recent_data) >= 14:
            # RSI calculation
            delta = recent_data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            features.rsi = (100 - (100 / (1 + rs))).iloc[-1] if not rs.iloc[-1] == 0 else 50.0
            
            # Bollinger Bands position
            sma = recent_data['close'].rolling(20).mean()
            std = recent_data['close'].rolling(20).std()
            if not (std.iloc[-1] == 0 or np.isnan(std.iloc[-1])):
                features.bollinger_position = (recent_data['close'].iloc[-1] - sma.iloc[-1]) / (2 * std.iloc[-1]) + 0.5
            
            # ATR normalized
            high_low = recent_data['high'] - recent_data['low']
            high_close = abs(recent_data['high'] - recent_data['close'].shift())
            low_close = abs(recent_data['low'] - recent_data['close'].shift())
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            atr = tr.rolling(14).mean()
            if not (recent_data['close'].iloc[-1] == 0 or np.isnan(atr.iloc[-1])):
                features.atr_normalized = atr.iloc[-1] / recent_data['close'].iloc[-1]
        
        return features


class MarketRegimeDetector:
    """
    Sophisticated market regime detection system
    
    Uses multiple approaches:
    - Statistical features from price/volume data
    - Technical indicators
    - Machine learning classification
    - Hidden Markov Models (simplified)
    """
    
    def __init__(self, n_regimes: int = 4):
        self.n_regimes = n_regimes
        self.feature_scaler = StandardScaler()
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_fitted = False
        self.regime_names = ['bull', 'bear', 'sideways', 'high_volatility']
        
        # Regime transition probabilities (simplified HMM)
        self.transition_matrix = np.array([
            [0.7, 0.1, 0.15, 0.05],  # bull -> [bull, bear, sideways, high_vol]
            [0.1, 0.7, 0.15, 0.05],  # bear -> [bull, bear, sideways, high_vol]
            [0.2, 0.2, 0.5, 0.1],    # sideways -> [bull, bear, sideways, high_vol]
            [0.25, 0.25, 0.3, 0.2]   # high_vol -> [bull, bear, sideways, high_vol]
        ])
        
        self.current_regime = 2  # Start with sideways
        self.regime_confidence = 0.5
        self.regime_history = []
    
    def extract_features(self, data: pd.DataFrame) -> np.ndarray:
        """Extract features for regime classification"""
        features = MarketRegimeFeatures.from_market_data(data)
        return features.to_array().reshape(1, -1)
    
    def fit(self, historical_data: pd.DataFrame, regime_labels: Optional[np.ndarray] = None):
        """
        Fit the regime detector on historical data
        
        Args:
            historical_data: Historical market data
            regime_labels: Manual regime labels (if available)
        """
        try:
            if regime_labels is None:
                # Generate pseudo-labels based on market conditions
                regime_labels = self._generate_regime_labels(historical_data)
            
            # Extract features for each time window
            features_list = []
            labels_list = []
            
            window_size = 50
            for i in range(window_size, len(historical_data)):
                window_data = historical_data.iloc[i-window_size:i]
                features = self.extract_features(window_data)
                features_list.append(features.flatten())
                
                if i < len(regime_labels):
                    labels_list.append(regime_labels[i])
            
            if len(features_list) > 0:
                X = np.array(features_list)
                y = np.array(labels_list)
                
                # Fit scaler and classifier
                X_scaled = self.feature_scaler.fit_transform(X)
                self.classifier.fit(X_scaled, y)
                self.is_fitted = True
                
                logger.info(f"Regime detector fitted on {len(X)} samples")
                logger.info(f"Feature importance: {self.classifier.feature_importances_[:5]}")
            
        except Exception as e:
            logger.error(f"Error fitting regime detector: {e}")
    
    def _generate_regime_labels(self, data: pd.DataFrame) -> np.ndarray:
        """Generate regime labels based on market conditions"""
        returns = data['close'].pct_change().dropna()
        volatility = returns.rolling(20).std()
        
        labels = []
        for i in range(len(data)):
            if i < 20:
                labels.append(2)  # sideways for insufficient data
                continue
            
            recent_returns = returns.iloc[max(0, i-20):i].mean()
            recent_vol = volatility.iloc[i] if i < len(volatility) else 0.02
            
            # Simple rule-based labeling
            if recent_vol > 0.04:  # High volatility threshold
                label = 3  # high_volatility
            elif recent_returns > 0.002:  # Bull market threshold
                label = 0  # bull
            elif recent_returns < -0.002:  # Bear market threshold
                label = 1  # bear
            else:
                label = 2  # sideways
            
            labels.append(label)
        
        return np.array(labels)
    
    def predict_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Predict current market regime
        
        Returns:
            Dictionary with regime prediction and confidence
        """
        if not self.is_fitted:
            logger.warning("Regime detector not fitted, using default prediction")
            return {
                'regime': self.current_regime,
                'regime_name': self.regime_names[self.current_regime],
                'confidence': 0.5,
                'probabilities': np.array([0.25, 0.25, 0.25, 0.25])
            }
        
        try:
            # Extract features
            features = self.extract_features(data)
            features_scaled = self.feature_scaler.transform(features)
            
            # Get prediction and probabilities
            regime_pred = self.classifier.predict(features_scaled)[0]
            regime_probs = self.classifier.predict_proba(features_scaled)[0]
            
            # Apply transition smoothing (simple HMM-like approach)
            prior_probs = self.transition_matrix[self.current_regime]
            smoothed_probs = 0.7 * regime_probs + 0.3 * prior_probs
            smoothed_regime = np.argmax(smoothed_probs)
            
            # Update state
            self.current_regime = smoothed_regime
            self.regime_confidence = smoothed_probs[smoothed_regime]
            
            self.regime_history.append({
                'timestamp': datetime.now(),
                'regime': smoothed_regime,
                'confidence': self.regime_confidence
            })
            
            # Keep only recent history
            if len(self.regime_history) > 100:
                self.regime_history = self.regime_history[-100:]
            
            return {
                'regime': smoothed_regime,
                'regime_name': self.regime_names[smoothed_regime],
                'confidence': self.regime_confidence,
                'probabilities': smoothed_probs,
                'raw_probabilities': regime_probs
            }
            
        except Exception as e:
            logger.error(f"Error predicting regime: {e}")
            return {
                'regime': self.current_regime,
                'regime_name': self.regime_names[self.current_regime],
                'confidence': 0.5,
                'probabilities': np.array([0.25, 0.25, 0.25, 0.25])
            }
    
    def get_regime_stability(self, window: int = 10) -> float:
        """Calculate regime stability over recent history"""
        if len(self.regime_history) < window:
            return 0.5
        
        recent_regimes = [h['regime'] for h in self.regime_history[-window:]]
        most_common_regime = max(set(recent_regimes), key=recent_regimes.count)
        stability = recent_regimes.count(most_common_regime) / len(recent_regimes)
        
        return stability
    
    def save(self, filepath: str):
        """Save the fitted detector"""
        try:
            detector_data = {
                'feature_scaler': self.feature_scaler,
                'classifier': self.classifier,
                'is_fitted': self.is_fitted,
                'regime_names': self.regime_names,
                'transition_matrix': self.transition_matrix,
                'current_regime': self.current_regime
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(detector_data, f)
                
            logger.info(f"Regime detector saved to {filepath}")
            
        except Exception as e:
            logger.error(f"Error saving regime detector: {e}")
    
    def load(self, filepath: str):
        """Load a fitted detector"""
        try:
            with open(filepath, 'rb') as f:
                detector_data = pickle.load(f)
            
            self.feature_scaler = detector_data['feature_scaler']
            self.classifier = detector_data['classifier']
            self.is_fitted = detector_data['is_fitted']
            self.regime_names = detector_data['regime_names']
            self.transition_matrix = detector_data['transition_matrix']
            self.current_regime = detector_data['current_regime']
            
            logger.info(f"Regime detector loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Error loading regime detector: {e}")


@dataclass
class EnsembleConfig:
    """Configuration for ensemble agent"""
    # Agent specifications
    n_agents: int = 4
    agent_types: List[str] = field(default_factory=lambda: ['bull', 'bear', 'sideways', 'high_volatility'])
    
    # Ensemble behavior
    weighting_method: str = 'confidence'  # 'confidence', 'equal', 'performance'
    confidence_threshold: float = 0.7
    fallback_agent: str = 'sideways'  # Default agent when confidence is low
    
    # Performance tracking
    performance_window: int = 100
    rebalance_frequency: int = 50  # Steps between weight updates
    
    # Model paths
    models_dir: str = "/tmp/rl_models/ensemble"
    regime_detector_path: str = "/tmp/rl_models/regime_detector.pkl"
    
    # Training parameters
    train_individual_agents: bool = True
    specialization_data_split: float = 0.8  # Use 80% of relevant data for each specialist


class EnsembleAgent:
    """
    Ensemble trading agent that combines multiple specialized agents
    
    Features:
    - Multiple agents specialized for different market conditions
    - Automatic regime detection and agent selection
    - Dynamic weight allocation based on confidence and performance
    - Fallback mechanisms for uncertain conditions
    - Performance tracking and comparison
    """
    
    def __init__(self, env: gym.Env, config: Optional[EnsembleConfig] = None):
        self.env = env
        self.config = config or EnsembleConfig()
        
        # Initialize components
        self.regime_detector = MarketRegimeDetector(n_regimes=self.config.n_agents)
        self.agents: Dict[str, PPOAgent] = {}
        self.agent_weights = np.ones(self.config.n_agents) / self.config.n_agents
        self.agent_performance = {agent_type: [] for agent_type in self.config.agent_types}
        
        # State tracking
        self.current_regime = None
        self.regime_confidence = 0.5
        self.steps_since_rebalance = 0
        
        self._setup_directories()
        logger.info(f"Ensemble agent initialized with {self.config.n_agents} agents")
    
    def _setup_directories(self):
        """Setup model directories"""
        Path(self.config.models_dir).mkdir(parents=True, exist_ok=True)
    
    def create_specialized_agents(self, base_config: PPOConfig) -> Dict[str, PPOAgent]:
        """
        Create specialized agents for different market conditions
        
        Args:
            base_config: Base PPO configuration to customize for each agent
            
        Returns:
            Dictionary of specialized agents
        """
        agents = {}
        
        for agent_type in self.config.agent_types:
            # Customize config for each agent type
            specialized_config = self._customize_config_for_agent(base_config, agent_type)
            
            # Create agent
            agent = PPOAgent(self.env, specialized_config)
            agents[agent_type] = agent
            
            logger.info(f"Created {agent_type} specialist agent")
        
        self.agents = agents
        return agents
    
    def _customize_config_for_agent(self, base_config: PPOConfig, agent_type: str) -> PPOConfig:
        """Customize PPO config for specific market condition"""
        config = PPOConfig(**base_config.__dict__)
        
        # Customize based on agent type
        if agent_type == 'bull':
            # More aggressive for bull markets
            config.clip_range = 0.3
            config.ent_coef = 0.01  # Higher exploration
            config.policy_type = 'attention'
            
        elif agent_type == 'bear':
            # More conservative for bear markets
            config.clip_range = 0.1
            config.ent_coef = 0.001  # Lower exploration
            config.policy_type = 'risk_aware'
            
        elif agent_type == 'sideways':
            # Balanced for sideways markets
            config.clip_range = 0.2
            config.ent_coef = 0.005
            config.policy_type = 'standard'
            
        elif agent_type == 'high_volatility':
            # Risk-aware for high volatility
            config.clip_range = 0.15
            config.ent_coef = 0.002
            config.policy_type = 'risk_aware'
            config.gamma = 0.95  # Shorter horizon for volatile conditions
        
        # Set model name and path
        config.model_name = f"{agent_type}_specialist"
        config.save_path = str(Path(self.config.models_dir) / agent_type)
        
        return config
    
    def train_ensemble(self, historical_data: pd.DataFrame,
                      base_config: PPOConfig, total_timesteps: int = 100000) -> Dict[str, Any]:
        """
        Train the ensemble of specialized agents
        
        Args:
            historical_data: Historical market data for regime detection
            base_config: Base configuration for agents
            total_timesteps: Training timesteps per agent
            
        Returns:
            Training results for each agent
        """
        logger.info("Starting ensemble training")
        
        # Step 1: Fit regime detector
        logger.info("Fitting regime detector on historical data")
        self.regime_detector.fit(historical_data)
        
        # Save regime detector
        self.regime_detector.save(self.config.regime_detector_path)
        
        # Step 2: Create specialized agents
        if not self.agents or self.config.train_individual_agents:
            self.create_specialized_agents(base_config)
        
        # Step 3: Train each specialist (could be done in parallel)
        training_results = {}
        
        for agent_type, agent in self.agents.items():
            logger.info(f"Training {agent_type} specialist agent")
            
            try:
                # Train the agent
                training_summary = agent.train(total_timesteps=total_timesteps)
                
                # Evaluate performance
                eval_metrics = agent.evaluate(self.env, n_episodes=10)
                
                training_results[agent_type] = {
                    'training_summary': training_summary,
                    'eval_metrics': eval_metrics
                }
                
                # Save the trained agent
                agent_path = Path(self.config.models_dir) / agent_type / f"{agent_type}_final.zip"
                agent.model.save(agent_path)
                
                logger.info(f"{agent_type} agent training completed")
                logger.info(f"  Sharpe ratio: {eval_metrics.get('sharpe_ratio', 0):.2f}")
                logger.info(f"  Total return: {eval_metrics.get('mean_reward', 0):.2f}")
                
            except Exception as e:
                logger.error(f"Error training {agent_type} agent: {e}")
                training_results[agent_type] = {'error': str(e)}
        
        logger.info("Ensemble training completed")
        return training_results
    
    def predict(self, observation: np.ndarray, market_data: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Make ensemble prediction
        
        Args:
            observation: Environment observation
            market_data: Recent market data for regime detection
            
        Returns:
            Tuple of (action, additional_info)
        """
        try:
            # Step 1: Detect current market regime
            regime_info = self.regime_detector.predict_regime(market_data)
            self.current_regime = regime_info['regime']
            self.regime_confidence = regime_info['confidence']
            
            # Step 2: Update agent weights
            self._update_agent_weights(regime_info)
            
            # Step 3: Get predictions from all agents
            agent_predictions = {}
            agent_infos = {}
            
            for agent_type, agent in self.agents.items():
                if agent.model is not None:
                    action, info = agent.predict(observation, deterministic=True)
                    agent_predictions[agent_type] = action
                    agent_infos[agent_type] = info
            
            # Step 4: Combine predictions based on weights
            final_action = self._combine_predictions(agent_predictions)
            
            # Step 5: Create comprehensive info
            ensemble_info = {
                'regime_info': regime_info,
                'agent_weights': self.agent_weights.tolist(),
                'agent_predictions': {k: v.tolist() if hasattr(v, 'tolist') else v 
                                    for k, v in agent_predictions.items()},
                'final_action': final_action.tolist() if hasattr(final_action, 'tolist') else final_action,
                'agent_infos': agent_infos,
                'regime_stability': self.regime_detector.get_regime_stability()
            }
            
            return final_action, ensemble_info
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            # Fallback to default agent
            if self.config.fallback_agent in self.agents:
                return self.agents[self.config.fallback_agent].predict(observation)
            else:
                # Return hold action as ultimate fallback
                return np.array([0]), {'error': str(e)}
    
    def _update_agent_weights(self, regime_info: Dict[str, Any]):
        """Update agent weights based on regime and performance"""
        self.steps_since_rebalance += 1
        
        if self.steps_since_rebalance < self.config.rebalance_frequency:
            return  # Don't rebalance yet
        
        self.steps_since_rebalance = 0
        
        if self.config.weighting_method == 'confidence':
            # Weight based on regime confidence
            regime_probs = regime_info['probabilities']
            self.agent_weights = regime_probs
            
        elif self.config.weighting_method == 'performance':
            # Weight based on recent performance
            weights = []
            for agent_type in self.config.agent_types:
                if agent_type in self.agent_performance and self.agent_performance[agent_type]:
                    # Use recent average performance
                    recent_perf = np.mean(self.agent_performance[agent_type][-10:])
                    weights.append(max(0.1, recent_perf))  # Minimum weight 0.1
                else:
                    weights.append(0.25)  # Default weight
            
            # Normalize weights
            weights = np.array(weights)
            self.agent_weights = weights / weights.sum()
            
        else:  # equal weighting
            self.agent_weights = np.ones(self.config.n_agents) / self.config.n_agents
        
        # Apply confidence threshold
        if self.regime_confidence < self.config.confidence_threshold:
            # Increase weight for fallback agent
            fallback_idx = self.config.agent_types.index(self.config.fallback_agent)
            self.agent_weights = 0.5 * self.agent_weights
            self.agent_weights[fallback_idx] += 0.5
            self.agent_weights = self.agent_weights / self.agent_weights.sum()
    
    def _combine_predictions(self, agent_predictions: Dict[str, np.ndarray]) -> np.ndarray:
        """Combine agent predictions using weights"""
        if not agent_predictions:
            return np.array([0])  # Hold action
        
        # Handle case where predictions are scalar or arrays
        predictions_array = []
        for i, agent_type in enumerate(self.config.agent_types):
            if agent_type in agent_predictions:
                pred = agent_predictions[agent_type]
                if isinstance(pred, np.ndarray):
                    pred = pred.item() if pred.size == 1 else pred[0]
                predictions_array.append(pred)
            else:
                predictions_array.append(0)  # Default to hold
        
        predictions_array = np.array(predictions_array)
        
        # Weighted average (for continuous actions) or weighted voting (for discrete)
        if self.env.action_space.__class__.__name__ == 'Discrete':
            # Discrete actions: weighted voting
            action_votes = np.zeros(self.env.action_space.n)
            for i, pred in enumerate(predictions_array):
                if 0 <= pred < self.env.action_space.n:
                    action_votes[int(pred)] += self.agent_weights[i]
            
            final_action = np.argmax(action_votes)
        else:
            # Continuous actions: weighted average
            final_action = np.average(predictions_array, weights=self.agent_weights)
        
        return np.array([final_action])
    
    def evaluate_ensemble(self, eval_env: gym.Env, n_episodes: int = 10,
                         market_data: pd.DataFrame = None) -> Dict[str, Any]:
        """
        Evaluate ensemble performance
        
        Args:
            eval_env: Evaluation environment
            n_episodes: Number of episodes to evaluate
            market_data: Market data for regime detection
            
        Returns:
            Evaluation metrics
        """
        if not self.agents:
            raise ValueError("No agents loaded in ensemble")
        
        logger.info(f"Evaluating ensemble over {n_episodes} episodes")
        
        # Evaluate ensemble
        episode_rewards = []
        episode_lengths = []
        regime_switches = []
        
        for episode in range(n_episodes):
            obs, _ = eval_env.reset()
            episode_reward = 0
            episode_length = 0
            episode_regimes = []
            
            done = False
            while not done:
                # Get ensemble prediction
                if market_data is not None:
                    action, info = self.predict(obs, market_data)
                    episode_regimes.append(info['regime_info']['regime'])
                else:
                    # Use fallback agent if no market data
                    fallback_agent = self.agents[self.config.fallback_agent]
                    action, _ = fallback_agent.predict(obs)
                
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                episode_reward += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            # Track regime switches
            if episode_regimes:
                switches = sum(1 for i in range(1, len(episode_regimes)) 
                             if episode_regimes[i] != episode_regimes[i-1])
                regime_switches.append(switches)
        
        # Calculate metrics
        metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths),
            'total_episodes': len(episode_rewards),
            'regime_switches_per_episode': np.mean(regime_switches) if regime_switches else 0
        }
        
        # Calculate Sharpe ratio
        if metrics['std_reward'] > 0:
            metrics['sharpe_ratio'] = metrics['mean_reward'] / metrics['std_reward']
        else:
            metrics['sharpe_ratio'] = 0.0
        
        # Individual agent evaluation for comparison
        individual_metrics = {}
        for agent_type, agent in self.agents.items():
            if agent.model is not None:
                try:
                    agent_metrics = agent.evaluate(eval_env, n_episodes=5)
                    individual_metrics[agent_type] = agent_metrics
                except Exception as e:
                    logger.warning(f"Failed to evaluate {agent_type} agent: {e}")
        
        metrics['individual_agents'] = individual_metrics
        metrics['agent_weights'] = self.agent_weights.tolist()
        
        logger.info(f"Ensemble evaluation completed:")
        logger.info(f"  Mean reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
        logger.info(f"  Sharpe ratio: {metrics['sharpe_ratio']:.2f}")
        logger.info(f"  Regime switches per episode: {metrics['regime_switches_per_episode']:.1f}")
        
        return metrics
    
    def load_ensemble(self, models_dir: str = None):
        """Load pre-trained ensemble"""
        models_dir = models_dir or self.config.models_dir
        
        logger.info(f"Loading ensemble from {models_dir}")
        
        # Load regime detector
        if Path(self.config.regime_detector_path).exists():
            self.regime_detector.load(self.config.regime_detector_path)
        else:
            logger.warning("Regime detector not found, using untrained detector")
        
        # Load individual agents
        for agent_type in self.config.agent_types:
            agent_dir = Path(models_dir) / agent_type
            model_files = list(agent_dir.glob("*.zip"))
            
            if model_files:
                # Load the most recent model
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                
                if agent_type not in self.agents:
                    # Create agent with default config
                    base_config = PPOConfig()
                    specialized_config = self._customize_config_for_agent(base_config, agent_type)
                    self.agents[agent_type] = PPOAgent(self.env, specialized_config)
                
                # Load the model
                self.agents[agent_type].load_model(str(latest_model))
                logger.info(f"Loaded {agent_type} agent from {latest_model}")
            else:
                logger.warning(f"No model found for {agent_type} agent")
        
        logger.info(f"Ensemble loaded with {len(self.agents)} agents")
    
    def save_ensemble(self, save_dir: str = None):
        """Save the entire ensemble"""
        save_dir = save_dir or self.config.models_dir
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Save regime detector
        self.regime_detector.save(self.config.regime_detector_path)
        
        # Save ensemble configuration
        config_path = Path(save_dir) / "ensemble_config.json"
        with open(config_path, 'w') as f:
            # Convert dataclass to dict for JSON serialization
            config_dict = {
                'n_agents': self.config.n_agents,
                'agent_types': self.config.agent_types,
                'weighting_method': self.config.weighting_method,
                'confidence_threshold': self.config.confidence_threshold,
                'fallback_agent': self.config.fallback_agent
            }
            json.dump(config_dict, f, indent=2)
        
        # Individual agent models are saved during training
        logger.info(f"Ensemble configuration saved to {save_dir}")


def create_ensemble_agent(env: gym.Env, config_dict: Optional[Dict[str, Any]] = None) -> EnsembleAgent:
    """Factory function to create ensemble agent"""
    config = EnsembleConfig(**config_dict) if config_dict else EnsembleConfig()
    return EnsembleAgent(env, config)


if __name__ == "__main__":
    # Example usage
    from environment.trading_env import TradingEnvironment
    from rl_config import get_rl_config
    import pandas as pd
    
    # Create environment
    rl_config = get_rl_config()
    env = TradingEnvironment(config=rl_config, mode='train')
    
    # Load data
    from datetime import datetime
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 6, 30)
    
    if env.load_data(start_date, end_date):
        # Create dummy historical data
        dates = pd.date_range(start_date, end_date, freq='1H')
        historical_data = pd.DataFrame({
            'timestamp': dates,
            'close': 50000 + np.cumsum(np.random.randn(len(dates)) * 100),
            'high': 0,
            'low': 0,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        historical_data['high'] = historical_data['close'] * 1.01
        historical_data['low'] = historical_data['close'] * 0.99
        
        # Create ensemble agent
        ensemble_config = EnsembleConfig(
            n_agents=4,
            train_individual_agents=True
        )
        
        ensemble = EnsembleAgent(env, ensemble_config)
        
        # Create base PPO config
        base_config = PPOConfig(
            policy_type='attention',
            total_timesteps=20000  # Reduced for example
        )
        
        # Train ensemble
        print("Training ensemble agents...")
        training_results = ensemble.train_ensemble(historical_data, base_config, total_timesteps=20000)
        
        print("Training completed!")
        for agent_type, results in training_results.items():
            if 'eval_metrics' in results:
                sharpe = results['eval_metrics'].get('sharpe_ratio', 0)
                print(f"  {agent_type}: Sharpe ratio = {sharpe:.2f}")
        
        # Evaluate ensemble
        print("\nEvaluating ensemble...")
        ensemble_metrics = ensemble.evaluate_ensemble(env, n_episodes=3, market_data=historical_data)
        print(f"Ensemble Sharpe ratio: {ensemble_metrics['sharpe_ratio']:.2f}")
        
    else:
        print("Failed to load data for training")