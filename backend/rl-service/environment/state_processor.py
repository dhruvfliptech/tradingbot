"""
State Processor for RL Trading Environment
Handles feature engineering, normalization, and observation space preparation
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
import ta
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


@dataclass
class MarketState:
    """Container for current market state data"""
    timestamp: pd.Timestamp
    price_data: Dict[str, float]  # OHLCV data
    technical_indicators: Dict[str, float]
    sentiment_data: Dict[str, float]
    portfolio_state: Dict[str, float]
    alternative_data: Dict[str, float]
    raw_features: np.ndarray
    normalized_features: np.ndarray


class FeatureEngineer(ABC):
    """Abstract base class for feature engineering components"""
    
    @abstractmethod
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute features from raw data"""
        pass
    
    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        pass


class TechnicalIndicatorEngine(FeatureEngineer):
    """Computes technical indicators for price data"""
    
    def __init__(self, indicators_config: Dict[str, Any] = None):
        self.config = indicators_config or {}
        self.feature_names = []
        self._setup_indicators()
    
    def _setup_indicators(self):
        """Setup indicator configurations"""
        self.indicators = {
            # Trend indicators
            'sma_20': {'period': 20, 'func': ta.trend.sma_indicator},
            'sma_50': {'period': 50, 'func': ta.trend.sma_indicator},
            'sma_200': {'period': 200, 'func': ta.trend.sma_indicator},
            'ema_12': {'period': 12, 'func': ta.trend.ema_indicator},
            'ema_26': {'period': 26, 'func': ta.trend.ema_indicator},
            
            # Momentum indicators
            'rsi_14': {'period': 14, 'func': ta.momentum.rsi},
            'rsi_30': {'period': 30, 'func': ta.momentum.rsi},
            'stoch_k': {'func': ta.momentum.stoch},
            'stoch_d': {'func': ta.momentum.stoch_signal},
            'williams_r': {'period': 14, 'func': ta.momentum.williams_r},
            
            # Volume indicators
            'volume_sma': {'period': 20, 'func': ta.volume.volume_sma},
            'vwap': {'func': ta.volume.volume_weighted_average_price},
            'mfi': {'period': 14, 'func': ta.volume.money_flow_index},
            
            # Volatility indicators
            'bb_upper': {'func': ta.volatility.bollinger_hband},
            'bb_middle': {'func': ta.volatility.bollinger_mavg},
            'bb_lower': {'func': ta.volatility.bollinger_lband},
            'bb_width': {'func': ta.volatility.bollinger_wband},
            'atr': {'period': 14, 'func': ta.volatility.average_true_range},
            'kc_upper': {'func': ta.volatility.keltner_channel_hband},
            'kc_lower': {'func': ta.volatility.keltner_channel_lband},
            
            # Trend strength
            'adx': {'period': 14, 'func': ta.trend.adx},
            'cci': {'period': 20, 'func': ta.trend.cci},
            'dpo': {'period': 20, 'func': ta.trend.dpo},
        }
        
        self.feature_names = list(self.indicators.keys()) + [
            'macd', 'macd_signal', 'macd_histogram',
            'bb_percent', 'price_vs_sma20', 'price_vs_sma50',
            'rsi_divergence', 'volume_ratio'
        ]
    
    def compute_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Compute technical indicators from OHLCV data"""
        if len(data) < 200:  # Need sufficient history for indicators
            logger.warning(f"Insufficient data for technical indicators: {len(data)} rows")
            return pd.DataFrame()
        
        try:
            features = pd.DataFrame(index=data.index)
            
            # Basic indicators
            for name, config in self.indicators.items():
                try:
                    func = config['func']
                    if 'period' in config:
                        if name.startswith('sma'):
                            features[name] = func(data['close'], window=config['period'])
                        elif name.startswith('ema'):
                            features[name] = func(data['close'], window=config['period'])
                        elif name in ['rsi_14', 'rsi_30']:
                            features[name] = func(data['close'], window=config['period'])
                        elif name == 'atr':
                            features[name] = func(data['high'], data['low'], data['close'], window=config['period'])
                        elif name == 'adx':
                            features[name] = func(data['high'], data['low'], data['close'], window=config['period'])
                        elif name == 'williams_r':
                            features[name] = func(data['high'], data['low'], data['close'], lbp=config['period'])
                        elif name == 'mfi':
                            features[name] = func(data['high'], data['low'], data['close'], data['volume'], window=config['period'])
                        elif name == 'cci':
                            features[name] = func(data['high'], data['low'], data['close'], window=config['period'])
                        elif name == 'dpo':
                            features[name] = func(data['close'], window=config['period'])
                        elif name == 'volume_sma':
                            features[name] = func(data['volume'], window=config['period'])
                    else:
                        if name == 'stoch_k':
                            features[name] = func(data['high'], data['low'], data['close'])
                        elif name == 'stoch_d':
                            features[name] = func(data['high'], data['low'], data['close'])
                        elif name == 'vwap':
                            features[name] = func(data['high'], data['low'], data['close'], data['volume'])
                        elif name.startswith('bb_'):
                            if name == 'bb_upper':
                                features[name] = func(data['close'])
                            elif name == 'bb_middle':
                                features[name] = func(data['close'])
                            elif name == 'bb_lower':
                                features[name] = func(data['close'])
                            elif name == 'bb_width':
                                features[name] = func(data['close'])
                        elif name.startswith('kc_'):
                            if name == 'kc_upper':
                                features[name] = func(data['high'], data['low'], data['close'])
                            elif name == 'kc_lower':
                                features[name] = func(data['high'], data['low'], data['close'])
                except Exception as e:
                    logger.warning(f"Failed to compute {name}: {e}")
                    features[name] = np.nan
            
            # MACD components
            try:
                macd_line, macd_signal, macd_histogram = ta.trend.MACD(data['close']).macd(), \
                                                        ta.trend.MACD(data['close']).macd_signal(), \
                                                        ta.trend.MACD(data['close']).macd_diff()
                features['macd'] = macd_line
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_histogram
            except Exception as e:
                logger.warning(f"Failed to compute MACD: {e}")
                features['macd'] = np.nan
                features['macd_signal'] = np.nan
                features['macd_histogram'] = np.nan
            
            # Custom derived features
            try:
                # Bollinger Band percentage
                if 'bb_upper' in features and 'bb_lower' in features:
                    bb_range = features['bb_upper'] - features['bb_lower']
                    features['bb_percent'] = (data['close'] - features['bb_lower']) / bb_range
                
                # Price vs moving averages
                if 'sma_20' in features:
                    features['price_vs_sma20'] = (data['close'] - features['sma_20']) / features['sma_20']
                if 'sma_50' in features:
                    features['price_vs_sma50'] = (data['close'] - features['sma_50']) / features['sma_50']
                
                # RSI divergence (simplified)
                if 'rsi_14' in features:
                    features['rsi_divergence'] = features['rsi_14'].diff()
                
                # Volume ratio
                if 'volume_sma' in features:
                    features['volume_ratio'] = data['volume'] / features['volume_sma']
                
            except Exception as e:
                logger.warning(f"Failed to compute derived features: {e}")
            
            return features
            
        except Exception as e:
            logger.error(f"Error computing technical indicators: {e}")
            return pd.DataFrame()
    
    def get_feature_names(self) -> List[str]:
        """Get list of technical indicator feature names"""
        return self.feature_names


class SentimentFeatureEngine(FeatureEngineer):
    """Processes sentiment and market psychology features"""
    
    def __init__(self):
        self.feature_names = [
            'fear_greed_index', 'sentiment_score', 'news_sentiment',
            'social_sentiment', 'market_cap_rank', 'sentiment_momentum',
            'fear_greed_ma', 'sentiment_volatility'
        ]
    
    def compute_features(self, sentiment_data: Dict[str, Any]) -> pd.DataFrame:
        """Compute sentiment features from various sources"""
        try:
            features = {}
            
            # Fear & Greed Index
            features['fear_greed_index'] = sentiment_data.get('fear_greed_index', 50)
            
            # Groq sentiment analysis
            features['sentiment_score'] = sentiment_data.get('sentiment_score', 0.0)
            features['news_sentiment'] = sentiment_data.get('news_sentiment', 0.0)
            features['social_sentiment'] = sentiment_data.get('social_sentiment', 0.0)
            
            # Market data
            features['market_cap_rank'] = sentiment_data.get('market_cap_rank', 1)
            
            # Derived features
            features['sentiment_momentum'] = sentiment_data.get('sentiment_momentum', 0.0)
            features['fear_greed_ma'] = sentiment_data.get('fear_greed_ma', 50)
            features['sentiment_volatility'] = sentiment_data.get('sentiment_volatility', 0.0)
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error computing sentiment features: {e}")
            return pd.DataFrame()
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class AlternativeDataEngine(FeatureEngineer):
    """Processes alternative data features (on-chain, derivatives, etc.)"""
    
    def __init__(self):
        self.feature_names = [
            'funding_rate', 'open_interest', 'long_short_ratio',
            'whale_activity', 'exchange_inflows', 'exchange_outflows',
            'network_activity', 'active_addresses', 'transaction_volume',
            'nvt_ratio', 'mvrv_ratio', 'realized_cap'
        ]
    
    def compute_features(self, alt_data: Dict[str, Any]) -> pd.DataFrame:
        """Compute alternative data features"""
        try:
            features = {}
            
            # Derivatives data
            features['funding_rate'] = alt_data.get('funding_rate', 0.0)
            features['open_interest'] = alt_data.get('open_interest', 0.0)
            features['long_short_ratio'] = alt_data.get('long_short_ratio', 1.0)
            
            # Whale and exchange data
            features['whale_activity'] = alt_data.get('whale_activity', 0.0)
            features['exchange_inflows'] = alt_data.get('exchange_inflows', 0.0)
            features['exchange_outflows'] = alt_data.get('exchange_outflows', 0.0)
            
            # On-chain metrics
            features['network_activity'] = alt_data.get('network_activity', 0.0)
            features['active_addresses'] = alt_data.get('active_addresses', 0.0)
            features['transaction_volume'] = alt_data.get('transaction_volume', 0.0)
            
            # Valuation metrics
            features['nvt_ratio'] = alt_data.get('nvt_ratio', 0.0)
            features['mvrv_ratio'] = alt_data.get('mvrv_ratio', 1.0)
            features['realized_cap'] = alt_data.get('realized_cap', 0.0)
            
            return pd.DataFrame([features])
            
        except Exception as e:
            logger.error(f"Error computing alternative data features: {e}")
            return pd.DataFrame()
    
    def get_feature_names(self) -> List[str]:
        return self.feature_names


class StateProcessor:
    """Main state processor for RL trading environment"""
    
    def __init__(self, 
                 lookback_window: int = 50,
                 normalization_method: str = 'robust',
                 feature_selection: bool = True):
        self.lookback_window = lookback_window
        self.normalization_method = normalization_method
        self.feature_selection = feature_selection
        
        # Initialize feature engines
        self.technical_engine = TechnicalIndicatorEngine()
        self.sentiment_engine = SentimentFeatureEngine()
        self.alternative_engine = AlternativeDataEngine()
        
        # Initialize scalers
        self.scaler = self._get_scaler()
        self.imputer = SimpleImputer(strategy='median')
        
        # Feature tracking
        self.feature_names: List[str] = []
        self.is_fitted = False
        
        # Historical data storage
        self.price_history: pd.DataFrame = pd.DataFrame()
        self.feature_history: pd.DataFrame = pd.DataFrame()
        
        logger.info(f"StateProcessor initialized with {lookback_window} lookback window")
    
    def _get_scaler(self):
        """Get appropriate scaler based on normalization method"""
        if self.normalization_method == 'standard':
            return StandardScaler()
        elif self.normalization_method == 'minmax':
            return MinMaxScaler()
        elif self.normalization_method == 'robust':
            return RobustScaler()
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization_method}")
    
    def fit(self, 
            price_data: pd.DataFrame,
            sentiment_data: Optional[pd.DataFrame] = None,
            alternative_data: Optional[pd.DataFrame] = None):
        """Fit the state processor on historical data"""
        logger.info("Fitting StateProcessor on historical data...")
        
        try:
            # Compute all features
            features_df = self._compute_all_features(
                price_data, sentiment_data, alternative_data
            )
            
            if features_df.empty:
                raise ValueError("No features computed from input data")
            
            # Handle missing values
            features_df = self._handle_missing_values(features_df)
            
            # Fit scalers
            self.scaler.fit(features_df)
            self.imputer.fit(features_df)
            
            # Store feature names
            self.feature_names = list(features_df.columns)
            self.is_fitted = True
            
            logger.info(f"StateProcessor fitted with {len(self.feature_names)} features")
            return self
            
        except Exception as e:
            logger.error(f"Error fitting StateProcessor: {e}")
            raise
    
    def transform(self, 
                  current_data: Dict[str, Any],
                  update_history: bool = True) -> MarketState:
        """Transform current market data into a normalized state vector"""
        if not self.is_fitted:
            raise ValueError("StateProcessor must be fitted before transform")
        
        try:
            # Extract components from current data
            price_data = current_data.get('price_data', {})
            sentiment_data = current_data.get('sentiment_data', {})
            alternative_data = current_data.get('alternative_data', {})
            portfolio_state = current_data.get('portfolio_state', {})
            timestamp = current_data.get('timestamp', pd.Timestamp.now())
            
            # Update price history
            if update_history and price_data:
                self._update_price_history(price_data, timestamp)
            
            # Compute current features
            current_features = self._compute_current_features(
                price_data, sentiment_data, alternative_data, portfolio_state
            )
            
            # Create observation vector
            observation = self._create_observation_vector(current_features)
            
            # Create market state object
            market_state = MarketState(
                timestamp=timestamp,
                price_data=price_data,
                technical_indicators=current_features.get('technical', {}),
                sentiment_data=sentiment_data,
                portfolio_state=portfolio_state,
                alternative_data=alternative_data,
                raw_features=current_features.get('raw_vector', np.array([])),
                normalized_features=observation
            )
            
            return market_state
            
        except Exception as e:
            logger.error(f"Error transforming state: {e}")
            raise
    
    def _compute_all_features(self, 
                            price_data: pd.DataFrame,
                            sentiment_data: Optional[pd.DataFrame] = None,
                            alternative_data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """Compute all features for fitting"""
        all_features = pd.DataFrame(index=price_data.index)
        
        # Technical indicators
        tech_features = self.technical_engine.compute_features(price_data)
        if not tech_features.empty:
            all_features = pd.concat([all_features, tech_features], axis=1)
        
        # Add price features
        price_features = self._compute_price_features(price_data)
        all_features = pd.concat([all_features, price_features], axis=1)
        
        # Sentiment features (if available)
        if sentiment_data is not None and not sentiment_data.empty:
            # Resample sentiment data to match price data frequency
            sentiment_resampled = sentiment_data.reindex(price_data.index, method='ffill')
            all_features = pd.concat([all_features, sentiment_resampled], axis=1)
        
        # Alternative data features (if available)
        if alternative_data is not None and not alternative_data.empty:
            alt_resampled = alternative_data.reindex(price_data.index, method='ffill')
            all_features = pd.concat([all_features, alt_resampled], axis=1)
        
        return all_features
    
    def _compute_price_features(self, price_data: pd.DataFrame) -> pd.DataFrame:
        """Compute basic price-based features"""
        features = pd.DataFrame(index=price_data.index)
        
        try:
            # Returns
            features['return_1'] = price_data['close'].pct_change()
            features['return_5'] = price_data['close'].pct_change(5)
            features['return_20'] = price_data['close'].pct_change(20)
            
            # Volatility
            features['volatility_5'] = features['return_1'].rolling(5).std()
            features['volatility_20'] = features['return_1'].rolling(20).std()
            
            # Price ratios
            features['hl_ratio'] = (price_data['high'] - price_data['low']) / price_data['close']
            features['oc_ratio'] = (price_data['close'] - price_data['open']) / price_data['open']
            
            # Volume features
            features['volume_change'] = price_data['volume'].pct_change()
            features['price_volume'] = price_data['close'] * price_data['volume']
            
        except Exception as e:
            logger.warning(f"Error computing price features: {e}")
        
        return features
    
    def _compute_current_features(self, 
                                price_data: Dict[str, float],
                                sentiment_data: Dict[str, Any],
                                alternative_data: Dict[str, Any],
                                portfolio_state: Dict[str, float]) -> Dict[str, Any]:
        """Compute features for current state"""
        current_features = {
            'technical': {},
            'sentiment': {},
            'alternative': {},
            'portfolio': portfolio_state,
            'raw_vector': np.array([])
        }
        
        try:
            # Technical indicators (computed from recent history)
            if len(self.price_history) >= self.lookback_window:
                tech_features = self.technical_engine.compute_features(self.price_history)
                if not tech_features.empty:
                    current_features['technical'] = tech_features.iloc[-1].to_dict()
            
            # Sentiment features
            if sentiment_data:
                sentiment_df = self.sentiment_engine.compute_features(sentiment_data)
                if not sentiment_df.empty:
                    current_features['sentiment'] = sentiment_df.iloc[0].to_dict()
            
            # Alternative data features
            if alternative_data:
                alt_df = self.alternative_engine.compute_features(alternative_data)
                if not alt_df.empty:
                    current_features['alternative'] = alt_df.iloc[0].to_dict()
            
            # Combine all features into a vector
            feature_vector = []
            
            # Add technical features
            for name in self.technical_engine.get_feature_names():
                feature_vector.append(current_features['technical'].get(name, 0.0))
            
            # Add sentiment features
            for name in self.sentiment_engine.get_feature_names():
                feature_vector.append(current_features['sentiment'].get(name, 0.0))
            
            # Add alternative features
            for name in self.alternative_engine.get_feature_names():
                feature_vector.append(current_features['alternative'].get(name, 0.0))
            
            # Add portfolio features
            portfolio_feature_names = ['cash_balance', 'portfolio_value', 'total_equity',
                                     'position_size', 'unrealized_pnl', 'realized_pnl',
                                     'current_allocation', 'drawdown', 'win_rate']
            for name in portfolio_feature_names:
                feature_vector.append(portfolio_state.get(name, 0.0))
            
            current_features['raw_vector'] = np.array(feature_vector)
            
        except Exception as e:
            logger.error(f"Error computing current features: {e}")
        
        return current_features
    
    def _create_observation_vector(self, current_features: Dict[str, Any]) -> np.ndarray:
        """Create normalized observation vector for RL agent"""
        try:
            raw_vector = current_features.get('raw_vector', np.array([]))
            
            if len(raw_vector) == 0:
                # Return zero vector if no features
                return np.zeros(len(self.feature_names))
            
            # Ensure vector has correct dimensions
            if len(raw_vector) != len(self.feature_names):
                logger.warning(f"Feature vector size mismatch: {len(raw_vector)} vs {len(self.feature_names)}")
                # Pad or truncate as needed
                padded_vector = np.zeros(len(self.feature_names))
                min_len = min(len(raw_vector), len(self.feature_names))
                padded_vector[:min_len] = raw_vector[:min_len]
                raw_vector = padded_vector
            
            # Handle missing values
            raw_vector = np.nan_to_num(raw_vector, nan=0.0, posinf=1e6, neginf=-1e6)
            
            # Normalize
            raw_vector_2d = raw_vector.reshape(1, -1)
            normalized_vector = self.scaler.transform(raw_vector_2d)
            
            return normalized_vector.flatten()
            
        except Exception as e:
            logger.error(f"Error creating observation vector: {e}")
            return np.zeros(len(self.feature_names) if self.feature_names else 50)
    
    def _update_price_history(self, price_data: Dict[str, float], timestamp: pd.Timestamp):
        """Update price history with new data"""
        try:
            new_row = pd.Series(price_data, name=timestamp)
            self.price_history = pd.concat([self.price_history, new_row.to_frame().T])
            
            # Keep only recent history
            max_history = max(self.lookback_window * 2, 500)
            if len(self.price_history) > max_history:
                self.price_history = self.price_history.tail(max_history)
                
        except Exception as e:
            logger.error(f"Error updating price history: {e}")
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in feature data"""
        try:
            # Forward fill first, then backward fill, then use median
            df = df.fillna(method='ffill').fillna(method='bfill')
            
            # Use imputer for remaining NaN values
            if df.isnull().sum().sum() > 0:
                df_imputed = pd.DataFrame(
                    self.imputer.fit_transform(df),
                    index=df.index,
                    columns=df.columns
                )
                return df_imputed
            
            return df
            
        except Exception as e:
            logger.error(f"Error handling missing values: {e}")
            return df.fillna(0)  # Fallback: fill with zeros
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores (placeholder for future implementation)"""
        if not self.feature_names:
            return {}
        
        # Placeholder: equal importance for all features
        # In practice, this could use feature selection algorithms
        importance = 1.0 / len(self.feature_names)
        return {name: importance for name in self.feature_names}
    
    def get_observation_space_size(self) -> int:
        """Get the size of the observation space"""
        return len(self.feature_names) if self.feature_names else 0
    
    def reset(self):
        """Reset the state processor"""
        self.price_history = pd.DataFrame()
        self.feature_history = pd.DataFrame()
        logger.info("StateProcessor reset")


if __name__ == "__main__":
    # Example usage
    from rl_config import get_rl_config
    
    config = get_rl_config()
    processor = StateProcessor(
        lookback_window=config.observation.lookback_window,
        normalization_method=config.observation.normalization_method
    )
    
    print(f"StateProcessor initialized")
    print(f"Technical indicators: {len(processor.technical_engine.get_feature_names())}")
    print(f"Sentiment features: {len(processor.sentiment_engine.get_feature_names())}")
    print(f"Alternative features: {len(processor.alternative_engine.get_feature_names())}")