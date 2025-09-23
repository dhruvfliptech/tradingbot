"""
Market Regime Detection and Correlation Regime Changes

This module implements advanced regime detection methods to identify:
- Correlation regime changes
- Market regime shifts (risk-on/risk-off)
- Structural breaks in correlation patterns
- Volatility regime changes
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import chi2, norm
from scipy.signal import find_peaks
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RegimeConfig:
    """Configuration for regime detection"""
    min_regime_length: int = 20  # Minimum days for a regime
    correlation_threshold: float = 0.3  # Threshold for regime change
    volatility_windows: List[int] = field(default_factory=lambda: [10, 20, 60])
    n_regimes: int = 3  # Number of regimes to detect
    breakpoint_confidence: float = 0.95
    use_hmm: bool = True  # Use Hidden Markov Model
    use_structural_breaks: bool = True
    clustering_method: str = 'gmm'  # 'gmm', 'kmeans', 'dbscan'
    lookback_window: int = 252  # Days for regime detection


@dataclass
class RegimeState:
    """Container for regime state information"""
    regime_id: int
    start_date: datetime
    end_date: Optional[datetime]
    characteristics: Dict
    probability: float
    assets_affected: List[str]
    regime_type: str  # 'risk_on', 'risk_off', 'transition'
    confidence: float


@dataclass
class RegimeChangeSignal:
    """Signal for regime change detection"""
    timestamp: datetime
    from_regime: int
    to_regime: int
    confidence: float
    trigger: str  # What triggered the change
    affected_correlations: List[Tuple[str, str]]
    magnitude: float


class RegimeDetector:
    """
    Advanced regime detection for correlation and market regimes
    """
    
    def __init__(self, config: Optional[RegimeConfig] = None):
        """
        Initialize regime detector
        
        Args:
            config: Configuration for regime detection
        """
        self.config = config or RegimeConfig()
        self.current_regime: Optional[RegimeState] = None
        self.regime_history: List[RegimeState] = []
        self.correlation_history: Dict[datetime, pd.DataFrame] = {}
        self.regime_models: Dict = {}
        self.breakpoints: List[datetime] = []
        
        # Initialize models
        self._initialize_models()
        
    def _initialize_models(self):
        """Initialize regime detection models"""
        if self.config.clustering_method == 'gmm':
            self.regime_model = GaussianMixture(
                n_components=self.config.n_regimes,
                covariance_type='full',
                random_state=42
            )
        elif self.config.clustering_method == 'kmeans':
            self.regime_model = KMeans(
                n_clusters=self.config.n_regimes,
                random_state=42
            )
        elif self.config.clustering_method == 'dbscan':
            self.regime_model = DBSCAN(
                eps=0.3,
                min_samples=self.config.min_regime_length
            )
            
    def detect_correlation_regime(
        self,
        correlation_matrix: pd.DataFrame,
        timestamp: datetime
    ) -> RegimeState:
        """
        Detect current correlation regime
        
        Args:
            correlation_matrix: Current correlation matrix
            timestamp: Current timestamp
            
        Returns:
            RegimeState object describing current regime
        """
        # Store correlation history
        self.correlation_history[timestamp] = correlation_matrix
        
        # Extract features from correlation matrix
        features = self._extract_correlation_features(correlation_matrix)
        
        # Detect regime using clustering
        regime_id = self._classify_regime(features)
        
        # Determine regime characteristics
        characteristics = self._analyze_regime_characteristics(
            correlation_matrix,
            features
        )
        
        # Determine regime type
        regime_type = self._determine_regime_type(characteristics)
        
        # Calculate confidence
        confidence = self._calculate_regime_confidence(features, regime_id)
        
        regime_state = RegimeState(
            regime_id=regime_id,
            start_date=timestamp,
            end_date=None,
            characteristics=characteristics,
            probability=confidence,
            assets_affected=list(correlation_matrix.columns),
            regime_type=regime_type,
            confidence=confidence
        )
        
        # Check for regime change
        if self.current_regime is None or regime_id != self.current_regime.regime_id:
            self._handle_regime_change(regime_state, timestamp)
            
        self.current_regime = regime_state
        
        return regime_state
        
    def detect_structural_breaks(
        self,
        correlation_series: Dict[datetime, pd.DataFrame],
        method: str = 'chow'
    ) -> List[datetime]:
        """
        Detect structural breaks in correlation patterns
        
        Args:
            correlation_series: Time series of correlation matrices
            method: Detection method ('chow', 'cusum', 'bai_perron')
            
        Returns:
            List of breakpoint timestamps
        """
        if not self.config.use_structural_breaks:
            return []
            
        breakpoints = []
        
        # Convert correlation matrices to feature vectors
        features_series = []
        timestamps = []
        
        for timestamp, corr_matrix in sorted(correlation_series.items()):
            features = self._extract_correlation_features(corr_matrix)
            features_series.append(features)
            timestamps.append(timestamp)
            
        features_array = np.array(features_series)
        
        if method == 'chow':
            breakpoints = self._chow_test_breaks(features_array, timestamps)
        elif method == 'cusum':
            breakpoints = self._cusum_test_breaks(features_array, timestamps)
        elif method == 'bai_perron':
            breakpoints = self._bai_perron_breaks(features_array, timestamps)
        else:
            logger.warning(f"Unknown structural break method: {method}")
            
        self.breakpoints = breakpoints
        return breakpoints
        
    def detect_volatility_regime(
        self,
        returns_data: pd.DataFrame,
        timestamp: datetime
    ) -> Dict:
        """
        Detect volatility regime
        
        Args:
            returns_data: Returns data for assets
            timestamp: Current timestamp
            
        Returns:
            Dictionary with volatility regime information
        """
        vol_regimes = {}
        
        for window in self.config.volatility_windows:
            # Calculate rolling volatility
            rolling_vol = returns_data.rolling(window).std()
            
            # Current volatility
            current_vol = rolling_vol.iloc[-1]
            
            # Historical volatility percentiles
            vol_percentiles = rolling_vol.quantile([0.25, 0.5, 0.75])
            
            # Classify volatility regime
            if (current_vol < vol_percentiles.loc[0.25]).any():
                vol_regime = 'low_volatility'
            elif (current_vol > vol_percentiles.loc[0.75]).any():
                vol_regime = 'high_volatility'
            else:
                vol_regime = 'normal_volatility'
                
            vol_regimes[f'window_{window}'] = {
                'regime': vol_regime,
                'current_vol': current_vol.mean(),
                'percentile': (current_vol > rolling_vol).mean().mean(),
                'timestamp': timestamp
            }
            
        return vol_regimes
        
    def detect_risk_regime(
        self,
        correlation_matrix: pd.DataFrame,
        returns_data: pd.DataFrame,
        timestamp: datetime
    ) -> str:
        """
        Detect risk-on/risk-off regime
        
        Args:
            correlation_matrix: Current correlations
            returns_data: Returns data
            timestamp: Current timestamp
            
        Returns:
            Risk regime classification
        """
        # Average correlation as risk indicator
        avg_corr = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ].mean()
        
        # Volatility
        recent_vol = returns_data.tail(20).std().mean()
        
        # Dispersion
        cross_sectional_vol = returns_data.iloc[-1].std()
        
        # Risk-on/risk-off classification
        if avg_corr > 0.6 and recent_vol > returns_data.std().mean():
            risk_regime = 'risk_off'  # High correlation, high vol = risk-off
        elif avg_corr < 0.3 and recent_vol < returns_data.std().mean():
            risk_regime = 'risk_on'  # Low correlation, low vol = risk-on
        else:
            risk_regime = 'neutral'
            
        return risk_regime
        
    def predict_regime_change(
        self,
        current_features: np.ndarray,
        lookback: int = 20
    ) -> Tuple[float, int]:
        """
        Predict probability of regime change
        
        Args:
            current_features: Current regime features
            lookback: Number of periods to look back
            
        Returns:
            Tuple of (probability, predicted_regime)
        """
        if len(self.regime_history) < lookback:
            return 0.0, self.current_regime.regime_id if self.current_regime else 0
            
        # Get historical features
        historical_features = []
        for regime in self.regime_history[-lookback:]:
            historical_features.append(regime.characteristics.get('features', []))
            
        if not historical_features:
            return 0.0, 0
            
        # Calculate regime transition probabilities
        transition_probs = self._calculate_transition_probabilities()
        
        # Current regime probability
        if self.current_regime:
            current_prob = transition_probs.get(
                self.current_regime.regime_id, 
                {self.current_regime.regime_id: 0.5}
            )
            
            # Most likely next regime
            next_regime = max(current_prob.items(), key=lambda x: x[1])
            
            # Probability of change
            change_prob = 1.0 - current_prob.get(self.current_regime.regime_id, 0.5)
            
            return change_prob, next_regime[0]
            
        return 0.0, 0
        
    def get_regime_statistics(self) -> pd.DataFrame:
        """
        Get statistics about regime history
        
        Returns:
            DataFrame with regime statistics
        """
        if not self.regime_history:
            return pd.DataFrame()
            
        stats_data = []
        
        for regime in self.regime_history:
            duration = None
            if regime.end_date and regime.start_date:
                duration = (regime.end_date - regime.start_date).days
                
            stats_data.append({
                'regime_id': regime.regime_id,
                'regime_type': regime.regime_type,
                'start_date': regime.start_date,
                'end_date': regime.end_date,
                'duration_days': duration,
                'confidence': regime.confidence,
                'avg_correlation': regime.characteristics.get('avg_correlation', np.nan),
                'volatility': regime.characteristics.get('volatility', np.nan)
            })
            
        return pd.DataFrame(stats_data)
        
    def get_regime_transitions(self) -> pd.DataFrame:
        """
        Get regime transition matrix
        
        Returns:
            DataFrame with transition probabilities
        """
        transitions = self._calculate_transition_probabilities()
        
        # Convert to DataFrame
        regime_ids = sorted(set(r.regime_id for r in self.regime_history))
        transition_matrix = pd.DataFrame(
            0.0,
            index=regime_ids,
            columns=regime_ids
        )
        
        for from_regime, to_regimes in transitions.items():
            for to_regime, prob in to_regimes.items():
                transition_matrix.loc[from_regime, to_regime] = prob
                
        return transition_matrix
        
    # Private helper methods
    
    def _extract_correlation_features(self, correlation_matrix: pd.DataFrame) -> np.ndarray:
        """Extract features from correlation matrix for regime detection"""
        # Upper triangular correlation values
        upper_tri = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ]
        
        features = [
            np.mean(upper_tri),  # Average correlation
            np.median(upper_tri),  # Median correlation
            np.std(upper_tri),  # Correlation dispersion
            np.min(upper_tri),  # Minimum correlation
            np.max(upper_tri),  # Maximum correlation
            np.percentile(upper_tri, 25),  # Q1
            np.percentile(upper_tri, 75),  # Q3
            stats.skew(upper_tri),  # Skewness
            stats.kurtosis(upper_tri),  # Kurtosis
            np.sum(upper_tri > 0.5) / len(upper_tri),  # Fraction high correlation
            np.sum(upper_tri < 0) / len(upper_tri),  # Fraction negative correlation
        ]
        
        # Add eigenvalue features
        eigenvalues = np.linalg.eigvalsh(correlation_matrix)
        features.extend([
            eigenvalues[-1],  # Largest eigenvalue
            eigenvalues[-1] / np.sum(eigenvalues),  # Proportion of variance
            np.sum(eigenvalues[:3]) / np.sum(eigenvalues),  # First 3 components
        ])
        
        return np.array(features)
        
    def _classify_regime(self, features: np.ndarray) -> int:
        """Classify regime based on features"""
        # Reshape for single sample
        features = features.reshape(1, -1)
        
        # Scale features
        if not hasattr(self, 'feature_scaler'):
            self.feature_scaler = StandardScaler()
            # Fit on initial data if available
            if len(self.regime_history) > 0:
                historical_features = [
                    r.characteristics.get('features', features[0])
                    for r in self.regime_history
                ]
                self.feature_scaler.fit(historical_features)
            else:
                self.feature_scaler.fit(features)
                
        features_scaled = self.feature_scaler.transform(features)
        
        # Predict regime
        if hasattr(self.regime_model, 'predict'):
            regime_id = self.regime_model.predict(features_scaled)[0]
        else:
            # For models without predict (like during initialization)
            regime_id = 0
            
        return int(regime_id)
        
    def _analyze_regime_characteristics(
        self,
        correlation_matrix: pd.DataFrame,
        features: np.ndarray
    ) -> Dict:
        """Analyze characteristics of current regime"""
        upper_tri = correlation_matrix.values[
            np.triu_indices_from(correlation_matrix.values, k=1)
        ]
        
        characteristics = {
            'avg_correlation': np.mean(upper_tri),
            'median_correlation': np.median(upper_tri),
            'correlation_dispersion': np.std(upper_tri),
            'max_correlation': np.max(upper_tri),
            'min_correlation': np.min(upper_tri),
            'high_correlation_pairs': np.sum(upper_tri > 0.7),
            'negative_correlation_pairs': np.sum(upper_tri < 0),
            'features': features.tolist()
        }
        
        # Network properties
        threshold = 0.5
        adjacency = (correlation_matrix.abs() > threshold).astype(int)
        np.fill_diagonal(adjacency.values, 0)
        
        characteristics['network_density'] = adjacency.sum().sum() / (
            len(correlation_matrix) * (len(correlation_matrix) - 1)
        )
        
        # Clustering coefficient
        characteristics['clustering_coefficient'] = self._calculate_clustering_coefficient(
            adjacency.values
        )
        
        return characteristics
        
    def _determine_regime_type(self, characteristics: Dict) -> str:
        """Determine regime type based on characteristics"""
        avg_corr = characteristics['avg_correlation']
        dispersion = characteristics['correlation_dispersion']
        
        if avg_corr > 0.6:
            if dispersion < 0.2:
                return 'crisis'  # High correlation, low dispersion
            else:
                return 'risk_off'
        elif avg_corr < 0.3:
            if dispersion < 0.2:
                return 'divergence'  # Low correlation, low dispersion
            else:
                return 'risk_on'
        else:
            return 'transition'
            
    def _calculate_regime_confidence(
        self,
        features: np.ndarray,
        regime_id: int
    ) -> float:
        """Calculate confidence in regime classification"""
        if not hasattr(self.regime_model, 'predict_proba'):
            # For models without probability prediction
            return 0.5
            
        features = features.reshape(1, -1)
        features_scaled = self.feature_scaler.transform(features)
        
        try:
            probs = self.regime_model.predict_proba(features_scaled)[0]
            return probs[regime_id]
        except:
            return 0.5
            
    def _handle_regime_change(
        self,
        new_regime: RegimeState,
        timestamp: datetime
    ):
        """Handle regime change event"""
        if self.current_regime:
            # End current regime
            self.current_regime.end_date = timestamp
            self.regime_history.append(self.current_regime)
            
            # Log regime change
            logger.info(
                f"Regime change detected at {timestamp}: "
                f"{self.current_regime.regime_id} -> {new_regime.regime_id}"
            )
            
    def _calculate_transition_probabilities(self) -> Dict[int, Dict[int, float]]:
        """Calculate regime transition probabilities"""
        transitions = {}
        
        for i in range(len(self.regime_history) - 1):
            from_regime = self.regime_history[i].regime_id
            to_regime = self.regime_history[i + 1].regime_id
            
            if from_regime not in transitions:
                transitions[from_regime] = {}
                
            if to_regime not in transitions[from_regime]:
                transitions[from_regime][to_regime] = 0
                
            transitions[from_regime][to_regime] += 1
            
        # Normalize to probabilities
        for from_regime in transitions:
            total = sum(transitions[from_regime].values())
            for to_regime in transitions[from_regime]:
                transitions[from_regime][to_regime] /= total
                
        return transitions
        
    def _chow_test_breaks(
        self,
        features_array: np.ndarray,
        timestamps: List[datetime]
    ) -> List[datetime]:
        """Detect breaks using Chow test"""
        breakpoints = []
        n_features = features_array.shape[1]
        
        for i in range(self.config.min_regime_length, 
                      len(features_array) - self.config.min_regime_length):
            
            # Split data
            before = features_array[:i]
            after = features_array[i:]
            
            # Calculate F-statistic
            f_stat = self._calculate_chow_f_statistic(before, after)
            
            # Critical value
            df1 = n_features
            df2 = len(features_array) - 2 * n_features
            critical_value = stats.f.ppf(self.config.breakpoint_confidence, df1, df2)
            
            if f_stat > critical_value:
                breakpoints.append(timestamps[i])
                
        return breakpoints
        
    def _cusum_test_breaks(
        self,
        features_array: np.ndarray,
        timestamps: List[datetime]
    ) -> List[datetime]:
        """Detect breaks using CUSUM test"""
        breakpoints = []
        
        for feature_idx in range(features_array.shape[1]):
            feature = features_array[:, feature_idx]
            
            # Calculate CUSUM
            mean = np.mean(feature)
            cusum = np.cumsum(feature - mean)
            
            # Find peaks in CUSUM
            peaks, properties = find_peaks(np.abs(cusum), height=np.std(cusum) * 2)
            
            for peak in peaks:
                if peak >= self.config.min_regime_length and \
                   peak < len(timestamps) - self.config.min_regime_length:
                    breakpoints.append(timestamps[peak])
                    
        # Remove duplicates and sort
        breakpoints = sorted(list(set(breakpoints)))
        
        return breakpoints
        
    def _bai_perron_breaks(
        self,
        features_array: np.ndarray,
        timestamps: List[datetime]
    ) -> List[datetime]:
        """Detect breaks using Bai-Perron method"""
        # Simplified implementation of Bai-Perron
        # In production, use ruptures or similar package
        
        breakpoints = []
        max_breaks = 5
        
        # Use variance-based detection
        for feature_idx in range(features_array.shape[1]):
            feature = features_array[:, feature_idx]
            
            # Rolling variance
            window = self.config.min_regime_length
            rolling_var = pd.Series(feature).rolling(window).var()
            
            # Detect significant changes in variance
            var_changes = rolling_var.diff().abs()
            threshold = var_changes.quantile(0.95)
            
            potential_breaks = np.where(var_changes > threshold)[0]
            
            for break_idx in potential_breaks:
                if break_idx < len(timestamps):
                    breakpoints.append(timestamps[break_idx])
                    
        # Filter and return unique breakpoints
        breakpoints = sorted(list(set(breakpoints)))
        
        # Limit number of breaks
        if len(breakpoints) > max_breaks:
            # Keep most significant breaks
            breakpoints = breakpoints[:max_breaks]
            
        return breakpoints
        
    def _calculate_chow_f_statistic(
        self,
        before: np.ndarray,
        after: np.ndarray
    ) -> float:
        """Calculate Chow test F-statistic"""
        # Simplified Chow test
        n1, k = before.shape
        n2 = after.shape[0]
        n = n1 + n2
        
        # Pooled variance
        var_pooled = (np.var(before, axis=0) * n1 + np.var(after, axis=0) * n2) / n
        
        # Separate variances
        var1 = np.var(before, axis=0)
        var2 = np.var(after, axis=0)
        
        # F-statistic
        f_stat = np.mean((var1 + var2) / (2 * var_pooled))
        
        return f_stat
        
    def _calculate_clustering_coefficient(self, adjacency: np.ndarray) -> float:
        """Calculate clustering coefficient of correlation network"""
        n = len(adjacency)
        clustering_coeffs = []
        
        for i in range(n):
            neighbors = np.where(adjacency[i] == 1)[0]
            k = len(neighbors)
            
            if k < 2:
                clustering_coeffs.append(0)
                continue
                
            # Count edges between neighbors
            edges = 0
            for j in range(k):
                for l in range(j + 1, k):
                    if adjacency[neighbors[j], neighbors[l]] == 1:
                        edges += 1
                        
            # Clustering coefficient
            max_edges = k * (k - 1) / 2
            clustering_coeffs.append(edges / max_edges if max_edges > 0 else 0)
            
        return np.mean(clustering_coeffs)