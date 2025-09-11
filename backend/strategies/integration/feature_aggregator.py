"""
Feature Aggregator
==================

Aggregates and processes features from all institutional strategies into a unified
feature vector for the RL environment. Handles feature normalization, selection,
dimensionality reduction, and real-time updates.

Key Features:
- Multi-strategy feature aggregation
- Real-time feature normalization
- Feature selection and importance scoring
- Dimensionality reduction
- Feature correlation analysis
- Dynamic feature weighting
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd
from collections import deque, defaultdict
from threading import Lock
import pickle
import json

# Scientific computing libraries
from sklearn.preprocessing import RobustScaler, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.decomposition import PCA, FastICA
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import pdist, squareform
import scipy.cluster.hierarchy as sch

logger = logging.getLogger(__name__)


class NormalizationMethod(Enum):
    """Feature normalization methods"""
    ROBUST = "robust"
    STANDARD = "standard"
    MINMAX = "minmax"
    RANK = "rank"
    QUANTILE = "quantile"


class FeatureSelectionMethod(Enum):
    """Feature selection methods"""
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    F_REGRESSION = "f_regression"
    RANDOM_FOREST = "random_forest"
    VARIANCE = "variance"


class DimensionalityReduction(Enum):
    """Dimensionality reduction methods"""
    PCA = "pca"
    ICA = "ica"
    SELECTION = "selection"
    NONE = "none"


@dataclass
class FeatureMetadata:
    """Metadata for individual features"""
    name: str
    strategy: str
    data_type: str  # 'continuous', 'categorical', 'binary'
    importance: float = 0.0
    correlation_with_target: float = 0.0
    mutual_info: float = 0.0
    variance: float = 0.0
    missing_rate: float = 0.0
    outlier_rate: float = 0.0
    last_update: Optional[datetime] = None
    enabled: bool = True
    weight: float = 1.0


@dataclass
class AggregationStats:
    """Statistics for feature aggregation process"""
    total_features: int = 0
    selected_features: int = 0
    aggregation_time: float = 0.0
    normalization_time: float = 0.0
    selection_time: float = 0.0
    reduction_time: float = 0.0
    update_count: int = 0
    error_count: int = 0
    last_update: Optional[datetime] = None


class FeatureAggregator:
    """
    Advanced feature aggregation system for institutional trading strategies.
    
    Combines features from multiple strategies into optimized feature vectors
    for RL training and inference.
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize feature aggregator"""
        self.config = config or self._default_config()
        
        # Feature storage
        self.raw_features: Dict[str, float] = {}
        self.normalized_features: Dict[str, float] = {}
        self.selected_features: Dict[str, float] = {}
        self.reduced_features: np.ndarray = np.array([])
        
        # Feature metadata
        self.feature_metadata: Dict[str, FeatureMetadata] = {}
        self.feature_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.config['history_length'])
        )
        
        # Processing components
        self.scalers: Dict[str, Any] = {}
        self.feature_selector = None
        self.dimensionality_reducer = None
        
        # Strategy weights
        self.strategy_weights: Dict[str, float] = {}
        
        # State management
        self.is_fitted = False
        self.lock = Lock()
        self.stats = AggregationStats()
        
        # Target values for supervised feature selection
        self.target_history: deque = deque(maxlen=self.config['history_length'])
        
        # Feature correlation matrix
        self.correlation_matrix: Optional[np.ndarray] = None
        self.feature_clusters: Dict[int, List[str]] = {}
        
        logger.info("Feature Aggregator initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'max_features': 50,  # Maximum features to output
            'history_length': 1000,  # Feature history length
            'normalization_method': 'robust',
            'feature_selection_method': 'mutual_info',
            'dimensionality_reduction': 'pca',
            'correlation_threshold': 0.95,  # Remove highly correlated features
            'variance_threshold': 0.01,  # Minimum variance threshold
            'missing_threshold': 0.1,  # Maximum missing rate
            'outlier_threshold': 0.05,  # Maximum outlier rate
            'update_interval': 1.0,  # Seconds between updates
            'refit_interval': 3600,  # Seconds between refitting
            'enable_feature_selection': True,
            'enable_dimensionality_reduction': True,
            'enable_clustering': True,
            'n_clusters': 10,  # Number of feature clusters
            'selection_ratio': 0.8,  # Ratio of features to select
            'pca_variance_ratio': 0.95,  # Explained variance for PCA
            'ica_components': None,  # Number of ICA components (None = auto)
            'enable_outlier_detection': True,
            'outlier_method': 'isolation_forest',
            'enable_real_time_updates': True,
            'batch_size': 100,  # Batch processing size
            'parallel_processing': True,
            'cache_size': 10000,
            'enable_persistence': True,
            'persistence_path': 'feature_aggregator_state.pkl'
        }
    
    async def start(self):
        """Start the feature aggregator"""
        logger.info("Starting Feature Aggregator...")
        
        # Load persisted state if available
        if self.config.get('enable_persistence', True):
            await self._load_state()
        
        # Initialize processing components
        await self._initialize_components()
        
        # Start update loop if real-time updates enabled
        if self.config.get('enable_real_time_updates', True):
            asyncio.create_task(self._update_loop())
        
        logger.info("Feature Aggregator started")
    
    async def stop(self):
        """Stop the feature aggregator"""
        logger.info("Stopping Feature Aggregator...")
        
        # Save state if persistence enabled
        if self.config.get('enable_persistence', True):
            await self._save_state()
        
        logger.info("Feature Aggregator stopped")
    
    async def _initialize_components(self):
        """Initialize processing components"""
        # Initialize scalers for different normalization methods
        scaler_map = {
            'robust': RobustScaler(),
            'standard': StandardScaler(),
            'minmax': MinMaxScaler(),
            'rank': None,  # Custom implementation
            'quantile': None  # Custom implementation
        }
        
        norm_method = self.config.get('normalization_method', 'robust')
        if norm_method in scaler_map:
            self.scalers['primary'] = scaler_map[norm_method]
        
        # Initialize feature selector
        if self.config.get('enable_feature_selection', True):
            selection_method = self.config.get('feature_selection_method', 'mutual_info')
            k_features = min(self.config['max_features'], 
                           int(len(self.feature_metadata) * self.config.get('selection_ratio', 0.8)))
            
            if selection_method == 'mutual_info':
                self.feature_selector = SelectKBest(
                    score_func=mutual_info_regression,
                    k=k_features
                )
            elif selection_method == 'f_regression':
                self.feature_selector = SelectKBest(
                    score_func=f_regression,
                    k=k_features
                )
            elif selection_method == 'random_forest':
                self.feature_selector = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
        
        # Initialize dimensionality reducer
        if self.config.get('enable_dimensionality_reduction', True):
            reduction_method = self.config.get('dimensionality_reduction', 'pca')
            
            if reduction_method == 'pca':
                n_components = min(
                    self.config['max_features'],
                    len(self.feature_metadata)
                )
                self.dimensionality_reducer = PCA(
                    n_components=n_components,
                    random_state=42
                )
            elif reduction_method == 'ica':
                n_components = self.config.get('ica_components') or min(
                    self.config['max_features'],
                    len(self.feature_metadata)
                )
                self.dimensionality_reducer = FastICA(
                    n_components=n_components,
                    random_state=42
                )
    
    async def add_features(self, features: Dict[str, float], target: Optional[float] = None):
        """Add new features to the aggregator"""
        async with self._async_lock():
            try:
                start_time = time.time()
                
                # Validate and clean features
                clean_features = await self._validate_features(features)
                
                # Update raw features
                self.raw_features.update(clean_features)
                
                # Update feature metadata
                await self._update_feature_metadata(clean_features)
                
                # Update feature history
                for name, value in clean_features.items():
                    self.feature_history[name].append({
                        'value': value,
                        'timestamp': datetime.now()
                    })
                
                # Add target if provided
                if target is not None:
                    self.target_history.append(target)
                
                # Process features
                await self._process_features()
                
                # Update statistics
                self.stats.update_count += 1
                self.stats.aggregation_time = time.time() - start_time
                self.stats.last_update = datetime.now()
                self.stats.total_features = len(self.raw_features)
                
                logger.debug(f"Added {len(clean_features)} features in {self.stats.aggregation_time:.3f}s")
                
            except Exception as e:
                self.stats.error_count += 1
                logger.error(f"Error adding features: {e}")
                raise
    
    async def _validate_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Validate and clean input features"""
        clean_features = {}
        
        for name, value in features.items():
            try:
                # Check if value is numeric
                if not isinstance(value, (int, float, np.number)):
                    continue
                
                # Handle NaN and infinite values
                if np.isnan(value) or np.isinf(value):
                    # Use last known value or zero
                    if name in self.feature_history and self.feature_history[name]:
                        value = self.feature_history[name][-1]['value']
                    else:
                        value = 0.0
                
                # Check for reasonable bounds
                if abs(value) > 1e10:  # Very large values
                    value = np.sign(value) * 1e10
                
                clean_features[name] = float(value)
                
            except Exception as e:
                logger.warning(f"Error validating feature {name}: {e}")
                continue
        
        return clean_features
    
    async def _update_feature_metadata(self, features: Dict[str, float]):
        """Update metadata for features"""
        for name, value in features.items():
            if name not in self.feature_metadata:
                # Create new metadata
                strategy = name.split('_')[0] if '_' in name else 'unknown'
                self.feature_metadata[name] = FeatureMetadata(
                    name=name,
                    strategy=strategy,
                    data_type='continuous',  # Assume continuous for now
                    last_update=datetime.now()
                )
            else:
                # Update existing metadata
                metadata = self.feature_metadata[name]
                metadata.last_update = datetime.now()
                
                # Update variance (running calculation)
                if len(self.feature_history[name]) > 1:
                    values = [item['value'] for item in self.feature_history[name]]
                    metadata.variance = np.var(values)
                
                # Calculate missing rate
                total_updates = len(self.feature_history[name])
                missing_count = sum(1 for item in self.feature_history[name] 
                                  if item['value'] == 0.0)  # Assuming 0 means missing
                metadata.missing_rate = missing_count / max(1, total_updates)
    
    async def _process_features(self):
        """Process features through normalization, selection, and reduction"""
        if not self.raw_features:
            return
        
        try:
            # Normalize features
            start_time = time.time()
            await self._normalize_features()
            self.stats.normalization_time = time.time() - start_time
            
            # Select features
            if self.config.get('enable_feature_selection', True):
                start_time = time.time()
                await self._select_features()
                self.stats.selection_time = time.time() - start_time
            else:
                self.selected_features = self.normalized_features.copy()
            
            # Reduce dimensionality
            if self.config.get('enable_dimensionality_reduction', True):
                start_time = time.time()
                await self._reduce_dimensionality()
                self.stats.reduction_time = time.time() - start_time
            
            self.stats.selected_features = len(self.selected_features)
            
        except Exception as e:
            logger.error(f"Error processing features: {e}")
            raise
    
    async def _normalize_features(self):
        """Normalize features using configured method"""
        if not self.raw_features:
            return
        
        feature_names = list(self.raw_features.keys())
        feature_values = np.array([self.raw_features[name] for name in feature_names])
        
        # Reshape for sklearn
        if len(feature_values.shape) == 1:
            feature_values = feature_values.reshape(-1, 1)
        else:
            feature_values = feature_values.reshape(1, -1)
        
        normalization_method = self.config.get('normalization_method', 'robust')
        
        if normalization_method in ['robust', 'standard', 'minmax']:
            scaler = self.scalers.get('primary')
            if scaler is not None:
                if not self.is_fitted:
                    # For single sample, we can't fit properly
                    # Use running statistics or default normalization
                    normalized_values = self._default_normalize(feature_values.flatten())
                else:
                    normalized_values = scaler.transform(feature_values).flatten()
            else:
                normalized_values = self._default_normalize(feature_values.flatten())
        
        elif normalization_method == 'rank':
            normalized_values = self._rank_normalize(feature_values.flatten())
        
        elif normalization_method == 'quantile':
            normalized_values = self._quantile_normalize(feature_values.flatten())
        
        else:
            normalized_values = feature_values.flatten()
        
        # Update normalized features
        self.normalized_features = {
            name: value for name, value in zip(feature_names, normalized_values)
        }
    
    def _default_normalize(self, values: np.ndarray) -> np.ndarray:
        """Default normalization for single samples"""
        # Use z-score normalization with running statistics
        normalized = []
        for i, (name, value) in enumerate(zip(self.raw_features.keys(), values)):
            if name in self.feature_history and len(self.feature_history[name]) > 1:
                # Use historical statistics
                hist_values = [item['value'] for item in self.feature_history[name]]
                mean = np.mean(hist_values)
                std = np.std(hist_values)
                if std > 0:
                    normalized_value = (value - mean) / std
                else:
                    normalized_value = 0.0
            else:
                # Use simple clipping
                normalized_value = np.clip(value, -5, 5)
            
            normalized.append(normalized_value)
        
        return np.array(normalized)
    
    def _rank_normalize(self, values: np.ndarray) -> np.ndarray:
        """Rank-based normalization"""
        # For single sample, return as-is
        return values
    
    def _quantile_normalize(self, values: np.ndarray) -> np.ndarray:
        """Quantile normalization"""
        # For single sample, return as-is
        return values
    
    async def _select_features(self):
        """Select important features"""
        if not self.normalized_features:
            return
        
        # For now, select all features (would implement proper selection with historical data)
        self.selected_features = self.normalized_features.copy()
        
        # Apply variance threshold
        variance_threshold = self.config.get('variance_threshold', 0.01)
        filtered_features = {}
        
        for name, value in self.selected_features.items():
            metadata = self.feature_metadata.get(name)
            if metadata and metadata.variance >= variance_threshold:
                filtered_features[name] = value
            elif not metadata:  # New feature, include it
                filtered_features[name] = value
        
        self.selected_features = filtered_features
        
        # Apply correlation threshold (remove highly correlated features)
        if len(self.selected_features) > 1:
            await self._remove_correlated_features()
    
    async def _remove_correlated_features(self):
        """Remove highly correlated features"""
        if len(self.selected_features) < 2:
            return
        
        correlation_threshold = self.config.get('correlation_threshold', 0.95)
        
        # Build correlation matrix from historical data
        feature_names = list(self.selected_features.keys())
        correlation_matrix = np.eye(len(feature_names))
        
        for i, name1 in enumerate(feature_names):
            for j, name2 in enumerate(feature_names):
                if i >= j:
                    continue
                
                # Calculate correlation from history
                if (name1 in self.feature_history and name2 in self.feature_history and
                    len(self.feature_history[name1]) > 10 and len(self.feature_history[name2]) > 10):
                    
                    values1 = [item['value'] for item in self.feature_history[name1][-100:]]
                    values2 = [item['value'] for item in self.feature_history[name2][-100:]]
                    
                    if len(values1) == len(values2):
                        try:
                            corr, _ = pearsonr(values1, values2)
                            correlation_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
                            correlation_matrix[j, i] = correlation_matrix[i, j]
                        except:
                            correlation_matrix[i, j] = 0
                            correlation_matrix[j, i] = 0
        
        # Find and remove highly correlated features
        to_remove = set()
        for i in range(len(feature_names)):
            for j in range(i + 1, len(feature_names)):
                if correlation_matrix[i, j] > correlation_threshold:
                    # Remove feature with lower importance (or just the second one)
                    feature_to_remove = feature_names[j]
                    to_remove.add(feature_to_remove)
        
        # Remove correlated features
        for name in to_remove:
            if name in self.selected_features:
                del self.selected_features[name]
        
        if to_remove:
            logger.info(f"Removed {len(to_remove)} highly correlated features")
    
    async def _reduce_dimensionality(self):
        """Reduce feature dimensionality"""
        if not self.selected_features:
            self.reduced_features = np.array([])
            return
        
        feature_values = np.array(list(self.selected_features.values()))
        
        # For single sample, dimensionality reduction is not meaningful
        # Just return the selected features as-is
        self.reduced_features = feature_values
    
    async def _update_loop(self):
        """Background update loop for refitting and maintenance"""
        while True:
            try:
                await asyncio.sleep(self.config.get('refit_interval', 3600))
                
                # Refit components if enough data
                if len(self.feature_history) > 100:
                    await self._refit_components()
                
                # Update feature importance scores
                await self._update_feature_importance()
                
                # Cluster features
                if self.config.get('enable_clustering', True):
                    await self._cluster_features()
                
            except Exception as e:
                logger.error(f"Error in update loop: {e}")
                await asyncio.sleep(60)
    
    async def _refit_components(self):
        """Refit normalization and selection components with historical data"""
        try:
            # Collect historical data
            if not self.feature_history:
                return
            
            # Build feature matrix
            feature_names = list(self.feature_history.keys())
            min_length = min(len(self.feature_history[name]) for name in feature_names)
            
            if min_length < 50:  # Need sufficient data
                return
            
            # Use last N samples
            n_samples = min(min_length, 1000)
            feature_matrix = []
            
            for i in range(n_samples):
                sample = []
                for name in feature_names:
                    value = self.feature_history[name][-(n_samples-i)]['value']
                    sample.append(value)
                feature_matrix.append(sample)
            
            feature_matrix = np.array(feature_matrix)
            
            # Refit scalers
            if 'primary' in self.scalers and self.scalers['primary'] is not None:
                self.scalers['primary'].fit(feature_matrix)
            
            # Refit feature selector with targets if available
            if (self.feature_selector is not None and 
                len(self.target_history) >= n_samples):
                
                targets = list(self.target_history)[-n_samples:]
                
                if hasattr(self.feature_selector, 'fit'):
                    self.feature_selector.fit(feature_matrix, targets)
                elif hasattr(self.feature_selector, 'fit'):  # Random Forest
                    self.feature_selector.fit(feature_matrix, targets)
            
            # Refit dimensionality reducer
            if self.dimensionality_reducer is not None:
                self.dimensionality_reducer.fit(feature_matrix)
            
            self.is_fitted = True
            logger.info("Successfully refitted aggregator components")
            
        except Exception as e:
            logger.error(f"Error refitting components: {e}")
    
    async def _update_feature_importance(self):
        """Update feature importance scores"""
        for name, metadata in self.feature_metadata.items():
            if name not in self.feature_history or len(self.feature_history[name]) < 20:
                continue
            
            try:
                # Calculate correlation with targets if available
                if len(self.target_history) >= 20:
                    feature_values = [item['value'] for item in self.feature_history[name][-20:]]
                    target_values = list(self.target_history)[-20:]
                    
                    if len(feature_values) == len(target_values):
                        corr, _ = pearsonr(feature_values, target_values)
                        metadata.correlation_with_target = abs(corr) if not np.isnan(corr) else 0
                
                # Calculate mutual information
                # (Simplified version - would implement proper MI calculation)
                metadata.mutual_info = metadata.correlation_with_target * 0.5
                
                # Combined importance score
                metadata.importance = (
                    metadata.correlation_with_target * 0.4 +
                    metadata.mutual_info * 0.3 +
                    metadata.variance * 0.2 +
                    (1 - metadata.missing_rate) * 0.1
                )
                
            except Exception as e:
                logger.warning(f"Error updating importance for {name}: {e}")
    
    async def _cluster_features(self):
        """Cluster similar features to identify redundancy"""
        if len(self.feature_metadata) < 3:
            return
        
        try:
            # Build feature correlation matrix
            feature_names = list(self.feature_metadata.keys())
            n_features = len(feature_names)
            correlation_matrix = np.eye(n_features)
            
            # Calculate pairwise correlations from history
            for i, name1 in enumerate(feature_names):
                for j, name2 in enumerate(feature_names):
                    if i >= j:
                        continue
                    
                    if (name1 in self.feature_history and name2 in self.feature_history and
                        len(self.feature_history[name1]) > 10 and len(self.feature_history[name2]) > 10):
                        
                        values1 = [item['value'] for item in self.feature_history[name1][-50:]]
                        values2 = [item['value'] for item in self.feature_history[name2][-50:]]
                        
                        if len(values1) == len(values2):
                            try:
                                corr, _ = pearsonr(values1, values2)
                                correlation_matrix[i, j] = abs(corr) if not np.isnan(corr) else 0
                                correlation_matrix[j, i] = correlation_matrix[i, j]
                            except:
                                pass
            
            # Perform hierarchical clustering
            distance_matrix = 1 - correlation_matrix
            np.fill_diagonal(distance_matrix, 0)
            
            # Convert to condensed distance matrix
            condensed_distances = pdist(distance_matrix, metric='precomputed')
            
            # Perform clustering
            linkage_matrix = sch.linkage(condensed_distances, method='ward')
            
            # Get cluster assignments
            n_clusters = min(self.config.get('n_clusters', 10), n_features)
            cluster_assignments = sch.fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Update feature clusters
            self.feature_clusters.clear()
            for i, cluster_id in enumerate(cluster_assignments):
                if cluster_id not in self.feature_clusters:
                    self.feature_clusters[cluster_id] = []
                self.feature_clusters[cluster_id].append(feature_names[i])
            
            logger.info(f"Clustered {n_features} features into {len(self.feature_clusters)} clusters")
            
        except Exception as e:
            logger.error(f"Error clustering features: {e}")
    
    def get_aggregated_features(self) -> Dict[str, float]:
        """Get the final aggregated feature vector"""
        if self.config.get('enable_dimensionality_reduction', True) and len(self.reduced_features) > 0:
            # Return reduced features as named vector
            return {
                f"reduced_feature_{i}": value 
                for i, value in enumerate(self.reduced_features)
            }
        else:
            return self.selected_features.copy()
    
    def get_feature_vector(self) -> np.ndarray:
        """Get feature vector as numpy array"""
        aggregated = self.get_aggregated_features()
        return np.array(list(aggregated.values())) if aggregated else np.array([])
    
    def get_feature_names(self) -> List[str]:
        """Get names of final features"""
        aggregated = self.get_aggregated_features()
        return list(aggregated.keys())
    
    def get_feature_metadata(self, feature_name: str) -> Optional[FeatureMetadata]:
        """Get metadata for specific feature"""
        return self.feature_metadata.get(feature_name)
    
    def get_feature_importance_ranking(self) -> List[Tuple[str, float]]:
        """Get features ranked by importance"""
        rankings = [
            (name, metadata.importance) 
            for name, metadata in self.feature_metadata.items()
        ]
        return sorted(rankings, key=lambda x: x[1], reverse=True)
    
    def get_correlation_matrix(self) -> Optional[np.ndarray]:
        """Get feature correlation matrix"""
        return self.correlation_matrix
    
    def get_feature_clusters(self) -> Dict[int, List[str]]:
        """Get feature clusters"""
        return self.feature_clusters.copy()
    
    def update_strategy_weight(self, strategy_name: str, weight: float):
        """Update weight for specific strategy"""
        self.strategy_weights[strategy_name] = weight
        
        # Update weights for features from this strategy
        for name, metadata in self.feature_metadata.items():
            if metadata.strategy == strategy_name:
                metadata.weight = weight
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get aggregator metrics"""
        return {
            'stats': {
                'total_features': self.stats.total_features,
                'selected_features': self.stats.selected_features,
                'aggregation_time': self.stats.aggregation_time,
                'normalization_time': self.stats.normalization_time,
                'selection_time': self.stats.selection_time,
                'reduction_time': self.stats.reduction_time,
                'update_count': self.stats.update_count,
                'error_count': self.stats.error_count,
                'last_update': self.stats.last_update.isoformat() if self.stats.last_update else None
            },
            'features': {
                'raw_count': len(self.raw_features),
                'normalized_count': len(self.normalized_features),
                'selected_count': len(self.selected_features),
                'reduced_dimensions': len(self.reduced_features),
                'total_history_length': sum(len(hist) for hist in self.feature_history.values()),
                'clusters': len(self.feature_clusters)
            },
            'config': self.config,
            'is_fitted': self.is_fitted
        }
    
    async def _save_state(self):
        """Save aggregator state to disk"""
        try:
            state = {
                'feature_metadata': self.feature_metadata,
                'strategy_weights': self.strategy_weights,
                'scalers': self.scalers,
                'feature_selector': self.feature_selector,
                'dimensionality_reducer': self.dimensionality_reducer,
                'is_fitted': self.is_fitted,
                'stats': self.stats
            }
            
            with open(self.config['persistence_path'], 'wb') as f:
                pickle.dump(state, f)
            
            logger.info("Saved aggregator state")
            
        except Exception as e:
            logger.error(f"Error saving state: {e}")
    
    async def _load_state(self):
        """Load aggregator state from disk"""
        try:
            with open(self.config['persistence_path'], 'rb') as f:
                state = pickle.load(f)
            
            self.feature_metadata = state.get('feature_metadata', {})
            self.strategy_weights = state.get('strategy_weights', {})
            self.scalers = state.get('scalers', {})
            self.feature_selector = state.get('feature_selector')
            self.dimensionality_reducer = state.get('dimensionality_reducer')
            self.is_fitted = state.get('is_fitted', False)
            self.stats = state.get('stats', AggregationStats())
            
            logger.info("Loaded aggregator state")
            
        except FileNotFoundError:
            logger.info("No saved state found, starting fresh")
        except Exception as e:
            logger.error(f"Error loading state: {e}")
    
    async def _async_lock(self):
        """Async context manager for thread lock"""
        class AsyncLock:
            def __init__(self, lock):
                self.lock = lock
            
            async def __aenter__(self):
                self.lock.acquire()
                return self
            
            async def __aexit__(self, exc_type, exc_val, exc_tb):
                self.lock.release()
        
        return AsyncLock(self.lock)


if __name__ == "__main__":
    # Example usage
    import asyncio
    
    async def main():
        # Create feature aggregator
        aggregator = FeatureAggregator()
        
        try:
            await aggregator.start()
            
            # Add some sample features
            features = {
                'whale_tracker_sentiment': 0.75,
                'whale_tracker_volume': 1000000,
                'volume_profile_poc': 30000,
                'volume_profile_imbalance': 0.3,
                'order_book_spread': 0.001,
                'order_book_depth': 500000
            }
            
            await aggregator.add_features(features, target=0.02)
            
            # Get aggregated features
            final_features = aggregator.get_aggregated_features()
            print(f"Aggregated features: {final_features}")
            
            # Get metrics
            metrics = aggregator.get_metrics()
            print(f"Metrics: {metrics}")
            
        finally:
            await aggregator.stop()
    
    asyncio.run(main())