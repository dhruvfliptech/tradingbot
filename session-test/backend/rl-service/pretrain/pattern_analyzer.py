"""
Pattern Analyzer for RL Pre-training Pipeline

This module analyzes and categorizes trading patterns extracted from Composer strategies,
providing detailed insights for supervised pre-training of RL agents.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import pickle
import json
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from composer_extractor import StrategyPattern

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class PatternCluster:
    """Represents a cluster of similar trading patterns"""
    cluster_id: int
    pattern_type: str
    market_regime: str
    patterns: List[StrategyPattern]
    centroid: np.ndarray
    characteristics: Dict[str, Any]
    performance_stats: Dict[str, float]
    confidence_score: float

@dataclass
class PatternAnalysis:
    """Comprehensive analysis results for trading patterns"""
    total_patterns: int
    pattern_distribution: Dict[str, int]
    regime_distribution: Dict[str, int]
    performance_summary: Dict[str, float]
    clusters: List[PatternCluster]
    feature_importance: Dict[str, float]
    correlation_matrix: np.ndarray
    anomalies: List[StrategyPattern]
    recommendations: List[str]

class PatternAnalyzer:
    """Analyzes and categorizes trading patterns for RL pre-training"""
    
    def __init__(self, 
                 feature_dim: int = 50,
                 clustering_method: str = 'kmeans',
                 n_clusters: int = 8):
        self.feature_dim = feature_dim
        self.clustering_method = clustering_method
        self.n_clusters = n_clusters
        
        # ML components
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=min(feature_dim, 20))
        self.tsne = TSNE(n_components=2, random_state=42)
        
        # Pattern categorization
        self.pattern_categories = {
            'momentum': {
                'description': 'Patterns that capitalize on price momentum',
                'indicators': ['moving_average', 'rsi', 'momentum'],
                'typical_duration': 'short to medium',
                'market_conditions': ['trending', 'volatile']
            },
            'mean_reversion': {
                'description': 'Patterns that profit from price reversions',
                'indicators': ['bollinger_bands', 'rsi', 'z_score'],
                'typical_duration': 'short',
                'market_conditions': ['sideways', 'oversold', 'overbought']
            },
            'breakout': {
                'description': 'Patterns that capture breakout movements',
                'indicators': ['volume', 'volatility', 'support_resistance'],
                'typical_duration': 'medium',
                'market_conditions': ['consolidation', 'low_volatility']
            },
            'trend_following': {
                'description': 'Patterns that follow established trends',
                'indicators': ['trend_lines', 'moving_averages', 'adx'],
                'typical_duration': 'medium to long',
                'market_conditions': ['trending', 'bull', 'bear']
            }
        }
        
        # Market regime characteristics
        self.regime_characteristics = {
            'bull': {'volatility': 'low to medium', 'trend': 'upward', 'sentiment': 'positive'},
            'bear': {'volatility': 'high', 'trend': 'downward', 'sentiment': 'negative'},
            'sideways': {'volatility': 'low', 'trend': 'neutral', 'sentiment': 'neutral'},
            'volatile': {'volatility': 'high', 'trend': 'mixed', 'sentiment': 'uncertain'}
        }
    
    def analyze_patterns(self, patterns: List[StrategyPattern]) -> PatternAnalysis:
        """Perform comprehensive analysis of trading patterns"""
        logger.info(f"Analyzing {len(patterns)} trading patterns...")
        
        if not patterns:
            logger.warning("No patterns provided for analysis")
            return self._empty_analysis()
        
        # Prepare data
        features, labels = self._prepare_features(patterns)
        
        # Basic statistics
        pattern_distribution = self._calculate_pattern_distribution(patterns)
        regime_distribution = self._calculate_regime_distribution(patterns)
        performance_summary = self._calculate_performance_summary(patterns)
        
        # Feature analysis
        feature_importance = self._analyze_feature_importance(features, patterns)
        correlation_matrix = self._calculate_correlation_matrix(features)
        
        # Clustering analysis
        clusters = self._perform_clustering(patterns, features)
        
        # Anomaly detection
        anomalies = self._detect_anomalies(patterns, features)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(patterns, clusters, performance_summary)
        
        analysis = PatternAnalysis(
            total_patterns=len(patterns),
            pattern_distribution=pattern_distribution,
            regime_distribution=regime_distribution,
            performance_summary=performance_summary,
            clusters=clusters,
            feature_importance=feature_importance,
            correlation_matrix=correlation_matrix,
            anomalies=anomalies,
            recommendations=recommendations
        )
        
        logger.info("Pattern analysis completed successfully")
        return analysis
    
    def _prepare_features(self, patterns: List[StrategyPattern]) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and labels from patterns"""
        features_list = []
        labels_list = []
        
        for pattern in patterns:
            # Extract state features
            state_features = pattern.state_features
            
            # Pad or truncate to standard dimension
            if len(state_features) < self.feature_dim:
                padded_features = np.zeros(self.feature_dim)
                padded_features[:len(state_features)] = state_features
                state_features = padded_features
            else:
                state_features = state_features[:self.feature_dim]
            
            # Add derived features
            derived_features = self._extract_derived_features(pattern)
            combined_features = np.concatenate([state_features, derived_features])
            
            features_list.append(combined_features)
            labels_list.append(pattern.pattern_type)
        
        features = np.array(features_list)
        labels = np.array(labels_list)
        
        # Normalize features
        features = self.scaler.fit_transform(features)
        
        return features, labels
    
    def _extract_derived_features(self, pattern: StrategyPattern) -> np.ndarray:
        """Extract additional derived features from pattern"""
        derived = []
        
        # Reward-based features
        derived.extend([
            pattern.reward,
            pattern.confidence,
            np.log1p(abs(pattern.reward)) * np.sign(pattern.reward)
        ])
        
        # Entry condition features
        entry_conditions = pattern.entry_conditions
        derived.extend([
            entry_conditions.get('confidence', 0.5),
            entry_conditions.get('volatility', 0.0),
            1.0 if entry_conditions.get('side') == 'buy' else 0.0
        ])
        
        # Exit condition features
        exit_conditions = pattern.exit_conditions
        derived.extend([
            exit_conditions.get('pnl_percent', 0.0) / 100.0,
            np.log1p(exit_conditions.get('duration', 1.0)),
            1.0 if exit_conditions.get('pnl', 0) > 0 else 0.0
        ])
        
        # Performance-based features
        performance = pattern.performance_metrics
        derived.extend([
            performance.get('sharpeRatio', 0.0),
            performance.get('winRate', 0.5),
            performance.get('maxDrawdown', 0.1),
            performance.get('profitFactor', 1.0)
        ])
        
        # Temporal features
        hour = pattern.timestamp.hour
        day_of_week = pattern.timestamp.weekday()
        derived.extend([
            np.sin(2 * np.pi * hour / 24),
            np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * day_of_week / 7),
            np.cos(2 * np.pi * day_of_week / 7)
        ])
        
        return np.array(derived)
    
    def _calculate_pattern_distribution(self, patterns: List[StrategyPattern]) -> Dict[str, int]:
        """Calculate distribution of pattern types"""
        distribution = {}
        for pattern in patterns:
            pattern_type = pattern.pattern_type
            distribution[pattern_type] = distribution.get(pattern_type, 0) + 1
        return distribution
    
    def _calculate_regime_distribution(self, patterns: List[StrategyPattern]) -> Dict[str, int]:
        """Calculate distribution of market regimes"""
        distribution = {}
        for pattern in patterns:
            regime = pattern.market_regime
            distribution[regime] = distribution.get(regime, 0) + 1
        return distribution
    
    def _calculate_performance_summary(self, patterns: List[StrategyPattern]) -> Dict[str, float]:
        """Calculate summary statistics of pattern performance"""
        rewards = [p.reward for p in patterns]
        confidences = [p.confidence for p in patterns]
        
        return {
            'mean_reward': np.mean(rewards),
            'std_reward': np.std(rewards),
            'median_reward': np.median(rewards),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'mean_confidence': np.mean(confidences),
            'positive_reward_ratio': np.mean([r > 0 for r in rewards]),
            'high_confidence_ratio': np.mean([c > 0.7 for c in confidences])
        }
    
    def _analyze_feature_importance(self, features: np.ndarray, patterns: List[StrategyPattern]) -> Dict[str, float]:
        """Analyze importance of different features"""
        try:
            from sklearn.ensemble import RandomForestRegressor
            
            # Use rewards as target variable
            targets = np.array([p.reward for p in patterns])
            
            # Train random forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(features, targets)
            
            # Get feature importances
            importances = rf.feature_importances_
            
            # Create feature names
            feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Sort by importance
            sorted_indices = np.argsort(importances)[::-1]
            
            importance_dict = {}
            for i, idx in enumerate(sorted_indices[:20]):  # Top 20 features
                importance_dict[feature_names[idx]] = float(importances[idx])
            
            return importance_dict
            
        except Exception as e:
            logger.warning(f"Feature importance analysis failed: {e}")
            return {}
    
    def _calculate_correlation_matrix(self, features: np.ndarray) -> np.ndarray:
        """Calculate correlation matrix of features"""
        try:
            correlation_matrix = np.corrcoef(features.T)
            return correlation_matrix
        except Exception as e:
            logger.warning(f"Correlation matrix calculation failed: {e}")
            return np.eye(features.shape[1])
    
    def _perform_clustering(self, patterns: List[StrategyPattern], features: np.ndarray) -> List[PatternCluster]:
        """Perform clustering analysis on patterns"""
        clusters = []
        
        try:
            if self.clustering_method == 'kmeans':
                clusterer = KMeans(n_clusters=self.n_clusters, random_state=42)
            elif self.clustering_method == 'dbscan':
                clusterer = DBSCAN(eps=0.5, min_samples=5)
            else:
                logger.warning(f"Unknown clustering method: {self.clustering_method}")
                return clusters
            
            cluster_labels = clusterer.fit_predict(features)
            unique_labels = np.unique(cluster_labels)
            
            for label in unique_labels:
                if label == -1:  # Noise cluster in DBSCAN
                    continue
                
                # Get patterns in this cluster
                cluster_patterns = [patterns[i] for i, l in enumerate(cluster_labels) if l == label]
                
                if not cluster_patterns:
                    continue
                
                # Calculate cluster characteristics
                cluster_features = features[cluster_labels == label]
                centroid = np.mean(cluster_features, axis=0)
                
                # Analyze cluster patterns
                characteristics = self._analyze_cluster_characteristics(cluster_patterns)
                performance_stats = self._calculate_cluster_performance(cluster_patterns)
                
                # Calculate confidence score for cluster
                confidence_scores = [p.confidence for p in cluster_patterns]
                cluster_confidence = np.mean(confidence_scores)
                
                # Determine dominant pattern type and market regime
                pattern_types = [p.pattern_type for p in cluster_patterns]
                market_regimes = [p.market_regime for p in cluster_patterns]
                
                dominant_pattern = max(set(pattern_types), key=pattern_types.count)
                dominant_regime = max(set(market_regimes), key=market_regimes.count)
                
                cluster = PatternCluster(
                    cluster_id=int(label),
                    pattern_type=dominant_pattern,
                    market_regime=dominant_regime,
                    patterns=cluster_patterns,
                    centroid=centroid,
                    characteristics=characteristics,
                    performance_stats=performance_stats,
                    confidence_score=cluster_confidence
                )
                
                clusters.append(cluster)
            
            logger.info(f"Created {len(clusters)} pattern clusters")
            
        except Exception as e:
            logger.error(f"Clustering analysis failed: {e}")
        
        return clusters
    
    def _analyze_cluster_characteristics(self, patterns: List[StrategyPattern]) -> Dict[str, Any]:
        """Analyze characteristics of a pattern cluster"""
        characteristics = {}
        
        # Pattern type distribution
        pattern_types = [p.pattern_type for p in patterns]
        characteristics['pattern_type_distribution'] = {
            pt: pattern_types.count(pt) / len(pattern_types) 
            for pt in set(pattern_types)
        }
        
        # Market regime distribution
        regimes = [p.market_regime for p in patterns]
        characteristics['regime_distribution'] = {
            regime: regimes.count(regime) / len(regimes) 
            for regime in set(regimes)
        }
        
        # Action distribution
        actions = [p.action_taken for p in patterns]
        characteristics['action_distribution'] = {
            action: actions.count(action) / len(actions) 
            for action in set(actions)
        }
        
        # Temporal characteristics
        hours = [p.timestamp.hour for p in patterns]
        characteristics['peak_hours'] = sorted(set(hours), key=hours.count, reverse=True)[:3]
        
        days = [p.timestamp.weekday() for p in patterns]
        characteristics['peak_days'] = sorted(set(days), key=days.count, reverse=True)[:3]
        
        # Performance characteristics
        rewards = [p.reward for p in patterns]
        characteristics['reward_stats'] = {
            'mean': np.mean(rewards),
            'std': np.std(rewards),
            'positive_ratio': np.mean([r > 0 for r in rewards])
        }
        
        return characteristics
    
    def _calculate_cluster_performance(self, patterns: List[StrategyPattern]) -> Dict[str, float]:
        """Calculate performance statistics for a cluster"""
        rewards = [p.reward for p in patterns]
        confidences = [p.confidence for p in patterns]
        
        # Calculate Sharpe-like ratio for cluster
        mean_reward = np.mean(rewards)
        std_reward = np.std(rewards) if len(rewards) > 1 else 1.0
        sharpe_ratio = mean_reward / max(std_reward, 0.001)
        
        return {
            'mean_reward': mean_reward,
            'volatility': std_reward,
            'sharpe_ratio': sharpe_ratio,
            'success_rate': np.mean([r > 0 for r in rewards]),
            'mean_confidence': np.mean(confidences),
            'max_reward': np.max(rewards),
            'min_reward': np.min(rewards),
            'pattern_count': len(patterns)
        }
    
    def _detect_anomalies(self, patterns: List[StrategyPattern], features: np.ndarray) -> List[StrategyPattern]:
        """Detect anomalous patterns using isolation forest"""
        anomalies = []
        
        try:
            from sklearn.ensemble import IsolationForest
            
            # Fit isolation forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(features)
            
            # Extract anomalous patterns
            for i, label in enumerate(anomaly_labels):
                if label == -1:  # Anomaly
                    anomalies.append(patterns[i])
            
            logger.info(f"Detected {len(anomalies)} anomalous patterns")
            
        except Exception as e:
            logger.warning(f"Anomaly detection failed: {e}")
        
        return anomalies
    
    def _generate_recommendations(self, 
                                patterns: List[StrategyPattern],
                                clusters: List[PatternCluster],
                                performance_summary: Dict[str, float]) -> List[str]:
        """Generate actionable recommendations based on pattern analysis"""
        recommendations = []
        
        # Pattern diversity recommendations
        pattern_types = set(p.pattern_type for p in patterns)
        if len(pattern_types) < 3:
            recommendations.append(
                "Consider diversifying pattern types to improve robustness"
            )
        
        # Performance recommendations
        if performance_summary['positive_reward_ratio'] < 0.6:
            recommendations.append(
                "Low success rate detected - consider filtering patterns by confidence"
            )
        
        if performance_summary['mean_confidence'] < 0.7:
            recommendations.append(
                "Average confidence is low - consider stricter pattern selection criteria"
            )
        
        # Cluster-based recommendations
        if clusters:
            best_cluster = max(clusters, key=lambda c: c.performance_stats['sharpe_ratio'])
            recommendations.append(
                f"Focus on {best_cluster.pattern_type} patterns in {best_cluster.market_regime} "
                f"market conditions (best performing cluster)"
            )
            
            # Check for underrepresented but high-performing patterns
            for cluster in clusters:
                if (cluster.performance_stats['pattern_count'] < len(patterns) * 0.1 and
                    cluster.performance_stats['mean_reward'] > performance_summary['mean_reward']):
                    recommendations.append(
                        f"Underrepresented high-performing pattern detected: "
                        f"{cluster.pattern_type} in {cluster.market_regime} conditions"
                    )
        
        # Regime-specific recommendations
        regime_performance = {}
        for pattern in patterns:
            regime = pattern.market_regime
            if regime not in regime_performance:
                regime_performance[regime] = []
            regime_performance[regime].append(pattern.reward)
        
        for regime, rewards in regime_performance.items():
            mean_reward = np.mean(rewards)
            if mean_reward > performance_summary['mean_reward'] * 1.2:
                recommendations.append(
                    f"Strong performance in {regime} market conditions - "
                    f"consider increasing allocation"
                )
        
        return recommendations
    
    def _empty_analysis(self) -> PatternAnalysis:
        """Return empty analysis when no patterns are provided"""
        return PatternAnalysis(
            total_patterns=0,
            pattern_distribution={},
            regime_distribution={},
            performance_summary={},
            clusters=[],
            feature_importance={},
            correlation_matrix=np.array([]),
            anomalies=[],
            recommendations=["No patterns available for analysis"]
        )
    
    def visualize_patterns(self, analysis: PatternAnalysis, save_path: Optional[str] = None):
        """Create visualizations of pattern analysis"""
        if analysis.total_patterns == 0:
            logger.warning("No patterns to visualize")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Trading Pattern Analysis', fontsize=16)
        
        # Pattern type distribution
        if analysis.pattern_distribution:
            axes[0, 0].pie(
                analysis.pattern_distribution.values(),
                labels=analysis.pattern_distribution.keys(),
                autopct='%1.1f%%'
            )
            axes[0, 0].set_title('Pattern Type Distribution')
        
        # Market regime distribution
        if analysis.regime_distribution:
            axes[0, 1].pie(
                analysis.regime_distribution.values(),
                labels=analysis.regime_distribution.keys(),
                autopct='%1.1f%%'
            )
            axes[0, 1].set_title('Market Regime Distribution')
        
        # Performance summary
        if analysis.performance_summary:
            perf_keys = list(analysis.performance_summary.keys())[:6]
            perf_values = [analysis.performance_summary[k] for k in perf_keys]
            axes[0, 2].bar(range(len(perf_keys)), perf_values)
            axes[0, 2].set_xticks(range(len(perf_keys)))
            axes[0, 2].set_xticklabels(perf_keys, rotation=45, ha='right')
            axes[0, 2].set_title('Performance Summary')
        
        # Cluster performance
        if analysis.clusters:
            cluster_rewards = [c.performance_stats['mean_reward'] for c in analysis.clusters]
            cluster_labels = [f"C{c.cluster_id}" for c in analysis.clusters]
            axes[1, 0].bar(cluster_labels, cluster_rewards)
            axes[1, 0].set_title('Cluster Performance')
            axes[1, 0].tick_params(axis='x', rotation=45)
        
        # Feature importance (top 10)
        if analysis.feature_importance:
            features = list(analysis.feature_importance.keys())[:10]
            importances = [analysis.feature_importance[f] for f in features]
            axes[1, 1].barh(range(len(features)), importances)
            axes[1, 1].set_yticks(range(len(features)))
            axes[1, 1].set_yticklabels(features)
            axes[1, 1].set_title('Top Feature Importance')
        
        # Correlation heatmap (subset)
        if analysis.correlation_matrix.size > 0:
            subset_size = min(15, analysis.correlation_matrix.shape[0])
            corr_subset = analysis.correlation_matrix[:subset_size, :subset_size]
            im = axes[1, 2].imshow(corr_subset, cmap='coolwarm', vmin=-1, vmax=1)
            axes[1, 2].set_title('Feature Correlation Matrix')
            plt.colorbar(im, ax=axes[1, 2])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def export_analysis(self, analysis: PatternAnalysis, export_path: str):
        """Export analysis results to file"""
        try:
            # Convert analysis to dict for JSON serialization
            analysis_dict = asdict(analysis)
            
            # Handle numpy arrays
            if analysis.correlation_matrix.size > 0:
                analysis_dict['correlation_matrix'] = analysis.correlation_matrix.tolist()
            else:
                analysis_dict['correlation_matrix'] = []
            
            # Convert cluster centroids
            for i, cluster in enumerate(analysis_dict['clusters']):
                if 'centroid' in cluster:
                    cluster['centroid'] = cluster['centroid'].tolist()
                # Remove pattern objects (too complex for JSON)
                cluster['patterns'] = len(cluster['patterns'])
            
            # Remove anomaly pattern objects
            analysis_dict['anomalies'] = len(analysis.anomalies)
            
            with open(export_path, 'w') as f:
                json.dump(analysis_dict, f, indent=2, default=str)
            
            logger.info(f"Analysis exported to {export_path}")
            
        except Exception as e:
            logger.error(f"Failed to export analysis: {e}")
    
    def get_pattern_recommendations_for_rl(self, analysis: PatternAnalysis) -> Dict[str, Any]:
        """Get specific recommendations for RL pre-training"""
        recommendations = {
            'high_quality_patterns': [],
            'balanced_sampling_weights': {},
            'feature_selection': [],
            'training_strategies': []
        }
        
        # Identify high-quality patterns
        if analysis.clusters:
            for cluster in analysis.clusters:
                if (cluster.performance_stats['sharpe_ratio'] > 1.0 and
                    cluster.confidence_score > 0.7):
                    recommendations['high_quality_patterns'].append({
                        'pattern_type': cluster.pattern_type,
                        'market_regime': cluster.market_regime,
                        'expected_performance': cluster.performance_stats['mean_reward']
                    })
        
        # Calculate balanced sampling weights
        total_patterns = analysis.total_patterns
        for pattern_type, count in analysis.pattern_distribution.items():
            weight = total_patterns / (len(analysis.pattern_distribution) * count)
            recommendations['balanced_sampling_weights'][pattern_type] = weight
        
        # Feature selection recommendations
        if analysis.feature_importance:
            top_features = list(analysis.feature_importance.keys())[:20]
            recommendations['feature_selection'] = top_features
        
        # Training strategy recommendations
        if analysis.performance_summary['positive_reward_ratio'] > 0.7:
            recommendations['training_strategies'].append("Use reward shaping to emphasize positive patterns")
        
        if len(analysis.pattern_distribution) >= 4:
            recommendations['training_strategies'].append("Implement multi-task learning across pattern types")
        
        if analysis.performance_summary['mean_confidence'] > 0.8:
            recommendations['training_strategies'].append("Use confidence-weighted loss function")
        
        return recommendations

# Example usage
def main():
    """Example usage of PatternAnalyzer"""
    from composer_extractor import ComposerExtractor
    
    # Load patterns (assuming they exist)
    extractor = ComposerExtractor()
    patterns = extractor.load_patterns(min_confidence=0.5)
    
    if not patterns:
        logger.warning("No patterns found for analysis")
        return
    
    # Analyze patterns
    analyzer = PatternAnalyzer()
    analysis = analyzer.analyze_patterns(patterns)
    
    # Print summary
    print(f"Total patterns analyzed: {analysis.total_patterns}")
    print(f"Pattern types: {list(analysis.pattern_distribution.keys())}")
    print(f"Market regimes: {list(analysis.regime_distribution.keys())}")
    print(f"Number of clusters: {len(analysis.clusters)}")
    print(f"Anomalies detected: {len(analysis.anomalies)}")
    
    # Show recommendations
    print("\nRecommendations:")
    for rec in analysis.recommendations:
        print(f"- {rec}")
    
    # Get RL-specific recommendations
    rl_recommendations = analyzer.get_pattern_recommendations_for_rl(analysis)
    print("\nRL Pre-training Recommendations:")
    print(json.dumps(rl_recommendations, indent=2))
    
    # Create visualizations
    analyzer.visualize_patterns(analysis, save_path="pattern_analysis.png")
    
    # Export analysis
    analyzer.export_analysis(analysis, "pattern_analysis.json")

if __name__ == "__main__":
    main()