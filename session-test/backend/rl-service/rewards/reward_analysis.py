"""
Reward Analysis and Debugging System

This module provides comprehensive analysis tools for understanding and debugging
the multi-objective reward function, including visualization, sensitivity analysis,
and performance diagnostics.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import logging
import json
from datetime import datetime, timedelta
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

from .reward_components import MarketRegime

logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for reward analysis"""
    
    # Analysis parameters
    window_size: int = 100
    correlation_threshold: float = 0.7
    significance_level: float = 0.05
    
    # Visualization
    figure_size: Tuple[int, int] = (12, 8)
    save_plots: bool = True
    plot_directory: str = "./reward_analysis_plots"
    
    # Sensitivity analysis
    perturbation_range: float = 0.1  # ±10% for sensitivity
    n_perturbations: int = 20
    
    # Performance thresholds (SOW aligned)
    target_sharpe: float = 1.5
    target_weekly_return: float = 0.04
    max_acceptable_drawdown: float = 0.15
    target_win_rate: float = 0.60


class RewardAnalyzer:
    """
    Main reward analysis system
    
    Provides comprehensive analysis of reward function behavior and performance.
    """
    
    def __init__(self, config: Optional[AnalysisConfig] = None):
        self.config = config or AnalysisConfig()
        
        # Data storage
        self.reward_history: List[float] = []
        self.component_history: List[Dict[str, float]] = []
        self.state_history: List[Dict[str, Any]] = []
        self.action_history: List[int] = []
        self.performance_metrics: Dict[str, List[float]] = {}
        
        # Analysis results
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.sensitivity_results: Dict[str, Dict[str, float]] = {}
        self.regime_analysis: Dict[MarketRegime, Dict[str, Any]] = {}
        
        # Setup plotting
        plt.style.use('seaborn-v0_8-darkgrid')
        self._setup_plot_directory()
    
    def _setup_plot_directory(self):
        """Create plot directory if saving plots"""
        if self.config.save_plots:
            import os
            os.makedirs(self.config.plot_directory, exist_ok=True)
    
    def record(self,
              reward: float,
              components: Dict[str, float],
              state: Dict[str, Any],
              action: Optional[int] = None):
        """
        Record reward data for analysis
        
        Args:
            reward: Total reward value
            components: Individual component values
            state: Environment state
            action: Action taken
        """
        self.reward_history.append(reward)
        self.component_history.append(components.copy())
        self.state_history.append(state.copy())
        
        if action is not None:
            self.action_history.append(action)
        
        # Update performance metrics
        self._update_performance_metrics(state)
    
    def _update_performance_metrics(self, state: Dict[str, Any]):
        """Update tracked performance metrics"""
        metrics_to_track = [
            'portfolio_value', 'sharpe_ratio', 'drawdown',
            'win_rate', 'volatility', 'total_trades'
        ]
        
        for metric in metrics_to_track:
            if metric in state:
                if metric not in self.performance_metrics:
                    self.performance_metrics[metric] = []
                self.performance_metrics[metric].append(state[metric])
    
    def analyze_components(self) -> Dict[str, Any]:
        """
        Analyze individual reward components
        
        Returns:
            Analysis results including statistics and correlations
        """
        if not self.component_history:
            logger.warning("No component history available for analysis")
            return {}
        
        # Convert to DataFrame
        df = pd.DataFrame(self.component_history)
        
        # Basic statistics
        stats = {
            'mean': df.mean().to_dict(),
            'std': df.std().to_dict(),
            'min': df.min().to_dict(),
            'max': df.max().to_dict(),
            'skew': df.skew().to_dict(),
            'kurtosis': df.kurtosis().to_dict()
        }
        
        # Correlation analysis
        self.correlation_matrix = df.corr()
        
        # Identify highly correlated components
        high_correlations = []
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i + 1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) > self.config.correlation_threshold:
                    high_correlations.append({
                        'component1': self.correlation_matrix.columns[i],
                        'component2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Component contribution analysis
        total_rewards = np.array(self.reward_history)
        contributions = {}
        
        for component in df.columns:
            # Calculate correlation with total reward
            if len(total_rewards) == len(df):
                corr, p_value = stats.pearsonr(df[component], total_rewards)
                contributions[component] = {
                    'correlation': corr,
                    'p_value': p_value,
                    'significant': p_value < self.config.significance_level,
                    'avg_contribution': df[component].mean()
                }
        
        # Temporal analysis
        temporal_patterns = self._analyze_temporal_patterns(df)
        
        return {
            'statistics': stats,
            'correlations': self.correlation_matrix.to_dict(),
            'high_correlations': high_correlations,
            'contributions': contributions,
            'temporal_patterns': temporal_patterns
        }
    
    def _analyze_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze temporal patterns in components"""
        patterns = {}
        
        for component in df.columns:
            series = df[component].values
            
            # Trend analysis
            x = np.arange(len(series))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, series)
            
            # Autocorrelation
            if len(series) > 10:
                autocorr = pd.Series(series).autocorr(lag=1)
            else:
                autocorr = 0
            
            patterns[component] = {
                'trend_slope': slope,
                'trend_r2': r_value ** 2,
                'trend_significant': p_value < self.config.significance_level,
                'autocorrelation': autocorr
            }
        
        return patterns
    
    def sensitivity_analysis(self,
                           base_state: Dict[str, Any],
                           reward_function: callable) -> Dict[str, Dict[str, float]]:
        """
        Perform sensitivity analysis on reward function
        
        Args:
            base_state: Base state for analysis
            reward_function: Reward calculation function
            
        Returns:
            Sensitivity results for each component
        """
        results = {}
        base_reward, base_components = reward_function(base_state)
        
        # Analyze each state variable
        for key, value in base_state.items():
            if isinstance(value, (int, float)):
                sensitivities = []
                
                # Perturb the variable
                for i in range(self.config.n_perturbations):
                    perturbation = np.linspace(
                        -self.config.perturbation_range,
                        self.config.perturbation_range,
                        self.config.n_perturbations
                    )[i]
                    
                    perturbed_state = base_state.copy()
                    perturbed_state[key] = value * (1 + perturbation)
                    
                    try:
                        perturbed_reward, _ = reward_function(perturbed_state)
                        sensitivity = (perturbed_reward - base_reward) / (perturbation + 1e-10)
                        sensitivities.append(sensitivity)
                    except:
                        sensitivities.append(0)
                
                results[key] = {
                    'mean_sensitivity': np.mean(sensitivities),
                    'std_sensitivity': np.std(sensitivities),
                    'max_sensitivity': np.max(np.abs(sensitivities))
                }
        
        self.sensitivity_results = results
        return results
    
    def analyze_by_regime(self) -> Dict[MarketRegime, Dict[str, Any]]:
        """
        Analyze reward behavior by market regime
        
        Returns:
            Regime-specific analysis results
        """
        if not self.state_history:
            return {}
        
        # Group data by regime
        regime_data = {regime: {'rewards': [], 'components': []} 
                      for regime in MarketRegime}
        
        for i, state in enumerate(self.state_history):
            if 'market_regime' in state and i < len(self.reward_history):
                regime = MarketRegime[state['market_regime']]
                regime_data[regime]['rewards'].append(self.reward_history[i])
                
                if i < len(self.component_history):
                    regime_data[regime]['components'].append(self.component_history[i])
        
        # Analyze each regime
        for regime in MarketRegime:
            if regime_data[regime]['rewards']:
                rewards = np.array(regime_data[regime]['rewards'])
                
                analysis = {
                    'count': len(rewards),
                    'mean_reward': np.mean(rewards),
                    'std_reward': np.std(rewards),
                    'sharpe': np.mean(rewards) / (np.std(rewards) + 1e-10) * np.sqrt(252),
                    'positive_rate': np.sum(rewards > 0) / len(rewards)
                }
                
                # Component analysis
                if regime_data[regime]['components']:
                    comp_df = pd.DataFrame(regime_data[regime]['components'])
                    analysis['component_means'] = comp_df.mean().to_dict()
                    analysis['component_stds'] = comp_df.std().to_dict()
                
                self.regime_analysis[regime] = analysis
        
        return self.regime_analysis
    
    def plot_reward_breakdown(self, window: Optional[int] = None):
        """
        Plot reward component breakdown over time
        
        Args:
            window: Rolling window for smoothing
        """
        if not self.component_history:
            logger.warning("No component history to plot")
            return
        
        window = window or self.config.window_size
        df = pd.DataFrame(self.component_history)
        
        fig, axes = plt.subplots(2, 1, figsize=self.config.figure_size)
        
        # Plot 1: Stacked area chart of components
        df_smooth = df.rolling(window=window, min_periods=1).mean()
        df_smooth.plot.area(ax=axes[0], alpha=0.7)
        axes[0].set_title('Reward Components Breakdown (Smoothed)')
        axes[0].set_xlabel('Step')
        axes[0].set_ylabel('Component Value')
        axes[0].legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        # Plot 2: Total reward with components
        axes[1].plot(self.reward_history, label='Total Reward', linewidth=2)
        axes[1].plot(df_smooth.sum(axis=1), label='Sum of Components', 
                    linestyle='--', alpha=0.7)
        axes[1].set_title('Total Reward vs Component Sum')
        axes[1].set_xlabel('Step')
        axes[1].set_ylabel('Reward')
        axes[1].legend()
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_directory}/reward_breakdown.png")
        plt.show()
    
    def plot_correlation_heatmap(self):
        """Plot correlation heatmap of reward components"""
        if self.correlation_matrix is None:
            self.analyze_components()
        
        if self.correlation_matrix is None:
            logger.warning("No correlation matrix available")
            return
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(self.correlation_matrix, annot=True, cmap='coolwarm',
                   center=0, vmin=-1, vmax=1, square=True)
        plt.title('Reward Component Correlation Matrix')
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_directory}/correlation_heatmap.png")
        plt.show()
    
    def plot_performance_alignment(self):
        """Plot alignment with SOW performance targets"""
        if not self.performance_metrics:
            logger.warning("No performance metrics available")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
        
        # Sharpe ratio
        if 'sharpe_ratio' in self.performance_metrics:
            ax = axes[0, 0]
            sharpe = self.performance_metrics['sharpe_ratio']
            ax.plot(sharpe, label='Actual')
            ax.axhline(self.config.target_sharpe, color='r', 
                      linestyle='--', label='Target (1.5)')
            ax.set_title('Sharpe Ratio')
            ax.set_ylabel('Sharpe')
            ax.legend()
        
        # Drawdown
        if 'drawdown' in self.performance_metrics:
            ax = axes[0, 1]
            dd = self.performance_metrics['drawdown']
            ax.plot(dd, label='Actual')
            ax.axhline(-self.config.max_acceptable_drawdown, color='r',
                      linestyle='--', label='Max Acceptable (-15%)')
            ax.set_title('Drawdown')
            ax.set_ylabel('Drawdown')
            ax.legend()
        
        # Win rate
        if 'win_rate' in self.performance_metrics:
            ax = axes[1, 0]
            wr = self.performance_metrics['win_rate']
            ax.plot(wr, label='Actual')
            ax.axhline(self.config.target_win_rate, color='r',
                      linestyle='--', label='Target (60%)')
            ax.set_title('Win Rate')
            ax.set_ylabel('Win Rate')
            ax.legend()
        
        # Portfolio value growth
        if 'portfolio_value' in self.performance_metrics:
            ax = axes[1, 1]
            pv = self.performance_metrics['portfolio_value']
            returns = pd.Series(pv).pct_change().fillna(0)
            cumulative = (1 + returns).cumprod()
            
            # Calculate weekly return trajectory
            weeks = len(pv) / (252 / 52)  # Approximate weeks
            target_growth = (1 + self.config.target_weekly_return) ** weeks
            
            ax.plot(cumulative, label='Actual')
            ax.plot(np.linspace(1, target_growth, len(pv)), 
                   'r--', label='Target (4% weekly)')
            ax.set_title('Cumulative Returns')
            ax.set_ylabel('Cumulative Return')
            ax.legend()
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_directory}/performance_alignment.png")
        plt.show()
    
    def plot_sensitivity_analysis(self):
        """Plot sensitivity analysis results"""
        if not self.sensitivity_results:
            logger.warning("No sensitivity results available")
            return
        
        # Prepare data
        variables = list(self.sensitivity_results.keys())
        sensitivities = [self.sensitivity_results[v]['mean_sensitivity'] 
                        for v in variables]
        std_sensitivities = [self.sensitivity_results[v]['std_sensitivity']
                            for v in variables]
        
        # Sort by absolute sensitivity
        sorted_idx = np.argsort(np.abs(sensitivities))[::-1]
        variables = [variables[i] for i in sorted_idx]
        sensitivities = [sensitivities[i] for i in sorted_idx]
        std_sensitivities = [std_sensitivities[i] for i in sorted_idx]
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(variables) * 0.3)))
        
        y_pos = np.arange(len(variables))
        ax.barh(y_pos, sensitivities, xerr=std_sensitivities, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(variables)
        ax.set_xlabel('Sensitivity')
        ax.set_title('Reward Function Sensitivity Analysis')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_directory}/sensitivity_analysis.png")
        plt.show()
    
    def plot_regime_comparison(self):
        """Plot reward behavior comparison across market regimes"""
        if not self.regime_analysis:
            self.analyze_by_regime()
        
        if not self.regime_analysis:
            logger.warning("No regime analysis available")
            return
        
        # Prepare data
        regimes = list(self.regime_analysis.keys())
        mean_rewards = [self.regime_analysis[r].get('mean_reward', 0) for r in regimes]
        std_rewards = [self.regime_analysis[r].get('std_reward', 0) for r in regimes]
        counts = [self.regime_analysis[r].get('count', 0) for r in regimes]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Mean rewards by regime
        ax = axes[0]
        x = np.arange(len(regimes))
        ax.bar(x, mean_rewards, yerr=std_rewards, alpha=0.7, capsize=5)
        ax.set_xticks(x)
        ax.set_xticklabels([r.value for r in regimes], rotation=45)
        ax.set_ylabel('Mean Reward')
        ax.set_title('Average Reward by Market Regime')
        ax.grid(True, alpha=0.3)
        
        # Sample distribution
        ax = axes[1]
        ax.pie(counts, labels=[r.value for r in regimes], autopct='%1.1f%%')
        ax.set_title('Regime Distribution in Data')
        
        # Component means by regime
        ax = axes[2]
        component_data = {}
        for regime in regimes:
            if 'component_means' in self.regime_analysis[regime]:
                for comp, value in self.regime_analysis[regime]['component_means'].items():
                    if comp not in component_data:
                        component_data[comp] = []
                    component_data[comp].append(value)
        
        if component_data:
            comp_df = pd.DataFrame(component_data, index=[r.value for r in regimes])
            comp_df.plot(kind='bar', ax=ax, alpha=0.7)
            ax.set_xlabel('Market Regime')
            ax.set_ylabel('Component Value')
            ax.set_title('Component Values by Regime')
            ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if self.config.save_plots:
            plt.savefig(f"{self.config.plot_directory}/regime_comparison.png")
        plt.show()
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive analysis report
        
        Returns:
            Complete analysis report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'data_points': len(self.reward_history),
            'episodes_analyzed': len(self.reward_history) // 1000 if self.reward_history else 0
        }
        
        # Component analysis
        component_analysis = self.analyze_components()
        report['component_analysis'] = component_analysis
        
        # Regime analysis
        regime_analysis = self.analyze_by_regime()
        report['regime_analysis'] = {k.value: v for k, v in regime_analysis.items()}
        
        # Performance alignment
        if self.performance_metrics:
            alignment = {}
            
            if 'sharpe_ratio' in self.performance_metrics:
                current_sharpe = np.mean(self.performance_metrics['sharpe_ratio'][-100:])
                alignment['sharpe_ratio'] = {
                    'current': current_sharpe,
                    'target': self.config.target_sharpe,
                    'achieved': current_sharpe >= self.config.target_sharpe
                }
            
            if 'drawdown' in self.performance_metrics:
                max_dd = np.min(self.performance_metrics['drawdown'])
                alignment['max_drawdown'] = {
                    'current': max_dd,
                    'limit': -self.config.max_acceptable_drawdown,
                    'within_limit': max_dd > -self.config.max_acceptable_drawdown
                }
            
            if 'win_rate' in self.performance_metrics:
                current_wr = np.mean(self.performance_metrics['win_rate'][-100:])
                alignment['win_rate'] = {
                    'current': current_wr,
                    'target': self.config.target_win_rate,
                    'achieved': current_wr >= self.config.target_win_rate
                }
            
            report['target_alignment'] = alignment
        
        # Key insights
        insights = self._generate_insights(component_analysis, regime_analysis)
        report['insights'] = insights
        
        # Recommendations
        recommendations = self._generate_recommendations(report)
        report['recommendations'] = recommendations
        
        return report
    
    def _generate_insights(self,
                          component_analysis: Dict[str, Any],
                          regime_analysis: Dict[MarketRegime, Dict[str, Any]]) -> List[str]:
        """Generate key insights from analysis"""
        insights = []
        
        # Component insights
        if 'contributions' in component_analysis:
            significant_components = [
                k for k, v in component_analysis['contributions'].items()
                if v.get('significant', False)
            ]
            if significant_components:
                insights.append(f"Significant components: {', '.join(significant_components)}")
        
        # Correlation insights
        if 'high_correlations' in component_analysis:
            for corr in component_analysis['high_correlations']:
                insights.append(
                    f"High correlation ({corr['correlation']:.2f}) between "
                    f"{corr['component1']} and {corr['component2']}"
                )
        
        # Regime insights
        if regime_analysis:
            best_regime = max(regime_analysis.items(), 
                            key=lambda x: x[1].get('mean_reward', float('-inf')))
            worst_regime = min(regime_analysis.items(),
                             key=lambda x: x[1].get('mean_reward', float('inf')))
            
            insights.append(f"Best performing regime: {best_regime[0].value}")
            insights.append(f"Worst performing regime: {worst_regime[0].value}")
        
        return insights
    
    def _generate_recommendations(self, report: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on analysis"""
        recommendations = []
        
        # Target alignment recommendations
        if 'target_alignment' in report:
            alignment = report['target_alignment']
            
            if 'sharpe_ratio' in alignment and not alignment['sharpe_ratio']['achieved']:
                recommendations.append(
                    "Increase risk-adjusted returns weight to improve Sharpe ratio"
                )
            
            if 'max_drawdown' in alignment and not alignment['max_drawdown']['within_limit']:
                recommendations.append(
                    "Strengthen drawdown penalties to reduce maximum drawdown"
                )
            
            if 'win_rate' in alignment and not alignment['win_rate']['achieved']:
                recommendations.append(
                    "Increase consistency bonus to improve win rate"
                )
        
        # Component recommendations
        if 'component_analysis' in report:
            comp_stats = report['component_analysis'].get('statistics', {})
            if 'std' in comp_stats:
                high_variance_components = [
                    k for k, v in comp_stats['std'].items() if v > 0.5
                ]
                if high_variance_components:
                    recommendations.append(
                        f"Consider stabilizing high-variance components: "
                        f"{', '.join(high_variance_components)}"
                    )
        
        return recommendations
    
    def save_report(self, filepath: str):
        """Save analysis report to file"""
        report = self.generate_report()
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Analysis report saved to {filepath}")


def create_analyzer(config: Optional[AnalysisConfig] = None) -> RewardAnalyzer:
    """
    Factory function to create reward analyzer
    
    Args:
        config: Analysis configuration
        
    Returns:
        RewardAnalyzer instance
    """
    return RewardAnalyzer(config)