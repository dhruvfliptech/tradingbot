"""
Correlation-Based Portfolio Optimization with RL Integration

This module implements portfolio optimization using correlation analysis and integrates
with reinforcement learning for dynamic portfolio management:
- Mean-variance optimization with correlation constraints
- Risk parity with correlation adjustments
- Hierarchical risk parity using correlation clustering
- Dynamic rebalancing based on regime changes
- RL integration for adaptive portfolio management
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy.optimize import minimize, LinearConstraint, NonlinearConstraint
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform
import cvxpy as cp
import logging

# Import other modules
from .correlation_engine import CorrelationEngine, CorrelationConfig, CorrelationResult
from .regime_detection import RegimeDetector, RegimeConfig, RegimeState
from .correlation_risk import CorrelationRiskManager, RiskConfig, PortfolioRisk
from .network_analysis import CorrelationNetworkAnalyzer, NetworkConfig

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class OptimizationConfig:
    """Configuration for portfolio optimization"""
    optimization_method: str = 'hrp'  # 'mvo', 'risk_parity', 'hrp', 'max_diversification'
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    min_weight: float = 0.01  # Minimum position weight
    max_weight: float = 0.2  # Maximum position weight
    target_volatility: Optional[float] = 0.15  # Annual target volatility
    risk_free_rate: float = 0.02  # Annual risk-free rate
    transaction_cost: float = 0.001  # 10 bps transaction cost
    correlation_penalty: float = 0.1  # Penalty for high correlations
    regime_adaptive: bool = True  # Adapt to regime changes
    use_rl_signals: bool = True  # Use RL model signals
    max_turnover: float = 0.5  # Maximum portfolio turnover


@dataclass
class OptimizationResult:
    """Container for optimization results"""
    timestamp: datetime
    method: str
    weights: Dict[str, float]
    expected_return: float
    expected_volatility: float
    sharpe_ratio: float
    diversification_ratio: float
    effective_n_assets: float
    turnover: float
    transaction_costs: float
    regime: Optional[str]
    metadata: Dict


@dataclass
class RLSignal:
    """Signal from RL model for portfolio optimization"""
    timestamp: datetime
    action: str  # 'increase_risk', 'decrease_risk', 'rebalance', 'hold'
    confidence: float
    target_assets: List[str]
    suggested_weights: Optional[Dict[str, float]]
    risk_level: float  # 0-1 scale
    market_view: Dict[str, float]  # Expected returns by asset


class CorrelationBasedOptimizer:
    """
    Portfolio optimizer using correlation analysis and RL integration
    """
    
    def __init__(
        self,
        config: Optional[OptimizationConfig] = None,
        correlation_engine: Optional[CorrelationEngine] = None,
        regime_detector: Optional[RegimeDetector] = None,
        risk_manager: Optional[CorrelationRiskManager] = None,
        network_analyzer: Optional[CorrelationNetworkAnalyzer] = None
    ):
        """
        Initialize portfolio optimizer
        
        Args:
            config: Optimization configuration
            correlation_engine: Correlation calculation engine
            regime_detector: Regime detection system
            risk_manager: Risk management system
            network_analyzer: Network analysis system
        """
        self.config = config or OptimizationConfig()
        
        # Initialize components
        self.correlation_engine = correlation_engine or CorrelationEngine()
        self.regime_detector = regime_detector or RegimeDetector()
        self.risk_manager = risk_manager or CorrelationRiskManager()
        self.network_analyzer = network_analyzer or CorrelationNetworkAnalyzer()
        
        # Portfolio state
        self.current_weights: Dict[str, float] = {}
        self.optimization_history: List[OptimizationResult] = []
        self.rl_signals: List[RLSignal] = []
        
        # Market data
        self.returns_data: Optional[pd.DataFrame] = None
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.expected_returns: Optional[pd.Series] = None
        
    def initialize_data(
        self,
        returns_data: pd.DataFrame,
        initial_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize optimizer with market data
        
        Args:
            returns_data: Historical returns for assets
            initial_weights: Initial portfolio weights
        """
        self.returns_data = returns_data
        
        # Add data to correlation engine
        for asset in returns_data.columns:
            prices = (1 + returns_data[asset]).cumprod()
            self.correlation_engine.add_price_data(asset, prices)
            
        # Calculate initial correlation matrix
        self.correlation_matrix = self.correlation_engine.calculate_correlation_matrix(
            method='pearson',
            window=60
        ).correlation_matrix
        
        # Build network
        self.network_analyzer.build_network(self.correlation_matrix)
        
        # Detect initial regime
        current_regime = self.regime_detector.detect_correlation_regime(
            self.correlation_matrix,
            datetime.now()
        )
        
        # Set initial weights
        if initial_weights:
            self.current_weights = initial_weights
        else:
            # Equal weight initialization
            n_assets = len(returns_data.columns)
            self.current_weights = {
                asset: 1/n_assets for asset in returns_data.columns
            }
            
        # Calculate expected returns
        self._estimate_expected_returns()
        
        logger.info(f"Optimizer initialized with {len(returns_data.columns)} assets")
        
    def optimize_portfolio(
        self,
        method: Optional[str] = None,
        constraints: Optional[Dict] = None,
        rl_signal: Optional[RLSignal] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio weights
        
        Args:
            method: Optimization method
            constraints: Additional constraints
            rl_signal: Signal from RL model
            
        Returns:
            OptimizationResult with optimal weights
        """
        method = method or self.config.optimization_method
        timestamp = datetime.now()
        
        # Store RL signal if provided
        if rl_signal:
            self.rl_signals.append(rl_signal)
            
        # Detect current regime
        current_regime = self.regime_detector.detect_correlation_regime(
            self.correlation_matrix,
            timestamp
        )
        
        # Adjust method based on regime if adaptive
        if self.config.regime_adaptive:
            method = self._adapt_method_to_regime(method, current_regime)
            
        # Apply optimization method
        if method == 'mvo':
            weights = self._mean_variance_optimization(constraints, rl_signal)
        elif method == 'risk_parity':
            weights = self._risk_parity_optimization(constraints)
        elif method == 'hrp':
            weights = self._hierarchical_risk_parity()
        elif method == 'max_diversification':
            weights = self._max_diversification_optimization()
        elif method == 'correlation_minimization':
            weights = self._correlation_minimization()
        else:
            raise ValueError(f"Unknown optimization method: {method}")
            
        # Apply RL adjustments if enabled
        if self.config.use_rl_signals and rl_signal:
            weights = self._apply_rl_adjustments(weights, rl_signal)
            
        # Calculate portfolio metrics
        metrics = self._calculate_portfolio_metrics(weights)
        
        # Calculate turnover and transaction costs
        turnover = self._calculate_turnover(weights)
        transaction_costs = turnover * self.config.transaction_cost
        
        # Create result
        result = OptimizationResult(
            timestamp=timestamp,
            method=method,
            weights=weights,
            expected_return=metrics['expected_return'],
            expected_volatility=metrics['volatility'],
            sharpe_ratio=metrics['sharpe_ratio'],
            diversification_ratio=metrics['diversification_ratio'],
            effective_n_assets=metrics['effective_n_assets'],
            turnover=turnover,
            transaction_costs=transaction_costs,
            regime=current_regime.regime_type,
            metadata={
                'regime_confidence': current_regime.confidence,
                'correlation_risk': metrics.get('correlation_risk', 0),
                'network_metrics': self._get_network_metrics()
            }
        )
        
        # Update current weights
        self.current_weights = weights
        self.optimization_history.append(result)
        
        return result
        
    def _mean_variance_optimization(
        self,
        constraints: Optional[Dict],
        rl_signal: Optional[RLSignal]
    ) -> Dict[str, float]:
        """
        Mean-variance optimization with correlation constraints
        """
        assets = list(self.returns_data.columns)
        n_assets = len(assets)
        
        # Get covariance matrix
        cov_matrix = self.returns_data.cov() * 252  # Annualized
        
        # Expected returns
        if rl_signal and rl_signal.market_view:
            # Use RL market views
            expected_returns = np.array([
                rl_signal.market_view.get(asset, self.expected_returns[asset])
                for asset in assets
            ])
        else:
            expected_returns = self.expected_returns[assets].values * 252
            
        # Setup optimization problem using cvxpy
        weights = cp.Variable(n_assets)
        
        # Objective: Maximize Sharpe ratio (approximation)
        portfolio_return = expected_returns @ weights
        portfolio_variance = cp.quad_form(weights, cov_matrix.values)
        
        # Add correlation penalty
        correlation_penalty = 0
        if self.config.correlation_penalty > 0:
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    correlation_penalty += (
                        self.config.correlation_penalty *
                        self.correlation_matrix.iloc[i, j] *
                        weights[i] * weights[j]
                    )
                    
        objective = cp.Maximize(
            portfolio_return - 0.5 * portfolio_variance - correlation_penalty
        )
        
        # Constraints
        constraints_list = [
            cp.sum(weights) == 1,
            weights >= self.config.min_weight,
            weights <= self.config.max_weight
        ]
        
        # Target volatility constraint
        if self.config.target_volatility:
            constraints_list.append(
                cp.sqrt(portfolio_variance) <= self.config.target_volatility
            )
            
        # Turnover constraint
        if self.config.max_turnover and self.current_weights:
            current = np.array([self.current_weights.get(a, 0) for a in assets])
            constraints_list.append(
                cp.norm(weights - current, 1) <= self.config.max_turnover
            )
            
        # Solve
        problem = cp.Problem(objective, constraints_list)
        problem.solve(solver=cp.OSQP)
        
        if problem.status == 'optimal':
            return dict(zip(assets, weights.value))
        else:
            logger.warning(f"Optimization failed: {problem.status}")
            return self.current_weights
            
    def _risk_parity_optimization(
        self,
        constraints: Optional[Dict]
    ) -> Dict[str, float]:
        """
        Risk parity optimization with correlation adjustments
        """
        assets = list(self.returns_data.columns)
        n_assets = len(assets)
        
        # Covariance matrix
        cov_matrix = self.returns_data.cov().values * 252
        
        def risk_parity_objective(weights):
            # Portfolio volatility
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            # Marginal contributions
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            
            # Risk contributions
            risk_contrib = weights * marginal_contrib
            
            # Objective: minimize variance of risk contributions
            return np.var(risk_contrib)
            
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [
            (self.config.min_weight, self.config.max_weight)
            for _ in range(n_assets)
        ]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons,
            options={'maxiter': 1000}
        )
        
        if result.success:
            return dict(zip(assets, result.x))
        else:
            return dict(zip(assets, x0))
            
    def _hierarchical_risk_parity(self) -> Dict[str, float]:
        """
        Hierarchical Risk Parity (HRP) optimization
        """
        returns = self.returns_data
        
        # Step 1: Tree clustering
        corr_matrix = returns.corr()
        distance_matrix = np.sqrt(2 * (1 - corr_matrix))
        
        # Hierarchical clustering
        linkage_matrix = linkage(
            squareform(distance_matrix),
            method='single'
        )
        
        # Step 2: Quasi-diagonalization
        sorted_indices = self._get_quasi_diagonal_order(
            linkage_matrix,
            len(returns.columns)
        )
        
        # Step 3: Recursive bisection
        weights = self._recursive_bisection(
            returns.cov().values * 252,
            sorted_indices
        )
        
        # Map weights to assets
        assets = list(returns.columns)
        weight_dict = {}
        for i, idx in enumerate(sorted_indices):
            weight_dict[assets[idx]] = weights[i]
            
        return weight_dict
        
    def _max_diversification_optimization(self) -> Dict[str, float]:
        """
        Maximum diversification portfolio
        """
        assets = list(self.returns_data.columns)
        n_assets = len(assets)
        
        # Volatilities and correlation
        volatilities = self.returns_data.std().values * np.sqrt(252)
        corr_matrix = self.correlation_matrix.values
        
        def diversification_ratio(weights):
            # Weighted average volatility
            weighted_vol = np.sum(weights * volatilities)
            
            # Portfolio volatility
            cov_matrix = corr_matrix * np.outer(volatilities, volatilities)
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights)
            
            # Negative for maximization
            return -weighted_vol / portfolio_vol if portfolio_vol > 0 else 0
            
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [
            (self.config.min_weight, self.config.max_weight)
            for _ in range(n_assets)
        ]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            diversification_ratio,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            return dict(zip(assets, result.x))
        else:
            return dict(zip(assets, x0))
            
    def _correlation_minimization(self) -> Dict[str, float]:
        """
        Minimize portfolio correlation risk
        """
        assets = list(self.returns_data.columns)
        n_assets = len(assets)
        
        # Correlation matrix
        corr_matrix = self.correlation_matrix.values
        
        def correlation_objective(weights):
            # Sum of weighted correlations
            correlation_sum = 0
            for i in range(n_assets):
                for j in range(i+1, n_assets):
                    correlation_sum += (
                        weights[i] * weights[j] * abs(corr_matrix[i, j])
                    )
            return correlation_sum
            
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Add minimum return constraint
        expected_returns = self.expected_returns[assets].values * 252
        min_return = 0.05  # 5% minimum annual return
        cons.append({
            'type': 'ineq',
            'fun': lambda w: expected_returns @ w - min_return
        })
        
        # Bounds
        bounds = [
            (self.config.min_weight, self.config.max_weight)
            for _ in range(n_assets)
        ]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            correlation_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            return dict(zip(assets, result.x))
        else:
            return dict(zip(assets, x0))
            
    def _apply_rl_adjustments(
        self,
        weights: Dict[str, float],
        rl_signal: RLSignal
    ) -> Dict[str, float]:
        """
        Apply RL model adjustments to portfolio weights
        """
        adjusted_weights = weights.copy()
        
        # Apply action based on RL signal
        if rl_signal.action == 'increase_risk':
            # Increase weights of high-conviction assets
            if rl_signal.target_assets:
                for asset in rl_signal.target_assets:
                    if asset in adjusted_weights:
                        adjusted_weights[asset] *= (1 + 0.2 * rl_signal.confidence)
                        
        elif rl_signal.action == 'decrease_risk':
            # Move towards equal weight
            equal_weight = 1 / len(adjusted_weights)
            for asset in adjusted_weights:
                current = adjusted_weights[asset]
                adjusted_weights[asset] = (
                    current * (1 - 0.3 * rl_signal.confidence) +
                    equal_weight * 0.3 * rl_signal.confidence
                )
                
        elif rl_signal.action == 'rebalance':
            # Use suggested weights if provided
            if rl_signal.suggested_weights:
                blend_factor = rl_signal.confidence * 0.5
                for asset in adjusted_weights:
                    if asset in rl_signal.suggested_weights:
                        current = adjusted_weights[asset]
                        suggested = rl_signal.suggested_weights[asset]
                        adjusted_weights[asset] = (
                            current * (1 - blend_factor) +
                            suggested * blend_factor
                        )
                        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {
            asset: weight / total_weight
            for asset, weight in adjusted_weights.items()
        }
        
        # Apply bounds
        for asset in adjusted_weights:
            adjusted_weights[asset] = np.clip(
                adjusted_weights[asset],
                self.config.min_weight,
                self.config.max_weight
            )
            
        # Re-normalize after clipping
        total_weight = sum(adjusted_weights.values())
        adjusted_weights = {
            asset: weight / total_weight
            for asset, weight in adjusted_weights.items()
        }
        
        return adjusted_weights
        
    def _adapt_method_to_regime(
        self,
        method: str,
        regime: RegimeState
    ) -> str:
        """
        Adapt optimization method based on market regime
        """
        if regime.regime_type == 'crisis':
            # In crisis, use risk parity or minimum correlation
            return 'risk_parity'
        elif regime.regime_type == 'risk_off':
            # Risk-off: hierarchical risk parity
            return 'hrp'
        elif regime.regime_type == 'risk_on':
            # Risk-on: mean-variance or max diversification
            return 'max_diversification'
        elif regime.regime_type == 'transition':
            # Transition: stay with current method but reduce risk
            return method
        else:
            return method
            
    def _calculate_portfolio_metrics(
        self,
        weights: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate portfolio performance metrics
        """
        # Convert weights to array
        assets = list(weights.keys())
        w = np.array([weights[asset] for asset in assets])
        
        # Get returns and covariance
        returns = self.returns_data[assets]
        expected_ret = returns.mean().values * 252
        cov_matrix = returns.cov().values * 252
        
        # Portfolio metrics
        portfolio_return = w @ expected_ret
        portfolio_variance = w @ cov_matrix @ w
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe = (portfolio_return - self.config.risk_free_rate) / portfolio_vol
        
        # Diversification ratio
        individual_vols = np.sqrt(np.diag(cov_matrix))
        weighted_avg_vol = w @ individual_vols
        div_ratio = weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
        # Effective number of assets
        herfindahl = np.sum(w ** 2)
        effective_n = 1 / herfindahl if herfindahl > 0 else 1
        
        # Correlation risk
        corr_matrix = self.correlation_matrix.loc[assets, assets].values
        correlation_risk = 0
        for i in range(len(assets)):
            for j in range(i+1, len(assets)):
                correlation_risk += w[i] * w[j] * abs(corr_matrix[i, j])
                
        return {
            'expected_return': portfolio_return,
            'volatility': portfolio_vol,
            'sharpe_ratio': sharpe,
            'diversification_ratio': div_ratio,
            'effective_n_assets': effective_n,
            'correlation_risk': correlation_risk
        }
        
    def _calculate_turnover(self, new_weights: Dict[str, float]) -> float:
        """
        Calculate portfolio turnover
        """
        if not self.current_weights:
            return 1.0
            
        turnover = 0
        for asset in new_weights:
            old_weight = self.current_weights.get(asset, 0)
            new_weight = new_weights[asset]
            turnover += abs(new_weight - old_weight)
            
        return turnover / 2  # Two-way turnover
        
    def _estimate_expected_returns(self):
        """
        Estimate expected returns using various methods
        """
        # Simple historical mean
        historical_mean = self.returns_data.mean()
        
        # Exponentially weighted mean (more weight on recent)
        ewm_mean = self.returns_data.ewm(span=60).mean().iloc[-1]
        
        # Blend approaches
        self.expected_returns = 0.7 * historical_mean + 0.3 * ewm_mean
        
    def _get_network_metrics(self) -> Dict:
        """
        Get current network metrics
        """
        if self.network_analyzer.network:
            return self.network_analyzer.calculate_network_metrics()
        return {}
        
    def _get_quasi_diagonal_order(
        self,
        linkage_matrix: np.ndarray,
        n_assets: int
    ) -> List[int]:
        """
        Get quasi-diagonal ordering from linkage matrix
        """
        # This is a simplified version
        # In production, use proper dendrogram ordering
        
        # Get leaves order from linkage
        from scipy.cluster.hierarchy import leaves_list
        return leaves_list(linkage_matrix)
        
    def _recursive_bisection(
        self,
        cov_matrix: np.ndarray,
        sorted_indices: List[int]
    ) -> np.ndarray:
        """
        Recursive bisection for HRP
        """
        def get_cluster_variance(cov, indices):
            """Get variance of a cluster"""
            cluster_cov = cov[np.ix_(indices, indices)]
            inv_diag = 1 / np.diag(cluster_cov)
            weights = inv_diag / inv_diag.sum()
            return weights @ cluster_cov @ weights
            
        def recursive_allocation(indices, parent_weight=1.0):
            """Recursively allocate weights"""
            if len(indices) == 1:
                return {indices[0]: parent_weight}
                
            # Split into two clusters
            mid = len(indices) // 2
            left_indices = indices[:mid]
            right_indices = indices[mid:]
            
            # Calculate variances
            left_var = get_cluster_variance(cov_matrix, left_indices)
            right_var = get_cluster_variance(cov_matrix, right_indices)
            
            # Allocate weight inversely proportional to variance
            total_inv_var = 1/left_var + 1/right_var
            left_weight = parent_weight * (1/left_var) / total_inv_var
            right_weight = parent_weight * (1/right_var) / total_inv_var
            
            # Recursive allocation
            weights = {}
            weights.update(recursive_allocation(left_indices, left_weight))
            weights.update(recursive_allocation(right_indices, right_weight))
            
            return weights
            
        # Get weights
        weight_dict = recursive_allocation(list(range(len(sorted_indices))))
        
        # Convert to array in original order
        weights = np.zeros(len(sorted_indices))
        for i, weight in weight_dict.items():
            weights[sorted_indices[i]] = weight
            
        return weights
        
    def generate_rl_state(self) -> Dict[str, Any]:
        """
        Generate state representation for RL model
        
        Returns:
            Dictionary with state features for RL model
        """
        state = {
            'timestamp': datetime.now(),
            'current_weights': self.current_weights.copy(),
            'correlation_matrix': self.correlation_matrix.values.tolist() if self.correlation_matrix is not None else None,
            'returns_stats': {
                'mean': self.returns_data.mean().to_dict(),
                'std': self.returns_data.std().to_dict(),
                'skew': self.returns_data.skew().to_dict(),
                'kurtosis': self.returns_data.kurtosis().to_dict()
            } if self.returns_data is not None else None,
            'regime': {
                'type': self.regime_detector.current_regime.regime_type if self.regime_detector.current_regime else None,
                'confidence': self.regime_detector.current_regime.confidence if self.regime_detector.current_regime else 0
            },
            'network_metrics': self._get_network_metrics(),
            'portfolio_metrics': self._calculate_portfolio_metrics(self.current_weights) if self.current_weights else None,
            'recent_performance': {
                'returns': self._calculate_portfolio_returns()[-20:].tolist() if self.returns_data is not None else [],
                'volatility': self._calculate_portfolio_returns().rolling(20).std().iloc[-1] if self.returns_data is not None else 0
            }
        }
        
        return state
        
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate historical portfolio returns"""
        if self.returns_data is None or not self.current_weights:
            return pd.Series()
            
        weights = pd.Series(self.current_weights)
        aligned_weights = pd.Series(
            [weights.get(col, 0) for col in self.returns_data.columns],
            index=self.returns_data.columns
        )
        
        return (self.returns_data * aligned_weights).sum(axis=1)