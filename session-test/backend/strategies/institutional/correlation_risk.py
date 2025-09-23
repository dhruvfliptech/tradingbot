"""
Correlation-Based Risk Metrics and Portfolio Risk Management

This module calculates portfolio risk metrics based on correlation analysis:
- Portfolio VaR and CVaR with correlation adjustments
- Correlation risk contribution
- Diversification ratio and effective number of assets
- Concentration risk from correlation clusters
- Stress testing under correlation shocks
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.optimize import minimize
from scipy.linalg import sqrtm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RiskConfig:
    """Configuration for risk calculations"""
    confidence_levels: List[float] = field(default_factory=lambda: [0.95, 0.99])
    var_method: str = 'historical'  # 'historical', 'parametric', 'monte_carlo'
    cvar_enabled: bool = True
    n_simulations: int = 10000
    stress_scenarios: List[str] = field(default_factory=lambda: ['correlation_spike', 'decorrelation', 'contagion'])
    rebalance_frequency: str = 'daily'  # 'daily', 'weekly', 'monthly'
    max_position_size: float = 0.2  # Maximum 20% in single asset
    min_effective_assets: int = 10  # Minimum effective number of assets


@dataclass
class PortfolioRisk:
    """Container for portfolio risk metrics"""
    timestamp: datetime
    var_95: float
    var_99: float
    cvar_95: Optional[float]
    cvar_99: Optional[float]
    portfolio_volatility: float
    sharpe_ratio: float
    max_drawdown: float
    diversification_ratio: float
    effective_n_assets: float
    correlation_risk: float
    concentration_risk: float
    risk_contributions: Dict[str, float]
    correlation_contributions: Dict[Tuple[str, str], float]


@dataclass
class StressTestResult:
    """Results from stress testing"""
    scenario: str
    initial_portfolio_value: float
    stressed_portfolio_value: float
    loss: float
    loss_percentage: float
    affected_assets: List[str]
    new_correlations: pd.DataFrame
    risk_metrics: PortfolioRisk


class CorrelationRiskManager:
    """
    Manages portfolio risk based on correlation analysis
    """
    
    def __init__(self, config: Optional[RiskConfig] = None):
        """
        Initialize risk manager
        
        Args:
            config: Risk configuration
        """
        self.config = config or RiskConfig()
        self.portfolio_weights: Dict[str, float] = {}
        self.correlation_matrix: Optional[pd.DataFrame] = None
        self.returns_data: Optional[pd.DataFrame] = None
        self.risk_history: List[PortfolioRisk] = []
        
    def set_portfolio(
        self,
        weights: Dict[str, float],
        returns_data: pd.DataFrame,
        correlation_matrix: pd.DataFrame
    ):
        """
        Set portfolio data
        
        Args:
            weights: Portfolio weights by asset
            returns_data: Historical returns
            correlation_matrix: Correlation matrix
        """
        # Normalize weights
        total_weight = sum(weights.values())
        self.portfolio_weights = {
            asset: weight / total_weight
            for asset, weight in weights.items()
        }
        
        self.returns_data = returns_data
        self.correlation_matrix = correlation_matrix
        
        logger.info(f"Portfolio set with {len(weights)} assets")
        
    def calculate_portfolio_risk(
        self,
        timestamp: Optional[datetime] = None
    ) -> PortfolioRisk:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            timestamp: Timestamp for risk calculation
            
        Returns:
            PortfolioRisk object with all metrics
        """
        if self.returns_data is None or self.correlation_matrix is None:
            raise ValueError("Portfolio data not set")
            
        timestamp = timestamp or datetime.now()
        
        # Calculate VaR and CVaR
        var_95, cvar_95 = self._calculate_var_cvar(0.95)
        var_99, cvar_99 = self._calculate_var_cvar(0.99)
        
        # Portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility()
        
        # Sharpe ratio
        sharpe = self._calculate_sharpe_ratio()
        
        # Maximum drawdown
        max_dd = self._calculate_max_drawdown()
        
        # Diversification metrics
        div_ratio = self._calculate_diversification_ratio()
        eff_n = self._calculate_effective_n_assets()
        
        # Correlation risk
        corr_risk = self._calculate_correlation_risk()
        
        # Concentration risk
        conc_risk = self._calculate_concentration_risk()
        
        # Risk contributions
        risk_contrib = self._calculate_risk_contributions()
        
        # Correlation contributions
        corr_contrib = self._calculate_correlation_contributions()
        
        risk_metrics = PortfolioRisk(
            timestamp=timestamp,
            var_95=var_95,
            var_99=var_99,
            cvar_95=cvar_95,
            cvar_99=cvar_99,
            portfolio_volatility=portfolio_vol,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            diversification_ratio=div_ratio,
            effective_n_assets=eff_n,
            correlation_risk=corr_risk,
            concentration_risk=conc_risk,
            risk_contributions=risk_contrib,
            correlation_contributions=corr_contrib
        )
        
        self.risk_history.append(risk_metrics)
        
        return risk_metrics
        
    def calculate_var_cvar(
        self,
        confidence_level: float = 0.95,
        method: Optional[str] = None
    ) -> Tuple[float, float]:
        """
        Calculate Value at Risk and Conditional Value at Risk
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            method: VaR calculation method
            
        Returns:
            Tuple of (VaR, CVaR)
        """
        method = method or self.config.var_method
        
        # Get portfolio returns
        portfolio_returns = self._calculate_portfolio_returns()
        
        if method == 'historical':
            var, cvar = self._historical_var_cvar(portfolio_returns, confidence_level)
        elif method == 'parametric':
            var, cvar = self._parametric_var_cvar(portfolio_returns, confidence_level)
        elif method == 'monte_carlo':
            var, cvar = self._monte_carlo_var_cvar(confidence_level)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
            
        return var, cvar
        
    def stress_test_correlation(
        self,
        scenario: str = 'correlation_spike',
        magnitude: float = 0.3
    ) -> StressTestResult:
        """
        Stress test portfolio under correlation scenarios
        
        Args:
            scenario: Stress scenario type
            magnitude: Magnitude of stress
            
        Returns:
            StressTestResult with stressed metrics
        """
        # Store original state
        original_corr = self.correlation_matrix.copy()
        initial_value = self._calculate_portfolio_value()
        
        # Apply stress scenario
        if scenario == 'correlation_spike':
            stressed_corr = self._stress_correlation_spike(magnitude)
        elif scenario == 'decorrelation':
            stressed_corr = self._stress_decorrelation(magnitude)
        elif scenario == 'contagion':
            stressed_corr = self._stress_contagion(magnitude)
        elif scenario == 'regime_change':
            stressed_corr = self._stress_regime_change(magnitude)
        else:
            raise ValueError(f"Unknown stress scenario: {scenario}")
            
        # Calculate stressed portfolio value
        self.correlation_matrix = stressed_corr
        stressed_value = self._calculate_portfolio_value_stressed()
        
        # Calculate stressed risk metrics
        stressed_risk = self.calculate_portfolio_risk()
        
        # Restore original correlation
        self.correlation_matrix = original_corr
        
        # Identify most affected assets
        corr_changes = (stressed_corr - original_corr).abs()
        affected_assets = list(
            corr_changes.sum().nlargest(10).index
        )
        
        result = StressTestResult(
            scenario=scenario,
            initial_portfolio_value=initial_value,
            stressed_portfolio_value=stressed_value,
            loss=initial_value - stressed_value,
            loss_percentage=(initial_value - stressed_value) / initial_value * 100,
            affected_assets=affected_assets,
            new_correlations=stressed_corr,
            risk_metrics=stressed_risk
        )
        
        return result
        
    def calculate_marginal_risk_contribution(
        self,
        asset: str
    ) -> float:
        """
        Calculate marginal risk contribution of an asset
        
        Args:
            asset: Asset identifier
            
        Returns:
            Marginal risk contribution
        """
        if asset not in self.portfolio_weights:
            return 0.0
            
        # Calculate portfolio risk gradient
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        # Covariance matrix from correlation and volatilities
        volatilities = self.returns_data.std()
        cov_matrix = self.correlation_matrix * np.outer(volatilities, volatilities)
        
        # Marginal contribution to risk
        portfolio_var = weights @ cov_matrix @ weights.T
        portfolio_vol = np.sqrt(portfolio_var)
        
        asset_idx = list(self.correlation_matrix.columns).index(asset)
        marginal_contrib = (cov_matrix @ weights)[asset_idx] / portfolio_vol
        
        return marginal_contrib
        
    def calculate_incremental_var(
        self,
        asset: str,
        additional_weight: float
    ) -> float:
        """
        Calculate incremental VaR from adding to a position
        
        Args:
            asset: Asset to increase
            additional_weight: Additional weight
            
        Returns:
            Incremental VaR
        """
        # Current VaR
        current_var, _ = self.calculate_var_cvar(0.95)
        
        # New weights
        new_weights = self.portfolio_weights.copy()
        new_weights[asset] = new_weights.get(asset, 0) + additional_weight
        
        # Normalize
        total = sum(new_weights.values())
        new_weights = {k: v/total for k, v in new_weights.items()}
        
        # Calculate new VaR with updated weights
        old_weights = self.portfolio_weights
        self.portfolio_weights = new_weights
        new_var, _ = self.calculate_var_cvar(0.95)
        self.portfolio_weights = old_weights
        
        return new_var - current_var
        
    def optimize_portfolio_risk(
        self,
        target_risk: Optional[float] = None,
        constraints: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Optimize portfolio weights to minimize correlation risk
        
        Args:
            target_risk: Target risk level
            constraints: Additional constraints
            
        Returns:
            Optimized weights
        """
        assets = list(self.correlation_matrix.columns)
        n_assets = len(assets)
        
        # Covariance matrix
        volatilities = self.returns_data[assets].std()
        cov_matrix = self.correlation_matrix.values * np.outer(volatilities, volatilities)
        
        # Expected returns (using historical mean)
        expected_returns = self.returns_data[assets].mean().values
        
        # Objective: Minimize portfolio variance
        def objective(weights):
            return weights @ cov_matrix @ weights.T
            
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}  # Weights sum to 1
        ]
        
        if target_risk:
            cons.append({
                'type': 'ineq',
                'fun': lambda w: target_risk - np.sqrt(w @ cov_matrix @ w.T)
            })
            
        # Bounds
        bounds = [(0, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            optimized_weights = dict(zip(assets, result.x))
            return optimized_weights
        else:
            logger.warning("Optimization failed, returning equal weights")
            return dict(zip(assets, x0))
            
    def calculate_risk_parity_weights(self) -> Dict[str, float]:
        """
        Calculate risk parity portfolio weights
        
        Returns:
            Risk parity weights
        """
        assets = list(self.correlation_matrix.columns)
        n_assets = len(assets)
        
        # Covariance matrix
        volatilities = self.returns_data[assets].std()
        cov_matrix = self.correlation_matrix.values * np.outer(volatilities, volatilities)
        
        # Risk parity objective
        def risk_parity_objective(weights):
            portfolio_vol = np.sqrt(weights @ cov_matrix @ weights.T)
            marginal_contrib = (cov_matrix @ weights) / portfolio_vol
            risk_contrib = weights * marginal_contrib
            
            # Minimize variance of risk contributions
            return np.var(risk_contrib)
            
        # Constraints
        cons = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        ]
        
        # Bounds
        bounds = [(0.01, self.config.max_position_size) for _ in range(n_assets)]
        
        # Initial guess
        x0 = np.ones(n_assets) / n_assets
        
        # Optimize
        result = minimize(
            risk_parity_objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=cons
        )
        
        if result.success:
            return dict(zip(assets, result.x))
        else:
            return dict(zip(assets, x0))
            
    # Private helper methods
    
    def _calculate_var_cvar(
        self,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Calculate VaR and CVaR"""
        if self.config.var_method == 'historical':
            return self._historical_var_cvar(
                self._calculate_portfolio_returns(),
                confidence_level
            )
        elif self.config.var_method == 'parametric':
            return self._parametric_var_cvar(
                self._calculate_portfolio_returns(),
                confidence_level
            )
        else:
            return self._monte_carlo_var_cvar(confidence_level)
            
    def _historical_var_cvar(
        self,
        returns: pd.Series,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Historical VaR and CVaR"""
        var = returns.quantile(1 - confidence_level)
        
        if self.config.cvar_enabled:
            cvar = returns[returns <= var].mean()
        else:
            cvar = None
            
        return -var, -cvar if cvar else None
        
    def _parametric_var_cvar(
        self,
        returns: pd.Series,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Parametric VaR and CVaR assuming normal distribution"""
        mean = returns.mean()
        std = returns.std()
        
        z_score = stats.norm.ppf(1 - confidence_level)
        var = mean + z_score * std
        
        if self.config.cvar_enabled:
            # CVaR for normal distribution
            cvar = mean - std * stats.norm.pdf(z_score) / (1 - confidence_level)
        else:
            cvar = None
            
        return -var, -cvar if cvar else None
        
    def _monte_carlo_var_cvar(
        self,
        confidence_level: float
    ) -> Tuple[float, float]:
        """Monte Carlo VaR and CVaR"""
        # Get portfolio parameters
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        returns_mean = self.returns_data.mean().values
        returns_cov = self.correlation_matrix.values * np.outer(
            self.returns_data.std().values,
            self.returns_data.std().values
        )
        
        # Simulate returns
        simulated_returns = np.random.multivariate_normal(
            returns_mean,
            returns_cov,
            self.config.n_simulations
        )
        
        # Portfolio returns
        portfolio_returns = simulated_returns @ weights
        
        # Calculate VaR and CVaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        
        if self.config.cvar_enabled:
            cvar = portfolio_returns[portfolio_returns <= var].mean()
        else:
            cvar = None
            
        return -var, -cvar if cvar else None
        
    def _calculate_portfolio_returns(self) -> pd.Series:
        """Calculate historical portfolio returns"""
        weights = pd.Series(self.portfolio_weights)
        
        # Align weights with returns columns
        aligned_weights = pd.Series(
            [weights.get(col, 0) for col in self.returns_data.columns],
            index=self.returns_data.columns
        )
        
        portfolio_returns = (self.returns_data * aligned_weights).sum(axis=1)
        
        return portfolio_returns
        
    def _calculate_portfolio_volatility(self) -> float:
        """Calculate portfolio volatility"""
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        volatilities = self.returns_data.std().values
        cov_matrix = self.correlation_matrix.values * np.outer(volatilities, volatilities)
        
        portfolio_var = weights @ cov_matrix @ weights.T
        
        return np.sqrt(portfolio_var)
        
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio"""
        portfolio_returns = self._calculate_portfolio_returns()
        
        excess_returns = portfolio_returns.mean() - risk_free_rate / 252
        portfolio_vol = portfolio_returns.std()
        
        return excess_returns / portfolio_vol * np.sqrt(252) if portfolio_vol > 0 else 0
        
    def _calculate_max_drawdown(self) -> float:
        """Calculate maximum drawdown"""
        portfolio_returns = self._calculate_portfolio_returns()
        cumulative_returns = (1 + portfolio_returns).cumprod()
        
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        
        return abs(drawdown.min())
        
    def _calculate_diversification_ratio(self) -> float:
        """Calculate diversification ratio"""
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        volatilities = self.returns_data.std().values
        
        # Weighted average volatility
        weighted_avg_vol = np.sum(weights * volatilities)
        
        # Portfolio volatility
        portfolio_vol = self._calculate_portfolio_volatility()
        
        return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1
        
    def _calculate_effective_n_assets(self) -> float:
        """Calculate effective number of assets (Entropy-based)"""
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        # Remove zero weights
        weights = weights[weights > 0]
        
        # Herfindahl index
        herfindahl = np.sum(weights ** 2)
        
        # Effective N
        return 1 / herfindahl if herfindahl > 0 else 1
        
    def _calculate_correlation_risk(self) -> float:
        """Calculate correlation risk metric"""
        # Average pairwise correlation weighted by position sizes
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        # Weighted correlation matrix
        weighted_corr = self.correlation_matrix.values * np.outer(weights, weights)
        
        # Sum of off-diagonal elements
        np.fill_diagonal(weighted_corr, 0)
        correlation_risk = np.sum(np.abs(weighted_corr))
        
        return correlation_risk
        
    def _calculate_concentration_risk(self) -> float:
        """Calculate concentration risk from correlation clusters"""
        # Identify highly correlated clusters
        threshold = 0.7
        high_corr = (self.correlation_matrix.abs() > threshold).astype(int)
        np.fill_diagonal(high_corr.values, 0)
        
        # Find clusters using connected components
        clusters = self._find_correlation_clusters(high_corr.values)
        
        # Calculate concentration in each cluster
        cluster_concentrations = []
        
        for cluster in clusters:
            cluster_weight = sum(
                self.portfolio_weights.get(
                    self.correlation_matrix.columns[i], 0
                )
                for i in cluster
            )
            cluster_concentrations.append(cluster_weight)
            
        # Concentration risk is the maximum cluster weight
        return max(cluster_concentrations) if cluster_concentrations else 0
        
    def _calculate_risk_contributions(self) -> Dict[str, float]:
        """Calculate risk contribution by asset"""
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        volatilities = self.returns_data.std().values
        cov_matrix = self.correlation_matrix.values * np.outer(volatilities, volatilities)
        
        portfolio_vol = self._calculate_portfolio_volatility()
        
        # Marginal contributions
        marginal_contrib = (cov_matrix @ weights) / portfolio_vol
        
        # Risk contributions
        risk_contrib = weights * marginal_contrib
        
        return dict(zip(self.correlation_matrix.columns, risk_contrib))
        
    def _calculate_correlation_contributions(self) -> Dict[Tuple[str, str], float]:
        """Calculate risk contribution from each correlation pair"""
        weights = np.array([
            self.portfolio_weights.get(col, 0)
            for col in self.correlation_matrix.columns
        ])
        
        volatilities = self.returns_data.std().values
        portfolio_vol = self._calculate_portfolio_volatility()
        
        contributions = {}
        
        for i, asset1 in enumerate(self.correlation_matrix.columns):
            for j, asset2 in enumerate(self.correlation_matrix.columns):
                if i < j:
                    # Contribution from this correlation
                    contrib = (
                        2 * weights[i] * weights[j] *
                        volatilities[i] * volatilities[j] *
                        self.correlation_matrix.iloc[i, j]
                    ) / (portfolio_vol ** 2)
                    
                    contributions[(asset1, asset2)] = contrib
                    
        return contributions
        
    def _calculate_portfolio_value(self) -> float:
        """Calculate portfolio value (normalized to 1)"""
        return 1.0
        
    def _calculate_portfolio_value_stressed(self) -> float:
        """Calculate portfolio value under stress"""
        # Simplified stress impact calculation
        portfolio_vol = self._calculate_portfolio_volatility()
        
        # Assume 2-sigma shock
        stressed_value = 1.0 - 2 * portfolio_vol
        
        return stressed_value
        
    def _stress_correlation_spike(self, magnitude: float) -> pd.DataFrame:
        """Apply correlation spike stress"""
        stressed_corr = self.correlation_matrix.copy()
        
        # Increase all correlations
        for i in range(len(stressed_corr)):
            for j in range(len(stressed_corr)):
                if i != j:
                    current = stressed_corr.iloc[i, j]
                    stressed = current + magnitude * (1 - abs(current))
                    stressed_corr.iloc[i, j] = np.clip(stressed, -1, 1)
                    
        return stressed_corr
        
    def _stress_decorrelation(self, magnitude: float) -> pd.DataFrame:
        """Apply decorrelation stress"""
        stressed_corr = self.correlation_matrix.copy()
        
        # Reduce all correlations
        for i in range(len(stressed_corr)):
            for j in range(len(stressed_corr)):
                if i != j:
                    stressed_corr.iloc[i, j] *= (1 - magnitude)
                    
        return stressed_corr
        
    def _stress_contagion(self, magnitude: float) -> pd.DataFrame:
        """Apply contagion stress (increase in tail correlations)"""
        stressed_corr = self.correlation_matrix.copy()
        
        # Identify core assets (highest average correlation)
        avg_corr = stressed_corr.mean()
        core_assets = avg_corr.nlargest(5).index
        
        # Increase correlation with core assets
        for asset in core_assets:
            for other in stressed_corr.columns:
                if asset != other:
                    current = stressed_corr.loc[asset, other]
                    stressed = current + magnitude
                    stressed_corr.loc[asset, other] = np.clip(stressed, -1, 1)
                    stressed_corr.loc[other, asset] = np.clip(stressed, -1, 1)
                    
        return stressed_corr
        
    def _stress_regime_change(self, magnitude: float) -> pd.DataFrame:
        """Apply regime change stress"""
        # Randomly shuffle correlation structure
        stressed_corr = self.correlation_matrix.copy()
        
        # Add random noise
        noise = np.random.randn(*stressed_corr.shape) * magnitude
        stressed_corr += noise
        
        # Ensure symmetry and proper bounds
        stressed_corr = (stressed_corr + stressed_corr.T) / 2
        np.fill_diagonal(stressed_corr.values, 1)
        stressed_corr = stressed_corr.clip(-1, 1)
        
        return stressed_corr
        
    def _find_correlation_clusters(self, adjacency: np.ndarray) -> List[List[int]]:
        """Find clusters in correlation network"""
        n = len(adjacency)
        visited = [False] * n
        clusters = []
        
        def dfs(node, cluster):
            visited[node] = True
            cluster.append(node)
            
            for neighbor in range(n):
                if adjacency[node][neighbor] == 1 and not visited[neighbor]:
                    dfs(neighbor, cluster)
                    
        for i in range(n):
            if not visited[i]:
                cluster = []
                dfs(i, cluster)
                if len(cluster) > 1:
                    clusters.append(cluster)
                    
        return clusters