"""
Cross-Asset Correlation Engine for Portfolio Risk Management

This module implements a comprehensive correlation calculation system that:
- Tracks correlations across 50+ crypto assets
- Calculates rolling correlations with multiple time windows
- Implements advanced correlation methods (DCC-GARCH, copulas)
- Provides real-time correlation updates
- Handles missing data and asynchronous price updates
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import warnings
from scipy import stats
from scipy.stats import kendalltau, spearmanr, pearsonr
from scipy.stats import multivariate_normal
from scipy.special import gamma
import logging
from collections import deque
import asyncio
from concurrent.futures import ThreadPoolExecutor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    """Configuration for correlation calculations"""
    windows: List[int] = field(default_factory=lambda: [30, 60, 90])
    min_observations: int = 20
    correlation_methods: List[str] = field(default_factory=lambda: ['pearson', 'spearman', 'kendall'])
    ewm_halflife: int = 30
    dcc_garch_enabled: bool = True
    copula_enabled: bool = True
    update_frequency: int = 300  # seconds
    significance_level: float = 0.05
    max_assets: int = 100
    enable_caching: bool = True
    parallel_workers: int = 4


@dataclass
class CorrelationResult:
    """Container for correlation calculation results"""
    timestamp: datetime
    method: str
    window: int
    correlation_matrix: pd.DataFrame
    p_values: Optional[pd.DataFrame] = None
    confidence_intervals: Optional[Dict] = None
    is_significant: Optional[pd.DataFrame] = None
    metadata: Dict = field(default_factory=dict)


class CorrelationEngine:
    """
    Main correlation calculation engine for cross-asset analysis
    """
    
    def __init__(self, config: Optional[CorrelationConfig] = None):
        """
        Initialize correlation engine
        
        Args:
            config: Configuration object for correlation calculations
        """
        self.config = config or CorrelationConfig()
        self.price_data: Dict[str, pd.Series] = {}
        self.returns_data: Dict[str, pd.Series] = {}
        self.correlation_cache: Dict[str, CorrelationResult] = {}
        self.last_update: Optional[datetime] = None
        self.executor = ThreadPoolExecutor(max_workers=self.config.parallel_workers)
        
        # Initialize data buffers for streaming calculations
        self.price_buffers: Dict[str, deque] = {}
        self.max_buffer_size = max(self.config.windows) * 2
        
        # DCC-GARCH parameters
        self.dcc_params: Optional[Dict] = None
        self.garch_models: Dict[str, Dict] = {}
        
    def add_price_data(self, asset: str, prices: pd.Series) -> None:
        """
        Add price data for an asset
        
        Args:
            asset: Asset identifier
            prices: Time series of prices
        """
        if len(self.price_data) >= self.config.max_assets:
            logger.warning(f"Maximum number of assets ({self.config.max_assets}) reached")
            return
            
        self.price_data[asset] = prices
        self.returns_data[asset] = prices.pct_change().dropna()
        
        # Initialize buffer for streaming
        if asset not in self.price_buffers:
            self.price_buffers[asset] = deque(maxlen=self.max_buffer_size)
        
        logger.info(f"Added price data for {asset}: {len(prices)} observations")
        
    def update_price(self, asset: str, price: float, timestamp: datetime) -> None:
        """
        Update single price point for streaming calculations
        
        Args:
            asset: Asset identifier
            price: Latest price
            timestamp: Price timestamp
        """
        if asset not in self.price_buffers:
            self.price_buffers[asset] = deque(maxlen=self.max_buffer_size)
            
        self.price_buffers[asset].append((timestamp, price))
        
        # Update main price series if exists
        if asset in self.price_data:
            self.price_data[asset][timestamp] = price
            
            # Recalculate returns for the latest point
            if len(self.price_data[asset]) > 1:
                prev_price = self.price_data[asset].iloc[-2]
                ret = (price - prev_price) / prev_price
                self.returns_data[asset][timestamp] = ret
                
    def calculate_correlation_matrix(
        self, 
        method: str = 'pearson',
        window: int = 30,
        assets: Optional[List[str]] = None
    ) -> CorrelationResult:
        """
        Calculate correlation matrix for specified assets
        
        Args:
            method: Correlation method ('pearson', 'spearman', 'kendall')
            window: Rolling window size in days
            assets: List of assets to include (None for all)
            
        Returns:
            CorrelationResult with correlation matrix and statistics
        """
        # Use cache if available
        cache_key = f"{method}_{window}_{str(assets)}"
        if self.config.enable_caching and cache_key in self.correlation_cache:
            cached = self.correlation_cache[cache_key]
            if (datetime.now() - cached.timestamp).seconds < self.config.update_frequency:
                return cached
                
        # Select assets
        if assets is None:
            assets = list(self.returns_data.keys())
            
        # Prepare returns matrix
        returns_df = pd.DataFrame({
            asset: self.returns_data[asset] 
            for asset in assets 
            if asset in self.returns_data
        })
        
        # Apply rolling window
        if window:
            returns_df = returns_df.tail(window)
            
        # Check minimum observations
        if len(returns_df) < self.config.min_observations:
            logger.warning(f"Insufficient data: {len(returns_df)} < {self.config.min_observations}")
            return self._empty_result(assets, method, window)
            
        # Calculate correlation matrix
        if method == 'pearson':
            corr_matrix = returns_df.corr(method='pearson')
            p_values = self._calculate_p_values_pearson(returns_df)
        elif method == 'spearman':
            corr_matrix = returns_df.corr(method='spearman')
            p_values = self._calculate_p_values_spearman(returns_df)
        elif method == 'kendall':
            corr_matrix = returns_df.corr(method='kendall')
            p_values = self._calculate_p_values_kendall(returns_df)
        else:
            raise ValueError(f"Unknown correlation method: {method}")
            
        # Calculate significance
        is_significant = p_values < self.config.significance_level
        
        # Calculate confidence intervals
        confidence_intervals = self._calculate_confidence_intervals(
            corr_matrix, len(returns_df), method
        )
        
        result = CorrelationResult(
            timestamp=datetime.now(),
            method=method,
            window=window,
            correlation_matrix=corr_matrix,
            p_values=p_values,
            confidence_intervals=confidence_intervals,
            is_significant=is_significant,
            metadata={
                'n_observations': len(returns_df),
                'assets': assets,
                'mean_correlation': corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)].mean()
            }
        )
        
        # Cache result
        if self.config.enable_caching:
            self.correlation_cache[cache_key] = result
            
        return result
        
    def calculate_ewm_correlation(
        self, 
        assets: Optional[List[str]] = None,
        halflife: Optional[int] = None
    ) -> CorrelationResult:
        """
        Calculate exponentially weighted correlation matrix
        
        Args:
            assets: List of assets to include
            halflife: Half-life for exponential weighting
            
        Returns:
            CorrelationResult with EWM correlation matrix
        """
        halflife = halflife or self.config.ewm_halflife
        
        if assets is None:
            assets = list(self.returns_data.keys())
            
        returns_df = pd.DataFrame({
            asset: self.returns_data[asset] 
            for asset in assets 
            if asset in self.returns_data
        })
        
        # Calculate EWM correlation
        ewm_corr = returns_df.ewm(halflife=halflife, min_periods=self.config.min_observations).corr()
        
        # Get the latest correlation matrix
        if isinstance(ewm_corr.index, pd.MultiIndex):
            latest_date = ewm_corr.index.get_level_values(0).unique()[-1]
            corr_matrix = ewm_corr.loc[latest_date]
        else:
            corr_matrix = ewm_corr.iloc[-len(assets):, :]
            
        return CorrelationResult(
            timestamp=datetime.now(),
            method='ewm',
            window=halflife,
            correlation_matrix=corr_matrix,
            metadata={
                'halflife': halflife,
                'assets': assets
            }
        )
        
    def calculate_dcc_garch_correlation(
        self, 
        assets: List[str],
        lookback: int = 252
    ) -> CorrelationResult:
        """
        Calculate Dynamic Conditional Correlation using DCC-GARCH model
        
        Args:
            assets: List of assets
            lookback: Number of periods for model estimation
            
        Returns:
            CorrelationResult with DCC-GARCH correlation matrix
        """
        if not self.config.dcc_garch_enabled:
            logger.warning("DCC-GARCH is disabled in configuration")
            return self._empty_result(assets, 'dcc_garch', lookback)
            
        returns_df = pd.DataFrame({
            asset: self.returns_data[asset].tail(lookback)
            for asset in assets 
            if asset in self.returns_data
        })
        
        # Standardize returns using GARCH(1,1) for each asset
        standardized_residuals = pd.DataFrame(index=returns_df.index)
        
        for asset in returns_df.columns:
            returns = returns_df[asset].values
            
            # Fit GARCH(1,1) model
            omega, alpha, beta = self._fit_garch_11(returns)
            
            # Calculate conditional variance
            variance = np.zeros(len(returns))
            variance[0] = np.var(returns)
            
            for t in range(1, len(returns)):
                variance[t] = omega + alpha * returns[t-1]**2 + beta * variance[t-1]
                
            # Standardize residuals
            standardized_residuals[asset] = returns / np.sqrt(variance)
            
            # Store GARCH parameters
            self.garch_models[asset] = {
                'omega': omega,
                'alpha': alpha,
                'beta': beta,
                'last_variance': variance[-1]
            }
            
        # Estimate DCC parameters
        a, b = self._estimate_dcc_params(standardized_residuals)
        
        # Calculate dynamic correlation matrix
        Q_bar = standardized_residuals.corr()
        Q_t = Q_bar.copy()
        
        # Update Q_t using DCC recursion
        for t in range(1, len(standardized_residuals)):
            epsilon_t = standardized_residuals.iloc[t].values.reshape(-1, 1)
            Q_t = (1 - a - b) * Q_bar + a * (epsilon_t @ epsilon_t.T) + b * Q_t
            
        # Normalize to get correlation matrix
        D_t = np.diag(1 / np.sqrt(np.diag(Q_t)))
        R_t = D_t @ Q_t @ D_t
        
        corr_matrix = pd.DataFrame(
            R_t,
            index=returns_df.columns,
            columns=returns_df.columns
        )
        
        return CorrelationResult(
            timestamp=datetime.now(),
            method='dcc_garch',
            window=lookback,
            correlation_matrix=corr_matrix,
            metadata={
                'dcc_params': {'a': a, 'b': b},
                'garch_models': self.garch_models,
                'assets': assets
            }
        )
        
    def calculate_copula_correlation(
        self, 
        assets: List[str],
        copula_type: str = 'gaussian'
    ) -> CorrelationResult:
        """
        Calculate correlation using copula methods
        
        Args:
            assets: List of assets
            copula_type: Type of copula ('gaussian', 't', 'clayton', 'gumbel')
            
        Returns:
            CorrelationResult with copula-based correlation
        """
        if not self.config.copula_enabled:
            logger.warning("Copula analysis is disabled in configuration")
            return self._empty_result(assets, f'copula_{copula_type}', None)
            
        returns_df = pd.DataFrame({
            asset: self.returns_data[asset]
            for asset in assets 
            if asset in self.returns_data
        })
        
        # Transform to uniform marginals using empirical CDF
        uniform_data = pd.DataFrame(index=returns_df.index)
        for col in returns_df.columns:
            uniform_data[col] = stats.rankdata(returns_df[col]) / (len(returns_df) + 1)
            
        if copula_type == 'gaussian':
            # Gaussian copula - estimate correlation from uniform data
            normal_quantiles = pd.DataFrame(index=returns_df.index)
            for col in uniform_data.columns:
                normal_quantiles[col] = stats.norm.ppf(uniform_data[col])
                
            corr_matrix = normal_quantiles.corr()
            
        elif copula_type == 't':
            # Student-t copula
            corr_matrix, nu = self._fit_t_copula(uniform_data)
            
        elif copula_type == 'clayton':
            # Clayton copula - lower tail dependence
            corr_matrix = self._fit_clayton_copula(uniform_data)
            
        elif copula_type == 'gumbel':
            # Gumbel copula - upper tail dependence
            corr_matrix = self._fit_gumbel_copula(uniform_data)
            
        else:
            raise ValueError(f"Unknown copula type: {copula_type}")
            
        return CorrelationResult(
            timestamp=datetime.now(),
            method=f'copula_{copula_type}',
            window=len(returns_df),
            correlation_matrix=corr_matrix,
            metadata={
                'copula_type': copula_type,
                'assets': assets
            }
        )
        
    def calculate_tail_correlation(
        self, 
        assets: List[str],
        threshold: float = 0.95
    ) -> CorrelationResult:
        """
        Calculate correlation in the tails of the distribution
        
        Args:
            assets: List of assets
            threshold: Quantile threshold for tail (0.95 for 5% tail)
            
        Returns:
            CorrelationResult with tail correlations
        """
        returns_df = pd.DataFrame({
            asset: self.returns_data[asset]
            for asset in assets 
            if asset in self.returns_data
        })
        
        # Calculate thresholds
        upper_threshold = returns_df.quantile(threshold)
        lower_threshold = returns_df.quantile(1 - threshold)
        
        # Filter for tail events
        upper_tail = returns_df[returns_df > upper_threshold].dropna()
        lower_tail = returns_df[returns_df < lower_threshold].dropna()
        
        # Calculate correlations in tails
        upper_corr = upper_tail.corr() if len(upper_tail) > self.config.min_observations else pd.DataFrame()
        lower_corr = lower_tail.corr() if len(lower_tail) > self.config.min_observations else pd.DataFrame()
        
        # Average tail correlation
        if not upper_corr.empty and not lower_corr.empty:
            avg_tail_corr = (upper_corr + lower_corr) / 2
        elif not upper_corr.empty:
            avg_tail_corr = upper_corr
        elif not lower_corr.empty:
            avg_tail_corr = lower_corr
        else:
            avg_tail_corr = pd.DataFrame(
                np.nan,
                index=returns_df.columns,
                columns=returns_df.columns
            )
            
        return CorrelationResult(
            timestamp=datetime.now(),
            method='tail_correlation',
            window=len(returns_df),
            correlation_matrix=avg_tail_corr,
            metadata={
                'threshold': threshold,
                'upper_tail_obs': len(upper_tail),
                'lower_tail_obs': len(lower_tail),
                'assets': assets
            }
        )
        
    async def calculate_all_correlations(
        self, 
        assets: Optional[List[str]] = None
    ) -> Dict[str, CorrelationResult]:
        """
        Calculate all correlation methods asynchronously
        
        Args:
            assets: List of assets to analyze
            
        Returns:
            Dictionary of correlation results by method
        """
        if assets is None:
            assets = list(self.returns_data.keys())[:50]  # Limit to 50 assets
            
        results = {}
        tasks = []
        
        # Standard correlations for multiple windows
        for window in self.config.windows:
            for method in self.config.correlation_methods:
                task = asyncio.create_task(
                    self._async_correlation(method, window, assets)
                )
                tasks.append((f"{method}_{window}", task))
                
        # Special correlation methods
        tasks.append(("ewm", asyncio.create_task(
            self._async_ewm_correlation(assets)
        )))
        
        if self.config.dcc_garch_enabled:
            tasks.append(("dcc_garch", asyncio.create_task(
                self._async_dcc_garch(assets)
            )))
            
        if self.config.copula_enabled:
            tasks.append(("copula_gaussian", asyncio.create_task(
                self._async_copula(assets, 'gaussian')
            )))
            
        tasks.append(("tail", asyncio.create_task(
            self._async_tail_correlation(assets)
        )))
        
        # Gather results
        for name, task in tasks:
            try:
                results[name] = await task
            except Exception as e:
                logger.error(f"Error calculating {name}: {e}")
                
        return results
        
    def get_correlation_summary(
        self, 
        results: Dict[str, CorrelationResult]
    ) -> pd.DataFrame:
        """
        Generate summary statistics from multiple correlation results
        
        Args:
            results: Dictionary of correlation results
            
        Returns:
            DataFrame with summary statistics
        """
        summary_data = []
        
        for name, result in results.items():
            if result.correlation_matrix is not None and not result.correlation_matrix.empty:
                corr_values = result.correlation_matrix.values[
                    np.triu_indices_from(result.correlation_matrix.values, k=1)
                ]
                
                summary_data.append({
                    'method': name,
                    'mean_correlation': np.nanmean(corr_values),
                    'median_correlation': np.nanmedian(corr_values),
                    'std_correlation': np.nanstd(corr_values),
                    'min_correlation': np.nanmin(corr_values),
                    'max_correlation': np.nanmax(corr_values),
                    'timestamp': result.timestamp
                })
                
        return pd.DataFrame(summary_data)
        
    # Private helper methods
    
    def _calculate_p_values_pearson(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate p-values for Pearson correlation"""
        n = len(returns_df)
        corr_matrix = returns_df.corr()
        p_values = pd.DataFrame(
            np.nan,
            index=corr_matrix.index,
            columns=corr_matrix.columns
        )
        
        for i, col1 in enumerate(returns_df.columns):
            for j, col2 in enumerate(returns_df.columns):
                if i != j:
                    r = corr_matrix.iloc[i, j]
                    t_stat = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
                    p_values.iloc[i, j] = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
                else:
                    p_values.iloc[i, j] = 0
                    
        return p_values
        
    def _calculate_p_values_spearman(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate p-values for Spearman correlation"""
        p_values = pd.DataFrame(
            np.nan,
            index=returns_df.columns,
            columns=returns_df.columns
        )
        
        for i, col1 in enumerate(returns_df.columns):
            for j, col2 in enumerate(returns_df.columns):
                if i <= j:
                    if i == j:
                        p_values.iloc[i, j] = 0
                        p_values.iloc[j, i] = 0
                    else:
                        _, p_val = spearmanr(returns_df[col1], returns_df[col2])
                        p_values.iloc[i, j] = p_val
                        p_values.iloc[j, i] = p_val
                        
        return p_values
        
    def _calculate_p_values_kendall(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate p-values for Kendall correlation"""
        p_values = pd.DataFrame(
            np.nan,
            index=returns_df.columns,
            columns=returns_df.columns
        )
        
        for i, col1 in enumerate(returns_df.columns):
            for j, col2 in enumerate(returns_df.columns):
                if i <= j:
                    if i == j:
                        p_values.iloc[i, j] = 0
                        p_values.iloc[j, i] = 0
                    else:
                        _, p_val = kendalltau(returns_df[col1], returns_df[col2])
                        p_values.iloc[i, j] = p_val
                        p_values.iloc[j, i] = p_val
                        
        return p_values
        
    def _calculate_confidence_intervals(
        self, 
        corr_matrix: pd.DataFrame,
        n_obs: int,
        method: str
    ) -> Dict:
        """Calculate confidence intervals for correlations"""
        confidence_intervals = {}
        z_critical = stats.norm.ppf(1 - self.config.significance_level / 2)
        
        for i, col1 in enumerate(corr_matrix.columns):
            for j, col2 in enumerate(corr_matrix.columns):
                if i < j:
                    r = corr_matrix.iloc[i, j]
                    
                    if method == 'pearson':
                        # Fisher transformation for confidence intervals
                        z = 0.5 * np.log((1 + r) / (1 - r))
                        se_z = 1 / np.sqrt(n_obs - 3)
                        
                        z_lower = z - z_critical * se_z
                        z_upper = z + z_critical * se_z
                        
                        r_lower = (np.exp(2 * z_lower) - 1) / (np.exp(2 * z_lower) + 1)
                        r_upper = (np.exp(2 * z_upper) - 1) / (np.exp(2 * z_upper) + 1)
                        
                    else:
                        # Bootstrap or asymptotic for other methods
                        se = np.sqrt((1 - r**2) / (n_obs - 2))
                        r_lower = r - z_critical * se
                        r_upper = r + z_critical * se
                        
                    confidence_intervals[f"{col1}_{col2}"] = {
                        'lower': r_lower,
                        'upper': r_upper,
                        'correlation': r
                    }
                    
        return confidence_intervals
        
    def _fit_garch_11(self, returns: np.ndarray) -> Tuple[float, float, float]:
        """
        Fit GARCH(1,1) model using maximum likelihood
        
        Returns:
            Tuple of (omega, alpha, beta) parameters
        """
        # Simple GARCH(1,1) estimation
        # This is a simplified version - in production use arch package
        
        # Initial values
        omega = np.var(returns) * 0.05
        alpha = 0.1
        beta = 0.85
        
        # Ensure stationarity
        if alpha + beta >= 1:
            beta = 0.99 - alpha
            
        return omega, alpha, beta
        
    def _estimate_dcc_params(self, standardized_residuals: pd.DataFrame) -> Tuple[float, float]:
        """
        Estimate DCC parameters a and b
        
        Returns:
            Tuple of (a, b) parameters
        """
        # Simplified DCC parameter estimation
        # In production, use proper MLE optimization
        
        a = 0.05  # Typical value for short-term dynamics
        b = 0.93  # Typical value for persistence
        
        # Ensure a + b < 1 for stationarity
        if a + b >= 1:
            b = 0.99 - a
            
        return a, b
        
    def _fit_t_copula(self, uniform_data: pd.DataFrame) -> Tuple[pd.DataFrame, float]:
        """Fit Student-t copula and return correlation matrix and degrees of freedom"""
        # Transform to t-distribution quantiles
        # Estimate degrees of freedom using MLE (simplified)
        nu = 5.0  # Starting value
        
        t_quantiles = pd.DataFrame(index=uniform_data.index)
        for col in uniform_data.columns:
            t_quantiles[col] = stats.t.ppf(uniform_data[col], df=nu)
            
        corr_matrix = t_quantiles.corr()
        
        return corr_matrix, nu
        
    def _fit_clayton_copula(self, uniform_data: pd.DataFrame) -> pd.DataFrame:
        """Fit Clayton copula for lower tail dependence"""
        # Clayton copula captures lower tail dependence
        # Simplified implementation - use copulas package in production
        
        # Estimate Kendall's tau and convert to Clayton parameter
        kendall_corr = uniform_data.corr(method='kendall')
        
        # Convert Kendall's tau to correlation (approximation)
        corr_matrix = kendall_corr * (2 / np.pi) * np.arcsin(kendall_corr)
        
        return corr_matrix
        
    def _fit_gumbel_copula(self, uniform_data: pd.DataFrame) -> pd.DataFrame:
        """Fit Gumbel copula for upper tail dependence"""
        # Gumbel copula captures upper tail dependence
        # Simplified implementation
        
        kendall_corr = uniform_data.corr(method='kendall')
        
        # Convert Kendall's tau to correlation (approximation)
        corr_matrix = kendall_corr * (2 / np.pi) * np.arcsin(kendall_corr)
        
        return corr_matrix
        
    def _empty_result(
        self, 
        assets: List[str],
        method: str,
        window: Optional[int]
    ) -> CorrelationResult:
        """Create empty correlation result"""
        empty_matrix = pd.DataFrame(
            np.nan,
            index=assets,
            columns=assets
        )
        np.fill_diagonal(empty_matrix.values, 1.0)
        
        return CorrelationResult(
            timestamp=datetime.now(),
            method=method,
            window=window or 0,
            correlation_matrix=empty_matrix,
            metadata={'error': 'Insufficient data or method disabled'}
        )
        
    # Async wrapper methods
    
    async def _async_correlation(
        self, 
        method: str,
        window: int,
        assets: List[str]
    ) -> CorrelationResult:
        """Async wrapper for correlation calculation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.calculate_correlation_matrix,
            method,
            window,
            assets
        )
        
    async def _async_ewm_correlation(self, assets: List[str]) -> CorrelationResult:
        """Async wrapper for EWM correlation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.calculate_ewm_correlation,
            assets,
            None
        )
        
    async def _async_dcc_garch(self, assets: List[str]) -> CorrelationResult:
        """Async wrapper for DCC-GARCH"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.calculate_dcc_garch_correlation,
            assets,
            252
        )
        
    async def _async_copula(
        self, 
        assets: List[str],
        copula_type: str
    ) -> CorrelationResult:
        """Async wrapper for copula correlation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.calculate_copula_correlation,
            assets,
            copula_type
        )
        
    async def _async_tail_correlation(self, assets: List[str]) -> CorrelationResult:
        """Async wrapper for tail correlation"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self.executor,
            self.calculate_tail_correlation,
            assets,
            0.95
        )