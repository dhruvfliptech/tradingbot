"""
Portfolio-Level Risk Metrics and Analysis
Calculates VaR, CVaR, correlations, and portfolio optimization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from scipy import stats, optimize
from scipy.stats import norm, t
import warnings

logger = logging.getLogger(__name__)


@dataclass
class PortfolioMetrics:
    """Portfolio risk metrics"""
    total_value: float
    var_95: float  # 95% VaR
    var_99: float  # 99% VaR
    cvar_95: float  # 95% CVaR (Expected Shortfall)
    cvar_99: float  # 99% CVaR
    beta: float  # Portfolio beta vs market
    correlation_matrix: pd.DataFrame
    concentration_hhi: float  # Herfindahl-Hirschman Index
    max_drawdown: float
    current_drawdown: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    information_ratio: float
    treynor_ratio: float
    tracking_error: float
    downside_deviation: float
    ulcer_index: float
    recovery_time: int  # Days to recover from drawdown
    risk_adjusted_return: float


class PortfolioRiskAnalyzer:
    """
    Advanced portfolio risk analysis and optimization
    """
    
    def __init__(self, 
                 positions: Dict[str, Dict],
                 market_data: Optional[pd.DataFrame] = None,
                 risk_free_rate: float = 0.03):
        """
        Initialize Portfolio Risk Analyzer
        
        Args:
            positions: Dictionary of positions {symbol: {size, entry_price, current_price}}
            market_data: Historical market data
            risk_free_rate: Annual risk-free rate
        """
        self.positions = positions
        self.market_data = market_data
        self.risk_free_rate = risk_free_rate
        
        # Historical data storage
        self.returns_data = pd.DataFrame()
        self.correlation_matrix = pd.DataFrame()
        self.covariance_matrix = pd.DataFrame()
        
        # Risk calculations cache
        self.portfolio_metrics = None
        self.var_scenarios = {}
        self.stress_test_results = {}
        
        logger.info(f"Portfolio Risk Analyzer initialized with {len(positions)} positions")
    
    def calculate_portfolio_metrics(self, 
                                  lookback_days: int = 252,
                                  confidence_levels: List[float] = [0.95, 0.99]) -> PortfolioMetrics:
        """
        Calculate comprehensive portfolio risk metrics
        
        Args:
            lookback_days: Days of historical data to use
            confidence_levels: VaR confidence levels
            
        Returns:
            Portfolio metrics
        """
        # Calculate portfolio value
        portfolio_value = sum(
            pos['size'] * pos['current_price'] 
            for pos in self.positions.values()
        )
        
        # Get returns data
        returns = self._calculate_returns(lookback_days)
        
        if returns.empty:
            logger.warning("No returns data available")
            return self._get_default_metrics(portfolio_value)
        
        # Calculate VaR and CVaR
        var_95 = self._calculate_var(returns, 0.95)
        var_99 = self._calculate_var(returns, 0.99)
        cvar_95 = self._calculate_cvar(returns, 0.95)
        cvar_99 = self._calculate_cvar(returns, 0.99)
        
        # Calculate correlation and covariance
        self.correlation_matrix = returns.corr()
        self.covariance_matrix = returns.cov()
        
        # Calculate portfolio statistics
        portfolio_returns = self._calculate_portfolio_returns(returns)
        
        # Risk metrics
        sharpe = self._calculate_sharpe_ratio(portfolio_returns)
        sortino = self._calculate_sortino_ratio(portfolio_returns)
        calmar = self._calculate_calmar_ratio(portfolio_returns)
        
        # Drawdown analysis
        max_dd, current_dd, recovery_time = self._calculate_drawdown_metrics(portfolio_returns)
        
        # Concentration metrics
        hhi = self._calculate_concentration_hhi()
        
        # Beta calculation
        beta = self._calculate_portfolio_beta(portfolio_returns)
        
        # Additional metrics
        info_ratio = self._calculate_information_ratio(portfolio_returns)
        treynor = self._calculate_treynor_ratio(portfolio_returns, beta)
        tracking_error = self._calculate_tracking_error(portfolio_returns)
        downside_dev = self._calculate_downside_deviation(portfolio_returns)
        ulcer = self._calculate_ulcer_index(portfolio_returns)
        
        metrics = PortfolioMetrics(
            total_value=portfolio_value,
            var_95=var_95 * portfolio_value,
            var_99=var_99 * portfolio_value,
            cvar_95=cvar_95 * portfolio_value,
            cvar_99=cvar_99 * portfolio_value,
            beta=beta,
            correlation_matrix=self.correlation_matrix,
            concentration_hhi=hhi,
            max_drawdown=max_dd,
            current_drawdown=current_dd,
            sharpe_ratio=sharpe,
            sortino_ratio=sortino,
            calmar_ratio=calmar,
            information_ratio=info_ratio,
            treynor_ratio=treynor,
            tracking_error=tracking_error,
            downside_deviation=downside_dev,
            ulcer_index=ulcer,
            recovery_time=recovery_time,
            risk_adjusted_return=portfolio_returns.mean() * 252 / (portfolio_returns.std() * np.sqrt(252))
        )
        
        self.portfolio_metrics = metrics
        return metrics
    
    def calculate_var_parametric(self, 
                                confidence_level: float = 0.95,
                                time_horizon: int = 1) -> float:
        """
        Calculate VaR using parametric method (assumes normal distribution)
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon: Time horizon in days
            
        Returns:
            VaR value
        """
        portfolio_returns = self._calculate_portfolio_returns(self._calculate_returns())
        
        if portfolio_returns.empty:
            return 0
        
        # Calculate portfolio statistics
        mu = portfolio_returns.mean()
        sigma = portfolio_returns.std()
        
        # Calculate VaR
        alpha = 1 - confidence_level
        z_score = norm.ppf(alpha)
        var = -(mu * time_horizon - z_score * sigma * np.sqrt(time_horizon))
        
        return var
    
    def calculate_var_historical(self, 
                               confidence_level: float = 0.95,
                               lookback_days: int = 252) -> float:
        """
        Calculate VaR using historical simulation
        
        Args:
            confidence_level: Confidence level
            lookback_days: Historical period
            
        Returns:
            VaR value
        """
        returns = self._calculate_portfolio_returns(self._calculate_returns(lookback_days))
        
        if returns.empty:
            return 0
        
        # Calculate percentile
        var = np.percentile(returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def calculate_var_monte_carlo(self,
                                 confidence_level: float = 0.95,
                                 n_simulations: int = 10000,
                                 time_horizon: int = 1) -> float:
        """
        Calculate VaR using Monte Carlo simulation
        
        Args:
            confidence_level: Confidence level
            n_simulations: Number of simulations
            time_horizon: Time horizon in days
            
        Returns:
            VaR value
        """
        returns = self._calculate_portfolio_returns(self._calculate_returns())
        
        if returns.empty:
            return 0
        
        # Fit distribution
        mu = returns.mean()
        sigma = returns.std()
        
        # Run simulations
        simulated_returns = np.random.normal(mu, sigma, (n_simulations, time_horizon))
        portfolio_returns = np.sum(simulated_returns, axis=1)
        
        # Calculate VaR
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def calculate_marginal_var(self, symbol: str) -> float:
        """
        Calculate marginal VaR for a specific position
        
        Args:
            symbol: Position symbol
            
        Returns:
            Marginal VaR
        """
        if symbol not in self.positions:
            return 0
        
        # Calculate portfolio VaR with position
        var_with = self.calculate_var_parametric()
        
        # Calculate portfolio VaR without position
        temp_positions = self.positions.copy()
        del temp_positions[symbol]
        self.positions = temp_positions
        var_without = self.calculate_var_parametric()
        
        # Restore positions
        self.positions[symbol] = temp_positions[symbol]
        
        # Marginal VaR
        marginal_var = var_with - var_without
        
        return marginal_var
    
    def calculate_component_var(self) -> Dict[str, float]:
        """
        Calculate component VaR for all positions
        
        Returns:
            Dictionary of component VaRs
        """
        component_vars = {}
        
        for symbol in self.positions:
            marginal_var = self.calculate_marginal_var(symbol)
            position_value = self.positions[symbol]['size'] * self.positions[symbol]['current_price']
            portfolio_value = sum(p['size'] * p['current_price'] for p in self.positions.values())
            
            component_var = marginal_var * (position_value / portfolio_value)
            component_vars[symbol] = component_var
        
        return component_vars
    
    def optimize_portfolio(self,
                          target_return: Optional[float] = None,
                          max_risk: Optional[float] = None) -> Dict[str, float]:
        """
        Optimize portfolio weights using mean-variance optimization
        
        Args:
            target_return: Target portfolio return
            max_risk: Maximum acceptable risk (volatility)
            
        Returns:
            Optimal weights
        """
        symbols = list(self.positions.keys())
        n_assets = len(symbols)
        
        if n_assets < 2:
            return {symbols[0]: 1.0} if symbols else {}
        
        # Get returns
        returns = self._calculate_returns()
        if returns.empty:
            # Equal weight if no data
            return {symbol: 1.0 / n_assets for symbol in symbols}
        
        # Calculate expected returns and covariance
        expected_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Optimization constraints
        constraints = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Weights sum to 1
        
        if target_return is not None:
            constraints.append({
                'type': 'eq',
                'fun': lambda x: np.dot(x, expected_returns) - target_return
            })
        
        if max_risk is not None:
            constraints.append({
                'type': 'ineq',
                'fun': lambda x: max_risk - np.sqrt(np.dot(x, np.dot(cov_matrix, x)))
            })
        
        # Bounds (0 to 1 for each weight)
        bounds = tuple((0, 1) for _ in range(n_assets))
        
        # Initial guess (equal weights)
        x0 = np.array([1.0 / n_assets] * n_assets)
        
        # Objective function (minimize portfolio variance)
        def portfolio_variance(weights):
            return np.dot(weights, np.dot(cov_matrix, weights))
        
        # Optimize
        result = optimize.minimize(
            portfolio_variance,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        if result.success:
            optimal_weights = {symbols[i]: result.x[i] for i in range(n_assets)}
        else:
            logger.warning("Optimization failed, using equal weights")
            optimal_weights = {symbol: 1.0 / n_assets for symbol in symbols}
        
        return optimal_weights
    
    def calculate_efficient_frontier(self, n_points: int = 100) -> pd.DataFrame:
        """
        Calculate efficient frontier
        
        Args:
            n_points: Number of points on the frontier
            
        Returns:
            DataFrame with risk-return combinations
        """
        returns = self._calculate_returns()
        if returns.empty:
            return pd.DataFrame()
        
        expected_returns = returns.mean() * 252  # Annualized
        
        # Generate target returns
        min_return = expected_returns.min()
        max_return = expected_returns.max()
        target_returns = np.linspace(min_return, max_return, n_points)
        
        frontier_data = []
        
        for target_return in target_returns:
            weights = self.optimize_portfolio(target_return=target_return)
            
            # Calculate portfolio metrics
            portfolio_return = sum(weights[s] * expected_returns[s] for s in weights)
            portfolio_risk = self._calculate_portfolio_volatility(weights, returns)
            
            frontier_data.append({
                'return': portfolio_return,
                'risk': portfolio_risk,
                'sharpe': (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0,
                'weights': weights
            })
        
        return pd.DataFrame(frontier_data)
    
    def stress_test(self, scenarios: Optional[Dict[str, Dict]] = None) -> Dict[str, Dict]:
        """
        Run stress tests on portfolio
        
        Args:
            scenarios: Custom stress scenarios
            
        Returns:
            Stress test results
        """
        if scenarios is None:
            # Default stress scenarios
            scenarios = {
                "market_crash": {
                    "market_change": -0.20,  # 20% market drop
                    "volatility_multiplier": 3.0,
                    "correlation_increase": 0.3
                },
                "flash_crash": {
                    "market_change": -0.10,  # 10% instant drop
                    "volatility_multiplier": 5.0,
                    "correlation_increase": 0.5
                },
                "black_swan": {
                    "market_change": -0.35,  # 35% drop
                    "volatility_multiplier": 4.0,
                    "correlation_increase": 0.6
                },
                "liquidity_crisis": {
                    "market_change": -0.15,
                    "volatility_multiplier": 2.0,
                    "correlation_increase": 0.4,
                    "liquidity_discount": 0.10  # 10% liquidity discount
                },
                "sector_collapse": {
                    "market_change": -0.08,
                    "sector_impact": -0.30,  # 30% sector drop
                    "volatility_multiplier": 2.5,
                    "correlation_increase": 0.2
                }
            }
        
        results = {}
        current_value = sum(p['size'] * p['current_price'] for p in self.positions.values())
        
        for scenario_name, scenario_params in scenarios.items():
            # Apply scenario
            scenario_value = current_value
            
            # Market impact
            if 'market_change' in scenario_params:
                scenario_value *= (1 + scenario_params['market_change'])
            
            # Liquidity impact
            if 'liquidity_discount' in scenario_params:
                scenario_value *= (1 - scenario_params['liquidity_discount'])
            
            # Calculate stressed VaR
            stressed_var = self.calculate_var_parametric() * scenario_params.get('volatility_multiplier', 1.0)
            
            # Calculate losses
            loss = current_value - scenario_value
            loss_percent = loss / current_value
            
            results[scenario_name] = {
                "scenario_value": scenario_value,
                "loss": loss,
                "loss_percent": loss_percent,
                "stressed_var": stressed_var,
                "survival_probability": self._calculate_survival_probability(loss_percent),
                "recovery_time_estimate": self._estimate_recovery_time(loss_percent)
            }
        
        self.stress_test_results = results
        return results
    
    def _calculate_returns(self, lookback_days: int = 252) -> pd.DataFrame:
        """Calculate historical returns for all positions"""
        if self.market_data is None or self.market_data.empty:
            # Generate synthetic returns for demonstration
            symbols = list(self.positions.keys())
            dates = pd.date_range(end=datetime.now(), periods=lookback_days, freq='D')
            
            returns = pd.DataFrame(index=dates)
            for symbol in symbols:
                # Synthetic returns with some correlation
                base_returns = np.random.normal(0.0005, 0.02, lookback_days)
                market_component = np.random.normal(0, 0.01, lookback_days)
                returns[symbol] = base_returns + market_component * 0.5
            
            return returns
        
        # Use actual market data
        return self.market_data.pct_change().dropna()
    
    def _calculate_portfolio_returns(self, asset_returns: pd.DataFrame) -> pd.Series:
        """Calculate portfolio returns from asset returns"""
        if asset_returns.empty:
            return pd.Series()
        
        # Calculate weights
        total_value = sum(p['size'] * p['current_price'] for p in self.positions.values())
        weights = {}
        
        for symbol in self.positions:
            if symbol in asset_returns.columns:
                position_value = self.positions[symbol]['size'] * self.positions[symbol]['current_price']
                weights[symbol] = position_value / total_value
        
        # Calculate weighted returns
        portfolio_returns = pd.Series(0, index=asset_returns.index)
        for symbol, weight in weights.items():
            if symbol in asset_returns.columns:
                portfolio_returns += asset_returns[symbol] * weight
        
        return portfolio_returns
    
    def _calculate_var(self, returns: pd.DataFrame, confidence_level: float) -> float:
        """Calculate Value at Risk"""
        portfolio_returns = self._calculate_portfolio_returns(returns)
        if portfolio_returns.empty:
            return 0
        
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        return abs(var)
    
    def _calculate_cvar(self, returns: pd.DataFrame, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        portfolio_returns = self._calculate_portfolio_returns(returns)
        if portfolio_returns.empty:
            return 0
        
        var = np.percentile(portfolio_returns, (1 - confidence_level) * 100)
        cvar = portfolio_returns[portfolio_returns <= var].mean()
        return abs(cvar)
    
    def _calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        """Calculate Sharpe ratio"""
        if returns.empty or returns.std() == 0:
            return 0
        
        excess_returns = returns - self.risk_free_rate / 252
        return np.sqrt(252) * excess_returns.mean() / returns.std()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio"""
        if returns.empty:
            return 0
        
        excess_returns = returns - self.risk_free_rate / 252
        downside_returns = returns[returns < 0]
        
        if downside_returns.empty or downside_returns.std() == 0:
            return 0
        
        return np.sqrt(252) * excess_returns.mean() / downside_returns.std()
    
    def _calculate_calmar_ratio(self, returns: pd.Series) -> float:
        """Calculate Calmar ratio"""
        if returns.empty:
            return 0
        
        cumulative_returns = (1 + returns).cumprod()
        max_drawdown = self._calculate_max_drawdown(cumulative_returns)
        
        if max_drawdown == 0:
            return 0
        
        annual_return = returns.mean() * 252
        return annual_return / abs(max_drawdown)
    
    def _calculate_max_drawdown(self, cumulative_returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        if cumulative_returns.empty:
            return 0
        
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        return drawdown.min()
    
    def _calculate_drawdown_metrics(self, returns: pd.Series) -> Tuple[float, float, int]:
        """Calculate drawdown metrics"""
        if returns.empty:
            return 0, 0, 0
        
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = (cumulative_returns - peak) / peak
        
        max_drawdown = drawdown.min()
        current_drawdown = drawdown.iloc[-1] if not drawdown.empty else 0
        
        # Recovery time
        recovery_time = 0
        if max_drawdown < 0:
            max_dd_idx = drawdown.idxmin()
            recovery_idx = drawdown[max_dd_idx:][drawdown[max_dd_idx:] >= -0.01].index
            if len(recovery_idx) > 0:
                recovery_time = (recovery_idx[0] - max_dd_idx).days
        
        return max_drawdown, current_drawdown, recovery_time
    
    def _calculate_concentration_hhi(self) -> float:
        """Calculate Herfindahl-Hirschman Index for concentration"""
        total_value = sum(p['size'] * p['current_price'] for p in self.positions.values())
        
        if total_value == 0:
            return 0
        
        hhi = 0
        for position in self.positions.values():
            weight = (position['size'] * position['current_price']) / total_value
            hhi += weight ** 2
        
        return hhi
    
    def _calculate_portfolio_beta(self, portfolio_returns: pd.Series) -> float:
        """Calculate portfolio beta vs market"""
        if portfolio_returns.empty:
            return 1.0
        
        # Use SPY as market proxy (simplified)
        market_returns = pd.Series(np.random.normal(0.0004, 0.015, len(portfolio_returns)), 
                                 index=portfolio_returns.index)
        
        covariance = portfolio_returns.cov(market_returns)
        market_variance = market_returns.var()
        
        if market_variance == 0:
            return 1.0
        
        return covariance / market_variance
    
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio"""
        if returns.empty:
            return 0
        
        # Benchmark returns (simplified)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.012, len(returns)), 
                                     index=returns.index)
        
        active_returns = returns - benchmark_returns
        tracking_error = active_returns.std()
        
        if tracking_error == 0:
            return 0
        
        return np.sqrt(252) * active_returns.mean() / tracking_error
    
    def _calculate_treynor_ratio(self, returns: pd.Series, beta: float) -> float:
        """Calculate Treynor ratio"""
        if returns.empty or beta == 0:
            return 0
        
        excess_returns = returns.mean() - self.risk_free_rate / 252
        return excess_returns * 252 / beta
    
    def _calculate_tracking_error(self, returns: pd.Series) -> float:
        """Calculate tracking error"""
        if returns.empty:
            return 0
        
        # Benchmark returns (simplified)
        benchmark_returns = pd.Series(np.random.normal(0.0003, 0.012, len(returns)), 
                                     index=returns.index)
        
        active_returns = returns - benchmark_returns
        return active_returns.std() * np.sqrt(252)
    
    def _calculate_downside_deviation(self, returns: pd.Series) -> float:
        """Calculate downside deviation"""
        if returns.empty:
            return 0
        
        target_return = 0  # Can be adjusted
        downside_returns = returns[returns < target_return]
        
        if downside_returns.empty:
            return 0
        
        return downside_returns.std() * np.sqrt(252)
    
    def _calculate_ulcer_index(self, returns: pd.Series) -> float:
        """Calculate Ulcer Index"""
        if returns.empty:
            return 0
        
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding(min_periods=1).max()
        drawdown = ((cumulative_returns - peak) / peak) * 100
        
        ulcer_index = np.sqrt(np.mean(drawdown ** 2))
        return ulcer_index
    
    def _calculate_portfolio_volatility(self, weights: Dict[str, float], returns: pd.DataFrame) -> float:
        """Calculate portfolio volatility"""
        if returns.empty:
            return 0
        
        # Filter returns for symbols in weights
        symbols = [s for s in weights.keys() if s in returns.columns]
        if not symbols:
            return 0
        
        returns_subset = returns[symbols]
        cov_matrix = returns_subset.cov() * 252  # Annualized
        
        # Convert weights to array
        w = np.array([weights[s] for s in symbols])
        
        # Portfolio variance
        portfolio_variance = np.dot(w, np.dot(cov_matrix, w))
        
        return np.sqrt(portfolio_variance)
    
    def _calculate_survival_probability(self, loss_percent: float) -> float:
        """Estimate probability of surviving a loss"""
        # Simplified calculation based on loss magnitude
        if loss_percent <= 0.05:
            return 0.95
        elif loss_percent <= 0.10:
            return 0.85
        elif loss_percent <= 0.20:
            return 0.70
        elif loss_percent <= 0.30:
            return 0.50
        else:
            return max(0.20, 1 - loss_percent)
    
    def _estimate_recovery_time(self, loss_percent: float) -> int:
        """Estimate recovery time in days"""
        # Simplified estimation
        daily_expected_return = 0.0003  # 0.03% daily
        
        if daily_expected_return <= 0:
            return 999  # Cannot recover with negative returns
        
        # Time to recover = ln(1/(1-loss)) / ln(1+return)
        recovery_days = np.log(1 / (1 - loss_percent)) / np.log(1 + daily_expected_return)
        
        return int(recovery_days)
    
    def _get_default_metrics(self, portfolio_value: float) -> PortfolioMetrics:
        """Get default metrics when no data available"""
        return PortfolioMetrics(
            total_value=portfolio_value,
            var_95=0,
            var_99=0,
            cvar_95=0,
            cvar_99=0,
            beta=1.0,
            correlation_matrix=pd.DataFrame(),
            concentration_hhi=0,
            max_drawdown=0,
            current_drawdown=0,
            sharpe_ratio=0,
            sortino_ratio=0,
            calmar_ratio=0,
            information_ratio=0,
            treynor_ratio=0,
            tracking_error=0,
            downside_deviation=0,
            ulcer_index=0,
            recovery_time=0,
            risk_adjusted_return=0
        )