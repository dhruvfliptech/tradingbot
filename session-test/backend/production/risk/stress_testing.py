"""
Stress Testing and Scenario Analysis
Monte Carlo simulations and historical stress testing
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import logging
from scipy import stats
from scipy.stats import norm, t, skew, kurtosis
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor
import warnings

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    """Stress test scenario definition"""
    name: str
    description: str
    market_shock: float  # Market return shock
    volatility_multiplier: float  # Volatility increase
    correlation_increase: float  # Correlation increase
    liquidity_impact: float  # Liquidity reduction
    duration_days: int  # Scenario duration
    probability: float  # Scenario probability
    historical_reference: Optional[str] = None  # Historical event reference


@dataclass
class StressTestResult:
    """Stress test results"""
    scenario_name: str
    portfolio_loss: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    survival_probability: float
    recovery_days: int
    worst_position: str
    best_position: str
    recommendations: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


class StressTester:
    """
    Advanced stress testing and scenario analysis system
    """
    
    def __init__(self,
                 portfolio_data: pd.DataFrame,
                 market_data: Optional[pd.DataFrame] = None):
        """
        Initialize Stress Tester
        
        Args:
            portfolio_data: Portfolio positions and weights
            market_data: Historical market data
        """
        self.portfolio_data = portfolio_data
        self.market_data = market_data
        
        # Predefined scenarios
        self.scenarios = self._initialize_scenarios()
        
        # Results storage
        self.test_results = []
        self.monte_carlo_results = None
        self.historical_stress_results = None
        
        logger.info("Stress Tester initialized")
    
    def _initialize_scenarios(self) -> Dict[str, StressScenario]:
        """Initialize predefined stress scenarios"""
        scenarios = {
            "black_monday_1987": StressScenario(
                name="Black Monday 1987",
                description="Single day 20% market crash",
                market_shock=-0.22,
                volatility_multiplier=5.0,
                correlation_increase=0.5,
                liquidity_impact=0.30,
                duration_days=1,
                probability=0.001,
                historical_reference="October 19, 1987"
            ),
            "financial_crisis_2008": StressScenario(
                name="Financial Crisis 2008",
                description="Systemic financial crisis",
                market_shock=-0.38,
                volatility_multiplier=3.5,
                correlation_increase=0.6,
                liquidity_impact=0.50,
                duration_days=180,
                probability=0.01,
                historical_reference="September 2008 - March 2009"
            ),
            "covid_crash_2020": StressScenario(
                name="COVID-19 Crash 2020",
                description="Pandemic-driven market crash",
                market_shock=-0.34,
                volatility_multiplier=4.0,
                correlation_increase=0.7,
                liquidity_impact=0.40,
                duration_days=30,
                probability=0.005,
                historical_reference="February - March 2020"
            ),
            "flash_crash_2010": StressScenario(
                name="Flash Crash 2010",
                description="Intraday flash crash",
                market_shock=-0.09,
                volatility_multiplier=10.0,
                correlation_increase=0.8,
                liquidity_impact=0.70,
                duration_days=0.01,  # Intraday
                probability=0.002,
                historical_reference="May 6, 2010"
            ),
            "dot_com_burst_2000": StressScenario(
                name="Dot-Com Burst 2000",
                description="Tech bubble burst",
                market_shock=-0.49,
                volatility_multiplier=2.5,
                correlation_increase=0.4,
                liquidity_impact=0.35,
                duration_days=900,
                probability=0.02,
                historical_reference="2000-2002"
            ),
            "ltcm_crisis_1998": StressScenario(
                name="LTCM Crisis 1998",
                description="Hedge fund collapse and contagion",
                market_shock=-0.15,
                volatility_multiplier=3.0,
                correlation_increase=0.5,
                liquidity_impact=0.45,
                duration_days=60,
                probability=0.01,
                historical_reference="August - October 1998"
            ),
            "volmageddon_2018": StressScenario(
                name="Volmageddon 2018",
                description="Volatility spike event",
                market_shock=-0.10,
                volatility_multiplier=8.0,
                correlation_increase=0.6,
                liquidity_impact=0.20,
                duration_days=3,
                probability=0.005,
                historical_reference="February 5, 2018"
            ),
            "moderate_correction": StressScenario(
                name="Moderate Correction",
                description="Standard market correction",
                market_shock=-0.10,
                volatility_multiplier=1.5,
                correlation_increase=0.2,
                liquidity_impact=0.10,
                duration_days=30,
                probability=0.10,
                historical_reference="Typical correction"
            ),
            "severe_recession": StressScenario(
                name="Severe Recession",
                description="Economic recession scenario",
                market_shock=-0.25,
                volatility_multiplier=2.0,
                correlation_increase=0.4,
                liquidity_impact=0.25,
                duration_days=365,
                probability=0.05,
                historical_reference="Recession scenario"
            ),
            "geopolitical_crisis": StressScenario(
                name="Geopolitical Crisis",
                description="Major geopolitical event",
                market_shock=-0.15,
                volatility_multiplier=2.5,
                correlation_increase=0.3,
                liquidity_impact=0.30,
                duration_days=90,
                probability=0.03,
                historical_reference="War/conflict scenario"
            )
        }
        
        return scenarios
    
    def run_scenario_test(self, scenario_name: str) -> StressTestResult:
        """
        Run a specific stress test scenario
        
        Args:
            scenario_name: Name of scenario to test
            
        Returns:
            Stress test results
        """
        if scenario_name not in self.scenarios:
            raise ValueError(f"Unknown scenario: {scenario_name}")
        
        scenario = self.scenarios[scenario_name]
        logger.info(f"Running stress test: {scenario.name}")
        
        # Simulate portfolio under stress
        initial_value = self.portfolio_data['value'].sum()
        
        # Apply market shock
        shocked_returns = self._apply_market_shock(scenario)
        
        # Calculate stressed portfolio value
        stressed_value = initial_value * (1 + shocked_returns.sum())
        portfolio_loss = initial_value - stressed_value
        loss_percent = portfolio_loss / initial_value
        
        # Calculate risk metrics under stress
        stressed_var = self._calculate_stressed_var(shocked_returns, scenario)
        stressed_cvar = self._calculate_stressed_cvar(shocked_returns, scenario)
        
        # Calculate maximum drawdown
        max_drawdown = self._calculate_max_drawdown_scenario(shocked_returns, scenario)
        
        # Survival probability
        survival_prob = self._calculate_survival_probability(loss_percent, scenario)
        
        # Recovery time estimate
        recovery_days = self._estimate_recovery_time(loss_percent, scenario)
        
        # Position analysis
        position_impacts = self._analyze_position_impacts(scenario)
        worst_position = min(position_impacts, key=position_impacts.get)
        best_position = max(position_impacts, key=position_impacts.get)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(scenario, loss_percent)
        
        result = StressTestResult(
            scenario_name=scenario.name,
            portfolio_loss=portfolio_loss,
            max_drawdown=max_drawdown,
            var_95=stressed_var,
            cvar_95=stressed_cvar,
            survival_probability=survival_prob,
            recovery_days=recovery_days,
            worst_position=worst_position,
            best_position=best_position,
            recommendations=recommendations
        )
        
        self.test_results.append(result)
        return result
    
    def run_all_scenarios(self) -> pd.DataFrame:
        """
        Run all predefined stress scenarios
        
        Returns:
            DataFrame with all scenario results
        """
        results = []
        
        for scenario_name in self.scenarios:
            result = self.run_scenario_test(scenario_name)
            results.append({
                'scenario': result.scenario_name,
                'loss': result.portfolio_loss,
                'loss_percent': result.portfolio_loss / self.portfolio_data['value'].sum(),
                'max_drawdown': result.max_drawdown,
                'var_95': result.var_95,
                'cvar_95': result.cvar_95,
                'survival_prob': result.survival_probability,
                'recovery_days': result.recovery_days,
                'probability': self.scenarios[scenario_name].probability
            })
        
        df_results = pd.DataFrame(results)
        df_results['expected_loss'] = df_results['loss'] * df_results['probability']
        df_results = df_results.sort_values('expected_loss', ascending=False)
        
        return df_results
    
    def monte_carlo_simulation(self,
                              n_simulations: int = 10000,
                              time_horizon: int = 252,
                              confidence_levels: List[float] = [0.95, 0.99]) -> Dict[str, Any]:
        """
        Run Monte Carlo simulation for portfolio
        
        Args:
            n_simulations: Number of simulations
            time_horizon: Time horizon in days
            confidence_levels: VaR confidence levels
            
        Returns:
            Simulation results
        """
        logger.info(f"Running Monte Carlo simulation with {n_simulations} scenarios")
        
        # Get historical statistics
        if self.market_data is not None:
            returns = self.market_data.pct_change().dropna()
            mu = returns.mean()
            sigma = returns.cov()
        else:
            # Use simplified assumptions
            n_assets = len(self.portfolio_data)
            mu = pd.Series(np.random.uniform(-0.0005, 0.001, n_assets))
            sigma = pd.DataFrame(np.random.randn(n_assets, n_assets) * 0.01)
            sigma = sigma.T @ sigma  # Make positive definite
        
        # Portfolio weights
        weights = self.portfolio_data['weight'].values if 'weight' in self.portfolio_data else \
                 self.portfolio_data['value'].values / self.portfolio_data['value'].sum()
        
        # Run simulations
        portfolio_returns = []
        
        for _ in range(n_simulations):
            # Generate random returns
            random_returns = np.random.multivariate_normal(
                mu * time_horizon,
                sigma * np.sqrt(time_horizon),
                1
            )[0]
            
            # Calculate portfolio return
            portfolio_return = np.dot(weights, random_returns)
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate statistics
        results = {
            'mean_return': np.mean(portfolio_returns),
            'std_return': np.std(portfolio_returns),
            'skewness': skew(portfolio_returns),
            'kurtosis': kurtosis(portfolio_returns),
            'min_return': np.min(portfolio_returns),
            'max_return': np.max(portfolio_returns),
            'probability_loss': np.mean(portfolio_returns < 0),
            'var': {},
            'cvar': {}
        }
        
        # Calculate VaR and CVaR
        for conf_level in confidence_levels:
            var_value = np.percentile(portfolio_returns, (1 - conf_level) * 100)
            results['var'][f'{int(conf_level*100)}%'] = abs(var_value)
            
            cvar_value = np.mean(portfolio_returns[portfolio_returns <= var_value])
            results['cvar'][f'{int(conf_level*100)}%'] = abs(cvar_value)
        
        # Store full results
        self.monte_carlo_results = {
            'returns': portfolio_returns,
            'statistics': results,
            'n_simulations': n_simulations,
            'time_horizon': time_horizon
        }
        
        return results
    
    def historical_stress_test(self, 
                              crisis_periods: Optional[Dict[str, Tuple[str, str]]] = None) -> pd.DataFrame:
        """
        Test portfolio against historical crisis periods
        
        Args:
            crisis_periods: Dictionary of crisis periods {name: (start_date, end_date)}
            
        Returns:
            Historical stress test results
        """
        if self.market_data is None:
            logger.warning("No historical data available for stress testing")
            return pd.DataFrame()
        
        if crisis_periods is None:
            # Default crisis periods
            crisis_periods = {
                "Black Monday 1987": ("1987-10-01", "1987-11-01"),
                "Asian Crisis 1997": ("1997-07-01", "1997-12-01"),
                "LTCM 1998": ("1998-08-01", "1998-10-01"),
                "Dot-Com Crash": ("2000-03-01", "2002-10-01"),
                "9/11 Attack": ("2001-09-01", "2001-10-01"),
                "Financial Crisis": ("2008-09-01", "2009-03-01"),
                "Euro Crisis": ("2011-08-01", "2011-10-01"),
                "China Slowdown": ("2015-08-01", "2016-02-01"),
                "COVID-19": ("2020-02-01", "2020-04-01")
            }
        
        results = []
        
        for crisis_name, (start_date, end_date) in crisis_periods.items():
            try:
                # Get crisis period data
                crisis_data = self.market_data[start_date:end_date]
                
                if crisis_data.empty:
                    continue
                
                # Calculate crisis returns
                crisis_returns = crisis_data.pct_change().dropna()
                
                # Portfolio performance during crisis
                weights = self.portfolio_data['weight'].values if 'weight' in self.portfolio_data else \
                         self.portfolio_data['value'].values / self.portfolio_data['value'].sum()
                
                portfolio_crisis_returns = crisis_returns @ weights
                
                # Calculate metrics
                total_return = (1 + portfolio_crisis_returns).prod() - 1
                max_drawdown = self._calculate_max_drawdown(portfolio_crisis_returns)
                volatility = portfolio_crisis_returns.std() * np.sqrt(252)
                worst_day = portfolio_crisis_returns.min()
                best_day = portfolio_crisis_returns.max()
                
                results.append({
                    'crisis': crisis_name,
                    'start': start_date,
                    'end': end_date,
                    'total_return': total_return,
                    'max_drawdown': max_drawdown,
                    'volatility': volatility,
                    'worst_day': worst_day,
                    'best_day': best_day,
                    'duration_days': len(crisis_data)
                })
                
            except Exception as e:
                logger.warning(f"Could not process crisis period {crisis_name}: {e}")
        
        df_results = pd.DataFrame(results)
        self.historical_stress_results = df_results
        
        return df_results
    
    def tail_risk_analysis(self) -> Dict[str, Any]:
        """
        Analyze tail risk characteristics
        
        Returns:
            Tail risk metrics
        """
        if self.monte_carlo_results is None:
            self.monte_carlo_simulation()
        
        returns = self.monte_carlo_results['returns']
        
        # Define tail thresholds
        left_tail_threshold = np.percentile(returns, 5)
        right_tail_threshold = np.percentile(returns, 95)
        
        # Tail returns
        left_tail = returns[returns <= left_tail_threshold]
        right_tail = returns[returns >= right_tail_threshold]
        
        # Tail statistics
        tail_metrics = {
            'left_tail': {
                'mean': np.mean(left_tail),
                'std': np.std(left_tail),
                'min': np.min(left_tail),
                'expected_shortfall': np.mean(left_tail),
                'tail_index': self._estimate_tail_index(left_tail)
            },
            'right_tail': {
                'mean': np.mean(right_tail),
                'std': np.std(right_tail),
                'max': np.max(right_tail),
                'expected_gain': np.mean(right_tail),
                'tail_index': self._estimate_tail_index(right_tail)
            },
            'tail_ratio': len(left_tail) / len(right_tail) if len(right_tail) > 0 else np.inf,
            'tail_dependence': self._calculate_tail_dependence(returns)
        }
        
        return tail_metrics
    
    def generate_stress_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive stress testing report
        
        Returns:
            Complete stress test report
        """
        report = {
            'timestamp': datetime.now().isoformat(),
            'portfolio_value': self.portfolio_data['value'].sum(),
            'scenario_tests': [],
            'monte_carlo': None,
            'historical_stress': None,
            'tail_risk': None,
            'recommendations': [],
            'risk_score': 0
        }
        
        # Scenario test results
        if self.test_results:
            for result in self.test_results:
                report['scenario_tests'].append({
                    'scenario': result.scenario_name,
                    'loss': result.portfolio_loss,
                    'max_drawdown': result.max_drawdown,
                    'survival_prob': result.survival_probability,
                    'recovery_days': result.recovery_days,
                    'recommendations': result.recommendations
                })
        
        # Monte Carlo results
        if self.monte_carlo_results:
            report['monte_carlo'] = self.monte_carlo_results['statistics']
        
        # Historical stress results
        if self.historical_stress_results is not None:
            report['historical_stress'] = self.historical_stress_results.to_dict('records')
        
        # Tail risk analysis
        report['tail_risk'] = self.tail_risk_analysis()
        
        # Overall risk score (0-100)
        report['risk_score'] = self._calculate_overall_risk_score()
        
        # Generate overall recommendations
        report['recommendations'] = self._generate_overall_recommendations(report['risk_score'])
        
        return report
    
    def _apply_market_shock(self, scenario: StressScenario) -> pd.Series:
        """Apply market shock to portfolio"""
        # Simplified shock application
        base_shock = scenario.market_shock
        
        # Add volatility component
        volatility_component = np.random.normal(0, 0.01 * scenario.volatility_multiplier, 
                                               len(self.portfolio_data))
        
        # Correlation adjustment
        correlation_adjustment = scenario.correlation_increase * base_shock * 0.5
        
        # Total shock per position
        shocks = pd.Series(
            base_shock + volatility_component + correlation_adjustment,
            index=self.portfolio_data.index
        )
        
        return shocks
    
    def _calculate_stressed_var(self, returns: pd.Series, scenario: StressScenario) -> float:
        """Calculate VaR under stress"""
        # Adjust for increased volatility
        stressed_vol = returns.std() * scenario.volatility_multiplier
        
        # Parametric VaR with fat tails
        var_95 = norm.ppf(0.05) * stressed_vol * np.sqrt(scenario.duration_days)
        
        return abs(var_95) * self.portfolio_data['value'].sum()
    
    def _calculate_stressed_cvar(self, returns: pd.Series, scenario: StressScenario) -> float:
        """Calculate CVaR under stress"""
        var_threshold = np.percentile(returns, 5)
        cvar = returns[returns <= var_threshold].mean()
        
        # Adjust for scenario severity
        stressed_cvar = cvar * scenario.volatility_multiplier
        
        return abs(stressed_cvar) * self.portfolio_data['value'].sum()
    
    def _calculate_max_drawdown_scenario(self, returns: pd.Series, scenario: StressScenario) -> float:
        """Calculate maximum drawdown for scenario"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        
        # Adjust for scenario impact
        max_dd = drawdown.min() * (1 + scenario.liquidity_impact)
        
        return abs(max_dd)
    
    def _calculate_max_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        cumulative_returns = (1 + returns).cumprod()
        peak = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - peak) / peak
        return abs(drawdown.min())
    
    def _calculate_survival_probability(self, loss_percent: float, scenario: StressScenario) -> float:
        """Calculate probability of surviving the scenario"""
        # Base survival probability
        if loss_percent < 0.10:
            base_prob = 0.95
        elif loss_percent < 0.20:
            base_prob = 0.80
        elif loss_percent < 0.30:
            base_prob = 0.60
        elif loss_percent < 0.50:
            base_prob = 0.30
        else:
            base_prob = 0.10
        
        # Adjust for scenario duration
        duration_factor = max(0.5, 1 - scenario.duration_days / 365)
        
        # Adjust for liquidity
        liquidity_factor = 1 - scenario.liquidity_impact * 0.5
        
        return base_prob * duration_factor * liquidity_factor
    
    def _estimate_recovery_time(self, loss_percent: float, scenario: StressScenario) -> int:
        """Estimate recovery time in days"""
        # Base recovery time
        base_recovery = loss_percent * 1000  # Rough estimate
        
        # Adjust for market conditions
        recovery_days = base_recovery * (1 + scenario.volatility_multiplier * 0.2)
        
        # Minimum recovery time
        recovery_days = max(recovery_days, scenario.duration_days)
        
        return int(recovery_days)
    
    def _analyze_position_impacts(self, scenario: StressScenario) -> Dict[str, float]:
        """Analyze impact on individual positions"""
        impacts = {}
        
        for idx, position in self.portfolio_data.iterrows():
            # Simplified impact calculation
            base_impact = scenario.market_shock
            
            # Add position-specific factors
            if 'beta' in position:
                base_impact *= position['beta']
            
            if 'sector' in position:
                # Sector-specific adjustments
                if position['sector'] == 'Technology':
                    base_impact *= 1.2
                elif position['sector'] == 'Utilities':
                    base_impact *= 0.7
            
            impacts[position.get('symbol', str(idx))] = base_impact
        
        return impacts
    
    def _generate_recommendations(self, scenario: StressScenario, loss_percent: float) -> List[str]:
        """Generate scenario-specific recommendations"""
        recommendations = []
        
        if loss_percent > 0.20:
            recommendations.append("Consider reducing overall portfolio leverage")
            recommendations.append("Implement stop-loss orders for all positions")
        
        if scenario.volatility_multiplier > 3:
            recommendations.append("Increase cash reserves to 20-30% of portfolio")
            recommendations.append("Consider volatility hedges (VIX calls or put spreads)")
        
        if scenario.correlation_increase > 0.5:
            recommendations.append("Diversify across uncorrelated asset classes")
            recommendations.append("Reduce concentration in highly correlated positions")
        
        if scenario.liquidity_impact > 0.3:
            recommendations.append("Focus on liquid assets only")
            recommendations.append("Avoid illiquid positions during stress periods")
        
        if scenario.duration_days > 180:
            recommendations.append("Prepare for extended market stress")
            recommendations.append("Consider defensive sector rotation")
        
        return recommendations
    
    def _estimate_tail_index(self, tail_data: np.ndarray) -> float:
        """Estimate tail index using Hill estimator"""
        if len(tail_data) < 10:
            return np.nan
        
        sorted_data = np.sort(np.abs(tail_data))[::-1]
        k = min(len(sorted_data) // 4, 100)  # Use top 25% or max 100 observations
        
        if k < 2:
            return np.nan
        
        hill_estimate = k / np.sum(np.log(sorted_data[:k] / sorted_data[k]))
        
        return hill_estimate
    
    def _calculate_tail_dependence(self, returns: np.ndarray) -> float:
        """Calculate tail dependence coefficient"""
        # Simplified tail dependence measure
        threshold = np.percentile(np.abs(returns), 95)
        extreme_events = np.abs(returns) > threshold
        
        if np.sum(extreme_events) < 2:
            return 0
        
        # Calculate correlation in tails
        tail_correlation = np.corrcoef(returns[:-1][extreme_events[:-1]], 
                                      returns[1:][extreme_events[1:]])[0, 1]
        
        return tail_correlation if not np.isnan(tail_correlation) else 0
    
    def _calculate_overall_risk_score(self) -> float:
        """Calculate overall portfolio risk score (0-100)"""
        score = 0
        
        # Factor 1: Scenario test results (0-40)
        if self.test_results:
            avg_survival = np.mean([r.survival_probability for r in self.test_results])
            score += (1 - avg_survival) * 40
        
        # Factor 2: Monte Carlo results (0-30)
        if self.monte_carlo_results:
            prob_loss = self.monte_carlo_results['statistics']['probability_loss']
            score += prob_loss * 30
        
        # Factor 3: Historical stress (0-30)
        if self.historical_stress_results is not None and not self.historical_stress_results.empty:
            avg_drawdown = self.historical_stress_results['max_drawdown'].mean()
            score += min(30, abs(avg_drawdown) * 100)
        
        return min(100, score)
    
    def _generate_overall_recommendations(self, risk_score: float) -> List[str]:
        """Generate overall recommendations based on risk score"""
        recommendations = []
        
        if risk_score > 80:
            recommendations.extend([
                "CRITICAL: Portfolio has extreme risk exposure",
                "Immediately reduce leverage to maximum 2x",
                "Increase cash position to minimum 40%",
                "Implement comprehensive hedging strategy",
                "Review and tighten all stop-loss orders"
            ])
        elif risk_score > 60:
            recommendations.extend([
                "HIGH RISK: Portfolio vulnerable to market stress",
                "Reduce position sizes by 30-50%",
                "Diversify across uncorrelated assets",
                "Consider protective puts for large positions",
                "Increase monitoring frequency"
            ])
        elif risk_score > 40:
            recommendations.extend([
                "MODERATE RISK: Some vulnerability to adverse events",
                "Review position concentration",
                "Consider partial hedging for tail risk",
                "Maintain adequate cash reserves (15-20%)",
                "Regular stress testing recommended"
            ])
        else:
            recommendations.extend([
                "ACCEPTABLE RISK: Portfolio reasonably protected",
                "Continue regular monitoring",
                "Maintain current risk management practices",
                "Consider opportunistic hedging",
                "Document risk limits and stick to them"
            ])
        
        return recommendations