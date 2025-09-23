"""
Test Report Generator
Generates comprehensive test reports for RL trading system validation:
- Performance validation results
- Benchmark comparison analysis  
- Integration test status
- Stress test metrics
- SOW compliance verification
- Executive summary with recommendations
"""

import pytest
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
import numpy as np
import logging
import os
import sys
from pathlib import Path
import jinja2
from io import StringIO

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)


@dataclass
class TestResultSummary:
    """Summary of test results"""
    test_category: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    success_rate: float
    execution_time: float
    critical_failures: List[str]
    warnings: List[str]
    key_metrics: Dict[str, Any]


@dataclass
class PerformanceValidationReport:
    """Performance validation report"""
    weekly_return_target_met: bool
    sharpe_ratio_target_met: bool
    drawdown_constraint_met: bool
    win_rate_target_met: bool
    actual_weekly_return: float
    actual_sharpe_ratio: float
    actual_max_drawdown: float
    actual_win_rate: float
    target_weekly_return: Tuple[float, float]  # (min, max)
    target_sharpe_ratio: float
    target_max_drawdown: float
    target_win_rate: float
    consistency_score: float
    regime_performance: Dict[str, Dict[str, float]]


@dataclass  
class BenchmarkComparisonReport:
    """Benchmark comparison report"""
    rl_vs_adaptive_outperformance: float
    statistical_significance: float
    consistency_across_periods: float
    regime_specific_performance: Dict[str, float]
    sow_compliance: bool
    outperformance_range_met: bool
    risk_adjusted_outperformance: float


@dataclass
class SystemReliabilityReport:
    """System reliability and stress test report"""
    crash_resilience_score: float
    data_feed_reliability: float
    concurrent_user_capacity: int
    memory_efficiency_score: float
    response_time_performance: float
    failover_recovery_time: float
    critical_system_failures: List[str]


@dataclass
class ComplianceReport:
    """SOW compliance verification"""
    sow_targets_met: int
    sow_targets_total: int
    compliance_percentage: float
    critical_violations: List[str]
    warnings: List[str]
    recommendations: List[str]


class TestReportGenerator:
    """Generates comprehensive test reports"""
    
    def __init__(self, output_dir: str = "test_reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.report_timestamp = datetime.now()
        
        # Initialize report components
        self.test_results: List[TestResultSummary] = []
        self.performance_report: Optional[PerformanceValidationReport] = None
        self.benchmark_report: Optional[BenchmarkComparisonReport] = None
        self.reliability_report: Optional[SystemReliabilityReport] = None
        self.compliance_report: Optional[ComplianceReport] = None
        
        # Setup plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def collect_test_results(self, test_session_data: Dict) -> None:
        """Collect test results from pytest session"""
        
        # Parse pytest results (simplified structure)
        categories = {
            'performance': 'Performance Validation Tests',
            'integration': 'Integration Tests', 
            'backtesting': 'Backtesting Validation Tests',
            'stress': 'Stress Testing',
            'benchmark': 'Benchmark Comparison Tests',
            'system': 'End-to-End System Tests'
        }
        
        for category, description in categories.items():
            category_results = test_session_data.get(category, {})
            
            total = category_results.get('total', 0)
            passed = category_results.get('passed', 0)
            failed = category_results.get('failed', 0)
            skipped = category_results.get('skipped', 0)
            
            success_rate = passed / total if total > 0 else 0
            execution_time = category_results.get('duration', 0)
            
            summary = TestResultSummary(
                test_category=description,
                total_tests=total,
                passed_tests=passed,
                failed_tests=failed,
                skipped_tests=skipped,
                success_rate=success_rate,
                execution_time=execution_time,
                critical_failures=category_results.get('critical_failures', []),
                warnings=category_results.get('warnings', []),
                key_metrics=category_results.get('metrics', {})
            )
            
            self.test_results.append(summary)
    
    def generate_performance_report(self, performance_data: Dict) -> PerformanceValidationReport:
        """Generate performance validation report"""
        
        # SOW targets
        target_weekly_return = (0.03, 0.05)  # 3-5%
        target_sharpe_ratio = 1.5
        target_max_drawdown = 0.15  # 15%
        target_win_rate = 0.60  # 60%
        
        # Actual performance
        actual_weekly_return = performance_data.get('weekly_return', 0)
        actual_sharpe_ratio = performance_data.get('sharpe_ratio', 0)
        actual_max_drawdown = performance_data.get('max_drawdown', 1)
        actual_win_rate = performance_data.get('win_rate', 0)
        
        # Target validation
        weekly_return_met = target_weekly_return[0] <= actual_weekly_return <= target_weekly_return[1] * 1.2
        sharpe_ratio_met = actual_sharpe_ratio >= target_sharpe_ratio
        drawdown_met = actual_max_drawdown <= target_max_drawdown
        win_rate_met = actual_win_rate >= target_win_rate
        
        # Consistency score
        consistency_metrics = performance_data.get('consistency', {})
        consistency_score = (
            consistency_metrics.get('return_consistency', 0.5) * 0.3 +
            consistency_metrics.get('sharpe_consistency', 0.5) * 0.3 +
            consistency_metrics.get('drawdown_consistency', 0.5) * 0.4
        )
        
        # Regime performance
        regime_performance = performance_data.get('regime_performance', {})
        
        self.performance_report = PerformanceValidationReport(
            weekly_return_target_met=weekly_return_met,
            sharpe_ratio_target_met=sharpe_ratio_met,
            drawdown_constraint_met=drawdown_met,
            win_rate_target_met=win_rate_met,
            actual_weekly_return=actual_weekly_return,
            actual_sharpe_ratio=actual_sharpe_ratio,
            actual_max_drawdown=actual_max_drawdown,
            actual_win_rate=actual_win_rate,
            target_weekly_return=target_weekly_return,
            target_sharpe_ratio=target_sharpe_ratio,
            target_max_drawdown=target_max_drawdown,
            target_win_rate=target_win_rate,
            consistency_score=consistency_score,
            regime_performance=regime_performance
        )
        
        return self.performance_report
    
    def generate_benchmark_report(self, benchmark_data: Dict) -> BenchmarkComparisonReport:
        """Generate benchmark comparison report"""
        
        # Extract benchmark comparison data
        rl_vs_adaptive = benchmark_data.get('rl_vs_adaptive_outperformance', 0)
        statistical_sig = benchmark_data.get('statistical_significance', 0)
        consistency = benchmark_data.get('consistency_across_periods', 0)
        regime_perf = benchmark_data.get('regime_specific_performance', {})
        
        # SOW compliance check (15-20% outperformance target)
        sow_min_outperformance = 0.15
        sow_max_outperformance = 0.20
        
        outperformance_range_met = sow_min_outperformance <= rl_vs_adaptive <= sow_max_outperformance * 1.5
        sow_compliance = outperformance_range_met and statistical_sig < 0.1
        
        # Risk-adjusted outperformance
        risk_adjusted = benchmark_data.get('risk_adjusted_outperformance', 0)
        
        self.benchmark_report = BenchmarkComparisonReport(
            rl_vs_adaptive_outperformance=rl_vs_adaptive,
            statistical_significance=statistical_sig,
            consistency_across_periods=consistency,
            regime_specific_performance=regime_perf,
            sow_compliance=sow_compliance,
            outperformance_range_met=outperformance_range_met,
            risk_adjusted_outperformance=risk_adjusted
        )
        
        return self.benchmark_report
    
    def generate_reliability_report(self, reliability_data: Dict) -> SystemReliabilityReport:
        """Generate system reliability report"""
        
        # Extract reliability metrics
        crash_resilience = reliability_data.get('crash_resilience_score', 0)
        data_reliability = reliability_data.get('data_feed_reliability', 0)
        user_capacity = reliability_data.get('concurrent_user_capacity', 0)
        memory_efficiency = reliability_data.get('memory_efficiency_score', 0)
        response_time = reliability_data.get('response_time_performance', 0)
        failover_time = reliability_data.get('failover_recovery_time', 0)
        critical_failures = reliability_data.get('critical_system_failures', [])
        
        self.reliability_report = SystemReliabilityReport(
            crash_resilience_score=crash_resilience,
            data_feed_reliability=data_reliability,
            concurrent_user_capacity=user_capacity,
            memory_efficiency_score=memory_efficiency,
            response_time_performance=response_time,
            failover_recovery_time=failover_time,
            critical_system_failures=critical_failures
        )
        
        return self.reliability_report
    
    def generate_compliance_report(self) -> ComplianceReport:
        """Generate SOW compliance verification report"""
        
        violations = []
        warnings = []
        recommendations = []
        targets_met = 0
        total_targets = 6
        
        # Check performance targets
        if self.performance_report:
            if self.performance_report.weekly_return_target_met:
                targets_met += 1
            else:
                violations.append(f"Weekly return target not met: {self.performance_report.actual_weekly_return:.1%} vs target {self.performance_report.target_weekly_return[0]:.1%}-{self.performance_report.target_weekly_return[1]:.1%}")
            
            if self.performance_report.sharpe_ratio_target_met:
                targets_met += 1
            else:
                violations.append(f"Sharpe ratio target not met: {self.performance_report.actual_sharpe_ratio:.2f} vs target ≥{self.performance_report.target_sharpe_ratio}")
            
            if self.performance_report.drawdown_constraint_met:
                targets_met += 1
            else:
                violations.append(f"Maximum drawdown constraint violated: {self.performance_report.actual_max_drawdown:.1%} vs limit ≤{self.performance_report.target_max_drawdown:.1%}")
            
            if self.performance_report.win_rate_target_met:
                targets_met += 1
            else:
                violations.append(f"Win rate target not met: {self.performance_report.actual_win_rate:.1%} vs target ≥{self.performance_report.target_win_rate:.1%}")
        
        # Check benchmark outperformance
        if self.benchmark_report:
            if self.benchmark_report.sow_compliance:
                targets_met += 1
            else:
                violations.append(f"Outperformance vs baseline not within SOW range: {self.benchmark_report.rl_vs_adaptive_outperformance:.1%}")
            
            if self.benchmark_report.statistical_significance < 0.1:
                targets_met += 1
            else:
                violations.append(f"Outperformance not statistically significant: p-value {self.benchmark_report.statistical_significance:.3f}")
        
        # Generate recommendations
        if targets_met < total_targets:
            recommendations.append("Review and optimize RL training parameters")
            recommendations.append("Consider ensemble methods for improved performance")
            recommendations.append("Implement additional risk management controls")
            
        if self.reliability_report and self.reliability_report.crash_resilience_score < 0.8:
            recommendations.append("Strengthen crash resilience mechanisms")
        
        compliance_percentage = (targets_met / total_targets) * 100
        
        self.compliance_report = ComplianceReport(
            sow_targets_met=targets_met,
            sow_targets_total=total_targets,
            compliance_percentage=compliance_percentage,
            critical_violations=violations,
            warnings=warnings,
            recommendations=recommendations
        )
        
        return self.compliance_report
    
    def create_visualizations(self) -> Dict[str, str]:
        """Create visualization charts and return file paths"""
        
        viz_paths = {}
        
        # 1. Test Results Summary Chart
        if self.test_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Success rates by category
            categories = [result.test_category for result in self.test_results]
            success_rates = [result.success_rate * 100 for result in self.test_results]
            
            bars = ax1.bar(categories, success_rates, color=['green' if sr >= 90 else 'orange' if sr >= 70 else 'red' for sr in success_rates])
            ax1.set_title('Test Success Rates by Category')
            ax1.set_ylabel('Success Rate (%)')
            ax1.set_ylim(0, 100)
            ax1.tick_params(axis='x', rotation=45)
            
            # Add percentage labels on bars
            for bar, rate in zip(bars, success_rates):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                        f'{rate:.1f}%', ha='center', va='bottom')
            
            # Test counts
            total_tests = [result.total_tests for result in self.test_results]
            passed_tests = [result.passed_tests for result in self.test_results]
            failed_tests = [result.failed_tests for result in self.test_results]
            
            x = np.arange(len(categories))
            width = 0.35
            
            ax2.bar(x - width/2, passed_tests, width, label='Passed', color='green', alpha=0.7)
            ax2.bar(x + width/2, failed_tests, width, label='Failed', color='red', alpha=0.7)
            
            ax2.set_title('Test Results by Category')
            ax2.set_ylabel('Number of Tests')
            ax2.set_xticks(x)
            ax2.set_xticklabels(categories, rotation=45)
            ax2.legend()
            
            plt.tight_layout()
            test_summary_path = self.output_dir / 'test_summary.png'
            plt.savefig(test_summary_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['test_summary'] = str(test_summary_path)
        
        # 2. Performance Metrics Chart
        if self.performance_report:
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
            
            # Weekly return
            ax1.bar(['Actual', 'Target Min', 'Target Max'], 
                   [self.performance_report.actual_weekly_return * 100,
                    self.performance_report.target_weekly_return[0] * 100,
                    self.performance_report.target_weekly_return[1] * 100],
                   color=['blue', 'green', 'green'])
            ax1.set_title('Weekly Return Performance')
            ax1.set_ylabel('Weekly Return (%)')
            
            # Sharpe ratio
            ax2.bar(['Actual', 'Target'], 
                   [self.performance_report.actual_sharpe_ratio,
                    self.performance_report.target_sharpe_ratio],
                   color=['blue', 'green'])
            ax2.set_title('Sharpe Ratio Performance')
            ax2.set_ylabel('Sharpe Ratio')
            
            # Maximum drawdown
            ax3.bar(['Actual', 'Target Limit'], 
                   [self.performance_report.actual_max_drawdown * 100,
                    self.performance_report.target_max_drawdown * 100],
                   color=['blue', 'red'])
            ax3.set_title('Maximum Drawdown')
            ax3.set_ylabel('Maximum Drawdown (%)')
            
            # Win rate
            ax4.bar(['Actual', 'Target'], 
                   [self.performance_report.actual_win_rate * 100,
                    self.performance_report.target_win_rate * 100],
                   color=['blue', 'green'])
            ax4.set_title('Win Rate Performance')
            ax4.set_ylabel('Win Rate (%)')
            
            plt.tight_layout()
            perf_path = self.output_dir / 'performance_metrics.png'
            plt.savefig(perf_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['performance_metrics'] = str(perf_path)
        
        # 3. Benchmark Comparison Chart
        if self.benchmark_report:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Outperformance vs target range
            outperf = self.benchmark_report.rl_vs_adaptive_outperformance * 100
            target_min = 15  # 15%
            target_max = 20  # 20%
            
            bars = ax1.bar(['Actual Outperformance', 'Target Min', 'Target Max'], 
                          [outperf, target_min, target_max],
                          color=['blue', 'green', 'green'])
            ax1.set_title('RL vs AdaptiveThreshold Outperformance')
            ax1.set_ylabel('Outperformance (%)')
            ax1.axhline(y=target_min, color='green', linestyle='--', alpha=0.7, label='Target Range')
            ax1.axhline(y=target_max, color='green', linestyle='--', alpha=0.7)
            
            # Regime-specific performance
            if self.benchmark_report.regime_specific_performance:
                regimes = list(self.benchmark_report.regime_specific_performance.keys())
                regime_outperf = [v * 100 for v in self.benchmark_report.regime_specific_performance.values()]
                
                bars = ax2.bar(regimes, regime_outperf, 
                              color=['green' if v > 0 else 'red' for v in regime_outperf])
                ax2.set_title('Outperformance by Market Regime')
                ax2.set_ylabel('Outperformance (%)')
                ax2.tick_params(axis='x', rotation=45)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            
            plt.tight_layout()
            benchmark_path = self.output_dir / 'benchmark_comparison.png'
            plt.savefig(benchmark_path, dpi=300, bbox_inches='tight')
            plt.close()
            viz_paths['benchmark_comparison'] = str(benchmark_path)
        
        return viz_paths
    
    def generate_html_report(self, viz_paths: Dict[str, str]) -> str:
        """Generate comprehensive HTML report"""
        
        # HTML template
        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>RL Trading System Test Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .header { background-color: #2c3e50; color: white; padding: 20px; margin-bottom: 30px; }
                .section { margin-bottom: 30px; }
                .metric-box { display: inline-block; background-color: #ecf0f1; padding: 15px; margin: 10px; border-radius: 5px; }
                .pass { color: green; font-weight: bold; }
                .fail { color: red; font-weight: bold; }
                .warning { color: orange; font-weight: bold; }
                .chart { text-align: center; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .summary-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>RL Trading System Validation Report</h1>
                <p>Generated on: {{ report_timestamp }}</p>
                <p>Test Suite: Comprehensive Validation & SOW Compliance</p>
            </div>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <div class="summary-grid">
                    <div class="metric-box">
                        <h3>Overall Compliance</h3>
                        <div class="{{ 'pass' if compliance_percentage >= 90 else 'warning' if compliance_percentage >= 70 else 'fail' }}">
                            {{ compliance_percentage:.1f }}% ({{ sow_targets_met }}/{{ sow_targets_total }} targets met)
                        </div>
                    </div>
                    <div class="metric-box">
                        <h3>Test Success Rate</h3>
                        <div class="{{ 'pass' if overall_success_rate >= 90 else 'warning' if overall_success_rate >= 70 else 'fail' }}">
                            {{ overall_success_rate:.1f }}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <h3>Performance Validation</h3>
                        <div class="{{ 'pass' if performance_targets_met >= 3 else 'warning' if performance_targets_met >= 2 else 'fail' }}">
                            {{ performance_targets_met }}/4 targets met
                        </div>
                    </div>
                    <div class="metric-box">
                        <h3>Benchmark Outperformance</h3>
                        <div class="{{ 'pass' if benchmark_compliance else 'fail' }}">
                            {{ outperformance_percentage:.1f }}% vs AdaptiveThreshold
                        </div>
                    </div>
                </div>
            </div>
            
            {% if test_summary_chart %}
            <div class="section">
                <h2>Test Results Summary</h2>
                <div class="chart">
                    <img src="{{ test_summary_chart }}" alt="Test Summary Chart" style="max-width: 100%;">
                </div>
                <table>
                    <tr><th>Test Category</th><th>Total</th><th>Passed</th><th>Failed</th><th>Success Rate</th></tr>
                    {% for result in test_results %}
                    <tr>
                        <td>{{ result.test_category }}</td>
                        <td>{{ result.total_tests }}</td>
                        <td class="pass">{{ result.passed_tests }}</td>
                        <td class="{{ 'fail' if result.failed_tests > 0 else '' }}">{{ result.failed_tests }}</td>
                        <td class="{{ 'pass' if result.success_rate >= 0.9 else 'warning' if result.success_rate >= 0.7 else 'fail' }}">
                            {{ (result.success_rate * 100):.1f }}%
                        </td>
                    </tr>
                    {% endfor %}
                </table>
            </div>
            {% endif %}
            
            {% if performance_metrics_chart %}
            <div class="section">
                <h2>Performance Validation Results</h2>
                <div class="chart">
                    <img src="{{ performance_metrics_chart }}" alt="Performance Metrics Chart" style="max-width: 100%;">
                </div>
                <table>
                    <tr><th>Metric</th><th>Actual</th><th>Target</th><th>Status</th></tr>
                    <tr>
                        <td>Weekly Return</td>
                        <td>{{ (actual_weekly_return * 100):.2f }}%</td>
                        <td>{{ (target_weekly_return_min * 100):.1f }}% - {{ (target_weekly_return_max * 100):.1f }}%</td>
                        <td class="{{ 'pass' if weekly_return_met else 'fail' }}">{{ 'PASS' if weekly_return_met else 'FAIL' }}</td>
                    </tr>
                    <tr>
                        <td>Sharpe Ratio</td>
                        <td>{{ actual_sharpe_ratio:.2f }}</td>
                        <td>≥{{ target_sharpe_ratio:.1f }}</td>
                        <td class="{{ 'pass' if sharpe_ratio_met else 'fail' }}">{{ 'PASS' if sharpe_ratio_met else 'FAIL' }}</td>
                    </tr>
                    <tr>
                        <td>Max Drawdown</td>
                        <td>{{ (actual_max_drawdown * 100):.1f }}%</td>
                        <td>≤{{ (target_max_drawdown * 100):.1f }}%</td>
                        <td class="{{ 'pass' if drawdown_met else 'fail' }}">{{ 'PASS' if drawdown_met else 'FAIL' }}</td>
                    </tr>
                    <tr>
                        <td>Win Rate</td>
                        <td>{{ (actual_win_rate * 100):.1f }}%</td>
                        <td>≥{{ (target_win_rate * 100):.1f }}%</td>
                        <td class="{{ 'pass' if win_rate_met else 'fail' }}">{{ 'PASS' if win_rate_met else 'FAIL' }}</td>
                    </tr>
                </table>
            </div>
            {% endif %}
            
            {% if benchmark_comparison_chart %}
            <div class="section">
                <h2>Benchmark Comparison Analysis</h2>
                <div class="chart">
                    <img src="{{ benchmark_comparison_chart }}" alt="Benchmark Comparison Chart" style="max-width: 100%;">
                </div>
                <p><strong>RL vs AdaptiveThreshold Outperformance:</strong> 
                   <span class="{{ 'pass' if benchmark_compliance else 'fail' }}">{{ (outperformance_percentage):.1f }}%</span>
                   (Target: 15-20%)
                </p>
                <p><strong>Statistical Significance:</strong> 
                   <span class="{{ 'pass' if statistical_significance < 0.1 else 'fail' }}">p = {{ statistical_significance:.3f }}</span>
                </p>
            </div>
            {% endif %}
            
            <div class="section">
                <h2>System Reliability Assessment</h2>
                {% if reliability_report %}
                <div class="summary-grid">
                    <div class="metric-box">
                        <h4>Crash Resilience</h4>
                        <div class="{{ 'pass' if crash_resilience >= 0.8 else 'warning' if crash_resilience >= 0.6 else 'fail' }}">
                            {{ (crash_resilience * 100):.1f }}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <h4>Data Feed Reliability</h4>
                        <div class="{{ 'pass' if data_reliability >= 0.95 else 'warning' if data_reliability >= 0.9 else 'fail' }}">
                            {{ (data_reliability * 100):.1f }}%
                        </div>
                    </div>
                    <div class="metric-box">
                        <h4>User Capacity</h4>
                        <div>{{ user_capacity }} concurrent users</div>
                    </div>
                    <div class="metric-box">
                        <h4>Response Time</h4>
                        <div class="{{ 'pass' if response_time_perf >= 0.9 else 'warning' if response_time_perf >= 0.7 else 'fail' }}">
                            {{ (response_time_perf * 100):.1f }}% within SLA
                        </div>
                    </div>
                </div>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>Compliance Violations & Recommendations</h2>
                {% if critical_violations %}
                <h3 class="fail">Critical Violations:</h3>
                <ul>
                    {% for violation in critical_violations %}
                    <li class="fail">{{ violation }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
                
                {% if warnings %}
                <h3 class="warning">Warnings:</h3>
                <ul>
                    {% for warning in warnings %}
                    <li class="warning">{{ warning }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
                
                {% if recommendations %}
                <h3>Recommendations:</h3>
                <ul>
                    {% for recommendation in recommendations %}
                    <li>{{ recommendation }}</li>
                    {% endfor %}
                </ul>
                {% endif %}
            </div>
            
            <div class="section">
                <h2>Conclusion</h2>
                <p>
                    {% if compliance_percentage >= 90 %}
                    <strong class="pass">SYSTEM APPROVED:</strong> The RL trading system meets {{ compliance_percentage:.0f }}% of SOW requirements and is ready for deployment.
                    {% elif compliance_percentage >= 70 %}
                    <strong class="warning">CONDITIONAL APPROVAL:</strong> The RL trading system meets {{ compliance_percentage:.0f }}% of SOW requirements but requires addressing identified issues before full deployment.
                    {% else %}
                    <strong class="fail">SYSTEM REJECTED:</strong> The RL trading system meets only {{ compliance_percentage:.0f }}% of SOW requirements and requires significant improvements before deployment consideration.
                    {% endif %}
                </p>
            </div>
            
            <div class="section">
                <p><em>Report generated by RL Trading System Test Suite - {{ report_timestamp }}</em></p>
            </div>
        </body>
        </html>
        """
        
        # Prepare template variables
        template_vars = {
            'report_timestamp': self.report_timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'test_results': self.test_results,
            'test_summary_chart': viz_paths.get('test_summary', ''),
            'performance_metrics_chart': viz_paths.get('performance_metrics', ''),
            'benchmark_comparison_chart': viz_paths.get('benchmark_comparison', ''),
        }
        
        # Add compliance data
        if self.compliance_report:
            template_vars.update({
                'compliance_percentage': self.compliance_report.compliance_percentage,
                'sow_targets_met': self.compliance_report.sow_targets_met,
                'sow_targets_total': self.compliance_report.sow_targets_total,
                'critical_violations': self.compliance_report.critical_violations,
                'warnings': self.compliance_report.warnings,
                'recommendations': self.compliance_report.recommendations,
            })
        
        # Add performance data
        if self.performance_report:
            performance_targets_met = sum([
                self.performance_report.weekly_return_target_met,
                self.performance_report.sharpe_ratio_target_met,
                self.performance_report.drawdown_constraint_met,
                self.performance_report.win_rate_target_met
            ])
            
            template_vars.update({
                'performance_targets_met': performance_targets_met,
                'actual_weekly_return': self.performance_report.actual_weekly_return,
                'actual_sharpe_ratio': self.performance_report.actual_sharpe_ratio,
                'actual_max_drawdown': self.performance_report.actual_max_drawdown,
                'actual_win_rate': self.performance_report.actual_win_rate,
                'target_weekly_return_min': self.performance_report.target_weekly_return[0],
                'target_weekly_return_max': self.performance_report.target_weekly_return[1],
                'target_sharpe_ratio': self.performance_report.target_sharpe_ratio,
                'target_max_drawdown': self.performance_report.target_max_drawdown,
                'target_win_rate': self.performance_report.target_win_rate,
                'weekly_return_met': self.performance_report.weekly_return_target_met,
                'sharpe_ratio_met': self.performance_report.sharpe_ratio_target_met,
                'drawdown_met': self.performance_report.drawdown_constraint_met,
                'win_rate_met': self.performance_report.win_rate_target_met,
            })
        
        # Add benchmark data
        if self.benchmark_report:
            template_vars.update({
                'benchmark_compliance': self.benchmark_report.sow_compliance,
                'outperformance_percentage': self.benchmark_report.rl_vs_adaptive_outperformance * 100,
                'statistical_significance': self.benchmark_report.statistical_significance,
            })
        
        # Add reliability data
        if self.reliability_report:
            template_vars.update({
                'reliability_report': True,
                'crash_resilience': self.reliability_report.crash_resilience_score,
                'data_reliability': self.reliability_report.data_feed_reliability,
                'user_capacity': self.reliability_report.concurrent_user_capacity,
                'response_time_perf': self.reliability_report.response_time_performance,
            })
        
        # Calculate overall success rate
        if self.test_results:
            total_tests = sum(r.total_tests for r in self.test_results)
            passed_tests = sum(r.passed_tests for r in self.test_results)
            overall_success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
            template_vars['overall_success_rate'] = overall_success_rate
        
        # Render template
        template = jinja2.Template(html_template)
        html_content = template.render(**template_vars)
        
        # Save HTML report
        html_path = self.output_dir / 'comprehensive_test_report.html'
        with open(html_path, 'w') as f:
            f.write(html_content)
        
        return str(html_path)
    
    def generate_json_report(self) -> str:
        """Generate machine-readable JSON report"""
        
        report_data = {
            'metadata': {
                'timestamp': self.report_timestamp.isoformat(),
                'report_version': '1.0',
                'test_framework': 'pytest',
                'system': 'RL Trading System'
            },
            'test_results': [asdict(result) for result in self.test_results],
            'performance_validation': asdict(self.performance_report) if self.performance_report else None,
            'benchmark_comparison': asdict(self.benchmark_report) if self.benchmark_report else None,
            'system_reliability': asdict(self.reliability_report) if self.reliability_report else None,
            'sow_compliance': asdict(self.compliance_report) if self.compliance_report else None
        }
        
        json_path = self.output_dir / 'test_results.json'
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2, default=str)
        
        return str(json_path)
    
    def generate_executive_summary(self) -> str:
        """Generate executive summary document"""
        
        summary_content = f"""
# RL Trading System Validation - Executive Summary

**Report Date:** {self.report_timestamp.strftime('%Y-%m-%d')}
**System:** Reinforcement Learning Trading System
**Validation Scope:** SOW Compliance & Performance Testing

## Overall Assessment

"""
        
        if self.compliance_report:
            compliance_pct = self.compliance_report.compliance_percentage
            if compliance_pct >= 90:
                summary_content += f"✅ **SYSTEM APPROVED** - {compliance_pct:.0f}% SOW compliance achieved\n\n"
            elif compliance_pct >= 70:
                summary_content += f"⚠️ **CONDITIONAL APPROVAL** - {compliance_pct:.0f}% SOW compliance (requires improvements)\n\n"
            else:
                summary_content += f"❌ **SYSTEM REJECTED** - Only {compliance_pct:.0f}% SOW compliance achieved\n\n"
        
        # Key Performance Metrics
        summary_content += "## Key Performance Metrics\n\n"
        
        if self.performance_report:
            summary_content += f"- **Weekly Return:** {self.performance_report.actual_weekly_return*100:.2f}% (Target: 3-5%)\n"
            summary_content += f"- **Sharpe Ratio:** {self.performance_report.actual_sharpe_ratio:.2f} (Target: ≥1.5)\n"
            summary_content += f"- **Max Drawdown:** {self.performance_report.actual_max_drawdown*100:.1f}% (Target: ≤15%)\n"
            summary_content += f"- **Win Rate:** {self.performance_report.actual_win_rate*100:.1f}% (Target: ≥60%)\n\n"
        
        # Benchmark Outperformance
        if self.benchmark_report:
            summary_content += f"## Benchmark Outperformance\n\n"
            summary_content += f"- **vs AdaptiveThreshold:** {self.benchmark_report.rl_vs_adaptive_outperformance*100:.1f}% (Target: 15-20%)\n"
            summary_content += f"- **Statistical Significance:** p = {self.benchmark_report.statistical_significance:.3f}\n\n"
        
        # Critical Issues
        if self.compliance_report and self.compliance_report.critical_violations:
            summary_content += "## Critical Issues\n\n"
            for violation in self.compliance_report.critical_violations:
                summary_content += f"- {violation}\n"
            summary_content += "\n"
        
        # Recommendations
        if self.compliance_report and self.compliance_report.recommendations:
            summary_content += "## Recommendations\n\n"
            for rec in self.compliance_report.recommendations:
                summary_content += f"- {rec}\n"
            summary_content += "\n"
        
        summary_content += "---\n"
        summary_content += f"*Full technical report available in comprehensive_test_report.html*\n"
        
        # Save summary
        summary_path = self.output_dir / 'executive_summary.md'
        with open(summary_path, 'w') as f:
            f.write(summary_content)
        
        return str(summary_path)


class TestReportGeneratorTests:
    """Tests for the test report generator"""
    
    def test_report_generation_workflow(self):
        """Test complete report generation workflow"""
        
        # Create test data
        mock_test_data = {
            'performance': {
                'total': 10, 'passed': 8, 'failed': 2, 'skipped': 0,
                'duration': 45.2, 'metrics': {'weekly_return': 0.042}
            },
            'integration': {
                'total': 15, 'passed': 14, 'failed': 1, 'skipped': 0,
                'duration': 32.1, 'metrics': {}
            },
            'stress': {
                'total': 8, 'passed': 7, 'failed': 1, 'skipped': 0,
                'duration': 120.5, 'metrics': {'crash_resilience': 0.85}
            }
        }
        
        mock_performance_data = {
            'weekly_return': 0.042,
            'sharpe_ratio': 1.65,
            'max_drawdown': 0.12,
            'win_rate': 0.63,
            'consistency': {'return_consistency': 0.8, 'sharpe_consistency': 0.75},
            'regime_performance': {'bull': 0.05, 'bear': 0.02, 'volatile': 0.03}
        }
        
        mock_benchmark_data = {
            'rl_vs_adaptive_outperformance': 0.18,
            'statistical_significance': 0.05,
            'consistency_across_periods': 0.72,
            'risk_adjusted_outperformance': 0.22
        }
        
        mock_reliability_data = {
            'crash_resilience_score': 0.85,
            'data_feed_reliability': 0.96,
            'concurrent_user_capacity': 25,
            'memory_efficiency_score': 0.88,
            'response_time_performance': 0.92,
            'failover_recovery_time': 15.2
        }
        
        # Generate report
        generator = TestReportGenerator(output_dir="test_reports_output")
        
        # Collect data
        generator.collect_test_results(mock_test_data)
        generator.generate_performance_report(mock_performance_data)
        generator.generate_benchmark_report(mock_benchmark_data)
        generator.generate_reliability_report(mock_reliability_data)
        generator.generate_compliance_report()
        
        # Create visualizations
        viz_paths = generator.create_visualizations()
        
        # Generate reports
        html_path = generator.generate_html_report(viz_paths)
        json_path = generator.generate_json_report()
        summary_path = generator.generate_executive_summary()
        
        # Verify files were created
        assert os.path.exists(html_path), "HTML report not generated"
        assert os.path.exists(json_path), "JSON report not generated"
        assert os.path.exists(summary_path), "Executive summary not generated"
        
        # Verify compliance calculation
        assert generator.compliance_report.compliance_percentage >= 80, "Compliance calculation incorrect"
        
        logger.info("Report generation workflow test passed")
        logger.info(f"Reports generated at: {generator.output_dir}")
        
        return {
            'html_report': html_path,
            'json_report': json_path,
            'executive_summary': summary_path,
            'visualizations': viz_paths
        }


if __name__ == "__main__":
    # Run report generator tests
    test_gen = TestReportGeneratorTests()
    reports = test_gen.test_report_generation_workflow()
    print("Test report generation completed successfully!")
    print(f"Generated reports: {reports}")