"""
Trading Bot Final Testing and Validation Suite

This package contains comprehensive tests to validate that the trading bot
meets all SOW requirements and is ready for production deployment.

Test Categories:
- System Integration Tests
- SOW Compliance Validation
- Load and Performance Testing
- Security Assessment
- User Acceptance Testing

Usage:
    python run_final_tests.py

For detailed documentation, see README.md
"""

__version__ = "1.0.0"
__author__ = "Trading Bot Development Team"

# Test suite configuration
TEST_SUITE_CONFIG = {
    "version": __version__,
    "sow_requirements": {
        "weekly_return_target": (0.03, 0.05),  # 3-5%
        "min_sharpe_ratio": 1.5,
        "max_drawdown": 0.15,  # 15%
        "min_win_rate": 0.60,  # 60%
        "max_var_95": 0.02,  # 2% daily VaR
    },
    "performance_benchmarks": {
        "max_response_time_ms": 500,
        "min_throughput_rps": 1000,
        "max_error_rate": 0.01,  # 1%
        "max_memory_growth_mb": 100,
    },
    "security_requirements": {
        "min_security_score": 0.85,
        "max_critical_vulnerabilities": 0,
        "required_encryption": "AES-256",
        "min_password_strength": 8,
    }
}

# Export main components
from .generate_final_report import FinalReportGenerator

__all__ = [
    "FinalReportGenerator",
    "TEST_SUITE_CONFIG"
]