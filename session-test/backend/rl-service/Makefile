# Makefile for RL Trading System Testing

.PHONY: help install test test-unit test-integration test-performance test-stress test-benchmark test-all
.PHONY: coverage report clean lint format security check-all
.PHONY: smoke-test regression-test sow-compliance
.DEFAULT_GOAL := help

# Configuration
PYTHON := python3
PIP := pip3
PYTEST := pytest
BLACK := black
ISORT := isort
FLAKE8 := flake8
MYPY := mypy
BANDIT := bandit
SAFETY := safety

# Directories
TEST_DIR := tests
REPORTS_DIR := test_reports
COVERAGE_DIR := coverage_reports

# Test categories
UNIT_TESTS := tests/test_environment.py tests/test_rl_system.py
INTEGRATION_TESTS := tests/test_integration.py
PERFORMANCE_TESTS := tests/test_performance.py
STRESS_TESTS := tests/stress_testing.py
BENCHMARK_TESTS := tests/benchmark_comparison.py
BACKTEST_TESTS := tests/test_backtesting.py

help: ## Show this help message
	@echo "RL Trading System Test Suite"
	@echo "============================"
	@echo ""
	@echo "Available commands:"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)

##@ Setup and Installation

install: ## Install all dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt

install-dev: ## Install development dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -r requirements-test.txt
	$(PIP) install pre-commit
	pre-commit install

setup-dirs: ## Create necessary directories
	mkdir -p $(REPORTS_DIR)
	mkdir -p $(COVERAGE_DIR)
	mkdir -p $(COVERAGE_DIR)/html

##@ Code Quality

lint: ## Run linting checks
	$(FLAKE8) . --count --select=E9,F63,F7,F82 --show-source --statistics
	$(FLAKE8) . --count --exit-zero --max-complexity=10 --max-line-length=127 --statistics

format: ## Format code with black and isort
	$(BLACK) .
	$(ISORT) .

format-check: ## Check code formatting
	$(BLACK) --check --diff .
	$(ISORT) --check-only --diff .

type-check: ## Run type checking
	$(MYPY) . --ignore-missing-imports --no-strict-optional

security: ## Run security checks
	$(BANDIT) -r . -f json -o $(REPORTS_DIR)/security-report.json
	$(SAFETY) check --json --output $(REPORTS_DIR)/safety-report.json

check-all: lint format-check type-check security ## Run all code quality checks

##@ Testing

smoke-test: setup-dirs ## Run quick smoke tests
	$(PYTEST) -m smoke --tb=short --maxfail=5 -v \
		--html=$(REPORTS_DIR)/smoke-report.html --self-contained-html

test-unit: setup-dirs ## Run unit tests
	$(PYTEST) $(UNIT_TESTS) -m unit --cov --cov-report=html:$(COVERAGE_DIR)/unit \
		--cov-report=xml:$(COVERAGE_DIR)/unit-coverage.xml \
		--html=$(REPORTS_DIR)/unit-report.html --self-contained-html -v

test-integration: setup-dirs ## Run integration tests
	$(PYTEST) $(INTEGRATION_TESTS) -m integration --cov --cov-report=html:$(COVERAGE_DIR)/integration \
		--cov-report=xml:$(COVERAGE_DIR)/integration-coverage.xml \
		--html=$(REPORTS_DIR)/integration-report.html --self-contained-html -v --timeout=300

test-performance: setup-dirs ## Run performance validation tests
	$(PYTEST) $(PERFORMANCE_TESTS) -m performance --tb=short -v --timeout=3600 \
		--html=$(REPORTS_DIR)/performance-report.html --self-contained-html \
		--benchmark-only --benchmark-json=$(REPORTS_DIR)/benchmark.json

test-stress: setup-dirs ## Run stress tests
	$(PYTEST) $(STRESS_TESTS) -m stress --tb=short -v --timeout=2700 \
		--html=$(REPORTS_DIR)/stress-report.html --self-contained-html

test-benchmark: setup-dirs ## Run benchmark comparison tests
	$(PYTEST) $(BENCHMARK_TESTS) -m benchmark --tb=short -v --timeout=5400 \
		--html=$(REPORTS_DIR)/benchmark-report.html --self-contained-html

test-backtest: setup-dirs ## Run backtesting validation tests
	$(PYTEST) $(BACKTEST_TESTS) --tb=short -v --timeout=2400 \
		--html=$(REPORTS_DIR)/backtest-report.html --self-contained-html

test: test-unit test-integration ## Run basic test suite (unit + integration)

test-all: setup-dirs ## Run complete test suite
	$(PYTEST) $(TEST_DIR) --cov --cov-report=html:$(COVERAGE_DIR)/html \
		--cov-report=xml:$(COVERAGE_DIR)/coverage.xml --cov-report=term-missing \
		--html=$(REPORTS_DIR)/complete-report.html --self-contained-html \
		--junitxml=$(REPORTS_DIR)/junit.xml -v --tb=short

test-parallel: setup-dirs ## Run tests in parallel
	$(PYTEST) $(TEST_DIR) -n auto --cov --cov-report=html:$(COVERAGE_DIR)/html \
		--cov-report=xml:$(COVERAGE_DIR)/coverage.xml \
		--html=$(REPORTS_DIR)/parallel-report.html --self-contained-html -v

##@ Specialized Testing

regression-test: setup-dirs ## Run regression tests
	$(PYTEST) -m regression --tb=short -v \
		--html=$(REPORTS_DIR)/regression-report.html --self-contained-html

critical-test: setup-dirs ## Run critical path tests only
	$(PYTEST) -m critical --tb=short --maxfail=1 -v \
		--html=$(REPORTS_DIR)/critical-report.html --self-contained-html

sow-compliance: test-performance test-benchmark ## Validate SOW compliance
	@echo "=========================================="
	@echo "SOW COMPLIANCE VALIDATION"
	@echo "=========================================="
	@echo "Checking performance against SOW targets:"
	@echo "- Weekly returns: 3-5%"
	@echo "- Sharpe ratio: >1.5" 
	@echo "- Maximum drawdown: <15%"
	@echo "- Win rate: >60%"
	@echo "- Outperformance vs AdaptiveThreshold: 15-20%"
	@echo ""
	@echo "Review test reports for detailed compliance:"
	@echo "- Performance: $(REPORTS_DIR)/performance-report.html"
	@echo "- Benchmark: $(REPORTS_DIR)/benchmark-report.html"
	@echo "=========================================="

##@ Coverage and Reporting

coverage: ## Generate coverage report
	$(PYTEST) $(TEST_DIR) --cov --cov-report=html:$(COVERAGE_DIR)/html \
		--cov-report=xml:$(COVERAGE_DIR)/coverage.xml --cov-report=term-missing

coverage-open: coverage ## Generate and open coverage report
	@if command -v open >/dev/null 2>&1; then \
		open $(COVERAGE_DIR)/html/index.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(COVERAGE_DIR)/html/index.html; \
	else \
		echo "Coverage report generated at: $(COVERAGE_DIR)/html/index.html"; \
	fi

report: setup-dirs ## Generate comprehensive test report
	$(PYTHON) $(TEST_DIR)/test_report_generator.py
	@echo "Comprehensive test report generated in $(REPORTS_DIR)/"

report-open: report ## Generate and open comprehensive test report
	@if command -v open >/dev/null 2>&1; then \
		open $(REPORTS_DIR)/comprehensive_test_report.html; \
	elif command -v xdg-open >/dev/null 2>&1; then \
		xdg-open $(REPORTS_DIR)/comprehensive_test_report.html; \
	else \
		echo "Test report generated at: $(REPORTS_DIR)/comprehensive_test_report.html"; \
	fi

##@ CI/CD Support

ci-test: check-all test-all ## Run full CI test pipeline
	@echo "CI pipeline completed successfully!"

ci-performance: test-performance sow-compliance ## Run CI performance validation
	@echo "Performance validation completed!"

ci-benchmark: test-benchmark ## Run CI benchmark comparison
	@echo "Benchmark comparison completed!"

##@ Cleanup

clean: ## Clean up generated files
	rm -rf $(REPORTS_DIR)/*
	rm -rf $(COVERAGE_DIR)/*
	rm -rf .pytest_cache
	rm -rf __pycache__
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete

clean-all: clean ## Clean up all generated files and caches
	rm -rf .mypy_cache
	rm -rf .coverage.*
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/

##@ Development Workflows

dev-test: format lint test ## Development workflow: format, lint, test
	@echo "Development test cycle completed!"

pre-commit: check-all test-unit test-integration ## Pre-commit validation
	@echo "Pre-commit checks passed!"

pre-deploy: test-all sow-compliance report ## Pre-deployment validation
	@echo "Pre-deployment validation completed!"
	@echo "Review reports before deployment:"
	@echo "- Comprehensive: $(REPORTS_DIR)/comprehensive_test_report.html"
	@echo "- Coverage: $(COVERAGE_DIR)/html/index.html"

##@ Quick Commands

quick: smoke-test ## Quick validation (smoke tests only)

full: test-all report ## Full test suite with reporting

validate: sow-compliance ## Validate SOW compliance

##@ Information

list-tests: ## List all available tests
	$(PYTEST) --collect-only -q

test-markers: ## Show all test markers
	$(PYTEST) --markers

test-config: ## Show pytest configuration
	$(PYTEST) --help

##@ Docker Support (if applicable)

docker-test: ## Run tests in Docker container
	docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit

docker-clean: ## Clean up Docker test resources
	docker-compose -f docker-compose.test.yml down --volumes --remove-orphans