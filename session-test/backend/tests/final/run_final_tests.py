"""
Final Test Suite Runner

Orchestrates the execution of all final validation tests and generates comprehensive reports.
This script should be run as the final validation step before production deployment.
"""

import os
import sys
import argparse
import logging
import time
from datetime import datetime
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from generate_final_report import FinalReportGenerator


def setup_logging(log_level='INFO', log_file=None):
    """Setup logging configuration"""
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format=log_format,
        handlers=handlers
    )


def validate_environment():
    """Validate that the testing environment is properly set up"""
    logger = logging.getLogger(__name__)
    
    logger.info("Validating testing environment...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        logger.error("Python 3.8 or higher is required")
        return False
    
    # Check required packages
    required_packages = [
        'pytest', 'numpy', 'pandas', 'matplotlib', 'seaborn',
        'aiohttp', 'psutil', 'jwt', 'bcrypt', 'cryptography'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        logger.error(f"Missing required packages: {', '.join(missing_packages)}")
        logger.error("Install with: pip install -r requirements-test.txt")
        return False
    
    # Check test files exist
    test_files = [
        'test_complete_system.py',
        'test_sow_compliance.py',
        'test_load_performance.py',
        'test_security.py',
        'test_acceptance.py'
    ]
    
    current_dir = Path(__file__).parent
    missing_files = []
    for test_file in test_files:
        if not (current_dir / test_file).exists():
            missing_files.append(test_file)
    
    if missing_files:
        logger.error(f"Missing test files: {', '.join(missing_files)}")
        return False
    
    logger.info("Environment validation completed successfully")
    return True


def pre_test_setup():
    """Perform pre-test setup tasks"""
    logger = logging.getLogger(__name__)
    
    logger.info("Performing pre-test setup...")
    
    # Create necessary directories
    directories = [
        'reports',
        'reports/charts',
        'reports/data',
        'reports/exports',
        'logs'
    ]
    
    base_dir = Path(__file__).parent
    for directory in directories:
        (base_dir / directory).mkdir(parents=True, exist_ok=True)
    
    # Initialize test environment variables
    os.environ['TESTING_MODE'] = 'true'
    os.environ['TEST_DATABASE_URL'] = 'sqlite:///test_trading_bot.db'
    
    # Clear any existing test data
    test_db_path = base_dir / 'test_trading_bot.db'
    if test_db_path.exists():
        test_db_path.unlink()
    
    logger.info("Pre-test setup completed")


def run_individual_test_category(category_name, test_file, output_dir):
    """Run an individual test category"""
    logger = logging.getLogger(__name__)
    
    logger.info(f"Starting {category_name} tests...")
    
    start_time = time.time()
    
    try:
        # Import and run the specific test module
        test_module_path = Path(__file__).parent / test_file
        
        if not test_module_path.exists():
            logger.error(f"Test file not found: {test_file}")
            return False
        
        # Run pytest on the specific file
        import subprocess
        
        cmd = [
            sys.executable, '-m', 'pytest',
            str(test_module_path),
            '-v',
            '--tb=short',
            '--maxfail=5',
            f'--junitxml={output_dir}/junit_{category_name.lower().replace(" ", "_")}.xml'
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            logger.info(f"{category_name} tests completed successfully in {execution_time:.2f}s")
            return True
        else:
            logger.error(f"{category_name} tests failed after {execution_time:.2f}s")
            logger.error(f"Error output: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error(f"{category_name} tests timed out after 10 minutes")
        return False
    except Exception as e:
        logger.error(f"Error running {category_name} tests: {e}")
        return False


def post_test_cleanup():
    """Perform post-test cleanup tasks"""
    logger = logging.getLogger(__name__)
    
    logger.info("Performing post-test cleanup...")
    
    # Clean up test database
    test_db_path = Path(__file__).parent / 'test_trading_bot.db'
    if test_db_path.exists():
        test_db_path.unlink()
    
    # Clear test environment variables
    os.environ.pop('TESTING_MODE', None)
    os.environ.pop('TEST_DATABASE_URL', None)
    
    logger.info("Post-test cleanup completed")


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='Trading Bot Final Test Suite Runner')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Set the logging level')
    parser.add_argument('--output-dir', default=None,
                       help='Output directory for reports (default: ./reports)')
    parser.add_argument('--skip-categories', nargs='*', default=[],
                       help='Test categories to skip')
    parser.add_argument('--only-categories', nargs='*', default=[],
                       help='Only run these test categories')
    parser.add_argument('--generate-report-only', action='store_true',
                       help='Only generate the final report (skip test execution)')
    parser.add_argument('--continue-on-failure', action='store_true',
                       help='Continue running tests even if some categories fail')
    
    args = parser.parse_args()
    
    # Setup logging
    log_file = Path(__file__).parent / 'logs' / f"test_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(args.log_level, str(log_file))
    
    logger = logging.getLogger(__name__)
    
    # Print banner
    print("\n" + "="*80)
    print("TRADING BOT FINAL VALIDATION TEST SUITE")
    print("="*80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log File: {log_file}")
    print("="*80 + "\n")
    
    # Validate environment
    if not validate_environment():
        logger.error("Environment validation failed. Exiting.")
        return 1
    
    # Set output directory
    output_dir = args.output_dir or str(Path(__file__).parent / 'reports')
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Define test categories
    test_categories = [
        ('System Integration', 'test_complete_system.py'),
        ('SOW Compliance', 'test_sow_compliance.py'),
        ('Load Performance', 'test_load_performance.py'),
        ('Security', 'test_security.py'),
        ('Acceptance', 'test_acceptance.py')
    ]
    
    # Filter test categories based on arguments
    if args.only_categories:
        test_categories = [(name, file) for name, file in test_categories 
                          if name in args.only_categories]
    
    if args.skip_categories:
        test_categories = [(name, file) for name, file in test_categories 
                          if name not in args.skip_categories]
    
    # Run tests or generate report only
    if not args.generate_report_only:
        # Perform pre-test setup
        pre_test_setup()
        
        # Track test execution results
        test_results = {}
        overall_success = True
        
        # Run each test category
        for category_name, test_file in test_categories:
            logger.info(f"\n{'='*20} {category_name} {'='*20}")
            
            success = run_individual_test_category(category_name, test_file, output_dir)
            test_results[category_name] = success
            
            if not success:
                overall_success = False
                if not args.continue_on_failure:
                    logger.error(f"Test category '{category_name}' failed. Stopping execution.")
                    break
        
        # Perform post-test cleanup
        post_test_cleanup()
        
        # Log test execution summary
        logger.info(f"\n{'='*20} TEST EXECUTION SUMMARY {'='*20}")
        for category, success in test_results.items():
            status = "PASSED" if success else "FAILED"
            logger.info(f"{category}: {status}")
        
        if not overall_success and not args.continue_on_failure:
            logger.error("Test execution failed. Skipping report generation.")
            return 1
    
    # Generate comprehensive final report
    logger.info(f"\n{'='*20} GENERATING FINAL REPORT {'='*20}")
    
    try:
        generator = FinalReportGenerator(output_dir)
        final_report = generator.run_complete_test_suite()
        
        # Print final summary
        print("\n" + "="*80)
        print("FINAL VALIDATION RESULTS")
        print("="*80)
        
        exec_summary = final_report['executive_summary']
        overall_metrics = final_report['overall_metrics']
        
        print(f"Overall Status: {exec_summary['overall_status']}")
        print(f"Production Ready: {'YES' if exec_summary['production_ready'] else 'NO'}")
        print(f"SOW Compliant: {'YES' if exec_summary['sow_compliant'] else 'NO'}")
        print(f"Overall Score: {overall_metrics['overall_weighted_score']:.1%}")
        print(f"Total Tests: {overall_metrics['total_tests_executed']}")
        print(f"Tests Passed: {overall_metrics['total_tests_passed']}")
        
        # Print critical issues if any
        if final_report['production_readiness']['critical_issues']:
            print("\nCRITICAL ISSUES:")
            for issue in final_report['production_readiness']['critical_issues']:
                print(f"  - {issue}")
        
        # Print key achievements
        if exec_summary['key_achievements']:
            print("\nKEY ACHIEVEMENTS:")
            for achievement in exec_summary['key_achievements']:
                print(f"  - {achievement}")
        
        print(f"\nDetailed reports saved to: {output_dir}")
        print("="*80)
        
        # Return appropriate exit code
        if exec_summary['production_ready'] and exec_summary['sow_compliant']:
            logger.info("All validation criteria met. System ready for production.")
            return 0
        else:
            logger.warning("Some validation criteria not met. Review recommendations before production.")
            return 2
            
    except Exception as e:
        logger.error(f"Error generating final report: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)