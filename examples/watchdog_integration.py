"""
Example of integrating the watchdog system with analyzers.

This example shows how to use the watchdog system to monitor analyzers,
handle timeouts, and generate execution reports.
"""

import time
import logging
from typing import List

# Add the parent directory to path for imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from tailchasing.utils.watchdog import (
    create_watchdog, AnalyzerWatchdog, WatchdogConfig,
    SemanticAnalysisFallback, monitor_analyzer
)
from tailchasing.core.issues import Issue

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlowAnalyzer:
    """Example of a slow analyzer that might timeout."""
    
    def run(self, ctx=None):
        logger.info("SlowAnalyzer starting...")
        # Simulate long-running analysis
        for i in range(10):
            time.sleep(0.5)  # Simulate work
            logger.info(f"SlowAnalyzer progress: {i+1}/10")
        
        return [Issue(kind='slow_issue', message='Found after long analysis', severity=2)]


class FastAnalyzer:
    """Example of a fast analyzer."""
    
    def run(self, ctx=None):
        logger.info("FastAnalyzer running...")
        time.sleep(0.1)
        return [Issue(kind='fast_issue', message='Quick find', severity=1)]


class CrashingAnalyzer:
    """Example of an analyzer that crashes."""
    
    def run(self, ctx=None):
        logger.info("CrashingAnalyzer starting...")
        time.sleep(0.1)
        raise RuntimeError("Simulated analyzer crash")


def example_basic_usage():
    """Example of basic watchdog usage."""
    print("\n=== Basic Watchdog Usage ===")
    
    # Create watchdog with custom configuration
    watchdog = create_watchdog(
        timeout=3.0,  # 3 second timeout
        heartbeat_interval=0.5,  # Heartbeat every 0.5 seconds
        enable_fallback=True
    )
    
    analyzers = [
        (FastAnalyzer(), "fast_analyzer"),
        (SlowAnalyzer(), "slow_analyzer"),  # This will timeout
        (CrashingAnalyzer(), "crashing_analyzer")
    ]
    
    results = {}
    for analyzer, name in analyzers:
        print(f"\nRunning {name}...")
        wrapped_run = watchdog.wrap_analyzer(analyzer, name)
        issues = wrapped_run()
        results[name] = issues
        print(f"  → Found {len(issues)} issues")
    
    # Generate execution report
    report = watchdog.get_execution_report()
    print(f"\n=== Execution Report ===")
    print(f"Total analyzers: {report['summary']['total_analyzers']}")
    print(f"Total duration: {report['summary']['total_duration']:.2f}s")
    print(f"Total timeouts: {report['summary']['total_timeouts']}")
    print(f"Total errors: {report['summary']['total_errors']}")
    
    print("\nAnalyzer Details:")
    for name, details in report['analyzer_details'].items():
        print(f"  {name}: {details['status']} ({details['duration']:.2f}s)")
    
    watchdog.shutdown()
    return results


def example_decorator_usage():
    """Example using the decorator approach."""
    print("\n=== Decorator Usage ===")
    
    # Create global watchdog
    global_watchdog = create_watchdog(timeout=2.0)
    
    # Use decorator to wrap analyzer
    @monitor_analyzer
    def decorated_analyzer(ctx=None):
        time.sleep(0.2)
        return [Issue(kind='decorated', message='From decorated analyzer', severity=1)]
    
    # This won't work as expected because the decorator needs the analyzer object
    # Instead, let's show the manual approach:
    
    class DecoratedAnalyzer:
        def run(self, ctx=None):
            time.sleep(0.2)
            return [Issue(kind='decorated', message='From decorated analyzer', severity=1)]
    
    analyzer = DecoratedAnalyzer()
    wrapped_run = global_watchdog.wrap_analyzer(analyzer, "decorated_analyzer")
    issues = wrapped_run()
    
    print(f"Decorated analyzer found {len(issues)} issues")
    
    global_watchdog.shutdown()


def example_advanced_configuration():
    """Example of advanced watchdog configuration."""
    print("\n=== Advanced Configuration ===")
    
    # Create custom configuration
    config = WatchdogConfig(
        analyzer_timeout=5.0,
        heartbeat_interval=1.0,
        heartbeat_timeout_multiplier=3.0,
        enable_fallback=True,
        enable_threading=True,
        max_retries=2,
        verbose_logging=True,
        execution_report=True
    )
    
    watchdog = AnalyzerWatchdog(config)
    
    # Test with a borderline timeout case
    class BorderlineAnalyzer:
        def run(self, ctx=None):
            # Takes about 4 seconds, should complete within 5s timeout
            for i in range(8):
                time.sleep(0.5)
                logger.info(f"BorderlineAnalyzer step {i+1}/8")
            return [Issue(kind='borderline', message='Completed just in time', severity=1)]
    
    analyzer = BorderlineAnalyzer()
    wrapped_run = watchdog.wrap_analyzer(analyzer, "borderline_analyzer")
    issues = wrapped_run()
    
    print(f"Borderline analyzer found {len(issues)} issues")
    
    # Show detailed report
    report = watchdog.get_execution_report()
    if 'borderline_analyzer' in report['analyzer_details']:
        details = report['analyzer_details']['borderline_analyzer']
        print(f"Execution details: {details}")
    
    watchdog.shutdown()


def example_semantic_fallback():
    """Example of semantic analysis fallback."""
    print("\n=== Semantic Fallback Example ===")
    
    # Create sample function data for fallback
    sample_functions = [
        {
            'name': 'calculate_sum',
            'file': 'math_utils.py',
            'line': 10,
            'body': 'def calculate_sum(a, b): return a + b'
        },
        {
            'name': 'add_numbers',
            'file': 'helpers.py', 
            'line': 5,
            'body': 'def add_numbers(x, y): return x + y'
        },
        {
            'name': 'multiply',
            'file': 'math_utils.py',
            'line': 20,
            'body': 'def multiply(a, b): return a * b'
        }
    ]
    
    # Test TF-IDF fallback
    try:
        issues = SemanticAnalysisFallback.tfidf_fallback(sample_functions, threshold=0.6)
        print(f"TF-IDF fallback found {len(issues)} potential duplicates")
        for issue in issues:
            print(f"  - {issue.message}")
    except Exception as e:
        print(f"TF-IDF fallback not available: {e}")


if __name__ == "__main__":
    print("Watchdog Integration Examples")
    print("=" * 50)
    
    example_basic_usage()
    example_decorator_usage()
    example_advanced_configuration()
    example_semantic_fallback()
    
    print("\n✅ All examples completed successfully!")