#!/usr/bin/env python3
"""
Main benchmark runner script for TailChasing Fixer evaluation.

This script runs comprehensive benchmarks to evaluate convergence performance,
token usage, and fix quality across different scenarios and models.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List
import json
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from benchmarks.scenarios import (
    SimpleImportScenario,
    CircularDependencyScenario,
    SemanticDuplicateScenario,
    ComplexRefactoringScenario,
)

from benchmarks.runners import (
    BenchmarkRunner,
    MultiModelRunner,
    PerformanceTracker,
)

from benchmarks.runners.multi_model import ModelConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_mock_fix_function(model_name: str, success_rate: float = 0.8):
    """
    Create a mock fix function for testing the benchmark harness.
    
    In production, this would be replaced with actual LLM fix functions.
    """
    import random
    
    def fix_function(temp_dir: Path, current_code: Dict[str, str]) -> Dict[str, Any]:
        """Mock fix function that simulates LLM behavior."""
        # Simulate some processing time
        import time
        time.sleep(random.uniform(0.1, 0.5))
        
        # Simulate success/failure based on success rate
        success = random.random() < success_rate
        
        # Simulate token usage
        tokens_used = random.randint(500, 3000)
        
        if success:
            # For mock purposes, return a "fixed" version
            # In reality, this would be the LLM's attempt to fix the code
            return {
                "success": True,
                "fixed_code": current_code,  # Mock: return same code
                "tokens_used": tokens_used,
                "description": f"Mock fix by {model_name}",
                "files_changed": list(current_code.keys())
            }
        else:
            return {
                "success": False,
                "error": "Mock failure - simulating LLM error",
                "tokens_used": tokens_used
            }
    
    return fix_function


def run_single_model_benchmarks(args):
    """Run benchmarks for a single model."""
    logger.info(f"Running single model benchmarks: {args.model}")
    
    # Create scenarios based on selection
    scenarios = []
    
    if args.scenario == "all" or args.scenario == "simple":
        scenarios.append(SimpleImportScenario())
    
    if args.scenario == "all" or args.scenario == "circular":
        scenarios.append(CircularDependencyScenario())
    
    if args.scenario == "all" or args.scenario == "semantic":
        scenarios.append(SemanticDuplicateScenario())
    
    if args.scenario == "all" or args.scenario == "complex":
        scenarios.append(ComplexRefactoringScenario())
    
    # Create fix function (mock for demonstration)
    fix_function = create_mock_fix_function(args.model, success_rate=0.7)
    
    # Create runner
    runner = BenchmarkRunner(
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        timeout_seconds=args.timeout
    )
    
    # Run benchmarks
    summary = runner.run_suite(scenarios, fix_function, args.model)
    
    # Print results
    print("\n" + "="*60)
    print(f"BENCHMARK RESULTS - {args.model}")
    print("="*60)
    print(f"Scenarios run: {summary['scenarios_run']}")
    print(f"Success rate: {summary['success_rate']*100:.1f}%")
    print(f"Total time: {summary['total_time']:.2f}s")
    print(f"Total tokens: {summary['total_tokens']:,}")
    print(f"Total cost: ${summary['total_cost']:.4f}")
    print("\nPer-scenario results:")
    print("-"*40)
    
    for result in summary['results']:
        status = "✅" if result['success'] else "❌"
        print(f"{status} {result['scenario']}: {result['steps']} steps, "
              f"{result['time']:.2f}s, ${result['cost']:.4f}")
        if result['error']:
            print(f"   Error: {result['error']}")
    
    # Save summary
    output_file = Path(args.output_dir) / f"{args.model}_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"Results saved to: {output_file}")


def run_comparison_benchmarks(args):
    """Run comparison benchmarks across multiple models."""
    logger.info("Running multi-model comparison benchmarks")
    
    # Create scenarios
    scenarios = []
    
    if args.scenario == "all" or args.scenario == "simple":
        scenarios.append(SimpleImportScenario())
    
    if args.scenario == "all" or args.scenario == "circular":
        scenarios.append(CircularDependencyScenario())
    
    if args.scenario == "all" or args.scenario == "semantic":
        scenarios.append(SemanticDuplicateScenario())
    
    if args.scenario == "all" or args.scenario == "complex":
        scenarios.append(ComplexRefactoringScenario())
    
    # Create multi-model runner
    runner = MultiModelRunner(
        output_dir=Path(args.output_dir),
        max_steps=args.max_steps,
        timeout_seconds=args.timeout,
        parallel=args.parallel
    )
    
    # Add models to compare
    models_to_test = args.compare_models.split(',')
    
    for model_name in models_to_test:
        # Create mock fix functions with different success rates for demonstration
        # In production, these would be actual LLM integrations
        success_rates = {
            "gpt-4": 0.85,
            "gpt-3.5": 0.70,
            "claude-3": 0.80,
            "llama-70b": 0.65,
            "mistral": 0.60
        }
        
        success_rate = success_rates.get(model_name.lower(), 0.5)
        fix_function = create_mock_fix_function(model_name, success_rate)
        
        runner.add_model(ModelConfig(
            name=model_name,
            fix_function=fix_function
        ))
    
    # Run comparison
    report = runner.run_comparison(scenarios, save_report=True)
    
    # Print comparison summary
    runner.print_comparison_summary()
    
    # Print rankings
    if "rankings" in report:
        print("\n" + "="*60)
        print("MODEL RANKINGS")
        print("="*60)
        
        for metric, ranking in report["rankings"].items():
            print(f"\n{metric.replace('_', ' ').title()}:")
            for i, model in enumerate(ranking, 1):
                print(f"  {i}. {model}")


def track_performance(args):
    """Track and analyze performance trends."""
    logger.info("Analyzing performance trends")
    
    tracker = PerformanceTracker(Path(args.output_dir) / "performance.db")
    
    # Generate performance report
    report = tracker.generate_performance_report(
        Path(args.output_dir) / "performance_report.json"
    )
    
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # Print summary
    print("\nOverall Summary (last 30 days):")
    print("-"*40)
    summary = report["summary"]
    print(f"Total runs: {summary.get('total_runs', 0)}")
    print(f"Success rate: {summary.get('overall_success_rate', 0)*100:.1f}%")
    print(f"Average steps: {summary.get('avg_steps', 0):.1f}")
    print(f"Average time: {summary.get('avg_time', 0):.2f}s")
    print(f"Average cost: ${summary.get('avg_cost', 0):.4f}")
    
    # Print problem patterns
    patterns = report.get("problem_patterns", {})
    
    if patterns.get("frequently_failing"):
        print("\n⚠️  Frequently Failing Scenarios:")
        for item in patterns["frequently_failing"][:5]:
            print(f"  - {item['scenario_name']} ({item['model_name']}): "
                  f"{item['success_rate']*100:.1f}% success rate")
    
    if patterns.get("regression_prone"):
        print("\n⚠️  Regression-Prone Scenarios:")
        for item in patterns["regression_prone"][:5]:
            print(f"  - {item['scenario_name']} ({item['model_name']}): "
                  f"{item['avg_regressions']:.1f} avg regressions")
    
    if patterns.get("inefficient"):
        print("\n⚠️  Inefficient Scenarios:")
        for item in patterns["inefficient"][:5]:
            print(f"  - {item['scenario_name']} ({item['model_name']}): "
                  f"{item['avg_steps']:.1f} steps (expected max: {item['expected_max']:.0f})")
    
    # Plot trends if requested
    if args.plot:
        plot_path = Path(args.output_dir) / "performance_trends.png"
        tracker.plot_performance_trends(save_path=plot_path)
        print(f"\n📊 Performance plots saved to: {plot_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run benchmarks for TailChasing Fixer evaluation"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Single model benchmark
    single_parser = subparsers.add_parser('single', help='Run single model benchmark')
    single_parser.add_argument('--model', default='gpt-4', help='Model name')
    single_parser.add_argument('--scenario', choices=['all', 'simple', 'circular', 'semantic', 'complex'],
                              default='all', help='Scenario to run')
    single_parser.add_argument('--max-steps', type=int, default=20, help='Max steps per scenario')
    single_parser.add_argument('--timeout', type=int, default=300, help='Timeout per scenario (seconds)')
    single_parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    
    # Comparison benchmark
    compare_parser = subparsers.add_parser('compare', help='Compare multiple models')
    compare_parser.add_argument('--compare-models', default='gpt-4,gpt-3.5,claude-3',
                               help='Comma-separated list of models to compare')
    compare_parser.add_argument('--scenario', choices=['all', 'simple', 'circular', 'semantic', 'complex'],
                               default='all', help='Scenario to run')
    compare_parser.add_argument('--max-steps', type=int, default=20, help='Max steps per scenario')
    compare_parser.add_argument('--timeout', type=int, default=300, help='Timeout per scenario (seconds)')
    compare_parser.add_argument('--parallel', action='store_true', help='Run models in parallel')
    compare_parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    
    # Performance tracking
    track_parser = subparsers.add_parser('track', help='Track performance trends')
    track_parser.add_argument('--output-dir', default='benchmark_results', help='Output directory')
    track_parser.add_argument('--plot', action='store_true', help='Generate performance plots')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'single':
        run_single_model_benchmarks(args)
    elif args.command == 'compare':
        run_comparison_benchmarks(args)
    elif args.command == 'track':
        track_performance(args)


if __name__ == "__main__":
    main()