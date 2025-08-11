"""
Multi-model benchmark runner for comparing different LLMs.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime
import concurrent.futures

from .base import BenchmarkRunner
from ..scenarios.base import BenchmarkScenario


logger = logging.getLogger(__name__)


class ModelConfig:
    """Configuration for a model to test."""
    
    def __init__(self, 
                 name: str,
                 fix_function: Callable,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 **kwargs):
        self.name = name
        self.fix_function = fix_function
        self.api_key = api_key
        self.endpoint = endpoint
        self.extra_params = kwargs


class MultiModelRunner:
    """Run benchmarks across multiple models for comparison."""
    
    def __init__(self,
                 output_dir: Optional[Path] = None,
                 max_steps: int = 20,
                 timeout_seconds: int = 300,
                 parallel: bool = False):
        """
        Initialize multi-model runner.
        
        Args:
            output_dir: Directory for output files
            max_steps: Maximum steps per scenario
            timeout_seconds: Timeout per scenario
            parallel: Run models in parallel
        """
        self.output_dir = output_dir or Path("benchmark_results")
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        self.parallel = parallel
        
        self.models: List[ModelConfig] = []
        self.results: Dict[str, Dict] = {}
    
    def add_model(self, model_config: ModelConfig):
        """Add a model to test."""
        self.models.append(model_config)
        logger.info(f"Added model: {model_config.name}")
    
    def run_comparison(self, 
                      scenarios: List[BenchmarkScenario],
                      save_report: bool = True) -> Dict[str, Any]:
        """
        Run comparison across all configured models.
        
        Args:
            scenarios: List of scenarios to test
            save_report: Whether to save comparison report to file
            
        Returns:
            Comparison report dictionary
        """
        logger.info(f"Starting comparison with {len(self.models)} models and {len(scenarios)} scenarios")
        
        if self.parallel and len(self.models) > 1:
            return self._run_parallel(scenarios, save_report)
        else:
            return self._run_sequential(scenarios, save_report)
    
    def _run_sequential(self, scenarios: List[BenchmarkScenario], save_report: bool) -> Dict[str, Any]:
        """Run models sequentially."""
        model_results = {}
        
        for model in self.models:
            logger.info(f"Running benchmarks for model: {model.name}")
            
            runner = BenchmarkRunner(
                output_dir=self.output_dir / model.name,
                max_steps=self.max_steps,
                timeout_seconds=self.timeout_seconds
            )
            
            summary = runner.run_suite(scenarios, model.fix_function, model.name)
            model_results[model.name] = summary
        
        return self._generate_comparison_report(model_results, save_report)
    
    def _run_parallel(self, scenarios: List[BenchmarkScenario], save_report: bool) -> Dict[str, Any]:
        """Run models in parallel."""
        model_results = {}
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.models)) as executor:
            # Submit all model runs
            future_to_model = {}
            for model in self.models:
                runner = BenchmarkRunner(
                    output_dir=self.output_dir / model.name,
                    max_steps=self.max_steps,
                    timeout_seconds=self.timeout_seconds
                )
                
                future = executor.submit(
                    runner.run_suite,
                    scenarios,
                    model.fix_function,
                    model.name
                )
                future_to_model[future] = model.name
            
            # Collect results
            for future in concurrent.futures.as_completed(future_to_model):
                model_name = future_to_model[future]
                try:
                    summary = future.result()
                    model_results[model_name] = summary
                    logger.info(f"Completed benchmarks for model: {model_name}")
                except Exception as e:
                    logger.error(f"Error running model {model_name}: {e}")
                    model_results[model_name] = {"error": str(e)}
        
        return self._generate_comparison_report(model_results, save_report)
    
    def _generate_comparison_report(self, model_results: Dict[str, Dict], save_report: bool) -> Dict[str, Any]:
        """Generate comparison report from model results."""
        report = {
            "timestamp": datetime.now().isoformat(),
            "models_tested": len(model_results),
            "scenarios_count": len(model_results[list(model_results.keys())[0]].get("results", [])) if model_results else 0,
            "model_summaries": {},
            "scenario_comparisons": {},
            "rankings": {}
        }
        
        # Collect model summaries
        for model_name, summary in model_results.items():
            if "error" not in summary:
                report["model_summaries"][model_name] = {
                    "success_rate": summary["success_rate"],
                    "average_steps": sum(r["steps"] for r in summary["results"]) / len(summary["results"]) if summary["results"] else 0,
                    "average_time": summary["average_time"],
                    "average_tokens": summary["average_tokens"],
                    "average_cost": summary["average_cost"],
                    "total_regressions": sum(r["regressions"] for r in summary["results"])
                }
        
        # Compare by scenario
        if model_results:
            first_model = list(model_results.keys())[0]
            if "results" in model_results[first_model]:
                for result in model_results[first_model]["results"]:
                    scenario_name = result["scenario"]
                    report["scenario_comparisons"][scenario_name] = {}
                    
                    for model_name, summary in model_results.items():
                        if "error" not in summary:
                            # Find this scenario's result for this model
                            scenario_result = next(
                                (r for r in summary["results"] if r["scenario"] == scenario_name),
                                None
                            )
                            
                            if scenario_result:
                                report["scenario_comparisons"][scenario_name][model_name] = {
                                    "success": scenario_result["success"],
                                    "steps": scenario_result["steps"],
                                    "efficiency_score": scenario_result["efficiency_score"],
                                    "time": scenario_result["time"],
                                    "tokens": scenario_result["tokens"],
                                    "cost": scenario_result["cost"]
                                }
        
        # Generate rankings
        report["rankings"] = self._calculate_rankings(report["model_summaries"])
        
        # Save report if requested
        if save_report:
            report_path = self.output_dir / f"comparison_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            report_path.parent.mkdir(exist_ok=True, parents=True)
            
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Saved comparison report to: {report_path}")
        
        self.results = model_results
        return report
    
    def _calculate_rankings(self, model_summaries: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Calculate model rankings by different metrics."""
        if not model_summaries:
            return {}
        
        rankings = {}
        
        # Rank by success rate
        rankings["by_success_rate"] = sorted(
            model_summaries.keys(),
            key=lambda m: model_summaries[m]["success_rate"],
            reverse=True
        )
        
        # Rank by speed (average time)
        rankings["by_speed"] = sorted(
            model_summaries.keys(),
            key=lambda m: model_summaries[m]["average_time"]
        )
        
        # Rank by efficiency (fewest steps)
        rankings["by_efficiency"] = sorted(
            model_summaries.keys(),
            key=lambda m: model_summaries[m]["average_steps"]
        )
        
        # Rank by cost
        rankings["by_cost"] = sorted(
            model_summaries.keys(),
            key=lambda m: model_summaries[m]["average_cost"]
        )
        
        # Calculate overall score and rank
        overall_scores = {}
        for model in model_summaries:
            score = 0
            # Success rate is most important (40%)
            score += model_summaries[model]["success_rate"] * 40
            
            # Efficiency (30%) - normalize steps (fewer is better)
            max_steps = max(m["average_steps"] for m in model_summaries.values())
            if max_steps > 0:
                efficiency = 1 - (model_summaries[model]["average_steps"] / max_steps)
                score += efficiency * 30
            
            # Speed (20%) - normalize time (faster is better)
            max_time = max(m["average_time"] for m in model_summaries.values())
            if max_time > 0:
                speed = 1 - (model_summaries[model]["average_time"] / max_time)
                score += speed * 20
            
            # Cost (10%) - normalize cost (cheaper is better)
            max_cost = max(m["average_cost"] for m in model_summaries.values())
            if max_cost > 0:
                cost_efficiency = 1 - (model_summaries[model]["average_cost"] / max_cost)
                score += cost_efficiency * 10
            
            overall_scores[model] = score
        
        rankings["overall"] = sorted(
            overall_scores.keys(),
            key=lambda m: overall_scores[m],
            reverse=True
        )
        
        return rankings
    
    def print_comparison_summary(self):
        """Print a formatted comparison summary to console."""
        if not self.results:
            print("No results available. Run comparison first.")
            return
        
        print("\n" + "="*80)
        print("BENCHMARK COMPARISON SUMMARY")
        print("="*80)
        
        # Model performance table
        print("\nModel Performance:")
        print("-"*60)
        print(f"{'Model':<20} {'Success Rate':<15} {'Avg Steps':<12} {'Avg Time(s)':<12} {'Avg Cost':<10}")
        print("-"*60)
        
        for model_name, summary in self.results.items():
            if "error" not in summary:
                print(f"{model_name:<20} {summary['success_rate']*100:<14.1f}% "
                      f"{sum(r['steps'] for r in summary['results'])/len(summary['results']):<11.1f} "
                      f"{summary['average_time']:<11.2f} "
                      f"${summary['average_cost']:<9.4f}")
        
        print("-"*60)
        
        # Best performers by metric
        print("\nBest Performers:")
        print("-"*40)
        
        if len(self.results) > 1:
            # Find best by each metric
            best_success = max(self.results.keys(), 
                             key=lambda m: self.results[m].get("success_rate", 0))
            best_speed = min(self.results.keys(),
                           key=lambda m: self.results[m].get("average_time", float('inf')))
            best_efficiency = min(self.results.keys(),
                                key=lambda m: sum(r['steps'] for r in self.results[m].get("results", [{"steps": float('inf')}])))
            best_cost = min(self.results.keys(),
                          key=lambda m: self.results[m].get("average_cost", float('inf')))
            
            print(f"Highest Success Rate: {best_success}")
            print(f"Fastest: {best_speed}")
            print(f"Most Efficient: {best_efficiency}")
            print(f"Most Cost-Effective: {best_cost}")
        
        print("="*80)