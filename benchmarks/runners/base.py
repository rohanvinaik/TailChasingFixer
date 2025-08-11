"""
Base benchmark runner for executing scenarios.
"""

import time
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Callable
from abc import ABC, abstractmethod

from ..scenarios.base import BenchmarkScenario, ScenarioResult
from ..metrics import MetricsCollector, BenchmarkMetrics


logger = logging.getLogger(__name__)


class BenchmarkRunner:
    """Runs benchmark scenarios and collects metrics."""
    
    def __init__(self, 
                 output_dir: Optional[Path] = None,
                 max_steps: int = 20,
                 timeout_seconds: int = 300):
        """
        Initialize benchmark runner.
        
        Args:
            output_dir: Directory for output files
            max_steps: Maximum steps before giving up
            timeout_seconds: Timeout for each benchmark
        """
        self.output_dir = output_dir or Path("benchmark_results")
        self.max_steps = max_steps
        self.timeout_seconds = timeout_seconds
        
        self.metrics_collector = MetricsCollector(self.output_dir)
        self.current_scenario: Optional[BenchmarkScenario] = None
    
    def run_scenario(self, 
                    scenario: BenchmarkScenario,
                    fix_function: Callable,
                    model_name: str = "unknown") -> ScenarioResult:
        """
        Run a single benchmark scenario.
        
        Args:
            scenario: The scenario to run
            fix_function: Function that attempts to fix the code
            model_name: Name of the model being tested
            
        Returns:
            ScenarioResult with execution details
        """
        logger.info(f"Starting benchmark: {scenario.name} with {model_name}")
        
        # Setup scenario
        self.current_scenario = scenario
        temp_dir = scenario.setup()
        
        # Start metrics collection
        metrics = self.metrics_collector.start_benchmark(scenario.name, model_name)
        metrics.convergence.start_tracking(
            scenario.expected_steps[0],
            scenario.expected_steps[1]
        )
        
        # Track execution
        start_time = time.time()
        success = False
        error_message = None
        steps_taken = 0
        fixes_applied = []
        
        try:
            # Get initial code state
            initial_code = scenario.get_initial_code()
            current_code = initial_code.copy()
            
            # Attempt to fix iteratively
            for step in range(1, self.max_steps + 1):
                # Check timeout
                if time.time() - start_time > self.timeout_seconds:
                    error_message = f"Timeout after {self.timeout_seconds} seconds"
                    break
                
                logger.debug(f"Step {step} for {scenario.name}")
                
                # Run fix function
                try:
                    fix_result = fix_function(temp_dir, current_code)
                    
                    # Record metrics
                    tokens_used = fix_result.get("tokens_used", 0)
                    self.metrics_collector.record_step(
                        step, tokens_used, 
                        success=fix_result.get("success", False),
                        error=fix_result.get("error")
                    )
                    
                    if fix_result.get("success"):
                        # Update current code
                        if "fixed_code" in fix_result:
                            current_code = fix_result["fixed_code"]
                        
                        # Track fix
                        fixes_applied.append({
                            "step": step,
                            "description": fix_result.get("description", ""),
                            "files_changed": fix_result.get("files_changed", [])
                        })
                        
                        # Check if solution is valid
                        is_valid, validation_errors = scenario.validate_solution(current_code)
                        
                        if is_valid:
                            success = True
                            steps_taken = step
                            logger.info(f"Scenario {scenario.name} converged in {step} steps")
                            break
                        elif validation_errors:
                            logger.debug(f"Validation errors: {validation_errors}")
                    
                    else:
                        # Fix failed
                        error_message = fix_result.get("error", "Unknown error")
                        
                except Exception as e:
                    logger.error(f"Error in fix function: {e}")
                    error_message = str(e)
                    self.metrics_collector.record_step(step, 0, success=False, error=str(e))
                
                # Check for regressions
                before_code = current_code.copy()
                regressions = scenario.check_regressions(initial_code, current_code)
                for regression in regressions:
                    self.metrics_collector.record_regression(regression)
            
            # If we exhausted steps without success
            if not success and not error_message:
                error_message = f"Failed to converge after {self.max_steps} steps"
                steps_taken = self.max_steps
        
        except Exception as e:
            logger.error(f"Benchmark execution error: {e}")
            error_message = str(e)
        
        finally:
            # Finish metrics collection
            elapsed_time = time.time() - start_time
            final_metrics = self.metrics_collector.finish_benchmark(success, error_message)
            
            # Cleanup
            scenario.cleanup()
            self.current_scenario = None
        
        # Create result
        result = ScenarioResult(
            scenario_name=scenario.name,
            success=success,
            steps_taken=steps_taken,
            expected_steps=scenario.expected_steps,
            time_elapsed=elapsed_time,
            tokens_used=final_metrics.cost.total_tokens,
            cost_estimate=final_metrics.cost.total_cost,
            error_message=error_message,
            regressions_detected=final_metrics.convergence.regressions,
            fixes_applied=fixes_applied
        )
        
        return result
    
    def run_suite(self,
                 scenarios: List[BenchmarkScenario],
                 fix_function: Callable,
                 model_name: str = "unknown") -> Dict[str, Any]:
        """
        Run a suite of benchmark scenarios.
        
        Args:
            scenarios: List of scenarios to run
            fix_function: Function that attempts to fix code
            model_name: Name of the model being tested
            
        Returns:
            Summary report of all scenarios
        """
        logger.info(f"Starting benchmark suite with {len(scenarios)} scenarios")
        
        results = []
        successful = 0
        total_time = 0
        total_tokens = 0
        total_cost = 0
        
        for scenario in scenarios:
            result = self.run_scenario(scenario, fix_function, model_name)
            results.append(result)
            
            if result.success:
                successful += 1
            
            total_time += result.time_elapsed
            total_tokens += result.tokens_used
            total_cost += result.cost_estimate
        
        # Generate summary
        summary = {
            "model": model_name,
            "scenarios_run": len(scenarios),
            "scenarios_passed": successful,
            "success_rate": successful / len(scenarios) if scenarios else 0,
            "total_time": total_time,
            "total_tokens": total_tokens,
            "total_cost": total_cost,
            "average_time": total_time / len(scenarios) if scenarios else 0,
            "average_tokens": total_tokens / len(scenarios) if scenarios else 0,
            "average_cost": total_cost / len(scenarios) if scenarios else 0,
            "results": [
                {
                    "scenario": r.scenario_name,
                    "success": r.success,
                    "steps": r.steps_taken,
                    "expected_steps": r.expected_steps,
                    "efficiency_score": r.efficiency_score,
                    "time": r.time_elapsed,
                    "tokens": r.tokens_used,
                    "cost": r.cost_estimate,
                    "regressions": len(r.regressions_detected),
                    "error": r.error_message
                }
                for r in results
            ]
        }
        
        return summary