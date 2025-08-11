"""
Main metrics collector that aggregates all metrics.
"""

import time
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field, asdict
from datetime import datetime

from .convergence import ConvergenceMetrics
from .performance import PerformanceMetrics
from .cost import CostMetrics


@dataclass
class BenchmarkMetrics:
    """Complete metrics for a benchmark run."""
    
    # Identifiers
    scenario_name: str
    model_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Core metrics
    convergence: ConvergenceMetrics = field(default_factory=ConvergenceMetrics)
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    cost: CostMetrics = field(default_factory=CostMetrics)
    
    # Overall results
    success: bool = False
    error_message: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "scenario_name": self.scenario_name,
            "model_name": self.model_name,
            "timestamp": self.timestamp.isoformat(),
            "convergence": asdict(self.convergence),
            "performance": asdict(self.performance),
            "cost": asdict(self.cost),
            "success": self.success,
            "error_message": self.error_message
        }
    
    def calculate_overall_score(self) -> float:
        """Calculate overall benchmark score (0-100)."""
        if not self.success:
            return 0.0
        
        # Weight different aspects
        convergence_score = self.convergence.get_efficiency_score() * 0.4
        performance_score = self.performance.get_speed_score() * 0.3
        cost_score = self.cost.get_efficiency_score() * 0.3
        
        return convergence_score + performance_score + cost_score


class MetricsCollector:
    """Collects and aggregates metrics during benchmark execution."""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("benchmark_results")
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        self.current_metrics: Optional[BenchmarkMetrics] = None
        self.all_metrics: List[BenchmarkMetrics] = []
        
        # Timing tracking
        self.start_time: Optional[float] = None
        self.step_times: List[float] = []
    
    def start_benchmark(self, scenario_name: str, model_name: str) -> BenchmarkMetrics:
        """Start collecting metrics for a benchmark."""
        self.current_metrics = BenchmarkMetrics(
            scenario_name=scenario_name,
            model_name=model_name
        )
        
        self.start_time = time.time()
        self.step_times = []
        
        # Initialize sub-metrics
        self.current_metrics.convergence.start_tracking()
        self.current_metrics.performance.start_tracking()
        
        return self.current_metrics
    
    def record_step(self, step_number: int, tokens_used: int = 0, 
                    success: bool = True, error: Optional[str] = None):
        """Record metrics for a single step."""
        if not self.current_metrics:
            raise RuntimeError("No benchmark in progress")
        
        step_time = time.time()
        if self.step_times:
            step_duration = step_time - self.step_times[-1]
        else:
            step_duration = step_time - self.start_time
        
        self.step_times.append(step_time)
        
        # Update convergence metrics
        self.current_metrics.convergence.record_step(
            step_number, success, error
        )
        
        # Update performance metrics
        self.current_metrics.performance.record_step_time(step_duration)
        
        # Update cost metrics
        if tokens_used > 0:
            self.current_metrics.cost.record_tokens(tokens_used)
    
    def record_regression(self, regression_description: str):
        """Record a regression detected during fixing."""
        if not self.current_metrics:
            raise RuntimeError("No benchmark in progress")
        
        self.current_metrics.convergence.add_regression(regression_description)
    
    def finish_benchmark(self, success: bool, error_message: Optional[str] = None) -> BenchmarkMetrics:
        """Finish collecting metrics for the current benchmark."""
        if not self.current_metrics:
            raise RuntimeError("No benchmark in progress")
        
        # Calculate total time
        total_time = time.time() - self.start_time
        
        # Finalize sub-metrics
        self.current_metrics.convergence.finish_tracking(success)
        self.current_metrics.performance.finish_tracking(total_time)
        self.current_metrics.cost.calculate_total_cost()
        
        # Set overall results
        self.current_metrics.success = success
        self.current_metrics.error_message = error_message
        
        # Save metrics
        self.all_metrics.append(self.current_metrics)
        self.save_metrics(self.current_metrics)
        
        result = self.current_metrics
        self.current_metrics = None
        
        return result
    
    def save_metrics(self, metrics: BenchmarkMetrics):
        """Save metrics to file."""
        # Create filename with timestamp
        filename = f"{metrics.scenario_name}_{metrics.model_name}_{metrics.timestamp.strftime('%Y%m%d_%H%M%S')}.json"
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)
    
    def load_historical_metrics(self, scenario_name: Optional[str] = None,
                               model_name: Optional[str] = None) -> List[BenchmarkMetrics]:
        """Load historical metrics from files."""
        historical = []
        
        for filepath in self.output_dir.glob("*.json"):
            with open(filepath) as f:
                data = json.load(f)
            
            # Filter if requested
            if scenario_name and data["scenario_name"] != scenario_name:
                continue
            if model_name and data["model_name"] != model_name:
                continue
            
            # Reconstruct metrics object
            metrics = BenchmarkMetrics(
                scenario_name=data["scenario_name"],
                model_name=data["model_name"],
                timestamp=datetime.fromisoformat(data["timestamp"]),
                success=data["success"],
                error_message=data.get("error_message")
            )
            
            # Reconstruct sub-metrics
            metrics.convergence = ConvergenceMetrics(**data["convergence"])
            metrics.performance = PerformanceMetrics(**data["performance"])
            metrics.cost = CostMetrics(**data["cost"])
            
            historical.append(metrics)
        
        return sorted(historical, key=lambda m: m.timestamp)
    
    def generate_summary_report(self, metrics_list: Optional[List[BenchmarkMetrics]] = None) -> Dict[str, Any]:
        """Generate summary report from metrics."""
        if metrics_list is None:
            metrics_list = self.all_metrics
        
        if not metrics_list:
            return {"error": "No metrics available"}
        
        # Group by scenario
        by_scenario = {}
        for metrics in metrics_list:
            if metrics.scenario_name not in by_scenario:
                by_scenario[metrics.scenario_name] = []
            by_scenario[metrics.scenario_name].append(metrics)
        
        summary = {
            "total_benchmarks": len(metrics_list),
            "scenarios": {}
        }
        
        for scenario, scenario_metrics in by_scenario.items():
            successful = [m for m in scenario_metrics if m.success]
            
            summary["scenarios"][scenario] = {
                "total_runs": len(scenario_metrics),
                "successful_runs": len(successful),
                "success_rate": len(successful) / len(scenario_metrics) if scenario_metrics else 0,
                "average_steps": sum(m.convergence.total_steps for m in successful) / len(successful) if successful else 0,
                "average_time": sum(m.performance.total_time for m in successful) / len(successful) if successful else 0,
                "average_tokens": sum(m.cost.total_tokens for m in successful) / len(successful) if successful else 0,
                "average_cost": sum(m.cost.total_cost for m in successful) / len(successful) if successful else 0,
                "regression_rate": sum(len(m.convergence.regressions) > 0 for m in scenario_metrics) / len(scenario_metrics) if scenario_metrics else 0
            }
        
        return summary