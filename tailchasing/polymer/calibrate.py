"""
Calibration tools for tuning polymer physics parameters.

This module provides functionality for calibrating the polymer physics
parameters (alpha, weights, epsilon, kappa) based on real codebase data
and observed thrashing patterns.
"""

import logging
import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from rich.console import Console
from rich.progress import track
from rich.table import Table
from scipy.optimize import minimize
from scipy.stats import pearsonr


logger = logging.getLogger(__name__)


@dataclass
class ThrashEvent:
    """Represents an observed thrashing event for calibration."""
    
    file1: str
    file2: str
    distance_type: str  # 'tok', 'ast', 'mod', 'git'
    observed_frequency: float
    latency_ms: float
    timestamp: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "file1": self.file1,
            "file2": self.file2,
            "distance_type": self.distance_type,
            "observed_frequency": self.observed_frequency,
            "latency_ms": self.latency_ms,
            "timestamp": self.timestamp
        }


@dataclass
class CodebaseMetrics:
    """Metrics about the codebase structure for calibration."""
    
    total_files: int
    total_lines: int
    module_count: int
    avg_file_size: float
    complexity_score: float
    interaction_density: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "total_files": self.total_files,
            "total_lines": self.total_lines,
            "module_count": self.module_count,
            "avg_file_size": self.avg_file_size,
            "complexity_score": self.complexity_score,
            "interaction_density": self.interaction_density
        }


@dataclass
class CalibrationResult:
    """Results from parameter calibration."""
    
    alpha: float
    weights: Dict[str, float]
    epsilon: float
    kappa: float
    correlation_score: float
    mse: float
    iterations: int
    converged: bool
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return {
            "alpha": self.alpha,
            "weights": self.weights,
            "epsilon": self.epsilon,
            "kappa": self.kappa,
            "correlation_score": self.correlation_score,
            "mse": self.mse,
            "iterations": self.iterations,
            "converged": self.converged
        }


class CalibrationTool:
    """
    Tool for calibrating polymer physics parameters.
    
    Uses observed thrashing patterns to tune the parameters for
    accurate contact probability predictions.
    """
    
    def __init__(self):
        self.console = Console()
        self.default_weights = {
            "tok": 1.0,
            "ast": 2.0,
            "mod": 3.0,
            "git": 4.0
        }
        
    def fit_parameters(
        self,
        events: List[ThrashEvent],
        codebase: CodebaseMetrics,
        initial_alpha: float = 1.2,
        initial_epsilon: float = 1e-6,
        initial_kappa: float = 1.0,
        max_iterations: int = 100
    ) -> CalibrationResult:
        """
        Fit polymer physics parameters to observed thrashing events.
        
        Args:
            events: List of observed thrashing events
            codebase: Codebase metrics
            initial_alpha: Initial alpha value
            initial_epsilon: Initial epsilon value
            initial_kappa: Initial kappa value
            max_iterations: Maximum optimization iterations
            
        Returns:
            Calibration result with fitted parameters
        """
        if not events:
            self.console.print("[yellow]Warning: No events provided for calibration[/yellow]")
            return self._default_result()
        
        # Prepare data
        X, y = self._prepare_calibration_data(events, codebase)
        
        if len(X) < 5:
            self.console.print("[yellow]Warning: Insufficient data for calibration[/yellow]")
            return self._default_result()
        
        # Initial parameters
        initial_params = np.array([
            initial_alpha,
            initial_epsilon,
            initial_kappa,
            *[self.default_weights[k] for k in sorted(self.default_weights.keys())]
        ])
        
        # Bounds for parameters
        bounds = [
            (0.5, 3.0),    # alpha
            (1e-8, 1.0),   # epsilon
            (0.1, 10.0),   # kappa
            (0.1, 10.0),   # weight_tok
            (0.1, 10.0),   # weight_ast
            (0.1, 10.0),   # weight_mod
            (0.1, 10.0),   # weight_git
        ]
        
        # Optimize
        result = minimize(
            lambda params: self._objective_function(params, X, y),
            initial_params,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iterations}
        )
        
        # Extract results
        alpha = result.x[0]
        epsilon = result.x[1]
        kappa = result.x[2]
        weights = {
            "tok": result.x[3],
            "ast": result.x[4],
            "mod": result.x[5],
            "git": result.x[6]
        }
        
        # Calculate correlation
        y_pred = self._predict_contacts(result.x, X)
        correlation, _ = pearsonr(y, y_pred)
        mse = np.mean((y - y_pred) ** 2)
        
        return CalibrationResult(
            alpha=alpha,
            weights=weights,
            epsilon=epsilon,
            kappa=kappa,
            correlation_score=correlation,
            mse=mse,
            iterations=result.nit,
            converged=result.success
        )
    
    def grid_search(
        self,
        events: List[ThrashEvent],
        codebase: CodebaseMetrics,
        alpha_range: Tuple[float, float] = (0.8, 2.0),
        n_grid_points: int = 5
    ) -> CalibrationResult:
        """
        Perform grid search for optimal parameters.
        
        Args:
            events: Thrashing events
            codebase: Codebase metrics
            alpha_range: Range for alpha parameter
            n_grid_points: Number of grid points per dimension
            
        Returns:
            Best calibration result
        """
        best_result = None
        best_score = float('-inf')
        
        alphas = np.linspace(alpha_range[0], alpha_range[1], n_grid_points)
        epsilons = np.logspace(-8, -3, n_grid_points)
        kappas = np.linspace(0.5, 2.0, n_grid_points)
        
        total_combinations = n_grid_points ** 3
        
        self.console.print(f"[cyan]Running grid search with {total_combinations} combinations...[/cyan]")
        
        for alpha in track(alphas, description="Grid search"):
            for epsilon in epsilons:
                for kappa in kappas:
                    result = self.fit_parameters(
                        events, codebase,
                        initial_alpha=alpha,
                        initial_epsilon=epsilon,
                        initial_kappa=kappa,
                        max_iterations=20
                    )
                    
                    score = result.correlation_score - result.mse
                    if score > best_score:
                        best_score = score
                        best_result = result
        
        return best_result
    
    def validate_parameters(
        self,
        result: CalibrationResult,
        test_events: List[ThrashEvent]
    ) -> Dict[str, float]:
        """
        Validate calibrated parameters on test data.
        
        Args:
            result: Calibration result
            test_events: Test events for validation
            
        Returns:
            Validation metrics
        """
        if not test_events:
            return {"error": "No test events provided"}
        
        # Prepare test data
        X_test, y_test = self._prepare_calibration_data(test_events, None)
        
        # Predict using calibrated parameters
        params = np.array([
            result.alpha,
            result.epsilon,
            result.kappa,
            *[result.weights[k] for k in sorted(result.weights.keys())]
        ])
        
        y_pred = self._predict_contacts(params, X_test)
        
        # Calculate metrics
        correlation, _ = pearsonr(y_test, y_pred)
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        return {
            "correlation": correlation,
            "mse": mse,
            "mae": mae,
            "r_squared": correlation ** 2
        }
    
    def generate_demo_data(
        self,
        n_events: int = 100,
        seed: int = 42
    ) -> Tuple[List[ThrashEvent], CodebaseMetrics]:
        """
        Generate demo data for testing calibration.
        
        Args:
            n_events: Number of events to generate
            seed: Random seed
            
        Returns:
            Demo events and codebase metrics
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Generate codebase metrics
        codebase = CodebaseMetrics(
            total_files=random.randint(50, 500),
            total_lines=random.randint(5000, 50000),
            module_count=random.randint(5, 20),
            avg_file_size=random.uniform(100, 500),
            complexity_score=random.uniform(0.3, 0.8),
            interaction_density=random.uniform(0.1, 0.5)
        )
        
        # Generate events
        events = []
        distance_types = ["tok", "ast", "mod", "git"]
        
        for i in range(n_events):
            event = ThrashEvent(
                file1=f"file_{random.randint(1, codebase.total_files)}.py",
                file2=f"file_{random.randint(1, codebase.total_files)}.py",
                distance_type=random.choice(distance_types),
                observed_frequency=random.uniform(0, 1),
                latency_ms=random.uniform(10, 1000),
                timestamp=i * random.uniform(0.1, 1.0)
            )
            events.append(event)
        
        return events, codebase
    
    def display_results(self, result: CalibrationResult):
        """
        Display calibration results in a formatted table.
        
        Args:
            result: Calibration result to display
        """
        table = Table(title="Calibration Results")
        
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="magenta")
        
        table.add_row("Alpha (α)", f"{result.alpha:.3f}")
        table.add_row("Epsilon (ε)", f"{result.epsilon:.2e}")
        table.add_row("Kappa (κ)", f"{result.kappa:.3f}")
        
        for name, weight in sorted(result.weights.items()):
            table.add_row(f"Weight ({name})", f"{weight:.3f}")
        
        table.add_row("Correlation", f"{result.correlation_score:.3f}")
        table.add_row("MSE", f"{result.mse:.3f}")
        table.add_row("Iterations", str(result.iterations))
        table.add_row("Converged", "✓" if result.converged else "✗")
        
        self.console.print(table)
    
    def _prepare_calibration_data(
        self,
        events: List[ThrashEvent],
        codebase: Optional[CodebaseMetrics]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for calibration."""
        X = []
        y = []
        
        for event in events:
            # Features: distance metrics
            features = [
                1.0 if event.distance_type == "tok" else 0.0,
                1.0 if event.distance_type == "ast" else 0.0,
                1.0 if event.distance_type == "mod" else 0.0,
                1.0 if event.distance_type == "git" else 0.0,
                event.latency_ms / 1000.0  # Normalize latency
            ]
            
            X.append(features)
            y.append(event.observed_frequency)
        
        return np.array(X), np.array(y)
    
    def _objective_function(
        self,
        params: np.ndarray,
        X: np.ndarray,
        y: np.ndarray
    ) -> float:
        """Objective function for optimization."""
        y_pred = self._predict_contacts(params, X)
        mse = np.mean((y - y_pred) ** 2)
        
        # Add regularization
        reg = 0.01 * np.sum(params[3:] ** 2)  # Regularize weights
        
        return mse + reg
    
    def _predict_contacts(
        self,
        params: np.ndarray,
        X: np.ndarray
    ) -> np.ndarray:
        """Predict contact probabilities."""
        alpha = params[0]
        epsilon = params[1]
        kappa = params[2]
        weights = params[3:7]
        
        # Calculate weighted distances
        distances = X[:, :4] @ weights
        
        # Apply polymer physics model
        # P(d) = kappa * (d + epsilon)^(-alpha)
        probabilities = kappa * (distances + epsilon) ** (-alpha)
        
        # Clamp to [0, 1]
        return np.clip(probabilities, 0, 1)
    
    def _default_result(self) -> CalibrationResult:
        """Return default calibration result."""
        return CalibrationResult(
            alpha=1.2,
            weights=self.default_weights,
            epsilon=1e-6,
            kappa=1.0,
            correlation_score=0.0,
            mse=1.0,
            iterations=0,
            converged=False
        )


def run_calibration_demo():
    """Run a demo calibration with synthetic data."""
    console = Console()
    console.print("[bold cyan]Running Calibration Demo[/bold cyan]\n")
    
    # Create calibration tool
    tool = CalibrationTool()
    
    # Generate demo data
    console.print("[yellow]Generating demo data...[/yellow]")
    events, codebase = tool.generate_demo_data(n_events=100)
    
    # Split into train/test
    split_idx = int(len(events) * 0.8)
    train_events = events[:split_idx]
    test_events = events[split_idx:]
    
    console.print(f"Generated {len(train_events)} training events and {len(test_events)} test events\n")
    
    # Fit parameters
    console.print("[yellow]Fitting parameters...[/yellow]")
    result = tool.fit_parameters(train_events, codebase)
    
    # Display results
    tool.display_results(result)
    
    # Validate
    console.print("\n[yellow]Validating on test data...[/yellow]")
    validation_metrics = tool.validate_parameters(result, test_events)
    
    val_table = Table(title="Validation Metrics")
    val_table.add_column("Metric", style="cyan")
    val_table.add_column("Value", style="magenta")
    
    for metric, value in validation_metrics.items():
        val_table.add_row(metric.replace("_", " ").title(), f"{value:.3f}")
    
    console.print(val_table)
    
    return result


if __name__ == "__main__":
    run_calibration_demo()