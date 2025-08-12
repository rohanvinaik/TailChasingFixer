"""
Calibration tools for polymer physics parameters.

This module provides tools to calibrate polymer physics parameters based on
historical thrashing events and codebase structure, using grid search and
correlation maximization to find optimal parameter values.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats
from scipy.optimize import minimize

from .config import CalibrationResult, DistanceWeights, PolymerConfig


# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class ThrashEvent:
    """
    Represents a historical thrashing event for calibration.
    """

    file_path: str
    line_number: int
    timestamp: float
    severity: float  # 0.0 to 1.0
    frequency: int
    latency_impact: float
    related_files: List[str] = None

    def __post_init__(self):
        if self.related_files is None:
            self.related_files = []


@dataclass
class CodebaseMetrics:
    """
    Codebase structure metrics for calibration.
    """

    total_files: int
    total_lines: int
    package_structure: Dict[str, int]  # package -> file count
    dependency_graph: Dict[str, List[str]]  # file -> dependencies
    ast_complexity: Dict[str, float]  # file -> complexity score
    git_activity: Dict[str, float]  # file -> activity score


class CalibrationTool:
    """
    Tool for calibrating polymer physics parameters based on historical data.

    Uses grid search for alpha parameter optimization and correlation maximization
    for distance weight optimization to find parameters that best predict
    historical thrashing events.
    """

    def __init__(self, config: Optional[PolymerConfig] = None):
        self.config = config or PolymerConfig()
        self.logger = logging.getLogger(__name__)

    def fit_parameters(
        self,
        historical_thrash_events: List[ThrashEvent],
        codebase: CodebaseMetrics,
        alpha_range: Tuple[float, float] = (1.0, 1.5),
        alpha_steps: int = 26,
        max_iterations: int = 100,
        tolerance: float = 1e-6,
    ) -> CalibrationResult:
        """
        Fit optimal parameters using historical thrashing events.

        Args:
            historical_thrash_events: List of historical thrash events
            codebase: Codebase structure metrics
            alpha_range: Range for alpha parameter grid search
            alpha_steps: Number of alpha values to test
            max_iterations: Maximum optimization iterations
            tolerance: Convergence tolerance

        Returns:
            CalibrationResult with optimal parameters
        """
        if not historical_thrash_events:
            raise ValueError(
                "Need at least one historical thrash event for calibration"
            )

        self.logger.info(
            f"Starting parameter calibration with {len(historical_thrash_events)} events"
        )

        # Prepare feature matrix and target vector
        X, y = self._prepare_training_data(historical_thrash_events, codebase)

        if X.shape[0] == 0:
            raise ValueError("No valid training data could be prepared")

        # Grid search for optimal alpha
        self.logger.info(f"Grid searching alpha parameter in range {alpha_range}")
        optimal_alpha, best_alpha_score = self._grid_search_alpha(
            X, y, alpha_range, alpha_steps
        )

        self.logger.info(
            f"Optimal alpha: {optimal_alpha:.4f} (score: {best_alpha_score:.4f})"
        )

        # Optimize weights using the optimal alpha
        self.logger.info("Optimizing distance weights via correlation maximization")
        optimal_weights, weight_score, iterations, converged = self._optimize_weights(
            X, y, optimal_alpha, max_iterations, tolerance
        )

        self.logger.info(
            f"Weight optimization: score={weight_score:.4f}, iterations={iterations}, converged={converged}"
        )

        # Validate on held-out data
        validation_score = self._cross_validate(X, y, optimal_alpha, optimal_weights)

        result = CalibrationResult(
            optimal_alpha=optimal_alpha,
            optimal_weights=optimal_weights,
            correlation_score=weight_score,
            validation_score=validation_score,
            iterations=iterations,
            converged=converged,
        )

        self.logger.info(
            f"Calibration complete. Validation score: {validation_score:.4f}"
        )

        return result

    def validate_predictions(
        self,
        test_set: List[ThrashEvent],
        codebase: CodebaseMetrics,
        config: Optional[PolymerConfig] = None,
    ) -> Dict[str, float]:
        """
        Validate predictions against a test set.

        Args:
            test_set: Test set of thrash events
            codebase: Codebase metrics
            config: Configuration to test (default: current config)

        Returns:
            Dictionary with validation metrics
        """
        test_config = config or self.config

        if not test_set:
            return {"error": "Empty test set"}

        # Prepare test data
        X_test, y_test = self._prepare_training_data(test_set, codebase)

        if X_test.shape[0] == 0:
            return {"error": "No valid test data"}

        # Make predictions
        predictions = self._predict_thrash_probability(
            X_test, test_config.alpha, test_config.weights
        )

        # Calculate metrics
        correlation, p_value = stats.pearsonr(y_test, predictions)
        spearman_corr, _ = stats.spearmanr(y_test, predictions)

        # Calculate RMSE
        rmse = np.sqrt(np.mean((y_test - predictions) ** 2))

        # Calculate R-squared
        ss_res = np.sum((y_test - predictions) ** 2)
        ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        return {
            "pearson_correlation": correlation,
            "pearson_p_value": p_value,
            "spearman_correlation": spearman_corr,
            "rmse": rmse,
            "r_squared": r_squared,
            "sample_size": len(test_set),
        }

    def suggest_config_updates(
        self,
        calibration_result: CalibrationResult,
        current_config: Optional[PolymerConfig] = None,
    ) -> Dict[str, Union[str, float, Dict]]:
        """
        Suggest configuration updates based on calibration results.

        Args:
            calibration_result: Results from parameter calibration
            current_config: Current configuration (default: self.config)

        Returns:
            Dictionary with suggested updates and explanations
        """
        current = current_config or self.config

        suggestions = {
            "summary": "",
            "alpha_change": {},
            "weight_changes": {},
            "expected_improvement": calibration_result.correlation_score,
            "confidence": "high" if calibration_result.converged else "medium",
        }

        # Alpha suggestions
        alpha_diff = calibration_result.optimal_alpha - current.alpha
        if abs(alpha_diff) > 0.01:
            suggestions["alpha_change"] = {
                "from": current.alpha,
                "to": calibration_result.optimal_alpha,
                "change": alpha_diff,
                "explanation": self._explain_alpha_change(alpha_diff),
            }

        # Weight suggestions
        current_weights = current.weights
        optimal_weights = calibration_result.optimal_weights

        weight_changes = {}
        for weight_type in ["tok", "ast", "mod", "git"]:
            current_val = getattr(current_weights, weight_type)
            optimal_val = getattr(optimal_weights, weight_type)

            if abs(optimal_val - current_val) > 0.1:
                weight_changes[weight_type] = {
                    "from": current_val,
                    "to": optimal_val,
                    "change": optimal_val - current_val,
                    "explanation": self._explain_weight_change(
                        weight_type, optimal_val - current_val
                    ),
                }

        suggestions["weight_changes"] = weight_changes

        # Generate summary
        summary_parts = []
        if suggestions["alpha_change"]:
            summary_parts.append(
                f"Adjust alpha from {current.alpha:.3f} to {calibration_result.optimal_alpha:.3f}"
            )

        if weight_changes:
            summary_parts.append(f"Update {len(weight_changes)} distance weights")

        if not summary_parts:
            summary_parts.append("Current configuration is already well-tuned")

        suggestions["summary"] = "; ".join(summary_parts)

        return suggestions

    def _prepare_training_data(
        self, events: List[ThrashEvent], codebase: CodebaseMetrics
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare feature matrix and target vector from events."""
        features = []
        targets = []

        for event in events:
            # Extract features for this event
            feature_vector = self._extract_features(event, codebase)
            if feature_vector is not None:
                features.append(feature_vector)
                targets.append(event.severity * event.frequency * event.latency_impact)

        if not features:
            return np.array([]), np.array([])

        return np.array(features), np.array(targets)

    def _extract_features(
        self, event: ThrashEvent, codebase: CodebaseMetrics
    ) -> Optional[np.ndarray]:
        """Extract feature vector for a thrash event."""
        try:
            # Distance features (tok, ast, mod, git levels)
            tok_distance = self._calculate_token_distance(event, codebase)
            ast_distance = self._calculate_ast_distance(event, codebase)
            mod_distance = self._calculate_module_distance(event, codebase)
            git_distance = self._calculate_git_distance(event, codebase)

            # Structural features
            file_complexity = codebase.ast_complexity.get(event.file_path, 1.0)
            package_size = self._get_package_size(event.file_path, codebase)
            dependency_count = len(codebase.dependency_graph.get(event.file_path, []))

            return np.array(
                [
                    tok_distance,
                    ast_distance,
                    mod_distance,
                    git_distance,
                    file_complexity,
                    package_size,
                    dependency_count,
                ]
            )

        except Exception as e:
            self.logger.warning(
                f"Failed to extract features for event {event.file_path}:{event.line_number}: {e}"
            )
            return None

    def _calculate_token_distance(
        self, event: ThrashEvent, codebase: CodebaseMetrics
    ) -> float:
        """Calculate token-level distance metric."""
        # Simple line-based distance within file
        # In practice, this would use AST analysis
        return float(event.line_number) / 1000.0  # Normalize by typical file size

    def _calculate_ast_distance(
        self, event: ThrashEvent, codebase: CodebaseMetrics
    ) -> float:
        """Calculate AST-level distance metric."""
        complexity = codebase.ast_complexity.get(event.file_path, 1.0)
        # Higher complexity = more AST distance
        return min(complexity / 10.0, 5.0)  # Cap at reasonable level

    def _calculate_module_distance(
        self, event: ThrashEvent, codebase: CodebaseMetrics
    ) -> float:
        """Calculate module-level distance metric."""
        package_name = self._get_package_name(event.file_path)
        package_size = codebase.package_structure.get(package_name, 1)
        # Larger packages = more module distance
        return min(np.log(package_size + 1), 5.0)

    def _calculate_git_distance(
        self, event: ThrashEvent, codebase: CodebaseMetrics
    ) -> float:
        """Calculate git-level distance metric."""
        activity = codebase.git_activity.get(event.file_path, 0.1)
        # Higher activity = closer git distance
        return max(1.0 / activity, 0.1)

    def _get_package_name(self, file_path: str) -> str:
        """Extract package name from file path."""
        path = Path(file_path)
        if len(path.parts) > 1:
            return path.parts[0]  # First directory
        return "root"

    def _get_package_size(self, file_path: str, codebase: CodebaseMetrics) -> float:
        """Get package size for file."""
        package_name = self._get_package_name(file_path)
        return float(codebase.package_structure.get(package_name, 1))

    def _grid_search_alpha(
        self, X: np.ndarray, y: np.ndarray, alpha_range: Tuple[float, float], steps: int
    ) -> Tuple[float, float]:
        """Grid search for optimal alpha parameter."""
        alphas = np.linspace(alpha_range[0], alpha_range[1], steps)
        best_alpha = alphas[0]
        best_score = -np.inf

        # Use default weights for alpha search
        default_weights = DistanceWeights()

        for alpha in alphas:
            try:
                predictions = self._predict_thrash_probability(
                    X, alpha, default_weights
                )
                score, _ = stats.pearsonr(y, predictions)

                if np.isfinite(score) and score > best_score:
                    best_score = score
                    best_alpha = alpha

            except Exception as e:
                self.logger.warning(f"Failed to evaluate alpha={alpha:.4f}: {e}")
                continue

        return best_alpha, best_score

    def _optimize_weights(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        max_iterations: int,
        tolerance: float,
    ) -> Tuple[DistanceWeights, float, int, bool]:
        """Optimize distance weights via correlation maximization."""

        def objective(weight_vector):
            """Objective function: negative correlation."""
            weights = DistanceWeights(
                tok=weight_vector[0],
                ast=weight_vector[1],
                mod=weight_vector[2],
                git=weight_vector[3],
            )

            try:
                predictions = self._predict_thrash_probability(X, alpha, weights)
                correlation, _ = stats.pearsonr(y, predictions)

                if np.isfinite(correlation):
                    return (
                        -correlation
                    )  # Minimize negative correlation = maximize correlation
                else:
                    return 1.0  # Bad score for invalid correlation

            except Exception:
                return 1.0  # Bad score for errors

        # Initial guess (current weights)
        x0 = [
            self.config.weights.tok,
            self.config.weights.ast,
            self.config.weights.mod,
            self.config.weights.git,
        ]

        # Bounds: weights should be positive and reasonable
        bounds = [(0.1, 10.0) for _ in range(4)]

        # Optimize
        result = minimize(
            objective,
            x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iterations, "ftol": tolerance},
        )

        optimal_weights = DistanceWeights(
            tok=result.x[0], ast=result.x[1], mod=result.x[2], git=result.x[3]
        )

        return optimal_weights, -result.fun, result.nit, result.success

    def _predict_thrash_probability(
        self, X: np.ndarray, alpha: float, weights: DistanceWeights
    ) -> np.ndarray:
        """Predict thrash probability using polymer physics model."""
        if X.shape[0] == 0:
            return np.array([])

        # Extract distance components
        tok_distances = X[:, 0]
        ast_distances = X[:, 1]
        mod_distances = X[:, 2]
        git_distances = X[:, 3]

        # Calculate weighted polymer distance
        weighted_distance = (
            weights.tok * tok_distances
            + weights.ast * ast_distances
            + weights.mod * mod_distances
            + weights.git * git_distances
        )

        # Apply power law decay: P(d) = d^(-alpha)
        # Add epsilon for numerical stability
        probabilities = np.power(weighted_distance + self.config.epsilon, -alpha)

        # Normalize to [0, 1] range
        if probabilities.max() > probabilities.min():
            probabilities = (probabilities - probabilities.min()) / (
                probabilities.max() - probabilities.min()
            )

        return probabilities

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        alpha: float,
        weights: DistanceWeights,
        k_folds: int = 5,
    ) -> float:
        """Cross-validate the calibrated parameters."""
        if X.shape[0] < k_folds:
            # Not enough data for k-fold, use leave-one-out
            return self._leave_one_out_validate(X, y, alpha, weights)

        n_samples = X.shape[0]
        fold_size = n_samples // k_folds
        scores = []

        for i in range(k_folds):
            # Split data
            start_idx = i * fold_size
            end_idx = start_idx + fold_size if i < k_folds - 1 else n_samples

            test_indices = list(range(start_idx, end_idx))

            X_test = X[test_indices]
            y_test = y[test_indices]

            # Make predictions
            predictions = self._predict_thrash_probability(X_test, alpha, weights)

            # Calculate correlation
            if len(predictions) > 1 and np.std(predictions) > 0 and np.std(y_test) > 0:
                corr, _ = stats.pearsonr(y_test, predictions)
                if np.isfinite(corr):
                    scores.append(corr)

        return np.mean(scores) if scores else 0.0

    def _leave_one_out_validate(
        self, X: np.ndarray, y: np.ndarray, alpha: float, weights: DistanceWeights
    ) -> float:
        """Leave-one-out cross-validation for small datasets."""
        predictions = []
        actuals = []

        for i in range(X.shape[0]):
            # Leave one out
            X_test = X[i : i + 1]
            y_test = y[i]

            # Predict (using the given parameters)
            pred = self._predict_thrash_probability(X_test, alpha, weights)

            if len(pred) > 0:
                predictions.append(pred[0])
                actuals.append(y_test)

        if len(predictions) > 1:
            corr, _ = stats.pearsonr(actuals, predictions)
            return corr if np.isfinite(corr) else 0.0

        return 0.0

    def _explain_alpha_change(self, alpha_diff: float) -> str:
        """Explain what an alpha change means."""
        if alpha_diff > 0.1:
            return "Increase contact decay (stronger distance penalty)"
        elif alpha_diff < -0.1:
            return "Decrease contact decay (weaker distance penalty)"
        else:
            return "Minor adjustment to contact decay"

    def _explain_weight_change(self, weight_type: str, change: float) -> str:
        """Explain what a weight change means."""
        weight_names = {
            "tok": "token-level",
            "ast": "AST-level",
            "mod": "module-level",
            "git": "git-level",
        }

        name = weight_names.get(weight_type, weight_type)

        if change > 0.5:
            return f"Increase {name} distance importance significantly"
        elif change > 0.1:
            return f"Increase {name} distance importance"
        elif change < -0.5:
            return f"Decrease {name} distance importance significantly"
        elif change < -0.1:
            return f"Decrease {name} distance importance"
        else:
            return f"Minor adjustment to {name} distance importance"


def create_sample_thrash_events(
    file_paths: List[str], event_count: int = 50, random_seed: int = 42
) -> List[ThrashEvent]:
    """
    Create sample thrash events for testing/demo purposes.

    Args:
        file_paths: List of file paths in the codebase
        event_count: Number of events to generate
        random_seed: Random seed for reproducibility

    Returns:
        List of sample thrash events
    """
    np.random.seed(random_seed)

    events = []

    for i in range(event_count):
        file_path = np.random.choice(file_paths)

        event = ThrashEvent(
            file_path=file_path,
            line_number=np.random.randint(1, 500),
            timestamp=float(i * 1000 + np.random.randint(0, 1000)),
            severity=np.random.beta(2, 5),  # Biased toward lower severity
            frequency=np.random.poisson(3) + 1,
            latency_impact=np.random.exponential(50.0),
            related_files=np.random.choice(
                file_paths, size=np.random.randint(0, 3), replace=False
            ).tolist(),
        )

        events.append(event)

    return events


def create_sample_codebase_metrics(file_paths: List[str]) -> CodebaseMetrics:
    """
    Create sample codebase metrics for testing/demo purposes.

    Args:
        file_paths: List of file paths in the codebase

    Returns:
        Sample codebase metrics
    """
    np.random.seed(42)

    # Extract package structure
    packages = {}
    for file_path in file_paths:
        package = Path(file_path).parts[0] if Path(file_path).parts else "root"
        packages[package] = packages.get(package, 0) + 1

    # Create dependency graph (simplified)
    dependency_graph = {}
    for file_path in file_paths:
        dep_count = np.random.poisson(3)
        deps = np.random.choice(
            file_paths, size=min(dep_count, len(file_paths) - 1), replace=False
        )
        dependency_graph[file_path] = deps.tolist()

    # Create complexity and activity metrics
    ast_complexity = {fp: np.random.gamma(2, 2) for fp in file_paths}
    git_activity = {fp: np.random.exponential(0.5) for fp in file_paths}

    return CodebaseMetrics(
        total_files=len(file_paths),
        total_lines=sum(np.random.randint(50, 1000) for _ in file_paths),
        package_structure=packages,
        dependency_graph=dependency_graph,
        ast_complexity=ast_complexity,
        git_activity=git_activity,
    )