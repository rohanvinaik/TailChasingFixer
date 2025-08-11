"""
Convergence metrics tracking steps to solution.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class ConvergenceMetrics:
    """Metrics related to convergence behavior."""
    
    # Step tracking
    total_steps: int = 0
    successful_steps: int = 0
    failed_steps: int = 0
    
    # Expected vs actual
    expected_steps_min: int = 0
    expected_steps_max: int = 0
    
    # Convergence pattern
    step_history: List[Tuple[int, bool, Optional[str]]] = field(default_factory=list)
    
    # Regression tracking
    regressions: List[str] = field(default_factory=list)
    regression_count: int = 0
    
    # Efficiency metrics
    backtrack_count: int = 0
    retry_count: int = 0
    
    # Final state
    converged: bool = False
    convergence_quality: str = "unknown"  # perfect, good, acceptable, poor
    
    def start_tracking(self, expected_min: int = 0, expected_max: int = 0):
        """Start tracking convergence metrics."""
        self.expected_steps_min = expected_min
        self.expected_steps_max = expected_max
        self.step_history = []
        self.regressions = []
    
    def record_step(self, step_number: int, success: bool, error: Optional[str] = None):
        """Record a convergence step."""
        self.total_steps += 1
        
        if success:
            self.successful_steps += 1
        else:
            self.failed_steps += 1
            if error and "retry" in error.lower():
                self.retry_count += 1
        
        self.step_history.append((step_number, success, error))
        
        # Detect backtracking
        if len(self.step_history) > 1:
            prev_step = self.step_history[-2][0]
            if step_number <= prev_step:
                self.backtrack_count += 1
    
    def add_regression(self, regression: str):
        """Add a detected regression."""
        self.regressions.append(regression)
        self.regression_count += 1
    
    def finish_tracking(self, converged: bool):
        """Finish tracking and calculate final metrics."""
        self.converged = converged
        
        if not converged:
            self.convergence_quality = "failed"
        elif self.regression_count > 0:
            self.convergence_quality = "poor"
        elif self.total_steps <= self.expected_steps_min:
            self.convergence_quality = "perfect"
        elif self.total_steps <= self.expected_steps_max:
            self.convergence_quality = "good"
        elif self.total_steps <= self.expected_steps_max * 1.5:
            self.convergence_quality = "acceptable"
        else:
            self.convergence_quality = "poor"
    
    def get_efficiency_score(self) -> float:
        """Calculate efficiency score (0-100)."""
        if not self.converged:
            return 0.0
        
        base_score = 100.0
        
        # Penalize for exceeding expected steps
        if self.expected_steps_max > 0:
            if self.total_steps > self.expected_steps_max:
                excess_ratio = (self.total_steps - self.expected_steps_max) / self.expected_steps_max
                base_score -= min(50, excess_ratio * 50)
        
        # Penalize for regressions
        base_score -= self.regression_count * 10
        
        # Penalize for backtracks and retries
        base_score -= self.backtrack_count * 5
        base_score -= self.retry_count * 3
        
        # Penalize for failed steps
        if self.total_steps > 0:
            failure_ratio = self.failed_steps / self.total_steps
            base_score -= failure_ratio * 20
        
        return max(0.0, base_score)
    
    def get_convergence_rate(self) -> float:
        """Calculate convergence rate (successful steps / total steps)."""
        if self.total_steps == 0:
            return 0.0
        return self.successful_steps / self.total_steps
    
    def get_step_efficiency(self) -> float:
        """Calculate step efficiency compared to expected."""
        if self.expected_steps_min == 0:
            return 1.0
        
        if self.total_steps <= self.expected_steps_min:
            return 1.0
        elif self.total_steps <= self.expected_steps_max:
            # Linear interpolation
            range_size = self.expected_steps_max - self.expected_steps_min
            excess = self.total_steps - self.expected_steps_min
            return 1.0 - (excess / range_size * 0.3)  # Max 30% penalty
        else:
            # Over expected maximum
            excess_ratio = self.total_steps / self.expected_steps_max
            return max(0.1, 1.0 / excess_ratio)