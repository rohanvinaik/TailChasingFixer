"""
Comprehensive evaluation harness for TailChasing Fixer.

This module provides benchmarking capabilities to evaluate:
- Convergence performance across different problem types
- Token usage and cost tracking
- Success/failure rates
- Regression detection
- Multi-model comparison
"""

from .scenarios import (
    SimpleImportScenario,
    CircularDependencyScenario,
    SemanticDuplicateScenario,
    ComplexRefactoringScenario,
)

from .metrics import (
    MetricsCollector,
    ConvergenceMetrics,
    PerformanceMetrics,
    CostMetrics,
)

from .runners import (
    BenchmarkRunner,
    MultiModelRunner,
    PerformanceTracker,
)

__all__ = [
    # Scenarios
    'SimpleImportScenario',
    'CircularDependencyScenario',
    'SemanticDuplicateScenario',
    'ComplexRefactoringScenario',
    
    # Metrics
    'MetricsCollector',
    'ConvergenceMetrics',
    'PerformanceMetrics',
    'CostMetrics',
    
    # Runners
    'BenchmarkRunner',
    'MultiModelRunner',
    'PerformanceTracker',
]