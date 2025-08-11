"""
Benchmark scenarios for testing tail-chasing pattern detection and fixing.

Each scenario represents a specific type of problem with expected resolution steps.
"""

from .base import BenchmarkScenario, ScenarioResult
from .simple_import import SimpleImportScenario
from .circular_dependency import CircularDependencyScenario
from .semantic_duplicate import SemanticDuplicateScenario
from .complex_refactoring import ComplexRefactoringScenario

__all__ = [
    'BenchmarkScenario',
    'ScenarioResult',
    'SimpleImportScenario',
    'CircularDependencyScenario',
    'SemanticDuplicateScenario',
    'ComplexRefactoringScenario',
]