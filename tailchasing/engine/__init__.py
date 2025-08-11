"""
Tail-chasing fix engine - orchestrates convergent fix application.

Prevents infinite loops and ensures convergence to stable state.
"""

from .convergence import (
    ConvergenceTracker,
    PatchValidator, 
    FixOrchestrator,
    ConvergenceError,
    PatchValidationError,
    OrchestrationError
)

__all__ = [
    "ConvergenceTracker",
    "PatchValidator",
    "FixOrchestrator", 
    "ConvergenceError",
    "PatchValidationError",
    "OrchestrationError"
]