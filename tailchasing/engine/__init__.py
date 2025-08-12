"""
Engine module for convergence control and fix orchestration.

Provides a facade for backward compatibility while using the refactored modules.
Prevents infinite loops and ensures convergence to stable state.
"""

# Try to import from refactored modules first, fall back to convergence.py if needed
try:
    # Import from refactored modules
    from .errors import (
        EngineError,
        ConvergenceError,
        ValidationError,
        OrchestrationError,
        # Backward compatibility helpers
        is_iteration_limit_error,
        is_loop_detected_error,
        is_syntax_error,
        is_risk_error,
        is_test_failure,
        is_rollback_error
    )

    from .state import (
        RiskLevel,
        IterationState,
        PatchInfo,
        ConvergenceTracker,
        StateTrackerProtocol
    )

    from .validation import (
        SyntaxValidator,
        RiskAssessor,
        PatchValidator,
        ValidatorProtocol
    )

    from .orchestration import (
        FixPlan,
        ExecutionResult,
        FixOrchestrator,
        SimpleBackupManager,
        FixProviderProtocol,
        BackupManagerProtocol
    )
    
    # Backward compatibility aliases
    PatchValidationError = ValidationError

except ImportError:
    # Fall back to original convergence.py if refactored modules don't exist
    from .convergence import (
        ConvergenceTracker,
        PatchValidator, 
        FixOrchestrator,
        ConvergenceError,
        PatchValidationError,
        OrchestrationError
    )
    
    # Create placeholder classes for new types
    class EngineError(Exception):
        pass
    
    ValidationError = PatchValidationError
    RiskLevel = None
    IterationState = None
    PatchInfo = None
    FixPlan = None
    ExecutionResult = None
    SimpleBackupManager = None


# Factory functions for backward compatibility
def create_convergence_system(max_iterations: int = 8,
                             max_risk: str = 'high',
                             test_command: str = None):
    """
    Create a convergence system (orchestrator).
    
    Backward compatibility wrapper for FixOrchestrator.
    
    Args:
        max_iterations: Maximum iterations
        max_risk: Risk level ('low', 'medium', 'high', 'critical')
        test_command: Test command to run
        
    Returns:
        FixOrchestrator instance
    """
    try:
        risk_map = {
            'low': RiskLevel.LOW,
            'medium': RiskLevel.MEDIUM,
            'high': RiskLevel.HIGH,
            'critical': RiskLevel.CRITICAL
        }
        
        risk_level = risk_map.get(max_risk, RiskLevel.HIGH)
        
        return FixOrchestrator(
            max_iterations=max_iterations,
            max_risk=risk_level,
            test_command=test_command
        )
    except:
        # Fall back to old API if needed
        return FixOrchestrator()


def create_patch_validator(max_risk: str = 'high',
                          test_command: str = None):
    """
    Create a patch validator.
    
    Backward compatibility wrapper.
    
    Args:
        max_risk: Maximum risk level
        test_command: Test command
        
    Returns:
        PatchValidator instance
    """
    try:
        risk_map = {
            'low': RiskLevel.LOW,
            'medium': RiskLevel.MEDIUM,
            'high': RiskLevel.HIGH,
            'critical': RiskLevel.CRITICAL
        }
        
        risk_level = risk_map.get(max_risk, RiskLevel.HIGH)
        
        return PatchValidator(max_risk=risk_level, test_command=test_command)
    except:
        # Fall back to old API
        return PatchValidator()


# Maintain backward compatibility with original exports
__all__ = [
    # Original exports
    "ConvergenceTracker",
    "PatchValidator",
    "FixOrchestrator", 
    "ConvergenceError",
    "PatchValidationError",
    "OrchestrationError",
    
    # New exports (if available)
    'EngineError',
    'ValidationError',
    'RiskLevel',
    'IterationState',
    'PatchInfo',
    'FixPlan',
    'ExecutionResult',
    'SimpleBackupManager',
    
    # Factory functions
    'create_convergence_system',
    'create_patch_validator',
]