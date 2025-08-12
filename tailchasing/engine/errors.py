"""
Consolidated error types for the engine module.

Reduces 9 exception classes to 4 essential ones, eliminating redundancy.
"""

from typing import Optional, Any, Dict


class EngineError(Exception):
    """
    Base exception for all engine-related errors.
    
    Provides common functionality for error tracking and reporting.
    """
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize engine error.
        
        Args:
            message: Error message
            details: Optional detailed error context
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}


class ConvergenceError(EngineError):
    """
    Raised when convergence issues occur.
    
    Consolidates IterationLimitError and LoopDetectedError.
    """
    
    def __init__(self, message: str, 
                 iteration: Optional[int] = None,
                 loop_detected: bool = False,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize convergence error.
        
        Args:
            message: Error message
            iteration: Current iteration when error occurred
            loop_detected: Whether a loop was detected
            details: Additional error context
        """
        super().__init__(message, details)
        self.iteration = iteration
        self.loop_detected = loop_detected
        
        # Add to details for structured error handling
        self.details.update({
            'iteration': iteration,
            'loop_detected': loop_detected
        })


class ValidationError(EngineError):
    """
    Raised when validation fails.
    
    Consolidates PatchValidationError, SyntaxValidationError, and RiskThresholdError.
    """
    
    def __init__(self, message: str,
                 validation_type: str = 'general',
                 file_path: Optional[str] = None,
                 line_number: Optional[int] = None,
                 risk_level: Optional[float] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize validation error.
        
        Args:
            message: Error message
            validation_type: Type of validation that failed ('syntax', 'risk', 'patch', etc.)
            file_path: File where validation failed
            line_number: Line number where error occurred
            risk_level: Risk level if applicable
            details: Additional error context
        """
        super().__init__(message, details)
        self.validation_type = validation_type
        self.file_path = file_path
        self.line_number = line_number
        self.risk_level = risk_level
        
        # Add to details
        self.details.update({
            'validation_type': validation_type,
            'file_path': file_path,
            'line_number': line_number,
            'risk_level': risk_level
        })


class OrchestrationError(EngineError):
    """
    Raised when orchestration operations fail.
    
    Consolidates TestFailureError and RollbackError.
    """
    
    def __init__(self, message: str,
                 operation: str = 'general',
                 rollback_attempted: bool = False,
                 test_results: Optional[Dict[str, Any]] = None,
                 details: Optional[Dict[str, Any]] = None):
        """
        Initialize orchestration error.
        
        Args:
            message: Error message
            operation: Operation that failed ('test', 'rollback', 'apply', etc.)
            rollback_attempted: Whether rollback was attempted
            test_results: Test results if applicable
            details: Additional error context
        """
        super().__init__(message, details)
        self.operation = operation
        self.rollback_attempted = rollback_attempted
        self.test_results = test_results
        
        # Add to details
        self.details.update({
            'operation': operation,
            'rollback_attempted': rollback_attempted,
            'test_results': test_results
        })


# Helper functions for backward compatibility
def is_iteration_limit_error(error: Exception) -> bool:
    """Check if error is due to iteration limit."""
    return isinstance(error, ConvergenceError) and error.iteration is not None


def is_loop_detected_error(error: Exception) -> bool:
    """Check if error is due to loop detection."""
    return isinstance(error, ConvergenceError) and error.loop_detected


def is_syntax_error(error: Exception) -> bool:
    """Check if error is a syntax validation error."""
    return isinstance(error, ValidationError) and error.validation_type == 'syntax'


def is_risk_error(error: Exception) -> bool:
    """Check if error is a risk threshold error."""
    return isinstance(error, ValidationError) and error.validation_type == 'risk'


def is_test_failure(error: Exception) -> bool:
    """Check if error is a test failure."""
    return isinstance(error, OrchestrationError) and error.operation == 'test'


def is_rollback_error(error: Exception) -> bool:
    """Check if error is a rollback failure."""
    return isinstance(error, OrchestrationError) and error.operation == 'rollback'