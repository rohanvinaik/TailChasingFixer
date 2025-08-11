"""
Fix strategies for tail-chasing patterns.

This package contains modular fix strategies that can be applied to
different types of issues detected by the analyzers.
"""

from .base import (
    RiskLevel,
    Action,
    Patch,
    FixStrategy,
    BaseFixStrategy
)

from .imports import ImportResolutionStrategy

# TODO: Import other strategies as they are created
# from .duplicates import DuplicateMergeStrategy
# from .placeholders import PlaceholderImplementationStrategy
# from .circular import CircularDependencyBreaker
# from .async_sync import AsyncSyncMismatchFixer
# from .selector import StrategySelector

__all__ = [
    # Base classes and types
    'RiskLevel',
    'Action',
    'Patch',
    'FixStrategy',
    'BaseFixStrategy',
    
    # Concrete strategies
    'ImportResolutionStrategy',
    # 'DuplicateMergeStrategy',
    # 'PlaceholderImplementationStrategy',
    # 'CircularDependencyBreaker',
    # 'AsyncSyncMismatchFixer',
    # 'StrategySelector',
]