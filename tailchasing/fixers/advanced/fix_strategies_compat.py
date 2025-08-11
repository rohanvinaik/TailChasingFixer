"""
Compatibility layer for fix_strategies refactoring.

This module provides backward compatibility while we migrate to the new
modular structure in tailchasing/fixers/strategies/.

This file can be deleted once all imports are updated to use the new location.
"""

# Re-export from new location
from ..strategies import (
    RiskLevel,
    Action,
    Patch,
    FixStrategy,
    BaseFixStrategy,
    ImportResolutionStrategy,
)

# TODO: Re-export other strategies as they are migrated
# from ..strategies import (
#     DuplicateMergeStrategy,
#     PlaceholderImplementationStrategy,
#     CircularDependencyBreaker,
#     AsyncSyncMismatchFixer,
#     StrategySelector,
# )

# Keep imports from old file for now (will be removed)
from .fix_strategies import (
    DuplicateMergeStrategy,
    PlaceholderImplementationStrategy,
    CircularDependencyBreaker,
    AsyncSyncMismatchFixer,
    StrategySelector,
    FixOutcome,
    StrategyRanking,
    FixAttempt,
)

# Classes that don't exist but are imported by __init__.py
# Create dummy classes for now
class SemanticDuplicateFixer:
    """Placeholder - to be implemented."""
    pass

class PhantomImplementationFixer:
    """Placeholder - to be implemented."""
    pass

class CircularImportFixer:
    """Placeholder - to be implemented."""
    pass

class ImportAnxietyFixer:
    """Placeholder - to be implemented."""
    pass

__all__ = [
    # From new location
    'RiskLevel',
    'Action',
    'Patch',
    'FixStrategy',
    'BaseFixStrategy',
    'ImportResolutionStrategy',
    
    # From old location (temporary)
    'DuplicateMergeStrategy',
    'PlaceholderImplementationStrategy',
    'CircularDependencyBreaker',
    'AsyncSyncMismatchFixer',
    'StrategySelector',
    'FixOutcome',
    'StrategyRanking',
    'FixAttempt',
    
    # Placeholder classes
    'SemanticDuplicateFixer',
    'PhantomImplementationFixer',
    'CircularImportFixer',
    'ImportAnxietyFixer',
]