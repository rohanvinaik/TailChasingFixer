"""Advanced fixers for tail-chasing patterns."""

from .fix_strategies import (
    ImportResolutionStrategy,
    DuplicateMergeStrategy,
    PlaceholderImplementationStrategy,
    CircularDependencyBreaker,
    AsyncSyncMismatchFixer,
    StrategySelector,
    RiskLevel
)

__all__ = [
    "ImportResolutionStrategy",
    "DuplicateMergeStrategy",
    "PlaceholderImplementationStrategy",
    "CircularDependencyBreaker",
    "AsyncSyncMismatchFixer",
    "StrategySelector",
    "RiskLevel",
]
