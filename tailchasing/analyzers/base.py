"""Base analyzer interface and context."""

from __future__ import annotations
# Import consolidated types and protocols
from ..core.types import (
    Analyzer,
    AnalysisContext,
    BaseAnalyzer
)
from ..shared import common_functions

from typing import Dict, Any, TYPE_CHECKING
from pathlib import Path

if TYPE_CHECKING:
    pass

# Re-export for backward compatibility
__all__ = ['Analyzer', 'AnalysisContext', 'BaseAnalyzer']


# Legacy helper functions for backward compatibility
def get_confidence(base_confidence: float, modifiers: Dict[str, float]) -> float:
    """Calculate confidence score with modifiers.""" 
    return common_functions.get_confidence(base_confidence, modifiers)


def is_excluded(path: str, root_dir: Path, config: Dict[str, Any]) -> bool:
    """Check if a path should be excluded based on config."""
    return common_functions.is_excluded(path, root_dir, config)


def should_ignore_issue(issue_kind: str, config: Dict[str, Any]) -> bool:
    """Check if an issue type should be ignored."""
    return common_functions.should_ignore_issue(issue_kind, config)


def is_placeholder_allowed(symbol: str, config: Dict[str, Any]) -> bool:
    """Check if a placeholder is explicitly allowed."""
    return common_functions.is_placeholder_allowed(symbol, config)
