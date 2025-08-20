"""
Auto-fix engine package for TailChasingFixer.

This package provides intelligent automated fixing capabilities with comprehensive
safety mechanisms, dependency analysis, and rollback support.
"""

from .base import (
    FixStatus,
    FixPriority, 
    FixResult,
    FixPlan,
    FixAction
)

from .backup import BackupManager
from .dependency import DependencyAnalyzer
from .safety import SafetyValidator
from .registry import FixStrategyRegistry
from .engine import IntelligentAutoFixer

# Convenience factory functions
from .factory import create_auto_fixer, auto_fix_issues

__all__ = [
    # Base types
    'FixStatus',
    'FixPriority',
    'FixResult', 
    'FixPlan',
    'FixAction',
    
    # Core components
    'BackupManager',
    'DependencyAnalyzer',
    'SafetyValidator',
    'FixStrategyRegistry',
    'IntelligentAutoFixer',
    
    # Factory functions
    'create_auto_fixer',
    'auto_fix_issues'
]