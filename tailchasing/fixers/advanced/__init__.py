"""Advanced fixers for tail-chasing patterns."""

from .intelligent_fixer import IntelligentAutoFixer
from .fix_strategies import (
    SemanticDuplicateFixer,
    PhantomImplementationFixer,
    CircularImportFixer,
    ImportAnxietyFixer
)

__all__ = [
    'IntelligentAutoFixer',
    'SemanticDuplicateFixer',
    'PhantomImplementationFixer',
    'CircularImportFixer',
    'ImportAnxietyFixer'
]
