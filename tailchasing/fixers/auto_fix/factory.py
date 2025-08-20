"""
Factory functions for creating and using auto-fix engine components.

Provides convenient high-level interfaces for automated fixing.
"""

from typing import Dict, List, Optional, Any

from ...core.issues import Issue
from .base import FixResult
from .engine import IntelligentAutoFixer


def create_auto_fixer(
    dry_run: bool = True,
    backup_dir: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None
) -> IntelligentAutoFixer:
    """
    Create and configure an IntelligentAutoFixer instance.
    
    Args:
        dry_run: If True, simulate fixes without applying them
        backup_dir: Directory for backups. If None, uses temp directory
        config: Additional configuration options
        
    Returns:
        Configured IntelligentAutoFixer instance
    """
    full_config = {
        'dry_run': dry_run,
        'backup_dir': backup_dir,
        'enable_validation': True,
        'max_risk_level': 'medium',
        'require_tests': True,
        **(config or {})
    }
    
    return IntelligentAutoFixer(full_config)


def auto_fix_issues(
    issues: List[Issue],
    dry_run: bool = True,
    max_fixes: int = 10,
    config: Optional[Dict[str, Any]] = None
) -> List[FixResult]:
    """
    Automatically fix a list of issues using the intelligent auto-fixer.
    
    Args:
        issues: List of issues to fix
        dry_run: If True, simulate fixes without applying them  
        max_fixes: Maximum number of fixes to apply
        config: Additional configuration options
        
    Returns:
        List of FixResult objects describing what was fixed
    """
    # Create auto-fixer with configuration
    auto_fixer = create_auto_fixer(dry_run=dry_run, config=config)
    
    # Apply fixes
    results = auto_fixer.fix_issues(issues, max_fixes=max_fixes)
    
    return results


__all__ = [
    'create_auto_fixer',
    'auto_fix_issues'
]