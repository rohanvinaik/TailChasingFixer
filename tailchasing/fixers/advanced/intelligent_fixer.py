"""
Intelligent Auto-Fixer for detected tail-chasing patterns.
"""

import ast
import re
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass
from collections import defaultdict

from ...core.issues import Issue


@dataclass
class FixAction:
    """Represents a single fix action."""
    action_type: str
    target_file: str
    target_line: Optional[int]
    old_code: Optional[str]
    new_code: Optional[str]
    description: str


@dataclass
class FixPlan:
    """Complete fix plan for detected issues."""
    issues_addressed: List[Issue]
    actions: List[FixAction]
    estimated_impact: Dict[str, Any]
    rollback_plan: List[str]
    execution_order: List[str]


class IntelligentAutoFixer:
    """Automatically fix detected tail-chasing patterns."""
    
    def __init__(self):
        self.fix_strategies = {
            'semantic_duplicate': self._fix_semantic_duplicate,
            'semantic_duplicate_multimodal': self._fix_semantic_duplicate,
            'phantom_implementation': self._fix_phantom_implementation,
            'circular_import': self._fix_circular_import,
            'import_anxiety': self._fix_import_anxiety,
            'context_window_thrashing': self._fix_context_thrashing,
            'hallucination_cascade': self._fix_hallucination_cascade,
        }
    
    def generate_fix_plan(self, issues: List[Issue]) -> FixPlan:
        """Generate comprehensive fix plan for detected patterns."""
        # Group issues by type
        grouped_issues = defaultdict(list)
        for issue in issues:
            grouped_issues[issue.kind].append(issue)
        
        # Generate fixes in optimal order
        fix_order = [
            'circular_import',  # Fix these first as they block other fixes
            'import_anxiety',   # Clean up imports
            'hallucination_cascade',  # Remove fictional subsystems
            'semantic_duplicate',  # Merge duplicates
            'semantic_duplicate_multimodal',  # Merge multimodal duplicates
            'context_window_thrashing',  # Fix context issues
            'phantom_implementation'  # Implement stubs last
        ]
        
        all_actions = []
        execution_order = []
        
        for issue_type in fix_order:
            if issue_type in grouped_issues:
                strategy = self.fix_strategies.get(issue_type)
                if strategy:
                    actions = strategy(grouped_issues[issue_type])
                    all_actions.extend(actions)
                    execution_order.append(issue_type)
        
        # Estimate impact
        estimated_impact = self._estimate_fix_impact(all_actions)
        
        # Generate rollback plan
        rollback_plan = self._generate_rollback_plan(all_actions)
        
        return FixPlan(
            issues_addressed=issues,
            actions=all_actions,
            estimated_impact=estimated_impact,
            rollback_plan=rollback_plan,
            execution_order=execution_order
        )
    
    def _fix_semantic_duplicate(self, issues: List[Issue]) -> List[FixAction]:
        """Fix semantic duplicate functions by creating aliases and deprecation warnings."""
        actions = []
        
        # Group related issues
        duplicate_groups = self._group_duplicate_issues(issues)
        
        for group in duplicate_groups:
            # Choose primary implementation (highest confidence/quality)
            primary = self._choose_primary_duplicate(group)
            others = [issue for issue in group if issue != primary]
            
            for dup_issue in others:
                # Create deprecation alias
                actions.append(FixAction(
                    action_type="create_deprecation_alias",
                    target_file=dup_issue.file,
                    target_line=dup_issue.line,
                    old_code=f"def {dup_issue.evidence.get('function1', 'unknown')}",
                    new_code=f"# DEPRECATED: Use {primary.evidence.get('function1', 'primary')} instead\\n"
                             f"def {dup_issue.evidence.get('function1', 'unknown')}(*args, **kwargs):\\n"
                             f"    import warnings\\n"
                             f"    warnings.warn('This function is deprecated. Use {primary.evidence.get('function1', 'primary')} instead.', DeprecationWarning)\\n"
                             f"    return {primary.evidence.get('function1', 'primary')}(*args, **kwargs)",
                    description=f"Create deprecation alias for {dup_issue.evidence.get('function1', 'unknown')}"
                ))
                
                # Add TODO comment for manual review
                actions.append(FixAction(
                    action_type="add_todo_comment",
                    target_file=dup_issue.file,
                    target_line=dup_issue.line - 1,
                    old_code="",
                    new_code=f"# TODO: Review and remove this duplicate function after updating all references\\n"
                             f"# Similar to: {primary.file}:{primary.line}",
                    description="Add review comment for duplicate function"
                ))
        
        return actions
    
    def _fix_phantom_implementation(self, issues: List[Issue]) -> List[FixAction]:
        """Generate basic implementations for phantom functions."""
        actions = []
        
        for issue in issues:
            function_name = issue.symbol or "unknown_function"
            
            # Generate basic implementation template
            implementation = f"""def {function_name}(self, *args, **kwargs):
    \"\"\"
    Auto-generated implementation for {function_name}.
    
    TODO: Complete this implementation based on the function's intended purpose.
    This was generated because the original was a placeholder (pass/NotImplementedError).
    \"\"\"
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Using auto-generated implementation for {function_name} - please review and complete")
    
    # TODO: Add your implementation here
    # Based on the function name, it should probably:
    # - Validate input parameters
    # - Perform the main operation
    # - Return appropriate results
    
    raise NotImplementedError(f"Please implement {function_name} - this is an auto-generated stub")"""
            
            actions.append(FixAction(
                action_type="implement_phantom_function",
                target_file=issue.file,
                target_line=issue.line,
                old_code="pass",  # or NotImplementedError
                new_code=implementation,
                description=f"Generate implementation template for {function_name}"
            ))
        
        return actions
    
    def _fix_circular_import(self, issues: List[Issue]) -> List[FixAction]:
        """Fix circular imports through import restructuring."""
        actions = []
        
        for issue in issues:
            cycle = issue.evidence.get('cycle', [])
            if len(cycle) >= 2:
                # Strategy: Move import inside function to break cycle
                actions.append(FixAction(
                    action_type="move_import_local",
                    target_file=cycle[0],
                    target_line=1,  # Top of file
                    old_code=f"from {cycle[1]} import *",  # Simplified
                    new_code="# Import moved to function scope to break circular dependency",
                    description=f"Move import from {cycle[1]} to local scope"
                ))
                
                # Add comment explaining the fix
                actions.append(FixAction(
                    action_type="add_explanation_comment",
                    target_file=cycle[0],
                    target_line=1,
                    old_code="",
                    new_code="# Note: Some imports moved to function scope to resolve circular dependencies",
                    description="Add explanation for import restructuring"
                ))
        
        return actions
    
    def _fix_import_anxiety(self, issues: List[Issue]) -> List[FixAction]:
        """Remove unnecessary imports."""
        actions = []
        
        for issue in issues:
            unused_items = issue.evidence.get('unused_items', [])
            module = issue.evidence.get('module', 'unknown')
            
            if unused_items:
                actions.append(FixAction(
                    action_type="remove_unused_imports",
                    target_file=issue.file,
                    target_line=issue.line,
                    old_code=f"from {module} import {', '.join(unused_items)}",
                    new_code="# Unused imports removed by TailChasingFixer",
                    description=f"Remove {len(unused_items)} unused imports from {module}"
                ))
        
        return actions
    
    def _fix_context_thrashing(self, issues: List[Issue]) -> List[FixAction]:
        """Fix context window thrashing by consolidating similar functions."""
        actions = []
        
        for issue in issues:
            func1 = issue.evidence.get('function1', 'func1')
            func2 = issue.evidence.get('function2', 'func2')
            
            # Create consolidation comment
            actions.append(FixAction(
                action_type="add_consolidation_comment",
                target_file=issue.file,
                target_line=issue.line,
                old_code="",
                new_code=f"# CONTEXT THRASHING DETECTED: Functions '{func1}' and '{func2}' are very similar\\n"
                         f"# Consider consolidating these functions or extracting common functionality\\n"
                         f"# Similarity: {issue.evidence.get('similarity', 0):.0%}",
                description=f"Add consolidation suggestion for {func1} and {func2}"
            ))
        
        return actions
    
    def _fix_hallucination_cascade(self, issues: List[Issue]) -> List[FixAction]:
        """Fix hallucination cascades by marking for review."""
        actions = []
        
        for issue in issues:
            components = issue.evidence.get('components', [])
            
            # Add review comments to each component
            actions.append(FixAction(
                action_type="add_review_comment",
                target_file=issue.file,
                target_line=issue.line,
                old_code="",
                new_code=f"# HALLUCINATION CASCADE DETECTED\\n"
                         f"# This class is part of a group of {len(components)} related classes\\n"
                         f"# created together with minimal external references.\\n"
                         f"# Review if this entire subsystem is necessary or if existing\\n"
                         f"# functionality could be used instead.\\n"
                         f"# Related components: {', '.join(components[:5])}{'...' if len(components) > 5 else ''}",
                description=f"Mark hallucination cascade involving {len(components)} components"
            ))
        
        return actions
    
    def _group_duplicate_issues(self, issues: List[Issue]) -> List[List[Issue]]:
        """Group related duplicate issues together."""
        groups = []
        used_issues = set()
        
        for issue in issues:
            if id(issue) in used_issues:
                continue
            
            group = [issue]
            used_issues.add(id(issue))
            
            # Find related issues
            for other_issue in issues:
                if id(other_issue) not in used_issues and self._are_related_duplicates(issue, other_issue):
                    group.append(other_issue)
                    used_issues.add(id(other_issue))
            
            groups.append(group)
        
        return groups
    
    def _are_related_duplicates(self, issue1: Issue, issue2: Issue) -> bool:
        """Check if two duplicate issues are related."""
        # Simple heuristic: same functions involved
        func1_1 = issue1.evidence.get('function1', '')
        func1_2 = issue1.evidence.get('function2', '')
        func2_1 = issue2.evidence.get('function1', '')
        func2_2 = issue2.evidence.get('function2', '')
        
        return (func1_1 == func2_1 or func1_1 == func2_2 or 
                func1_2 == func2_1 or func1_2 == func2_2)
    
    def _choose_primary_duplicate(self, group: List[Issue]) -> Issue:
        """Choose the best implementation from a group of duplicates."""
        # Prefer issues with higher confidence/similarity scores
        return max(group, key=lambda issue: issue.evidence.get('similarity', 0))
    
    def _estimate_fix_impact(self, actions: List[FixAction]) -> Dict[str, Any]:
        """Estimate the impact of applying fixes."""
        files_affected = set()
        functions_modified = 0
        imports_changed = 0
        
        for action in actions:
            files_affected.add(action.target_file)
            
            if action.action_type in ['implement_phantom_function', 'create_deprecation_alias']:
                functions_modified += 1
            elif action.action_type in ['remove_unused_imports', 'move_import_local']:
                imports_changed += 1
        
        # Calculate risk level
        risk_level = 'low'
        if functions_modified > 10 or len(files_affected) > 20:
            risk_level = 'high'
        elif functions_modified > 5 or len(files_affected) > 10:
            risk_level = 'medium'
        
        return {
            'files_affected': len(files_affected),
            'functions_modified': functions_modified,
            'imports_changed': imports_changed,
            'risk_level': risk_level,
            'estimated_time_minutes': functions_modified * 2 + imports_changed * 0.5
        }
    
    def _generate_rollback_plan(self, actions: List[FixAction]) -> List[str]:
        """Generate rollback commands for the fixes."""
        rollback_commands = []
        
        files_to_backup = set(action.target_file for action in actions)
        for file in files_to_backup:
            rollback_commands.append(f"git checkout HEAD -- {file}")
        
        rollback_commands.append("# Run tests to ensure rollback was successful")
        rollback_commands.append("# Review any manual changes that may need to be re-applied")
        
        return rollback_commands
