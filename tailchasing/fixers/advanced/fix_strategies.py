"""
Fix Strategies

Specific strategies for fixing different types of tail-chasing patterns.
"""

import ast
import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ...core.issues import Issue
from .intelligent_fixer import Fix


class FixStrategy(ABC):
    """Base class for fix strategies."""
    
    @abstractmethod
    def generate_fixes(self, issues: List[Issue]) -> List[Fix]:
        """Generate fixes for a list of issues."""
        pass


class SemanticDuplicateFixer(FixStrategy):
    """Fix semantic duplicates by merging them intelligently."""
    
    def generate_fixes(self, issues: List[Issue]) -> List[Fix]:
        """Generate fixes for semantic duplicate issues."""
        fixes = []
        
        # Group related duplicates
        duplicate_groups = self._group_related_duplicates(issues)
        
        for group in duplicate_groups:
            # Choose the best implementation as primary
            primary = self._choose_primary_implementation(group)
            others = [issue for issue in group if issue != primary]
            
            # Generate merge fixes
            for duplicate in others:
                fixes.extend(self._create_merge_fixes(primary, duplicate))
        
        return fixes
    
    def _group_related_duplicates(self, issues: List[Issue]) -> List[List[Issue]]:
        """Group semantically related duplicate issues."""
        # Simple grouping by evidence pairs
        groups = []
        used = set()
        
        for i, issue in enumerate(issues):
            if i in used:
                continue
            
            group = [issue]
            used.add(i)
            
            # Find related issues
            if 'pair' in issue.evidence:
                pair = issue.evidence['pair']
                for j, other in enumerate(issues[i+1:], i+1):
                    if j not in used and 'pair' in other.evidence:
                        other_pair = other.evidence['pair']
                        # Check if they share any functions
                        if (pair[0][0] in [p[0] for p in other_pair] or
                            pair[1][0] in [p[0] for p in other_pair]):
                            group.append(other)
                            used.add(j)
            
            groups.append(group)
        
        return groups
    
    def _choose_primary_implementation(self, group: List[Issue]) -> Issue:
        """Choose the best implementation from a group of duplicates."""
        # Simple heuristic: prefer the one with more evidence
        # In real implementation would analyze code quality
        return max(group, key=lambda i: i.evidence.get('z_score', 0))
    
    def _create_merge_fixes(self, primary: Issue, duplicate: Issue) -> List[Fix]:
        """Create fixes to merge duplicate into primary."""
        fixes = []
        
        # Extract function names from evidence
        if 'pair' in duplicate.evidence:
            dup_func = duplicate.evidence['pair'][0][0]
            primary_func = duplicate.evidence['pair'][1][0]
            
            # Create deprecation alias
            alias_code = f"""# TODO: Deprecated - use {primary_func} instead
{dup_func} = {primary_func}

import warnings
def _{dup_func}_deprecated(*args, **kwargs):
    warnings.warn(
        f"{dup_func} is deprecated, use {primary_func} instead",
        DeprecationWarning,
        stacklevel=2
    )
    return {primary_func}(*args, **kwargs)

{dup_func} = _{dup_func}_deprecated
"""
            
            fixes.append(Fix(
                type='merge_duplicates',
                file=duplicate.file,
                line=duplicate.line,
                original_code=f"def {dup_func}(...): ...",  # Placeholder
                fixed_code=alias_code,
                description=f"Create deprecation alias from {dup_func} to {primary_func}",
                confidence=0.8,
                impact={'deprecation': True, 'backward_compatible': True}
            ))
        
        return fixes


class PhantomImplementationFixer(FixStrategy):
    """Fix phantom/stub implementations by generating real code."""
    
    def generate_fixes(self, issues: List[Issue]) -> List[Fix]:
        """Generate implementations for phantom functions."""
        fixes = []
        
        for issue in issues:
            # Infer intended behavior
            intended_behavior = self._infer_intended_behavior(issue)
            
            # Generate implementation
            implementation = self._generate_implementation(
                issue.symbol or "unknown_function",
                intended_behavior
            )
            
            fixes.append(Fix(
                type='implement_phantom',
                file=issue.file,
                line=issue.line,
                original_code="pass  # or NotImplementedError",
                fixed_code=implementation,
                description=f"Generate implementation for phantom function {issue.symbol}",
                confidence=0.6,  # Lower confidence for generated code
                impact={'generated_code': True, 'needs_review': True}
            ))
        
        return fixes
    
    def _infer_intended_behavior(self, issue: Issue) -> Dict[str, Any]:
        """Infer what a phantom function should do from context."""
        behavior = {
            'purpose': 'unknown',
            'parameters': [],
            'return_type': 'Any',
            'pattern': 'basic'
        }
        
        # Analyze function name
        if issue.symbol:
            name_lower = issue.symbol.lower()
            
            # Common patterns
            if name_lower.startswith('get_') or name_lower.startswith('fetch_'):
                behavior['pattern'] = 'getter'
                behavior['purpose'] = f"Retrieve {name_lower.replace('get_', '').replace('fetch_', '')}"
            elif name_lower.startswith('set_') or name_lower.startswith('update_'):
                behavior['pattern'] = 'setter'
                behavior['purpose'] = f"Update {name_lower.replace('set_', '').replace('update_', '')}"
            elif name_lower.startswith('is_') or name_lower.startswith('has_'):
                behavior['pattern'] = 'predicate'
                behavior['return_type'] = 'bool'
                behavior['purpose'] = f"Check {name_lower}"
            elif 'process' in name_lower or 'handle' in name_lower:
                behavior['pattern'] = 'processor'
                behavior['purpose'] = f"Process input for {name_lower}"
            elif 'validate' in name_lower:
                behavior['pattern'] = 'validator'
                behavior['purpose'] = f"Validate input for {name_lower}"
        
        return behavior
    
    def _generate_implementation(self, func_name: str, behavior: Dict[str, Any]) -> str:
        """Generate code implementation based on inferred behavior."""
        
        templates = {
            'getter': '''def {func_name}(self, key=None):
    """
    Auto-generated getter implementation.
    TODO: Review and complete this implementation.
    """
    if not hasattr(self, '_cache'):
        self._cache = {{}}
    
    if key is None:
        return self._cache
    
    return self._cache.get(key, None)
''',
            
            'setter': '''def {func_name}(self, key, value):
    """
    Auto-generated setter implementation.
    TODO: Review and complete this implementation.
    """
    if not hasattr(self, '_cache'):
        self._cache = {{}}
    
    self._cache[key] = value
    return True
''',
            
            'predicate': '''def {func_name}(self, value):
    """
    Auto-generated predicate implementation.
    TODO: Review and complete this implementation.
    """
    if value is None:
        return False
    
    # TODO: Add actual validation logic
    return True
''',
            
            'validator': '''def {func_name}(self, data):
    """
    Auto-generated validator implementation.
    TODO: Review and complete this implementation.
    """
    errors = []
    
    if not data:
        errors.append("Data is required")
    
    # TODO: Add specific validation rules
    
    if errors:
        raise ValueError(f"Validation failed: {{errors}}")
    
    return True
''',
            
            'processor': '''def {func_name}(self, *args, **kwargs):
    """
    Auto-generated processor implementation.
    TODO: Review and complete this implementation.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"Processing with {{len(args)}} args and {{len(kwargs)}} kwargs")
    
    try:
        # TODO: Add actual processing logic
        result = {{"status": "processed", "args": args, "kwargs": kwargs}}
        
        return result
    except Exception as e:
        logger.error(f"Error in {func_name}: {{e}}")
        raise
''',
            
            'basic': '''def {func_name}(self, *args, **kwargs):
    """
    Auto-generated implementation.
    TODO: Review and complete this implementation.
    Purpose: {purpose}
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.warning("Using auto-generated implementation - please review")
    
    # TODO: Implement actual logic here
    # Current behavior: log and return None
    logger.info(f"{func_name} called with args={{args}} kwargs={{kwargs}}")
    
    return None
'''
        }
        
        template = templates.get(behavior['pattern'], templates['basic'])
        return template.format(func_name=func_name, purpose=behavior['purpose'])


class CircularImportFixer(FixStrategy):
    """Fix circular imports through refactoring."""
    
    def generate_fixes(self, issues: List[Issue]) -> List[Fix]:
        """Generate fixes for circular import issues."""
        fixes = []
        
        for issue in issues:
            if 'cycle' in issue.evidence:
                cycle = issue.evidence['cycle']
                fixes.extend(self._break_cycle(cycle))
        
        return fixes
    
    def _break_cycle(self, cycle: List[str]) -> List[Fix]:
        """Generate fixes to break an import cycle."""
        fixes = []
        
        if len(cycle) < 2:
            return fixes
        
        # Strategy 1: Move import inside function
        module_a = cycle[0]
        module_b = cycle[1]
        
        move_import_code = '''# Move import inside function to break circular dependency
# Before:
# from {module_b} import something

def function_that_needs_import():
    """Function that uses the circular import."""
    # Import moved here to break circular dependency
    from {module_b} import something
    
    # Use 'something' here
    return something()
'''
        
        fixes.append(Fix(
            type='break_circular_import',
            file=module_a,
            line=1,
            original_code=f"from {module_b} import something",
            fixed_code=move_import_code.format(module_b=module_b),
            description=f"Move import of {module_b} inside function to break cycle",
            confidence=0.7,
            impact={'import_location': 'function', 'performance': 'minimal'}
        ))
        
        # Strategy 2: Create interface module
        interface_name = f"{module_a.replace('.py', '')}_interface.py"
        interface_code = '''"""
Interface module to break circular dependencies.

Move shared types, protocols, and abstract classes here.
"""

from typing import Protocol

# Example protocol to break circular dependency
class SharedProtocol(Protocol):
    """Define shared interface here."""
    def shared_method(self) -> None:
        ...

# Move other shared definitions here
'''
        
        fixes.append(Fix(
            type='create_interface',
            file=interface_name,
            line=1,
            original_code="",
            fixed_code=interface_code,
            description=f"Create interface module to break circular dependency",
            confidence=0.6,
            impact={'new_file': True, 'refactoring_required': True}
        ))
        
        return fixes


class ImportAnxietyFixer(FixStrategy):
    """Fix import anxiety by removing unnecessary imports."""
    
    def generate_fixes(self, issues: List[Issue]) -> List[Fix]:
        """Generate fixes for import anxiety issues."""
        fixes = []
        
        for issue in issues:
            if 'unused_items' in issue.evidence:
                module = issue.evidence.get('module', 'unknown')
                unused = issue.evidence['unused_items']
                
                # Generate fix to remove unused imports
                if unused:
                    fixes.append(self._create_import_cleanup_fix(
                        issue.file,
                        module,
                        unused,
                        issue.line
                    ))
        
        return fixes
    
    def _create_import_cleanup_fix(self,
                                  file: str,
                                  module: str,
                                  unused_items: List[str],
                                  line: Optional[int]) -> Fix:
        """Create fix to clean up unused imports."""
        
        # Generate cleaned import statement
        if len(unused_items) <= 3:
            unused_str = ', '.join(unused_items)
            description = f"Remove unused imports: {unused_str}"
        else:
            unused_str = f"{', '.join(unused_items[:3])}, and {len(unused_items)-3} more"
            description = f"Remove {len(unused_items)} unused imports from {module}"
        
        # Example cleaned import
        cleaned_code = f'''# Cleaned imports - removed unused items
# Removed: {unused_str}

# Keep only used imports
from {module} import (
    # TODO: List only the actually used items here
    used_function_1,
    used_function_2,
    UsedClass
)
'''
        
        return Fix(
            type='clean_imports',
            file=file,
            line=line or 1,
            original_code=f"from {module} import *  # or long list",
            fixed_code=cleaned_code,
            description=description,
            confidence=0.9,  # High confidence for unused import removal
            impact={'imports_removed': len(unused_items), 'safe': True}
        )
