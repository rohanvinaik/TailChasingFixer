"""
Duplicate code consolidation strategies.

Handles merging of semantic duplicates and similar function consolidation.
Extracted from fix_strategies.py to address context window thrashing.
"""

from typing import List, Dict, Any, Optional, Tuple

from .base import BaseFixStrategy, Action, Patch, RiskLevel
from ...core.issues import Issue


class DuplicateMergeStrategy(BaseFixStrategy):
    """
    Strategy for consolidating semantic duplicates.
    
    Handles:
    - Semantic duplicate functions
    - Duplicate function implementations
    - Duplicate class definitions
    
    Creates deprecation wrappers to maintain backward compatibility.
    """
    
    def __init__(self):
        super().__init__("DuplicateMerge")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle duplicate function and class issues."""
        return issue.kind in [
            "semantic_duplicate_function",
            "duplicate_function", 
            "duplicate_class"
        ]
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix to merge duplicate implementations."""
        actions = self._generate_fix_actions(issue, context)
        
        if not actions:
            return None
        
        # Calculate estimated time
        estimated_time = self._estimate_time(actions)
        
        # Generate validation tests
        validation_tests = self._generate_validation_tests(issue)
        validation_tests.extend([
            f"# Test that merged function works correctly",
            f"# TODO: Verify deprecation warnings are shown"
        ])
        
        # Create rollback plan
        rollback_plan = []
        for action in actions:
            rollback_plan.append(self.create_backup_action(action))
        
        return Patch(
            actions=actions,
            description=f"Merge duplicate implementation: {issue.symbol or 'unknown'}",
            confidence=0.8,
            risk_level=RiskLevel.MEDIUM,
            estimated_time=estimated_time,
            dependencies=self.get_issue_dependencies(issue),
            rollback_plan=rollback_plan,
            validation_tests=validation_tests,
            side_effects=["Creates deprecation warnings", "Changes function location"]
        )
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to merge duplicate code."""
        actions = []
        
        if not issue.evidence or 'pair' not in issue.evidence:
            return actions
        
        pair = issue.evidence['pair']
        if len(pair) < 2:
            return actions
        
        # Choose the better implementation as primary
        primary, secondary = self._choose_primary_and_secondary(pair)
        
        # Generate merge actions
        actions.extend(self._create_merge_actions(primary, secondary, issue))
        
        return actions
    
    def _choose_primary_and_secondary(self, pair: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Choose which implementation to keep as primary."""
        
        def score_implementation(impl: Dict[str, Any]) -> float:
            score = 0.0
            
            # Prefer longer implementations (more complete)
            code = impl.get('code', '')
            if code and len(code) > 20:
                score += 1.0
            
            # Prefer implementations with docstrings
            if '"""' in code or "'''" in code:
                score += 0.5
            
            # Prefer implementations without TODO/FIXME
            if 'TODO' not in code and 'FIXME' not in code:
                score += 0.5
            
            # Prefer implementations with better names
            name = impl.get('name', '')
            if name and not name.startswith('_') and len(name) > 3:
                score += 0.3
            
            # Prefer implementations in main modules over test files
            file_path = impl.get('file', '')
            if 'test' not in file_path.lower():
                score += 0.2
            
            return score
        
        scores = [(impl, score_implementation(impl)) for impl in pair]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[0][0], scores[1][0]
    
    def _create_merge_actions(self, primary: Dict[str, Any], secondary: Dict[str, Any], issue: Issue) -> List[Action]:
        """Create actions to merge secondary into primary."""
        actions = []
        
        primary_name = primary.get('name', 'primary_func')
        secondary_name = secondary.get('name', 'secondary_func')
        secondary_file = secondary.get('file', issue.file)
        primary_file = primary.get('file', issue.file)
        
        try:
            # Read secondary file to backup
            with open(secondary_file, 'r') as f:
                original_content = f.read()
            
            # Create deprecation wrapper for secondary
            deprecation_code = self._generate_deprecation_wrapper(
                secondary_name, primary_name, primary_file
            )
            
            # Replace secondary implementation with deprecation wrapper
            new_content = self._replace_function_in_content(
                original_content, 
                secondary_name,
                deprecation_code
            )
            
            actions.append(Action(
                type="modify_file",
                target=secondary_file,
                content=new_content,
                backup_content=original_content,
                metadata={
                    "merged_into": primary_name,
                    "deprecated": secondary_name,
                    "type": "deprecation_wrapper",
                    "primary_file": primary_file
                }
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not create merge action: {e}")
        
        return actions
    
    def _generate_deprecation_wrapper(self, secondary_name: str, primary_name: str, primary_file: str) -> str:
        """Generate a deprecation wrapper function."""
        # Convert file path to module name
        module_name = primary_file.replace('.py', '').replace('/', '.').replace('\\', '.')
        if module_name.startswith('.'):
            module_name = module_name[1:]
        
        return f'''# Deprecated: {secondary_name} merged into {primary_name}
import warnings

def {secondary_name}(*args, **kwargs):
    """
    DEPRECATED: This function has been merged into {primary_name}.
    Please use {primary_name} instead.
    """
    warnings.warn(
        f"{secondary_name} is deprecated, use {primary_name} instead",
        DeprecationWarning,
        stacklevel=2
    )
    try:
        from {module_name} import {primary_name}
        return {primary_name}(*args, **kwargs)
    except ImportError:
        # Fallback for import issues
        raise NotImplementedError(
            f"{secondary_name} has been deprecated and merged into {primary_name}. "
            f"Please update imports to use {primary_name} from {module_name}."
        )
'''
    
    def _replace_function_in_content(self, content: str, func_name: str, replacement: str) -> str:
        """Replace a function definition with new content."""
        lines = content.split('\n')
        
        # Find function definition
        func_start = -1
        func_end = -1
        indent_level = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith(f'def {func_name}('):
                func_start = i
                indent_level = len(line) - len(line.lstrip())
                continue
            
            if func_start != -1 and line.strip():
                current_indent = len(line) - len(line.lstrip())
                if current_indent <= indent_level and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                    func_end = i
                    break
        
        if func_start != -1:
            if func_end == -1:
                func_end = len(lines)
            
            # Replace function with proper indentation
            replacement_lines = replacement.split('\n')
            if func_start > 0:
                # Preserve original indentation
                base_indent = ' ' * indent_level
                indented_replacement = []
                for line in replacement_lines:
                    if line.strip():
                        indented_replacement.append(base_indent + line)
                    else:
                        indented_replacement.append('')
                replacement_lines = indented_replacement
            
            # Replace function
            new_lines = lines[:func_start] + replacement_lines + lines[func_end:]
            return '\n'.join(new_lines)
        
        return content


class SemanticSimilarityMerger(BaseFixStrategy):
    """
    Strategy for merging functions with high semantic similarity.
    
    Uses more sophisticated analysis to identify functions that serve
    the same purpose but have slightly different implementations.
    """
    
    def __init__(self):
        super().__init__("SemanticSimilarityMerger")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle semantic similarity issues."""
        return issue.kind in [
            "semantic_duplicate_function",
            "similar_function_pattern",
            "context_window_thrashing"  # When similar functions are far apart
        ]
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix to consolidate semantically similar functions."""
        if not issue.evidence or 'similarity_score' not in issue.evidence:
            return None
        
        similarity_score = issue.evidence.get('similarity_score', 0.0)
        
        # Only proceed if similarity is high enough
        if similarity_score < 0.8:
            return None
        
        actions = self._generate_consolidation_actions(issue, context)
        
        if not actions:
            return None
        
        return Patch(
            actions=actions,
            description=f"Consolidate semantically similar functions (similarity: {similarity_score:.2f})",
            confidence=min(similarity_score, 0.9),  # Cap confidence
            risk_level=RiskLevel.MEDIUM,
            estimated_time=self._estimate_time(actions),
            dependencies=self.get_issue_dependencies(issue),
            rollback_plan=[self.create_backup_action(action) for action in actions],
            validation_tests=self._generate_validation_tests(issue),
            side_effects=["Function consolidation", "Possible API changes"]
        )
    
    def _generate_consolidation_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to consolidate similar functions."""
        actions = []
        
        if not issue.evidence or 'functions' not in issue.evidence:
            return actions
        
        functions = issue.evidence['functions']
        if len(functions) < 2:
            return actions
        
        # Choose the best function as the canonical implementation
        canonical = self._choose_canonical_function(functions)
        
        # Create consolidation actions for other functions
        for func in functions:
            if func != canonical:
                actions.extend(self._create_redirect_actions(func, canonical, issue))
        
        return actions
    
    def _choose_canonical_function(self, functions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Choose the canonical function to keep."""
        
        def score_function(func: Dict[str, Any]) -> float:
            score = 0.0
            
            # Prefer functions with better documentation
            code = func.get('code', '')
            if '"""' in code or "'''" in code:
                score += 2.0
            
            # Prefer functions with type hints
            if '->' in func.get('signature', ''):
                score += 1.0
            
            # Prefer functions with more comprehensive error handling
            if 'try:' in code or 'except' in code:
                score += 1.0
            
            # Prefer functions that are called more frequently (if available)
            usage_count = func.get('usage_count', 0)
            score += min(usage_count * 0.1, 2.0)  # Cap at 2.0
            
            # Prefer functions in main modules
            file_path = func.get('file', '')
            if 'test' not in file_path.lower() and 'util' not in file_path.lower():
                score += 1.0
            
            return score
        
        scored_functions = [(func, score_function(func)) for func in functions]
        scored_functions.sort(key=lambda x: x[1], reverse=True)
        
        return scored_functions[0][0]
    
    def _create_redirect_actions(self, func: Dict[str, Any], canonical: Dict[str, Any], issue: Issue) -> List[Action]:
        """Create actions to redirect a function to the canonical implementation."""
        actions = []
        
        func_name = func.get('name', 'unknown_func')
        canonical_name = canonical.get('name', 'canonical_func')
        func_file = func.get('file', issue.file)
        canonical_file = canonical.get('file', issue.file)
        
        try:
            # Read file content
            with open(func_file, 'r') as f:
                original_content = f.read()
            
            # Create redirect implementation
            redirect_code = self._generate_redirect_function(func, canonical)
            
            # Replace function with redirect
            new_content = self._replace_function_in_content(
                original_content,
                func_name,
                redirect_code
            )
            
            actions.append(Action(
                type="modify_file",
                target=func_file,
                content=new_content,
                backup_content=original_content,
                metadata={
                    "redirected_to": canonical_name,
                    "original_function": func_name,
                    "type": "function_redirect",
                    "canonical_file": canonical_file
                }
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not create redirect action for {func_name}: {e}")
        
        return actions
    
    def _generate_redirect_function(self, func: Dict[str, Any], canonical: Dict[str, Any]) -> str:
        """Generate a redirect function implementation."""
        func_name = func.get('name', 'unknown')
        canonical_name = canonical.get('name', 'canonical')
        canonical_file = canonical.get('file', '')
        
        # Convert file to module
        module_name = canonical_file.replace('.py', '').replace('/', '.').replace('\\', '.')
        if module_name.startswith('.'):
            module_name = module_name[1:]
        
        # Extract function signature for compatibility
        signature = func.get('signature', f'{func_name}()')
        
        return f'''def {signature}:
    """
    Redirect to canonical implementation.
    
    This function has been consolidated with {canonical_name} for better maintainability.
    """
    from {module_name} import {canonical_name}
    return {canonical_name}(*args, **kwargs)
'''