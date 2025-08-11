"""
Modular Fix Strategies System

Implements specific, targeted strategies for fixing different types of tail-chasing patterns
with risk assessment, rollback capabilities, and intelligent selection.
"""

import ast
import re
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Tuple, Set
from enum import Enum
import logging

from ...core.issues import Issue
from ...utils.logging_setup import get_logger


class RiskLevel(Enum):
    """Risk levels for fix strategies."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Action:
    """Represents a single action in a fix or rollback plan."""
    type: str  # "modify_file", "create_file", "delete_lines", etc.
    target: str  # File path or identifier
    content: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    backup_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Patch:
    """Represents a proposed fix with metadata."""
    actions: List[Action]
    description: str
    confidence: float  # 0.0 - 1.0
    risk_level: RiskLevel
    estimated_time: float  # seconds
    dependencies: List[str] = field(default_factory=list)  # Other patches this depends on
    rollback_plan: List[Action] = field(default_factory=list)
    validation_tests: List[str] = field(default_factory=list)  # Test commands to verify fix
    side_effects: List[str] = field(default_factory=list)  # Potential side effects
    
    def estimate_impact(self) -> Dict[str, Any]:
        """Estimate the impact of applying this patch."""
        lines_modified = sum(1 for action in self.actions if action.type == "modify_file")
        files_affected = len(set(action.target for action in self.actions))
        
        return {
            "files_affected": files_affected,
            "lines_modified": lines_modified,
            "risk_level": self.risk_level.name,
            "confidence": self.confidence,
            "estimated_time": self.estimated_time
        }


class FixStrategy(Protocol):
    """Protocol defining the interface for fix strategies."""
    
    name: str
    
    def can_handle(self, issue: Issue) -> bool:
        """Check if this strategy can handle the given issue."""
        ...
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix for the issue with full metadata."""
        ...
    
    def estimate_risk(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> RiskLevel:
        """Estimate the risk level of fixing this issue."""
        ...
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Get list of other issue types that should be fixed first."""
        ...
    
    def learn_from_outcome(self, issue: Issue, patch: Patch, success: bool, feedback: str) -> None:
        """Learn from the outcome of applying this strategy."""
        ...


class BaseFixStrategy(ABC):
    """Base implementation of FixStrategy with common functionality."""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"strategy.{name}")
        self.success_history: List[Dict[str, Any]] = []
        self.failure_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}
    
    @abstractmethod
    def can_handle(self, issue: Issue) -> bool:
        """Check if this strategy can handle the given issue."""
        pass
    
    @abstractmethod
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate the specific actions needed to fix the issue."""
        pass
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix for the issue."""
        if not self.can_handle(issue):
            return None
        
        try:
            actions = self._generate_fix_actions(issue, context)
            if not actions:
                return None
            
            confidence = self._calculate_confidence(issue, context)
            risk_level = self.estimate_risk(issue, context)
            rollback_plan = self._generate_rollback_plan(actions)
            validation_tests = self._generate_validation_tests(issue)
            
            return Patch(
                actions=actions,
                description=f"{self.name} fix for {issue.kind}",
                confidence=confidence,
                risk_level=risk_level,
                estimated_time=self._estimate_time(actions),
                dependencies=self.get_dependencies(issue),
                rollback_plan=rollback_plan,
                validation_tests=validation_tests,
                side_effects=self._identify_side_effects(issue, actions)
            )
            
        except Exception as e:
            self.logger.error(f"Error generating fix for {issue}: {e}")
            return None
    
    def estimate_risk(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> RiskLevel:
        """Estimate risk based on issue characteristics and strategy history."""
        base_risk = self._get_base_risk_level()
        
        # Adjust based on issue severity
        if issue.severity >= 4:
            base_risk = min(RiskLevel.CRITICAL, RiskLevel(base_risk.value + 1))
        elif issue.severity <= 1:
            base_risk = max(RiskLevel.LOW, RiskLevel(base_risk.value - 1))
        
        # Adjust based on success history
        recent_failures = len([h for h in self.failure_history[-10:] if h.get('issue_kind') == issue.kind])
        if recent_failures > 3:
            base_risk = min(RiskLevel.CRITICAL, RiskLevel(base_risk.value + 1))
        
        return base_risk
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Get dependencies - override in subclasses."""
        return []
    
    def learn_from_outcome(self, issue: Issue, patch: Patch, success: bool, feedback: str) -> None:
        """Learn from the outcome of applying this strategy."""
        outcome = {
            "timestamp": time.time(),
            "issue_kind": issue.kind,
            "confidence": patch.confidence,
            "risk_level": patch.risk_level.name,
            "success": success,
            "feedback": feedback,
            "actions_count": len(patch.actions)
        }
        
        if success:
            self.success_history.append(outcome)
            self._update_learned_patterns(issue, patch, True)
        else:
            self.failure_history.append(outcome)
            self._update_learned_patterns(issue, patch, False)
        
        # Keep only recent history
        self.success_history = self.success_history[-50:]
        self.failure_history = self.failure_history[-50:]
    
    def _calculate_confidence(self, issue: Issue, context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence based on issue and history."""
        base_confidence = 0.7  # Default confidence
        
        # Adjust based on success rate for this issue type
        issue_successes = [h for h in self.success_history if h.get('issue_kind') == issue.kind]
        issue_failures = [h for h in self.failure_history if h.get('issue_kind') == issue.kind]
        
        if issue_successes or issue_failures:
            total = len(issue_successes) + len(issue_failures)
            success_rate = len(issue_successes) / total
            base_confidence = 0.3 + (success_rate * 0.7)  # 0.3 to 1.0 range
        
        # Adjust based on issue evidence quality
        if issue.evidence and len(issue.evidence) > 2:
            base_confidence += 0.1
        
        return min(1.0, max(0.1, base_confidence))
    
    def _get_base_risk_level(self) -> RiskLevel:
        """Get the base risk level for this strategy type."""
        return RiskLevel.MEDIUM  # Override in subclasses
    
    def _generate_rollback_plan(self, actions: List[Action]) -> List[Action]:
        """Generate a rollback plan for the given actions."""
        rollback_actions = []
        
        for action in reversed(actions):  # Reverse order for rollback
            if action.type == "modify_file" and action.backup_content:
                rollback_actions.append(Action(
                    type="restore_file",
                    target=action.target,
                    content=action.backup_content,
                    metadata={"rollback_for": action.type}
                ))
            elif action.type == "create_file":
                rollback_actions.append(Action(
                    type="delete_file",
                    target=action.target,
                    metadata={"rollback_for": action.type}
                ))
        
        return rollback_actions
    
    def _generate_validation_tests(self, issue: Issue) -> List[str]:
        """Generate validation test commands."""
        tests = []
        
        if issue.file:
            # Basic syntax check
            tests.append(f"python -m py_compile {issue.file}")
            
            # Import check if it's a Python file
            if issue.file.endswith('.py'):
                module_name = issue.file.replace('.py', '').replace('/', '.')
                tests.append(f"python -c \"import {module_name}\"")
        
        return tests
    
    def _estimate_time(self, actions: List[Action]) -> float:
        """Estimate time to apply actions in seconds."""
        time_per_action = {
            "modify_file": 2.0,
            "create_file": 3.0,
            "delete_file": 1.0,
            "move_file": 2.5
        }
        
        total_time = sum(time_per_action.get(action.type, 2.0) for action in actions)
        return total_time
    
    def _identify_side_effects(self, issue: Issue, actions: List[Action]) -> List[str]:
        """Identify potential side effects of applying actions."""
        side_effects = []
        
        files_modified = [action.target for action in actions if action.type == "modify_file"]
        if len(files_modified) > 1:
            side_effects.append("Multiple file modifications")
        
        if any(action.type == "create_file" for action in actions):
            side_effects.append("New file creation")
        
        return side_effects
    
    def _update_learned_patterns(self, issue: Issue, patch: Patch, success: bool) -> None:
        """Update learned patterns from outcomes."""
        pattern_key = f"{issue.kind}_{patch.risk_level.name}"
        
        if pattern_key not in self.learned_patterns:
            self.learned_patterns[pattern_key] = {
                "successes": 0,
                "failures": 0,
                "avg_confidence": [],
                "effective_actions": []
            }
        
        pattern = self.learned_patterns[pattern_key]
        
        if success:
            pattern["successes"] += 1
            pattern["avg_confidence"].append(patch.confidence)
            pattern["effective_actions"].extend([a.type for a in patch.actions])
        else:
            pattern["failures"] += 1


class ImportResolutionStrategy(BaseFixStrategy):
    """Strategy for handling missing imports and import-related issues."""
    
    def __init__(self):
        super().__init__("ImportResolution")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle missing imports, import errors, and related issues."""
        return issue.kind in [
            "missing_symbol",
            "import_anxiety", 
            "unused_import",
            "import_error"
        ]
    
    def _get_base_risk_level(self) -> RiskLevel:
        return RiskLevel.LOW  # Import fixes are generally low risk
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to fix import issues."""
        actions = []
        
        if issue.kind == "missing_symbol":
            actions.extend(self._fix_missing_symbol(issue, context))
        elif issue.kind == "import_anxiety":
            actions.extend(self._fix_import_anxiety(issue, context))
        elif issue.kind == "unused_import":
            actions.extend(self._fix_unused_imports(issue, context))
        
        return actions
    
    def _fix_missing_symbol(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Fix missing symbol by adding appropriate import."""
        actions = []
        
        symbol = issue.symbol or "unknown_symbol"
        
        # Try to determine correct import
        common_imports = {
            'json': 'import json',
            'os': 'import os',
            'sys': 'import sys',
            're': 'import re',
            'datetime': 'from datetime import datetime',
            'Path': 'from pathlib import Path',
            'Optional': 'from typing import Optional',
            'List': 'from typing import List',
            'Dict': 'from typing import Dict'
        }
        
        if symbol in common_imports:
            import_statement = common_imports[symbol]
        else:
            # Generic import suggestion
            import_statement = f"# TODO: Add import for {symbol}\n# from module import {symbol}"
        
        # Read current file content for backup
        file_content = ""
        try:
            with open(issue.file, 'r') as f:
                file_content = f.read()
        except Exception:
            pass
        
        # Add import at top of file
        new_content = f"{import_statement}\n{file_content}"
        
        actions.append(Action(
            type="modify_file",
            target=issue.file,
            content=new_content,
            backup_content=file_content,
            metadata={"added_import": import_statement, "symbol": symbol}
        ))
        
        return actions
    
    def _fix_import_anxiety(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Fix import anxiety by organizing and cleaning imports."""
        actions = []
        
        if not issue.file:
            return actions
        
        try:
            with open(issue.file, 'r') as f:
                content = f.read()
            
            # Parse AST to analyze imports
            tree = ast.parse(content)
            
            # Collect all imports
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(f"import {alias.name}")
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ""
                    names = [alias.name for alias in node.names]
                    if len(names) == 1:
                        imports.append(f"from {module} import {names[0]}")
                    else:
                        imports.append(f"from {module} import ({', '.join(names)})")
            
            # Remove duplicates and sort
            unique_imports = list(set(imports))
            unique_imports.sort()
            
            # Create organized import section
            organized_imports = "\n".join(unique_imports)
            
            # Remove old imports and add organized ones
            new_content = self._replace_imports_in_content(content, organized_imports)
            
            actions.append(Action(
                type="modify_file", 
                target=issue.file,
                content=new_content,
                backup_content=content,
                metadata={"organized_imports": len(unique_imports)}
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not analyze imports in {issue.file}: {e}")
        
        return actions
    
    def _fix_unused_imports(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Remove unused imports."""
        actions = []
        
        if not issue.file or not issue.evidence:
            return actions
        
        unused_items = issue.evidence.get('unused_items', [])
        if not unused_items:
            return actions
        
        try:
            with open(issue.file, 'r') as f:
                lines = f.readlines()
            
            # Remove lines containing unused imports
            new_lines = []
            for line in lines:
                should_remove = False
                for unused in unused_items:
                    if f"import {unused}" in line or f"from {unused}" in line:
                        should_remove = True
                        break
                
                if not should_remove:
                    new_lines.append(line)
                else:
                    # Add comment about removal
                    new_lines.append(f"# Removed unused import: {line.strip()}\n")
            
            new_content = ''.join(new_lines)
            
            actions.append(Action(
                type="modify_file",
                target=issue.file, 
                content=new_content,
                backup_content=''.join(lines),
                metadata={"removed_imports": unused_items}
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not remove unused imports from {issue.file}: {e}")
        
        return actions
    
    def _replace_imports_in_content(self, content: str, new_imports: str) -> str:
        """Replace imports section in file content."""
        lines = content.split('\n')
        
        # Find the extent of import statements
        first_import = -1
        last_import = -1
        
        for i, line in enumerate(lines):
            stripped = line.strip()
            if (stripped.startswith('import ') or stripped.startswith('from ')) and not stripped.startswith('#'):
                if first_import == -1:
                    first_import = i
                last_import = i
        
        if first_import != -1:
            # Replace import section
            new_lines = lines[:first_import] + [new_imports] + lines[last_import+1:]
            return '\n'.join(new_lines)
        else:
            # Add imports at the beginning
            return f"{new_imports}\n{content}"
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Import issues should be fixed before other issues."""
        return []  # No dependencies - imports are usually foundational


class DuplicateMergeStrategy(BaseFixStrategy):
    """Strategy for consolidating semantic duplicates."""
    
    def __init__(self):
        super().__init__("DuplicateMerge")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle duplicate function and class issues."""
        return issue.kind in [
            "semantic_duplicate_function",
            "duplicate_function", 
            "duplicate_class"
        ]
    
    def _get_base_risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM  # Merging duplicates has moderate risk
    
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
        
        try:
            # Read secondary file to backup
            with open(secondary_file, 'r') as f:
                original_content = f.read()
            
            # Create deprecation wrapper for secondary
            deprecation_code = f'''# Deprecated: {secondary_name} merged into {primary_name}
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
    from {primary.get('file', 'module')} import {primary_name}
    return {primary_name}(*args, **kwargs)
'''
            
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
                    "type": "deprecation_wrapper"
                }
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not create merge action: {e}")
        
        return actions
    
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
            
            # Replace function
            new_lines = lines[:func_start] + [replacement] + lines[func_end:]
            return '\n'.join(new_lines)
        
        return content
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Duplicate merging should happen after import fixes."""
        return ["missing_symbol", "import_error"]


class PlaceholderImplementationStrategy(BaseFixStrategy):
    """Strategy for implementing placeholder functions and TODOs."""
    
    def __init__(self):
        super().__init__("PlaceholderImplementation")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle phantom functions and placeholder implementations."""
        return issue.kind in [
            "phantom_function",
            "placeholder",
            "todo_implementation",
            "stub_function"
        ]
    
    def _get_base_risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH  # Generated implementations are high risk
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate implementation for placeholder functions."""
        actions = []
        
        if not issue.file or not issue.symbol:
            return actions
        
        implementation = self._generate_implementation(issue, context)
        if not implementation:
            return actions
        
        try:
            with open(issue.file, 'r') as f:
                original_content = f.read()
            
            # Replace placeholder with implementation
            new_content = self._replace_placeholder_in_content(
                original_content,
                issue.symbol,
                implementation
            )
            
            actions.append(Action(
                type="modify_file",
                target=issue.file,
                content=new_content,
                backup_content=original_content,
                metadata={
                    "implemented_function": issue.symbol,
                    "type": "generated_implementation",
                    "needs_review": True
                }
            ))
            
        except Exception as e:
            self.logger.warning(f"Could not generate implementation: {e}")
        
        return actions
    
    def _generate_implementation(self, issue: Issue, context: Optional[Dict[str, Any]]) -> str:
        """Generate implementation based on function name and context."""
        func_name = issue.symbol
        
        # Analyze function name to determine pattern
        name_lower = func_name.lower()
        
        if name_lower.startswith(('get_', 'fetch_', 'retrieve_')):
            return self._generate_getter_implementation(func_name)
        elif name_lower.startswith(('set_', 'update_', 'save_')):
            return self._generate_setter_implementation(func_name)
        elif name_lower.startswith(('is_', 'has_', 'can_')):
            return self._generate_predicate_implementation(func_name)
        elif name_lower.startswith(('validate_', 'check_')):
            return self._generate_validator_implementation(func_name)
        elif 'process' in name_lower or 'handle' in name_lower:
            return self._generate_processor_implementation(func_name)
        else:
            return self._generate_generic_implementation(func_name)
    
    def _generate_getter_implementation(self, func_name: str) -> str:
        """Generate getter implementation."""
        return f'''def {func_name}(self, key=None):
    """
    Auto-generated getter implementation.
    
    TODO: Review and customize this implementation.
    Generated by TailChasingFixer PlaceholderImplementationStrategy.
    """
    if not hasattr(self, '_data_cache'):
        self._data_cache = {{}}
    
    if key is None:
        return self._data_cache
    
    return self._data_cache.get(key)
'''
    
    def _generate_setter_implementation(self, func_name: str) -> str:
        """Generate setter implementation."""
        return f'''def {func_name}(self, key, value):
    """
    Auto-generated setter implementation.
    
    TODO: Review and customize this implementation.
    Generated by TailChasingFixer PlaceholderImplementationStrategy.
    """
    if not hasattr(self, '_data_cache'):
        self._data_cache = {{}}
    
    old_value = self._data_cache.get(key)
    self._data_cache[key] = value
    
    return old_value
'''
    
    def _generate_predicate_implementation(self, func_name: str) -> str:
        """Generate predicate (boolean) implementation."""
        return f'''def {func_name}(self, value):
    """
    Auto-generated predicate implementation.
    
    TODO: Review and customize this implementation.
    Generated by TailChasingFixer PlaceholderImplementationStrategy.
    """
    if value is None:
        return False
    
    # TODO: Add specific validation logic
    # Currently returns True for any non-None value
    return bool(value)
'''
    
    def _generate_validator_implementation(self, func_name: str) -> str:
        """Generate validator implementation."""
        return f'''def {func_name}(self, data):
    """
    Auto-generated validator implementation.
    
    TODO: Review and customize this implementation.
    Generated by TailChasingFixer PlaceholderImplementationStrategy.
    
    Raises:
        ValueError: If validation fails
    """
    errors = []
    
    if data is None:
        errors.append("Data cannot be None")
    
    # TODO: Add specific validation rules here
    # Example validation rules:
    # if not isinstance(data, dict):
    #     errors.append("Data must be a dictionary")
    
    if errors:
        raise ValueError(f"Validation failed: {{', '.join(errors)}}")
    
    return True
'''
    
    def _generate_processor_implementation(self, func_name: str) -> str:
        """Generate processor implementation."""
        return f'''def {func_name}(self, *args, **kwargs):
    """
    Auto-generated processor implementation.
    
    TODO: Review and customize this implementation.
    Generated by TailChasingFixer PlaceholderImplementationStrategy.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"{func_name} called with {{len(args)}} args, {{len(kwargs)}} kwargs")
    
    try:
        # TODO: Implement actual processing logic
        result = {{
            "status": "processed",
            "timestamp": __import__('time').time(),
            "input_args": len(args),
            "input_kwargs": len(kwargs)
        }}
        
        logger.info(f"{func_name} completed successfully")
        return result
        
    except Exception as e:
        logger.error(f"Error in {func_name}: {{e}}")
        raise
'''
    
    def _generate_generic_implementation(self, func_name: str) -> str:
        """Generate generic implementation."""
        return f'''def {func_name}(self, *args, **kwargs):
    """
    Auto-generated implementation.
    
    TODO: Review and customize this implementation.
    Generated by TailChasingFixer PlaceholderImplementationStrategy.
    
    This is a generic implementation that logs calls and returns None.
    Please replace with actual business logic.
    """
    import logging
    logger = logging.getLogger(__name__)
    
    logger.warning(
        f"{func_name} is using auto-generated implementation - please review"
    )
    
    logger.debug(
        f"{func_name} called with args={{args}} kwargs={{kwargs}}"
    )
    
    # TODO: Implement actual functionality here
    return None
'''
    
    def _replace_placeholder_in_content(self, content: str, func_name: str, implementation: str) -> str:
        """Replace placeholder function with implementation."""
        lines = content.split('\n')
        
        # Find function definition
        func_start = -1
        func_end = -1
        indent_level = 0
        
        for i, line in enumerate(lines):
            if line.strip().startswith(f'def {func_name}('):
                func_start = i
                indent_level = len(line) - len(line.lstrip())
                
                # Find end of function (next def or class at same or lower indentation)
                for j in range(i + 1, len(lines)):
                    next_line = lines[j]
                    if next_line.strip():
                        next_indent = len(next_line) - len(next_line.lstrip())
                        if (next_indent <= indent_level and 
                            (next_line.strip().startswith('def ') or 
                             next_line.strip().startswith('class '))):
                            func_end = j
                            break
                
                if func_end == -1:
                    func_end = len(lines)
                
                break
        
        if func_start != -1:
            # Replace function with implementation
            new_lines = lines[:func_start] + [implementation] + lines[func_end:]
            return '\n'.join(new_lines)
        
        return content
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Implementation should happen after imports and merging."""
        return ["missing_symbol", "import_error", "semantic_duplicate_function"]
    
    def _generate_validation_tests(self, issue: Issue) -> List[str]:
        """Generate validation tests for generated implementations."""
        tests = super()._generate_validation_tests(issue)
        
        # Add specific tests for generated code
        if issue.symbol:
            tests.extend([
                f"# Test that {issue.symbol} can be called without errors",
                f"# python -c \"from {issue.file.replace('.py', '')} import {issue.symbol}; print('Import successful')\"",
                f"# TODO: Add unit tests for generated implementation of {issue.symbol}"
            ])
        
        return tests


class CircularDependencyBreaker(BaseFixStrategy):
    """Strategy for breaking circular import dependencies."""
    
    def __init__(self):
        super().__init__("CircularDependencyBreaker")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle circular import issues."""
        return issue.kind in [
            "circular_import",
            "circular_dependency",
            "import_cycle"
        ]
    
    def _get_base_risk_level(self) -> RiskLevel:
        return RiskLevel.HIGH  # Circular dependency fixes can be risky
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to break circular dependencies."""
        actions = []
        
        if not issue.evidence or 'cycle' not in issue.evidence:
            return actions
        
        cycle = issue.evidence['cycle']
        if len(cycle) < 2:
            return actions
        
        # Strategy 1: Move imports to function level
        actions.extend(self._create_local_import_actions(cycle, issue))
        
        # Strategy 2: Create interface module (if cycle is complex)
        if len(cycle) > 2:
            actions.extend(self._create_interface_module_actions(cycle, issue))
        
        return actions
    
    def _create_local_import_actions(self, cycle: List[str], issue: Issue) -> List[Action]:
        """Create actions to move imports to local scope."""
        actions = []
        
        if len(cycle) < 2:
            return actions
        
        file_a = cycle[0]
        file_b = cycle[1]
        
        try:
            with open(file_a, 'r') as f:
                content_a = f.read()
            
            # Find imports of file_b in file_a
            new_content = self._move_imports_to_functions(content_a, file_b)
            
            if new_content != content_a:
                actions.append(Action(
                    type="modify_file",
                    target=file_a,
                    content=new_content,
                    backup_content=content_a,
                    metadata={
                        "moved_imports_from": file_b,
                        "strategy": "local_imports"
                    }
                ))
        
        except Exception as e:
            self.logger.warning(f"Could not move imports in {file_a}: {e}")
        
        return actions
    
    def _create_interface_module_actions(self, cycle: List[str], issue: Issue) -> List[Action]:
        """Create interface module to break complex cycles."""
        actions = []
        
        # Create interface module name
        base_names = [f.replace('.py', '').replace('/', '_') for f in cycle[:2]]
        interface_name = f"{base_names[0]}_{base_names[1]}_interface.py"
        
        # Generate interface module content
        interface_content = f'''"""
Interface module to break circular dependency.

This module contains shared types, protocols, and abstract classes
that were causing circular imports between:
{', '.join(cycle)}

Generated by TailChasingFixer CircularDependencyBreaker.
"""

from typing import Protocol, runtime_checkable
from abc import ABC, abstractmethod


@runtime_checkable
class SharedProtocol(Protocol):
    """Protocol for shared interface between modules."""
    
    def shared_method(self) -> None:
        """Shared method that modules can implement."""
        ...


class SharedInterface(ABC):
    """Abstract interface for shared functionality."""
    
    @abstractmethod
    def process(self, data) -> None:
        """Process data - implement in concrete classes."""
        pass
    
    @abstractmethod
    def validate(self, data) -> bool:
        """Validate data - implement in concrete classes."""
        pass


# TODO: Move shared types and constants here from the circular modules
# TODO: Update the circular modules to import from this interface module
'''
        
        actions.append(Action(
            type="create_file",
            target=interface_name,
            content=interface_content,
            metadata={
                "breaks_cycle": cycle,
                "strategy": "interface_module",
                "requires_refactoring": True
            }
        ))
        
        return actions
    
    def _move_imports_to_functions(self, content: str, imported_module: str) -> str:
        """Move imports of specified module to function level."""
        lines = content.split('\n')
        new_lines = []
        import_line = None
        
        # Find and remove top-level import
        module_basename = imported_module.replace('.py', '').split('/')[-1]
        
        for line in lines:
            stripped = line.strip()
            if (f'from {module_basename} import' in stripped or 
                f'import {module_basename}' in stripped):
                import_line = stripped
                new_lines.append(f'# Moved to function level to break circular import: {stripped}')
            else:
                new_lines.append(line)
        
        if import_line:
            # Find functions that might need the import
            content_str = '\n'.join(new_lines)
            
            # Look for function definitions and add local import
            function_pattern = r'def\s+(\w+)\s*\('
            functions = re.findall(function_pattern, content_str)
            
            if functions:
                # Add import to first function as example
                new_content = re.sub(
                    r'(def\s+' + functions[0] + r'\s*\([^)]*\):\s*\n)(.*?)(\n)',
                    rf'\1    # Local import to break circular dependency\n    {import_line}\2\3',
                    content_str,
                    flags=re.DOTALL
                )
                return new_content
        
        return content
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Circular dependency breaking has no dependencies."""
        return []
    
    def _generate_validation_tests(self, issue: Issue) -> List[str]:
        """Generate validation tests for circular dependency fixes."""
        tests = super()._generate_validation_tests(issue)
        
        if issue.evidence and 'cycle' in issue.evidence:
            cycle = issue.evidence['cycle']
            for module in cycle:
                tests.append(f"python -c \"import {module.replace('.py', '')}\"")
        
        return tests


class AsyncSyncMismatchFixer(BaseFixStrategy):
    """Strategy for fixing async/await mismatches."""
    
    def __init__(self):
        super().__init__("AsyncSyncMismatchFixer")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle async/sync mismatches."""
        return issue.kind in [
            "async_sync_mismatch",
            "missing_await",
            "unnecessary_await",
            "async_in_sync_context"
        ]
    
    def _get_base_risk_level(self) -> RiskLevel:
        return RiskLevel.MEDIUM  # Async fixes have moderate risk
    
    def _generate_fix_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to fix async/sync mismatches."""
        actions = []
        
        if not issue.file:
            return actions
        
        try:
            with open(issue.file, 'r') as f:
                original_content = f.read()
            
            new_content = original_content
            
            if issue.kind == "missing_await":
                new_content = self._add_missing_await(original_content, issue)
            elif issue.kind == "unnecessary_await":
                new_content = self._remove_unnecessary_await(original_content, issue)
            elif issue.kind == "async_in_sync_context":
                new_content = self._fix_async_in_sync_context(original_content, issue)
            
            if new_content != original_content:
                actions.append(Action(
                    type="modify_file",
                    target=issue.file,
                    content=new_content,
                    backup_content=original_content,
                    metadata={
                        "async_fix_type": issue.kind,
                        "line": issue.line
                    }
                ))
        
        except Exception as e:
            self.logger.warning(f"Could not fix async issue in {issue.file}: {e}")
        
        return actions
    
    def _add_missing_await(self, content: str, issue: Issue) -> str:
        """Add missing await keywords."""
        lines = content.split('\n')
        
        if issue.line and 0 < issue.line <= len(lines):
            line = lines[issue.line - 1]
            
            # Look for async function calls without await
            async_call_pattern = r'(\s*)(\w+\([^)]*\))'
            
            def add_await(match):
                indent = match.group(1)
                call = match.group(2)
                return f"{indent}await {call}"
            
            new_line = re.sub(async_call_pattern, add_await, line)
            if new_line != line:
                new_line += "  # Added await - please verify this is correct"
                lines[issue.line - 1] = new_line
        
        return '\n'.join(lines)
    
    def _remove_unnecessary_await(self, content: str, issue: Issue) -> str:
        """Remove unnecessary await keywords."""
        lines = content.split('\n')
        
        if issue.line and 0 < issue.line <= len(lines):
            line = lines[issue.line - 1]
            
            # Remove await from non-async calls
            new_line = re.sub(r'\s*await\s+(\w+\([^)]*\))', r'\1', line)
            if new_line != line:
                new_line += "  # Removed await - please verify this is correct"
                lines[issue.line - 1] = new_line
        
        return '\n'.join(lines)
    
    def _fix_async_in_sync_context(self, content: str, issue: Issue) -> str:
        """Fix async function used in sync context."""
        lines = content.split('\n')
        
        if issue.line and 0 < issue.line <= len(lines):
            line = lines[issue.line - 1]
            
            # Add asyncio.run() wrapper for async calls in sync context
            async_call_pattern = r'(\s*)(\w+\([^)]*\))'
            
            def wrap_with_asyncio_run(match):
                indent = match.group(1)
                call = match.group(2)
                return f"{indent}import asyncio; asyncio.run({call})"
            
            new_line = re.sub(async_call_pattern, wrap_with_asyncio_run, line)
            if new_line != line:
                new_line += "  # Wrapped with asyncio.run - please verify"
                lines[issue.line - 1] = new_line
        
        return '\n'.join(lines)
    
    def get_dependencies(self, issue: Issue) -> List[str]:
        """Async fixes should happen after import resolution."""
        return ["missing_symbol", "import_error"]


@dataclass
class StrategyRanking:
    """Ranking information for a strategy-issue pair."""
    strategy: BaseFixStrategy
    confidence: float
    risk_level: RiskLevel
    estimated_time: float
    success_rate: float
    dependencies_satisfied: bool


class StrategySelector:
    """
    Selects the best fix strategies for issues based on various criteria.
    
    Features:
    - Ranks strategies by confidence and success history
    - Considers dependencies between fixes
    - Learns from success/failure outcomes
    - Manages strategy application order
    """
    
    def __init__(self):
        self.strategies: List[BaseFixStrategy] = [
            ImportResolutionStrategy(),
            DuplicateMergeStrategy(),
            PlaceholderImplementationStrategy(),
            CircularDependencyBreaker(),
            AsyncSyncMismatchFixer()
        ]
        self.logger = get_logger("strategy_selector")
        self.application_history: List[Dict[str, Any]] = []
    
    def select_strategies(self, 
                         issues: List[Issue], 
                         context: Optional[Dict[str, Any]] = None) -> List[Tuple[Issue, StrategyRanking]]:
        """
        Select and rank the best strategies for a list of issues.
        
        Returns list of (issue, strategy_ranking) pairs sorted by priority.
        """
        rankings = []
        
        # Get all applicable strategy-issue pairs
        for issue in issues:
            for strategy in self.strategies:
                if strategy.can_handle(issue):
                    ranking = self._rank_strategy_for_issue(strategy, issue, context)
                    if ranking.confidence > 0.3:  # Minimum confidence threshold
                        rankings.append((issue, ranking))
        
        # Sort by priority (dependency order, confidence, risk)
        sorted_rankings = self._sort_by_priority(rankings, context)
        
        return sorted_rankings
    
    def _rank_strategy_for_issue(self, 
                                strategy: BaseFixStrategy, 
                                issue: Issue, 
                                context: Optional[Dict[str, Any]]) -> StrategyRanking:
        """Rank a strategy for a specific issue."""
        
        # Calculate success rate for this strategy-issue combination
        success_rate = self._calculate_success_rate(strategy, issue)
        
        # Get confidence from strategy
        patch = strategy.propose_fix(issue, context)
        confidence = patch.confidence if patch else 0.0
        risk_level = patch.risk_level if patch else RiskLevel.HIGH
        estimated_time = patch.estimated_time if patch else 10.0
        
        # Check if dependencies are satisfied
        dependencies_satisfied = self._check_dependencies_satisfied(strategy, issue, context)
        
        return StrategyRanking(
            strategy=strategy,
            confidence=confidence,
            risk_level=risk_level,
            estimated_time=estimated_time,
            success_rate=success_rate,
            dependencies_satisfied=dependencies_satisfied
        )
    
    def _calculate_success_rate(self, strategy: BaseFixStrategy, issue: Issue) -> float:
        """Calculate success rate for strategy on this issue type."""
        successes = [h for h in strategy.success_history if h.get('issue_kind') == issue.kind]
        failures = [h for h in strategy.failure_history if h.get('issue_kind') == issue.kind]
        
        total = len(successes) + len(failures)
        if total == 0:
            return 0.7  # Default success rate for new combinations
        
        return len(successes) / total
    
    def _check_dependencies_satisfied(self, 
                                     strategy: BaseFixStrategy, 
                                     issue: Issue, 
                                     context: Optional[Dict[str, Any]]) -> bool:
        """Check if strategy dependencies are satisfied."""
        dependencies = strategy.get_dependencies(issue)
        
        if not dependencies:
            return True
        
        # Check if dependent issue types have been resolved
        # This would require access to current issue state
        # For now, assume dependencies are satisfied
        return True
    
    def _sort_by_priority(self, 
                         rankings: List[Tuple[Issue, StrategyRanking]], 
                         context: Optional[Dict[str, Any]]) -> List[Tuple[Issue, StrategyRanking]]:
        """Sort strategy rankings by priority."""
        
        def priority_key(item: Tuple[Issue, StrategyRanking]) -> Tuple[bool, int, float, float]:
            issue, ranking = item
            
            # Priority factors (lower values = higher priority):
            # 1. Dependencies satisfied (False = 0, True = 1, so False comes first)
            # 2. Risk level (lower risk first) 
            # 3. Negative confidence (higher confidence first)
            # 4. Estimated time (faster fixes first)
            
            return (
                not ranking.dependencies_satisfied,  # Dependencies first
                ranking.risk_level.value,           # Lower risk first
                -ranking.confidence,                # Higher confidence first  
                ranking.estimated_time              # Faster fixes first
            )
        
        return sorted(rankings, key=priority_key)
    
    def record_application_outcome(self, 
                                  issue: Issue, 
                                  strategy: BaseFixStrategy,
                                  patch: Patch,
                                  success: bool, 
                                  feedback: str) -> None:
        """Record the outcome of applying a strategy."""
        
        # Record in strategy's own history
        strategy.learn_from_outcome(issue, patch, success, feedback)
        
        # Record in selector's history
        outcome = {
            "timestamp": time.time(),
            "issue_kind": issue.kind,
            "strategy_name": strategy.name,
            "success": success,
            "confidence": patch.confidence,
            "risk_level": patch.risk_level.name,
            "estimated_time": patch.estimated_time,
            "actual_feedback": feedback
        }
        
        self.application_history.append(outcome)
        
        # Keep only recent history
        self.application_history = self.application_history[-200:]
        
        self.logger.info(
            f"Recorded {strategy.name} outcome for {issue.kind}: "
            f"{'SUCCESS' if success else 'FAILURE'}"
        )
    
    def get_strategy_statistics(self) -> Dict[str, Any]:
        """Get statistics about strategy performance."""
        stats = {}
        
        for strategy in self.strategies:
            strategy_stats = {
                "name": strategy.name,
                "total_successes": len(strategy.success_history),
                "total_failures": len(strategy.failure_history),
                "success_rate": 0.0,
                "avg_confidence": 0.0,
                "learned_patterns": len(strategy.learned_patterns)
            }
            
            total_attempts = len(strategy.success_history) + len(strategy.failure_history)
            if total_attempts > 0:
                strategy_stats["success_rate"] = len(strategy.success_history) / total_attempts
                
                confidences = [h.get("confidence", 0) for h in strategy.success_history + strategy.failure_history]
                strategy_stats["avg_confidence"] = sum(confidences) / len(confidences)
            
            stats[strategy.name] = strategy_stats
        
        # Overall statistics
        stats["overall"] = {
            "total_applications": len(self.application_history),
            "overall_success_rate": len([h for h in self.application_history if h["success"]]) / max(1, len(self.application_history))
        }
        
        return stats
    
    def recommend_strategy_improvements(self) -> List[str]:
        """Recommend improvements based on performance data."""
        recommendations = []
        
        for strategy in self.strategies:
            total_attempts = len(strategy.success_history) + len(strategy.failure_history)
            
            if total_attempts < 5:
                recommendations.append(
                    f"Strategy '{strategy.name}' needs more usage data (only {total_attempts} attempts)"
                )
            
            elif total_attempts > 10:
                success_rate = len(strategy.success_history) / total_attempts
                
                if success_rate < 0.4:
                    recommendations.append(
                        f"Strategy '{strategy.name}' has low success rate ({success_rate:.1%}) - consider improvement"
                    )
                elif success_rate > 0.8:
                    recommendations.append(
                        f"Strategy '{strategy.name}' is performing well ({success_rate:.1%}) - consider expanding scope"
                    )
        
        return recommendations