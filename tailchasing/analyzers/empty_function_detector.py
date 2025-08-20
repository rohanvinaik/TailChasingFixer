"""
Empty Function Detector - Identifies stub functions and empty implementations.

This analyzer detects functions that are either completely empty, only contain
pass/NotImplementedError, or have trivial implementations that suggest they
were created as placeholders but never completed.
"""

import ast
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..core.issues import Issue
from .base import BaseAnalyzer


@dataclass
class EmptyFunctionPattern:
    """Represents an empty or stub function pattern."""
    
    function_name: str
    file_path: str
    line_number: int
    pattern_type: str  # 'empty', 'pass_only', 'not_implemented', 'trivial_return'
    body_lines: int
    has_docstring: bool
    is_abstract: bool
    is_protocol: bool
    decorator_hints: List[str]  # @abstractmethod, @property, etc.


class EmptyFunctionDetector(BaseAnalyzer):
    """
    Detects empty, stub, and placeholder functions that indicate incomplete implementations.
    
    Patterns detected:
    1. Completely empty functions (only docstring)
    2. Functions with only 'pass'
    3. Functions raising NotImplementedError
    4. Functions with trivial returns (return None, return {}, etc.)
    5. Functions with only ellipsis (...)
    """
    
    def __init__(self):
        super().__init__()
        self.empty_patterns: List[EmptyFunctionPattern] = []
        self.current_file = ""
        self.is_protocol_file = False
        self.is_abstract_class = False
        
    def analyze(self, context) -> List[Issue]:
        """Analyze codebase for empty function patterns."""
        issues = []
        
        # Track patterns across files for clustering
        pattern_clusters: Dict[str, List[EmptyFunctionPattern]] = {
            'empty': [],
            'pass_only': [],
            'not_implemented': [],
            'trivial_return': [],
            'ellipsis': []
        }
        
        for file_path, ast_tree in context.ast_index.items():
            self.current_file = file_path
            self.is_protocol_file = self._is_protocol_file(ast_tree)
            
            # Find all empty/stub functions
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    pattern = self._analyze_function(node)
                    if pattern and not self._should_ignore_pattern(pattern):
                        self.empty_patterns.append(pattern)
                        pattern_clusters[pattern.pattern_type].append(pattern)
                        
                elif isinstance(node, ast.ClassDef):
                    self.is_abstract_class = self._is_abstract_class(node)
        
        # Generate issues from patterns
        issues.extend(self._generate_issues_from_patterns(pattern_clusters))
        
        # Add summary issue if many empty functions found
        total_empty = len(self.empty_patterns)
        if total_empty > 10:
            issues.append(Issue(
                type="empty_function_epidemic",
                severity="HIGH",
                message=f"Found {total_empty} empty/stub functions indicating incomplete implementation",
                file_path="<global>",
                line_number=0,
                details={
                    "total_count": total_empty,
                    "breakdown": {
                        k: len(v) for k, v in pattern_clusters.items()
                    },
                    "most_affected_files": self._get_most_affected_files()
                }
            ))
        
        return issues
    
    def _analyze_function(self, node: ast.FunctionDef) -> Optional[EmptyFunctionPattern]:
        """Analyze a function node for empty/stub patterns."""
        # Extract basic info
        func_name = node.name
        line_number = node.lineno
        decorators = [d.id if isinstance(d, ast.Name) else 
                     d.attr if isinstance(d, ast.Attribute) else None 
                     for d in node.decorator_list]
        
        # Check if it's legitimately abstract
        is_abstract = 'abstractmethod' in decorators or 'abc.abstractmethod' in str(decorators)
        is_property = 'property' in decorators
        
        # Analyze function body
        body = node.body
        has_docstring = (len(body) > 0 and 
                        isinstance(body[0], ast.Expr) and 
                        isinstance(body[0].value, (ast.Str, ast.Constant)))
        
        # Skip docstring if present
        actual_body = body[1:] if has_docstring else body
        body_lines = len(actual_body)
        
        # Detect pattern type
        pattern_type = None
        
        if body_lines == 0:
            pattern_type = 'empty'
        elif body_lines == 1:
            stmt = actual_body[0]
            
            if isinstance(stmt, ast.Pass):
                pattern_type = 'pass_only'
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant) and stmt.value.value is ...:
                pattern_type = 'ellipsis'
            elif isinstance(stmt, ast.Raise):
                if self._is_not_implemented_error(stmt):
                    pattern_type = 'not_implemented'
            elif isinstance(stmt, ast.Return):
                if self._is_trivial_return(stmt):
                    pattern_type = 'trivial_return'
        
        if pattern_type:
            return EmptyFunctionPattern(
                function_name=func_name,
                file_path=self.current_file,
                line_number=line_number,
                pattern_type=pattern_type,
                body_lines=body_lines,
                has_docstring=has_docstring,
                is_abstract=is_abstract,
                is_protocol=self.is_protocol_file,
                decorator_hints=decorators
            )
        
        return None
    
    def _is_not_implemented_error(self, raise_node: ast.Raise) -> bool:
        """Check if a raise statement is NotImplementedError."""
        if not raise_node.exc:
            return False
            
        if isinstance(raise_node.exc, ast.Call):
            if isinstance(raise_node.exc.func, ast.Name):
                return raise_node.exc.func.id == 'NotImplementedError'
        elif isinstance(raise_node.exc, ast.Name):
            return raise_node.exc.id == 'NotImplementedError'
            
        return False
    
    def _is_trivial_return(self, return_node: ast.Return) -> bool:
        """Check if a return statement is trivial."""
        if not return_node.value:
            return True  # return None
            
        value = return_node.value
        
        # Check for literal returns
        if isinstance(value, ast.Constant):
            return value.value in (None, True, False, 0, 1, "", [], {})
        elif isinstance(value, (ast.List, ast.Dict, ast.Set)) and len(value.elts) == 0:
            return True  # Empty collection
        elif isinstance(value, ast.NameConstant):
            return value.value in (None, True, False)
            
        return False
    
    def _is_protocol_file(self, tree: ast.AST) -> bool:
        """Check if file contains Protocol classes."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and 'typing' in node.module:
                    for alias in node.names:
                        if alias.name == 'Protocol':
                            return True
        return False
    
    def _is_abstract_class(self, class_node: ast.ClassDef) -> bool:
        """Check if class is abstract (inherits from ABC)."""
        for base in class_node.bases:
            if isinstance(base, ast.Name) and base.id in ('ABC', 'ABCMeta'):
                return True
            elif isinstance(base, ast.Attribute) and base.attr in ('ABC', 'ABCMeta'):
                return True
        return False
    
    def _should_ignore_pattern(self, pattern: EmptyFunctionPattern) -> bool:
        """Determine if pattern should be ignored as legitimate."""
        # Ignore abstract methods
        if pattern.is_abstract:
            return True
            
        # Ignore protocol methods
        if pattern.is_protocol:
            return True
            
        # Ignore special methods in protocols/ABCs
        if pattern.function_name.startswith('__') and pattern.function_name.endswith('__'):
            if self.is_abstract_class or self.is_protocol_file:
                return True
                
        # Ignore test stubs
        if 'test' in pattern.file_path.lower() and pattern.function_name.startswith('test_'):
            if pattern.pattern_type == 'pass_only':
                return True  # Legitimate test placeholder
                
        return False
    
    def _generate_issues_from_patterns(self, pattern_clusters: Dict[str, List[EmptyFunctionPattern]]) -> List[Issue]:
        """Generate issues from detected patterns."""
        issues = []
        
        # NotImplementedError patterns (highest priority)
        for pattern in pattern_clusters['not_implemented']:
            issues.append(Issue(
                type="not_implemented_stub",
                severity="HIGH",
                message=f"Function '{pattern.function_name}' raises NotImplementedError",
                file_path=pattern.file_path,
                line_number=pattern.line_number,
                details={
                    "pattern_type": "not_implemented",
                    "has_docstring": pattern.has_docstring,
                    "recommendation": "Implement the function or remove if unused"
                }
            ))
        
        # Empty functions (no body at all)
        for pattern in pattern_clusters['empty']:
            if not pattern.has_docstring:
                issues.append(Issue(
                    type="empty_function",
                    severity="MEDIUM",
                    message=f"Function '{pattern.function_name}' is completely empty",
                    file_path=pattern.file_path,
                    line_number=pattern.line_number,
                    details={
                        "pattern_type": "empty",
                        "recommendation": "Add implementation or remove"
                    }
                ))
        
        # Pass-only functions
        for pattern in pattern_clusters['pass_only']:
            issues.append(Issue(
                type="pass_only_stub",
                severity="MEDIUM",
                message=f"Function '{pattern.function_name}' contains only 'pass'",
                file_path=pattern.file_path,
                line_number=pattern.line_number,
                details={
                    "pattern_type": "pass_only",
                    "has_docstring": pattern.has_docstring,
                    "recommendation": "Implement or mark as abstract"
                }
            ))
        
        # Trivial returns
        for pattern in pattern_clusters['trivial_return']:
            issues.append(Issue(
                type="trivial_return_stub",
                severity="LOW",
                message=f"Function '{pattern.function_name}' has trivial return",
                file_path=pattern.file_path,
                line_number=pattern.line_number,
                details={
                    "pattern_type": "trivial_return",
                    "recommendation": "Implement actual logic or remove"
                }
            ))
        
        return issues
    
    def _get_most_affected_files(self) -> List[Tuple[str, int]]:
        """Get files with most empty functions."""
        file_counts: Dict[str, int] = {}
        for pattern in self.empty_patterns:
            file_counts[pattern.file_path] = file_counts.get(pattern.file_path, 0) + 1
        
        # Return top 5 files
        sorted_files = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
        return sorted_files[:5]