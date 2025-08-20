"""
Enhanced placeholder and phantom function analyzer with better detection and suggestions.

This analyzer detects various placeholder patterns and provides actionable fixes.
"""

import ast
import re
from typing import List, Optional, Iterable, Dict, Set
from dataclasses import dataclass
from enum import Enum

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


class PlaceholderSeverity(Enum):
    """Severity levels for placeholder issues."""
    HIGH = 3      # Called elsewhere, critical logic implied
    MEDIUM = 2    # Important logic implied by name
    LOW = 1       # Never called or test file


@dataclass
class PlaceholderPattern:
    """Represents a detected placeholder pattern."""
    pattern_type: str
    severity: PlaceholderSeverity
    description: str
    suggestions: List[str]
    confidence: float = 0.9


class EnhancedPlaceholderAnalyzer(BaseAnalyzer):
    """
    Enhanced detector for placeholder and phantom function implementations.
    
    Detects:
    - Functions with only 'pass' or 'return None'
    - Low-complexity functions with names implying logic
    - Functions that just return constants
    - Empty exception handlers
    - Stub methods in non-abstract classes
    """
    
    name = "enhanced_placeholders"
    
    # Patterns that imply logic should exist
    LOGIC_PATTERNS = [
        r'_?check_',
        r'_?validate_',
        r'_?verify_',
        r'_?detect_',
        r'_?analyze_',
        r'_?process_',
        r'_?compute_',
        r'_?calculate_',
        r'_?parse_',
        r'_?handle_',
        r'_?filter_',
        r'_?transform_',
        r'_?convert_',
        r'_?ensure_',
        r'_?enforce_',
        r'is_valid',
        r'should_',
        r'can_',
        r'has_',
        r'needs_'
    ]
    
    # Interface/protocol method names that are often stubs
    INTERFACE_METHODS = {
        '__enter__', '__exit__', '__iter__', '__next__',
        '__getitem__', '__setitem__', '__delitem__',
        '__len__', '__contains__', '__repr__', '__str__'
    }
    
    def __init__(self):
        super().__init__()
        self.function_calls: Dict[str, Set[str]] = {}  # Track function calls
        self.class_inheritance: Dict[str, List[str]] = {}  # Track class bases
        
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Find enhanced placeholder patterns."""
        # First pass: collect call information
        self._collect_call_information(ctx)
        
        # Second pass: detect placeholders with context
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            visitor = EnhancedPlaceholderVisitor(file, ctx, self.function_calls)
            visitor.visit(tree)
            
            for issue in visitor.issues:
                yield issue


    def _collect_call_information(self, ctx: AnalysisContext):
        """Collect information about function calls across the codebase."""
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            collector = CallCollector(file)
            collector.visit(tree)
            
            for func_name, callers in collector.function_calls.items():
                if func_name not in self.function_calls:
                    self.function_calls[func_name] = set()
                self.function_calls[func_name].update(callers)


class CallCollector(ast.NodeVisitor):
    """Collects function call information."""
    
    def __init__(self, file: str):
        self.file = file
        self.function_calls: Dict[str, Set[str]] = {}
        self.current_function = None
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef):
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        old_func = self.current_function
        if self.current_class:
            self.current_function = f"{self.current_class}.{node.name}"
        else:
            self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_func
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        self.visit_FunctionDef(node)
        
    def visit_Call(self, node: ast.Call):
        """Track function calls."""
        if self.current_function:
            called_name = None
            
            # Direct function call
            if isinstance(node.func, ast.Name):
                called_name = node.func.id
            # Method call
            elif isinstance(node.func, ast.Attribute):
                called_name = node.func.attr
                
            if called_name:
                if called_name not in self.function_calls:
                    self.function_calls[called_name] = set()
                self.function_calls[called_name].add(self.current_function)
                
        self.generic_visit(node)


class EnhancedPlaceholderVisitor(ast.NodeVisitor):
    """Enhanced visitor for finding placeholder implementations."""
    
    def __init__(self, file: str, ctx: AnalysisContext, function_calls: Dict[str, Set[str]]):
        self.file = file
        self.ctx = ctx
        self.function_calls = function_calls
        self.issues: List[Issue] = []
        self.current_class = None
        self.class_bases: List[str] = []
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class context and inheritance."""
        old_class = self.current_class
        old_bases = self.class_bases
        
        self.current_class = node.name
        self.class_bases = self._extract_base_names(node)
        
        self.generic_visit(node)
        
        self.current_class = old_class
        self.class_bases = old_bases
        
    def _extract_base_names(self, node: ast.ClassDef) -> List[str]:
        """Extract base class names."""
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(base.attr)
        return bases
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function for enhanced placeholder patterns."""
        self._check_function(node)
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Check async function for placeholder patterns."""
        self._check_function(node)
        self.generic_visit(node)
        
    def _check_function(self, node: ast.FunctionDef):
        """Enhanced function checking with pattern detection."""
        # Get full name
        if self.current_class:
            full_name = f"{self.current_class}.{node.name}"
        else:
            full_name = node.name
            
        # Check if explicitly allowed
        if self.ctx.is_placeholder_allowed(full_name):
            return
            
        # Skip abstract methods
        if self._is_abstract_method(node):
            return
            
        # Detect placeholder pattern
        pattern = self._detect_placeholder_pattern(node)
        if pattern:
            # Calculate severity based on usage
            severity = self._calculate_severity(node, pattern)
            
            # Generate fix suggestions
            suggestions = self._generate_suggestions(node, pattern)
            
            # Create issue
            self._add_enhanced_issue(node, pattern, severity, suggestions)
    
    def _is_abstract_method(self, node: ast.FunctionDef) -> bool:
        """Check if method is abstract."""
        # Check decorators
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name):
                if 'abstract' in decorator.id.lower():
                    return True
            elif isinstance(decorator, ast.Attribute):
                if 'abstract' in decorator.attr.lower():
                    return True
                    
        # Check if class inherits from ABC
        if 'ABC' in self.class_bases or 'Protocol' in self.class_bases:
            # These often have stub methods intentionally
            return True
            
        return False
    
    def _detect_placeholder_pattern(self, node: ast.FunctionDef) -> Optional[PlaceholderPattern]:
        """Detect various placeholder patterns."""
        # Check for TODO-only first (more specific)
        if self._is_todo_only(node):
            return PlaceholderPattern(
                pattern_type="todo_only",
                severity=PlaceholderSeverity.MEDIUM,
                description="Function contains only TODO comment",
                suggestions=[]
            )
            
        # Check for pass-only
        if self._is_pass_only(node):
            return PlaceholderPattern(
                pattern_type="pass_only",
                severity=PlaceholderSeverity.MEDIUM,
                description="Function contains only 'pass'",
                suggestions=[]
            )
            
        # Check for trivial return
        trivial_return = self._check_trivial_return(node)
        if trivial_return:
            return trivial_return
            
        # Check for NotImplementedError
        if self._is_not_implemented(node):
            return PlaceholderPattern(
                pattern_type="not_implemented",
                severity=PlaceholderSeverity.LOW,
                description="Function raises NotImplementedError",
                suggestions=[]
            )
            
        # Check for low complexity with logic-implying name
        if self._has_logic_implying_name(node.name):
            complexity = self._calculate_complexity(node)
            if complexity <= 1:  # Only flag truly trivial functions
                return PlaceholderPattern(
                    pattern_type="low_complexity_logic",
                    severity=PlaceholderSeverity.HIGH,
                    description=f"Function '{node.name}' implies logic but has complexity {complexity}",
                    suggestions=[]
                )
                
        # Check for ellipsis
        if self._is_ellipsis_only(node):
            return PlaceholderPattern(
                pattern_type="ellipsis_only",
                severity=PlaceholderSeverity.MEDIUM,
                description="Function contains only ellipsis (...)",
                suggestions=[]
            )
            
        return None
    
    def _is_pass_only(self, node: ast.FunctionDef) -> bool:
        """Check if function contains only pass."""
        body = self._skip_docstring(node.body)
        return len(body) == 1 and isinstance(body[0], ast.Pass)
    
    def _check_trivial_return(self, node: ast.FunctionDef) -> Optional[PlaceholderPattern]:
        """Check for trivial return patterns."""
        body = self._skip_docstring(node.body)
        
        if len(body) != 1 or not isinstance(body[0], ast.Return):
            return None
            
        ret_val = body[0].value
        
        # return None or no return value
        if ret_val is None:
            return PlaceholderPattern(
                pattern_type="return_none",
                severity=PlaceholderSeverity.MEDIUM,
                description="Function only returns None",
                suggestions=[]
            )
            
        # return constant
        if isinstance(ret_val, ast.Constant):
            const_val = ret_val.value
            return PlaceholderPattern(
                pattern_type="return_constant",
                severity=PlaceholderSeverity.LOW,
                description=f"Function only returns constant: {repr(const_val)}",
                suggestions=[]
            )
            
        # return empty collection
        if isinstance(ret_val, (ast.List, ast.Tuple)) and not ret_val.elts:
            return PlaceholderPattern(
                pattern_type="return_empty",
                severity=PlaceholderSeverity.LOW,
                description="Function only returns empty collection",
                suggestions=[]
            )
            
        if isinstance(ret_val, ast.Dict) and not ret_val.keys:
            return PlaceholderPattern(
                pattern_type="return_empty",
                severity=PlaceholderSeverity.LOW,
                description="Function only returns empty dict",
                suggestions=[]
            )
            
        return None
    
    def _is_not_implemented(self, node: ast.FunctionDef) -> bool:
        """Check if function raises NotImplementedError."""
        body = self._skip_docstring(node.body)
        
        if len(body) != 1 or not isinstance(body[0], ast.Raise):
            return False
            
        exc = body[0].exc
        if isinstance(exc, ast.Call):
            if isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                return True
        elif isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
            
        return False
    
    def _is_todo_only(self, node: ast.FunctionDef) -> bool:
        """Check if function contains only TODO."""
        body = node.body
        
        if not body:
            return False
            
        # Check docstring for TODO
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            docstring = body[0].value.value
            if isinstance(docstring, str) and 'todo' in docstring.lower():
                remaining = body[1:]
                # Function with TODO docstring and either nothing else or just pass
                if not remaining or (len(remaining) == 1 and isinstance(remaining[0], ast.Pass)):
                    return True
                # Also check if it's just an empty return
                if len(remaining) == 1 and isinstance(remaining[0], ast.Return) and remaining[0].value is None:
                    return True
                    
        return False
    
    def _is_ellipsis_only(self, node: ast.FunctionDef) -> bool:
        """Check if function contains only ellipsis."""
        body = self._skip_docstring(node.body)
        
        if len(body) != 1:
            return False
            
        stmt = body[0]
        return (isinstance(stmt, ast.Expr) and 
                isinstance(stmt.value, ast.Constant) and
                stmt.value.value is ...)
    
    def _has_logic_implying_name(self, name: str) -> bool:
        """Check if function name implies logic should exist."""
        name_lower = name.lower()
        for pattern in EnhancedPlaceholderAnalyzer.LOGIC_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        return False
    
    def _calculate_complexity(self, node: ast.FunctionDef) -> int:
        """Calculate cyclomatic complexity of function."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            # Control flow statements add complexity
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # Each additional boolean operator adds complexity
                complexity += len(child.values) - 1
                
        return complexity
    
    def _skip_docstring(self, body: List[ast.stmt]) -> List[ast.stmt]:
        """Skip docstring if present."""
        if (body and isinstance(body[0], ast.Expr) and 
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
            return body[1:]
        return body
    
    def _calculate_severity(self, node: ast.FunctionDef, pattern: PlaceholderPattern) -> PlaceholderSeverity:
        """Calculate severity based on usage and context."""
        func_name = node.name
        
        # Check if function is called elsewhere
        is_called = False
        if func_name in self.function_calls:
            callers = self.function_calls[func_name]
            # Filter out self-calls
            external_callers = [c for c in callers if not c.endswith(f".{func_name}")]
            is_called = bool(external_callers)
        
        # Check if in test file
        is_test = 'test' in self.file.lower()
        
        # Check if name implies important logic
        implies_logic = self._has_logic_implying_name(func_name)
        
        # Determine severity
        if is_called and implies_logic:
            return PlaceholderSeverity.HIGH
        elif is_called or implies_logic:
            return PlaceholderSeverity.MEDIUM
        elif is_test:
            return PlaceholderSeverity.LOW
        else:
            return pattern.severity
    
    def _generate_suggestions(self, node: ast.FunctionDef, pattern: PlaceholderPattern) -> List[str]:
        """Generate specific fix suggestions based on pattern."""
        func_name = node.name
        suggestions = []
        
        if pattern.pattern_type in ["pass_only", "ellipsis_only", "todo_only"]:
            # Suggest NotImplementedError
            suggestions.append(
                f"Replace with: raise NotImplementedError(\"TODO: {func_name} needs implementation\")"
            )
            
        if pattern.pattern_type == "return_none":
            if self._has_logic_implying_name(func_name):
                suggestions.append(
                    f"Implement {func_name} logic or raise NotImplementedError if not ready"
                )
                
        if pattern.pattern_type == "low_complexity_logic":
            suggestions.append(
                f"Function name '{func_name}' implies logic - implement it or rename to clarify intent"
            )
        
        # Add logic-specific suggestions for functions with logic-implying names
        if self._has_logic_implying_name(func_name):
            suggestions.append(
                f"Implement the {func_name} logic that the name implies"
            )
            
        # Check if this looks like an interface method
        if func_name in EnhancedPlaceholderAnalyzer.INTERFACE_METHODS:
            suggestions.append(
                "Consider using Protocol or ABC for interface definitions"
            )
            
        # Check if in a base class context
        if self.current_class and not self.class_bases:
            suggestions.append(
                "If this is a base class, consider inheriting from ABC or Protocol"
            )
            
        # Add general suggestions
        suggestions.extend([
            "Remove the function if it's not needed",
            "Add to 'placeholders.allow' in config if this is intentional",
            "Document why this placeholder exists with a detailed comment"
        ])
        
        return suggestions
    
    def _add_enhanced_issue(self, node: ast.FunctionDef, pattern: PlaceholderPattern, 
                           severity: PlaceholderSeverity, suggestions: List[str]):
        """Add an enhanced placeholder issue."""
        if self.current_class:
            symbol = f"{self.current_class}.{node.name}"
        else:
            symbol = node.name
            
        # Map severity enum to integer
        severity_int = severity.value
        
        # Create detailed evidence
        evidence = {
            "function_name": node.name,
            "pattern_type": pattern.pattern_type,
            "description": pattern.description,
            "complexity": self._calculate_complexity(node),
            "is_called": symbol in self.function_calls,
            "implies_logic": self._has_logic_implying_name(node.name)
        }
        
        # Add caller information if available
        if symbol in self.function_calls:
            evidence["called_by"] = list(self.function_calls[symbol])[:5]  # Limit to 5
            
        issue = Issue(
            kind="enhanced_placeholder",
            message=f"{pattern.description} in {symbol}",
            severity=severity_int,
            file=self.file,
            line=safe_get_lineno(node),
            end_line=node.end_lineno,
            symbol=symbol,
            evidence=evidence,
            suggestions=suggestions,
            confidence=pattern.confidence
        )
        
        self.issues.append(issue)