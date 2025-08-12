"""Placeholder and phantom function analyzer."""

import ast
from typing import List, Optional, Iterable

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


class PlaceholderAnalyzer(BaseAnalyzer):
    """Detects placeholder and phantom function implementations."""
    
    name = "placeholders"
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Find placeholder functions."""
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            visitor = PlaceholderVisitor(file, ctx)
            visitor.visit(tree)
            
            for issue in visitor.issues:
                yield issue
                
                
class PlaceholderVisitor(ast.NodeVisitor):
    """Visits AST nodes to find placeholder implementations."""
    
    def __init__(self, file: str, ctx: AnalysisContext):
        self.file = file
        self.ctx = ctx
        self.issues: List[Issue] = []
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Check function for placeholder patterns."""
        self._check_function(node)
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Check async function for placeholder patterns."""
        self._check_function(node)
        self.generic_visit(node)
        
    def _check_function(self, node: ast.FunctionDef):
        """Check if a function is a placeholder."""
        # Get full name for checking against allowed placeholders
        if self.current_class:
            full_name = f"{self.current_class}.{node.name}"
        else:
            full_name = node.name
            
        # Check if this placeholder is explicitly allowed
        if self.ctx.is_placeholder_allowed(full_name):
            return
            
        # Check for various placeholder patterns
        if self._is_pass_only(node):
            self._add_placeholder_issue(node, "Pass-only function", 2)
        elif self._is_not_implemented(node):
            self._add_placeholder_issue(node, "NotImplementedError stub", 2)
        elif self._is_trivial_return(node):
            self._add_placeholder_issue(node, "Trivial return stub", 1)
        elif self._is_todo_comment_only(node):
            self._add_placeholder_issue(node, "TODO-only function", 2)
        elif self._is_ellipsis_only(node):
            self._add_placeholder_issue(node, "Ellipsis-only function", 2)
            
    def _is_pass_only(self, node: ast.FunctionDef) -> bool:
        """Check if function contains only a pass statement."""
        body = node.body
        
        # Skip docstring if present
        if (body and isinstance(body[0], ast.Expr) and 
            isinstance(body[0].value, ast.Constant)):
            body = body[1:]
            
        return len(body) == 1 and isinstance(body[0], ast.Pass)
        
    def _is_not_implemented(self, node: ast.FunctionDef) -> bool:
        """Check if function raises NotImplementedError."""
        body = node.body
        
        # Skip docstring
        if (body and isinstance(body[0], ast.Expr) and 
            isinstance(body[0].value, ast.Constant)):
            body = body[1:]
            
        if len(body) != 1:
            return False
            
        stmt = body[0]
        if not isinstance(stmt, ast.Raise):
            return False
            
        # Check if it's raising NotImplementedError
        exc = stmt.exc
        if isinstance(exc, ast.Call):
            if isinstance(exc.func, ast.Name) and exc.func.id == "NotImplementedError":
                return True
        elif isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
            
        return False
        
    def _is_trivial_return(self, node: ast.FunctionDef) -> bool:
        """Check if function only returns a constant."""
        body = node.body
        
        # Skip docstring
        if (body and isinstance(body[0], ast.Expr) and 
            isinstance(body[0].value, ast.Constant)):
            body = body[1:]
            
        if len(body) != 1:
            return False
            
        stmt = body[0]
        if not isinstance(stmt, ast.Return):
            return False
            
        # Check if returning a constant
        value = stmt.value
        if value is None:  # return without value
            return True
        if isinstance(value, ast.Constant):
            return True
        if isinstance(value, ast.List) and not value.elts:
            return True  # Empty list
        if isinstance(value, ast.Dict) and not value.keys:
            return True  # Empty dict
        if isinstance(value, ast.Tuple) and not value.elts:
            return True  # Empty tuple
            
        return False
        
    def _is_todo_comment_only(self, node: ast.FunctionDef) -> bool:
        """Check if function contains only TODO comments."""
        body = node.body
        
        # Must have at least one statement
        if not body:
            return False
            
        # Check if first statement is a docstring with TODO
        if isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            docstring = body[0].value.value
            if isinstance(docstring, str) and 'todo' in docstring.lower():
                # If only docstring or docstring + pass
                if len(body) == 1 or (len(body) == 2 and isinstance(body[1], ast.Pass)):
                    return True
                    
        return False
        
    def _is_ellipsis_only(self, node: ast.FunctionDef) -> bool:
        """Check if function contains only ellipsis."""
        body = node.body
        
        # Skip docstring
        if (body and isinstance(body[0], ast.Expr) and 
            isinstance(body[0].value, ast.Constant) and
            isinstance(body[0].value.value, str)):
            body = body[1:]
            
        if len(body) != 1:
            return False
            
        stmt = body[0]
        return (isinstance(stmt, ast.Expr) and 
                isinstance(stmt.value, ast.Constant) and
                stmt.value.value is ...)
        
    def _add_placeholder_issue(self, node: ast.FunctionDef, description: str, severity: int):
        """Add a placeholder issue."""
        if self.current_class:
            symbol = f"{self.current_class}.{node.name}"
        else:
            symbol = node.name
            
        # Check for test files - lower severity
        if 'test' in self.file.lower():
            severity = max(1, severity - 1)
            
        # Check if it's an abstract method (has decorator)
        if any(isinstance(d, ast.Name) and 'abstract' in d.id.lower() 
               for d in node.decorator_list):
            # Skip abstract methods
            return
            
        issue = Issue(
            kind="phantom_function",
            message=f"{description}: {symbol}",
            severity=severity,
            file=self.file,
            line=safe_get_lineno(node),
            end_line=node.end_lineno,
            symbol=symbol,
            evidence={
                "function_name": node.name,
                "placeholder_type": description.lower().replace(' ', '_')
            },
            suggestions=[
                "Implement the function or remove if not needed",
                "If this is intentional, add to 'placeholders.allow' in config",
                "Consider using abstract base classes for interface definitions"
            ],
            confidence=0.9
        )
        
        self.issues.append(issue)
