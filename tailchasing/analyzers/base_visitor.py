"""
Base visitor class for AST analyzers to avoid code duplication.

This module provides common visitor implementations that are shared
across multiple analyzers, reducing duplicate code in the codebase.
"""

import ast
from typing import Optional, Set, Dict, Any, List
from abc import ABC, abstractmethod

# Public API exports
__all__ = [
    'BaseASTVisitor',
    'CollectingVisitor', 
    'AnalysisVisitor'
]


class BaseASTVisitor(ast.NodeVisitor, ABC):
    """
    Base class for AST visitors with common functionality.
    
    Provides shared implementations of visitor methods that are
    commonly duplicated across different analyzers.
    """
    
    def __init__(self):
        """Initialize the base visitor."""
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.in_class: bool = False
        self.in_function: bool = False
        self.scope_stack: List[str] = []
        
    def visit_ClassDef(self, node: ast.ClassDef) -> Any:
        """
        Visit a class definition node.
        
        Maintains class context for child nodes.
        """
        old_class = self.current_class
        old_in_class = self.in_class
        
        self.current_class = node.name
        self.in_class = True
        self.scope_stack.append(f"class:{node.name}")
        
        # Allow subclasses to process the class
        self._process_class(node)
        
        # Visit child nodes
        self.generic_visit(node)
        
        # Restore context
        self.scope_stack.pop()
        self.current_class = old_class
        self.in_class = old_in_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef) -> Any:
        """
        Visit a function definition node.
        
        Maintains function context for child nodes.
        """
        return self._visit_function_common(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> Any:
        """
        Visit an async function definition node.
        
        Maintains function context for child nodes.
        """
        return self._visit_function_common(node, is_async=True)
        
    def _visit_function_common(self, node: ast.FunctionDef | ast.AsyncFunctionDef, 
                              is_async: bool = False) -> Any:
        """
        Common logic for visiting function definitions.
        
        Args:
            node: The function definition node
            is_async: Whether this is an async function
        """
        old_function = self.current_function
        old_in_function = self.in_function
        
        self.current_function = node.name
        self.in_function = True
        
        func_type = "async_func" if is_async else "func"
        self.scope_stack.append(f"{func_type}:{node.name}")
        
        # Allow subclasses to process the function
        self._process_function(node, is_async=is_async)
        
        # Visit child nodes
        self.generic_visit(node)
        
        # Restore context
        self.scope_stack.pop()
        self.current_function = old_function
        self.in_function = old_in_function
        
    def get_full_name(self, name: str) -> str:
        """
        Get the fully qualified name within current scope.
        
        Args:
            name: The local name
            
        Returns:
            Fully qualified name including class/function scope
        """
        parts = []
        
        if self.current_class:
            parts.append(self.current_class)
            
        if self.current_function:
            parts.append(self.current_function)
            
        parts.append(name)
        
        return ".".join(parts)
    
    def get_current_scope(self) -> str:
        """
        Get the current scope as a string.
        
        Returns:
            Current scope path (e.g., "ClassName.method_name")
        """
        if not self.scope_stack:
            return "<module>"
            
        # Extract just the names from the scope stack
        scope_parts = []
        for scope in self.scope_stack:
            scope_type, scope_name = scope.split(":", 1)
            scope_parts.append(scope_name)
            
        return ".".join(scope_parts) if scope_parts else "<module>"
    
    # Abstract methods that subclasses can override
    @abstractmethod
    def _process_class(self, node: ast.ClassDef) -> None:
        """
        Process a class definition.
        
        Subclasses should override this to add custom class processing.
        
        Args:
            node: The class definition node
        """
        pass
        
    @abstractmethod  
    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, 
                         is_async: bool = False) -> None:
        """
        Process a function definition.
        
        Subclasses should override this to add custom function processing.
        
        Args:
            node: The function definition node
            is_async: Whether this is an async function
        """
        pass


class CollectingVisitor(BaseASTVisitor):
    """
    A visitor that collects information about classes and functions.
    
    This is a common pattern used by many analyzers to build up
    a collection of defined symbols.
    """
    
    def __init__(self):
        """Initialize the collecting visitor."""
        super().__init__()
        self.classes: Set[str] = set()
        self.functions: Set[str] = set()
        self.methods: Set[str] = set()
        self.async_functions: Set[str] = set()
        
    def _process_class(self, node: ast.ClassDef) -> None:
        """Collect class definitions."""
        self.classes.add(node.name)
        
    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, 
                         is_async: bool = False) -> None:
        """Collect function/method definitions."""
        full_name = self.get_full_name(node.name)
        
        if self.in_class:
            self.methods.add(full_name)
        else:
            self.functions.add(full_name)
            
        if is_async:
            self.async_functions.add(full_name)


class AnalysisVisitor(BaseASTVisitor):
    """
    A visitor for analyzing code patterns.
    
    Provides additional utilities for pattern detection and analysis.
    """
    
    def __init__(self):
        """Initialize the analysis visitor."""
        super().__init__()
        self.issues: List[Dict[str, Any]] = []
        
    def _process_class(self, node: ast.ClassDef) -> None:
        """Analyze class definitions for patterns."""
        # Check for empty classes
        if self._is_empty_class(node):
            self.add_issue("empty_class", node, "Empty class definition")
            
    def _process_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef, 
                         is_async: bool = False) -> None:
        """Analyze function definitions for patterns."""
        # Check for stub functions
        if self._is_stub_function(node):
            issue_type = "async_stub" if is_async else "stub_function"
            self.add_issue(issue_type, node, "Stub function with no implementation")
            
    def _is_empty_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is empty (only has pass or docstring)."""
        if not node.body:
            return True
            
        # Check if only contains pass and/or docstring
        non_trivial = False
        for item in node.body:
            if isinstance(item, ast.Pass):
                continue
            elif isinstance(item, ast.Expr) and isinstance(item.value, ast.Constant):
                # Docstring
                continue
            else:
                non_trivial = True
                break
                
        return not non_trivial
        
    def _is_stub_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> bool:
        """Check if a function is a stub (pass, ..., NotImplementedError)."""
        if not node.body:
            return True
            
        # Skip docstring if present
        body_start = 0
        if (len(node.body) > 0 and 
            isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant)):
            body_start = 1
            
        # Check remaining body
        if len(node.body) <= body_start:
            return True
            
        remaining_body = node.body[body_start:]
        
        # Check for stub patterns
        if len(remaining_body) == 1:
            stmt = remaining_body[0]
            
            # pass statement
            if isinstance(stmt, ast.Pass):
                return True
                
            # ... (Ellipsis)
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value is Ellipsis:
                    return True
                    
            # raise NotImplementedError
            if isinstance(stmt, ast.Raise):
                if stmt.exc:
                    if isinstance(stmt.exc, ast.Call):
                        if isinstance(stmt.exc.func, ast.Name):
                            if stmt.exc.func.id == "NotImplementedError":
                                return True
                    elif isinstance(stmt.exc, ast.Name):
                        if stmt.exc.id == "NotImplementedError":
                            return True
                            
        return False
        
    def add_issue(self, issue_type: str, node: ast.AST, message: str) -> None:
        """
        Add an issue to the collection.
        
        Args:
            issue_type: Type of issue detected
            node: AST node where issue was found
            message: Description of the issue
        """
        self.issues.append({
            "type": issue_type,
            "line": getattr(node, "lineno", 0),
            "column": getattr(node, "col_offset", 0),
            "name": getattr(node, "name", ""),
            "scope": self.get_current_scope(),
            "message": message
        })