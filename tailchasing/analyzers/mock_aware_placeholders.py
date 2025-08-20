"""
Mock-aware placeholder detection that distinguishes test doubles from incomplete code.

This analyzer properly handles:
- Mock implementations (MockKyber, MockDilithium, etc.)
- Test stubs and fixtures
- Intentional test doubles
- Actual incomplete implementations
"""

import ast
import re
from typing import List, Optional
from pathlib import Path

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


class MockAwarePlaceholderAnalyzer(BaseAnalyzer):
    """Detects true placeholders while recognizing legitimate mocks and test doubles."""
    
    name = "mock_aware_placeholders"
    
    def __init__(self):
        super().__init__()
        # Patterns that indicate intentional test doubles
        self.mock_indicators = {
            'class_prefixes': {'Mock', 'Fake', 'Stub', 'Dummy', 'Spy', 'Test'},
            'class_suffixes': {'Mock', 'Stub', 'Fake', 'Test', 'TestCase'},
            'function_prefixes': {'mock_', 'fake_', 'stub_', 'dummy_', 'make_mock_'},
            'module_patterns': {'mock', 'fake', 'stub', 'test', 'conftest', 'fixture'},
            'decorator_patterns': {'mock', 'patch', 'fixture', 'parametrize'},
        }
        
        # Patterns that indicate actual placeholders
        self.placeholder_indicators = {
            'todo_patterns': [
                r'#\s*TODO:?\s*implement',
                r'#\s*FIXME:?\s*implement',
                r'raise\s+NotImplementedError\s*\(\s*["\']?TODO',
                r'pass\s*#\s*TODO',
            ],
            'generic_messages': [
                'Not implemented',
                'To be implemented',
                'Implementation pending',
                'Placeholder implementation'
            ]
        }
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Find true placeholders, excluding legitimate mocks."""
        issues = []
        
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
            
            # Determine file context
            file_context = self._analyze_file_context(file)
            
            # Visit nodes looking for placeholders
            visitor = MockAwarePlaceholderVisitor(file, ctx, self, file_context)
            visitor.visit(tree)
            issues.extend(visitor.issues)
        
        return issues
    
    def _analyze_file_context(self, file: str) -> dict:
        """Analyze the context of a file to determine if it's test-related."""
        file_path = Path(file)
        context = {
            'is_test_file': False,
            'is_mock_module': False,
            'is_fixture_file': False,
            'module_type': 'production'
        }
        
        # Check file name and path
        file_name = file_path.name.lower()
        path_str = str(file_path).lower()
        
        # Test file detection
        if any(pattern in path_str for pattern in ['test', 'tests', 'testing']):
            context['is_test_file'] = True
            context['module_type'] = 'test'
        
        # Mock module detection
        if any(pattern in file_name for pattern in self.mock_indicators['module_patterns']):
            context['is_mock_module'] = True
            context['module_type'] = 'mock'
        
        # Fixture file detection
        if 'fixture' in file_name or 'conftest' in file_name:
            context['is_fixture_file'] = True
            context['module_type'] = 'fixture'
        
        return context
    
    def is_legitimate_mock(self, node: ast.AST, file_context: dict) -> bool:
        """Check if a node represents a legitimate mock/test double."""
        # Check class names
        if isinstance(node, ast.ClassDef):
            # Check prefixes
            for prefix in self.mock_indicators['class_prefixes']:
                if node.name.startswith(prefix):
                    return True
            
            # Check suffixes
            for suffix in self.mock_indicators['class_suffixes']:
                if node.name.endswith(suffix):
                    return True
            
            # Check base classes for test patterns
            for base in node.bases:
                if isinstance(base, ast.Name):
                    if 'Mock' in base.id or 'Test' in base.id:
                        return True
        
        # Check function names
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for prefix in self.mock_indicators['function_prefixes']:
                if node.name.startswith(prefix):
                    return True
            
            # Check decorators
            for decorator in node.decorator_list:
                decorator_name = self._get_decorator_name(decorator)
                for pattern in self.mock_indicators['decorator_patterns']:
                    if pattern in decorator_name.lower():
                        return True
        
        # In test/mock files, be more lenient
        if file_context['module_type'] in ['test', 'mock', 'fixture']:
            return True
        
        return False
    
    def _get_decorator_name(self, decorator: ast.AST) -> str:
        """Extract decorator name from AST node."""
        if isinstance(decorator, ast.Name):
            return decorator.id
        elif isinstance(decorator, ast.Attribute):
            return decorator.attr
        elif isinstance(decorator, ast.Call):
            if isinstance(decorator.func, ast.Name):
                return decorator.func.id
            elif isinstance(decorator.func, ast.Attribute):
                return decorator.func.attr
        return ""


class MockAwarePlaceholderVisitor(ast.NodeVisitor):
    """Visitor that identifies true placeholders vs legitimate mocks."""
    
    def __init__(self, file: str, ctx: AnalysisContext, analyzer: MockAwarePlaceholderAnalyzer,
                 file_context: dict):
        self.file = file
        self.ctx = ctx
        self.analyzer = analyzer
        self.file_context = file_context
        self.issues: List[Issue] = []
        self.current_class: Optional[str] = None
    
    def visit_ClassDef(self, node: ast.ClassDef):
        """Visit class definition."""
        old_class = self.current_class
        self.current_class = node.name
        
        # Check if entire class is a placeholder
        if not self.analyzer.is_legitimate_mock(node, self.file_context):
            if self._is_placeholder_class(node):
                self._add_placeholder_issue(node, "class", "Empty or stub class")
        
        self.generic_visit(node)
        self.current_class = old_class
    
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Visit function definition."""
        self._check_function(node)
        self.generic_visit(node)
    
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Visit async function definition."""
        self._check_function(node, is_async=True)
        self.generic_visit(node)
    
    def _check_function(self, node: ast.FunctionDef, is_async: bool = False):
        """Check if a function is a true placeholder."""
        # Skip if it's a legitimate mock
        if self.analyzer.is_legitimate_mock(node, self.file_context):
            return
        
        # Skip abstract methods
        if any('abstract' in self.analyzer._get_decorator_name(d).lower() 
               for d in node.decorator_list):
            return
        
        # Check for placeholder patterns
        if self._is_placeholder_function(node):
            func_type = "async function" if is_async else "function"
            self._add_placeholder_issue(node, func_type, self._get_placeholder_reason(node))
    
    def _is_placeholder_class(self, node: ast.ClassDef) -> bool:
        """Check if a class is a placeholder."""
        # Empty class with just pass
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return True
        
        # Class with only docstring and pass
        if len(node.body) == 2:
            if isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
                if isinstance(node.body[1], ast.Pass):
                    return True
        
        # Class with only ellipsis
        if len(node.body) == 1 and isinstance(node.body[0], ast.Expr):
            if isinstance(node.body[0].value, ast.Constant) and node.body[0].value.value is ...:
                return True
        
        return False
    
    def _is_placeholder_function(self, node: ast.FunctionDef) -> bool:
        """Check if a function is a placeholder."""
        body = node.body
        
        # Skip docstring if present
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        
        if not body:
            return True
        
        # Single pass statement
        if len(body) == 1 and isinstance(body[0], ast.Pass):
            return True
        
        # Single ellipsis
        if len(body) == 1 and isinstance(body[0], ast.Expr):
            if isinstance(body[0].value, ast.Constant) and body[0].value.value is ...:
                return True
        
        # Raises NotImplementedError (check if it's a real placeholder)
        if len(body) == 1 and isinstance(body[0], ast.Raise):
            if self._is_placeholder_not_implemented(body[0]):
                return True
        
        # Contains only TODO comment
        if self._contains_only_todo(node):
            return True
        
        return False
    
    def _is_placeholder_not_implemented(self, raise_node: ast.Raise) -> bool:
        """Check if NotImplementedError indicates a placeholder."""
        if raise_node.exc:
            # Check if it's NotImplementedError
            is_not_implemented = False
            
            if isinstance(raise_node.exc, ast.Call):
                if isinstance(raise_node.exc.func, ast.Name):
                    is_not_implemented = raise_node.exc.func.id == "NotImplementedError"
            elif isinstance(raise_node.exc, ast.Name):
                is_not_implemented = raise_node.exc.id == "NotImplementedError"
            
            if is_not_implemented:
                # Check if message indicates placeholder
                if isinstance(raise_node.exc, ast.Call) and raise_node.exc.args:
                    first_arg = raise_node.exc.args[0]
                    if isinstance(first_arg, ast.Constant):
                        message = str(first_arg.value).lower()
                        
                        # Check for placeholder indicators
                        for indicator in self.analyzer.placeholder_indicators['generic_messages']:
                            if indicator.lower() in message:
                                return True
                        
                        # If no message or generic message, it's likely a placeholder
                        if not message or message in ['', 'not implemented']:
                            return True
                else:
                    # No message - likely a placeholder
                    return True
        
        return False
    
    def _contains_only_todo(self, node: ast.FunctionDef) -> bool:
        """Check if function contains only TODO comments."""
        # Get function source if possible
        try:
            import inspect
            source = inspect.getsource(node)
            
            # Check for TODO patterns
            for pattern in self.analyzer.placeholder_indicators['todo_patterns']:
                if re.search(pattern, source, re.IGNORECASE):
                    # Check if there's any real implementation
                    has_implementation = False
                    for stmt in node.body:
                        if not isinstance(stmt, (ast.Pass, ast.Expr)):
                            has_implementation = True
                            break
                    
                    if not has_implementation:
                        return True
        except:
            pass
        
        return False
    
    def _get_placeholder_reason(self, node: ast.FunctionDef) -> str:
        """Get specific reason why function is a placeholder."""
        body = node.body
        
        # Skip docstring
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]
        
        if not body:
            return "Empty function body"
        
        if len(body) == 1:
            stmt = body[0]
            if isinstance(stmt, ast.Pass):
                return "Contains only 'pass'"
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
                if stmt.value.value is ...:
                    return "Contains only ellipsis"
            elif isinstance(stmt, ast.Raise):
                return "Raises NotImplementedError without implementation"
        
        return "Incomplete implementation"
    
    def _add_placeholder_issue(self, node: ast.AST, node_type: str, reason: str):
        """Add a placeholder issue."""
        name = node.name if hasattr(node, 'name') else 'unknown'
        
        if self.current_class:
            full_name = f"{self.current_class}.{name}"
        else:
            full_name = name
        
        # Don't flag mocks in test files as issues
        if self.file_context['module_type'] in ['test', 'mock', 'fixture']:
            # Only flag if it's truly incomplete (not a valid mock)
            if 'Mock' in full_name or 'mock' in name.lower():
                return  # Valid mock, don't flag
        
        # Adjust severity based on context
        if self.file_context['module_type'] == 'production':
            severity = 2
            kind = "true_placeholder"
        else:
            severity = 1
            kind = "incomplete_test_helper"
        
        issue = Issue(
            kind=kind,
            message=f"Placeholder {node_type}: {full_name} - {reason}",
            severity=severity,
            file=self.file,
            line=safe_get_lineno(node),
            symbol=full_name,
            evidence={
                "placeholder_type": node_type,
                "reason": reason,
                "in_test_file": self.file_context['is_test_file'],
                "module_type": self.file_context['module_type']
            },
            suggestions=[
                "Implement the functionality",
                "Remove if not needed",
                "Convert to abstract method if this is an interface"
            ] if self.file_context['module_type'] == 'production' else [
                "Complete the mock implementation",
                "Add mock behavior for testing"
            ],
            confidence=0.7 if self.file_context['module_type'] != 'production' else 0.9
        )
        self.issues.append(issue)