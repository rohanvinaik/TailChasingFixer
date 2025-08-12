"""
Tests for enhanced placeholder detection.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from tailchasing.analyzers.enhanced_placeholders import (
    EnhancedPlaceholderAnalyzer,
    EnhancedPlaceholderVisitor,
    PlaceholderSeverity,
    PlaceholderPattern
)
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.symbols import SymbolTable


def create_test_context(code: str, filename: str = "test.py"):
    """Create a test analysis context."""
    tree = ast.parse(code)
    
    ctx = MagicMock(spec=AnalysisContext)
    ctx.ast_index = {filename: tree}
    ctx.symbol_table = SymbolTable()
    ctx.config = {"enhanced_placeholders": {"enabled": True}}
    ctx.root_dir = Path(".")
    ctx.is_excluded.return_value = False
    ctx.is_placeholder_allowed.return_value = False
    
    return ctx, tree


class TestEnhancedPlaceholderDetection:
    """Test enhanced placeholder pattern detection."""
    
    def test_detect_pass_only_function(self):
        """Test detection of functions with only pass."""
        code = '''
def my_function():
    pass
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        assert len(issues) == 1
        assert issues[0].kind == "enhanced_placeholder"
        assert "pass" in issues[0].message.lower()
        assert "raise NotImplementedError" in issues[0].suggestions[0]
    
    def test_detect_return_none_function(self):
        """Test detection of functions that only return None."""
        code = '''
def get_data():
    return None
    
def process():
    return
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        assert len(issues) == 2
        for issue in issues:
            assert "return" in issue.message.lower()
    
    def test_detect_return_constant_function(self):
        """Test detection of functions that only return constants."""
        code = '''
def get_status():
    return "TODO"
    
def get_count():
    return 0
    
def get_empty_list():
    return []
    
def get_empty_dict():
    return {}
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        assert len(issues) == 4
        assert all(issue.severity <= 2 for issue in issues)  # Low to medium severity
    
    def test_detect_low_complexity_logic_functions(self):
        """Test detection of low-complexity functions with logic-implying names."""
        code = '''
def check_validity():
    return True
    
def validate_input(data):
    pass
    
def process_data(items):
    return items
    
def detect_anomalies(values):
    return []
    
def calculate_score():
    return 0
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        # All these should be detected due to logic-implying names
        assert len(issues) >= 5
        
        # Check that logic-implying functions are flagged
        for issue in issues:
            evidence = issue.evidence
            if "implies_logic" in evidence:
                assert evidence["implies_logic"] is True
    
    def test_not_implemented_pattern(self):
        """Test detection of NotImplementedError patterns."""
        code = '''
def future_feature():
    raise NotImplementedError("Coming soon")
    
def abstract_method():
    raise NotImplementedError
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        assert len(issues) == 2
        for issue in issues:
            assert "NotImplementedError" in issue.message
            assert issue.severity == 1  # Low severity
    
    def test_todo_only_function(self):
        """Test detection of functions with only TODO comments."""
        code = '''
def future_implementation():
    """TODO: Implement this feature."""
    pass
    
def another_todo():
    """TODO: Add logic here"""
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        assert len(issues) == 2
        for issue in issues:
            assert "TODO" in issue.message
    
    def test_ellipsis_pattern(self):
        """Test detection of ellipsis placeholder."""
        code = '''
def placeholder():
    ...
    
def with_docstring():
    """This is a placeholder."""
    ...
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        assert len(issues) == 2
        for issue in issues:
            assert "ellipsis" in issue.message.lower()
    
    def test_severity_based_on_usage(self):
        """Test that severity increases when function is called."""
        code = '''
def validate_data(data):
    return True
    
def process():
    if validate_data(input_data):
        print("Valid")
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        
        # Collect call information first
        analyzer._collect_call_information(ctx)
        
        # Run detection
        issues = list(analyzer.run(ctx))
        
        # validate_data should have higher severity because it's called
        validate_issue = [i for i in issues if "validate_data" in i.symbol][0]
        assert validate_issue.severity >= 2  # Medium or high
        assert validate_issue.evidence["is_called"] is True
    
    def test_skip_abstract_methods(self):
        """Test that abstract methods are skipped."""
        code = '''
from abc import ABC, abstractmethod

class MyInterface(ABC):
    @abstractmethod
    def required_method(self):
        pass
        
class MyProtocol:
    def protocol_method(self):
        ...
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        # Abstract method should be skipped
        assert not any("required_method" in issue.symbol for issue in issues)
        
        # Non-abstract placeholder should still be detected
        assert any("protocol_method" in issue.symbol for issue in issues)
    
    def test_interface_method_suggestions(self):
        """Test suggestions for interface-like methods."""
        code = '''
class MyClass:
    def __enter__(self):
        pass
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
        
    def __iter__(self):
        return []
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        # Check that interface methods get appropriate suggestions
        for issue in issues:
            if "__enter__" in issue.symbol or "__exit__" in issue.symbol:
                assert any("Protocol" in s or "ABC" in s for s in issue.suggestions)
    
    def test_complexity_calculation(self):
        """Test that complexity is calculated correctly."""
        code = '''
def simple():
    return 1
    
def with_if(x):
    if x > 0:
        return x
    return 0
    
def with_loop(items):
    for item in items:
        if item:
            return item
    return None
'''
        ctx, tree = create_test_context(code)
        visitor = EnhancedPlaceholderVisitor("test.py", ctx, {})
        
        # Parse and find functions
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        # Check complexity
        assert visitor._calculate_complexity(functions[0]) == 1  # simple
        assert visitor._calculate_complexity(functions[1]) == 2  # with_if
        assert visitor._calculate_complexity(functions[2]) == 3  # with_loop
    
    def test_test_file_severity_reduction(self):
        """Test that placeholders in test files get lower severity."""
        code = '''
def mock_function():
    pass
    
def stub_for_testing():
    return None
'''
        # Use test file name
        ctx, tree = create_test_context(code, "test_module.py")
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        # All issues in test files should have low severity
        assert all(issue.severity == 1 for issue in issues)
    
    def test_generate_suggestions(self):
        """Test that appropriate suggestions are generated."""
        code = '''
def validate_input(data):
    pass
    
def check_permission():
    return True
    
def transform_data(items):
    return items
'''
        ctx, tree = create_test_context(code)
        analyzer = EnhancedPlaceholderAnalyzer()
        issues = list(analyzer.run(ctx))
        
        for issue in issues:
            # Should have multiple suggestions
            assert len(issue.suggestions) >= 3
            
            # Check for specific suggestions based on pattern
            if "pass" in issue.message.lower():
                assert any("NotImplementedError" in s for s in issue.suggestions)
            
            if "validate" in issue.symbol or "check" in issue.symbol:
                assert any("implement" in s.lower() and "logic" in s.lower() 
                          for s in issue.suggestions)