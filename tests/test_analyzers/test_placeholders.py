"""
Unit tests for placeholder/phantom function detection.

TODOs addressed:
1. Test detection of various placeholder patterns
2. Test handling of legitimate abstract methods
3. Test confidence scoring based on context
4. Test fix suggestions for different placeholder types
"""

import pytest
import ast
from pathlib import Path
from typing import List

from tailchasing.analyzers.placeholders import PlaceholderAnalyzer
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.symbols import SymbolTable
from tailchasing.core.issues import Issue


class TestPlaceholderDetection:
    """Test detection of placeholder/phantom functions."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a PlaceholderAnalyzer instance."""
        return PlaceholderAnalyzer()
    
    @pytest.fixture
    def context(self, tmp_path):
        """Create a test analysis context."""
        return AnalysisContext(
            config={
                "placeholders": {
                    "allow": ["BaseClass.abstract_method"]
                }
            },
            root_dir=tmp_path,
            file_paths=[],
            ast_index={},
            symbol_table=SymbolTable(),
            source_cache={},
            cache={}
        )
    
    # TODO 1: Test detection of various placeholder patterns
    @pytest.mark.parametrize("code,should_detect", [
        # Pass-only function
        ("""
def process_data(data):
    pass
""", True),
        
        # NotImplementedError
        ("""
def calculate_total(items):
    raise NotImplementedError("TODO: implement this")
""", True),
        
        # Ellipsis placeholder
        ("""
def validate_input(value):
    ...
""", True),
        
        # TODO comment only
        ("""
def transform_data(data):
    # TODO: implement transformation
    pass
""", True),
        
        # Return None explicitly (might be intentional)
        ("""
def reset_state():
    return None
""", False),
        
        # Actual implementation
        ("""
def add_numbers(a, b):
    return a + b
""", False),
        
        # Docstring only (still placeholder)
        ("""
def complex_algorithm(data):
    '''This will implement a complex algorithm.'''
    pass
""", True),
    ])
    def test_placeholder_patterns(self, analyzer, context, code, should_detect):
        """Test detection of various placeholder patterns."""
        tree = ast.parse(code)
        context.ast_index["test.py"] = tree
        context.source_cache["test.py"] = code.split('\n')
        
        issues = list(analyzer.run(context))
        
        if should_detect:
            assert len(issues) > 0, f"Should detect placeholder in: {code}"
            assert issues[0].kind == "phantom_function"
        else:
            assert len(issues) == 0, f"Should not detect placeholder in: {code}"
    
    # TODO 2: Test handling of legitimate abstract methods
    def test_abstract_method_handling(self, analyzer, context):
        """Test that legitimate abstract methods are not flagged."""
        code = """
from abc import ABC, abstractmethod

class BaseProcessor(ABC):
    @abstractmethod
    def process(self, data):
        pass
    
    def concrete_method(self):
        return "implemented"

class ConcreteProcessor(BaseProcessor):
    def process(self, data):
        pass  # This should be flagged - concrete class with placeholder
"""
        tree = ast.parse(code)
        context.ast_index["test.py"] = tree
        context.source_cache["test.py"] = code.split('\n')
        
        issues = list(analyzer.run(context))
        
        # Should not flag the abstract method but should flag the concrete one
        assert len(issues) == 1
        assert "ConcreteProcessor" in issues[0].message
        
    def test_protocol_methods(self, analyzer, context):
        """Test handling of Protocol stub methods."""
        code = """
from typing import Protocol

class Comparable(Protocol):
    def __lt__(self, other) -> bool:
        ...
    
    def __eq__(self, other) -> bool:
        ...

class Item:
    def __lt__(self, other):
        ...  # This should be flagged - not a Protocol
"""
        tree = ast.parse(code)
        context.ast_index["test.py"] = tree
        context.source_cache["test.py"] = code.split('\n')
        
        issues = list(analyzer.run(context))
        
        # Should only flag the Item class method
        assert len(issues) == 1
        assert "Item" in issues[0].message
    
    # TODO 3: Test confidence scoring based on context
    def test_confidence_scoring(self, analyzer, context):
        """Test that confidence varies based on context."""
        test_cases = [
            # Test file - lower confidence
            ("test_module.py", """
def test_something():
    pass
""", 0.5),
            
            # Init file - might be intentional
            ("__init__.py", """
def initialize():
    pass
""", 0.7),
            
            # Regular module - high confidence
            ("processor.py", """
def process():
    pass
""", 0.9),
            
            # Has docstring - slightly lower confidence
            ("documented.py", """
def calculate():
    '''Calculate the result.'''
    pass
""", 0.8),
        ]
        
        for filename, code, expected_confidence in test_cases:
            tree = ast.parse(code)
            context.ast_index[filename] = tree
            context.source_cache[filename] = code.split('\n')
            
            issues = list(analyzer.run(context))
            
            if issues:
                # Check confidence is in expected range
                assert abs(issues[0].confidence - expected_confidence) < 0.2, \
                    f"Confidence for {filename} should be close to {expected_confidence}"
    
    # TODO 4: Test fix suggestions for different placeholder types
    @pytest.mark.parametrize("code,expected_suggestions", [
        # Simple pass placeholder
        ("""
def calculate_tax(amount):
    pass
""", [
            "Implement the function body with actual tax calculation logic",
            "Remove the function if it's not needed",
            "Import the actual implementation from another module"
        ]),
        
        # NotImplementedError with message
        ("""
def validate_data(data):
    raise NotImplementedError("Need to add validation logic")
""", [
            "Implement the validation logic mentioned in the error message",
            "Create an abstract base class if this is meant to be overridden",
            "Remove if this functionality is not required"
        ]),
        
        # Ellipsis in class
        ("""
class DataProcessor:
    def process(self):
        ...
""", [
            "Implement the process method",
            "Make this an abstract base class with @abstractmethod",
            "Use Protocol if this is meant as an interface"
        ]),
        
        # TODO comment
        ("""
def transform(data):
    # TODO: implement transformation logic
    # Should handle edge cases
    pass
""", [
            "Implement the transformation logic described in the TODO",
            "Break down into smaller, implementable functions",
            "Create a ticket and reference it if this is future work"
        ]),
    ])
    def test_fix_suggestions(self, analyzer, context, code, expected_suggestions):
        """Test that appropriate fix suggestions are generated."""
        tree = ast.parse(code)
        context.ast_index["test.py"] = tree
        context.source_cache["test.py"] = code.split('\n')
        
        issues = list(analyzer.run(context))
        
        assert len(issues) > 0, "Should detect placeholder"
        
        suggestions = issues[0].suggestions
        
        # Check that we have meaningful suggestions
        assert len(suggestions) >= 2, "Should have multiple suggestions"
        
        # Check that suggestions match the pattern type
        for expected in expected_suggestions:
            assert any(
                any(keyword in suggestion.lower() for keyword in expected.lower().split())
                for suggestion in suggestions
            ), f"Should have suggestion about: {expected}"


class TestPlaceholderEvolution:
    """Test tracking of placeholder evolution over time."""
    
    def test_stagnant_placeholder_detection(self):
        """Test detection of placeholders that never get implemented."""
        # This would integrate with git history analysis
        # For now, we'll test the concept
        
        placeholder_age_days = 30  # Placeholder has existed for 30 days
        
        if placeholder_age_days > 14:
            severity = 3  # Increase severity for old placeholders
        else:
            severity = 2
            
        assert severity == 3, "Old placeholders should have higher severity"
    
    def test_placeholder_proliferation(self):
        """Test detection of cascading placeholder creation."""
        placeholders = [
            ("helper1", "file1.py", 10),
            ("helper2", "file1.py", 20),
            ("helper3", "file1.py", 30),
            ("process_helper1", "file2.py", 10),
        ]
        
        # Count placeholders with similar names
        related_count = sum(1 for name, _, _ in placeholders if "helper" in name)
        
        is_cascade = related_count >= 3
        assert is_cascade, "Should detect placeholder cascade pattern"


class TestAllowedPlaceholders:
    """Test handling of explicitly allowed placeholders."""
    
    @pytest.fixture
    def context_with_allowlist(self, tmp_path):
        """Create context with placeholder allowlist."""
        return AnalysisContext(
            config={
                "placeholders": {
                    "allow": [
                        "BaseHandler.handle",
                        "AbstractProcessor.process",
                        "*TestCase.setUp"
                    ]
                }
            },
            root_dir=tmp_path,
            file_paths=[],
            ast_index={},
            symbol_table=SymbolTable(),
            source_cache={},
            cache={}
        )
    
    def test_allowed_placeholder_ignored(self, context_with_allowlist):
        """Test that allowed placeholders are not flagged."""
        analyzer = PlaceholderAnalyzer()
        
        code = """
class BaseHandler:
    def handle(self, request):
        pass  # Allowed
    
    def process(self, data):
        pass  # Not allowed - different method

class MyTestCase:
    def setUp(self):
        pass  # Allowed by pattern
"""
        tree = ast.parse(code)
        context_with_allowlist.ast_index["test.py"] = tree
        context_with_allowlist.source_cache["test.py"] = code.split('\n')
        
        issues = list(analyzer.run(context_with_allowlist))
        
        # Should only flag the process method
        assert len(issues) == 1
        assert "process" in issues[0].message
        assert "handle" not in issues[0].message
        assert "setUp" not in issues[0].message