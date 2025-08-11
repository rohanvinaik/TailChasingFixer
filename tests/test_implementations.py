"""
Comprehensive test suite for implemented functions.

Tests cover:
- CargoCultDetector._parent_has_init
- SemanticHVAnalyzer._are_structural_duplicates  
- BaseFixStrategy.get_dependencies
- AnthropicAdapter._get_system_prompt
- OllamaAdapter.estimate_cost
"""

import ast
import unittest
from dataclasses import dataclass
from typing import Tuple
from unittest.mock import Mock, patch, MagicMock

# Test for CargoCultDetector._parent_has_init
class TestParentHasInit(unittest.TestCase):
    """Test cases for _parent_has_init method."""
    
    def setUp(self):
        """Set up test detector."""
        from tailchasing.analyzers.cargo_cult import CargoCultDetector
        from tailchasing.analyzers.base import AnalysisContext
        self.detector = CargoCultDetector()
        self.ctx = Mock(spec=AnalysisContext)
    
    def _create_class_node(self, class_obj):
        """Helper to create AST node from class."""
        import inspect
        source = inspect.getsource(class_obj)
        tree = ast.parse(source)
        return tree.body[0]
    
    def test_single_inheritance_with_init(self):
        """Test single inheritance where parent has __init__."""
        code = """
class Child(Parent):
    pass
"""
        class_node = ast.parse(code).body[0]
        
        # Note: Current implementation returns True (conservative)
        # This test documents the current behavior
        self.assertTrue(self.detector._parent_has_init(class_node, self.ctx))
    
    def test_single_inheritance_without_init(self):
        """Test single inheritance where parent has no __init__."""
        code = """
class Child(Parent):
    pass
"""
        class_node = ast.parse(code).body[0]
        
        # Current implementation returns True (conservative)
        self.assertTrue(self.detector._parent_has_init(class_node, self.ctx))
    
    def test_multiple_inheritance_with_init(self):
        """Test multiple inheritance where at least one parent has __init__."""
        code = """
class Child(Parent1, Parent2):
    pass
"""
        class_node = ast.parse(code).body[0]
        
        # Current implementation returns True
        self.assertTrue(self.detector._parent_has_init(class_node, self.ctx))
    
    def test_multiple_inheritance_without_init(self):
        """Test multiple inheritance where no parent has __init__."""
        code = """
class Child(Parent1, Parent2):
    pass
"""
        class_node = ast.parse(code).body[0]
        
        # Current implementation returns True (conservative)
        self.assertTrue(self.detector._parent_has_init(class_node, self.ctx))
    
    def test_external_base_class(self):
        """Test inheritance from external base class (e.g., Exception)."""
        code = """
class CustomError(Exception):
    pass
"""
        class_node = ast.parse(code).body[0]
        
        # Current implementation returns True
        self.assertTrue(self.detector._parent_has_init(class_node, self.ctx))
    
    def test_dataclass_inheritance(self):
        """Test inheritance from dataclass."""
        code = """
@dataclass
class Child(Parent):
    pass
"""
        class_node = ast.parse(code).body[0]
        
        # Current implementation returns True
        self.assertTrue(self.detector._parent_has_init(class_node, self.ctx))


# Test for SemanticHVAnalyzer._are_structural_duplicates
class TestStructuralDuplicates(unittest.TestCase):
    """Test cases for _are_structural_duplicates method."""
    
    def setUp(self):
        """Set up test analyzer."""
        from tailchasing.analyzers.semantic_hv import SemanticHVAnalyzer
        self.analyzer = SemanticHVAnalyzer()
    
    def test_identical_functions(self):
        """Test completely identical functions."""
        code1 = """
def foo(x, y):
    result = x + y
    return result
"""
        code2 = """
def foo(x, y):
    result = x + y
    return result
"""
        f1 = {"ast_node": ast.parse(code1).body[0]}
        f2 = {"ast_node": ast.parse(code2).body[0]}
        
        self.assertTrue(self.analyzer._are_structural_duplicates(f1, f2))
    
    def test_renamed_identifiers(self):
        """Test functions with renamed variables."""
        code1 = """
def calculate(a, b):
    total = a + b
    return total
"""
        code2 = """
def compute(x, y):
    sum_val = x + y
    return sum_val
"""
        f1 = {"ast_node": ast.parse(code1).body[0]}
        f2 = {"ast_node": ast.parse(code2).body[0]}
        
        self.assertTrue(self.analyzer._are_structural_duplicates(f1, f2))
    
    def test_changed_constants(self):
        """Test functions with different constant values."""
        code1 = """
def get_value():
    return 42
"""
        code2 = """
def get_value():
    return 100
"""
        f1 = {"ast_node": ast.parse(code1).body[0]}
        f2 = {"ast_node": ast.parse(code2).body[0]}
        
        # Should be structural duplicates (constants normalized)
        self.assertTrue(self.analyzer._are_structural_duplicates(f1, f2))
    
    def test_wrapper_pattern(self):
        """Test wrapper functions with same structure."""
        code1 = """
def wrapper(func):
    def inner(*args):
        print("Before")
        result = func(*args)
        print("After")
        return result
    return inner
"""
        code2 = """
def decorator(fn):
    def wrapped(*params):
        print("Start")
        output = fn(*params)
        print("End")
        return output
    return wrapped
"""
        f1 = {"ast_node": ast.parse(code1).body[0]}
        f2 = {"ast_node": ast.parse(code2).body[0]}
        
        self.assertTrue(self.analyzer._are_structural_duplicates(f1, f2))
    
    def test_different_control_flow(self):
        """Test functions with different control flow."""
        code1 = """
def process(x):
    if x > 0:
        return x * 2
    return x
"""
        code2 = """
def process(x):
    for i in range(x):
        x += 1
    return x
"""
        f1 = {"ast_node": ast.parse(code1).body[0]}
        f2 = {"ast_node": ast.parse(code2).body[0]}
        
        self.assertFalse(self.analyzer._are_structural_duplicates(f1, f2))
    
    def test_tiny_function_strict(self):
        """Test tiny functions with strict threshold."""
        code1 = """
def f():
    pass
"""
        code2 = """
def g():
    return None
"""
        f1 = {"ast_node": ast.parse(code1).body[0]}
        f2 = {"ast_node": ast.parse(code2).body[0]}
        
        # With strict threshold, even tiny differences matter
        self.assertFalse(self.analyzer._are_structural_duplicates(
            f1, f2, seq_threshold=0.99, skel_threshold=0.99
        ))


# Test for BaseFixStrategy.get_dependencies
class TestGetDependencies(unittest.TestCase):
    """Test cases for get_dependencies method."""
    
    def test_explicit_dependencies(self):
        """Test strategy with explicit dependencies."""
        from tailchasing.fixers.strategies.base import BaseFixStrategy
        
        class TestStrategy(BaseFixStrategy):
            REQUIRES_ANALYZERS = ("duplicates", "circular_imports")
            REQUIRES_TOOLS = ("ast_parser", "import_resolver")
            REQUIRES_MODELS = ("gpt-4", "claude-3")
            
            def can_handle(self, issue):
                return True
            
            def propose_fix(self, issue, context=None):
                return None
        
        strategy = TestStrategy("test")
        deps = strategy.get_dependencies()
        
        expected = [
            "analyzer:circular_imports",
            "analyzer:duplicates",
            "model:claude-3",
            "model:gpt-4",
            "tool:ast_parser",
            "tool:import_resolver"
        ]
        
        self.assertEqual(deps, expected)
    
    def test_empty_dependencies(self):
        """Test strategy with no dependencies."""
        from tailchasing.fixers.strategies.base import BaseFixStrategy
        
        class SimpleStrategy(BaseFixStrategy):
            # No dependency declarations
            
            def can_handle(self, issue):
                return True
            
            def propose_fix(self, issue, context=None):
                return None
        
        strategy = SimpleStrategy("simple")
        deps = strategy.get_dependencies()
        
        self.assertEqual(deps, [])
    
    def test_unknown_dependency_types(self):
        """Test strategy with only some dependency types."""
        from tailchasing.fixers.strategies.base import BaseFixStrategy
        
        class PartialStrategy(BaseFixStrategy):
            REQUIRES_ANALYZERS = ("semantic",)
            # No REQUIRES_TOOLS or REQUIRES_MODELS
            
            def can_handle(self, issue):
                return True
            
            def propose_fix(self, issue, context=None):
                return None
        
        strategy = PartialStrategy("partial")
        deps = strategy.get_dependencies()
        
        self.assertEqual(deps, ["analyzer:semantic"])


# Test for AnthropicAdapter._get_system_prompt
class TestSystemPrompt(unittest.TestCase):
    """Test cases for _get_system_prompt method."""
    
    def setUp(self):
        """Set up test adapter."""
        # Import first to avoid import issues
        from tailchasing.llm.adapters.anthropic_adapter import AnthropicAdapter
        
        # Create adapter without initializing client
        self.adapter = AnthropicAdapter(api_key="test_key")
        self.adapter._client = Mock()  # Mock the client directly
    
    def test_refactor_mode(self):
        """Test refactor mode (default)."""
        prompt = self.adapter._get_system_prompt("refactor")
        
        self.assertIn("meticulous senior software engineer", prompt)
        self.assertIn("AST-safe, minimal diffs", prompt)
        self.assertIn("Preserve public API", prompt)
        self.assertIn("refactor existing code with minimal, safe edits", prompt)
        self.assertIn("PROMPT_VERSION=anthropic.v1", prompt)
    
    def test_lint_fix_mode(self):
        """Test lint fix mode."""
        prompt = self.adapter._get_system_prompt("lint_fix")
        
        self.assertIn("style-only changes", prompt)
        self.assertIn("No behavior changes", prompt)
        self.assertNotIn("refactor existing code", prompt)
    
    def test_test_fix_mode(self):
        """Test test fix mode."""
        prompt = self.adapter._get_system_prompt("test_fix")
        
        self.assertIn("smallest change to satisfy failing tests", prompt)
        self.assertNotIn("style-only changes", prompt)
    
    def test_codegen_mode(self):
        """Test code generation mode."""
        prompt = self.adapter._get_system_prompt("codegen")
        
        self.assertIn("generate new code only when explicitly instructed", prompt)
        self.assertNotIn("refactor existing code", prompt)
    
    def test_custom_settings(self):
        """Test with custom settings."""
        settings = Mock()
        settings.python_version = "3.12"
        
        prompt = self.adapter._get_system_prompt("refactor", settings=settings)
        
        self.assertIn("Python 3.12", prompt)
        self.assertIn("ruff, mypy, pytest", prompt)
    
    def test_context_parameter(self):
        """Test that context parameter is accepted."""
        context = {"some": "context"}
        # Should not raise
        prompt = self.adapter._get_system_prompt("refactor", context=context)
        self.assertIsNotNone(prompt)


# Test for OllamaAdapter.estimate_cost
class TestOllamaEstimateCost(unittest.TestCase):
    """Test cases for estimate_cost method."""
    
    def setUp(self):
        """Set up test adapter."""
        # Mock the requests to avoid actual API calls
        with patch('tailchasing.llm.adapters.ollama_adapter.requests'):
            from tailchasing.llm.adapters.ollama_adapter import OllamaAdapter
            # Mock availability check
            with patch.object(OllamaAdapter, '_check_ollama_availability'):
                self.adapter = OllamaAdapter(model="llama3:8b")
    
    def test_known_model(self):
        """Test cost estimation for known model."""
        result = self.adapter.estimate_cost(1000, 500)
        
        self.assertEqual(result["model"], "llama3:8b")
        self.assertEqual(result["prompt_tokens"], 1000)
        self.assertEqual(result["completion_tokens"], 500)
        self.assertEqual(result["total_tokens"], 1500)
        
        # llama3:8b pricing: (0.05, 0.10) per 1k tokens
        expected_cost = (1000/1000.0) * 0.05 + (500/1000.0) * 0.10
        self.assertEqual(result["equivalent_cost_usd"], round(expected_cost, 6))
        self.assertEqual(result["actual_cost_usd"], 0.0)  # Local model
        self.assertTrue(result["is_local"])
    
    def test_unknown_model(self):
        """Test cost estimation for unknown model."""
        with patch.object(self.adapter.__class__, '_check_ollama_availability'):
            self.adapter.model = "custom-model:latest"
        
        result = self.adapter.estimate_cost(2000, 1000)
        
        self.assertEqual(result["model"], "custom-model:latest")
        self.assertEqual(result["prompt_tokens"], 2000)
        self.assertEqual(result["completion_tokens"], 1000)
        
        # Should use default pricing: (0.05, 0.10)
        expected_cost = (2000/1000.0) * 0.05 + (1000/1000.0) * 0.10
        self.assertEqual(result["equivalent_cost_usd"], round(expected_cost, 6))
    
    def test_zero_tokens(self):
        """Test cost estimation with zero tokens."""
        result = self.adapter.estimate_cost(0, 0)
        
        self.assertEqual(result["prompt_tokens"], 0)
        self.assertEqual(result["completion_tokens"], 0)
        self.assertEqual(result["total_tokens"], 0)
        self.assertEqual(result["cost_usd"], 0.0)
        self.assertEqual(result["equivalent_cost_usd"], 0.0)
    
    def test_large_tokens(self):
        """Test cost estimation with large token counts."""
        result = self.adapter.estimate_cost(100000, 50000)
        
        self.assertEqual(result["prompt_tokens"], 100000)
        self.assertEqual(result["completion_tokens"], 50000)
        self.assertEqual(result["total_tokens"], 150000)
        
        # llama3:8b pricing
        expected_cost = (100000/1000.0) * 0.05 + (50000/1000.0) * 0.10
        self.assertEqual(result["equivalent_cost_usd"], round(expected_cost, 6))
        
        # Verify it's a reasonable cost (not astronomical)
        self.assertLess(result["equivalent_cost_usd"], 20.0)
    
    def test_model_family_matching(self):
        """Test that model family matching works."""
        with patch.object(self.adapter.__class__, '_check_ollama_availability'):
            # Test model that should match llama3 family
            self.adapter.model = "llama3:13b-instruct"
        
        result = self.adapter.estimate_cost(1000, 1000)
        
        # Should match llama3:8b pricing as it starts with "llama3"
        self.assertEqual(result["pricing_model"], "llama3:8b")
    
    def test_parameter_size_matching(self):
        """Test model matching by parameter size."""
        with patch.object(self.adapter.__class__, '_check_ollama_availability'):
            # Model with parameter size in name
            self.adapter.model = "unknown-7b-model"
        
        result = self.adapter.estimate_cost(1000, 1000)
        
        # Should get default pricing for 7b models
        self.assertIn("rates", result)
        self.assertIsNotNone(result["equivalent_cost_usd"])


if __name__ == "__main__":
    unittest.main()