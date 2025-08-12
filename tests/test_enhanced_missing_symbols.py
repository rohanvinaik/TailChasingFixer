"""
Tests for enhanced missing symbol analyzer with stub generation.
"""

import ast
from pathlib import Path
from unittest.mock import MagicMock, patch
from textwrap import dedent

import pytest

from tailchasing.analyzers.enhanced_missing_symbols import (
    EnhancedMissingSymbolAnalyzer,
    CallSite,
    ParameterInfo,
    InferredSignature
)
from tailchasing.analyzers.base import AnalysisContext


def create_test_context(code: str, filename: str = "test.py"):
    """Create a test analysis context."""
    tree = ast.parse(code)
    
    ctx = MagicMock(spec=AnalysisContext)
    ctx.ast_index = {filename: tree}
    ctx.root_dir = Path(".")
    ctx.is_excluded.return_value = False
    
    return ctx, tree


class TestCallSiteCollection:
    """Test callsite information collection."""
    
    def test_collect_simple_function_call(self):
        """Test collecting a simple function call."""
        code = '''
def main():
    result = undefined_function(42, "hello")
    return result
'''
        ctx, _ = create_test_context(code)
        analyzer = EnhancedMissingSymbolAnalyzer()
        list(analyzer.run(ctx))
        
        assert "undefined_function" in analyzer.missing_symbols
        callsites = analyzer.missing_symbols["undefined_function"]
        assert len(callsites) == 1
        
        callsite = callsites[0]
        assert callsite.line == 3
        assert callsite.context == "main"
        assert callsite.in_assignment is True
        assert callsite.assignment_target == "result"
        assert len(callsite.args) == 2
    
    def test_collect_keyword_arguments(self):
        """Test collecting calls with keyword arguments."""
        code = '''
def process():
    missing_func(x=10, y=20, name="test")
'''
        ctx, _ = create_test_context(code)
        analyzer = EnhancedMissingSymbolAnalyzer()
        list(analyzer.run(ctx))
        
        callsites = analyzer.missing_symbols["missing_func"]
        assert len(callsites) == 1
        
        callsite = callsites[0]
        assert len(callsite.keywords) == 3
        keyword_names = [kw.arg for kw in callsite.keywords]
        assert "x" in keyword_names
        assert "y" in keyword_names
        assert "name" in keyword_names
    
    def test_collect_multiple_callsites(self):
        """Test collecting multiple callsites of the same function."""
        code = '''
def func1():
    validate_data("input1")
    
def func2():
    if validate_data("input2"):
        print("Valid")
    
def func3():
    return validate_data("input3")
'''
        ctx, _ = create_test_context(code)
        analyzer = EnhancedMissingSymbolAnalyzer()
        list(analyzer.run(ctx))
        
        callsites = analyzer.missing_symbols["validate_data"]
        assert len(callsites) == 3
        
        # Check different contexts
        contexts = [cs.context for cs in callsites]
        assert "func1" in contexts
        assert "func2" in contexts
        assert "func3" in contexts
        
        # Check usage patterns
        assert any(cs.in_condition for cs in callsites)  # func2
        assert any(cs.in_return for cs in callsites)      # func3
    
    def test_context_tracking(self):
        """Test that different usage contexts are tracked."""
        code = '''
def example():
    # In assignment
    x = unknown_func1()
    
    # In condition
    if unknown_func2():
        pass
    
    # In return
    return unknown_func3()
    
    # In comparison
    if unknown_func4() > 5:
        pass
'''
        ctx, _ = create_test_context(code)
        analyzer = EnhancedMissingSymbolAnalyzer()
        list(analyzer.run(ctx))
        
        # Check assignment context
        cs1 = analyzer.missing_symbols["unknown_func1"][0]
        assert cs1.in_assignment is True
        
        # Check condition context
        cs2 = analyzer.missing_symbols["unknown_func2"][0]
        assert cs2.in_condition is True
        
        # Check return context
        cs3 = analyzer.missing_symbols["unknown_func3"][0]
        assert cs3.in_return is True
        
        # Check comparison context
        cs4 = analyzer.missing_symbols["unknown_func4"][0]
        assert cs4.comparison_op == ">"


class TestSignatureInference:
    """Test signature inference from callsites."""
    
    def test_infer_parameter_count(self):
        """Test inferring parameter count from callsites."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        # Create mock callsites
        callsites = [
            CallSite("test.py", 1, 0, "main", 
                    args=[ast.Constant(1), ast.Constant(2)], keywords=[]),
            CallSite("test.py", 2, 0, "main",
                    args=[ast.Constant(3), ast.Constant(4), ast.Constant(5)], keywords=[])
        ]
        
        signature = analyzer._infer_signature("test_func", callsites)
        
        # Should infer max argument count (3)
        assert len(signature.parameters) == 3
        # Check parameter names (excluding 'self' if present)
        non_self_params = [p for p in signature.parameters if p.name != "self"]
        assert all(p.name.startswith("arg") or p.name.startswith("int_param") for p in non_self_params)
    
    def test_infer_keyword_parameters(self):
        """Test inferring keyword parameters."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        # Create callsites with keywords
        kw1 = ast.keyword(arg="name", value=ast.Constant("test"))
        kw2 = ast.keyword(arg="age", value=ast.Constant(25))
        
        callsites = [
            CallSite("test.py", 1, 0, "main", args=[], keywords=[kw1, kw2])
        ]
        
        signature = analyzer._infer_signature("test_func", callsites)
        
        # Should have keyword parameters
        param_names = [p.name for p in signature.parameters]
        assert "name" in param_names
        assert "age" in param_names
        
        # Check keyword-only flag
        name_param = [p for p in signature.parameters if p.name == "name"][0]
        assert name_param.is_keyword_only is True
    
    def test_infer_parameter_types(self):
        """Test inferring parameter types from argument values."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        # Create callsites with typed values
        callsites = [
            CallSite("test.py", 1, 0, "main",
                    args=[ast.Constant("hello"), ast.Constant(42)], keywords=[]),
            CallSite("test.py", 2, 0, "main",
                    args=[ast.Constant("world"), ast.Constant(100)], keywords=[])
        ]
        
        signature = analyzer._infer_signature("test_func", callsites)
        
        # Should infer consistent types
        assert signature.parameters[0].inferred_type == "str"
        assert signature.parameters[1].inferred_type == "int"
    
    def test_infer_return_type_from_condition(self):
        """Test inferring return type from usage in conditions."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        callsites = [
            CallSite("test.py", 1, 0, "main", args=[], keywords=[],
                    in_condition=True)
        ]
        
        signature = analyzer._infer_signature("check_something", callsites)
        
        # Should infer bool return type
        assert signature.return_type == "bool"
    
    def test_infer_method_signature(self):
        """Test inferring that a function is a method."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        # Function with underscore prefix and class context
        callsites = [
            CallSite("test.py", 1, 0, "MyClass.method", args=[], keywords=[])
        ]
        
        signature = analyzer._infer_signature("_private_method", callsites)
        
        assert signature.is_method is True
        # Should have 'self' parameter
        assert signature.parameters[0].name == "self"
    
    def test_mixed_type_inference(self):
        """Test handling mixed types with Optional."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        # Callsites with mixed types including None
        callsites = [
            CallSite("test.py", 1, 0, "main",
                    args=[ast.Constant("text")], keywords=[]),
            CallSite("test.py", 2, 0, "main",
                    args=[ast.Constant(None)], keywords=[])
        ]
        
        signature = analyzer._infer_signature("maybe_func", callsites)
        
        # Should infer Optional type
        assert "Optional" in signature.parameters[0].inferred_type
        assert "str" in signature.parameters[0].inferred_type


class TestStubGeneration:
    """Test stub file generation."""
    
    def test_generate_simple_function_stub(self):
        """Test generating a simple function stub."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        # Add inferred signature
        analyzer.inferred_signatures["simple_func"] = InferredSignature(
            name="simple_func",
            parameters=[
                ParameterInfo("arg1", 0, inferred_type="str"),
                ParameterInfo("arg2", 1, inferred_type="int")
            ],
            return_type="bool"
        )
        
        stub_content = analyzer.generate_stub_file(Path("test.py"))
        
        assert "def simple_func(arg1: str, arg2: int) -> bool:" in stub_content
        assert "raise NotImplementedError" in stub_content
        assert "from typing import" in stub_content
    
    def test_generate_async_function_stub(self):
        """Test generating an async function stub."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        analyzer.inferred_signatures["async_func"] = InferredSignature(
            name="async_func",
            parameters=[],
            return_type="Any",
            is_async=True
        )
        
        stub_content = analyzer.generate_stub_file(Path("test.py"))
        
        assert "async def async_func() -> Any:" in stub_content
    
    def test_generate_method_stub(self):
        """Test generating a method stub."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        analyzer.inferred_signatures["MyClass.my_method"] = InferredSignature(
            name="my_method",
            parameters=[
                ParameterInfo("self", 0),
                ParameterInfo("value", 1, inferred_type="int")
            ],
            return_type=None,
            is_method=True
        )
        
        stub_content = analyzer.generate_stub_file(Path("test.py"))
        
        assert "class MyClass:" in stub_content
        assert "def my_method(self, value: int):" in stub_content
    
    def test_generate_with_optional_types(self):
        """Test generating stubs with Optional types."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        analyzer.inferred_signatures["optional_func"] = InferredSignature(
            name="optional_func",
            parameters=[
                ParameterInfo("maybe_str", 0, inferred_type="Optional[str]")
            ],
            return_type="Optional[int]"
        )
        
        stub_content = analyzer.generate_stub_file(Path("test.py"))
        
        assert "maybe_str: Optional[str]" in stub_content
        assert "-> Optional[int]:" in stub_content
    
    def test_generate_with_union_types(self):
        """Test generating stubs with Union types."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        analyzer.inferred_signatures["union_func"] = InferredSignature(
            name="union_func",
            parameters=[
                ParameterInfo("mixed", 0, inferred_type="Union[int, str]")
            ],
            return_type="Union[bool, None]"
        )
        
        stub_content = analyzer.generate_stub_file(Path("test.py"))
        
        assert "mixed: Union[int, str]" in stub_content
        assert "-> Union[bool, None]:" in stub_content
    
    def test_generate_with_docstring(self):
        """Test that generated stubs include docstrings."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        analyzer.inferred_signatures["documented_func"] = InferredSignature(
            name="documented_func",
            parameters=[ParameterInfo("data", 0, inferred_type="str")],
            return_type="bool",
            docstring="Stub for missing function 'documented_func'.\n\nArgs:\n    data (str): TODO: Add description"
        )
        
        stub_content = analyzer.generate_stub_file(Path("test.py"))
        
        assert '"""' in stub_content
        assert "TODO: Add description" in stub_content
    
    def test_confidence_in_stub(self):
        """Test that confidence level is included in stub comments."""
        analyzer = EnhancedMissingSymbolAnalyzer()
        
        analyzer.inferred_signatures["confident_func"] = InferredSignature(
            name="confident_func",
            parameters=[],
            confidence=0.85
        )
        
        stub_content = analyzer.generate_stub_file(Path("test.py"))
        
        assert "Confidence: 85%" in stub_content


class TestIntegration:
    """Integration tests for the complete workflow."""
    
    def test_complete_stub_generation_workflow(self):
        """Test the complete workflow from code to stub generation."""
        code = '''
class DataProcessor:
    def process(self):
        # Multiple calls with different signatures
        result1 = validate_input("data", strict=True)
        result2 = validate_input("other", strict=False, max_length=100)
        
        if check_permission(self.user_id):
            transformed = transform_data(result1, result2)
            return transformed
        
        return None
        
def main():
    processor = DataProcessor()
    if processor.process():
        log_success("Processing complete")
'''
        
        ctx, _ = create_test_context(code)
        analyzer = EnhancedMissingSymbolAnalyzer()
        issues = list(analyzer.run(ctx))
        
        # Should detect missing symbols
        missing_names = [sig for sig in analyzer.inferred_signatures.keys()]
        assert "validate_input" in missing_names
        assert "check_permission" in missing_names
        assert "transform_data" in missing_names
        assert "log_success" in missing_names
        
        # Check validate_input signature
        validate_sig = analyzer.inferred_signatures["validate_input"]
        assert len(validate_sig.parameters) >= 1  # At least the positional arg
        param_names = [p.name for p in validate_sig.parameters]
        assert "strict" in param_names
        assert "max_length" in param_names
        
        # Check return type inference
        check_sig = analyzer.inferred_signatures["check_permission"]
        assert check_sig.return_type == "bool"  # Used in condition
        
        # Generate stub file
        stub_content = analyzer.generate_stub_file(Path("stubs.py"))
        
        # Verify stub content
        assert "def validate_input" in stub_content
        assert "def check_permission" in stub_content
        assert "def transform_data" in stub_content
        assert "def log_success" in stub_content
        assert "raise NotImplementedError" in stub_content