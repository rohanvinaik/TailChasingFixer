"""
Unit tests for cargo cult pattern detection.

TODOs addressed:
1. Test detection of unnecessary abstractions
2. Test wrapper function identification
3. Test over-engineering pattern detection
4. Test distinction between legitimate and cargo-cult patterns
5. Test fix suggestions for simplification
"""

import pytest
import ast
from pathlib import Path

from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.symbols import SymbolTable
from tailchasing.core.issues import Issue


class CargoCultAnalyzer:
    """Analyzer for cargo cult programming patterns."""
    
    name = "cargo_cult"
    
    def run(self, ctx: AnalysisContext):
        """Detect cargo cult patterns."""
        for file_path, tree in ctx.ast_index.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check for trivial wrappers
                    if self._is_trivial_wrapper(node):
                        yield Issue(
                            kind="wrapper_abstraction",
                            message=f"Trivial wrapper function: {node.name}",
                            severity=2,
                            file=file_path,
                            line=node.lineno,
                            suggestions=[
                                "Remove the wrapper and call the wrapped function directly",
                                "Add actual logic if the wrapper is intended to add functionality",
                                "Use functools.partial if you need to preset arguments"
                            ]
                        )
                    
                    # Check for unnecessary abstraction
                    if self._is_unnecessary_abstraction(node, ctx):
                        yield Issue(
                            kind="unnecessary_abstraction",
                            message=f"Unnecessary abstraction: {node.name}",
                            severity=2,
                            file=file_path,
                            line=node.lineno,
                            suggestions=[
                                "Inline this function if it's only used once",
                                "Simplify the abstraction hierarchy",
                                "Follow YAGNI (You Aren't Gonna Need It) principle"
                            ]
                        )
    
    def _is_trivial_wrapper(self, node: ast.FunctionDef) -> bool:
        """Check if function is just a trivial wrapper."""
        if len(node.body) != 1:
            return False
            
        stmt = node.body[0]
        
        # Check for simple return of function call
        if isinstance(stmt, ast.Return) and isinstance(stmt.value, ast.Call):
            # Check if it's just passing through arguments
            call = stmt.value
            if isinstance(call.func, ast.Name):
                # Simple wrapper pattern: def wrapper(*args): return wrapped(*args)
                if (len(node.args.args) > 0 and 
                    node.args.vararg and 
                    len(call.args) == 1 and
                    isinstance(call.args[0], ast.Starred)):
                    return True
                    
                # Direct pass-through: def wrapper(x, y): return other(x, y)
                if len(node.args.args) == len(call.args):
                    args_match = all(
                        isinstance(arg, ast.Name) and arg.id == param.arg
                        for arg, param in zip(call.args, node.args.args)
                    )
                    if args_match:
                        return True
                        
        return False
    
    def _is_unnecessary_abstraction(self, node: ast.FunctionDef, ctx: AnalysisContext) -> bool:
        """Check if function represents unnecessary abstraction."""
        # Single-use functions that don't add value
        if node.name.startswith("_"):  # Private function
            # Check how many times it's called
            call_count = self._count_function_calls(node.name, ctx)
            if call_count <= 1 and len(node.body) <= 3:
                return True
                
        # Factory functions that just return a constructor call
        if "factory" in node.name.lower() or "create" in node.name.lower():
            if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                if isinstance(node.body[0].value, ast.Call):
                    return True
                    
        return False
    
    def _count_function_calls(self, func_name: str, ctx: AnalysisContext) -> int:
        """Count how many times a function is called."""
        count = 0
        for tree in ctx.ast_index.values():
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id == func_name:
                        count += 1
        return count


class TestCargoCultDetection:
    """Test detection of cargo cult programming patterns."""
    
    @pytest.fixture
    def analyzer(self):
        """Create a CargoCultAnalyzer instance."""
        return CargoCultAnalyzer()
    
    @pytest.fixture
    def context(self, tmp_path):
        """Create a test analysis context."""
        return AnalysisContext(
            config={},
            root_dir=tmp_path,
            file_paths=[],
            ast_index={},
            symbol_table=SymbolTable(),
            source_cache={},
            cache={}
        )
    
    # TODO 1: Test detection of unnecessary abstractions
    @pytest.mark.parametrize("code,should_detect", [
        # Unnecessary single-use abstraction
        ("""
def _format_string(s):
    return s.strip().lower()

def process_input(data):
    return _format_string(data)
""", True),
        
        # Legitimate abstraction (used multiple times)
        ("""
def _validate_input(data):
    if not data:
        raise ValueError("Empty input")
    return data.strip()

def process_a(data):
    clean = _validate_input(data)
    return clean.upper()

def process_b(data):
    clean = _validate_input(data)
    return clean.lower()
""", False),
        
        # Over-abstracted factory
        ("""
def create_processor():
    return DataProcessor()

processor = create_processor()
""", True),
        
        # Legitimate factory with configuration
        ("""
def create_processor(config):
    processor = DataProcessor()
    processor.configure(config)
    return processor
""", False),
    ])
    def test_unnecessary_abstraction_detection(self, analyzer, context, code, should_detect):
        """Test detection of unnecessary abstractions."""
        tree = ast.parse(code)
        context.ast_index["test.py"] = tree
        
        issues = list(analyzer.run(context))
        
        has_unnecessary = any(i.kind == "unnecessary_abstraction" for i in issues)
        
        if should_detect:
            assert has_unnecessary, f"Should detect unnecessary abstraction in: {code}"
        else:
            assert not has_unnecessary, f"Should not detect unnecessary abstraction in: {code}"
    
    # TODO 2: Test wrapper function identification
    def test_wrapper_detection(self, analyzer, context):
        """Test identification of trivial wrapper functions."""
        test_cases = [
            # Trivial wrapper
            ("""
def my_sum(*args):
    return sum(*args)
""", True),
            
            # Direct pass-through
            ("""
def calculate(x, y):
    return add(x, y)
""", True),
            
            # Wrapper with added logic (legitimate)
            ("""
def safe_divide(x, y):
    if y == 0:
        return None
    return x / y
""", False),
            
            # Wrapper with logging (legitimate)
            ("""
def logged_operation(data):
    logger.info(f"Processing {len(data)} items")
    result = process(data)
    logger.info(f"Processed successfully")
    return result
""", False),
        ]
        
        for code, should_detect in test_cases:
            tree = ast.parse(code)
            context.ast_index["test.py"] = tree
            
            issues = list(analyzer.run(context))
            has_wrapper = any(i.kind == "wrapper_abstraction" for i in issues)
            
            assert has_wrapper == should_detect, \
                f"Wrapper detection failed for: {code}"
    
    # TODO 3: Test over-engineering pattern detection
    def test_over_engineering_patterns(self, analyzer, context):
        """Test detection of over-engineered solutions."""
        # Abstract factory for simple objects
        over_engineered = """
class ProcessorFactory:
    def create_processor(self, type):
        if type == "simple":
            return SimpleProcessor()
        elif type == "complex":
            return ComplexProcessor()

class ProcessorBuilder:
    def __init__(self):
        self.processor = None
    
    def with_type(self, type):
        self.processor = ProcessorFactory().create_processor(type)
        return self
    
    def build(self):
        return self.processor

# Usage
processor = ProcessorBuilder().with_type("simple").build()
"""
        
        tree = ast.parse(over_engineered)
        context.ast_index["test.py"] = tree
        
        issues = list(analyzer.run(context))
        
        # Should detect multiple patterns
        assert len(issues) >= 1, "Should detect over-engineering patterns"
        
    # TODO 4: Test distinction between legitimate and cargo-cult patterns
    def test_legitimate_vs_cargo_cult(self, analyzer, context):
        """Test that legitimate patterns are not flagged as cargo cult."""
        legitimate_patterns = [
            # Strategy pattern with actual different strategies
            """
class SortStrategy:
    def sort(self, data):
        raise NotImplementedError

class QuickSort(SortStrategy):
    def sort(self, data):
        # Actual quicksort implementation
        if len(data) <= 1:
            return data
        pivot = data[0]
        less = [x for x in data[1:] if x < pivot]
        greater = [x for x in data[1:] if x >= pivot]
        return self.sort(less) + [pivot] + self.sort(greater)

class MergeSort(SortStrategy):
    def sort(self, data):
        # Actual mergesort implementation
        if len(data) <= 1:
            return data
        mid = len(data) // 2
        left = self.sort(data[:mid])
        right = self.sort(data[mid:])
        return self.merge(left, right)
    
    def merge(self, left, right):
        result = []
        i = j = 0
        while i < len(left) and j < len(right):
            if left[i] <= right[j]:
                result.append(left[i])
                i += 1
            else:
                result.append(right[j])
                j += 1
        result.extend(left[i:])
        result.extend(right[j:])
        return result
""",
            # Decorator with actual functionality
            """
def retry_on_error(max_attempts=3):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    time.sleep(2 ** attempt)
            return None
        return wrapper
    return decorator
""",
        ]
        
        for code in legitimate_patterns:
            tree = ast.parse(code)
            context.ast_index["test.py"] = tree
            
            issues = list(analyzer.run(context))
            
            # Should not flag legitimate patterns as cargo cult
            cargo_cult_issues = [i for i in issues if "wrapper" in i.kind or "unnecessary" in i.kind]
            assert len(cargo_cult_issues) == 0, \
                f"Should not flag legitimate pattern as cargo cult: {code[:50]}..."
    
    # TODO 5: Test fix suggestions for simplification
    def test_simplification_suggestions(self, analyzer, context):
        """Test that appropriate simplification suggestions are provided."""
        code_with_issues = [
            ("""
def get_data():
    return fetch_data()
""", ["Remove the wrapper", "call.*directly"]),
            
            ("""
def create_object():
    return Object()
""", ["Inline", "simplify"]),
            
            ("""
def process_wrapper(x, y, z):
    return process(x, y, z)
""", ["Remove.*wrapper", "functools.partial"]),
        ]
        
        for code, expected_keywords in code_with_issues:
            tree = ast.parse(code)
            context.ast_index["test.py"] = tree
            
            issues = list(analyzer.run(context))
            
            assert len(issues) > 0, f"Should detect issue in: {code}"
            
            suggestions = issues[0].suggestions
            suggestions_text = " ".join(suggestions).lower()
            
            for keyword in expected_keywords:
                assert any(keyword.lower() in s.lower() for s in suggestions), \
                    f"Should suggest '{keyword}' for: {code}"


class TestOverEngineeringMetrics:
    """Test metrics for measuring over-engineering."""
    
    def test_abstraction_depth_metric(self):
        """Test measurement of abstraction hierarchy depth."""
        code = """
class AbstractBase:
    pass

class MiddleLayer(AbstractBase):
    pass

class ConcreteImpl(MiddleLayer):
    pass

class ActualUsage(ConcreteImpl):
    pass
"""
        tree = ast.parse(code)
        
        # Count inheritance depth
        depth = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if node.bases:
                    depth = max(depth, len(node.bases) + 1)
        
        # Deep inheritance might indicate over-engineering
        assert depth >= 3, "Should detect deep inheritance hierarchy"
    
    def test_indirection_metric(self):
        """Test measurement of unnecessary indirection levels."""
        code = """
def step1(data):
    return step2(data)

def step2(data):
    return step3(data)

def step3(data):
    return step4(data)

def step4(data):
    return data.process()
"""
        tree = ast.parse(code)
        
        # Count chain of single-return functions
        indirection_count = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                if len(node.body) == 1 and isinstance(node.body[0], ast.Return):
                    if isinstance(node.body[0].value, ast.Call):
                        indirection_count += 1
        
        assert indirection_count >= 4, "Should detect excessive indirection"