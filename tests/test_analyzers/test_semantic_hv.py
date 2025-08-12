"""
Unit tests for semantic hypervector analysis.

TODOs addressed:
1. Test hypervector encoding accuracy
2. Test similarity threshold calibration  
3. Test handling of edge cases (empty functions, lambdas, etc.)
"""

import pytest
import numpy as np
from typing import List, Tuple
import ast

from tailchasing.semantic.hv_space import HVSpace
from tailchasing.semantic.encoder import FunctionFeatureExtractor as FunctionEncoder
from tailchasing.semantic.index import SemanticIndex
from tailchasing.core.issues import Issue


class TestHypervectorEncoding:
    """Test hypervector encoding of functions."""
    
    @pytest.fixture
    def hv_space(self):
        """Create a hypervector space for testing."""
        return HVSpace(dim=1024, bipolar=True, seed=42)
    
    @pytest.fixture
    def encoder(self, hv_space):
        """Create a function encoder."""
        return FunctionEncoder(hv_space)
    
    # TODO 1: Test hypervector encoding accuracy
    def test_encoding_preserves_similarity(self, encoder):
        """Test that similar functions have similar hypervectors."""
        # Very similar functions
        func1 = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
"""
        
        func2 = """
def compute_sum(values):
    result = 0
    for val in values:
        result += val
    return result
"""
        
        # Different function
        func3 = """
def find_maximum(numbers):
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val
"""
        
        tree1 = ast.parse(func1).body[0]
        tree2 = ast.parse(func2).body[0]
        tree3 = ast.parse(func3).body[0]
        
        hv1 = encoder.encode_function(tree1)
        hv2 = encoder.encode_function(tree2)
        hv3 = encoder.encode_function(tree3)
        
        # Calculate similarities
        sim_12 = encoder.space.similarity(hv1, hv2)
        sim_13 = encoder.space.similarity(hv1, hv3)
        sim_23 = encoder.space.similarity(hv2, hv3)
        
        # Similar functions should have higher similarity
        assert sim_12 > sim_13, "Similar functions should have higher similarity"
        assert sim_12 > sim_23, "Similar functions should have higher similarity"
        
        # Actual similarity should be quite high for near-duplicates
        assert sim_12 > 0.7, "Near-duplicate functions should have high similarity"
        
    def test_encoding_distinguishes_different_functions(self, encoder):
        """Test that different functions have different hypervectors."""
        functions = [
            "def add(a, b): return a + b",
            "def subtract(a, b): return a - b",
            "def multiply(a, b): return a * b",
            "def divide(a, b): return a / b if b != 0 else None",
        ]
        
        hypervectors = []
        for func_str in functions:
            tree = ast.parse(func_str).body[0]
            hv = encoder.encode_function(tree)
            hypervectors.append(hv)
        
        # Check that all pairs have reasonable distance
        for i in range(len(hypervectors)):
            for j in range(i + 1, len(hypervectors)):
                sim = encoder.space.similarity(hypervectors[i], hypervectors[j])
                # Different operations should have lower similarity
                assert sim < 0.6, f"Different functions should have lower similarity: {functions[i]} vs {functions[j]}"
    
    def test_encoding_components(self, encoder):
        """Test that different components contribute to encoding."""
        # Function with just name
        func_name_only = "def process(): pass"
        
        # Function with parameters
        func_with_params = "def process(data, options): pass"
        
        # Function with body
        func_with_body = """
def process():
    result = []
    for i in range(10):
        result.append(i * 2)
    return result
"""
        
        trees = [ast.parse(f).body[0] for f in [func_name_only, func_with_params, func_with_body]]
        hvs = [encoder.encode_function(t) for t in trees]
        
        # Each should be different
        for i in range(len(hvs)):
            for j in range(i + 1, len(hvs)):
                sim = encoder.space.similarity(hvs[i], hvs[j])
                assert sim < 0.95, "Different function structures should produce different encodings"
    
    # TODO 2: Test similarity threshold calibration
    def test_threshold_calibration(self):
        """Test calibration of similarity thresholds."""
        index = SemanticIndex(config={'z_threshold': 2.5, 'hv_dim': 1024})
        
        # Add a variety of functions
        test_functions = [
            ("sum_list", "def sum_list(lst): return sum(lst)"),
            ("sum_items", "def sum_items(items): return sum(items)"),  # Very similar
            ("count_items", "def count_items(items): return len(items)"),  # Somewhat similar
            ("reverse_list", "def reverse_list(lst): return lst[::-1]"),  # Different
            ("sort_list", "def sort_list(lst): return sorted(lst)"),  # Different
        ]
        
        encoder = FunctionEncoder(index.space)
        
        for name, func_str in test_functions:
            tree = ast.parse(func_str).body[0]
            hv = encoder.encode_function(tree)
            index.add(name, "test.py", tree.lineno, hv)
        
        # Compute background statistics
        mean, std = index.get_background_stats()
        
        # Statistics should be reasonable
        assert 0.3 < mean < 0.7, "Mean distance should be moderate"
        assert 0.05 < std < 0.3, "Standard deviation should be reasonable"
        
        # Find similar pairs
        similar_pairs = index.find_all_similar_pairs(z_threshold=2.0)
        
        # Should find the very similar pair
        pair_names = [(p[0].split('@')[0], p[1].split('@')[0]) for p in similar_pairs]
        assert ("sum_list", "sum_items") in pair_names or ("sum_items", "sum_list") in pair_names, \
            "Should detect very similar functions"
        
        # Should not find too many false positives
        assert len(similar_pairs) <= 2, "Should not have too many false positives with proper threshold"
    
    def test_adaptive_threshold(self):
        """Test that threshold adapts to codebase characteristics."""
        # Small codebase - need different threshold
        small_index = SemanticIndex(config={'z_threshold': 2.5, 'hv_dim': 1024})
        
        # Large codebase simulation
        large_index = SemanticIndex(config={'z_threshold': 2.5, 'hv_dim': 1024})
        
        encoder_small = FunctionEncoder(small_index.space)
        encoder_large = FunctionEncoder(large_index.space)
        
        # Add 5 functions to small index
        for i in range(5):
            tree = ast.parse(f"def func_{i}(x): return x + {i}").body[0]
            hv = encoder_small.encode_function(tree)
            small_index.add(f"func_{i}", "small.py", i, hv)
        
        # Add 100 functions to large index
        for i in range(100):
            tree = ast.parse(f"def func_{i}(x): return x + {i}").body[0]
            hv = encoder_large.encode_function(tree)
            large_index.add(f"func_{i}", "large.py", i, hv)
        
        # Background statistics should differ
        small_mean, small_std = small_index.get_background_stats()
        large_mean, large_std = large_index.get_background_stats()
        
        # Large codebase should have more stable statistics
        assert large_std <= small_std * 1.5, "Large codebase should have more stable statistics"
    
    # TODO 3: Test handling of edge cases
    def test_edge_case_encoding(self, encoder):
        """Test encoding of edge cases like empty functions, lambdas, etc."""
        edge_cases = [
            # Empty function
            ("empty", "def empty(): pass", True),
            
            # Function with just docstring
            ("docstring_only", '''def documented():
    """This function does something."""
    pass''', True),
            
            # Lambda (should skip)
            ("lambda", "lambda x: x + 1", False),
            
            # Nested function
            ("nested", """def outer():
    def inner():
        return 42
    return inner()""", True),
            
            # Generator function
            ("generator", """def gen():
    for i in range(10):
        yield i""", True),
            
            # Async function
            ("async", "async def fetch(): pass", True),
            
            # Property decorator
            ("property", """@property
def value(self):
    return self._value""", True),
        ]
        
        for name, code, should_encode in edge_cases:
            try:
                # Parse based on type
                if name == "lambda":
                    tree = ast.parse(code).body[0].value
                else:
                    tree = ast.parse(code).body[0]
                
                if should_encode:
                    hv = encoder.encode_function(tree)
                    assert hv is not None, f"Should encode {name}"
                    assert len(hv) == encoder.space.dim, f"Hypervector dimension mismatch for {name}"
                else:
                    # Should handle gracefully
                    pass
                    
            except Exception as e:
                if should_encode:
                    pytest.fail(f"Failed to encode {name}: {e}")
    
    def test_empty_function_handling(self, encoder):
        """Test specific handling of empty/stub functions."""
        empty_variants = [
            "def empty(): pass",
            "def empty(): ...",
            "def empty(): return",
            "def empty(): return None",
            '''def empty():
    """TODO: implement"""
    pass''',
        ]
        
        hypervectors = []
        for func_str in empty_variants:
            tree = ast.parse(func_str).body[0]
            hv = encoder.encode_function(tree)
            hypervectors.append(hv)
        
        # Empty functions should be similar to each other
        for i in range(len(hypervectors)):
            for j in range(i + 1, len(hypervectors)):
                sim = encoder.space.similarity(hypervectors[i], hypervectors[j])
                assert sim > 0.8, "Empty function variants should be detected as similar"


class TestSemanticDuplicateDetection:
    """Test semantic duplicate detection using hypervectors."""
    
    def test_semantic_duplicate_detection(self):
        """Test detection of semantic duplicates."""
        code = """
# Duplicate 1: Traditional loop
def calculate_total(items):
    total = 0
    for item in items:
        total += item.price
    return total

# Duplicate 2: List comprehension with sum
def compute_total_price(products):
    return sum(p.price for p in products)

# Duplicate 3: Reduce variant
def get_total_cost(items):
    from functools import reduce
    return reduce(lambda acc, item: acc + item.price, items, 0)

# Not duplicate: Different calculation
def calculate_average(items):
    if not items:
        return 0
    total = sum(item.price for item in items)
    return total / len(items)
"""
        
        tree = ast.parse(code)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        # Encode all functions
        space = HVSpace(dim=1024)
        encoder = FunctionEncoder(space)
        
        func_hvs = []
        for func in functions:
            hv = encoder.encode_function(func)
            func_hvs.append((func.name, hv))
        
        # Check similarities
        duplicates = []
        for i in range(len(func_hvs)):
            for j in range(i + 1, len(func_hvs)):
                name1, hv1 = func_hvs[i]
                name2, hv2 = func_hvs[j]
                sim = space.similarity(hv1, hv2)
                
                if sim > 0.7:  # Threshold for semantic similarity
                    duplicates.append((name1, name2, sim))
        
        # Should detect the three total calculation functions as similar
        duplicate_names = {name for pair in duplicates for name in pair[:2]}
        assert "calculate_total" in duplicate_names
        assert "compute_total_price" in duplicate_names
        assert "get_total_cost" in duplicate_names
        
        # Should not flag calculate_average as duplicate
        avg_pairs = [p for p in duplicates if "calculate_average" in p[:2]]
        assert len(avg_pairs) == 0, "calculate_average should not be flagged as duplicate"


class TestHypervectorPerformance:
    """Test performance characteristics of hypervector operations."""
    
    def test_encoding_performance(self):
        """Test that encoding is reasonably fast."""
        import time
        
        space = HVSpace(dim=8192)  # Larger dimension
        encoder = FunctionEncoder(space)
        
        # Generate a moderately complex function
        complex_func = """
def process_data(data, options=None):
    '''Process data with various transformations.'''
    if options is None:
        options = {}
    
    result = []
    for item in data:
        if isinstance(item, dict):
            processed = {k: v.upper() if isinstance(v, str) else v 
                        for k, v in item.items()}
        elif isinstance(item, list):
            processed = [x * 2 for x in item if isinstance(x, (int, float))]
        else:
            processed = str(item)
        
        if options.get('validate'):
            if not self._validate(processed):
                continue
        
        result.append(processed)
    
    return result
"""
        
        tree = ast.parse(complex_func).body[0]
        
        # Time encoding
        start = time.time()
        for _ in range(100):
            hv = encoder.encode_function(tree)
        elapsed = time.time() - start
        
        # Should be reasonably fast (< 1ms per encoding on average)
        assert elapsed < 0.1, f"Encoding too slow: {elapsed:.4f}s for 100 encodings"
    
    def test_similarity_computation_performance(self):
        """Test that similarity computation is fast."""
        import time
        
        space = HVSpace(dim=8192)
        
        # Generate random hypervectors
        hv1 = space.random()
        hv2 = space.random()
        
        # Time similarity computation
        start = time.time()
        for _ in range(10000):
            sim = space.similarity(hv1, hv2)
        elapsed = time.time() - start
        
        # Should be very fast (< 0.01ms per comparison)
        assert elapsed < 0.1, f"Similarity computation too slow: {elapsed:.4f}s for 10000 comparisons"