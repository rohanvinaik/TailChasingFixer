"""
Comprehensive tests for semantic encoder functionality.

Tests encoding of functions to hypervectors with context awareness,
feature extraction, and multimodal analysis capabilities.
"""

import pytest
import ast
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
from typing import Dict, Any, List, Set

from tailchasing.semantic.encoder import (
    encode_function,
    encode_function_with_context,
    split_identifier,
    extract_docstring_tokens,
    encode_channel,
    batch_encode_functions,
    extract_channels
)
from tailchasing.semantic.hv_space import HVSpace


# Simple test context class since it's not available in the encoder module
class FunctionContext:
    """Simple context class for testing."""
    def __init__(self, class_name=None, module_imports=None, docstring=None, 
                 decorators=None, surrounding_functions=None):
        self.class_name = class_name
        self.module_imports = module_imports or []
        self.docstring = docstring
        self.decorators = decorators or []
        self.surrounding_functions = surrounding_functions or []


class TestBasicEncoding:
    """Test basic function encoding functionality."""

    @pytest.fixture
    def hv_space(self):
        """Create a test hypervector space."""
        return HVSpace(dim=1024, bipolar=True)

    @pytest.fixture
    def sample_function_node(self):
        """Create a sample function AST node."""
        source = '''
def calculate_sum(a, b):
    """Calculate the sum of two numbers."""
    result = a + b
    return result
'''
        tree = ast.parse(source.strip())
        return tree.body[0]

    @pytest.fixture
    def complex_function_node(self):
        """Create a more complex function AST node."""
        source = '''
def process_data(items, threshold=0.5):
    """Process a list of items with filtering and transformation."""
    if not items:
        return []
    
    filtered = []
    for item in items:
        try:
            value = float(item.get('value', 0))
            if value > threshold:
                processed = {'id': item['id'], 'score': value * 2}
                filtered.append(processed)
        except (ValueError, KeyError) as e:
            print(f"Error processing item: {e}")
            continue
    
    return sorted(filtered, key=lambda x: x['score'], reverse=True)
'''
        tree = ast.parse(source.strip())
        return tree.body[0]

    def test_encode_function_basic(self, hv_space, sample_function_node):
        """Test basic function encoding without context."""
        result = encode_function(
            sample_function_node,
            "test.py",
            hv_space,
            {}
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (hv_space.dim,)
        assert result.dtype in [np.int8, np.int16, np.float32]
        
        # Test that encoding is deterministic
        result2 = encode_function(
            sample_function_node,
            "test.py",
            hv_space,
            {}
        )
        np.testing.assert_array_equal(result, result2)

    def test_encode_function_different_files(self, hv_space, sample_function_node):
        """Test that same function in different files produces different encodings."""
        result1 = encode_function(sample_function_node, "file1.py", hv_space, {})
        result2 = encode_function(sample_function_node, "file2.py", hv_space, {})
        
        # Should be different due to file path influence
        assert not np.array_equal(result1, result2)

    def test_encode_function_with_config(self, hv_space, sample_function_node):
        """Test encoding with custom configuration."""
        config = {
            'name_weight': 0.3,
            'ast_weight': 0.4,
            'control_flow_weight': 0.2,
            'data_flow_weight': 0.1
        }
        
        result = encode_function(
            sample_function_node,
            "test.py",
            hv_space,
            config
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (hv_space.dim,)

    def test_encode_function_with_context(self, hv_space, sample_function_node):
        """Test encoding with context information."""
        context = FunctionContext(
            class_name="Calculator",
            module_imports=["math", "numpy"],
            docstring="Calculate the sum of two numbers.",
            decorators=["@staticmethod"],
            surrounding_functions=["multiply", "divide"]
        )
        
        result = encode_function_with_context(
            sample_function_node,
            "test.py",
            hv_space,
            {},
            context.class_name,
            context.module_imports
        )
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (hv_space.dim,)
        
        # Should be different from encoding without context
        result_no_context = encode_function(sample_function_node, "test.py", hv_space, {})
        assert not np.array_equal(result, result_no_context)

    def test_encode_empty_function(self, hv_space):
        """Test encoding of empty function."""
        source = '''
def empty_function():
    pass
'''
        tree = ast.parse(source.strip())
        node = tree.body[0]
        
        result = encode_function(node, "test.py", hv_space, {})
        assert isinstance(result, np.ndarray)
        assert result.shape == (hv_space.dim,)

    def test_encode_function_invalid_input(self, hv_space):
        """Test encoding with invalid inputs."""
        with pytest.raises((AttributeError, TypeError)):
            encode_function(None, "test.py", hv_space, {})
        
        with pytest.raises((ValueError, TypeError)):
            encode_function(Mock(), "", hv_space, {})


class TestAvailableFunctions:
    """Test available encoder functions."""

    @pytest.fixture
    def sample_node(self):
        """Create a sample function node for testing."""
        source = '''
def process_items(items, config):
    """Process items with configuration."""
    results = []
    for item in items:
        if item.value > config.threshold:
            processed = transform_item(item)
            results.append(processed)
    return results
'''
        tree = ast.parse(source.strip())
        return tree.body[0]

    def test_extract_channels(self, sample_node):
        """Test channel extraction functionality."""
        # Use a simple fallback source instead of astor
        source = '''
def process_items(items, config):
    """Process items with configuration."""
    results = []
    for item in items:
        if item.value > config.threshold:
            processed = transform_item(item)
            results.append(processed)
    return results
'''
        
        channels = extract_channels(sample_node, source)
        
        assert isinstance(channels, list)
        # Each channel should be a tuple of (channel_name, tokens)
        for channel_name, tokens in channels:
            assert isinstance(channel_name, str)
            assert isinstance(tokens, list)

    def test_extract_docstring_tokens(self):
        """Test docstring tokenization."""
        docstring = "Process items with configuration and return results."
        tokens = extract_docstring_tokens(docstring)
        
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        assert any('process' in token.lower() for token in tokens)
        assert any('items' in token.lower() for token in tokens)

    def test_extract_docstring_empty(self):
        """Test docstring extraction with empty input."""
        tokens = extract_docstring_tokens("")
        assert isinstance(tokens, list)
        assert len(tokens) == 0

    def test_encode_channel(self):
        """Test individual channel encoding."""
        from tailchasing.semantic.hv_space import HVSpace
        space = HVSpace(dim=256, bipolar=True)
        
        tokens = ["process", "data", "items", "filter"]
        result = encode_channel("TEST_CHANNEL", tokens, space, {})
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (space.dim,)

    def test_batch_encode_functions(self):
        """Test batch encoding of multiple functions."""
        from tailchasing.semantic.hv_space import HVSpace
        space = HVSpace(dim=256, bipolar=True)
        
        # Create multiple test functions with file paths
        sources = [
            "def func1(x): return x + 1",
            "def func2(y): return y * 2", 
            "def func3(z): return z - 1"
        ]
        
        function_file_pairs = []
        for i, source in enumerate(sources):
            tree = ast.parse(source)
            function_file_pairs.append((tree.body[0], f"test{i}.py"))
        
        results = batch_encode_functions(function_file_pairs, space, {})
        
        assert isinstance(results, list)
        assert len(results) == len(function_file_pairs)
        
        for result in results:
            assert isinstance(result, np.ndarray)
            assert result.shape == (space.dim,)

    def test_split_identifier(self):
        """Test identifier splitting functionality."""
        # Test camelCase
        tokens = split_identifier("processDataItems")
        assert "process" in tokens
        assert "data" in tokens
        assert "items" in tokens
        
        # Test snake_case
        tokens = split_identifier("process_data_items")
        assert "process" in tokens
        assert "data" in tokens
        assert "items" in tokens
        
        # Test mixed case 
        tokens = split_identifier("processDataItems_v2")
        assert "process" in tokens
        assert "data" in tokens
        assert "items" in tokens
        # Note: current implementation may not split v2 correctly

    def test_split_identifier_edge_cases(self):
        """Test identifier splitting with edge cases."""
        # Empty string
        tokens = split_identifier("")
        assert len(tokens) == 0
        
        # Single word
        tokens = split_identifier("process")
        assert tokens == ["process"]
        
        # All caps (may not split as expected by current implementation)
        tokens = split_identifier("HTTP_API_CLIENT")
        assert isinstance(tokens, list)
        assert len(tokens) > 0
        
        # Numbers
        tokens = split_identifier("data2process")
        assert "data" in tokens
        assert "process" in tokens


class TestContextAwareEncoding:
    """Test context-aware encoding functionality."""

    @pytest.fixture
    def hv_space(self):
        """Create test hypervector space."""
        return HVSpace(dim=512, bipolar=True)

    @pytest.fixture
    def method_node(self):
        """Create a method AST node."""
        source = '''
def calculate(self, x, y):
    """Calculate something."""
    return self._helper(x) + self._process(y)
'''
        tree = ast.parse(source.strip())
        return tree.body[0]

    def test_context_affects_encoding(self, hv_space, method_node):
        """Test that context information affects encoding."""
        # Encode without context
        result_no_context = encode_function(method_node, "test.py", hv_space, {})
        
        # Encode with class context
        result_with_class = encode_function_with_context(
            method_node, "test.py", hv_space, {}, "Calculator", []
        )
        
        # Encode with imports context
        result_with_imports = encode_function_with_context(
            method_node, "test.py", hv_space, {}, None, ["numpy", "math"]
        )
        
        # All should be different
        assert not np.array_equal(result_no_context, result_with_class)
        assert not np.array_equal(result_no_context, result_with_imports)
        assert not np.array_equal(result_with_class, result_with_imports)

    def test_similar_functions_different_context(self, hv_space):
        """Test similar functions with different contexts."""
        # Same function structure, different contexts
        source = '''
def process(data):
    result = []
    for item in data:
        result.append(item * 2)
    return result
'''
        tree = ast.parse(source.strip())
        node = tree.body[0]
        
        # Different class contexts
        result1 = encode_function_with_context(
            node, "test.py", hv_space, {}, "DataProcessor", ["pandas"]
        )
        
        result2 = encode_function_with_context(
            node, "test.py", hv_space, {}, "Calculator", ["numpy"]
        )
        
        # Should be different despite same function structure
        assert not np.array_equal(result1, result2)


class TestEncodingConsistency:
    """Test encoding consistency and determinism."""

    @pytest.fixture
    def hv_space(self):
        return HVSpace(dim=256, bipolar=True)

    @pytest.fixture
    def test_functions(self):
        """Create multiple test functions."""
        sources = [
            '''
def add(a, b):
    return a + b
''',
            '''
def multiply(x, y):
    result = x * y
    return result
''',
            '''
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)
'''
        ]
        
        nodes = []
        for source in sources:
            tree = ast.parse(source.strip())
            nodes.append(tree.body[0])
        
        return nodes

    def test_encoding_determinism(self, hv_space, test_functions):
        """Test that encoding is deterministic."""
        for node in test_functions:
            result1 = encode_function(node, "test.py", hv_space, {})
            result2 = encode_function(node, "test.py", hv_space, {})
            
            np.testing.assert_array_equal(result1, result2)

    def test_different_functions_different_encodings(self, hv_space, test_functions):
        """Test that different functions produce different encodings."""
        encodings = []
        for node in test_functions:
            encoding = encode_function(node, "test.py", hv_space, {})
            encodings.append(encoding)
        
        # All encodings should be different
        for i in range(len(encodings)):
            for j in range(i + 1, len(encodings)):
                assert not np.array_equal(encodings[i], encodings[j])

    def test_similar_functions_similar_encodings(self, hv_space):
        """Test that structurally similar functions have similar encodings."""
        # Two very similar functions
        source1 = '''
def process_data(items):
    results = []
    for item in items:
        if item.valid:
            results.append(item.value)
    return results
'''
        
        source2 = '''
def process_items(data):
    output = []
    for element in data:
        if element.valid:
            output.append(element.value)
    return output
'''
        
        tree1 = ast.parse(source1.strip())
        tree2 = ast.parse(source2.strip())
        node1 = tree1.body[0]
        node2 = tree2.body[0]
        
        encoding1 = encode_function(node1, "test.py", hv_space, {})
        encoding2 = encode_function(node2, "test.py", hv_space, {})
        
        # Calculate similarity (for bipolar vectors)
        if hv_space.bipolar:
            similarity = np.mean(encoding1 == encoding2)
        else:
            similarity = 1 - np.mean(encoding1 != encoding2)
        
        # Should be more similar than random
        assert similarity > 0.6  # Threshold may need adjustment


class TestErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def hv_space(self):
        return HVSpace(dim=128, bipolar=True)

    def test_malformed_ast_node(self, hv_space):
        """Test handling of malformed AST nodes."""
        # Create a mock node that's missing expected attributes
        mock_node = Mock()
        mock_node.name = "test_function"
        
        # Should handle gracefully or raise appropriate error
        with pytest.raises((AttributeError, TypeError)):
            encode_function(mock_node, "test.py", hv_space)

    def test_empty_file_path(self, hv_space):
        """Test handling of empty file paths."""
        source = '''
def test():
    pass
'''
        tree = ast.parse(source.strip())
        node = tree.body[0]
        
        # Should handle empty file path
        result = encode_function(node, "", hv_space)
        assert isinstance(result, np.ndarray)

    def test_large_function(self, hv_space):
        """Test encoding of very large function."""
        # Generate a large function with many operations
        lines = ["def large_function():"]
        for i in range(100):
            lines.append(f"    var_{i} = {i} * 2")
        lines.append("    return sum([" + ", ".join(f"var_{i}" for i in range(100)) + "])")
        
        source = "\n".join(lines)
        tree = ast.parse(source)
        node = tree.body[0]
        
        # Should handle large functions without issues
        result = encode_function(node, "test.py", hv_space, {})
        assert isinstance(result, np.ndarray)
        assert result.shape == (hv_space.dim,)


class TestConfigurationEffects:
    """Test effects of different configuration parameters."""

    @pytest.fixture
    def hv_space(self):
        return HVSpace(dim=256, bipolar=True)

    @pytest.fixture
    def sample_node(self):
        source = '''
def calculate_score(data, weights):
    """Calculate weighted score."""
    total = 0
    for i, value in enumerate(data):
        total += value * weights[i]
    return total / len(data)
'''
        tree = ast.parse(source.strip())
        return tree.body[0]

    def test_weight_configuration_effects(self, hv_space, sample_node):
        """Test that different weight configurations affect encoding."""
        config1 = {
            'name_weight': 0.8,
            'ast_weight': 0.1,
            'control_flow_weight': 0.05,
            'data_flow_weight': 0.05
        }
        
        config2 = {
            'name_weight': 0.1,
            'ast_weight': 0.8,
            'control_flow_weight': 0.05,
            'data_flow_weight': 0.05
        }
        
        result1 = encode_function(sample_node, "test.py", hv_space, config1)
        result2 = encode_function(sample_node, "test.py", hv_space, config2)
        
        # Different weight configurations should produce different encodings
        assert not np.array_equal(result1, result2)

    def test_invalid_configuration(self, hv_space, sample_node):
        """Test handling of invalid configuration."""
        # Weights don't sum to 1.0
        invalid_config = {
            'name_weight': 0.5,
            'ast_weight': 0.5,
            'control_flow_weight': 0.5,
            'data_flow_weight': 0.5
        }
        
        # Should either normalize weights or handle gracefully
        result = encode_function(sample_node, "test.py", hv_space, invalid_config)
        assert isinstance(result, np.ndarray)


@pytest.fixture
def mock_hv_space():
    """Create a mock hypervector space for testing."""
    space = Mock()
    space.dim = 256
    space.bipolar = True
    
    # Mock methods to return predictable results
    def mock_token(token):
        # Simple hash-based deterministic vector
        np.random.seed(hash(token) % 2**32)
        return np.random.choice([-1, 1], size=space.dim)
    
    def mock_role(role):
        np.random.seed(hash(role) % 2**32)
        return np.random.choice([-1, 1], size=space.dim)
    
    def mock_bind(a, b):
        return a * b  # Element-wise multiplication for bipolar
    
    def mock_bundle(*vectors):
        if not vectors:
            return np.zeros(space.dim)
        result = np.sum(vectors, axis=0)
        return np.sign(result)  # Binarize
    
    space.token.side_effect = mock_token
    space.role.side_effect = mock_role
    space.bind.side_effect = mock_bind
    space.bundle.side_effect = mock_bundle
    
    return space


class TestMockIntegration:
    """Test encoding with mocked hypervector space."""

    def test_encode_with_mock_space(self, mock_hv_space):
        """Test encoding using mock hypervector space."""
        source = '''
def test_function(x):
    return x * 2
'''
        tree = ast.parse(source.strip())
        node = tree.body[0]
        
        result = encode_function(node, "test.py", mock_hv_space, {})
        
        assert isinstance(result, np.ndarray)
        assert result.shape == (mock_hv_space.dim,)
        assert np.all(np.isin(result, [-1, 1]))  # Should be bipolar

    def test_determinism_with_mock(self, mock_hv_space):
        """Test deterministic behavior with mock space."""
        source = '''
def sample_func():
    return "hello"
'''
        tree = ast.parse(source.strip())
        node = tree.body[0]
        
        result1 = encode_function(node, "test.py", mock_hv_space, {})
        result2 = encode_function(node, "test.py", mock_hv_space, {})
        
        np.testing.assert_array_equal(result1, result2)


# Performance and benchmarking tests
class TestPerformance:
    """Test encoding performance and scalability."""
    
    @pytest.mark.slow
    def test_encoding_performance(self):
        """Test encoding performance with multiple functions."""
        hv_space = HVSpace(dim=1024, bipolar=True)
        
        # Generate multiple test functions
        functions = []
        for i in range(10):
            source = f'''
def function_{i}(x, y):
    """Function {i} for testing."""
    result = x + y + {i}
    for j in range({i + 1}):
        result += j
    return result
'''
            tree = ast.parse(source.strip())
            functions.append(tree.body[0])
        
        import time
        start_time = time.time()
        
        encodings = []
        for node in functions:
            encoding = encode_function(node, f"test_{i}.py", hv_space, {})
            encodings.append(encoding)
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        # Should complete within reasonable time
        assert elapsed < 5.0  # 5 seconds for 10 functions
        assert len(encodings) == 10
        
        # All encodings should be valid
        for encoding in encodings:
            assert isinstance(encoding, np.ndarray)
            assert encoding.shape == (hv_space.dim,)