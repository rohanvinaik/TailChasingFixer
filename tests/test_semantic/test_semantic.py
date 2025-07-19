"""Test cases for semantic hypervector analyzer."""

import pytest
from pathlib import Path

from tailchasing.semantic.hv_space import HVSpace
from tailchasing.semantic.encoder import encode_function, split_identifier
from tailchasing.semantic.index import SemanticIndex


class TestHVSpace:
    """Test hypervector space operations."""
    
    def test_init(self):
        """Test space initialization."""
        space = HVSpace(dim=1024, bipolar=True, seed=42)
        assert space.dim == 1024
        assert space.bipolar == True
        
    def test_token_consistency(self):
        """Test that tokens are encoded consistently."""
        space = HVSpace(dim=1024, seed=42)
        
        # Same token should give same vector
        token1 = space.token("hello")
        token2 = space.token("hello")
        assert space.similarity(token1, token2) == 1.0
        
        # Different tokens should be dissimilar
        token3 = space.token("world")
        assert space.similarity(token1, token3) < 0.6
        
    def test_bind_operation(self):
        """Test binding operation properties."""
        space = HVSpace(dim=1024, bipolar=True)
        
        a = space.token("role")
        b = space.token("filler")
        
        # Bind should be dissimilar to inputs
        bound = space.bind(a, b)
        assert space.similarity(bound, a) < 0.6
        assert space.similarity(bound, b) < 0.6
        
        # Binding is its own inverse
        unbound = space.bind(bound, a)
        assert space.similarity(unbound, b) > 0.9
        
    def test_bundle_operation(self):
        """Test bundling operation properties."""
        space = HVSpace(dim=1024)
        
        tokens = [space.token(f"token{i}") for i in range(5)]
        bundled = space.bundle(tokens)
        
        # Bundle should be similar to all inputs
        for tok in tokens:
            assert space.similarity(bundled, tok) > 0.3
            
    def test_capacity_estimate(self):
        """Test capacity estimation."""
        space = HVSpace(dim=8192)
        capacity = space.capacity_estimate()
        assert capacity > 800  # ~0.1 * dim
        assert capacity < 1000


class TestEncoder:
    """Test AST to hypervector encoding."""
    
    def test_split_identifier(self):
        """Test identifier splitting."""
        assert split_identifier("camelCase") == ["camel", "case"]
        assert split_identifier("snake_case") == ["snake", "case"]
        assert split_identifier("CONSTANT_CASE") == ["constant", "case"]
        assert split_identifier("HTTPSConnection") == ["https", "connection"]
        
    def test_encode_simple_function(self):
        """Test encoding a simple function."""
        import ast
        
        code = '''
def add(a, b):
    """Add two numbers."""
    return a + b
'''
        tree = ast.parse(code)
        func = tree.body[0]
        
        space = HVSpace(dim=1024)
        config = {"channel_weights": {}}
        
        hv = encode_function(func, "test.py", space, config)
        assert hv is not None
        assert len(hv) == 1024
        
    def test_encode_complex_function(self):
        """Test encoding a complex function."""
        import ast
        
        code = '''
def process_data(data, config=None):
    """Process data according to configuration.
    
    This function handles various data types and applies
    transformations based on the provided configuration.
    """
    if not data:
        raise ValueError("Data cannot be empty")
    
    result = []
    for item in data:
        try:
            if isinstance(item, str):
                processed = item.strip().lower()
            elif isinstance(item, (int, float)):
                processed = item * 2
            else:
                processed = str(item)
            
            result.append(processed)
        except Exception as e:
            print(f"Error processing {item}: {e}")
            continue
    
    return result
'''
        tree = ast.parse(code)
        func = tree.body[0]
        
        space = HVSpace(dim=1024)
        config = {"channel_weights": {}}
        
        hv = encode_function(func, "test.py", space, config)
        assert hv is not None
        
        # Encode a similar function
        code2 = '''
def transform_data(items, settings=None):
    """Transform items based on settings."""
    if not items:
        raise ValueError("Items required")
    
    output = []
    for i in items:
        try:
            if isinstance(i, str):
                transformed = i.strip().lower()
            elif isinstance(i, (int, float)):
                transformed = i * 2
            else:
                transformed = str(i)
            output.append(transformed)
        except Exception:
            continue
    return output
'''
        tree2 = ast.parse(code2)
        func2 = tree2.body[0]
        hv2 = encode_function(func2, "test.py", space, config)
        
        # Should be semantically similar
        similarity = space.similarity(hv, hv2)
        assert similarity > 0.7  # High similarity expected


class TestSemanticIndex:
    """Test semantic index functionality."""
    
    def test_index_operations(self):
        """Test basic index operations."""
        config = {
            "hv_dim": 1024,
            "bipolar": True,
            "z_threshold": 2.5,
            "max_pairs_sample": 1000
        }
        
        index = SemanticIndex(config)
        
        # Add some functions
        for i in range(10):
            hv = index.space.token(f"function_{i}")
            index.add(f"func{i}", f"file{i}.py", i*10, hv)
        
        # Check stats
        stats = index.get_stats()
        assert stats["total_functions"] == 10
        
    def test_similarity_detection(self):
        """Test finding similar functions."""
        config = {
            "hv_dim": 1024,
            "bipolar": True,
            "z_threshold": 2.0,
            "max_pairs_sample": 100
        }
        
        index = SemanticIndex(config)
        
        # Add dissimilar functions
        for i in range(5):
            hv = index.space.token(f"unique_function_{i}")
            index.add(f"func{i}", f"file.py", i*10, hv)
        
        # Add similar functions
        base_hv = index.space.token("base_pattern")
        for i in range(5, 8):
            # Add slight noise
            noise = index.space.token(f"noise_{i}")
            hv = index.space.bundle([base_hv, base_hv, base_hv, noise])
            index.add(f"similar{i}", f"file.py", i*10, hv)
        
        # Find similar pairs
        pairs = index.find_all_similar_pairs()
        
        # Should find the similar functions
        assert len(pairs) > 0
        
        # Check that similar functions are paired
        similar_names = set()
        for id1, id2, dist, z, _ in pairs:
            name1 = id1.split('@')[0]
            name2 = id2.split('@')[0]
            if name1.startswith("similar"):
                similar_names.add(name1)
            if name2.startswith("similar"):
                similar_names.add(name2)
        
        assert len(similar_names) >= 2  # At least 2 similar functions found