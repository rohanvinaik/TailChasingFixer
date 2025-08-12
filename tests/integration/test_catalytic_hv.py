"""
Integration tests for the catalytic hypervector system.

Tests deterministic encoding, similarity thresholds, performance benchmarks,
and memory-mapped persistence.
"""

import ast
import tempfile
import shutil
import hashlib
import time
import pytest
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

from tailchasing.catalytic.hv_encoder import (
    HypervectorEncoder, 
    EncodingConfig, 
    ASTNormalizer
)
from tailchasing.catalytic.catalytic_index import CatalyticIndex, IndexMetadata, LSHIndex
from tailchasing.catalytic.similarity_pipeline import SimilarityPipeline, QueryResult
from tailchasing.catalytic.catalytic_analyzer import CatalyticDuplicateAnalyzer


class TestHypervectorEncoder:
    """Test the hypervector encoder for deterministic behavior."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.encoder = HypervectorEncoder()
        self.test_functions = self._create_test_functions()
    
    def _create_test_functions(self) -> List[ast.FunctionDef]:
        """Create a variety of test functions."""
        functions = []
        
        # Simple function
        simple = ast.parse("""
def simple_func(x, y):
    return x + y
        """).body[0]
        functions.append(simple)
        
        # Complex function
        complex_func = ast.parse("""
def complex_func(items):
    result = []
    for item in items:
        if item > 0:
            result.append(item * 2)
        else:
            result.append(0)
    return result
        """).body[0]
        functions.append(complex_func)
        
        # Function with nested structures
        nested = ast.parse("""
def nested_func(data):
    def inner(x):
        return x ** 2
    
    processed = []
    for key, value in data.items():
        if isinstance(value, list):
            processed.append([inner(v) for v in value])
        else:
            processed.append(inner(value))
    return processed
        """).body[0]
        functions.append(nested)
        
        # Identical to simple (for duplicate testing)
        identical = ast.parse("""
def another_simple(a, b):
    return a + b
        """).body[0]
        functions.append(identical)
        
        return functions
    
    def test_deterministic_encoding(self):
        """Test that encoding is deterministic across runs."""
        func = self.test_functions[0]
        
        # Encode multiple times
        encodings = []
        for _ in range(5):
            hv = self.encoder.encode_ast(func)
            encodings.append(hv)
        
        # All encodings should be identical
        for i in range(1, len(encodings)):
            np.testing.assert_array_equal(encodings[0], encodings[i])
    
    def test_ternary_values(self):
        """Test that all encoded values are ternary (-1, 0, 1)."""
        func = self.test_functions[1]
        hv = self.encoder.encode_ast(func)
        
        # Check that all values are ternary
        unique_values = np.unique(hv)
        assert all(val in [-1, 0, 1] for val in unique_values)
    
    def test_vector_dimensions(self):
        """Test correct vector dimensions."""
        func = self.test_functions[0]
        hv = self.encoder.encode_ast(func)
        
        assert hv.shape == (8192,)
        assert hv.dtype == np.int8
    
    def test_similarity_computation(self):
        """Test similarity computation between vectors."""
        func1 = self.test_functions[0]  # simple_func
        func2 = self.test_functions[3]  # identical structure
        func3 = self.test_functions[1]  # complex_func
        
        hv1 = self.encoder.encode_ast(func1)
        hv2 = self.encoder.encode_ast(func2)
        hv3 = self.encoder.encode_ast(func3)
        
        # Similar functions should have high similarity
        sim_12 = self.encoder.similarity(hv1, hv2)
        assert sim_12 > 0.8, f"Similar functions have low similarity: {sim_12}"
        
        # Different functions should have lower similarity
        sim_13 = self.encoder.similarity(hv1, hv3)
        assert sim_13 < 0.7, f"Different functions have high similarity: {sim_13}"
        
        # Self-similarity should be 1.0
        sim_11 = self.encoder.similarity(hv1, hv1)
        assert abs(sim_11 - 1.0) < 1e-10
    
    def test_hamming_similarity(self):
        """Test Hamming similarity for ternary vectors."""
        func1 = self.test_functions[0]
        func2 = self.test_functions[3]
        
        hv1 = self.encoder.encode_ast(func1)
        hv2 = self.encoder.encode_ast(func2)
        
        hamming_sim = self.encoder.hamming_similarity(hv1, hv2)
        cosine_sim = self.encoder.similarity(hv1, hv2)
        
        # Both should indicate high similarity
        assert hamming_sim > 0.6
        assert cosine_sim > 0.6
        
        # Hamming and cosine should be correlated but not identical
        assert abs(hamming_sim - cosine_sim) < 0.5
    
    def test_ast_normalization(self):
        """Test that AST normalization works correctly."""
        normalizer = ASTNormalizer()
        
        # Test function with specific variable names
        func_with_names = ast.parse("""
def test_func(user_data, session_key):
    result = user_data.get('name', 'default')
    if result == 'admin':
        return session_key
    return None
        """).body[0]
        
        # Normalize
        normalized = normalizer.visit(func_with_names)
        
        # Should have canonical names
        assert normalized.name == 'FUNC'
        
        # Check that variables are normalized
        names_found = []
        for node in ast.walk(normalized):
            if isinstance(node, ast.Name) and node.id not in normalizer.BUILTINS:
                names_found.append(node.id)
        
        # Most names should be 'ID'
        assert 'ID' in names_found
    
    def test_context_encoding(self):
        """Test that context affects encoding."""
        func = self.test_functions[0]
        
        # Encode without context
        hv_no_context = self.encoder.encode_function(func)
        
        # Encode with context
        context = {
            'imports': ['numpy', 'pandas'],
            'class_name': 'Calculator'
        }
        hv_with_context = self.encoder.encode_function(func, context)
        
        # Should be different
        similarity = self.encoder.similarity(hv_no_context, hv_with_context)
        assert similarity < 1.0
    
    def test_performance_encoding_speed(self):
        """Benchmark encoding speed."""
        func = self.test_functions[2]  # Complex nested function
        
        start_time = time.time()
        num_encodings = 100
        
        for _ in range(num_encodings):
            hv = self.encoder.encode_ast(func)
        
        end_time = time.time()
        total_time = end_time - start_time
        avg_time = total_time / num_encodings
        
        print(f"Average encoding time: {avg_time*1000:.2f}ms")
        
        # Should encode reasonably fast
        assert avg_time < 0.1, f"Encoding too slow: {avg_time}s per function"


class TestCatalyticIndex:
    """Test the memory-mapped catalytic index."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.encoder = HypervectorEncoder()
        self.test_vectors = self._create_test_vectors()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def _create_test_vectors(self) -> List[np.ndarray]:
        """Create test hypervectors."""
        vectors = []
        
        # Create some random ternary vectors
        rng = np.random.RandomState(42)
        for _ in range(10):
            vec = np.zeros(8192, dtype=np.int8)
            nonzero_idx = rng.choice(8192, 2000, replace=False)
            vec[nonzero_idx] = rng.choice([-1, 1], 2000)
            vectors.append(vec)
        
        return vectors
    
    def test_index_creation(self):
        """Test basic index creation and opening."""
        # Create new index
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            assert index.num_functions == 0
            assert index.total_bytes == 0
        
        # Verify files were created
        assert (Path(self.temp_dir) / 'metadata.jsonl').exists()
        assert (Path(self.temp_dir) / 'vectors.dat').exists()
    
    def test_add_and_retrieve_vectors(self):
        """Test adding and retrieving vectors."""
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            # Add test vectors
            for i, vec in enumerate(self.test_vectors[:3]):
                metadata = index.add_function(
                    function_id=f"test_func_{i}",
                    hypervector=vec,
                    file_path=f"test_{i}.py",
                    function_name=f"func_{i}",
                    line_number=i * 10,
                    ast_hash=f"hash_{i}"
                )
                
                assert metadata.function_id == f"test_func_{i}"
                assert metadata.file_path == f"test_{i}.py"
                assert index.num_functions == i + 1
        
        # Reopen and retrieve
        with CatalyticIndex(self.temp_dir, mode='r') as index:
            assert index.num_functions == 3
            
            # Retrieve vectors
            for i in range(3):
                retrieved = index.get_vector(f"test_func_{i}")
                assert retrieved is not None
                np.testing.assert_array_equal(retrieved, self.test_vectors[i])
    
    def test_append_mode(self):
        """Test appending to existing index."""
        # Create initial index
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            index.add_function(
                function_id="initial",
                hypervector=self.test_vectors[0],
                file_path="initial.py",
                function_name="initial_func",
                line_number=1,
                ast_hash="initial_hash"
            )
        
        # Append to existing index
        with CatalyticIndex(self.temp_dir, mode='a') as index:
            assert index.num_functions == 1
            
            index.add_function(
                function_id="appended",
                hypervector=self.test_vectors[1],
                file_path="appended.py",
                function_name="appended_func",
                line_number=2,
                ast_hash="appended_hash"
            )
        
        # Verify both are present
        with CatalyticIndex(self.temp_dir, mode='r') as index:
            assert index.num_functions == 2
            
            initial = index.get_vector("initial")
            appended = index.get_vector("appended")
            
            np.testing.assert_array_equal(initial, self.test_vectors[0])
            np.testing.assert_array_equal(appended, self.test_vectors[1])
    
    def test_lsh_querying(self):
        """Test LSH-based similarity querying."""
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            # Add vectors
            for i, vec in enumerate(self.test_vectors):
                index.add_function(
                    function_id=f"func_{i}",
                    hypervector=vec,
                    file_path=f"file_{i}.py",
                    function_name=f"function_{i}",
                    line_number=i,
                    ast_hash=f"hash_{i}"
                )
            
            # Query with one of the vectors
            query_vec = self.test_vectors[0]
            candidates = index.query_similar(query_vec, max_candidates=5)
            
            # Should find itself and possibly others
            assert len(candidates) >= 1
            assert "func_0" in candidates
    
    def test_metadata_persistence(self):
        """Test metadata persistence and retrieval."""
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            metadata = index.add_function(
                function_id="test_persistence",
                hypervector=self.test_vectors[0],
                file_path="/path/to/test.py",
                function_name="test_function",
                line_number=42,
                ast_hash="abcd1234"
            )
        
        # Reopen and check metadata
        with CatalyticIndex(self.temp_dir, mode='r') as index:
            retrieved_metadata = index.get_metadata("test_persistence")
            
            assert retrieved_metadata is not None
            assert retrieved_metadata.function_id == "test_persistence"
            assert retrieved_metadata.file_path == "/path/to/test.py"
            assert retrieved_metadata.function_name == "test_function"
            assert retrieved_metadata.line_number == 42
            assert retrieved_metadata.ast_hash == "abcd1234"
    
    def test_vector_iteration(self):
        """Test iterating over all vectors."""
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            # Add several vectors
            for i in range(5):
                index.add_function(
                    function_id=f"iter_test_{i}",
                    hypervector=self.test_vectors[i],
                    file_path=f"iter_{i}.py",
                    function_name=f"iter_func_{i}",
                    line_number=i,
                    ast_hash=f"iter_hash_{i}"
                )
        
        # Iterate and verify
        with CatalyticIndex(self.temp_dir, mode='r') as index:
            count = 0
            for func_id, vector in index.iterate_vectors(batch_size=2):
                assert func_id.startswith("iter_test_")
                assert vector.shape == (8192,)
                count += 1
            
            assert count == 5
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            # Add some vectors
            for i in range(10):
                index.add_function(
                    function_id=f"memory_test_{i}",
                    hypervector=self.test_vectors[i % len(self.test_vectors)],
                    file_path=f"memory_{i}.py",
                    function_name=f"memory_func_{i}",
                    line_number=i,
                    ast_hash=f"memory_hash_{i}"
                )
            
            stats = index.get_stats()
            assert stats['num_functions'] == 10
            assert stats['working_memory_mb'] > 0
            assert stats['working_memory_mb'] < 100  # Should be efficient
    
    def test_performance_large_scale(self):
        """Test performance with larger number of vectors."""
        num_vectors = 1000
        
        with CatalyticIndex(self.temp_dir, mode='w') as index:
            start_time = time.time()
            
            # Add many vectors
            for i in range(num_vectors):
                vec = self.test_vectors[i % len(self.test_vectors)]
                index.add_function(
                    function_id=f"perf_test_{i}",
                    hypervector=vec,
                    file_path=f"perf_{i}.py",
                    function_name=f"perf_func_{i}",
                    line_number=i,
                    ast_hash=f"perf_hash_{i}"
                )
                
                # Log progress
                if (i + 1) % 100 == 0:
                    elapsed = time.time() - start_time
                    rate = (i + 1) / elapsed
                    print(f"Added {i + 1} vectors, rate: {rate:.1f} vectors/sec")
            
            add_time = time.time() - start_time
            
            # Test querying performance
            query_start = time.time()
            query_vec = self.test_vectors[0]
            
            for _ in range(10):
                candidates = index.query_similar(query_vec, max_candidates=50)
            
            query_time = (time.time() - query_start) / 10
            
            stats = index.get_stats()
            
        print(f"Performance results for {num_vectors} vectors:")
        print(f"  Add time: {add_time:.2f}s ({num_vectors/add_time:.1f} vectors/sec)")
        print(f"  Query time: {query_time*1000:.2f}ms per query")
        print(f"  Working memory: {stats['working_memory_mb']:.1f}MB")
        
        # Performance assertions
        assert add_time < 60, f"Adding {num_vectors} vectors too slow: {add_time}s"
        assert query_time < 0.1, f"Querying too slow: {query_time}s"
        assert stats['working_memory_mb'] < 100, f"Memory usage too high: {stats['working_memory_mb']}MB"


class TestLSHIndex:
    """Test the LSH index component."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.lsh = LSHIndex(dim=8192, n_tables=8, n_hyperplanes=16)
        
        # Create test vectors
        rng = np.random.RandomState(42)
        self.test_vectors = []
        for i in range(20):
            vec = np.zeros(8192, dtype=np.int8)
            nonzero_idx = rng.choice(8192, 2000, replace=False)
            vec[nonzero_idx] = rng.choice([-1, 1], 2000)
            self.test_vectors.append(vec)
    
    def test_hash_consistency(self):
        """Test that hashing is consistent."""
        vec = self.test_vectors[0]
        
        # Hash multiple times
        hashes = []
        for _ in range(5):
            hash_vals = self.lsh.hash_vector(vec)
            hashes.append(hash_vals)
        
        # All should be identical
        for i in range(1, len(hashes)):
            assert hashes[0] == hashes[i]
    
    def test_similar_vectors_same_buckets(self):
        """Test that similar vectors hash to similar buckets."""
        vec1 = self.test_vectors[0]
        
        # Create similar vector (90% overlap)
        vec2 = vec1.copy()
        flip_indices = np.random.choice(np.where(vec2 != 0)[0], size=200, replace=False)
        vec2[flip_indices] *= -1
        
        hash1 = self.lsh.hash_vector(vec1)
        hash2 = self.lsh.hash_vector(vec2)
        
        # Should have some matching hash values
        matches = sum(1 for h1, h2 in zip(hash1, hash2) if h1 == h2)
        assert matches > 0, "Similar vectors should hash to some common buckets"
    
    def test_add_and_query(self):
        """Test adding vectors and querying."""
        # Add vectors
        for i, vec in enumerate(self.test_vectors[:10]):
            self.lsh.add(vec, f"vector_{i}")
        
        # Query
        query_vec = self.test_vectors[0]
        candidates = self.lsh.query(query_vec, max_candidates=5)
        
        # Should find itself
        assert "vector_0" in candidates
        assert len(candidates) <= 5


class TestSimilarityPipeline:
    """Test the complete similarity pipeline."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.index = CatalyticIndex(self.temp_dir, mode='w')
        self.pipeline = SimilarityPipeline(self.index)
        
        # Create test functions
        self.test_functions = self._create_test_functions()
        
        # Populate index
        self._populate_index()
    
    def teardown_method(self):
        """Clean up test fixtures."""
        self.index.close()
        shutil.rmtree(self.temp_dir)
    
    def _create_test_functions(self) -> List[ast.FunctionDef]:
        """Create test functions with known similarities."""
        functions = []
        
        # Function 1: Simple addition
        func1 = ast.parse("""
def add_numbers(a, b):
    return a + b
        """).body[0]
        functions.append(func1)
        
        # Function 2: Identical to func1 (different names)
        func2 = ast.parse("""
def sum_values(x, y):
    return x + y
        """).body[0]
        functions.append(func2)
        
        # Function 3: Similar but with multiplication
        func3 = ast.parse("""
def multiply_numbers(a, b):
    return a * b
        """).body[0]
        functions.append(func3)
        
        # Function 4: Complex logic
        func4 = ast.parse("""
def process_data(items):
    result = []
    for item in items:
        if item % 2 == 0:
            result.append(item * 2)
        else:
            result.append(item + 1)
    return result
        """).body[0]
        functions.append(func4)
        
        # Function 5: Different implementation, same logic as func4
        func5 = ast.parse("""
def transform_list(data):
    output = []
    for element in data:
        if element % 2 == 0:
            output.append(element * 2)
        else:
            output.append(element + 1)
    return output
        """).body[0]
        functions.append(func5)
        
        return functions
    
    def _populate_index(self):
        """Populate index with test functions."""
        for i, func in enumerate(self.test_functions):
            self.pipeline.update_index(
                func_ast=func,
                file_path=f"test_{i}.py",
                function_name=func.name,
                line_number=i * 10 + 1
            )
    
    def test_query_exact_matches(self):
        """Test querying for exact matches."""
        query_func = self.test_functions[0]  # add_numbers
        
        results = self.pipeline.query_function(query_func, top_k=3)
        
        # Should find itself and the identical function
        assert len(results) >= 1
        
        # Check that high similarity results are found
        high_sim_results = [r for r in results if r.similarity_score > 0.8]
        assert len(high_sim_results) >= 1
    
    def test_confidence_scoring(self):
        """Test confidence scoring mechanism."""
        query_func = self.test_functions[0]
        results = self.pipeline.query_function(query_func, top_k=5)
        
        # All results should have valid confidence scores
        for result in results:
            assert 0.0 <= result.confidence <= 1.0
            assert 0.0 <= result.hv_similarity <= 1.0
            assert 0.0 <= result.ast_similarity <= 1.0
    
    def test_batch_query(self):
        """Test batch querying functionality."""
        query_functions = [
            (self.test_functions[0], {}),
            (self.test_functions[3], {})
        ]
        
        batch_results = self.pipeline.batch_query(query_functions, top_k=3)
        
        assert len(batch_results) == 2
        assert 0 in batch_results
        assert 1 in batch_results
        
        # Each query should return results
        for results in batch_results.values():
            assert len(results) >= 1
    
    def test_find_duplicates(self):
        """Test duplicate detection across all functions."""
        duplicates = self.pipeline.find_duplicates(min_similarity=0.85)
        
        # Should find some duplicates (func1 and func2 are identical)
        assert len(duplicates) >= 1
        
        # Verify duplicate pairs have high similarity
        for func1_id, func2_id, similarity in duplicates:
            assert similarity >= 0.85
            assert func1_id != func2_id
    
    def test_similarity_thresholds(self):
        """Test different similarity thresholds."""
        query_func = self.test_functions[0]
        
        # Test with high threshold
        pipeline_high = SimilarityPipeline(
            self.index,
            hv_threshold=0.95,
            ast_threshold=0.95
        )
        results_high = pipeline_high.query_function(query_func, top_k=5)
        
        # Test with low threshold
        pipeline_low = SimilarityPipeline(
            self.index,
            hv_threshold=0.7,
            ast_threshold=0.7
        )
        results_low = pipeline_low.query_function(query_func, top_k=5)
        
        # Low threshold should return more results
        assert len(results_low) >= len(results_high)


class TestCatalyticAnalyzer:
    """Test the complete catalytic analyzer integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = CatalyticDuplicateAnalyzer()
    
    def test_analyzer_creation(self):
        """Test analyzer creation and configuration."""
        assert self.analyzer.name == "catalytic_duplicates"
        assert hasattr(self.analyzer, 'encoder')
        assert hasattr(self.analyzer, 'hv_threshold')
        assert hasattr(self.analyzer, 'ast_threshold')
    
    def test_threshold_validation(self):
        """Test threshold validation."""
        # Valid thresholds
        assert 0.0 <= self.analyzer.hv_threshold <= 1.0
        assert 0.0 <= self.analyzer.ast_threshold <= 1.0
        assert 0.0 <= self.analyzer.min_similarity_for_issue <= 1.0
        
        # Reasonable defaults
        assert self.analyzer.hv_threshold > 0.8
        assert self.analyzer.ast_threshold > 0.8


def test_end_to_end_pipeline():
    """Test complete end-to-end pipeline performance."""
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Create test functions
        test_code = """
def duplicate_func_1(x, y):
    result = x + y
    return result * 2

def duplicate_func_2(a, b):
    temp = a + b
    return temp * 2

def different_func(data):
    processed = []
    for item in data:
        if item > 0:
            processed.append(item)
    return processed

def another_duplicate(m, n):
    sum_val = m + n
    return sum_val * 2
        """
        
        # Parse functions
        tree = ast.parse(test_code)
        functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        
        # Set up pipeline
        with CatalyticIndex(temp_dir, mode='w') as index:
            pipeline = SimilarityPipeline(index, hv_threshold=0.85, ast_threshold=0.8)
            
            # Add functions to index
            for i, func in enumerate(functions):
                pipeline.update_index(
                    func_ast=func,
                    file_path="test_file.py",
                    function_name=func.name,
                    line_number=i * 5 + 1
                )
            
            # Test duplicate detection
            duplicates = pipeline.find_duplicates(min_similarity=0.9)
            
            # Should find duplicate functions
            assert len(duplicates) >= 2  # At least 2 duplicate pairs
            
            # Test querying
            query_results = pipeline.query_function(functions[0], top_k=3)
            assert len(query_results) >= 1
            
            # Performance test
            start_time = time.time()
            for func in functions:
                pipeline.query_function(func, top_k=5)
            end_time = time.time()
            
            query_time_per_func = (end_time - start_time) / len(functions)
            print(f"Average query time per function: {query_time_per_func*1000:.2f}ms")
            
            # Should be reasonably fast
            assert query_time_per_func < 0.5, f"Query too slow: {query_time_per_func}s per function"
    
    finally:
        shutil.rmtree(temp_dir)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])