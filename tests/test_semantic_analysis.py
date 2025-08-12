"""
Tests for semantic analysis and hypervector operations.

Comprehensive tests for hypervector computing, semantic encoding,
and similarity analysis with statistical significance testing.
"""

import unittest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import textwrap
import time

from hypothesis import given, strategies as st, settings, assume
import pytest

from tailchasing.semantic.hv_space import HVSpace
from tailchasing.semantic.encoder import SemanticEncoder
from tailchasing.semantic.index import SemanticIndex, FunctionEntry, SimilarityPair
from tailchasing.semantic.similarity import (
    SimilarityAnalyzer,
    ChannelContribution,
    PrototypeCluster,
    TemporalDrift,
    benjamini_hochberg_fdr,
    z_to_p_value
)


class TestHVSpace(unittest.TestCase):
    """Test hypervector space operations."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'dimensions': 8192,
            'sparsity': 0.01,
            'seed': 42
        }
        self.space = HVSpace(self.config)
    
    def test_vector_generation(self):
        """Test random vector generation."""
        # Test binary vectors
        binary_hv = self.space.random_hv(binary=True)
        self.assertEqual(len(binary_hv), self.config['dimensions'])
        self.assertTrue(np.all(np.isin(binary_hv, [0, 1])))
        
        # Test bipolar vectors
        bipolar_hv = self.space.random_hv(binary=False)
        self.assertEqual(len(bipolar_hv), self.config['dimensions'])
        self.assertTrue(np.all(np.isin(bipolar_hv, [-1, 1])))
        
        # Test sparsity
        sparse_hv = self.space.random_hv(sparse=True)
        sparsity_ratio = np.sum(sparse_hv != 0) / len(sparse_hv)
        self.assertAlmostEqual(sparsity_ratio, self.config['sparsity'], places=2)
    
    def test_token_caching(self):
        """Test token vector caching."""
        token1 = "test_token"
        
        # First call should generate
        hv1 = self.space.get_token_hv(token1)
        
        # Second call should return cached
        hv2 = self.space.get_token_hv(token1)
        
        # Should be identical
        np.testing.assert_array_equal(hv1, hv2)
        
        # Different token should be different
        hv3 = self.space.get_token_hv("different_token")
        self.assertFalse(np.array_equal(hv1, hv3))
    
    def test_bundle_operation(self):
        """Test bundling (superposition) operation."""
        hv1 = self.space.random_hv()
        hv2 = self.space.random_hv()
        hv3 = self.space.random_hv()
        
        bundled = self.space.bundle([hv1, hv2, hv3])
        
        # Bundled vector should be similar to all components
        sim1 = self.space.similarity(bundled, hv1)
        sim2 = self.space.similarity(bundled, hv2)
        sim3 = self.space.similarity(bundled, hv3)
        
        # Should have some similarity to each component
        self.assertGreater(sim1, 0.2)
        self.assertGreater(sim2, 0.2)
        self.assertGreater(sim3, 0.2)
    
    def test_bind_operation(self):
        """Test binding operation."""
        role_hv = self.space.get_role_hv("subject")
        filler_hv = self.space.get_token_hv("user")
        
        bound = self.space.bind(role_hv, filler_hv)
        
        # Bound vector should be dissimilar to components
        sim_role = self.space.similarity(bound, role_hv)
        sim_filler = self.space.similarity(bound, filler_hv)
        
        # Should have low similarity to components
        self.assertLess(abs(sim_role), 0.2)
        self.assertLess(abs(sim_filler), 0.2)
        
        # Unbinding should recover original
        unbound = self.space.bind(bound, role_hv)  # Bind is self-inverse
        sim_recovered = self.space.similarity(unbound, filler_hv)
        self.assertGreater(sim_recovered, 0.8)
    
    def test_similarity_metrics(self):
        """Test similarity and distance calculations."""
        hv1 = self.space.random_hv()
        hv2 = self.space.random_hv()
        
        # Test similarity
        sim = self.space.similarity(hv1, hv2)
        self.assertGreaterEqual(sim, -1.0)
        self.assertLessEqual(sim, 1.0)
        
        # Test self-similarity
        self_sim = self.space.similarity(hv1, hv1)
        self.assertAlmostEqual(self_sim, 1.0)
        
        # Test distance
        dist = self.space.distance(hv1, hv2)
        self.assertGreaterEqual(dist, 0.0)
        self.assertLessEqual(dist, 1.0)
        
        # Test self-distance
        self_dist = self.space.distance(hv1, hv1)
        self.assertAlmostEqual(self_dist, 0.0)
        
        # Test relationship between similarity and distance
        expected_dist = (1 - sim) / 2
        self.assertAlmostEqual(dist, expected_dist, places=5)
    
    @given(
        num_vectors=st.integers(min_value=2, max_value=10),
        weights=st.lists(st.floats(min_value=0.1, max_value=10.0), min_size=2, max_size=10)
    )
    @settings(max_examples=20)
    def test_weighted_bundle_property(self, num_vectors: int, weights: List[float]):
        """Property-based test for weighted bundling."""
        # Generate vectors
        vectors = [self.space.random_hv() for _ in range(num_vectors)]
        
        # Adjust weights to match number of vectors
        weights = weights[:num_vectors]
        while len(weights) < num_vectors:
            weights.append(1.0)
        
        # Create weighted bundle
        bundled = self.space.weighted_bundle(vectors, weights)
        
        # Verify properties
        self.assertEqual(len(bundled), self.config['dimensions'])
        
        # Higher weight should mean higher similarity
        similarities = [self.space.similarity(bundled, v) for v in vectors]
        
        # Normalize weights
        total_weight = sum(weights)
        norm_weights = [w / total_weight for w in weights]
        
        # Check that similarity roughly correlates with weight
        # (This is approximate due to interference)
        max_weight_idx = np.argmax(weights)
        max_sim_idx = np.argmax(similarities)
        
        # The highest weighted vector should often have high similarity
        # (not always due to interference, but statistically true)
        if weights[max_weight_idx] > 2 * np.mean(weights):
            self.assertGreater(similarities[max_weight_idx], np.mean(similarities))


class TestSemanticEncoder(unittest.TestCase):
    """Test semantic encoding of code."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'dimensions': 8192,
            'channels': [
                'structure', 'data_flow', 'control_flow',
                'identifiers', 'literals', 'operations'
            ]
        }
        self.encoder = SemanticEncoder(self.config)
    
    def test_encode_simple_function(self):
        """Test encoding of a simple function."""
        code = """
def add(a, b):
    return a + b
        """
        
        hv, features = self.encoder.encode_function(code)
        
        # Check hypervector properties
        self.assertEqual(len(hv), self.config['dimensions'])
        self.assertFalse(np.all(hv == 0))  # Should not be zero vector
        
        # Check features extracted
        self.assertIn('structure', features)
        self.assertIn('identifiers', features)
        self.assertIn('operations', features)
        
        # Should have identified the addition operation
        self.assertIn('Add', features.get('operations', []))
    
    def test_encode_complex_function(self):
        """Test encoding of a complex function."""
        code = """
def process_data(items):
    result = []
    for item in items:
        if item > 0:
            processed = item * 2
            result.append(processed)
        else:
            result.append(0)
    return result
        """
        
        hv, features = self.encoder.encode_function(code)
        
        # Check control flow features
        self.assertIn('For', features.get('control_flow', []))
        self.assertIn('If', features.get('control_flow', []))
        
        # Check data flow
        self.assertIn('result', features.get('data_flow', []))
        
        # Check operations
        ops = features.get('operations', [])
        self.assertIn('Mult', ops)  # Multiplication
        self.assertIn('Gt', ops)     # Greater than
    
    def test_encode_similarity(self):
        """Test that similar functions have similar encodings."""
        code1 = """
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        total += num
    return total
        """
        
        code2 = """
def compute_total(values):
    sum_val = 0
    for v in values:
        sum_val += v
    return sum_val
        """
        
        code3 = """
def find_maximum(numbers):
    max_val = numbers[0]
    for num in numbers:
        if num > max_val:
            max_val = num
    return max_val
        """
        
        hv1, _ = self.encoder.encode_function(code1)
        hv2, _ = self.encoder.encode_function(code2)
        hv3, _ = self.encoder.encode_function(code3)
        
        # Similar functions should have high similarity
        sim_12 = self.encoder.space.similarity(hv1, hv2)
        self.assertGreater(sim_12, 0.7)
        
        # Different functions should have lower similarity
        sim_13 = self.encoder.space.similarity(hv1, hv3)
        sim_23 = self.encoder.space.similarity(hv2, hv3)
        
        self.assertLess(sim_13, sim_12)
        self.assertLess(sim_23, sim_12)
    
    def test_channel_contributions(self):
        """Test channel contribution analysis."""
        code = """
def validate_user(user):
    if not user:
        return False
    if not user.get('email'):
        return False
    if user.get('age', 0) < 18:
        return False
    return True
        """
        
        hv, features = self.encoder.encode_function(code)
        
        # Analyze channel contributions
        analyzer = SimilarityAnalyzer({'fdr_q': 0.05, 'z_threshold': 2.5})
        
        # Compare with slightly modified version
        code2 = """
def check_user(u):
    if not u:
        return False
    if not u.get('email'):
        return False
    if u.get('age', 0) < 18:
        return False
    return True
        """
        
        hv2, features2 = self.encoder.encode_function(code2)
        
        contributions = analyzer.analyze_channel_contributions(
            hv, hv2, self.encoder.space, features, features2
        )
        
        # Control flow should have high contribution (identical structure)
        self.assertGreater(contributions.get('control_flow', 0), 0.2)
        
        # Identifiers should have lower contribution (different names)
        self.assertLess(contributions.get('identifiers', 1), 
                       contributions.get('control_flow', 0))


class TestSemanticIndex(unittest.TestCase):
    """Test semantic indexing and retrieval."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'dimensions': 8192,
            'z_threshold': 2.5,
            'fdr_alpha': 0.05
        }
        self.index = SemanticIndex(self.config)
        self.encoder = SemanticEncoder(self.config)
    
    def test_add_and_retrieve(self):
        """Test adding and retrieving functions."""
        code = """
def test_function():
    return 42
        """
        
        hv, features = self.encoder.encode_function(code)
        
        entry = self.index.add_function(
            function_id="test_1",
            file_path="test.py",
            name="test_function",
            line_number=1,
            hypervector=hv,
            features=features
        )
        
        self.assertIsNotNone(entry)
        self.assertEqual(entry.function_id, "test_1")
        
        # Retrieve
        retrieved = self.index.get_function("test_1")
        self.assertIsNotNone(retrieved)
        self.assertEqual(retrieved.name, "test_function")
    
    def test_find_similar_pairs(self):
        """Test finding similar function pairs."""
        # Add several functions
        functions = [
            ("def add(a, b): return a + b", "add"),
            ("def sum(x, y): return x + y", "sum"),
            ("def multiply(a, b): return a * b", "multiply"),
            ("def add_numbers(n1, n2): return n1 + n2", "add_numbers")
        ]
        
        for i, (code, name) in enumerate(functions):
            hv, features = self.encoder.encode_function(code)
            self.index.add_function(
                function_id=f"func_{i}",
                file_path="test.py",
                name=name,
                line_number=i * 10,
                hypervector=hv,
                features=features
            )
        
        # Compute background statistics
        self.index.compute_background_stats()
        
        # Find similar pairs
        pairs = self.index.find_similar_pairs(top_k=10)
        
        # Should find that add functions are similar
        self.assertGreater(len(pairs), 0)
        
        # Check that similar functions are paired
        pair_names = set()
        for pair in pairs:
            func1 = self.index.get_function(pair.id1)
            func2 = self.index.get_function(pair.id2)
            pair_names.add((func1.name, func2.name))
        
        # Should find add, sum, and add_numbers are similar
        similar_found = any(
            ('add' in p[0] and 'add' in p[1]) or
            ('add' in p[0] and 'sum' in p[1]) or
            ('sum' in p[0] and 'add' in p[1])
            for p in pair_names
        )
        self.assertTrue(similar_found)
    
    def test_statistical_significance(self):
        """Test statistical significance of similarities."""
        # Add many random functions
        np.random.seed(42)
        
        for i in range(50):
            # Generate random hypervector
            hv = np.random.choice([-1, 1], size=self.config['dimensions'])
            
            self.index.add_function(
                function_id=f"random_{i}",
                file_path="random.py",
                name=f"func_{i}",
                line_number=i,
                hypervector=hv,
                features={}
            )
        
        # Add two very similar functions
        base_hv = np.random.choice([-1, 1], size=self.config['dimensions'])
        
        # Create near-duplicate (95% same)
        duplicate_hv = base_hv.copy()
        flip_indices = np.random.choice(
            self.config['dimensions'],
            size=int(0.05 * self.config['dimensions']),
            replace=False
        )
        duplicate_hv[flip_indices] *= -1
        
        self.index.add_function(
            function_id="similar_1",
            file_path="test.py",
            name="similar_func_1",
            line_number=100,
            hypervector=base_hv,
            features={}
        )
        
        self.index.add_function(
            function_id="similar_2",
            file_path="test.py",
            name="similar_func_2",
            line_number=200,
            hypervector=duplicate_hv,
            features={}
        )
        
        # Compute statistics
        self.index.compute_background_stats()
        
        # Find significant pairs
        pairs = self.index.find_similar_pairs(top_k=100)
        
        # The planted similar pair should be detected with high significance
        similar_pair_found = False
        for pair in pairs:
            if (pair.id1 == "similar_1" and pair.id2 == "similar_2") or \
               (pair.id1 == "similar_2" and pair.id2 == "similar_1"):
                similar_pair_found = True
                # Should have high z-score
                self.assertGreater(pair.z_score, 3.0)
                # Should be FDR significant
                self.assertTrue(pair.metadata.get('fdr_significant', False))
                break
        
        self.assertTrue(similar_pair_found)


class TestSimilarityAnalysis(unittest.TestCase):
    """Test advanced similarity analysis."""
    
    def setUp(self):
        """Set up test environment."""
        self.config = {
            'dimensions': 8192,
            'fdr_q': 0.05,
            'z_threshold': 2.5
        }
        self.analyzer = SimilarityAnalyzer(self.config)
        self.space = HVSpace(self.config)
    
    def test_fdr_correction(self):
        """Test Benjamini-Hochberg FDR correction."""
        # Test with known p-values
        p_values = [0.001, 0.005, 0.01, 0.03, 0.04, 0.2, 0.5, 0.9]
        
        significant = benjamini_hochberg_fdr(p_values, alpha=0.05)
        
        # First few should be significant
        self.assertTrue(significant[0])  # p=0.001
        self.assertTrue(significant[1])  # p=0.005
        
        # Last few should not be significant
        self.assertFalse(significant[-1])  # p=0.9
        self.assertFalse(significant[-2])  # p=0.5
    
    def test_z_score_to_p_value(self):
        """Test z-score to p-value conversion."""
        # Test known values
        p_val_0 = z_to_p_value(0.0)
        self.assertAlmostEqual(p_val_0, 1.0, places=3)
        
        p_val_2 = z_to_p_value(2.0)
        self.assertAlmostEqual(p_val_2, 0.0455, places=3)
        
        p_val_3 = z_to_p_value(3.0)
        self.assertAlmostEqual(p_val_3, 0.0027, places=3)
    
    def test_cluster_similar_functions(self):
        """Test clustering of similar functions."""
        # Create groups of similar vectors
        entries = []
        
        # Group 1: Very similar vectors
        base1 = self.space.random_hv()
        for i in range(3):
            hv = base1.copy()
            # Add small noise
            noise_idx = np.random.choice(8192, size=100, replace=False)
            hv[noise_idx] *= -1
            entries.append((f"group1_{i}", hv, {}))
        
        # Group 2: Another set of similar vectors
        base2 = self.space.random_hv()
        for i in range(3):
            hv = base2.copy()
            noise_idx = np.random.choice(8192, size=100, replace=False)
            hv[noise_idx] *= -1
            entries.append((f"group2_{i}", hv, {}))
        
        # Singleton: Unrelated vector
        entries.append(("singleton", self.space.random_hv(), {}))
        
        # Perform clustering
        clusters = self.analyzer.cluster_similar_functions(
            entries, self.space, min_cluster_size=2
        )
        
        # Should find 2 clusters
        self.assertEqual(len(clusters), 2)
        
        # Check cluster membership
        cluster_members = []
        for cluster in clusters:
            cluster_members.extend(cluster['functions'])
        
        # Group 1 members should be in same cluster
        group1_members = [f"group1_{i}" for i in range(3)]
        group1_cluster = None
        for cluster in clusters:
            if any(m in cluster['functions'] for m in group1_members):
                group1_cluster = cluster
                break
        
        self.assertIsNotNone(group1_cluster)
        for member in group1_members:
            self.assertIn(member, group1_cluster['functions'])
    
    def test_performance_benchmark(self):
        """Benchmark similarity analysis performance."""
        # Generate many vectors
        num_vectors = 1000
        vectors = []
        
        start_gen = time.time()
        for i in range(num_vectors):
            hv = self.space.random_hv()
            vectors.append((f"vec_{i}", hv, {}))
        gen_time = time.time() - start_gen
        
        # Benchmark clustering
        start_cluster = time.time()
        clusters = self.analyzer.cluster_similar_functions(
            vectors, self.space, min_cluster_size=5
        )
        cluster_time = time.time() - start_cluster
        
        print(f"Vector generation: {gen_time:.3f}s for {num_vectors} vectors")
        print(f"Clustering: {cluster_time:.3f}s, found {len(clusters)} clusters")
        
        # Should complete in reasonable time
        self.assertLess(cluster_time, 30.0)  # 30 seconds max
        
        # Calculate throughput
        comparisons = (num_vectors * (num_vectors - 1)) / 2
        throughput = comparisons / cluster_time
        print(f"Throughput: {throughput:.0f} comparisons/second")
        
        self.assertGreater(throughput, 1000)  # At least 1000 comparisons/second


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_empty_input(self):
        """Test handling of empty inputs."""
        encoder = SemanticEncoder({'dimensions': 8192})
        
        # Empty code
        hv, features = encoder.encode_function("")
        self.assertIsNotNone(hv)
        self.assertEqual(len(features), len(encoder.config.get('channels', [])))
    
    def test_malformed_code(self):
        """Test handling of malformed code."""
        encoder = SemanticEncoder({'dimensions': 8192})
        
        # Syntax error
        malformed = "def broken(:\n    return"
        
        hv, features = encoder.encode_function(malformed)
        # Should still return a vector (might be less meaningful)
        self.assertIsNotNone(hv)
        self.assertEqual(len(hv), 8192)
    
    def test_very_large_function(self):
        """Test handling of very large functions."""
        encoder = SemanticEncoder({'dimensions': 8192})
        
        # Generate large function
        lines = ["def large_function():"]
        for i in range(1000):
            lines.append(f"    var_{i} = {i}")
            lines.append(f"    if var_{i} > 0:")
            lines.append(f"        result_{i} = var_{i} * 2")
        lines.append("    return None")
        
        large_code = "\n".join(lines)
        
        start = time.time()
        hv, features = encoder.encode_function(large_code)
        elapsed = time.time() - start
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 5.0)
        
        # Should still produce valid encoding
        self.assertIsNotNone(hv)
        self.assertEqual(len(hv), 8192)


if __name__ == '__main__':
    unittest.main()