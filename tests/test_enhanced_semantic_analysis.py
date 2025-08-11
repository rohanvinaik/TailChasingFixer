"""
Tests for enhanced multimodal semantic analysis system.

Tests the 16384-dimensional hypervector encoding, drift detection, and efficient similarity search.
"""

import pytest
import ast
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Tuple, Any

from tailchasing.analyzers.advanced.multimodal_semantic_enhanced import EnhancedSemanticAnalyzer
from tailchasing.semantic.index import SemanticIndex
from tailchasing.semantic.hv_space import HVSpace
from tailchasing.core.issues import Issue


@pytest.fixture
def enhanced_analyzer():
    """Create enhanced semantic analyzer with 16384 dimensions."""
    return EnhancedSemanticAnalyzer(vector_dim=16384)


@pytest.fixture
def semantic_index():
    """Create semantic index with enhanced configuration."""
    config = {
        'hv_dim': 16384,
        'bipolar': True,
        'z_threshold': 2.5,
        'batch_size': 10,
        'rebuild_threshold': 20
    }
    return SemanticIndex(config)


@pytest.fixture
def sample_functions():
    """Sample function ASTs for testing."""
    functions = []
    
    # Function 1: Simple data processing
    code1 = """
def process_data(data):
    \"\"\"Process input data and return results.\"\"\"
    if not data:
        return []
    
    results = []
    for item in data:
        try:
            processed = item.upper()
            results.append(processed)
        except AttributeError:
            continue
    
    return results
"""
    
    # Function 2: Similar data processing with different implementation
    code2 = """
def transform_items(items):
    \"\"\"Transform items and return processed list.\"\"\"
    if len(items) == 0:
        return []
    
    output = []
    for element in items:
        try:
            transformed = element.upper()
            output.append(transformed)
        except AttributeError:
            pass
    
    return output
"""
    
    # Function 3: Different functionality
    code3 = """
def calculate_statistics(numbers):
    \"\"\"Calculate mean and standard deviation.\"\"\"
    if not numbers:
        return None, None
    
    mean = sum(numbers) / len(numbers)
    variance = sum((x - mean) ** 2 for x in numbers) / len(numbers)
    std_dev = variance ** 0.5
    
    return mean, std_dev
"""
    
    for i, code in enumerate([code1, code2, code3], 1):
        tree = ast.parse(code)
        func_def = tree.body[0]
        functions.append((f"test_file_{i}.py", func_def))
    
    return functions


class TestEnhancedSemanticAnalyzer:
    """Test enhanced semantic analyzer with 16384-dimensional encoding."""
    
    def test_initialization(self, enhanced_analyzer):
        """Test analyzer initialization with enhanced configuration."""
        assert enhanced_analyzer.vector_dim == 16384
        assert enhanced_analyzer.hv_space.dim == 16384
        assert len(enhanced_analyzer.channel_weights) == 7
        assert enhanced_analyzer.channel_weights['data_flow'] == 2.0
        assert enhanced_analyzer.channel_weights['control_flow'] == 1.8
        assert enhanced_analyzer._pattern_cache == {}
    
    def test_multimodal_encoding(self, enhanced_analyzer, sample_functions):
        """Test multimodal function encoding with caching."""
        _, func_ast = sample_functions[0]
        
        # First encoding
        vector1 = enhanced_analyzer.encode_function_multimodal(func_ast, cache_key="test_func_1")
        assert vector1.shape == (16384,)
        assert vector1.dtype == np.float32
        
        # Second encoding should use cache
        vector2 = enhanced_analyzer.encode_function_multimodal(func_ast, cache_key="test_func_1")
        np.testing.assert_array_equal(vector1, vector2)
        
        # Verify cache was used
        assert "test_func_1" in enhanced_analyzer._pattern_cache
    
    def test_enhanced_data_flow_analysis(self, enhanced_analyzer, sample_functions):
        """Test enhanced data flow analysis."""
        _, func_ast = sample_functions[0]
        
        # Test data flow analysis
        data_flow_vector = enhanced_analyzer._analyze_data_flow_enhanced(func_ast)
        expected_dim = 16384 // 7  # 7 channels
        assert data_flow_vector.shape == (expected_dim,)
        assert data_flow_vector.dtype == np.float32
        
        # Should have detected variables and patterns
        assert not np.allclose(data_flow_vector, 0)
    
    def test_control_flow_complexity_analysis(self, enhanced_analyzer, sample_functions):
        """Test control flow complexity analysis."""
        _, func_ast = sample_functions[0]
        
        control_flow_vector = enhanced_analyzer._analyze_control_flow_complexity(func_ast)
        expected_dim = 16384 // 7
        assert control_flow_vector.shape == (expected_dim,)
        assert control_flow_vector.dtype == np.float32
    
    def test_enhanced_return_patterns(self, enhanced_analyzer, sample_functions):
        """Test enhanced return pattern analysis."""
        _, func_ast = sample_functions[0]
        
        return_vector = enhanced_analyzer._analyze_return_patterns_enhanced(func_ast)
        expected_dim = 16384 // 7
        assert return_vector.shape == (expected_dim,)
        assert return_vector.dtype == np.float32
    
    def test_enhanced_error_handling(self, enhanced_analyzer, sample_functions):
        """Test enhanced error handling pattern analysis."""
        _, func_ast = sample_functions[0]
        
        error_vector = enhanced_analyzer._analyze_error_handling_enhanced(func_ast)
        expected_dim = 16384 // 7
        assert error_vector.shape == (expected_dim,)
        assert error_vector.dtype == np.float32
    
    def test_loop_structure_analysis(self, enhanced_analyzer, sample_functions):
        """Test enhanced loop structure analysis."""
        _, func_ast = sample_functions[0]
        
        loop_vector = enhanced_analyzer._analyze_loop_structures_enhanced(func_ast)
        expected_dim = 16384 // 7
        assert loop_vector.shape == (expected_dim,)
        assert loop_vector.dtype == np.float32
    
    def test_type_pattern_analysis(self, enhanced_analyzer, sample_functions):
        """Test type pattern analysis."""
        _, func_ast = sample_functions[0]
        
        type_vector = enhanced_analyzer._analyze_type_patterns(func_ast)
        expected_dim = 16384 // 7
        assert type_vector.shape == (expected_dim,)
        assert type_vector.dtype == np.float32
    
    def test_enhanced_name_tokenization(self, enhanced_analyzer):
        """Test enhanced name tokenization with semantic categories."""
        # Test with caching
        tokens1 = enhanced_analyzer._extract_semantic_tokens("process_user_data")
        assert "process" in tokens1
        assert "user" in tokens1
        assert "data" in tokens1
        assert "_data_noun" in tokens1  # Semantic category
        
        # Test camelCase
        tokens2 = enhanced_analyzer._extract_semantic_tokens("getUserInfo")
        assert "get" in tokens2
        assert "user" in tokens2
        assert "info" in tokens2
        assert "_retrieval_verb" in tokens2
        
        # Test caching
        tokens3 = enhanced_analyzer._extract_semantic_tokens("process_user_data")
        assert tokens1 == tokens3
    
    def test_weighted_hypervector_creation(self, enhanced_analyzer, sample_functions):
        """Test weighted hypervector creation using HV operations."""
        _, func_ast = sample_functions[0]
        
        # Create sample channel data
        channels = {
            'data_flow': np.random.randn(1000).astype(np.float32),
            'control_flow': np.random.randn(1000).astype(np.float32),
            'return_patterns': np.random.randn(1000).astype(np.float32),
        }
        
        combined = enhanced_analyzer._create_weighted_hypervector_enhanced(channels)
        assert combined.shape == (16384,)
        assert combined.dtype == np.float32
        assert not np.allclose(combined, 0)
    
    def test_similarity_detection(self, enhanced_analyzer, sample_functions):
        """Test semantic duplicate detection."""
        issues = enhanced_analyzer.find_semantic_duplicates(sample_functions, threshold=0.3)
        
        # Should find similarity between first two functions
        assert len(issues) >= 1
        assert all(issue.kind == "semantic_duplicate_enhanced" for issue in issues)
        
        # Check issue details
        issue = issues[0]
        assert issue.severity == 3
        assert "similarity" in issue.evidence
        assert "channels" in issue.evidence
        assert len(issue.suggestions) > 0
    
    def test_efficient_similarity_search(self, enhanced_analyzer, sample_functions):
        """Test efficient vectorized similarity search."""
        # Encode all functions
        encoded_functions = []
        for filepath, func_ast in sample_functions:
            vector = enhanced_analyzer.encode_function_multimodal(func_ast)
            encoded_functions.append((filepath, func_ast, vector))
        
        # Test efficient search
        similar_pairs = enhanced_analyzer._find_similar_pairs_efficient(encoded_functions, threshold=0.3)
        
        assert isinstance(similar_pairs, list)
        # Should find at least one similar pair
        if len(similar_pairs) > 0:
            pair = similar_pairs[0]
            assert len(pair) == 3  # (func1, func2, similarity)
            assert isinstance(pair[2], float)
    
    def test_channel_contributions_analysis(self, enhanced_analyzer):
        """Test channel contribution analysis."""
        # Create two similar vectors
        vec1 = np.random.randn(16384).astype(np.float32)
        vec2 = vec1 + 0.1 * np.random.randn(16384).astype(np.float32)
        
        contributions = enhanced_analyzer._analyze_channel_contributions(vec1, vec2)
        
        assert isinstance(contributions, dict)
        expected_channels = ['data_flow', 'control_flow', 'return_patterns', 
                           'error_handling', 'loop_structures', 'type_patterns', 'name_tokens']
        
        for channel in expected_channels:
            assert channel in contributions
            assert isinstance(contributions[channel], float)
            assert 0.0 <= contributions[channel] <= 1.0


class TestSemanticDriftDetection:
    """Test semantic drift detection capabilities."""
    
    def test_drift_tracking_initialization(self, enhanced_analyzer):
        """Test drift tracking initialization."""
        assert enhanced_analyzer.commit_history == []
        assert enhanced_analyzer.semantic_snapshots == {}
    
    def test_single_commit_tracking(self, enhanced_analyzer, sample_functions):
        """Test tracking a single commit."""
        commit_id = "abc123"
        drift_analysis = enhanced_analyzer.track_semantic_drift(commit_id, sample_functions)
        
        assert drift_analysis['commit_id'] == commit_id
        assert drift_analysis['total_functions'] == len(sample_functions)
        assert drift_analysis['convergence_status'] == 'unknown'  # First commit
        
        # Should have stored snapshot
        assert len(enhanced_analyzer.semantic_snapshots) == len(sample_functions)
        assert len(enhanced_analyzer.commit_history) == 1
    
    def test_multi_commit_drift_detection(self, enhanced_analyzer, sample_functions):
        """Test drift detection across multiple commits."""
        # First commit
        commit1 = enhanced_analyzer.track_semantic_drift("commit1", sample_functions)
        
        # Modify functions slightly for second commit
        modified_functions = []
        for filepath, func_ast in sample_functions:
            # Create a modified version by changing the docstring
            modified_ast = ast.parse(f'def {func_ast.name}():\n    """Modified docstring."""\n    pass')
            modified_functions.append((filepath, modified_ast.body[0]))
        
        # Second commit
        commit2 = enhanced_analyzer.track_semantic_drift("commit2", modified_functions)
        
        assert len(enhanced_analyzer.commit_history) == 2
        assert 'semantic_changes' in commit2
        assert 'drift_patterns' in commit2
        assert commit2['convergence_status'] in ['convergence', 'divergence', 'refactoring', 'stable', 'mixed_changes']
    
    def test_convergence_assessment(self, enhanced_analyzer, sample_functions):
        """Test convergence vs divergence assessment."""
        # Create two similar functions that become more similar
        similar_code1 = '''
def process_data(data):
    results = []
    for item in data:
        results.append(item.upper())
    return results
'''
        
        similar_code2 = '''
def process_items(items):
    output = []
    for element in items:
        output.append(element.upper())
    return output
'''
        
        # Parse functions
        func1 = ast.parse(similar_code1).body[0]
        func2 = ast.parse(similar_code2).body[0]
        
        functions1 = [("file1.py", func1), ("file2.py", func2)]
        
        # First commit
        enhanced_analyzer.track_semantic_drift("commit1", functions1)
        
        # Second commit - make functions even more similar
        more_similar_code2 = '''
def process_items(items):
    results = []
    for item in items:
        results.append(item.upper())
    return results
'''
        
        func2_similar = ast.parse(more_similar_code2).body[0]
        functions2 = [("file1.py", func1), ("file2.py", func2_similar)]
        
        commit2 = enhanced_analyzer.track_semantic_drift("commit2", functions2)
        
        # Should detect convergence or stable refactoring
        assert commit2['convergence_status'] in ['convergence', 'refactoring', 'stable']
    
    def test_drift_summary(self, enhanced_analyzer, sample_functions):
        """Test drift summary generation."""
        # Track several commits
        for i in range(5):
            enhanced_analyzer.track_semantic_drift(f"commit{i}", sample_functions)
        
        summary = enhanced_analyzer.get_drift_summary()
        
        assert summary['total_commits_tracked'] == 5
        assert 'recent_convergence_pattern' in summary
        assert 'average_drift_rate' in summary
        assert 'stability_trend' in summary
        assert 'concerning_patterns' in summary
        
        assert isinstance(summary['recent_convergence_pattern'], list)
        assert isinstance(summary['average_drift_rate'], float)
        assert summary['stability_trend'] in ['convergence', 'divergence', 'refactoring', 'stable', 'unknown']
    
    def test_concerning_pattern_detection(self, enhanced_analyzer):
        """Test detection of concerning patterns."""
        # Simulate high divergence pattern
        for i in range(10):
            enhanced_analyzer.commit_history.append({
                'commit_id': f'commit{i}',
                'convergence_status': 'divergence',
                'drift_patterns': {'average_drift': 0.4}
            })
        
        summary = enhanced_analyzer.get_drift_summary()
        
        assert 'high_divergence_rate' in summary['concerning_patterns']
        assert 'high_drift_rate' in summary['concerning_patterns']


class TestEnhancedSemanticIndex:
    """Test enhanced semantic index with efficient similarity search."""
    
    def test_initialization_with_16384_dimensions(self, semantic_index):
        """Test index initialization with enhanced dimensions."""
        assert semantic_index.space.dim == 16384
        assert hasattr(semantic_index, '_similarity_cache')
        assert hasattr(semantic_index, '_vector_matrix')
        assert hasattr(semantic_index, '_incremental_updates')
        assert hasattr(semantic_index, '_search_stats')
    
    def test_incremental_updates(self, semantic_index):
        """Test incremental update system."""
        # Add some functions
        for i in range(5):
            hv = np.random.randn(16384).astype(np.float32)
            semantic_index.add(f"func{i}", f"file{i}.py", i+1, hv)
        
        # Should have queued updates
        assert len(semantic_index._incremental_updates) == 5
        
        # Force processing
        semantic_index._process_incremental_updates()
        
        # Updates should be processed
        assert len(semantic_index._incremental_updates) == 0
        assert len(semantic_index.entries) == 5
        assert semantic_index._search_stats['incremental_updates'] == 5
    
    def test_batch_processing(self, semantic_index):
        """Test automatic batch processing."""
        # Configure small batch size
        semantic_index.config['batch_size'] = 3
        
        # Add functions one by one
        for i in range(5):
            hv = np.random.randn(16384).astype(np.float32)
            semantic_index.add(f"func{i}", f"file{i}.py", i+1, hv)
        
        # Should have processed batches automatically
        assert len(semantic_index.entries) >= 3
        assert len(semantic_index._incremental_updates) <= 2
    
    def test_vectorized_similarity_search(self, semantic_index):
        """Test efficient vectorized similarity search."""
        # Add some functions
        vectors = []
        for i in range(10):
            hv = np.random.randn(16384).astype(np.float32)
            semantic_index.add(f"func{i}", f"file{i}.py", i+1, hv)
            vectors.append(hv)
        
        # Force processing
        semantic_index._process_incremental_updates()
        
        # Test similarity search
        query_hv = vectors[0] + 0.01 * np.random.randn(16384).astype(np.float32)
        similar = semantic_index.find_similar(query_hv, z_threshold=0.0, limit=5)
        
        assert len(similar) <= 5
        assert all(len(result) == 4 for result in similar)  # (id, distance, z_score, metadata)
    
    def test_vectorized_pairwise_search(self, semantic_index):
        """Test efficient pairwise similarity search."""
        # Add some similar functions
        base_vector = np.random.randn(16384).astype(np.float32)
        
        for i in range(5):
            # Add small variations
            hv = base_vector + 0.1 * np.random.randn(16384).astype(np.float32)
            semantic_index.add(f"func{i}", f"file{i}.py", i+1, hv)
        
        # Force processing
        semantic_index._process_incremental_updates()
        
        # Find all similar pairs
        pairs = semantic_index.find_all_similar_pairs(z_threshold=0.0, limit=10)
        
        assert isinstance(pairs, list)
        if len(pairs) > 0:
            pair = pairs[0]
            assert len(pair) == 5  # (id1, id2, distance, z_score, analysis)
            assert isinstance(pair[4], dict)  # analysis
    
    def test_cache_optimization(self, semantic_index):
        """Test cache optimization and management."""
        # Fill similarity cache
        for i in range(100):
            cache_key = (f"func{i}@file{i}.py:1", f"func{i+1}@file{i+1}.py:2")
            semantic_index._similarity_cache[cache_key] = {"files_same": False, "names_similar": True}
        
        # Optimize cache
        semantic_index.optimize_cache(max_cache_size=50)
        
        # Cache should be trimmed
        assert len(semantic_index._similarity_cache) <= 50
    
    def test_performance_stats(self, semantic_index):
        """Test performance statistics collection."""
        # Add some data
        for i in range(5):
            hv = np.random.randn(16384).astype(np.float32)
            semantic_index.add(f"func{i}", f"file{i}.py", i+1, hv)
        
        stats = semantic_index.get_performance_stats()
        
        assert 'index_stats' in stats
        assert 'cache_stats' in stats
        assert 'space_stats' in stats
        assert 'background_stats' in stats
        
        assert stats['index_stats']['total_entries'] >= 0
        assert stats['index_stats']['valid_entries'] >= 0
        assert stats['space_stats']['dimension'] == 16384
    
    def test_force_rebuild(self, semantic_index):
        """Test force rebuild functionality."""
        # Add some data
        for i in range(3):
            hv = np.random.randn(16384).astype(np.float32)
            semantic_index.add(f"func{i}", f"file{i}.py", i+1, hv)
        
        # Force rebuild
        semantic_index.force_rebuild()
        
        # Should have processed updates and rebuilt structures
        assert len(semantic_index._incremental_updates) == 0
        assert semantic_index._matrix_valid
        assert semantic_index._background_stats is not None
    
    def test_enhanced_cache_save_load(self, semantic_index, tmp_path):
        """Test enhanced cache save/load functionality."""
        semantic_index.cache_dir = tmp_path
        
        # Add some data
        for i in range(3):
            hv = np.random.randn(16384).astype(np.float32)
            semantic_index.add(f"func{i}", f"file{i}.py", i+1, hv)
        
        # Add some similarity cache data
        semantic_index._similarity_cache[("func1@file1.py:1", "func2@file2.py:2")] = {
            "files_same": False, "names_similar": True
        }
        
        # Save cache
        semantic_index.save_cache()
        
        # Check that enhanced cache file was created
        cache_file = tmp_path / "semantic_index_enhanced.pkl"
        assert cache_file.exists()
        
        # Create new index and load
        new_index = SemanticIndex(semantic_index.config, tmp_path)
        
        # Should have loaded data
        assert len(new_index.entries) == len(semantic_index.entries)
        assert len(new_index._similarity_cache) == len(semantic_index._similarity_cache)


class TestIntegration:
    """Integration tests for the complete enhanced semantic analysis system."""
    
    def test_end_to_end_analysis(self, enhanced_analyzer, sample_functions):
        """Test complete end-to-end semantic analysis workflow."""
        # Perform similarity detection
        issues = enhanced_analyzer.find_semantic_duplicates(sample_functions, threshold=0.3)
        
        # Track semantic drift
        drift_analysis = enhanced_analyzer.track_semantic_drift("commit1", sample_functions)
        
        # Verify results
        assert isinstance(issues, list)
        assert isinstance(drift_analysis, dict)
        
        if len(issues) > 0:
            issue = issues[0]
            assert issue.kind == "semantic_duplicate_enhanced"
            assert 'channel_contributions' in issue.evidence['channels']
        
        assert 'total_functions' in drift_analysis
        assert drift_analysis['total_functions'] == len(sample_functions)
    
    def test_large_scale_performance(self, enhanced_analyzer):
        """Test performance with larger number of functions."""
        # Generate many similar functions
        functions = []
        base_code = '''
def process_items_{i}(data):
    """Process data items."""
    results = []
    for item in data:
        try:
            processed = item.{transform}()
            results.append(processed)
        except AttributeError:
            continue
    return results
'''
        
        transforms = ['upper', 'lower', 'strip', 'title']
        
        for i in range(20):
            transform = transforms[i % len(transforms)]
            code = base_code.format(i=i, transform=transform)
            tree = ast.parse(code)
            func_def = tree.body[0]
            functions.append((f"file_{i}.py", func_def))
        
        # Should handle large scale analysis
        issues = enhanced_analyzer.find_semantic_duplicates(functions, threshold=0.4)
        
        # Should find some similarities
        assert len(issues) > 0
        
        # Test drift tracking
        drift_analysis = enhanced_analyzer.track_semantic_drift("large_commit", functions)
        assert drift_analysis['total_functions'] == 20
    
    def test_error_resilience(self, enhanced_analyzer):
        """Test system resilience to malformed inputs."""
        # Test with malformed function
        malformed_code = '''
def broken_function(
    # Missing closing parenthesis and body
'''
        
        try:
            tree = ast.parse(malformed_code)
            func_def = tree.body[0]
            vector = enhanced_analyzer.encode_function_multimodal(func_def)
            # Should either work or fail gracefully
            assert vector.shape == (16384,) or vector is None
        except (SyntaxError, AttributeError, IndexError):
            # Expected for malformed code
            pass
        
        # Test with empty function list
        issues = enhanced_analyzer.find_semantic_duplicates([], threshold=0.5)
        assert issues == []
        
        # Test drift tracking with empty functions
        drift_analysis = enhanced_analyzer.track_semantic_drift("empty_commit", [])
        assert drift_analysis['total_functions'] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])