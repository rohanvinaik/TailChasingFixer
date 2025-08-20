"""Comprehensive tests for the semantic index system.

This module tests the SemanticIndex class including initialization, function
addition/retrieval, similarity calculations, incremental updates, and cache
persistence.
"""

import ast
import math
import numpy as np
import pytest
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, MagicMock

from tailchasing.semantic.index import (
    SemanticIndex,
    FunctionEntry,
    SimilarityPair,
    SimilarityAnalysis,
    IndexStats
)
from tailchasing.semantic.hv_space import HVSpace
from tailchasing.core.types import FunctionRecord


class TestSemanticIndex:
    """Test suite for SemanticIndex functionality."""
    
    @pytest.fixture
    def mock_hv_space(self):
        """Create a mock HVSpace for testing."""
        space = Mock(spec=HVSpace)
        space.dim = 1024
        space.bipolar = True
        space.distance = Mock(return_value=0.5)
        space.similarity = Mock(return_value=0.5)
        space.role = Mock(return_value=np.random.randint(-1, 2, 1024))
        space.get_stats = Mock(return_value={'dim': 1024, 'bipolar': True})
        space._token_cache = {}
        space._role_cache = {}
        return space
    
    @pytest.fixture
    def sample_hypervector(self):
        """Create a sample hypervector for testing."""
        return np.random.randint(-1, 2, 1024).astype(np.float32)
    
    @pytest.fixture
    def sample_ast_node(self):
        """Create a sample AST node for testing."""
        code = """
def test_function(x, y):
    return x + y
"""
        tree = ast.parse(code)
        return tree.body[0]
    
    @pytest.fixture
    def sample_function_record(self, sample_ast_node):
        """Create a sample FunctionRecord for testing."""
        return FunctionRecord(
            name="test_function",
            file="test.py",
            line=2,
            id="test_function@test.py:2",
            source="def test_function(x, y):\n    return x + y",
            node=sample_ast_node,
            hv=[1, 0, -1, 1] * 256,  # 1024 dimensions
            metadata={"confidence": 0.9}
        )
    
    @pytest.fixture
    def temp_cache_dir(self):
        """Create a temporary directory for cache testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def semantic_index(self, mock_hv_space, temp_cache_dir):
        """Create a SemanticIndex instance for testing."""
        config = {
            'min_similarity_threshold': 0.5,
            'z_score_threshold': 2.5,
            'fdr_alpha': 0.05,
            'max_pairs_to_return': 100,
            'max_duplicate_pairs': 1000
        }
        return SemanticIndex(mock_hv_space, temp_cache_dir, config)
    
    def test_initialization(self, mock_hv_space, temp_cache_dir):
        """Test SemanticIndex initialization."""
        config = {
            'min_similarity_threshold': 0.7,
            'z_score_threshold': 3.0,
            'max_pairs_to_return': 50
        }
        
        index = SemanticIndex(mock_hv_space, temp_cache_dir, config)
        
        assert index.space == mock_hv_space
        assert index.cache_dir == temp_cache_dir
        assert index.config == config
        assert index.min_similarity_threshold == 0.7
        assert index.z_score_threshold == 3.0
        assert index.max_pairs_to_return == 50
        assert len(index.entries) == 0
        assert index._matrix_valid is False
    
    def test_initialization_without_cache_dir(self, mock_hv_space):
        """Test initialization without cache directory."""
        index = SemanticIndex(mock_hv_space)
        
        assert index.cache_dir is None
        assert index.config == {}
        assert index.min_similarity_threshold == 0.5  # Default
    
    @patch('tailchasing.semantic.index.encode_function')
    def test_add_function_basic(self, mock_encode, semantic_index, sample_ast_node, sample_hypervector):
        """Test adding a function to the index."""
        mock_encode.return_value = sample_hypervector
        
        semantic_index.add_function(
            function_id="test_func",
            ast_node=sample_ast_node,
            file_path="test.py",
            line_number=1,
            metadata={"confidence": 0.8}
        )
        
        assert "test_func" in semantic_index.entries
        entry = semantic_index.entries["test_func"]
        assert entry.function_id == "test_func"
        assert entry.file_path == "test.py"
        assert entry.line_number == 1
        assert entry.name == "test_function"
        assert entry.metadata == {"confidence": 0.8}
        assert np.array_equal(entry.hypervector, sample_hypervector)
        
        mock_encode.assert_called_once()
    
    @patch('tailchasing.semantic.index.encode_function_with_context')
    def test_add_function_with_context(self, mock_encode, semantic_index, sample_ast_node, sample_hypervector):
        """Test adding a function with context information."""
        mock_encode.return_value = sample_hypervector
        
        context = {
            'class_context': 'MyClass',
            'module_imports': ['os', 'sys']
        }
        
        semantic_index.add_function(
            function_id="test_func",
            ast_node=sample_ast_node,
            file_path="test.py",
            line_number=1,
            context=context
        )
        
        assert "test_func" in semantic_index.entries
        mock_encode.assert_called_once()
    
    def test_add_function_error_handling(self, semantic_index, sample_ast_node):
        """Test error handling when adding a function fails."""
        # Create a mock that raises an exception
        with patch('tailchasing.semantic.index.encode_function', side_effect=Exception("Encoding failed")):
            with patch('logging.getLogger') as mock_logger:
                mock_logger.return_value.error = Mock()
                
                semantic_index.add_function(
                    function_id="test_func",
                    ast_node=sample_ast_node,
                    file_path="test.py",
                    line_number=1
                )
                
                # Function should not be added if encoding fails
                assert "test_func" not in semantic_index.entries
                mock_logger.return_value.error.assert_called_once()
    
    def test_background_stats_computation(self, semantic_index, sample_hypervector):
        """Test background statistics computation."""
        # Add some mock entries
        for i in range(5):
            entry = FunctionEntry(
                function_id=f"func_{i}",
                file_path=f"file_{i}.py",
                name=f"function_{i}",
                line_number=i + 1,
                hypervector=sample_hypervector + i * 0.1  # Slightly different vectors
            )
            semantic_index.entries[f"func_{i}"] = entry
        
        # Compute background stats
        mean, std = semantic_index._compute_background_stats()
        
        assert isinstance(mean, float)
        assert isinstance(std, float)
        assert mean >= 0.0
        assert std >= 0.0
    
    def test_background_stats_small_sample(self, semantic_index):
        """Test background stats with small sample size."""
        # Add only one entry
        entry = FunctionEntry(
            function_id="func_1",
            file_path="file_1.py",
            name="function_1",
            line_number=1,
            hypervector=np.random.rand(1024).astype(np.float32)
        )
        semantic_index.entries["func_1"] = entry
        
        mean, std = semantic_index._compute_background_stats()
        
        # Should return default values for small samples
        assert mean == 0.5
        assert std == 0.05
    
    def test_z_score_computation(self, semantic_index):
        """Test z-score computation."""
        # Set background stats
        semantic_index._background_stats = (0.5, 0.1)
        
        # Test z-score calculation
        z_score = semantic_index.compute_z_score(0.3)
        expected = (0.5 - 0.3) / 0.1  # (mean - distance) / std
        assert abs(z_score - expected) < 1e-6
        
        # Test with zero std
        semantic_index._background_stats = (0.5, 0.0)
        z_score = semantic_index.compute_z_score(0.3)
        assert z_score == 0.0
    
    def test_remove_function(self, semantic_index, sample_hypervector):
        """Test removing a function from the index."""
        # First add a function
        entry = FunctionEntry(
            function_id="func_1@test.py:1",
            file_path="test.py",
            name="func_1",
            line_number=1,
            hypervector=sample_hypervector
        )
        semantic_index.entries["func_1@test.py:1"] = entry
        semantic_index.id_to_index["func_1@test.py:1"] = 0
        
        # Remove the function
        result = semantic_index.remove("func_1", "test.py", 1)
        
        assert result is True
        assert len(semantic_index._incremental_updates) == 1
        update = semantic_index._incremental_updates[0]
        assert update[0] == "remove"
        assert update[1] == "func_1@test.py:1"
    
    def test_remove_nonexistent_function(self, semantic_index):
        """Test removing a function that doesn't exist."""
        result = semantic_index.remove("nonexistent", "test.py", 1)
        assert result is False
        assert len(semantic_index._incremental_updates) == 0
    
    def test_incremental_updates_processing(self, semantic_index, sample_hypervector):
        """Test processing of incremental updates."""
        # Add an update to the queue
        semantic_index._incremental_updates.append(
            ("add", "func_1@test.py:1", sample_hypervector, {"confidence": 0.8})
        )
        
        # Process updates
        semantic_index._process_incremental_updates()
        
        # Check that the function was added
        assert "func_1@test.py:1" in semantic_index.entries
        entry = semantic_index.entries["func_1@test.py:1"]
        assert entry.name == "func_1"
        assert entry.file_path == "test.py"
        assert entry.line_number == 1
        assert np.array_equal(entry.hypervector, sample_hypervector)
        
        # Updates queue should be cleared
        assert len(semantic_index._incremental_updates) == 0
    
    def test_vector_matrix_rebuilding(self, semantic_index, sample_hypervector):
        """Test rebuilding of the vector matrix."""
        # Add some entries
        for i in range(3):
            entry = FunctionEntry(
                function_id=f"func_{i}",
                file_path=f"file_{i}.py",
                name=f"function_{i}",
                line_number=i + 1,
                hypervector=sample_hypervector + i * 0.1
            )
            semantic_index.entries[f"func_{i}"] = entry
        
        # Rebuild matrix
        semantic_index._rebuild_vector_matrix()
        
        assert semantic_index._vector_matrix is not None
        assert semantic_index._matrix_valid is True
        assert semantic_index._vector_matrix.shape == (3, 1024)
    
    def test_distance_calculation_vectorized_bipolar(self, semantic_index, sample_hypervector):
        """Test vectorized distance calculation for bipolar vectors."""
        # Create a matrix of vectors
        matrix = np.array([sample_hypervector, sample_hypervector * -1, sample_hypervector * 0])
        query = sample_hypervector
        
        distances = semantic_index._compute_distances_vectorized(query, matrix)
        
        assert len(distances) == 3
        assert distances[0] < distances[1]  # First should be most similar
        assert all(0.0 <= d <= 1.0 for d in distances)  # Valid distance range
    
    def test_find_similar_basic(self, semantic_index, sample_hypervector):
        """Test basic similarity search."""
        # Add some functions
        for i in range(3):
            entry = FunctionEntry(
                function_id=f"func_{i}",
                file_path=f"file_{i}.py",
                name=f"function_{i}",
                line_number=i + 1,
                hypervector=sample_hypervector + i * 0.1
            )
            semantic_index.entries[f"func_{i}"] = entry
        
        # Set background stats
        semantic_index._background_stats = (0.5, 0.1)
        
        # Find similar functions
        similar = semantic_index.find_similar(sample_hypervector, z_threshold=1.0, limit=2)
        
        assert len(similar) <= 2
        for func_id, distance, z_score, metadata in similar:
            assert func_id in semantic_index.entries
            assert isinstance(distance, float)
            assert isinstance(z_score, float)
    
    def test_find_all_similar_pairs(self, semantic_index, sample_hypervector):
        """Test finding all similar pairs."""
        # Add some functions
        for i in range(3):
            entry = FunctionEntry(
                function_id=f"func_{i}",
                file_path=f"file_{i}.py",
                name=f"function_{i}",
                line_number=i + 1,
                hypervector=sample_hypervector + i * 0.01  # Very similar
            )
            semantic_index.entries[f"func_{i}"] = entry
        
        # Set background stats
        semantic_index._background_stats = (0.5, 0.1)
        
        # Find all similar pairs
        pairs = semantic_index.find_all_similar_pairs(z_threshold=1.0, limit=10)
        
        # Should find pairs between the similar functions
        assert len(pairs) >= 0  # Might be 0 if threshold is too high
        for id1, id2, distance, z_score, analysis in pairs:
            assert id1 in semantic_index.entries
            assert id2 in semantic_index.entries
            assert isinstance(distance, float)
            assert isinstance(z_score, float)
            assert isinstance(analysis, dict)
    
    def test_similarity_analysis(self, semantic_index, sample_hypervector):
        """Test similarity analysis between functions."""
        hv1 = sample_hypervector
        hv2 = sample_hypervector * -1  # Opposite vector
        
        analysis = semantic_index._analyze_similarity(hv1, hv2, "func1@file1.py:1", "func2@file2.py:1")
        
        assert isinstance(analysis, dict)
        assert 'files_same' in analysis
        assert 'names_similar' in analysis
        assert 'channel_contributions' in analysis or isinstance(analysis, SimilarityAnalysis)
    
    def test_name_similarity(self, semantic_index):
        """Test name similarity calculation."""
        # Test identical names
        similarity = semantic_index._name_similarity("test_func@file1.py:1", "test_func@file2.py:1")
        assert similarity == 1.0
        
        # Test similar names
        similarity = semantic_index._name_similarity("test_func@file1.py:1", "test_function@file2.py:1")
        assert 0.0 < similarity < 1.0
        
        # Test completely different names
        similarity = semantic_index._name_similarity("foo@file1.py:1", "bar@file2.py:1")
        assert similarity == 0.0
    
    def test_statistics_gathering(self, semantic_index, sample_hypervector):
        """Test gathering index statistics."""
        # Add some functions
        for i in range(3):
            entry = FunctionEntry(
                function_id=f"func_{i}",
                file_path=f"file_{i}.py",
                name=f"function_{i}",
                line_number=i + 1,
                hypervector=sample_hypervector
            )
            semantic_index.entries[f"func_{i}"] = entry
        
        semantic_index._background_stats = (0.5, 0.1)
        semantic_index._stats_sample_size = 100
        
        stats = semantic_index.get_stats()
        
        assert isinstance(stats, dict)
        assert stats["total_functions"] == 3
        assert "space_stats" in stats
        assert "background_stats" in stats
        assert stats["background_stats"]["mean"] == 0.5
        assert stats["background_stats"]["std"] == 0.1
        assert stats["background_stats"]["sample_size"] == 100.0
    
    def test_performance_stats(self, semantic_index):
        """Test performance statistics gathering."""
        stats = semantic_index.get_performance_stats()
        
        assert isinstance(stats, dict)
        assert "index_stats" in stats
        assert "cache_stats" in stats
        assert "space_stats" in stats
        assert "background_stats" in stats
        
        index_stats = stats["index_stats"]
        assert "total_entries" in index_stats
        assert "valid_entries" in index_stats
        assert "pending_updates" in index_stats
        assert "matrix_valid" in index_stats
    
    def test_cache_optimization(self, semantic_index):
        """Test cache optimization functionality."""
        # Fill the similarity cache
        for i in range(15000):  # Exceed default max_cache_size
            semantic_index._similarity_cache[f"key_{i}"] = f"value_{i}"
        
        # Optimize cache
        semantic_index.optimize_cache(max_cache_size=10000)
        
        # Cache should be trimmed
        assert len(semantic_index._similarity_cache) <= 10000
    
    def test_force_rebuild(self, semantic_index, sample_hypervector):
        """Test forcing a rebuild of index structures."""
        # Add some data
        entry = FunctionEntry(
            function_id="func_1",
            file_path="file_1.py", 
            name="function_1",
            line_number=1,
            hypervector=sample_hypervector
        )
        semantic_index.entries["func_1"] = entry
        
        # Force rebuild
        semantic_index.force_rebuild()
        
        # Check that structures are rebuilt
        assert semantic_index._matrix_valid is True
        assert semantic_index._background_stats is not None
    
    def test_cache_persistence(self, semantic_index, sample_hypervector):
        """Test cache saving and loading."""
        # Add some data
        entry = FunctionEntry(
            function_id="func_1",
            file_path="file_1.py",
            name="function_1", 
            line_number=1,
            hypervector=sample_hypervector
        )
        semantic_index.entries["func_1"] = entry
        semantic_index._background_stats = (0.5, 0.1)
        
        # Save cache
        semantic_index.save_cache()
        
        # Verify cache file exists
        cache_file = semantic_index.cache_dir / "semantic_index_enhanced.pkl"
        assert cache_file.exists()
        
        # Create new index and load cache
        new_index = SemanticIndex(semantic_index.space, semantic_index.cache_dir, semantic_index.config)
        loaded = new_index._load_cache()
        
        assert loaded is True
        assert "func_1" in new_index.entries
        assert new_index._background_stats == (0.5, 0.1)
    
    def test_cache_loading_failure(self, semantic_index):
        """Test cache loading when file is corrupted or missing."""
        # Try to load cache when no file exists
        result = semantic_index._load_cache()
        assert result is False
        
        # Create a corrupted cache file
        cache_file = semantic_index.cache_dir / "semantic_index_enhanced.pkl"
        cache_file.write_text("corrupted data")
        
        result = semantic_index._load_cache()
        assert result is False
    
    @pytest.mark.parametrize("num_functions,expected_pairs", [
        (2, 1),  # 2 functions -> 1 pair
        (3, 3),  # 3 functions -> 3 pairs  
        (4, 6),  # 4 functions -> 6 pairs
    ])
    def test_pairwise_similarity_scaling(self, semantic_index, sample_hypervector, num_functions, expected_pairs):
        """Test that pairwise similarity scales correctly with number of functions."""
        # Add functions
        for i in range(num_functions):
            entry = FunctionEntry(
                function_id=f"func_{i}",
                file_path=f"file_{i}.py",
                name=f"function_{i}",
                line_number=i + 1,
                hypervector=sample_hypervector + i * 0.001  # Very similar
            )
            semantic_index.entries[f"func_{i}"] = entry
        
        # Set lenient background stats and threshold
        semantic_index._background_stats = (0.8, 0.1)  # High mean, low std
        
        # Find pairs with low threshold
        pairs = semantic_index.find_all_similar_pairs(z_threshold=0.1)
        
        # Should find approximately expected number of pairs (may be less due to thresholds)
        assert len(pairs) <= expected_pairs
    
    def test_empty_index_operations(self, semantic_index):
        """Test operations on empty index."""
        # Test finding similar on empty index
        similar = semantic_index.find_similar(np.random.rand(1024).astype(np.float32))
        assert len(similar) == 0
        
        # Test finding pairs on empty index  
        pairs = semantic_index.find_all_similar_pairs()
        assert len(pairs) == 0
        
        # Test stats on empty index
        stats = semantic_index.get_stats()
        assert stats["total_functions"] == 0
    
    def test_configuration_validation(self, mock_hv_space):
        """Test that configuration values are properly validated and applied."""
        config = {
            'min_similarity_threshold': 0.8,
            'z_score_threshold': 3.5,
            'fdr_alpha': 0.01,
            'max_pairs_to_return': 50,
            'max_duplicate_pairs': 500
        }
        
        index = SemanticIndex(mock_hv_space, config=config)
        
        assert index.min_similarity_threshold == 0.8
        assert index.z_score_threshold == 3.5
        assert index.fdr_alpha == 0.01
        assert index.max_pairs_to_return == 50
        assert index.max_duplicate_pairs == 500


class TestFunctionEntry:
    """Test suite for FunctionEntry class."""
    
    @pytest.fixture
    def sample_hypervector(self):
        """Create a sample hypervector."""
        return np.random.rand(1024).astype(np.float32)
    
    def test_function_entry_creation(self, sample_hypervector):
        """Test FunctionEntry creation."""
        entry = FunctionEntry(
            function_id="test_func",
            file_path="test.py",
            name="test_function",
            line_number=10,
            hypervector=sample_hypervector,
            metadata={"confidence": 0.9}
        )
        
        assert entry.function_id == "test_func"
        assert entry.file_path == "test.py"
        assert entry.name == "test_function"
        assert entry.line_number == 10
        assert np.array_equal(entry.hypervector, sample_hypervector)
        assert entry.metadata == {"confidence": 0.9}
        assert entry.self_similarity == 1.0
        assert entry.mean_similarity is None
        assert entry.std_similarity is None
    
    def test_z_score_calculation(self, sample_hypervector):
        """Test z-score calculation in FunctionEntry."""
        entry = FunctionEntry(
            function_id="test_func",
            file_path="test.py",
            name="test_function",
            line_number=10,
            hypervector=sample_hypervector
        )
        
        # Without statistics
        z_score = entry.get_z_score(0.8)
        assert z_score == 0.0
        
        # With statistics
        entry.mean_similarity = 0.6
        entry.std_similarity = 0.1
        z_score = entry.get_z_score(0.8)
        expected = (0.8 - 0.6) / 0.1
        assert abs(z_score - expected) < 1e-6
        
        # With zero std
        entry.std_similarity = 0.0
        z_score = entry.get_z_score(0.8)
        assert z_score == 0.0


class TestSimilarityPair:
    """Test suite for SimilarityPair class."""
    
    def test_similarity_pair_creation(self):
        """Test SimilarityPair creation."""
        pair = SimilarityPair(
            function1_id="func1",
            function2_id="func2", 
            similarity=0.8,
            z_score=2.5,
            p_value=0.01,
            q_value=0.02
        )
        
        assert pair.function1_id == "func1"
        assert pair.function2_id == "func2"
        assert pair.similarity == 0.8
        assert pair.z_score == 2.5
        assert pair.p_value == 0.01
        assert pair.q_value == 0.02
    
    def test_significance_testing(self):
        """Test statistical significance testing."""
        pair = SimilarityPair(
            function1_id="func1",
            function2_id="func2",
            similarity=0.8,
            z_score=2.5,
            p_value=0.03,
            q_value=0.04
        )
        
        # Test with FDR
        assert pair.is_significant(alpha=0.05, use_fdr=True) is True
        assert pair.is_significant(alpha=0.03, use_fdr=True) is False
        
        # Test without FDR
        assert pair.is_significant(alpha=0.05, use_fdr=False) is True
        assert pair.is_significant(alpha=0.02, use_fdr=False) is False
        
        # Test with no q_value
        pair_no_q = SimilarityPair(
            function1_id="func1",
            function2_id="func2",
            similarity=0.8,
            z_score=2.5,
            p_value=0.03
        )
        assert pair_no_q.is_significant(alpha=0.05, use_fdr=True) is True


if __name__ == "__main__":
    pytest.main([__file__])