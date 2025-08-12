"""
Tests for cache management functionality.
"""

import ast
import json
import gzip
import tempfile
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest

from tailchasing.core.cache import CacheManager, CachedFileInfo


class TestCachedFileInfo:
    """Tests for CachedFileInfo class."""
    
    def test_from_file(self, tmp_path):
        """Test creating CachedFileInfo from a file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_content = "def test():\n    pass\n"
        test_file.write_text(test_content)
        
        # Create CachedFileInfo
        info = CachedFileInfo.from_file(str(test_file))
        
        assert info.file_path == str(test_file.absolute())
        assert info.mtime == test_file.stat().st_mtime
        assert info.size == len(test_content)
        assert info.content_hash is not None
        assert len(info.content_hash) == 64  # SHA-256 hex digest
    
    def test_is_valid_for_same_file(self, tmp_path):
        """Test cache validity for unchanged file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_content = "def test():\n    pass\n"
        test_file.write_text(test_content)
        
        # Create CachedFileInfo
        info = CachedFileInfo.from_file(str(test_file))
        
        # Should be valid for the same file
        assert info.is_valid_for(str(test_file)) is True
    
    def test_is_valid_for_modified_file(self, tmp_path):
        """Test cache invalidity for modified file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test():\n    pass\n")
        
        # Create CachedFileInfo
        info = CachedFileInfo.from_file(str(test_file))
        
        # Modify the file
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("def test():\n    return 42\n")
        
        # Should be invalid for modified file
        assert info.is_valid_for(str(test_file)) is False
    
    def test_is_valid_for_nonexistent_file(self, tmp_path):
        """Test cache invalidity for nonexistent file."""
        # Create a test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def test():\n    pass\n")
        
        # Create CachedFileInfo
        info = CachedFileInfo.from_file(str(test_file))
        
        # Delete the file
        test_file.unlink()
        
        # Should be invalid for nonexistent file
        assert info.is_valid_for(str(test_file)) is False


class TestCacheManager:
    """Tests for CacheManager class."""
    
    def test_initialization(self, tmp_path):
        """Test cache manager initialization."""
        manager = CacheManager(tmp_path, enabled=True)
        
        assert manager.root_dir == tmp_path
        assert manager.enabled is True
        assert manager.cache_dir == tmp_path / ".tailchasing_cache"
        assert manager.cache_dir.exists()
        
        # Check version file
        version_file = manager.cache_dir / "version"
        assert version_file.exists()
        assert version_file.read_text() == CacheManager.CACHE_VERSION
        
        # Check gitignore
        gitignore = manager.cache_dir / ".gitignore"
        assert gitignore.exists()
        assert gitignore.read_text() == "*\n"
    
    def test_disabled_cache(self, tmp_path):
        """Test disabled cache manager."""
        manager = CacheManager(tmp_path, enabled=False)
        
        assert manager.enabled is False
        assert not (tmp_path / ".tailchasing_cache").exists()
        
        # Operations should return None/do nothing
        assert manager.get_cached_ast("test.py") is None
        manager.cache_ast("test.py", ast.parse("pass"))  # Should not raise
    
    def test_cache_ast(self, tmp_path):
        """Test caching and retrieving AST."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_content = "def foo():\n    return 42\n"
        test_file.write_text(test_content)
        
        # Create cache manager
        manager = CacheManager(tmp_path, enabled=True)
        
        # Parse and cache AST
        tree = ast.parse(test_content)
        manager.cache_ast(str(test_file), tree)
        
        # Retrieve cached AST
        cached_tree = manager.get_cached_ast(str(test_file))
        
        assert cached_tree is not None
        assert ast.dump(cached_tree) == ast.dump(tree)
        assert manager.stats['hits'] == 1
        assert manager.stats['misses'] == 0
    
    def test_cache_invalidation_on_file_change(self, tmp_path):
        """Test cache invalidation when file changes."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    return 42\n")
        
        # Create cache manager and cache AST
        manager = CacheManager(tmp_path, enabled=True)
        tree = ast.parse("def foo():\n    return 42\n")
        manager.cache_ast(str(test_file), tree)
        
        # Modify file
        time.sleep(0.01)  # Ensure mtime changes
        test_file.write_text("def bar():\n    return 0\n")
        
        # Cache should be invalid
        cached_tree = manager.get_cached_ast(str(test_file))
        assert cached_tree is None
        assert manager.stats['misses'] == 1
    
    def test_cache_analysis_results(self, tmp_path):
        """Test caching analysis results."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        
        # Create cache manager
        manager = CacheManager(tmp_path, enabled=True)
        
        # Cache analysis results
        analysis_data = {
            'issues': ['issue1', 'issue2'],
            'metrics': {'complexity': 5}
        }
        manager.cache_analysis(str(test_file), 'test_analyzer', analysis_data)
        
        # Retrieve cached results
        cached_data = manager.get_cached_analysis(str(test_file), 'test_analyzer')
        
        assert cached_data == analysis_data
        assert manager.stats['hits'] == 1
    
    def test_cache_signatures(self, tmp_path):
        """Test caching duplicate detection signatures."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        
        # Create cache manager
        manager = CacheManager(tmp_path, enabled=True)
        
        # Cache signatures
        signatures = {
            'content_hash': 'abc123',
            'shingle_hashes': [1, 2, 3, 4, 5],
            'minhash_signature': [0.1, 0.2, 0.3]
        }
        manager.cache_signatures(str(test_file), signatures)
        
        # Retrieve cached signatures
        cached_sigs = manager.get_cached_signatures(str(test_file))
        
        assert cached_sigs == signatures
        assert manager.stats['hits'] == 1
    
    def test_cache_hypervector(self, tmp_path):
        """Test caching hypervectors."""
        # Create test file
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        
        # Create cache manager
        manager = CacheManager(tmp_path, enabled=True)
        
        # Cache hypervector
        hypervector = np.random.randn(1024).astype(np.float32)
        manager.cache_hypervector(str(test_file), hypervector)
        
        # Retrieve cached hypervector
        cached_hv = manager.get_cached_hypervector(str(test_file))
        
        assert cached_hv is not None
        assert np.array_equal(cached_hv, hypervector)
        assert manager.stats['hits'] == 1
    
    def test_clear_cache(self, tmp_path):
        """Test clearing all cache."""
        # Create cache manager and cache some data
        manager = CacheManager(tmp_path, enabled=True)
        
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        
        manager.cache_ast(str(test_file), ast.parse("pass"))
        manager.cache_analysis(str(test_file), 'test', {'data': 'value'})
        
        # Clear cache
        manager.clear_cache()
        
        # Cache should be empty
        assert manager.get_cached_ast(str(test_file)) is None
        assert manager.get_cached_analysis(str(test_file), 'test') is None
        assert len(manager.file_cache) == 0
        
        # Cache directory should be recreated
        assert manager.cache_dir.exists()
    
    def test_clear_file_cache(self, tmp_path):
        """Test clearing cache for specific file."""
        # Create cache manager
        manager = CacheManager(tmp_path, enabled=True)
        
        # Cache data for multiple files
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        file1.write_text("pass")
        file2.write_text("pass")
        
        manager.cache_ast(str(file1), ast.parse("pass"))
        manager.cache_ast(str(file2), ast.parse("pass"))
        
        # Clear cache for file1 only
        manager.clear_file_cache(str(file1))
        
        # file1 cache should be cleared
        assert manager.get_cached_ast(str(file1)) is None
        
        # file2 cache should remain
        assert manager.get_cached_ast(str(file2)) is not None
    
    def test_flush(self, tmp_path):
        """Test flushing cache to disk."""
        # Create cache manager
        manager = CacheManager(tmp_path, enabled=True)
        
        # Cache some data
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        manager.cache_ast(str(test_file), ast.parse("pass"))
        
        # Mark as dirty
        manager.dirty_files.add(str(test_file))
        
        # Flush
        manager.flush()
        
        # Check index file exists
        index_path = manager.cache_dir / "index.json.gz"
        assert index_path.exists()
        
        # Load and verify index
        with gzip.open(index_path, 'rt', encoding='utf-8') as f:
            index_data = json.load(f)
        
        assert str(test_file) in index_data['files']
        assert len(manager.dirty_files) == 0
    
    def test_statistics(self, tmp_path):
        """Test cache statistics."""
        # Create cache manager
        manager = CacheManager(tmp_path, enabled=True)
        
        # Create test file
        test_file = tmp_path / "test.py"
        test_content = "def test(): pass"
        test_file.write_text(test_content)
        
        # Perform some operations
        manager.cache_ast(str(test_file), ast.parse(test_content))
        manager.get_cached_ast(str(test_file))  # Hit
        manager.get_cached_ast("nonexistent.py")  # Miss
        
        # Get statistics
        stats = manager.get_statistics()
        
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['cached_files'] == 1
        assert stats['cache_size_mb'] >= 0
        assert stats['bytes_saved_mb'] >= 0
    
    def test_context_manager(self, tmp_path):
        """Test using cache manager as context manager."""
        test_file = tmp_path / "test.py"
        test_file.write_text("pass")
        
        with CacheManager(tmp_path, enabled=True) as manager:
            manager.cache_ast(str(test_file), ast.parse("pass"))
            manager.dirty_files.add(str(test_file))
        
        # After exiting context, cache should be flushed
        index_path = tmp_path / ".tailchasing_cache" / "index.json.gz"
        assert index_path.exists()
    
    def test_version_mismatch(self, tmp_path):
        """Test handling of cache version mismatch."""
        # Create cache with old version
        cache_dir = tmp_path / ".tailchasing_cache"
        cache_dir.mkdir()
        version_file = cache_dir / "version"
        version_file.write_text("0.1")
        
        # Create some cache files
        test_cache = cache_dir / "test.cache"
        test_cache.write_text("old cache data")
        
        # Create manager (should clear old cache)
        manager = CacheManager(tmp_path, enabled=True)
        
        # Version should be updated
        assert version_file.read_text() == CacheManager.CACHE_VERSION
        
        # Old cache should be cleared
        assert not test_cache.exists()
    
    def test_numpy_conversion(self, tmp_path):
        """Test numpy array conversion for JSON serialization."""
        manager = CacheManager(tmp_path, enabled=True)
        
        # Test various numpy types
        data = {
            'array': np.array([1, 2, 3]),
            'nested': {
                'matrix': np.array([[1, 2], [3, 4]]),
                'scalar': np.float32(3.14)  # Test numpy scalar conversion
            },
            'list_with_array': [np.array([5, 6]), 'text'],
            'tuple_with_array': (np.array([7, 8]), 42)
        }
        
        # Convert
        converted = manager._convert_numpy_to_list(data)
        
        # Check conversions
        assert converted['array'] == [1, 2, 3]
        assert converted['nested']['matrix'] == [[1, 2], [3, 4]]
        assert isinstance(converted['nested']['scalar'], float)
        assert abs(converted['nested']['scalar'] - 3.14) < 0.01  # Check value is close
        assert converted['list_with_array'][0] == [5, 6]
        assert converted['tuple_with_array'][0] == [7, 8]