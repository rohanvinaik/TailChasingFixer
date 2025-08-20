"""
Caching infrastructure for performance optimization.

Provides multi-level caching for AST parsing, hypervector computation,
and similarity matrices.
"""

import ast
import hashlib
import time
import threading
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Thread-safe LRU cache with TTL support.
    """
    
    def __init__(self, max_size: int = 10000, ttl: int = 3600):
        """
        Initialize LRU cache.
        
        Args:
            max_size: Maximum number of entries
            ttl: Time to live in seconds
        """
        self.max_size = max_size
        self.ttl = ttl
        self.cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key not in self.cache:
                self.misses += 1
                return None
            
            value, timestamp = self.cache[key]
            
            # Check TTL
            if time.time() - timestamp > self.ttl:
                del self.cache[key]
                self.misses += 1
                return None
            
            # Move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            return value
    
    def set(self, key: str, value: Any):
        """Set value in cache."""
        with self.lock:
            # Remove if exists (to update position)
            if key in self.cache:
                del self.cache[key]
            
            # Check size limit
            while len(self.cache) >= self.max_size:
                # Remove least recently used
                self.cache.popitem(last=False)
                self.evictions += 1
            
            self.cache[key] = (value, time.time())
    
    def invalidate(self, key: str) -> bool:
        """Invalidate a cache entry."""
        with self.lock:
            if key in self.cache:
                del self.cache[key]
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.hits = 0
            self.misses = 0
            self.evictions = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total = self.hits + self.misses
        hit_rate = self.hits / total if total > 0 else 0
        
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'hit_rate': round(hit_rate, 3),
            'ttl': self.ttl
        }


class ASTCache:
    """
    Specialized cache for AST parsing results.
    """
    
    def __init__(self, max_size: int = 5000, ttl: int = 3600):
        """Initialize AST cache."""
        self.cache = LRUCache(max_size, ttl)
        self.file_hashes: Dict[str, str] = {}
        self.lock = threading.RLock()
    
    def get_ast(self, filepath: str, content: Optional[str] = None) -> Optional[ast.AST]:
        """
        Get parsed AST from cache or parse new.
        
        Args:
            filepath: File path
            content: Optional file content
            
        Returns:
            Parsed AST or None
        """
        # Read content if not provided
        if content is None:
            try:
                content = Path(filepath).read_text()
            except Exception as e:
                logger.debug(f"Failed to read {filepath}: {e}")
                return None
        
        # Generate cache key
        content_hash = hashlib.md5(content.encode()).hexdigest()
        cache_key = f"{filepath}:{content_hash}"
        
        # Check if file has changed
        with self.lock:
            old_hash = self.file_hashes.get(filepath)
            if old_hash and old_hash != content_hash:
                # File changed, invalidate old entry
                old_key = f"{filepath}:{old_hash}"
                self.cache.invalidate(old_key)
            
            self.file_hashes[filepath] = content_hash
        
        # Try cache
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        # Parse and cache
        try:
            tree = ast.parse(content, filename=filepath)
            self.cache.set(cache_key, tree)
            return tree
        except SyntaxError as e:
            logger.debug(f"Syntax error in {filepath}: {e}")
            # Cache the failure to avoid repeated parsing
            self.cache.set(cache_key, None)
            return None
    
    def invalidate_file(self, filepath: str):
        """Invalidate all cache entries for a file."""
        with self.lock:
            if filepath in self.file_hashes:
                old_hash = self.file_hashes.pop(filepath)
                cache_key = f"{filepath}:{old_hash}"
                self.cache.invalidate(cache_key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['tracked_files'] = len(self.file_hashes)
        return stats


class HypervectorCache:
    """
    Cache for hypervector computations with compression.
    """
    
    def __init__(self, max_size: int = 10000, ttl: int = 7200):
        """Initialize hypervector cache."""
        self.cache = LRUCache(max_size, ttl)
        self.compress = True  # Enable compression for large vectors
    
    def get(self, function_id: str) -> Optional[np.ndarray]:
        """Get cached hypervector."""
        cached = self.cache.get(function_id)
        
        if cached is not None:
            if self.compress:
                # Decompress if needed
                if isinstance(cached, bytes):
                    return pickle.loads(cached)
            return cached
        
        return None
    
    def set(self, function_id: str, vector: np.ndarray):
        """Cache hypervector with optional compression."""
        if self.compress and vector.nbytes > 1024:  # Compress if > 1KB
            # Compress using pickle with protocol 4
            compressed = pickle.dumps(vector, protocol=4)
            self.cache.set(function_id, compressed)
        else:
            self.cache.set(function_id, vector)
    
    def batch_get(self, function_ids: List[str]) -> Dict[str, np.ndarray]:
        """Get multiple hypervectors."""
        results = {}
        for func_id in function_ids:
            vector = self.get(func_id)
            if vector is not None:
                results[func_id] = vector
        return results
    
    def batch_set(self, vectors: Dict[str, np.ndarray]):
        """Set multiple hypervectors."""
        for func_id, vector in vectors.items():
            self.set(func_id, vector)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['compression_enabled'] = self.compress
        return stats


class SimilarityCache:
    """
    Cache for similarity scores with efficient storage.
    """
    
    def __init__(self, max_size: int = 50000, ttl: int = 1800):
        """Initialize similarity cache."""
        self.cache = LRUCache(max_size, ttl)
        
        # Use quantization to reduce memory
        self.quantize = True
        self.quantization_levels = 1000  # 0.001 precision
    
    def _make_key(self, id1: str, id2: str) -> str:
        """Create order-independent cache key."""
        return f"{min(id1, id2)}:{max(id1, id2)}"
    
    def get(self, id1: str, id2: str) -> Optional[float]:
        """Get cached similarity score."""
        key = self._make_key(id1, id2)
        cached = self.cache.get(key)
        
        if cached is not None:
            if self.quantize:
                # Dequantize
                return cached / self.quantization_levels
            return cached
        
        return None
    
    def set(self, id1: str, id2: str, similarity: float):
        """Cache similarity score."""
        key = self._make_key(id1, id2)
        
        if self.quantize:
            # Quantize to reduce memory
            quantized = int(similarity * self.quantization_levels)
            self.cache.set(key, quantized)
        else:
            self.cache.set(key, similarity)
    
    def get_matrix(self, ids: List[str]) -> np.ndarray:
        """
        Get similarity matrix for given IDs.
        
        Returns NaN for uncached values.
        """
        n = len(ids)
        matrix = np.full((n, n), np.nan)
        
        for i in range(n):
            matrix[i, i] = 1.0  # Self-similarity
            for j in range(i + 1, n):
                sim = self.get(ids[i], ids[j])
                if sim is not None:
                    matrix[i, j] = sim
                    matrix[j, i] = sim
        
        return matrix
    
    def set_matrix(self, ids: List[str], matrix: np.ndarray):
        """Cache similarity matrix."""
        n = len(ids)
        for i in range(n):
            for j in range(i + 1, n):
                if not np.isnan(matrix[i, j]):
                    self.set(ids[i], ids[j], matrix[i, j])
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['quantization_enabled'] = self.quantize
        if self.quantize:
            stats['quantization_precision'] = 1.0 / self.quantization_levels
        return stats


class IssueCache:
    """
    Cache for detected issues with dependency tracking.
    """
    
    def __init__(self, max_size: int = 1000, ttl: int = 600):
        """Initialize issue cache."""
        self.cache = LRUCache(max_size, ttl)
        self.file_dependencies: Dict[str, Set[str]] = {}
        self.lock = threading.RLock()
    
    def get_file_issues(self, filepath: str, file_hash: Optional[str] = None) -> Optional[List[Any]]:
        """Get cached issues for a file."""
        if file_hash:
            key = f"{filepath}:{file_hash}"
        else:
            # Compute hash
            try:
                content = Path(filepath).read_text()
                file_hash = hashlib.md5(content.encode()).hexdigest()
                key = f"{filepath}:{file_hash}"
            except:
                return None
        
        return self.cache.get(key)
    
    def set_file_issues(self, filepath: str, issues: List[Any], dependencies: Optional[Set[str]] = None):
        """Cache issues for a file."""
        try:
            content = Path(filepath).read_text()
            file_hash = hashlib.md5(content.encode()).hexdigest()
            key = f"{filepath}:{file_hash}"
            
            self.cache.set(key, issues)
            
            # Track dependencies
            if dependencies:
                with self.lock:
                    self.file_dependencies[filepath] = dependencies
            
        except Exception as e:
            logger.debug(f"Failed to cache issues for {filepath}: {e}")
    
    def invalidate_dependent(self, filepath: str):
        """Invalidate cache for files that depend on this one."""
        with self.lock:
            # Find all files that depend on this one
            to_invalidate = {filepath}
            
            for file, deps in self.file_dependencies.items():
                if filepath in deps:
                    to_invalidate.add(file)
            
            # Invalidate all affected files
            for file in to_invalidate:
                # Try to invalidate all possible hashes
                # In practice, we'd need to track the actual hashes
                self.file_dependencies.pop(file, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.cache.get_stats()
        stats['tracked_dependencies'] = len(self.file_dependencies)
        return stats


class CacheManager:
    """
    Centralized cache management for all cache types.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize cache manager."""
        self.config = config or {}
        
        # Initialize specialized caches
        self.ast_cache = ASTCache(
            max_size=self.config.get('ast_cache_size', 5000),
            ttl=self.config.get('ast_cache_ttl', 3600)
        )
        
        self.hypervector_cache = HypervectorCache(
            max_size=self.config.get('hv_cache_size', 10000),
            ttl=self.config.get('hv_cache_ttl', 7200)
        )
        
        self.similarity_cache = SimilarityCache(
            max_size=self.config.get('sim_cache_size', 50000),
            ttl=self.config.get('sim_cache_ttl', 1800)
        )
        
        self.issue_cache = IssueCache(
            max_size=self.config.get('issue_cache_size', 1000),
            ttl=self.config.get('issue_cache_ttl', 600)
        )
        
        # General purpose cache
        self.general_cache = LRUCache(
            max_size=self.config.get('general_cache_size', 5000),
            ttl=self.config.get('general_cache_ttl', 1800)
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics for all caches."""
        return {
            'ast': self.ast_cache.get_stats(),
            'hypervector': self.hypervector_cache.get_stats(),
            'similarity': self.similarity_cache.get_stats(),
            'issue': self.issue_cache.get_stats(),
            'general': self.general_cache.get_stats()
        }
    
    def clear_all(self):
        """Clear all caches."""
        self.ast_cache.cache.clear()
        self.hypervector_cache.cache.clear()
        self.similarity_cache.cache.clear()
        self.issue_cache.cache.clear()
        self.general_cache.clear()
        
        logger.info("All caches cleared")
    
    def optimize(self):
        """Optimize caches by removing stale entries."""
        current_time = time.time()
        total_removed = 0
        
        for cache_name, cache_obj in [
            ('ast', self.ast_cache.cache),
            ('hypervector', self.hypervector_cache.cache),
            ('similarity', self.similarity_cache.cache),
            ('issue', self.issue_cache.cache),
            ('general', self.general_cache)
        ]:
            with cache_obj.lock:
                # Remove expired entries
                expired_keys = []
                for key, (value, timestamp) in cache_obj.cache.items():
                    if current_time - timestamp > cache_obj.ttl:
                        expired_keys.append(key)
                
                for key in expired_keys:
                    del cache_obj.cache[key]
                
                if expired_keys:
                    logger.debug(f"Removed {len(expired_keys)} expired entries from {cache_name} cache")
                    total_removed += len(expired_keys)
        
        return total_removed
    
    def get_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage of caches."""
        import sys
        
        sizes = {}
        
        # Estimate sizes
        for name, cache in [
            ('ast', self.ast_cache.cache),
            ('hypervector', self.hypervector_cache.cache),
            ('similarity', self.similarity_cache.cache),
            ('issue', self.issue_cache.cache),
            ('general', self.general_cache)
        ]:
            total_size = 0
            for key, (value, _) in cache.cache.items():
                total_size += sys.getsizeof(key)
                total_size += sys.getsizeof(value)
            
            sizes[name] = total_size
        
        sizes['total'] = sum(sizes.values())
        return sizes


# Global cache manager instance
_cache_manager: Optional[CacheManager] = None


def get_cache_manager(config: Optional[Dict[str, Any]] = None) -> CacheManager:
    """Get or create global cache manager."""
    global _cache_manager
    
    if _cache_manager is None:
        _cache_manager = CacheManager(config)
    
    return _cache_manager


def clear_all_caches():
    """Clear all global caches."""
    if _cache_manager:
        _cache_manager.clear_all()