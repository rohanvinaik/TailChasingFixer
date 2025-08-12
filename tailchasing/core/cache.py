"""
Cache management for avoiding re-analysis of unchanged files.

This module provides persistent caching of analysis results to speed up
subsequent runs by skipping unchanged files.
"""

import ast
import gzip
import hashlib
import json
import logging
import os
import pickle
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple
import time

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CachedFileInfo:
    """Information about a cached file."""
    file_path: str
    mtime: float
    size: int
    content_hash: str
    ast_hash: Optional[str] = None
    
    # Analysis results
    issues: List[Dict[str, Any]] = None
    symbols: Dict[str, Any] = None
    
    # Duplicate detection caches
    content_signature: Optional[str] = None
    shingle_hashes: Optional[List[int]] = None
    minhash_signature: Optional[List[float]] = None
    ast_features: Optional[List[str]] = None
    
    # Semantic analysis caches
    hypervector: Optional[List[float]] = None
    semantic_features: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_file(cls, file_path: str) -> "CachedFileInfo":
        """Create CachedFileInfo from a file path."""
        path = Path(file_path)
        stat = path.stat()
        
        # Compute content hash
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while chunk := f.read(8192):
                hasher.update(chunk)
        
        return cls(
            file_path=str(path.absolute()),
            mtime=stat.st_mtime,
            size=stat.st_size,
            content_hash=hasher.hexdigest()
        )
    
    def is_valid_for(self, file_path: str) -> bool:
        """Check if this cache entry is valid for the given file."""
        try:
            path = Path(file_path)
            stat = path.stat()
            
            # Quick check: mtime and size
            if stat.st_mtime != self.mtime or stat.st_size != self.size:
                return False
            
            # If mtime/size match, verify content hash for safety
            hasher = hashlib.sha256()
            with open(file_path, 'rb') as f:
                while chunk := f.read(8192):
                    hasher.update(chunk)
            
            return hasher.hexdigest() == self.content_hash
            
        except (OSError, IOError):
            return False


class CacheManager:
    """
    Manages persistent cache for analysis results.
    
    The cache stores:
    - File metadata (mtime, size, hash)
    - AST trees
    - Analysis results
    - Computed signatures and features
    """
    
    CACHE_DIR_NAME = ".tailchasing_cache"
    CACHE_VERSION = "1.0"
    
    def __init__(self, root_dir: Path, enabled: bool = True):
        """
        Initialize cache manager.
        
        Args:
            root_dir: Root directory of the project
            enabled: Whether caching is enabled
        """
        self.root_dir = Path(root_dir)
        self.enabled = enabled
        self.cache_dir = self.root_dir / self.CACHE_DIR_NAME
        
        # In-memory cache
        self.file_cache: Dict[str, CachedFileInfo] = {}
        self.dirty_files: Set[str] = set()
        
        # Statistics
        self.stats = {
            'hits': 0,
            'misses': 0,
            'saves': 0,
            'loads': 0,
            'bytes_saved': 0
        }
        
        if self.enabled:
            self._initialize_cache()
    
    def _initialize_cache(self) -> None:
        """Initialize cache directory and load existing cache."""
        # Create cache directory if needed
        self.cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Add .gitignore to cache directory
        gitignore = self.cache_dir / ".gitignore"
        if not gitignore.exists():
            gitignore.write_text("*\n")
        
        # Check cache version
        version_file = self.cache_dir / "version"
        if version_file.exists():
            stored_version = version_file.read_text().strip()
            if stored_version != self.CACHE_VERSION:
                logger.info(f"Cache version mismatch ({stored_version} != {self.CACHE_VERSION}), clearing cache")
                self.clear_cache()
        
        # Write current version
        version_file.write_text(self.CACHE_VERSION)
        
        # Load index
        self._load_index()
    
    def _get_cache_path(self, file_path: str, suffix: str = ".json.gz") -> Path:
        """Get cache file path for a given source file."""
        # Use hash of absolute path to avoid collisions
        path_hash = hashlib.md5(file_path.encode()).hexdigest()[:16]
        
        # Create subdirectory based on first 2 chars of hash for better file system performance
        subdir = self.cache_dir / path_hash[:2]
        subdir.mkdir(exist_ok=True)
        
        return subdir / f"{path_hash}{suffix}"
    
    def _load_index(self) -> None:
        """Load the cache index."""
        index_path = self.cache_dir / "index.json.gz"
        
        if not index_path.exists():
            return
        
        try:
            with gzip.open(index_path, 'rt', encoding='utf-8') as f:
                index_data = json.load(f)
            
            # Load file cache entries
            for file_path, cache_data in index_data.get('files', {}).items():
                self.file_cache[file_path] = CachedFileInfo(**cache_data)
            
            self.stats['loads'] = len(self.file_cache)
            logger.info(f"Loaded cache index with {len(self.file_cache)} entries")
            
        except Exception as e:
            logger.warning(f"Failed to load cache index: {e}")
    
    def _save_index(self) -> None:
        """Save the cache index."""
        if not self.enabled:
            return
        
        index_path = self.cache_dir / "index.json.gz"
        
        try:
            # Prepare index data
            index_data = {
                'version': self.CACHE_VERSION,
                'timestamp': time.time(),
                'files': {}
            }
            
            # Convert CachedFileInfo objects to dicts
            for file_path, info in self.file_cache.items():
                # Only save basic metadata in index
                index_entry = {
                    'file_path': info.file_path,
                    'mtime': info.mtime,
                    'size': info.size,
                    'content_hash': info.content_hash,
                    'ast_hash': info.ast_hash
                }
                index_data['files'][file_path] = index_entry
            
            # Save compressed
            with gzip.open(index_path, 'wt', encoding='utf-8') as f:
                json.dump(index_data, f, indent=2)
            
            self.stats['saves'] += 1
            logger.debug(f"Saved cache index with {len(self.file_cache)} entries")
            
        except Exception as e:
            logger.error(f"Failed to save cache index: {e}")
    
    def get_cached_ast(self, file_path: str) -> Optional[ast.AST]:
        """
        Get cached AST for a file if available and valid.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Cached AST or None if not cached/invalid
        """
        if not self.enabled:
            return None
        
        # Check if we have valid cache entry
        cache_info = self.file_cache.get(file_path)
        if not cache_info or not cache_info.is_valid_for(file_path):
            self.stats['misses'] += 1
            return None
        
        # Load AST from cache file
        ast_cache_path = self._get_cache_path(file_path, ".ast.pkl.gz")
        
        if not ast_cache_path.exists():
            self.stats['misses'] += 1
            return None
        
        try:
            with gzip.open(ast_cache_path, 'rb') as f:
                cached_ast = pickle.load(f)
            
            self.stats['hits'] += 1
            self.stats['bytes_saved'] += Path(file_path).stat().st_size
            logger.debug(f"Cache hit for AST: {file_path}")
            return cached_ast
            
        except Exception as e:
            logger.debug(f"Failed to load cached AST for {file_path}: {e}")
            self.stats['misses'] += 1
            return None
    
    def cache_ast(self, file_path: str, ast_tree: ast.AST) -> None:
        """
        Cache an AST for a file.
        
        Args:
            file_path: Path to the Python file
            ast_tree: The AST to cache
        """
        if not self.enabled:
            return
        
        # Create or update cache info
        if file_path not in self.file_cache:
            self.file_cache[file_path] = CachedFileInfo.from_file(file_path)
        
        cache_info = self.file_cache[file_path]
        
        # Compute AST hash
        ast_dump = ast.dump(ast_tree, annotate_fields=False)
        cache_info.ast_hash = hashlib.md5(ast_dump.encode()).hexdigest()
        
        # Save AST to cache file
        ast_cache_path = self._get_cache_path(file_path, ".ast.pkl.gz")
        
        try:
            with gzip.open(ast_cache_path, 'wb', compresslevel=6) as f:
                pickle.dump(ast_tree, f)
            
            self.dirty_files.add(file_path)
            logger.debug(f"Cached AST for {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to cache AST for {file_path}: {e}")
    
    def get_cached_analysis(self, file_path: str, analyzer_name: str) -> Optional[Any]:
        """
        Get cached analysis results for a file and analyzer.
        
        Args:
            file_path: Path to the file
            analyzer_name: Name of the analyzer
            
        Returns:
            Cached analysis results or None
        """
        if not self.enabled:
            return None
        
        cache_info = self.file_cache.get(file_path)
        if not cache_info or not cache_info.is_valid_for(file_path):
            self.stats['misses'] += 1
            return None
        
        # Load analysis cache
        analysis_cache_path = self._get_cache_path(file_path, f".{analyzer_name}.json.gz")
        
        if not analysis_cache_path.exists():
            self.stats['misses'] += 1
            return None
        
        try:
            with gzip.open(analysis_cache_path, 'rt', encoding='utf-8') as f:
                cached_data = json.load(f)
            
            self.stats['hits'] += 1
            logger.debug(f"Cache hit for {analyzer_name}: {file_path}")
            return cached_data
            
        except Exception as e:
            logger.debug(f"Failed to load cached {analyzer_name} for {file_path}: {e}")
            self.stats['misses'] += 1
            return None
    
    def cache_analysis(self, file_path: str, analyzer_name: str, data: Any) -> None:
        """
        Cache analysis results for a file and analyzer.
        
        Args:
            file_path: Path to the file
            analyzer_name: Name of the analyzer
            data: Analysis results to cache
        """
        if not self.enabled:
            return
        
        # Create or update cache info
        if file_path not in self.file_cache:
            self.file_cache[file_path] = CachedFileInfo.from_file(file_path)
        
        # Save analysis cache
        analysis_cache_path = self._get_cache_path(file_path, f".{analyzer_name}.json.gz")
        
        try:
            # Convert numpy arrays to lists for JSON serialization
            if isinstance(data, np.ndarray):
                data = data.tolist()
            elif isinstance(data, dict):
                data = self._convert_numpy_to_list(data)
            
            with gzip.open(analysis_cache_path, 'wt', encoding='utf-8') as f:
                json.dump(data, f)
            
            self.dirty_files.add(file_path)
            logger.debug(f"Cached {analyzer_name} for {file_path}")
            
        except Exception as e:
            logger.error(f"Failed to cache {analyzer_name} for {file_path}: {e}")
    
    def _convert_numpy_to_list(self, data: Any) -> Any:
        """Recursively convert numpy arrays to lists for JSON serialization."""
        if isinstance(data, np.ndarray):
            return data.tolist()
        elif isinstance(data, (np.integer, np.floating)):
            # Handle numpy scalar types
            return data.item()
        elif isinstance(data, dict):
            return {k: self._convert_numpy_to_list(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_numpy_to_list(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(self._convert_numpy_to_list(item) for item in data)
        else:
            return data
    
    def get_cached_signatures(self, file_path: str) -> Optional[Dict[str, Any]]:
        """
        Get cached signatures for duplicate detection.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary of cached signatures or None
        """
        return self.get_cached_analysis(file_path, "signatures")
    
    def cache_signatures(self, file_path: str, signatures: Dict[str, Any]) -> None:
        """
        Cache signatures for duplicate detection.
        
        Args:
            file_path: Path to the file
            signatures: Dictionary of signatures to cache
        """
        self.cache_analysis(file_path, "signatures", signatures)
    
    def get_cached_hypervector(self, file_path: str) -> Optional[np.ndarray]:
        """
        Get cached hypervector for semantic analysis.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Cached hypervector or None
        """
        cached = self.get_cached_analysis(file_path, "hypervector")
        if cached is not None:
            return np.array(cached, dtype=np.float32)
        return None
    
    def cache_hypervector(self, file_path: str, hypervector: np.ndarray) -> None:
        """
        Cache hypervector for semantic analysis.
        
        Args:
            file_path: Path to the file
            hypervector: The hypervector to cache
        """
        self.cache_analysis(file_path, "hypervector", hypervector)
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        if self.cache_dir.exists():
            try:
                shutil.rmtree(self.cache_dir)
                logger.info(f"Cleared cache directory: {self.cache_dir}")
            except Exception as e:
                logger.error(f"Failed to clear cache: {e}")
        
        # Reset in-memory cache
        self.file_cache.clear()
        self.dirty_files.clear()
        
        # Reinitialize if enabled
        if self.enabled:
            self._initialize_cache()
    
    def clear_file_cache(self, file_path: str) -> None:
        """Clear cache for a specific file."""
        # Remove from in-memory cache
        if file_path in self.file_cache:
            del self.file_cache[file_path]
        
        # Remove cache files
        for suffix in [".ast.pkl.gz", ".json.gz", ".signatures.json.gz", ".hypervector.json.gz"]:
            cache_path = self._get_cache_path(file_path, suffix)
            if cache_path.exists():
                try:
                    cache_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to remove cache file {cache_path}: {e}")
    
    def flush(self) -> None:
        """Flush any pending cache writes."""
        if self.dirty_files:
            self._save_index()
            self.dirty_files.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        stats = self.stats.copy()
        
        if self.stats['hits'] + self.stats['misses'] > 0:
            stats['hit_rate'] = self.stats['hits'] / (self.stats['hits'] + self.stats['misses'])
        else:
            stats['hit_rate'] = 0.0
        
        # Calculate cache size
        if self.cache_dir.exists():
            cache_size = sum(f.stat().st_size for f in self.cache_dir.rglob('*') if f.is_file())
            stats['cache_size_mb'] = cache_size / (1024 * 1024)
        else:
            stats['cache_size_mb'] = 0.0
        
        stats['cached_files'] = len(self.file_cache)
        stats['bytes_saved_mb'] = self.stats['bytes_saved'] / (1024 * 1024)
        
        return stats
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - flush cache."""
        self.flush()


def create_cache_manager(root_dir: Path, config: Dict[str, Any]) -> CacheManager:
    """
    Factory function to create a cache manager.
    
    Args:
        root_dir: Root directory of the project
        config: Configuration dictionary
        
    Returns:
        Configured CacheManager instance
    """
    cache_config = config.get('cache', {})
    enabled = cache_config.get('enabled', True)
    
    return CacheManager(root_dir, enabled=enabled)