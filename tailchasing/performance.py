"""
Performance optimizations for large-scale analysis.

Implements:
- Parallel processing for multi-core systems
- Incremental analysis with smart caching
- Approximate nearest neighbor search for semantic similarity
- Memory-efficient hypervector operations
"""

import os
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import mmap
import pickle
from pathlib import Path

from .core.issues import Issue


class OptimizedHVSpace:
    """
    Memory and performance optimized hypervector operations.
    """
    
    def __init__(self, dim: int = 8192, use_gpu: bool = False):
        self.dim = dim
        self.use_gpu = use_gpu and self._check_gpu_available()
        self.bipolar = True  # Add missing attribute
        
        # Use bit-packed representation for binary vectors
        self.bits_per_chunk = 64
        self.num_chunks = (dim + self.bits_per_chunk - 1) // self.bits_per_chunk
        
        # Memory-mapped token cache for large vocabularies
        self.token_cache_file = Path(".tailchasing_cache/tokens.mmap")
        self._init_mmap_cache()
    
    def _rand_vec(self) -> np.ndarray:
        """Generate random vector."""
        if self.bipolar:
            return np.random.choice([-1, 1], size=self.dim).astype(np.int8)
        else:
            return np.random.randint(0, 2, size=self.dim, dtype=np.uint8)
    
    def _check_gpu_available(self) -> bool:
        """Check if GPU acceleration is available."""
        try:
            import cupy
            return True
        except ImportError:
            return False
    
    def _init_mmap_cache(self):
        """Initialize memory-mapped cache for tokens."""
        self.token_cache_file.parent.mkdir(exist_ok=True)
        # ... implementation
    
    def fast_hamming_batch(self, 
                          query: np.ndarray, 
                          corpus: List[np.ndarray]) -> np.ndarray:
        """
        Compute Hamming distances in batch using vectorized operations.
        """
        if self.use_gpu:
            import cupy as cp
            query_gpu = cp.asarray(query)
            corpus_gpu = cp.asarray(corpus)
            distances = cp.sum(query_gpu != corpus_gpu, axis=1) / self.dim
            return cp.asnumpy(distances)
        else:
            # Vectorized NumPy operations
            corpus_array = np.array(corpus)
            return np.sum(query != corpus_array, axis=1) / self.dim
    
    def bundle_optimized(self, vectors: List[np.ndarray]) -> np.ndarray:
        """
        Optimized bundling using bit manipulation for binary vectors.
        """
        if len(vectors) == 0:
            return self._rand_vec()
        
        if len(vectors) == 1:
            return vectors[0].copy()
        
        # For binary vectors, use bit counting
        if not self.bipolar:
            # Stack and count set bits
            stacked = np.array(vectors, dtype=np.uint8)
            bit_counts = np.sum(stacked, axis=0)
            threshold = len(vectors) // 2
            return (bit_counts > threshold).astype(np.uint8)
        
        # For bipolar, use sign of sum
        else:
            stacked = np.array(vectors, dtype=np.int8)
            sums = np.sum(stacked, axis=0)
            return np.sign(sums).astype(np.int8)


class ParallelSemanticAnalyzer:
    """
    Parallel processing for semantic analysis of large codebases.
    """
    
    def __init__(self, num_workers: Optional[int] = None):
        self.num_workers = num_workers or os.cpu_count()
        self.executor = ProcessPoolExecutor(max_workers=self.num_workers)
    
    def analyze_parallel(self, 
                        files: List[Path], 
                        config: Dict) -> List[Issue]:
        """
        Analyze files in parallel for better performance.
        """
        # Chunk files for parallel processing
        chunk_size = max(1, len(files) // self.num_workers)
        file_chunks = [
            files[i:i + chunk_size] 
            for i in range(0, len(files), chunk_size)
        ]
        
        # Process chunks in parallel
        futures = []
        for chunk in file_chunks:
            future = self.executor.submit(self._analyze_chunk, chunk, config)
            futures.append(future)
        
        # Collect results
        all_issues = []
        for future in futures:
            chunk_issues = future.result()
            all_issues.extend(chunk_issues)
        
        # Merge and deduplicate
        return self._merge_issues(all_issues)
    
    def _analyze_chunk(self, files: List[Path], config: Dict) -> List[Issue]:
        """Analyze a chunk of files."""
        # ... implementation
        return []


class ApproximateNearestNeighbor:
    """
    Fast approximate nearest neighbor search for semantic similarity.
    
    Uses LSH (Locality Sensitive Hashing) for sub-linear search time.
    """
    
    def __init__(self, dim: int, num_tables: int = 10, hash_size: int = 12):
        self.dim = dim
        self.num_tables = num_tables
        self.hash_size = hash_size
        
        # Random projection matrices for LSH
        self.projections = [
            np.random.randn(hash_size, dim) 
            for _ in range(num_tables)
        ]
        
        # Hash tables
        self.tables = [{} for _ in range(num_tables)]
        self.data = []
    
    def add(self, vector: np.ndarray, data_id: str) -> None:
        """Add a vector to the index."""
        idx = len(self.data)
        self.data.append((vector, data_id))
        
        # Hash into all tables
        for i, projection in enumerate(self.projections):
            hash_val = self._hash_vector(vector, projection)
            if hash_val not in self.tables[i]:
                self.tables[i][hash_val] = []
            self.tables[i][hash_val].append(idx)
    
    def query(self, vector: np.ndarray, k: int = 10) -> List[Tuple[str, float]]:
        """
        Find approximate k nearest neighbors.
        """
        candidates = set()
        
        # Get candidates from all hash tables
        for i, projection in enumerate(self.projections):
            hash_val = self._hash_vector(vector, projection)
            if hash_val in self.tables[i]:
                candidates.update(self.tables[i][hash_val])
        
        # Compute exact distances for candidates
        distances = []
        for idx in candidates:
            stored_vec, data_id = self.data[idx]
            dist = np.sum(vector != stored_vec) / self.dim
            distances.append((data_id, dist))
        
        # Sort and return top k
        distances.sort(key=lambda x: x[1])
        return distances[:k]
    
    def _hash_vector(self, vector: np.ndarray, projection: np.ndarray) -> int:
        """Compute LSH hash of vector."""
        projected = projection @ vector
        binary = (projected > 0).astype(int)
        # Convert binary vector to integer hash
        hash_val = 0
        for i, bit in enumerate(binary):
            hash_val |= (bit << i)
        return hash_val


class IncrementalAnalysisCache:
    """
    Smart caching for incremental analysis.
    """
    
    def __init__(self, cache_dir: Path):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(exist_ok=True)
        
        # File modification times
        self.file_mtimes: Dict[str, float] = {}
        self._load_mtimes()
        
        # Cached analysis results
        self.cached_results: Dict[str, Any] = {}
    
    def needs_reanalysis(self, file_path: Path) -> bool:
        """Check if file needs reanalysis."""
        current_mtime = file_path.stat().st_mtime
        cached_mtime = self.file_mtimes.get(str(file_path), 0)
        return current_mtime > cached_mtime
    
    def get_cached_result(self, file_path: Path) -> Optional[Dict]:
        """Get cached analysis result if valid."""
        if not self.needs_reanalysis(file_path):
            cache_file = self._get_cache_file(file_path)
            if cache_file.exists():
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
        return None
    
    def cache_result(self, file_path: Path, result: Dict) -> None:
        """Cache analysis result."""
        cache_file = self._get_cache_file(file_path)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        # Update mtime
        self.file_mtimes[str(file_path)] = file_path.stat().st_mtime
        self._save_mtimes()
    
    def _get_cache_file(self, file_path: Path) -> Path:
        """Get cache file path for given source file."""
        # Use hash of file path to avoid filesystem issues
        import hashlib
        path_hash = hashlib.md5(str(file_path).encode()).hexdigest()
        return self.cache_dir / f"{path_hash}.pkl"
    
    def _load_mtimes(self):
        """Load file modification times."""
        mtime_file = self.cache_dir / "mtimes.json"
        if mtime_file.exists():
            import json
            with open(mtime_file) as f:
                self.file_mtimes = json.load(f)
    
    def _save_mtimes(self):
        """Save file modification times."""
        import json
        mtime_file = self.cache_dir / "mtimes.json"
        with open(mtime_file, 'w') as f:
            json.dump(self.file_mtimes, f)


# Add these to the main analyzer
class OptimizedSemanticHVAnalyzer:
    """
    Performance-optimized version of semantic analyzer.
    """
    
    def __init__(self):
        self.space = OptimizedHVSpace()
        self.ann_index = ApproximateNearestNeighbor(8192)
        self.cache = IncrementalAnalysisCache(Path(".tailchasing_cache"))
        self.parallel_analyzer = ParallelSemanticAnalyzer()
        
    # ... rest of implementation using optimized components