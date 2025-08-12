"""
Semantic index for hypervector storage and similarity search.

Manages the collection of function hypervectors, computes background
statistics, and identifies significantly similar pairs.
"""

import json
import pickle
import random
import math
import logging
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional, Set, Any, TypedDict, cast
from collections import defaultdict
import numpy as np
from numpy.typing import NDArray

from .hv_space import HVSpace


class FunctionEntry(TypedDict, total=False):
    """Type definition for function metadata."""
    removed: bool
    name: str
    file: str
    line: int
    complexity: float
    tokens: List[str]


class SimilarityAnalysis(TypedDict):
    """Type definition for similarity analysis results."""
    files_same: bool
    names_similar: bool


class IndexStats(TypedDict):
    """Type definition for index statistics."""
    total_functions: int
    space_stats: Dict[str, Any]
    background_stats: Dict[str, Optional[float]]


class SemanticIndex:
    """
    Index for storing and querying function hypervectors.
    
    Provides similarity search with statistical significance testing.
    """
    
    def __init__(self, config: Dict[str, Any], cache_dir: Optional[Path] = None) -> None:
        """
        Initialize enhanced semantic index with 16384-dimensional vectors.
        
        Args:
            config: Semantic configuration
            cache_dir: Directory for persisting index
        """
        self.config = config
        self.cache_dir = cache_dir
        self.logger = logging.getLogger(__name__)
        
        # Resource limits
        self.max_duplicate_pairs = config.get('resource_limits', {}).get('max_duplicate_pairs', 200000)
        self.lsh_bucket_cap = config.get('resource_limits', {}).get('lsh_bucket_cap', 2000)
        
        # Initialize hypervector space with enhanced dimensions
        self.space = HVSpace(
            dim=config.get('hv_dim', 16384),  # Enhanced to 16384 dimensions
            bipolar=config.get('bipolar', True),
            seed=config.get('random_seed', 42)
        )
        
        # Function entries: (id, hv, metadata)
        self.entries: List[Tuple[str, Optional[NDArray[np.float32]], FunctionEntry]] = []
        
        # ID to entry index mapping
        self.id_to_index: Dict[str, int] = {}
        
        # Background statistics cache
        self._background_stats: Optional[Tuple[float, float]] = None
        self._stats_sample_size = 0
        
        # Enhanced indexing structures for efficient similarity search
        self._similarity_cache: Dict[Tuple[str, str], Any] = {}
        self._vector_matrix: Optional[NDArray[np.float32]] = None
        self._matrix_valid = False
        self._incremental_updates: List[Tuple[str, str, NDArray[np.float32], FunctionEntry]] = []
        
        # Performance monitoring
        self._last_rebuild_time = 0.0
        self._search_stats = {
            'cache_hits': 0,
            'cache_misses': 0,
            'matrix_rebuilds': 0,
            'incremental_updates': 0
        }
        
        # Load cache if available
        if cache_dir:
            self._load_cache()
    
    def add(self, func_id: str, file: str, line: int, hv: NDArray[np.float32],
            metadata: Optional[FunctionEntry] = None) -> None:
        """Add a function hypervector to the index with incremental updates."""
        full_id = f"{func_id}@{file}:{line}"
        
        # Store incremental update for batch processing
        entry_meta = metadata or cast(FunctionEntry, {})
        self._incremental_updates.append(("add", full_id, hv, entry_meta))
        
        # Process incremental updates if batch is large enough
        if len(self._incremental_updates) >= self.config.get('batch_size', 100):
            self._process_incremental_updates()
        
        # Invalidate caches
        self._invalidate_caches()
    
    def remove(self, func_id: str, file: str, line: int) -> bool:
        """Remove a function from the index with incremental updates."""
        full_id = f"{func_id}@{file}:{line}"
        
        if full_id not in self.id_to_index and not any(update[1] == full_id for update in self._incremental_updates):
            return False
        
        # Store incremental update for batch processing
        removed_meta: FunctionEntry = {"removed": True}
        self._incremental_updates.append(("remove", full_id, np.zeros(self.space.dim, dtype=np.float32), removed_meta))
        
        # Process incremental updates if batch is large enough
        if len(self._incremental_updates) >= self.config.get('batch_size', 100):
            self._process_incremental_updates()
        
        # Invalidate caches
        self._invalidate_caches()
        return True
    
    def _compute_background_stats(self) -> Tuple[float, float]:
        """
        Compute background distance distribution.
        
        Returns (mean, std) of random pair distances.
        """
        valid_entries: List[Tuple[str, NDArray[np.float32], FunctionEntry]] = [
            (id, hv, meta) for id, hv, meta in self.entries
            if hv is not None and not meta.get("removed", False)
        ]
        
        n = len(valid_entries)
        if n < 2:
            return (0.5, 0.05)  # Default for small samples
        
        # Sample random pairs more efficiently
        max_pairs = min(
            self.config.get('max_pairs_sample', 1000),  # Reduced default
            n * (n - 1) // 2
        )
        
        # For large sets, use random sampling instead of reservoir sampling
        distances: List[float] = []
        
        if n * (n - 1) // 2 <= max_pairs:
            # Small enough to compute all pairs
            for i in range(n):
                for j in range(i + 1, n):
                    d = self.space.distance(valid_entries[i][1], valid_entries[j][1])
                    distances.append(d)
        else:
            # Random sampling for large sets
            sampled = 0
            attempts = 0
            max_attempts = max_pairs * 3  # Prevent infinite loops
            seen_pairs = set()
            
            while sampled < max_pairs and attempts < max_attempts:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                attempts += 1
                
                if i != j and (i, j) not in seen_pairs and (j, i) not in seen_pairs:
                    seen_pairs.add((i, j))
                    d = self.space.distance(valid_entries[i][1], valid_entries[j][1])
                    distances.append(d)
                    sampled += 1
        
        if not distances:
            return (0.5, 0.05)
        
        # Compute statistics
        mean = sum(distances) / len(distances)
        variance = sum((d - mean) ** 2 for d in distances) / len(distances)
        std = math.sqrt(variance + 1e-12)  # Add small epsilon
        
        self._stats_sample_size = len(distances)
        return (mean, std)
    
    def _ensure_background_stats(self) -> None:
        """Ensure background statistics are computed."""
        if self._background_stats is None:
            self._background_stats = self._compute_background_stats()
    
    def get_background_stats(self) -> Tuple[float, float]:
        """Get background distance statistics."""
        self._ensure_background_stats()
        assert self._background_stats is not None
        return self._background_stats
    
    def compute_z_score(self, distance: float) -> float:
        """
        Compute z-score for a distance.
        
        Higher z-score indicates more significant similarity.
        """
        mean, std = self.get_background_stats()
        if std == 0:
            return 0.0
        
        # Lower distance = higher similarity = higher z-score
        return (mean - distance) / std
    
    def find_similar(self, hv: NDArray[np.float32], 
                    z_threshold: Optional[float] = None,
                    limit: int = 10) -> List[Tuple[str, float, float, FunctionEntry]]:
        """
        Find functions similar to given hypervector using efficient search.
        
        Returns list of (id, distance, z_score, metadata) tuples.
        """
        if z_threshold is None:
            z_threshold = self.config.get('z_threshold', 2.5)
        
        # Process any pending incremental updates
        if self._incremental_updates:
            self._process_incremental_updates()
        
        # Use efficient vectorized search
        return self._find_similar_vectorized(hv, z_threshold, limit)
    
    def find_all_similar_pairs(self, 
                              z_threshold: Optional[float] = None,
                              limit: Optional[int] = None) -> List[Tuple[str, str, float, float, SimilarityAnalysis]]:
        """
        Find all pairs of similar functions using efficient vectorized computation.
        
        Returns list of (id1, id2, distance, z_score, analysis) tuples.
        """
        if z_threshold is None:
            z_threshold = self.config.get('z_threshold', 2.5)
        
        # Process any pending incremental updates
        if self._incremental_updates:
            self._process_incremental_updates()
        
        # Use efficient vectorized search
        return self._find_all_similar_pairs_vectorized(z_threshold, limit)
    
    def _analyze_similarity(self, hv1: NDArray[np.float32], hv2: NDArray[np.float32],
                           id1: str, id2: str) -> SimilarityAnalysis:
        """
        Analyze what contributes to similarity between two functions.
        
        This is approximate since we can't perfectly decompose bound vectors.
        """
        analysis: SimilarityAnalysis = {
            "files_same": id1.split('@')[1].split(':')[0] == id2.split('@')[1].split(':')[0],
            "names_similar": self._name_similarity(id1, id2) > 0.5,
        }
        
        # Implement enhanced channel contribution analysis
        # Use approximation techniques with channel prototypes
        
        # Add channel contribution analysis
        channel_contributions = {}
        
        # Define channel prototypes based on common patterns
        channels = [
            'DATA_FLOW', 'CONTROL_FLOW', 'RETURN_PATTERNS', 
            'ERROR_HANDLING', 'LOOP_STRUCTURES', 'TYPE_PATTERNS', 'NAME_TOKENS'
        ]
        
        for channel in channels:
            try:
                # Get or create prototype for this channel
                prototype = self.space.role(f"CHANNEL_{channel}")
                
                # Measure similarity to prototype for both vectors
                sim1 = self.space.similarity(hv1, prototype)
                sim2 = self.space.similarity(hv2, prototype)
                
                # Channel contribution is the product of similarities
                channel_contributions[channel.lower()] = float(sim1 * sim2)
            except Exception:
                # Skip if prototype generation fails
                channel_contributions[channel.lower()] = 0.0
        
        # Add to analysis dictionary
        analysis['channel_contributions'] = channel_contributions  # type: ignore
        
        return analysis
    
    def _name_similarity(self, id1: str, id2: str) -> float:
        """Compute name similarity between two function IDs."""
        name1 = id1.split('@')[0]
        name2 = id2.split('@')[0]
        
        if name1 == name2:
            return 1.0
        
        # Simple Jaccard similarity on name tokens
        from ..semantic.encoder import split_identifier
        tokens1 = set(split_identifier(name1))
        tokens2 = set(split_identifier(name2))
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2
        
        return len(intersection) / len(union)
    
    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        valid_count = sum(1 for _, hv, meta in self.entries
                         if hv is not None and not meta.get("removed", False))
        
        stats: IndexStats = {
            "total_functions": valid_count,
            "space_stats": self.space.get_stats(),
            "background_stats": {
                "mean": self._background_stats[0] if self._background_stats else None,
                "std": self._background_stats[1] if self._background_stats else None,
                "sample_size": float(self._stats_sample_size)
            }
        }
        
        return stats
    
    def save_cache(self) -> None:
        """Save enhanced index to cache directory."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "semantic_index_enhanced.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Process any pending updates before saving
        if hasattr(self, '_incremental_updates') and self._incremental_updates:
            self._process_incremental_updates()
        
        cache_data = {
            "entries": self.entries,
            "id_to_index": self.id_to_index,
            "background_stats": self._background_stats,
            "stats_sample_size": self._stats_sample_size,
            "space_token_cache": self.space._token_cache,
            "space_role_cache": self.space._role_cache,
            "similarity_cache": getattr(self, '_similarity_cache', {}),
            "search_stats": getattr(self, '_search_stats', {}),
            "last_rebuild_time": getattr(self, '_last_rebuild_time', 0.0),
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_cache(self) -> bool:
        """Load index from cache directory."""
        if not self.cache_dir:
            return False
        
        cache_file = self.cache_dir / "semantic_index_enhanced.pkl"
        if not cache_file.exists():
            # Try legacy cache file
            legacy_file = self.cache_dir / "semantic_index.pkl"
            if legacy_file.exists():
                return self._load_legacy_cache(legacy_file)
            return False
        
        try:
            with open(cache_file, 'rb') as f:
                cache_data = pickle.load(f)
            
            self.entries = cache_data.get("entries", [])
            self.id_to_index = cache_data.get("id_to_index", {})
            self._background_stats = cache_data.get("background_stats")
            self._stats_sample_size = cache_data.get("stats_sample_size", 0)
            
            # Restore hypervector caches
            if "space_token_cache" in cache_data:
                self.space._token_cache = cache_data["space_token_cache"]
            if "space_role_cache" in cache_data:
                self.space._role_cache = cache_data["space_role_cache"]
            
            return True
            
        except Exception:
            # Cache corrupted or incompatible
            return False
    
    def _process_incremental_updates(self) -> None:
        """Process batched incremental updates efficiently."""
        if not self._incremental_updates:
            return
        
        start_time = time.time()
        updates_processed = 0
        
        for action, full_id, hv, entry_meta in self._incremental_updates:
            if action == "add":
                if full_id in self.id_to_index:
                    # Update existing entry
                    idx = self.id_to_index[full_id]
                    self.entries[idx] = (full_id, hv, entry_meta)
                else:
                    # Add new entry
                    idx = len(self.entries)
                    self.entries.append((full_id, hv, entry_meta))
                    self.id_to_index[full_id] = idx
                updates_processed += 1
                
            elif action == "remove":
                if full_id in self.id_to_index:
                    # Mark as removed
                    idx = self.id_to_index[full_id]
                    self.entries[idx] = (full_id, None, entry_meta)
                    del self.id_to_index[full_id]
                    updates_processed += 1
        
        # Clear processed updates
        self._incremental_updates.clear()
        
        # Update statistics
        self._search_stats['incremental_updates'] += updates_processed
        
        # Rebuild vector matrix if significant changes
        if updates_processed > self.config.get('rebuild_threshold', 50):
            self._rebuild_vector_matrix()
        
        process_time = time.time() - start_time
        self.logger.debug(f"Processed {updates_processed} incremental updates in {process_time:.3f}s")
    
    def _invalidate_caches(self) -> None:
        """Invalidate caches after modifications."""
        self._background_stats = None
        self._matrix_valid = False
        # Keep similarity cache but mark it as potentially stale
        # Full invalidation would be too expensive
    
    def _rebuild_vector_matrix(self) -> None:
        """Rebuild vector matrix for efficient vectorized operations."""
        valid_entries = [(id, hv, meta) for id, hv, meta in self.entries
                        if hv is not None and not meta.get("removed", False)]
        
        if not valid_entries:
            self._vector_matrix = None
            self._matrix_valid = False
            return
        
        # Stack vectors into matrix and convert to float32 for efficient computation
        vectors = [hv for _, hv, _ in valid_entries]
        self._vector_matrix = np.stack(vectors).astype(np.float32)
        self._matrix_valid = True
        self._last_rebuild_time = time.time()
        self._search_stats['matrix_rebuilds'] += 1
        
        self.logger.debug(f"Rebuilt vector matrix with {len(valid_entries)} entries")
    
    def _find_similar_vectorized(self, query_hv: NDArray[np.float32], 
                               z_threshold: float, limit: int) -> List[Tuple[str, float, float, FunctionEntry]]:
        """Efficient vectorized similarity search."""
        if not self._matrix_valid or self._vector_matrix is None:
            self._rebuild_vector_matrix()
            
        if self._vector_matrix is None:
            return []
        
        # Get valid entries for metadata lookup
        valid_entries = [(id, hv, meta) for id, hv, meta in self.entries
                        if hv is not None and not meta.get("removed", False)]
        
        # Compute distances using vectorized operations
        distances = self._compute_distances_vectorized(query_hv, self._vector_matrix)
        
        # Compute z-scores
        mean, std = self.get_background_stats()
        if std == 0:
            z_scores = np.zeros_like(distances)
        else:
            z_scores = (mean - distances) / std
        
        # Filter by threshold and get top results
        valid_indices = np.where(z_scores >= z_threshold)[0]
        
        if len(valid_indices) == 0:
            return []
        
        # Sort by z-score (descending)
        sorted_indices = valid_indices[np.argsort(-z_scores[valid_indices])]
        
        # Build results
        results = []
        for idx in sorted_indices[:limit]:
            entry_id, _, entry_meta = valid_entries[idx]
            results.append((entry_id, distances[idx], z_scores[idx], entry_meta))
        
        return results
    
    def _find_all_similar_pairs_vectorized(self, z_threshold: float, 
                                         limit: Optional[int]) -> List[Tuple[str, str, float, float, SimilarityAnalysis]]:
        """Efficient vectorized computation of all similar pairs."""
        if not self._matrix_valid or self._vector_matrix is None:
            self._rebuild_vector_matrix()
            
        if self._vector_matrix is None:
            return []
        
        # Get valid entries for metadata lookup
        valid_entries = [(id, hv, meta) for id, hv, meta in self.entries
                        if hv is not None and not meta.get("removed", False)]
        
        n = len(valid_entries)
        if n < 2:
            return []
        
        # Compute all pairwise distances efficiently
        distances = self._compute_pairwise_distances_vectorized(self._vector_matrix)
        
        # Compute z-scores
        mean, std = self.get_background_stats()
        if std == 0:
            z_scores = np.zeros_like(distances)
        else:
            z_scores = (mean - distances) / std
        
        # Find pairs above threshold
        pairs = []
        pair_count = 0
        max_pairs = min(self.max_duplicate_pairs, limit) if limit else self.max_duplicate_pairs
        
        # Log warning if we have many potential pairs
        total_pairs = n * (n - 1) // 2
        if total_pairs > max_pairs * 2:
            self.logger.warning(
                f"Large number of potential pairs ({total_pairs}), limiting to {max_pairs} pairs. "
                f"Consider increasing max_duplicate_pairs or using stricter thresholds."
            )
        
        for i in range(n):
            for j in range(i + 1, n):
                if z_scores[i, j] >= z_threshold:
                    # Check if we've reached the limit
                    if pair_count >= max_pairs:
                        self.logger.warning(
                            f"Reached max_duplicate_pairs limit ({max_pairs}), stopping pair search. "
                            f"Found {pair_count} pairs so far."
                        )
                        break
                    
                    id_i, hv_i, _ = valid_entries[i]
                    id_j, hv_j, _ = valid_entries[j]
                    
                    # Use cached analysis if available
                    cache_key = (id_i, id_j) if id_i < id_j else (id_j, id_i)
                    if cache_key in self._similarity_cache:
                        analysis = self._get_cached_similarity_analysis(cache_key)
                        self._search_stats['cache_hits'] += 1
                    else:
                        analysis = self._analyze_similarity(hv_i, hv_j, id_i, id_j)
                        self._similarity_cache[cache_key] = analysis
                        self._search_stats['cache_misses'] += 1
                    
                    pairs.append((id_i, id_j, distances[i, j], z_scores[i, j], analysis))
                    pair_count += 1
            
            # Break outer loop if limit reached
            if pair_count >= max_pairs:
                break
        
        # Sort by z-score (descending)
        pairs.sort(key=lambda x: -x[3])
        
        if limit:
            pairs = pairs[:limit]
        
        return pairs
    
    def _compute_distances_vectorized(self, query_hv: NDArray[np.float32], 
                                    matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute distances between query vector and matrix rows."""
        # Use Hamming distance for bipolar vectors
        if self.space.bipolar:
            # For bipolar vectors: Hamming distance = (dim - dot_product) / (2 * dim)
            dot_products = np.dot(matrix, query_hv)
            distances = (self.space.dim - dot_products) / (2 * self.space.dim)
        else:
            # For binary vectors: direct Hamming distance
            distances = np.mean(matrix != query_hv, axis=1)
        
        return distances.astype(np.float32)
    
    def _compute_pairwise_distances_vectorized(self, matrix: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute all pairwise distances efficiently."""
        n = matrix.shape[0]
        
        # For large matrices, warn about memory usage
        if n > 1000:
            self.logger.warning(f"Computing pairwise distances for {n} vectors - this may take time")
        
        if self.space.bipolar:
            # For bipolar vectors - ensure float32 computation
            # Normalize to [-1, 1] if needed
            if matrix.dtype != np.float32:
                matrix = matrix.astype(np.float32)
            
            # Compute dot products in chunks for very large matrices
            if n > 500:
                # Process in chunks to avoid memory issues
                chunk_size = 100
                distances = np.zeros((n, n), dtype=np.float32)
                for i in range(0, n, chunk_size):
                    end_i = min(i + chunk_size, n)
                    for j in range(0, n, chunk_size):
                        end_j = min(j + chunk_size, n)
                        chunk_dot = np.dot(matrix[i:end_i], matrix[j:end_j].T)
                        distances[i:end_i, j:end_j] = (self.space.dim - chunk_dot) / (2 * self.space.dim)
            else:
                dot_products = np.dot(matrix, matrix.T)
                distances = (self.space.dim - dot_products) / (2 * self.space.dim)
        else:
            # For binary vectors
            distances = np.zeros((n, n), dtype=np.float32)
            for i in range(n):
                for j in range(i + 1, n):
                    dist = np.mean(matrix[i] != matrix[j])
                    distances[i, j] = distances[j, i] = dist
        
        return distances
    
    def _get_cached_similarity_analysis(self, cache_key: Tuple[str, str]) -> SimilarityAnalysis:
        """Get cached similarity analysis."""
        cached = self._similarity_cache.get(cache_key)
        if isinstance(cached, dict):
            return cast(SimilarityAnalysis, cached)
        else:
            # Legacy cache format, create minimal analysis
            id1, id2 = cache_key
            return {
                "files_same": id1.split('@')[1].split(':')[0] == id2.split('@')[1].split(':')[0],
                "names_similar": self._name_similarity(id1, id2) > 0.5,
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for the enhanced index."""
        valid_count = sum(1 for _, hv, meta in self.entries
                         if hv is not None and not meta.get("removed", False))
        
        stats = {
            "index_stats": {
                "total_entries": len(self.entries),
                "valid_entries": valid_count,
                "pending_updates": len(getattr(self, '_incremental_updates', [])),
                "matrix_valid": getattr(self, '_matrix_valid', False),
                "last_rebuild_time": getattr(self, '_last_rebuild_time', 0.0),
            },
            "cache_stats": {
                "similarity_cache_size": len(getattr(self, '_similarity_cache', {})),
                "cache_hit_rate": (
                    self._search_stats['cache_hits'] / 
                    max(1, self._search_stats['cache_hits'] + self._search_stats['cache_misses'])
                ) if hasattr(self, '_search_stats') else 0.0,
                "matrix_rebuilds": getattr(self, '_search_stats', {}).get('matrix_rebuilds', 0),
                "incremental_updates": getattr(self, '_search_stats', {}).get('incremental_updates', 0),
            },
            "space_stats": self.space.get_stats(),
            "background_stats": {
                "mean": self._background_stats[0] if self._background_stats else None,
                "std": self._background_stats[1] if self._background_stats else None,
                "sample_size": self._stats_sample_size
            }
        }
        
        return stats
    
    def optimize_cache(self, max_cache_size: int = 10000) -> None:
        """Optimize cache sizes to prevent memory bloat."""
        if not hasattr(self, '_similarity_cache'):
            return
            
        # Trim similarity cache if too large
        if len(self._similarity_cache) > max_cache_size:
            # Keep most recently accessed items (simple LRU approximation)
            cache_items = list(self._similarity_cache.items())
            # Keep the last half
            keep_size = max_cache_size // 2
            self._similarity_cache = dict(cache_items[-keep_size:])
            
            if hasattr(self, 'logger'):
                self.logger.info(f"Trimmed similarity cache from {len(cache_items)} to {len(self._similarity_cache)} items")
    
    def force_rebuild(self) -> None:
        """Force rebuild of all internal structures."""
        if hasattr(self, 'logger'):
            self.logger.info("Forcing rebuild of semantic index structures")
        
        # Process pending updates
        if hasattr(self, '_incremental_updates') and self._incremental_updates:
            self._process_incremental_updates()
        
        # Rebuild vector matrix
        self._rebuild_vector_matrix()
        
        # Recompute background statistics
        self._background_stats = None
        self._ensure_background_stats()
        
        # Clear and rebuild similarity cache
        if hasattr(self, '_similarity_cache'):
            self._similarity_cache.clear()
        
        if hasattr(self, 'logger'):
            self.logger.info("Semantic index rebuild complete")