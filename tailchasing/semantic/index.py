"""
Semantic index for hypervector storage and similarity search.

Manages the collection of function hypervectors, computes background
statistics, and identifies significantly similar pairs.
"""

import json
import pickle
import random
import math
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
        Initialize semantic index.
        
        Args:
            config: Semantic configuration
            cache_dir: Directory for persisting index
        """
        self.config = config
        self.cache_dir = cache_dir
        
        # Initialize hypervector space
        self.space = HVSpace(
            dim=config.get('hv_dim', 8192),
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
        
        # Load cache if available
        if cache_dir:
            self._load_cache()
    
    def add(self, func_id: str, file: str, line: int, hv: NDArray[np.float32],
            metadata: Optional[FunctionEntry] = None) -> None:
        """Add a function hypervector to the index."""
        full_id = f"{func_id}@{file}:{line}"
        
        # Check if already exists
        if full_id in self.id_to_index:
            # Update existing entry
            idx = self.id_to_index[full_id]
            entry_meta = metadata or cast(FunctionEntry, {})
            self.entries[idx] = (full_id, hv, entry_meta)
        else:
            # Add new entry
            idx = len(self.entries)
            entry_meta = metadata or cast(FunctionEntry, {})
            self.entries.append((full_id, hv, entry_meta))
            self.id_to_index[full_id] = idx
        
        # Invalidate background stats
        self._background_stats = None
    
    def remove(self, func_id: str, file: str, line: int) -> bool:
        """Remove a function from the index."""
        full_id = f"{func_id}@{file}:{line}"
        
        if full_id not in self.id_to_index:
            return False
        
        # Mark as removed (don't shift indices)
        idx = self.id_to_index[full_id]
        removed_meta: FunctionEntry = {"removed": True}
        self.entries[idx] = (full_id, None, removed_meta)
        del self.id_to_index[full_id]
        
        # Invalidate stats
        self._background_stats = None
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
        
        # Sample random pairs
        max_pairs = min(
            self.config.get('max_pairs_sample', 10000),
            n * (n - 1) // 2
        )
        
        # Use reservoir sampling for large spaces
        distances: List[float] = []
        pairs_seen: int = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                pairs_seen += 1
                
                if len(distances) < max_pairs:
                    # Haven't filled reservoir yet
                    d = self.space.distance(valid_entries[i][1], valid_entries[j][1])
                    distances.append(d)
                else:
                    # Reservoir sampling
                    k = random.randint(0, pairs_seen - 1)
                    if k < max_pairs:
                        d = self.space.distance(valid_entries[i][1], valid_entries[j][1])
                        distances[k] = d
        
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
        Find functions similar to given hypervector.
        
        Returns list of (id, distance, z_score, metadata) tuples.
        """
        if z_threshold is None:
            z_threshold = self.config.get('z_threshold', 2.5)
        
        results: List[Tuple[str, float, float, FunctionEntry]] = []
        
        for entry_id, entry_hv, entry_meta in self.entries:
            if entry_hv is None or entry_meta.get("removed", False):
                continue
            
            distance = self.space.distance(hv, entry_hv)
            z_score = self.compute_z_score(distance)
            
            if z_score >= z_threshold:
                results.append((entry_id, distance, z_score, entry_meta))
        
        # Sort by z-score (descending)
        results.sort(key=lambda x: -x[2])
        
        return results[:limit]
    
    def find_all_similar_pairs(self, 
                              z_threshold: Optional[float] = None,
                              limit: Optional[int] = None) -> List[Tuple[str, str, float, float, SimilarityAnalysis]]:
        """
        Find all pairs of similar functions.
        
        Returns list of (id1, id2, distance, z_score, analysis) tuples.
        """
        if z_threshold is None:
            z_threshold = self.config.get('z_threshold', 2.5)
        
        valid_entries: List[Tuple[str, NDArray[np.float32], FunctionEntry]] = [
            (id, hv, meta) for id, hv, meta in self.entries
            if hv is not None and not meta.get("removed", False)
        ]
        
        pairs: List[Tuple[str, str, float, float, SimilarityAnalysis]] = []
        
        for i in range(len(valid_entries)):
            for j in range(i + 1, len(valid_entries)):
                id_i, hv_i, meta_i = valid_entries[i]
                id_j, hv_j, meta_j = valid_entries[j]
                
                distance = self.space.distance(hv_i, hv_j)
                z_score = self.compute_z_score(distance)
                
                if z_score >= z_threshold:
                    # Compute channel contribution analysis
                    analysis = self._analyze_similarity(hv_i, hv_j, id_i, id_j)
                    pairs.append((id_i, id_j, distance, z_score, analysis))
        
        # Sort by z-score
        pairs.sort(key=lambda x: -x[3])
        
        if limit:
            pairs = pairs[:limit]
        
        return pairs
    
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
        
        # TODO: Implement channel contribution analysis
        # This would require storing channel vectors separately
        # or using masking techniques
        
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
        """Save index to cache directory."""
        if not self.cache_dir:
            return
        
        cache_file = self.cache_dir / "semantic_index.pkl"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        cache_data = {
            "entries": self.entries,
            "id_to_index": self.id_to_index,
            "background_stats": self._background_stats,
            "stats_sample_size": self._stats_sample_size,
            "space_token_cache": self.space._token_cache,
            "space_role_cache": self.space._role_cache,
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
    
    def _load_cache(self) -> bool:
        """Load index from cache directory."""
        if not self.cache_dir:
            return False
        
        cache_file = self.cache_dir / "semantic_index.pkl"
        if not cache_file.exists():
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