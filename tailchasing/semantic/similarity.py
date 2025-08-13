# tailchasing/semantic/similarity.py
from __future__ import annotations

import hashlib
import math
import threading
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Tuple, Union

# Optional: import your FunctionRecord for better typing; falls back gracefully
try:
    from tailchasing.semantic.index import FunctionRecord  # noqa: F401
    _HasFunctionRecord = True
except Exception:  # pragma: no cover
    _HasFunctionRecord = False


# ----------------------------
# Core Bloom Filter
# ----------------------------

@dataclass
class BloomConfig:
    capacity: int = 1_000_000
    error_rate: float = 0.001
    seed: int = 0xB10F_11AA  # affects hash families


class BloomFilter:
    """
    Lightweight Bloom filter with bytearray-backed bitset.

    - m (bits) and k (hash functions) are derived from capacity and error rate:
        m = -n * ln(p) / (ln(2)^2)
        k = (m / n) * ln(2)
    - Uses blake2b with salted variants to simulate k independent hashes.
    - Thread-safe for add/contains via a single lock (low contention).
    """
    __slots__ = ("m", "k", "_bits", "_lock", "_salts")

    def __init__(self, capacity: int, error_rate: float = 0.001, seed: int = 0xB10F_11AA) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be > 0")
        if not (0.0 < error_rate < 1.0):
            raise ValueError("error_rate must be in (0,1)")

        ln2 = math.log(2.0)
        m_float = -capacity * math.log(error_rate) / (ln2 * ln2)
        m = int(math.ceil(m_float))  # total bits
        # round up to full bytes
        num_bytes = (m + 7) // 8
        self.m = num_bytes * 8  # effective bit count after rounding

        k_float = (self.m / capacity) * ln2
        self.k = max(1, int(round(k_float)))

        self._bits = bytearray(num_bytes)
        self._lock = threading.Lock()

        # Precompute k salt values (8 bytes each) for blake2b salt param
        # Note: blake2b salt must be <=16 bytes; we use 8
        rng = _SplitMix64(seed)
        self._salts = [rng.next_bytes(8) for _ in range(self.k)]

    # --- public API ---

    def add(self, item: Union[str, bytes, bytearray]) -> None:
        """Add item to the Bloom filter."""
        indices = self._indexes(item)
        with self._lock:
            for idx in indices:
                self._set_bit(idx)

    def __contains__(self, item: Union[str, bytes, bytearray]) -> bool:
        indices = self._indexes(item)
        with self._lock:
            return all(self._get_bit(idx) for idx in indices)

    # --- internals ---

    def _indexes(self, item: Union[str, bytes, bytearray]) -> Iterable[int]:
        if isinstance(item, str):
            data = item.encode("utf-8", errors="ignore")
        elif isinstance(item, (bytes, bytearray)):
            data = bytes(item)
        else:
            # last-resort stable repr
            data = repr(item).encode("utf-8", errors="ignore")

        # For each salt, compute a 64-bit digest and mod by bit length
        for salt in self._salts:
            hv = hashlib.blake2b(data, digest_size=8, salt=salt).digest()
            pos = int.from_bytes(hv, "little") % self.m
            yield pos

    def _get_bit(self, pos: int) -> int:
        byte_index = pos >> 3
        bit_mask = 1 << (pos & 7)
        return 1 if (self._bits[byte_index] & bit_mask) else 0

    def _set_bit(self, pos: int) -> None:
        byte_index = pos >> 3
        bit_mask = 1 << (pos & 7)
        self._bits[byte_index] |= bit_mask


class _SplitMix64:
    """
    Small deterministic PRNG for salt generation.
    """
    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFFFFFFFFFF

    def next(self) -> int:
        # SplitMix64 step
        z = (self.state + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF
        self.state = z
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
        z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
        z = z ^ (z >> 31)
        return z

    def next_bytes(self, n: int) -> bytes:
        out = bytearray()
        while len(out) < n:
            out.extend(self.next().to_bytes(8, "little"))
        return bytes(out[:n])


# ----------------------------
# Comparison Cache
# ----------------------------

class ComparisonCache:
    """
    Prevents redundant comparisons of the same unordered function pair.

    Usage:
        cache = ComparisonCache(expected_comparisons=1_000_000)
        if cache.should_compare(func_a, func_b):
            # do expensive comparison
            ...
    """
    __slots__ = ("bloom",)

    def __init__(self, expected_comparisons: int = 1_000_000, error_rate: float = 0.001) -> None:
        self.bloom = BloomFilter(capacity=expected_comparisons, error_rate=error_rate)

    def should_compare(self, func1: Union[str, Any], func2: Union[str, Any]) -> bool:
        """
        Returns False if this unordered pair was seen before; True otherwise (and records it).

        func1/func2 can be:
          - FunctionRecord (we use .id)
          - str (treated as ID)
          - anything with a stable string repr (fallback: repr(obj))
        """
        key = self.get_canonical_key(func1, func2)
        if key in self.bloom:
            return False
        self.bloom.add(key)
        return True

    # --- helpers ---

    @staticmethod
    def _to_id(x: Union[str, Any]) -> str:
        # Prefer FunctionRecord.id if available
        if _HasFunctionRecord:
            from tailchasing.semantic.index import FunctionRecord as _FR  # type: ignore
            if isinstance(x, _FR):
                return x.id  # type: ignore[attr-defined]
        # string ID
        if isinstance(x, str):
            return x
        # fallback stable repr
        return repr(x)

    @classmethod
    def get_canonical_key(cls, a: Union[str, Any], b: Union[str, Any]) -> str:
        """
        Order-invariant key for the pair (a,b). Stable across runs.
        """
        ia, ib = cls._to_id(a), cls._to_id(b)
        if ia == ib:
            # Degenerate self-pair; still generate a stable key
            payload = f"SELF::{ia}".encode("utf-8")
        else:
            lo, hi = sorted((ia, ib))
            payload = f"{lo}||{hi}".encode("utf-8")

        digest = hashlib.blake2b(payload, digest_size=16).hexdigest()
        return f"cmp::{digest}"


# ----------------------------
# Optional: Additional similarity utilities
# ----------------------------

class SimilarityAnalyzer:
    """
    Placeholder for compatibility with existing code that imports SimilarityAnalyzer.
    
    This can be extended with actual similarity analysis methods as needed.
    """
    
    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.comparison_cache = ComparisonCache(
            expected_comparisons=self.config.get('max_comparisons', 1_000_000),
            error_rate=self.config.get('bloom_error_rate', 0.001)
        )
    
    def filter_significant_pairs(self, pairs: list) -> list:
        """
        Filter pairs to only include those that haven't been compared before.
        
        Args:
            pairs: List of candidate pairs
            
        Returns:
            Filtered list of pairs
        """
        filtered = []
        for pair in pairs:
            # Assume pair is (id1, id2, ...) tuple
            if len(pair) >= 2:
                if self.comparison_cache.should_compare(pair[0], pair[1]):
                    filtered.append(pair)
        return filtered
    
    def cluster_similar_functions(self, entries: dict, space: Any) -> list:
        """
        Placeholder for clustering similar functions.
        
        Args:
            entries: Dictionary of function entries
            space: Hypervector space (unused in this simple implementation)
            
        Returns:
            List of clusters (empty for now)
        """
        # This is a placeholder implementation
        # Real clustering would group similar functions based on their hypervectors
        return []
    
    def analyze_channel_contributions(self, hv1: Any, hv2: Any, space: Any, 
                                     features1: dict, features2: dict) -> dict:
        """
        Placeholder for analyzing channel contributions.
        
        Args:
            hv1: First hypervector
            hv2: Second hypervector  
            space: Hypervector space
            features1: Features of first function
            features2: Features of second function
            
        Returns:
            Dictionary of channel contributions (empty for now)
        """
        # This is a placeholder implementation
        # Real implementation would analyze which channels contribute most to similarity
        return {}