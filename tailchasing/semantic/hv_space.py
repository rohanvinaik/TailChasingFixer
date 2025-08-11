"""
Hypervector space implementation for semantic encoding.

This module implements a high-dimensional computing (HDC) approach to detect
semantic duplicates and tail-chasing patterns beyond lexical/structural analysis.

Key concepts:
- Distributed representation: meaning emerges from patterns across dimensions
- Compositional semantics: combine channel vectors via binding/bundling
- Robustness: similar concepts have similar vectors despite surface variations
"""

import random
import math
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


class HVSpace:
    """
    High-dimensional vector space for semantic encoding.
    
    Uses random hypervectors to encode tokens and roles, supporting
    binding and bundling operations for compositional semantics.
    """
    
    def __init__(self, dim: int = 8192, bipolar: bool = True, seed: int = 42):
        """
        Initialize hypervector space.
        
        Args:
            dim: Vector dimensionality (>= 1000 for good separation)
            bipolar: Use +1/-1 (True) or 0/1 (False) encoding
            seed: Random seed for reproducible encodings
        """
        self.dim = dim
        self.bipolar = bipolar
        random.seed(seed)
        np.random.seed(seed)
        
        # Cache for token and role vectors
        self._token_cache: Dict[str, np.ndarray] = {}
        self._role_cache: Dict[str, np.ndarray] = {}
        
        # Pre-generate common role vectors
        self._init_common_roles()
    
    def _init_common_roles(self):
        """Pre-generate role vectors for common channels."""
        channels = [
            "NAME_TOKENS", "DOC_TOKENS", "CALLS", "CONTROL",
            "LITERALS", "ARG_SIG", "RETURN_KIND", "DECORATORS",
            "IMPORTS", "EXCEPTIONS"
        ]
        for channel in channels:
            self.role(channel)
    
    def _rand_vec(self) -> np.ndarray:
        """Generate a random hypervector."""
        if self.bipolar:
            # +1/-1 encoding for cosine similarity
            # Use int16 to avoid overflow issues with large dimensions
            return np.random.choice([-1, 1], size=self.dim).astype(np.int16)
        else:
            # 0/1 encoding for Hamming distance
            return np.random.randint(0, 2, size=self.dim, dtype=np.uint8)
    
    def token(self, tok: str) -> np.ndarray:
        """Get or create hypervector for a token."""
        if tok not in self._token_cache:
            self._token_cache[tok] = self._rand_vec()
        return self._token_cache[tok]
    
    def role(self, name: str) -> np.ndarray:
        """Get or create hypervector for a role/channel."""
        if name not in self._role_cache:
            self._role_cache[name] = self._rand_vec()
        return self._role_cache[name]
    
    def bind(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Bind two hypervectors (role-filler binding).
        
        For bipolar: element-wise multiplication
        For binary: XOR operation
        
        This creates a new vector dissimilar to both inputs,
        preserving the binding relationship.
        """
        if self.bipolar:
            return a * b
        else:
            return np.bitwise_xor(a, b)
    
    def bundle(self, vecs: List[np.ndarray]) -> np.ndarray:
        """
        Bundle multiple hypervectors (superposition).
        
        Uses majority voting to combine vectors.
        This preserves similarity to all input vectors.
        """
        if not vecs:
            return self._rand_vec()
        
        if len(vecs) == 1:
            return vecs[0].copy()
        
        # Stack vectors for efficient computation
        stack = np.stack(vecs)
        
        if self.bipolar:
            # Sum and take sign (majority vote)
            sums = np.sum(stack, axis=0)
            return np.sign(sums).astype(np.int16)
        else:
            # Binary majority vote
            sums = np.sum(stack, axis=0)
            threshold = len(vecs) / 2
            return (sums >= threshold).astype(np.uint8)
    
    def permute(self, vec: np.ndarray, shift: int = 1) -> np.ndarray:
        """
        Permute hypervector (circular shift).
        
        Used to encode sequence/position information.
        """
        return np.roll(vec, shift)
    
    def hamming_distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute normalized Hamming distance between vectors."""
        if self.bipolar:
            # For bipolar, convert to agreement ratio
            agreements = np.sum(a == b)
            return 1.0 - (agreements / self.dim)
        else:
            # Direct Hamming distance for binary
            differences = np.sum(a != b)
            return differences / self.dim
    
    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between vectors."""
        if not self.bipolar:
            # Convert binary to bipolar for cosine
            a = 2 * a - 1
            b = 2 * b - 1
        
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def distance(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute distance between vectors.
        
        Uses Hamming distance for both binary and bipolar.
        """
        return self.hamming_distance(a, b)
    
    def similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        Compute similarity between vectors.
        
        Returns value in [0, 1] where 1 is identical.
        """
        return 1.0 - self.distance(a, b)
    
    def capacity_estimate(self) -> int:
        """
        Estimate the number of quasi-orthogonal vectors.
        
        Based on dimensionality and desired separation.
        """
        # Conservative estimate: ~0.1 * dim for good separation
        return int(0.1 * self.dim)
    
    def get_stats(self) -> Dict[str, int]:
        """Get space usage statistics."""
        return {
            "dimension": self.dim,
            "bipolar": self.bipolar,
            "tokens_encoded": len(self._token_cache),
            "roles_encoded": len(self._role_cache),
            "capacity_estimate": self.capacity_estimate(),
            "memory_bytes": self._estimate_memory()
        }
    
    def _estimate_memory(self) -> int:
        """Estimate memory usage in bytes."""
        # 2 bytes per element for int16, 1 byte for uint8
        vec_size = self.dim * 2 if self.bipolar else self.dim
        total_vecs = len(self._token_cache) + len(self._role_cache)
        return total_vecs * vec_size