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
from typing import Dict, List, Tuple, Any
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
    
    def weighted_bundle(self, vecs_weights: List[Tuple[np.ndarray, float]]) -> np.ndarray:
        """
        Bundle vectors with weights.
        
        Args:
            vecs_weights: List of (vector, weight) tuples
            
        Returns:
            Weighted bundle of vectors
        """
        if not vecs_weights:
            return self._rand_vec()
        
        if len(vecs_weights) == 1:
            return vecs_weights[0][0].copy()
        
        # Expand vectors based on weights
        expanded_vecs = []
        for vec, weight in vecs_weights:
            repeat_count = max(1, int(weight * 10))  # Scale weights to reasonable repeat counts
            for _ in range(repeat_count):
                expanded_vecs.append(vec)
        
        return self.bundle(expanded_vecs)
    
    def unbind(self, bound: np.ndarray, key: np.ndarray) -> np.ndarray:
        """
        Unbind a vector using a key (approximate inverse of bind).
        
        For bipolar: multiply by key (since bind is multiplication)
        For binary: XOR with key (since bind is XOR)
        
        Args:
            bound: Bound vector (result of bind operation)
            key: Key vector used in original bind
            
        Returns:
            Approximation of original value vector
        """
        return self.bind(bound, key)  # Bind is its own inverse
    
    def cleanup(self, vec: np.ndarray, candidates: List[np.ndarray]) -> np.ndarray:
        """
        Clean up a vector by finding the most similar candidate.
        
        Args:
            vec: Vector to clean up
            candidates: List of candidate vectors
            
        Returns:
            Most similar candidate vector
        """
        if not candidates:
            return vec
        
        max_similarity = -1.0
        best_candidate = vec
        
        for candidate in candidates:
            sim = self.similarity(vec, candidate)
            if sim > max_similarity:
                max_similarity = sim
                best_candidate = candidate
        
        return best_candidate
    
    def generate_orthogonal_set(self, n: int) -> List[np.ndarray]:
        """
        Generate a set of approximately orthogonal vectors.
        
        Args:
            n: Number of vectors to generate
            
        Returns:
            List of approximately orthogonal vectors
        """
        if n > self.capacity_estimate():
            raise ValueError(f"Requested {n} vectors exceeds capacity estimate {self.capacity_estimate()}")
        
        vectors = []
        max_attempts = n * 10
        attempts = 0
        
        while len(vectors) < n and attempts < max_attempts:
            candidate = self._rand_vec()
            
            # Check orthogonality with existing vectors
            is_orthogonal = True
            min_distance = 0.4  # Minimum distance threshold
            
            for existing in vectors:
                if self.distance(candidate, existing) < min_distance:
                    is_orthogonal = False
                    break
            
            if is_orthogonal:
                vectors.append(candidate)
            
            attempts += 1
        
        return vectors
    
    def create_sequence_vector(self, items: List[str], use_permutation: bool = True) -> np.ndarray:
        """
        Create a hypervector representing a sequence of items.
        
        Args:
            items: List of items in sequence
            use_permutation: Whether to use permutation for order encoding
            
        Returns:
            Hypervector encoding the sequence
        """
        if not items:
            return self._rand_vec()
        
        if not use_permutation:
            # Simple bundle without order information
            item_vecs = [self.token(item) for item in items]
            return self.bundle(item_vecs)
        
        # Encode sequence with positional information
        sequence_vecs = []
        for i, item in enumerate(items):
            item_vec = self.token(item)
            # Permute by position to encode order
            pos_vec = self.permute(item_vec, i)
            sequence_vecs.append(pos_vec)
        
        return self.bundle(sequence_vecs)
    
    def threshold_cleanup(self, vec: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Clean up a vector by applying a threshold.
        
        Args:
            vec: Input vector
            threshold: Threshold for cleanup
            
        Returns:
            Cleaned up vector
        """
        if self.bipolar:
            # For bipolar, threshold on magnitude
            return np.sign(vec).astype(np.int16)
        else:
            # For binary, apply threshold
            float_vec = vec.astype(float) / np.max(vec) if np.max(vec) > 0 else vec.astype(float)
            return (float_vec >= threshold).astype(np.uint8)
    
    def analyze_vector(self, vec: np.ndarray) -> Dict[str, Any]:
        """
        Analyze properties of a hypervector.
        
        Args:
            vec: Vector to analyze
            
        Returns:
            Dictionary with vector statistics
        """
        stats = {
            'dimension': len(vec),
            'dtype': str(vec.dtype),
            'min_val': int(np.min(vec)),
            'max_val': int(np.max(vec)),
            'mean': float(np.mean(vec)),
            'std': float(np.std(vec))
        }
        
        if self.bipolar:
            positive_count = int(np.sum(vec > 0))
            negative_count = int(np.sum(vec < 0))
            zero_count = int(np.sum(vec == 0))
            
            stats.update({
                'positive_elements': positive_count,
                'negative_elements': negative_count,
                'zero_elements': zero_count,
                'balance_ratio': positive_count / (positive_count + negative_count) if (positive_count + negative_count) > 0 else 0.0
            })
        else:
            ones_count = int(np.sum(vec == 1))
            zeros_count = int(np.sum(vec == 0))
            
            stats.update({
                'ones': ones_count,
                'zeros': zeros_count,
                'density': ones_count / len(vec)
            })
        
        return stats
    
    def clear_caches(self):
        """Clear token and role caches to free memory."""
        self._token_cache.clear()
        self._role_cache.clear()
        # Re-initialize common roles
        self._init_common_roles()
    
    def save_space(self, filepath: str):
        """
        Save the hypervector space to a file.
        
        Args:
            filepath: Path to save the space
        """
        import pickle
        
        data = {
            'dim': self.dim,
            'bipolar': self.bipolar,
            'token_cache': self._token_cache,
            'role_cache': self._role_cache
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    @classmethod
    def load_space(cls, filepath: str, seed: int = None) -> 'HVSpace':
        """
        Load a hypervector space from a file.
        
        Args:
            filepath: Path to load the space from
            seed: Optional seed override
            
        Returns:
            Loaded HVSpace instance
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        space = cls(
            dim=data['dim'],
            bipolar=data['bipolar'],
            seed=seed if seed is not None else 42
        )
        
        space._token_cache = data['token_cache']
        space._role_cache = data['role_cache']
        
        return space