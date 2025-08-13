"""
Memory-mapped append-only index for hypervector storage.

Implements catalytic computing principles with minimal working memory
and efficient large-scale indexing using LSH bucketing.
"""

import json
import mmap
import os
import struct
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set, Any, Iterator
from dataclasses import dataclass, asdict
import numpy as np
import tempfile
import shutil
from datetime import datetime


@dataclass
class IndexMetadata:
    """Metadata for an indexed function."""
    function_id: str
    file_path: str
    function_name: str
    line_number: int
    ast_hash: str
    timestamp: str
    vector_offset: int
    vector_length: int
    lsh_buckets: List[int]
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_json(cls, json_str: str) -> 'IndexMetadata':
        """Create from JSON string."""
        return cls(**json.loads(json_str))


class LSHIndex:
    """
    Locality-Sensitive Hashing for approximate nearest neighbor search.
    
    Uses random hyperplanes for ternary vectors.
    """
    
    def __init__(self, dim: int = 8192, n_tables: int = 8, n_hyperplanes: int = 16):
        """
        Initialize LSH index.
        
        Args:
            dim: Vector dimension
            n_tables: Number of hash tables
            n_hyperplanes: Number of hyperplanes per table
        """
        self.dim = dim
        self.n_tables = n_tables
        self.n_hyperplanes = n_hyperplanes
        
        # Generate random hyperplanes (deterministic)
        rng = np.random.RandomState(42)
        self.hyperplanes = []
        for _ in range(n_tables):
            # Ternary hyperplanes for ternary vectors
            hp = rng.choice([-1, 0, 1], (n_hyperplanes, dim))
            self.hyperplanes.append(hp)
        
        # Bucket storage
        self.buckets: List[Dict[int, Set[str]]] = [
            {} for _ in range(n_tables)
        ]
    
    def hash_vector(self, vec: np.ndarray) -> List[int]:
        """
        Compute LSH hashes for a vector.
        
        Args:
            vec: Input vector
            
        Returns:
            List of hash values (one per table)
        """
        hashes = []
        for hyperplanes in self.hyperplanes:
            # Project vector onto hyperplanes
            projections = np.dot(hyperplanes, vec)
            # Convert to binary hash
            bits = (projections > 0).astype(int)
            # Convert bits to integer
            hash_val = int(''.join(map(str, bits)), 2) if len(bits) > 0 else 0
            hashes.append(hash_val)
        return hashes
    
    def add(self, vec: np.ndarray, item_id: str) -> List[int]:
        """
        Add vector to LSH index.
        
        Args:
            vec: Vector to add
            item_id: Identifier for the vector
            
        Returns:
            LSH bucket IDs
        """
        hashes = self.hash_vector(vec)
        for i, hash_val in enumerate(hashes):
            if hash_val not in self.buckets[i]:
                self.buckets[i][hash_val] = set()
            self.buckets[i][hash_val].add(item_id)
        return hashes
    
    def query(self, vec: np.ndarray, max_candidates: int = 100) -> Set[str]:
        """
        Query for similar vectors.
        
        Args:
            vec: Query vector
            max_candidates: Maximum candidates to return
            
        Returns:
            Set of candidate item IDs
        """
        hashes = self.hash_vector(vec)
        candidates = set()
        
        for i, hash_val in enumerate(hashes):
            if hash_val in self.buckets[i]:
                candidates.update(self.buckets[i][hash_val])
                if len(candidates) >= max_candidates:
                    break
        
        return candidates


class CatalyticIndex:
    """
    Memory-mapped append-only index for hypervectors.
    
    Features:
    - Append-only operations (never mutates existing data)
    - Memory-mapped files for efficient large-scale storage
    - JSONL metadata with fast seeking
    - LSH indexing for ANN search
    - Sharded storage for versioning
    - Working memory <100MB for 100k+ functions
    """
    
    # Constants
    VECTOR_DTYPE = np.int8
    VECTOR_SIZE = 8192  # bytes per vector
    METADATA_FILE = 'metadata.jsonl'
    VECTORS_FILE = 'vectors.dat'
    LSH_FILE = 'lsh_index.pkl'
    SHARD_SIZE = 10000  # Functions per shard
    
    def __init__(self, index_dir: str, mode: str = 'r'):
        """
        Initialize catalytic index.
        
        Args:
            index_dir: Directory for index storage
            mode: 'r' for read-only, 'w' for write, 'a' for append
        """
        self.index_dir = Path(index_dir)
        self.mode = mode
        
        # Create directory if needed
        if mode in ('w', 'a'):
            self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage
        self.metadata_path = self.index_dir / self.METADATA_FILE
        self.vectors_path = self.index_dir / self.VECTORS_FILE
        
        # Memory-mapped files
        self.vectors_mmap: Optional[mmap.mmap] = None
        self.vectors_file = None
        
        # LSH index
        self.lsh_index = LSHIndex()
        
        # Metadata cache (small working set)
        self._metadata_cache: Dict[str, IndexMetadata] = {}
        self._id_to_offset: Dict[str, int] = {}
        
        # Statistics
        self.num_functions = 0
        self.total_bytes = 0
        
        # Open files based on mode
        self._open_files()
        
        # Load existing metadata if appending or reading
        if mode in ('r', 'a'):
            self._load_metadata()
    
    def _open_files(self) -> None:
        """Open memory-mapped files based on mode."""
        if self.mode == 'r':
            if self.vectors_path.exists():
                self.vectors_file = open(self.vectors_path, 'rb')
                self.vectors_mmap = mmap.mmap(
                    self.vectors_file.fileno(), 0, access=mmap.ACCESS_READ
                )
        elif self.mode in ('w', 'a'):
            # Open for append
            mode = 'ab' if self.mode == 'a' else 'wb'
            self.vectors_file = open(self.vectors_path, mode)
    
    def _load_metadata(self) -> None:
        """Load metadata from JSONL file."""
        if not self.metadata_path.exists():
            return
        
        with open(self.metadata_path, 'r') as f:
            for line_num, line in enumerate(f):
                if line.strip():
                    try:
                        metadata = IndexMetadata.from_json(line)
                        self._metadata_cache[metadata.function_id] = metadata
                        self._id_to_offset[metadata.function_id] = metadata.vector_offset
                        
                        # Rebuild LSH index
                        for bucket_id in metadata.lsh_buckets:
                            table_idx = bucket_id % self.lsh_index.n_tables
                            if bucket_id not in self.lsh_index.buckets[table_idx]:
                                self.lsh_index.buckets[table_idx][bucket_id] = set()
                            self.lsh_index.buckets[table_idx][bucket_id].add(metadata.function_id)
                        
                        self.num_functions += 1
                    except json.JSONDecodeError:
                        pass  # Skip corrupted lines
        
        # Update statistics
        if self.vectors_path.exists():
            self.total_bytes = self.vectors_path.stat().st_size
    
    def add_function(self, function_id: str, hypervector: np.ndarray,
                    file_path: str, function_name: str, line_number: int,
                    ast_hash: str) -> IndexMetadata:
        """
        Add a function to the index (append-only).
        
        Args:
            function_id: Unique identifier
            hypervector: Ternary hypervector encoding
            file_path: Source file path
            function_name: Function name
            line_number: Line number in source
            ast_hash: Hash of normalized AST
            
        Returns:
            Metadata for the indexed function
        """
        if self.mode not in ('w', 'a'):
            raise ValueError("Index opened in read-only mode")
        
        # Ensure vector is correct type and shape
        if hypervector.dtype != self.VECTOR_DTYPE:
            hypervector = hypervector.astype(self.VECTOR_DTYPE)
        if hypervector.shape != (self.VECTOR_SIZE,):
            raise ValueError(f"Vector must be shape ({self.VECTOR_SIZE},)")
        
        # Compute LSH buckets
        lsh_buckets = self.lsh_index.add(hypervector, function_id)
        
        # Create metadata
        metadata = IndexMetadata(
            function_id=function_id,
            file_path=file_path,
            function_name=function_name,
            line_number=line_number,
            ast_hash=ast_hash,
            timestamp=datetime.utcnow().isoformat(),
            vector_offset=self.total_bytes,
            vector_length=self.VECTOR_SIZE,
            lsh_buckets=lsh_buckets
        )
        
        # Append vector to memory-mapped file
        self.vectors_file.write(hypervector.tobytes())
        self.vectors_file.flush()
        
        # Append metadata to JSONL
        with open(self.metadata_path, 'a') as f:
            f.write(metadata.to_json() + '\n')
        
        # Update caches
        self._metadata_cache[function_id] = metadata
        self._id_to_offset[function_id] = metadata.vector_offset
        
        # Update statistics
        self.num_functions += 1
        self.total_bytes += self.VECTOR_SIZE
        
        return metadata
    
    def get_vector(self, function_id: str) -> Optional[np.ndarray]:
        """
        Retrieve hypervector for a function.
        
        Args:
            function_id: Function identifier
            
        Returns:
            Hypervector or None if not found
        """
        if function_id not in self._id_to_offset:
            return None
        
        if self.vectors_mmap is None:
            # Reopen for reading if needed
            if self.vectors_file:
                self.vectors_file.close()
            self.vectors_file = open(self.vectors_path, 'rb')
            self.vectors_mmap = mmap.mmap(
                self.vectors_file.fileno(), 0, access=mmap.ACCESS_READ
            )
        
        offset = self._id_to_offset[function_id]
        self.vectors_mmap.seek(offset)
        data = self.vectors_mmap.read(self.VECTOR_SIZE)
        
        return np.frombuffer(data, dtype=self.VECTOR_DTYPE)
    
    def query_similar(self, query_vector: np.ndarray, 
                     max_candidates: int = 100) -> List[str]:
        """
        Query for similar functions using LSH.
        
        Args:
            query_vector: Query hypervector
            max_candidates: Maximum candidates to retrieve
            
        Returns:
            List of candidate function IDs
        """
        candidates = self.lsh_index.query(query_vector, max_candidates)
        return list(candidates)
    
    def iterate_vectors(self, batch_size: int = 100) -> Iterator[Tuple[str, np.ndarray]]:
        """
        Iterate over all vectors in batches.
        
        Args:
            batch_size: Number of vectors per batch
            
        Yields:
            Tuples of (function_id, hypervector)
        """
        if self.vectors_mmap is None and self.vectors_path.exists():
            if self.vectors_file:
                self.vectors_file.close()
            self.vectors_file = open(self.vectors_path, 'rb')
            self.vectors_mmap = mmap.mmap(
                self.vectors_file.fileno(), 0, access=mmap.ACCESS_READ
            )
        
        if self.vectors_mmap is None:
            return
        
        batch = []
        for function_id, offset in self._id_to_offset.items():
            self.vectors_mmap.seek(offset)
            data = self.vectors_mmap.read(self.VECTOR_SIZE)
            vector = np.frombuffer(data, dtype=self.VECTOR_DTYPE)
            batch.append((function_id, vector))
            
            if len(batch) >= batch_size:
                yield from batch
                batch = []
        
        if batch:
            yield from batch
    
    def get_metadata(self, function_id: str) -> Optional[IndexMetadata]:
        """Get metadata for a function."""
        return self._metadata_cache.get(function_id)
    
    def create_shard(self, shard_name: str) -> 'CatalyticIndex':
        """
        Create a new shard for versioning.
        
        Args:
            shard_name: Name for the shard
            
        Returns:
            New CatalyticIndex for the shard
        """
        shard_dir = self.index_dir / f'shard_{shard_name}'
        return CatalyticIndex(str(shard_dir), mode='w')
    
    def merge_shard(self, shard: 'CatalyticIndex') -> None:
        """
        Merge another shard into this index.
        
        Args:
            shard: Shard to merge
        """
        if self.mode not in ('w', 'a'):
            raise ValueError("Index opened in read-only mode")
        
        # Copy vectors
        for function_id, vector in shard.iterate_vectors():
            metadata = shard.get_metadata(function_id)
            if metadata and function_id not in self._id_to_offset:
                self.add_function(
                    function_id=function_id,
                    hypervector=vector,
                    file_path=metadata.file_path,
                    function_name=metadata.function_name,
                    line_number=metadata.line_number,
                    ast_hash=metadata.ast_hash
                )
    
    def close(self) -> None:
        """Close memory-mapped files."""
        if self.vectors_mmap:
            self.vectors_mmap.close()
            self.vectors_mmap = None
        if self.vectors_file:
            self.vectors_file.close()
            self.vectors_file = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        return {
            'num_functions': self.num_functions,
            'total_bytes': self.total_bytes,
            'avg_bytes_per_function': self.total_bytes / max(1, self.num_functions),
            'metadata_cache_size': len(self._metadata_cache),
            'working_memory_mb': self._estimate_memory_usage() / (1024 * 1024)
        }
    
    def _estimate_memory_usage(self) -> int:
        """Estimate current memory usage in bytes."""
        # Metadata cache
        metadata_size = len(self._metadata_cache) * 500  # ~500 bytes per entry
        
        # LSH index
        lsh_size = 0
        for table in self.lsh_index.buckets:
            for bucket_set in table.values():
                lsh_size += len(bucket_set) * 50  # ~50 bytes per ID
        
        # Hyperplanes
        hyperplane_size = (
            self.lsh_index.n_tables * 
            self.lsh_index.n_hyperplanes * 
            self.lsh_index.dim
        )
        
        return metadata_size + lsh_size + hyperplane_size
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()