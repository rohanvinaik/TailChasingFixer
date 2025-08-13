"""
Catalytic hypervector similarity system for efficient duplicate detection.

This module implements a catalytic computing approach to solve the O(N²) 
duplicate detection problem using:
- High-dimensional ternary vectors for semantic encoding
- Memory-mapped append-only storage for efficient large-scale indexing  
- LSH-based approximate nearest neighbor search
- Minimal working memory footprint (<100MB for 100k+ functions)

The system follows catalytic computing principles:
1. Borrow large read-only memory without mutation
2. Keep working set tiny and reversible
3. Use append-only operations for persistence
"""

from .hv_encoder import HypervectorEncoder, ASTNormalizer
from .catalytic_index import CatalyticIndex, IndexMetadata
from .similarity_pipeline import SimilarityPipeline, QueryResult

__all__ = [
    'HypervectorEncoder',
    'ASTNormalizer', 
    'CatalyticIndex',
    'IndexMetadata',
    'SimilarityPipeline',
    'QueryResult'
]