"""
Semantic analysis module for TailChasingFixer.

This module provides semantic analysis capabilities for detecting similar
functions and patterns that go beyond simple structural matching.
"""

# Import key semantic analysis components
from .hv_space import HVSpace
from .encoder import encode_function, encode_function_with_context, split_identifier
from .index import SemanticIndex, FunctionEntry, SimilarityPair

# Import advanced semantic features if available
try:
    from .smart_filter import SmartFilter
except ImportError:
    SmartFilter = None

try:
    from .similarity import calculate_semantic_similarity
except ImportError:
    calculate_semantic_similarity = None

# Public API
__all__ = [
    # Core components
    'HVSpace',
    'SemanticIndex',
    'FunctionEntry', 
    'SimilarityPair',
    
    # Encoding functions
    'encode_function',
    'encode_function_with_context',
    'split_identifier',
    
    # Advanced features (if available)
    'SmartFilter',
    'calculate_semantic_similarity',
]

# Remove None values from __all__ for missing optional components
__all__ = [item for item in __all__ if globals().get(item) is not None]