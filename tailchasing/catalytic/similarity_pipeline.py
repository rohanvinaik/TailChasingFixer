"""
Query pipeline for efficient similarity search with verification.

Combines hypervector similarity with exact AST matching for high precision.
"""

import ast
import hashlib
from typing import List, Optional, Tuple, Dict, Any, Set
from dataclasses import dataclass
import numpy as np
from difflib import SequenceMatcher
import logging

from .hv_encoder import HypervectorEncoder, ASTNormalizer, EncodingConfig
from .catalytic_index import CatalyticIndex, IndexMetadata


@dataclass
class QueryResult:
    """Result from similarity query."""
    function_id: str
    file_path: str
    function_name: str
    line_number: int
    similarity_score: float
    hv_similarity: float
    ast_similarity: float
    confidence: float
    is_exact_match: bool
    
    def __lt__(self, other: 'QueryResult') -> bool:
        """Compare by similarity score for sorting."""
        return self.similarity_score < other.similarity_score


class ASTSkeletonMatcher:
    """
    Verifies similarity using exact AST skeleton matching.
    
    Compares normalized AST structure after removing implementation details.
    """
    
    def __init__(self):
        """Initialize the matcher."""
        self.normalizer = ASTNormalizer()
    
    def extract_skeleton(self, node: ast.AST) -> str:
        """
        Extract skeleton representation of AST.
        
        Args:
            node: AST node
            
        Returns:
            String representation of skeleton
        """
        # Normalize the AST
        normalized = self.normalizer.visit(node)
        
        # Convert to skeleton string (structure only)
        skeleton = self._node_to_skeleton(normalized)
        return skeleton
    
    def _node_to_skeleton(self, node: ast.AST, depth: int = 0) -> str:
        """Convert node to skeleton string."""
        if depth > 10:  # Limit depth
            return "..."
        
        node_type = type(node).__name__
        
        # For certain nodes, include key info
        if isinstance(node, ast.FunctionDef):
            args = len(node.args.args) if node.args else 0
            children = [self._node_to_skeleton(child, depth + 1) 
                       for child in node.body[:3]]  # First 3 statements
            return f"Func({args}):{','.join(children)}"
        
        elif isinstance(node, ast.If):
            then_branch = self._node_to_skeleton(node.body[0], depth + 1) if node.body else ""
            else_branch = self._node_to_skeleton(node.orelse[0], depth + 1) if node.orelse else ""
            return f"If({then_branch},{else_branch})"
        
        elif isinstance(node, ast.For):
            body = self._node_to_skeleton(node.body[0], depth + 1) if node.body else ""
            return f"For({body})"
        
        elif isinstance(node, ast.While):
            body = self._node_to_skeleton(node.body[0], depth + 1) if node.body else ""
            return f"While({body})"
        
        elif isinstance(node, ast.Return):
            return "Return"
        
        elif isinstance(node, ast.Call):
            n_args = len(node.args)
            return f"Call({n_args})"
        
        else:
            # Generic handling
            children = list(ast.iter_child_nodes(node))
            if children and depth < 5:
                child_skeletons = [self._node_to_skeleton(c, depth + 1) 
                                 for c in children[:2]]  # First 2 children
                return f"{node_type}({','.join(child_skeletons)})"
            return node_type
    
    def compute_similarity(self, skeleton1: str, skeleton2: str) -> float:
        """
        Compute similarity between two skeletons.
        
        Args:
            skeleton1: First skeleton
            skeleton2: Second skeleton
            
        Returns:
            Similarity score in [0, 1]
        """
        return SequenceMatcher(None, skeleton1, skeleton2).ratio()
    
    def match_asts(self, ast1: ast.AST, ast2: ast.AST, 
                  threshold: float = 0.85) -> Tuple[bool, float]:
        """
        Check if two ASTs match structurally.
        
        Args:
            ast1: First AST
            ast2: Second AST
            threshold: Similarity threshold
            
        Returns:
            Tuple of (is_match, similarity_score)
        """
        skeleton1 = self.extract_skeleton(ast1)
        skeleton2 = self.extract_skeleton(ast2)
        
        similarity = self.compute_similarity(skeleton1, skeleton2)
        return similarity >= threshold, similarity


class SimilarityPipeline:
    """
    Complete pipeline for similarity detection.
    
    Combines:
    1. LSH-based candidate retrieval
    2. Hypervector similarity filtering
    3. AST skeleton verification
    4. Confidence scoring
    """
    
    def __init__(self, index: CatalyticIndex, 
                 hv_threshold: float = 0.88,
                 ast_threshold: float = 0.85,
                 max_candidates: int = 100):
        """
        Initialize the pipeline.
        
        Args:
            index: Catalytic index for retrieval
            hv_threshold: Hypervector similarity threshold
            ast_threshold: AST similarity threshold
            max_candidates: Maximum candidates to consider
        """
        self.index = index
        self.hv_threshold = hv_threshold
        self.ast_threshold = ast_threshold
        self.max_candidates = max_candidates
        
        self.encoder = HypervectorEncoder()
        self.ast_matcher = ASTSkeletonMatcher()
        
        # Cache for AST lookups
        self._ast_cache: Dict[str, ast.AST] = {}
        
        self.logger = logging.getLogger(__name__)
    
    def query_function(self, func_ast: ast.AST, 
                      context: Optional[Dict[str, Any]] = None,
                      top_k: int = 10) -> List[QueryResult]:
        """
        Query for similar functions.
        
        Args:
            func_ast: Query function AST
            context: Optional context (imports, class info)
            top_k: Number of results to return
            
        Returns:
            List of query results sorted by similarity
        """
        # Encode query function
        query_hv = self.encoder.encode_function(func_ast, context)
        
        # Get candidates using LSH
        candidate_ids = self.index.query_similar(query_hv, self.max_candidates)
        
        if not candidate_ids:
            return []
        
        # Score candidates
        results = []
        query_skeleton = self.ast_matcher.extract_skeleton(func_ast)
        query_hash = hashlib.md5(query_skeleton.encode()).hexdigest()
        
        for candidate_id in candidate_ids:
            # Get candidate hypervector
            candidate_hv = self.index.get_vector(candidate_id)
            if candidate_hv is None:
                continue
            
            # Compute hypervector similarity
            hv_sim = self.encoder.similarity(query_hv, candidate_hv)
            
            # Pre-filter by hypervector similarity
            if hv_sim < self.hv_threshold:
                continue
            
            # Get metadata
            metadata = self.index.get_metadata(candidate_id)
            if not metadata:
                continue
            
            # Check for exact match by hash
            is_exact = metadata.ast_hash == query_hash
            
            # Compute AST similarity if needed
            ast_sim = 1.0 if is_exact else 0.0
            
            if not is_exact and hv_sim >= self.hv_threshold:
                # Load and compare AST skeletons
                candidate_ast = self._get_ast(metadata.file_path, 
                                            metadata.function_name)
                if candidate_ast:
                    _, ast_sim = self.ast_matcher.match_asts(
                        func_ast, candidate_ast, self.ast_threshold
                    )
            
            # Compute combined score
            similarity_score = 0.7 * hv_sim + 0.3 * ast_sim
            
            # Compute confidence
            confidence = self._compute_confidence(hv_sim, ast_sim, is_exact)
            
            # Create result
            result = QueryResult(
                function_id=candidate_id,
                file_path=metadata.file_path,
                function_name=metadata.function_name,
                line_number=metadata.line_number,
                similarity_score=similarity_score,
                hv_similarity=hv_sim,
                ast_similarity=ast_sim,
                confidence=confidence,
                is_exact_match=is_exact
            )
            
            results.append(result)
        
        # Sort by similarity score
        results.sort(reverse=True)
        
        return results[:top_k]
    
    def batch_query(self, functions: List[Tuple[ast.AST, Dict[str, Any]]],
                   top_k: int = 10) -> Dict[int, List[QueryResult]]:
        """
        Process batch of queries efficiently.
        
        Args:
            functions: List of (ast, context) tuples
            top_k: Results per query
            
        Returns:
            Dictionary mapping query index to results
        """
        results = {}
        
        # Process in batches to reuse computations
        for i, (func_ast, context) in enumerate(functions):
            results[i] = self.query_function(func_ast, context, top_k)
        
        return results
    
    def find_duplicates(self, min_similarity: float = 0.9) -> List[Tuple[str, str, float]]:
        """
        Find all duplicate pairs in the index.
        
        Args:
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of (id1, id2, similarity) tuples
        """
        duplicates = []
        processed = set()
        
        # Iterate through all vectors
        for func_id, hv in self.index.iterate_vectors(batch_size=100):
            if func_id in processed:
                continue
            
            # Query for similar functions
            candidates = self.index.query_similar(hv, self.max_candidates)
            
            for candidate_id in candidates:
                if candidate_id == func_id or candidate_id in processed:
                    continue
                
                # Get candidate vector
                candidate_hv = self.index.get_vector(candidate_id)
                if candidate_hv is None:
                    continue
                
                # Compute similarity
                similarity = self.encoder.similarity(hv, candidate_hv)
                
                if similarity >= min_similarity:
                    duplicates.append((func_id, candidate_id, similarity))
            
            processed.add(func_id)
        
        return duplicates
    
    def _get_ast(self, file_path: str, function_name: str) -> Optional[ast.AST]:
        """
        Get AST for a function from file.
        
        Args:
            file_path: Source file path
            function_name: Function name
            
        Returns:
            Function AST or None
        """
        cache_key = f"{file_path}:{function_name}"
        
        if cache_key in self._ast_cache:
            return self._ast_cache[cache_key]
        
        try:
            with open(file_path, 'r') as f:
                tree = ast.parse(f.read())
            
            # Find the function
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if node.name == function_name:
                        self._ast_cache[cache_key] = node
                        return node
        except Exception as e:
            self.logger.warning(f"Failed to load AST from {file_path}: {e}")
        
        return None
    
    def _compute_confidence(self, hv_sim: float, ast_sim: float, 
                          is_exact: bool) -> float:
        """
        Compute confidence score for a match.
        
        Args:
            hv_sim: Hypervector similarity
            ast_sim: AST skeleton similarity
            is_exact: Whether hash matches exactly
            
        Returns:
            Confidence score in [0, 1]
        """
        if is_exact:
            return 1.0
        
        # Weight factors
        hv_weight = 0.6
        ast_weight = 0.4
        
        # Compute weighted confidence
        confidence = hv_weight * hv_sim + ast_weight * ast_sim
        
        # Boost for very high similarities
        if hv_sim > 0.95 and ast_sim > 0.9:
            confidence = min(1.0, confidence * 1.1)
        
        return confidence
    
    def update_index(self, func_ast: ast.AST, file_path: str, 
                    function_name: str, line_number: int,
                    context: Optional[Dict[str, Any]] = None) -> IndexMetadata:
        """
        Add a new function to the index.
        
        Args:
            func_ast: Function AST
            file_path: Source file
            function_name: Function name
            line_number: Line number
            context: Optional context
            
        Returns:
            Metadata for indexed function
        """
        # Generate hypervector
        hv = self.encoder.encode_function(func_ast, context)
        
        # Generate skeleton hash
        skeleton = self.ast_matcher.extract_skeleton(func_ast)
        ast_hash = hashlib.md5(skeleton.encode()).hexdigest()
        
        # Generate function ID
        function_id = f"{file_path}:{function_name}:{line_number}"
        
        # Add to index
        metadata = self.index.add_function(
            function_id=function_id,
            hypervector=hv,
            file_path=file_path,
            function_name=function_name,
            line_number=line_number,
            ast_hash=ast_hash
        )
        
        return metadata
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        return {
            'index_stats': self.index.get_stats(),
            'cache_size': len(self._ast_cache),
            'hv_threshold': self.hv_threshold,
            'ast_threshold': self.ast_threshold,
            'max_candidates': self.max_candidates
        }