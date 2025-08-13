"""
Enhanced multimodal semantic analysis for detecting semantic duplicates.

This module implements sophisticated semantic analysis using multiple channels
of information to detect semantically similar functions that may not be
structurally identical. It uses vector-based encoding to capture different
aspects of function behavior and semantics.
"""

from __future__ import annotations
import ast
import hashlib
import logging
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set, Optional, Any, Union
from dataclasses import dataclass, field
import numpy as np
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.spatial.distance')
    from scipy.spatial.distance import cosine, euclidean
from scipy import stats

from ..base import BaseAnalyzer, AnalysisContext  
from ...core.issues import Issue
from ...core.utils import safe_get_lineno
from .pattern_types import TailChasingPattern, PatternEvidence, PatternSeverity

logger = logging.getLogger(__name__)


@dataclass
class SemanticChannelEncoding:
    """Represents the encoding of a function across multiple semantic channels."""
    
    function_id: str
    filepath: str
    function_name: str
    line_number: int
    
    # Channel vectors
    data_flow_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    return_patterns_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    error_handling_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    loop_structures_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    control_flow_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Combined vector
    combined_vector: np.ndarray = field(default_factory=lambda: np.array([]))
    
    # Metadata
    channel_weights: Dict[str, float] = field(default_factory=dict)
    encoding_timestamp: Optional[str] = None
    
    def get_channel_vector(self, channel: str) -> np.ndarray:
        """Get vector for a specific channel."""
        channel_map = {
            'data_flow': self.data_flow_vector,
            'return_patterns': self.return_patterns_vector,
            'error_handling': self.error_handling_vector,
            'loop_structures': self.loop_structures_vector,
            'control_flow': self.control_flow_vector
        }
        return channel_map.get(channel, np.array([]))
    
    def set_channel_vector(self, channel: str, vector: np.ndarray) -> None:
        """Set vector for a specific channel."""
        if channel == 'data_flow':
            self.data_flow_vector = vector
        elif channel == 'return_patterns':
            self.return_patterns_vector = vector
        elif channel == 'error_handling':
            self.error_handling_vector = vector
        elif channel == 'loop_structures':
            self.loop_structures_vector = vector
        elif channel == 'control_flow':
            self.control_flow_vector = vector
    
    def is_valid(self) -> bool:
        """Check if the encoding has valid vectors."""
        return (
            len(self.data_flow_vector) > 0 or
            len(self.return_patterns_vector) > 0 or
            len(self.error_handling_vector) > 0 or
            len(self.loop_structures_vector) > 0 or
            len(self.control_flow_vector) > 0
        )


@dataclass
class SemanticSimilarity:
    """Represents similarity between two semantic encodings."""
    
    function1_id: str
    function2_id: str
    overall_similarity: float
    channel_similarities: Dict[str, float] = field(default_factory=dict)
    statistical_significance: Optional[float] = None
    confidence: float = 0.0
    
    def is_significant(self, threshold: float = 0.05) -> bool:
        """Check if similarity is statistically significant."""
        return (self.statistical_significance is not None and 
                self.statistical_significance < threshold)


class SemanticDuplicateEnhancer(BaseAnalyzer):
    """
    Enhanced semantic analysis using multimodal encoding.
    
    This analyzer encodes functions using multiple channels of semantic
    information and uses vector similarity to detect semantic duplicates
    that may not be structurally identical.
    """
    
    name = "multimodal_semantic"
    
    # Default channel weights
    DEFAULT_CHANNEL_WEIGHTS = {
        'data_flow': 0.25,
        'return_patterns': 0.20,
        'error_handling': 0.15,
        'loop_structures': 0.20,
        'control_flow': 0.20
    }
    
    # Vector dimensions for each channel
    CHANNEL_DIMENSIONS = {
        'data_flow': 64,
        'return_patterns': 32,
        'error_handling': 24,
        'loop_structures': 32,
        'control_flow': 48
    }
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # Configure channel weights
        self.channel_weights = self.config.get('channel_weights', self.DEFAULT_CHANNEL_WEIGHTS.copy())
        self._normalize_channel_weights()
        
        # Similarity thresholds
        self.similarity_threshold = self.config.get('similarity_threshold', 0.85)
        self.statistical_significance_threshold = self.config.get('statistical_significance', 0.05)
        
        # Caches for performance
        self._encoding_cache: Dict[str, SemanticChannelEncoding] = {}
        self._similarity_cache: Dict[Tuple[str, str], SemanticSimilarity] = {}
        
        # Background statistics for significance testing
        self._background_similarities: List[float] = []
        
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run multimodal semantic analysis to detect semantic duplicates.
        
        Args:
            ctx: Analysis context containing AST index and configuration
            
        Returns:
            List of Issue objects representing semantic duplicates
        """
        issues = []
        
        try:
            logger.info(f"Running multimodal semantic analysis on {len(ctx.ast_index)} files")
            
            # Extract and encode all functions
            logger.debug("Encoding functions with multimodal semantic vectors")
            function_encodings = self._encode_all_functions(ctx)
            
            if len(function_encodings) < 2:
                logger.debug("Not enough functions for semantic analysis")
                return issues
            
            # Build background distribution for significance testing
            logger.debug("Building background similarity distribution")
            self._build_background_distribution(function_encodings)
            
            # Detect semantic clones
            logger.debug("Detecting semantic clones")
            semantic_clones = self.detect_semantic_clones(function_encodings, ctx)
            
            # Convert to issues
            for clone_group in semantic_clones:
                issue = self._create_semantic_clone_issue(clone_group, ctx)
                if issue:
                    issues.append(issue)
            
            logger.info(f"Multimodal semantic analysis complete: found {len(issues)} semantic clone groups")
            
        except Exception as e:
            logger.error(f"Error in multimodal semantic analysis: {e}", exc_info=True)
            
        return issues
    
    def encode_function_multimodal(
        self, 
        func_node: ast.FunctionDef, 
        filepath: str, 
        function_id: Optional[str] = None
    ) -> SemanticChannelEncoding:
        """
        Encode a function using multiple semantic channels.
        
        Args:
            func_node: AST node of the function
            filepath: Path to the file containing the function
            function_id: Optional unique identifier for the function
            
        Returns:
            SemanticChannelEncoding with vectors for all channels
        """
        if function_id is None:
            function_id = f"{filepath}:{func_node.name}:{safe_get_lineno(func_node)}"
        
        # Check cache first
        if function_id in self._encoding_cache:
            return self._encoding_cache[function_id]
        
        encoding = SemanticChannelEncoding(
            function_id=function_id,
            filepath=filepath,
            function_name=func_node.name,
            line_number=safe_get_lineno(func_node),
            channel_weights=self.channel_weights.copy()
        )
        
        try:
            # Encode each channel
            encoding.data_flow_vector = self._encode_data_flow_channel(func_node)
            encoding.return_patterns_vector = self._encode_return_patterns_channel(func_node)
            encoding.error_handling_vector = self._encode_error_handling_channel(func_node)
            encoding.loop_structures_vector = self._encode_loop_structures_channel(func_node)
            encoding.control_flow_vector = self._encode_control_flow_channel(func_node)
            
            # Combine all channels into a single vector
            encoding.combined_vector = self._combine_channel_vectors(encoding)
            
            # Cache the result
            self._encoding_cache[function_id] = encoding
            
        except Exception as e:
            logger.error(f"Error encoding function {function_id}: {e}", exc_info=True)
            # Return partial encoding
            
        return encoding
    
    def compute_semantic_similarity(
        self, 
        encoding1: SemanticChannelEncoding, 
        encoding2: SemanticChannelEncoding
    ) -> SemanticSimilarity:
        """
        Compute weighted semantic similarity between two function encodings.
        
        Args:
            encoding1: First function encoding
            encoding2: Second function encoding
            
        Returns:
            SemanticSimilarity object with detailed similarity metrics
        """
        cache_key = (encoding1.function_id, encoding2.function_id)
        if cache_key in self._similarity_cache:
            return self._similarity_cache[cache_key]
        
        similarity = SemanticSimilarity(
            function1_id=encoding1.function_id,
            function2_id=encoding2.function_id,
            overall_similarity=0.0
        )
        
        try:
            channel_similarities = {}
            weighted_similarities = []
            
            # Calculate similarity for each channel
            for channel in self.channel_weights.keys():
                vector1 = encoding1.get_channel_vector(channel)
                vector2 = encoding2.get_channel_vector(channel)
                
                if len(vector1) == 0 or len(vector2) == 0:
                    channel_sim = 0.0
                else:
                    # Check for zero vectors to avoid division by zero warning
                    norm1 = np.linalg.norm(vector1)
                    norm2 = np.linalg.norm(vector2)
                    
                    if norm1 == 0 or norm2 == 0:
                        # Zero vector - no meaningful similarity
                        channel_sim = 0.0
                    else:
                        # Use cosine similarity as primary metric
                        try:
                            import warnings
                            with warnings.catch_warnings():
                                warnings.filterwarnings('ignore', category=RuntimeWarning)
                                channel_sim = 1 - cosine(vector1, vector2)
                            if np.isnan(channel_sim):
                                channel_sim = 0.0
                        except:
                            channel_sim = 0.0
                
                channel_similarities[channel] = channel_sim
                weighted_similarities.append(channel_sim * self.channel_weights[channel])
            
            similarity.channel_similarities = channel_similarities
            similarity.overall_similarity = sum(weighted_similarities)
            
            # Calculate statistical significance
            if self._background_similarities:
                similarity.statistical_significance = self._calculate_significance(
                    similarity.overall_similarity
                )
            
            # Calculate confidence based on agreement across channels
            similarity.confidence = self._calculate_confidence(channel_similarities)
            
            # Cache the result
            self._similarity_cache[cache_key] = similarity
            
        except Exception as e:
            logger.error(f"Error computing similarity between {encoding1.function_id} and {encoding2.function_id}: {e}")
            
        return similarity
    
    def detect_semantic_clones(
        self, 
        function_encodings: List[SemanticChannelEncoding], 
        ctx: Optional[AnalysisContext] = None
    ) -> List[List[SemanticChannelEncoding]]:
        """
        Detect groups of semantically similar functions (clones).
        
        Args:
            function_encodings: List of function encodings to analyze
            ctx: Optional analysis context
            
        Returns:
            List of clone groups, where each group contains similar functions
        """
        clone_groups = []
        processed = set()
        
        try:
            for i, encoding1 in enumerate(function_encodings):
                if encoding1.function_id in processed:
                    continue
                
                if not encoding1.is_valid():
                    continue
                
                current_group = [encoding1]
                
                for j, encoding2 in enumerate(function_encodings[i+1:], i+1):
                    if encoding2.function_id in processed:
                        continue
                    
                    if not encoding2.is_valid():
                        continue
                    
                    # Skip if same file and likely the same function
                    if (encoding1.filepath == encoding2.filepath and 
                        abs(encoding1.line_number - encoding2.line_number) < 5):
                        continue
                    
                    # Compute similarity
                    similarity = self.compute_semantic_similarity(encoding1, encoding2)
                    
                    # Check if similar enough to be considered a clone
                    if self._is_semantic_clone(similarity):
                        current_group.append(encoding2)
                        processed.add(encoding2.function_id)
                
                # Only keep groups with multiple functions
                if len(current_group) > 1:
                    clone_groups.append(current_group)
                    processed.add(encoding1.function_id)
                    
                    logger.debug(
                        f"Semantic clone group found: {len(current_group)} functions, "
                        f"primary: {encoding1.function_name}"
                    )
        
        except Exception as e:
            logger.error(f"Error detecting semantic clones: {e}", exc_info=True)
        
        return clone_groups
    
    def find_semantic_duplicates(self, functions: List[Tuple[str, ast.FunctionDef]]) -> List[Issue]:
        """
        Legacy interface for finding semantic duplicates.
        
        Args:
            functions: List of (filepath, function_node) tuples
            
        Returns:
            List of Issue objects
        """
        # Convert to encodings
        encodings = []
        for filepath, func_node in functions:
            encoding = self.encode_function_multimodal(func_node, filepath)
            if encoding.is_valid():
                encodings.append(encoding)
        
        # Detect clones
        clone_groups = self.detect_semantic_clones(encodings)
        
        # Convert to issues
        issues = []
        for group in clone_groups:
            issue = self._create_semantic_clone_issue(group)
            if issue:
                issues.append(issue)
        
        return issues
    
    # Channel encoding methods
    
    def _encode_data_flow_channel(self, func_node: ast.FunctionDef) -> np.ndarray:
        """
        Encode data flow patterns in the function.
        
        Captures:
        - Variable assignments and usage patterns
        - Data transformation chains
        - Input/output relationships
        - Variable scoping patterns
        """
        features = np.zeros(self.CHANNEL_DIMENSIONS['data_flow'])
        
        try:
            # Collect variable operations
            variable_ops = defaultdict(list)
            assignment_patterns = []
            data_transformations = 0
            
            for node in ast.walk(func_node):
                if isinstance(node, ast.Assign):
                    # Track assignment patterns
                    if len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                        var_name = node.targets[0].id
                        variable_ops[var_name].append('assign')
                        
                        # Check for data transformation patterns
                        if isinstance(node.value, ast.Call):
                            data_transformations += 1
                            
                elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    variable_ops[node.id].append('load')
                    
                elif isinstance(node, ast.AugAssign):
                    if isinstance(node.target, ast.Name):
                        variable_ops[node.target.id].append('augassign')
            
            # Encode variable usage patterns
            features[0] = len(variable_ops)  # Number of variables
            features[1] = sum(len(ops) for ops in variable_ops.values())  # Total operations
            features[2] = data_transformations  # Data transformations
            
            # Variable lifecycle patterns
            assignment_to_usage_ratios = []
            for var, ops in variable_ops.items():
                assigns = ops.count('assign') + ops.count('augassign')
                loads = ops.count('load')
                if assigns > 0:
                    assignment_to_usage_ratios.append(loads / assigns)
            
            if assignment_to_usage_ratios:
                features[3] = np.mean(assignment_to_usage_ratios)
                features[4] = np.std(assignment_to_usage_ratios)
            
            # Function parameter usage
            param_usage = set()
            if func_node.args.args:
                param_names = {arg.arg for arg in func_node.args.args}
                for var in variable_ops:
                    if var in param_names:
                        param_usage.add(var)
                        
                features[5] = len(param_usage) / len(param_names) if param_names else 0
            
            # Encode more complex patterns
            self._encode_data_flow_complexity(func_node, features[6:32])
            
            # Normalize features
            features = self._normalize_vector(features)
            
        except Exception as e:
            logger.debug(f"Error encoding data flow channel: {e}")
            
        return features
    
    def _encode_return_patterns_channel(self, func_node: ast.FunctionDef) -> np.ndarray:
        """
        Encode return value patterns.
        
        Captures:
        - Number and types of return statements
        - Return value patterns (constants, variables, expressions)
        - Conditional return logic
        - Early return patterns
        """
        features = np.zeros(self.CHANNEL_DIMENSIONS['return_patterns'])
        
        try:
            return_statements = []
            return_types = defaultdict(int)
            conditional_returns = 0
            early_returns = 0
            
            # Collect return statements with context
            for node in ast.walk(func_node):
                if isinstance(node, ast.Return):
                    return_statements.append(node)
                    
                    # Classify return value type
                    if node.value is None:
                        return_types['none'] += 1
                    elif isinstance(node.value, ast.Constant):
                        return_types['constant'] += 1
                    elif isinstance(node.value, ast.Name):
                        return_types['variable'] += 1
                    elif isinstance(node.value, ast.Call):
                        return_types['function_call'] += 1
                    elif isinstance(node.value, (ast.List, ast.Dict, ast.Tuple)):
                        return_types['collection'] += 1
                    else:
                        return_types['expression'] += 1
            
            # Check for conditional and early returns
            for i, stmt in enumerate(func_node.body):
                if isinstance(stmt, ast.If):
                    if self._contains_return(stmt):
                        conditional_returns += 1
                        # Early return if not the last statement
                        if i < len(func_node.body) - 1:
                            early_returns += 1
                elif isinstance(stmt, ast.Return) and i < len(func_node.body) - 1:
                    early_returns += 1
            
            # Encode features
            features[0] = len(return_statements)
            features[1] = conditional_returns
            features[2] = early_returns
            
            # Return type distribution
            total_returns = len(return_statements) if return_statements else 1
            features[3] = return_types['none'] / total_returns
            features[4] = return_types['constant'] / total_returns
            features[5] = return_types['variable'] / total_returns
            features[6] = return_types['function_call'] / total_returns
            features[7] = return_types['collection'] / total_returns
            features[8] = return_types['expression'] / total_returns
            
            # Return complexity patterns
            self._encode_return_complexity(func_node, features[9:])
            
            # Normalize
            features = self._normalize_vector(features)
            
        except Exception as e:
            logger.debug(f"Error encoding return patterns channel: {e}")
            
        return features
    
    def _encode_error_handling_channel(self, func_node: ast.FunctionDef) -> np.ndarray:
        """
        Encode error handling patterns.
        
        Captures:
        - Try/except block usage
        - Exception types handled
        - Error propagation patterns
        - Assertion usage
        - Input validation patterns
        """
        features = np.zeros(self.CHANNEL_DIMENSIONS['error_handling'])
        
        try:
            try_blocks = 0
            except_handlers = 0
            finally_blocks = 0
            exception_types = set()
            raises = 0
            assertions = 0
            
            for node in ast.walk(func_node):
                if isinstance(node, ast.Try):
                    try_blocks += 1
                    if node.handlers:
                        except_handlers += len(node.handlers)
                        for handler in node.handlers:
                            if handler.type:
                                if isinstance(handler.type, ast.Name):
                                    exception_types.add(handler.type.id)
                    if node.finalbody:
                        finally_blocks += 1
                        
                elif isinstance(node, ast.Raise):
                    raises += 1
                    
                elif isinstance(node, ast.Assert):
                    assertions += 1
            
            # Basic error handling metrics
            features[0] = try_blocks
            features[1] = except_handlers
            features[2] = finally_blocks
            features[3] = raises
            features[4] = assertions
            features[5] = len(exception_types)
            
            # Error handling density
            total_statements = len(list(ast.walk(func_node)))
            if total_statements > 0:
                features[6] = (try_blocks + raises + assertions) / total_statements
            
            # Common exception patterns
            common_exceptions = {'ValueError', 'TypeError', 'KeyError', 'AttributeError', 'IndexError'}
            features[7] = len(exception_types & common_exceptions) / len(common_exceptions)
            
            # Input validation patterns
            self._encode_validation_patterns(func_node, features[8:])
            
            features = self._normalize_vector(features)
            
        except Exception as e:
            logger.debug(f"Error encoding error handling channel: {e}")
            
        return features
    
    def _encode_loop_structures_channel(self, func_node: ast.FunctionDef) -> np.ndarray:
        """
        Encode loop structure patterns.
        
        Captures:
        - For/while loop usage
        - Loop nesting depth
        - Iterator patterns
        - Loop control statements (break, continue)
        - Comprehension usage
        """
        features = np.zeros(self.CHANNEL_DIMENSIONS['loop_structures'])
        
        try:
            for_loops = 0
            while_loops = 0
            max_nesting = 0
            breaks = 0
            continues = 0
            comprehensions = 0
            
            # Track nesting depth
            def count_nesting(node, depth=0):
                nonlocal max_nesting
                max_nesting = max(max_nesting, depth)
                
                for child in ast.walk(node):
                    if isinstance(child, (ast.For, ast.While)):
                        count_nesting(child, depth + 1)
            
            for node in ast.walk(func_node):
                if isinstance(node, ast.For):
                    for_loops += 1
                    count_nesting(node, 1)
                elif isinstance(node, ast.While):
                    while_loops += 1
                    count_nesting(node, 1)
                elif isinstance(node, ast.Break):
                    breaks += 1
                elif isinstance(node, ast.Continue):
                    continues += 1
                elif isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
                    comprehensions += 1
            
            # Encode features
            features[0] = for_loops
            features[1] = while_loops
            features[2] = max_nesting
            features[3] = breaks
            features[4] = continues
            features[5] = comprehensions
            
            # Loop complexity ratio
            total_loops = for_loops + while_loops
            if total_loops > 0:
                features[6] = (breaks + continues) / total_loops
                features[7] = max_nesting / total_loops
            
            # Iterator pattern analysis
            self._encode_iterator_patterns(func_node, features[8:])
            
            features = self._normalize_vector(features)
            
        except Exception as e:
            logger.debug(f"Error encoding loop structures channel: {e}")
            
        return features
    
    def _encode_control_flow_channel(self, func_node: ast.FunctionDef) -> np.ndarray:
        """
        Encode control flow complexity.
        
        Captures:
        - If/elif/else patterns
        - Nesting complexity
        - Cyclomatic complexity
        - Branch patterns
        - Function call patterns
        """
        features = np.zeros(self.CHANNEL_DIMENSIONS['control_flow'])
        
        try:
            if_statements = 0
            elif_branches = 0
            else_branches = 0
            max_if_nesting = 0
            function_calls = 0
            recursive_calls = 0
            
            # Analyze control flow structures
            def analyze_if_depth(node, depth=0):
                nonlocal max_if_nesting
                max_if_nesting = max(max_if_nesting, depth)
                
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, ast.If):
                        analyze_if_depth(child, depth + 1)
            
            for node in ast.walk(func_node):
                if isinstance(node, ast.If):
                    if_statements += 1
                    elif_branches += len(node.orelse) if node.orelse else 0
                    analyze_if_depth(node, 1)
                    
                elif isinstance(node, ast.Call):
                    function_calls += 1
                    if isinstance(node.func, ast.Name) and node.func.id == func_node.name:
                        recursive_calls += 1
            
            # Calculate cyclomatic complexity (simplified)
            cyclomatic_complexity = self._calculate_cyclomatic_complexity(func_node)
            
            # Encode features
            features[0] = if_statements
            features[1] = elif_branches
            features[2] = else_branches
            features[3] = max_if_nesting
            features[4] = function_calls
            features[5] = recursive_calls
            features[6] = cyclomatic_complexity
            
            # Control flow ratios
            if function_calls > 0:
                features[7] = recursive_calls / function_calls
            
            # Branch complexity
            total_statements = len(list(ast.walk(func_node)))
            if total_statements > 0:
                features[8] = (if_statements + elif_branches) / total_statements
            
            # Additional complexity measures
            self._encode_complexity_patterns(func_node, features[9:])
            
            features = self._normalize_vector(features)
            
        except Exception as e:
            logger.debug(f"Error encoding control flow channel: {e}")
            
        return features
    
    # Helper methods
    
    def _encode_all_functions(self, ctx: AnalysisContext) -> List[SemanticChannelEncoding]:
        """Extract and encode all functions in the codebase."""
        encodings = []
        
        for filepath, tree in ctx.ast_index.items():
            if ctx.is_excluded(filepath):
                continue
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Skip very small functions (likely trivial)
                    if len(node.body) < 2:
                        continue
                    
                    encoding = self.encode_function_multimodal(node, filepath)
                    if encoding.is_valid():
                        encodings.append(encoding)
        
        return encodings
    
    def _combine_channel_vectors(self, encoding: SemanticChannelEncoding) -> np.ndarray:
        """Combine all channel vectors into a single weighted vector."""
        combined_parts = []
        
        for channel, weight in self.channel_weights.items():
            vector = encoding.get_channel_vector(channel)
            if len(vector) > 0:
                # Weight and add to combined vector
                weighted_vector = vector * weight
                combined_parts.append(weighted_vector)
        
        if combined_parts:
            return np.concatenate(combined_parts)
        else:
            return np.array([])
    
    def _normalize_channel_weights(self) -> None:
        """Ensure channel weights sum to 1.0."""
        total = sum(self.channel_weights.values())
        if total > 0:
            for channel in self.channel_weights:
                self.channel_weights[channel] /= total
    
    def _normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        """Normalize a vector to unit length."""
        norm = np.linalg.norm(vector)
        if norm > 0:
            return vector / norm
        return vector
    
    def _build_background_distribution(self, encodings: List[SemanticChannelEncoding]) -> None:
        """Build background distribution of similarities for significance testing."""
        if len(encodings) < 10:  # Need sufficient data
            return
        
        # Sample random pairs to build background
        import random
        background_samples = min(1000, len(encodings) * (len(encodings) - 1) // 4)
        
        for _ in range(background_samples):
            i, j = random.sample(range(len(encodings)), 2)
            similarity = self.compute_semantic_similarity(encodings[i], encodings[j])
            self._background_similarities.append(similarity.overall_similarity)
    
    def _calculate_significance(self, similarity: float) -> float:
        """Calculate statistical significance of a similarity score."""
        if not self._background_similarities:
            return 1.0
        
        background_array = np.array(self._background_similarities)
        # Calculate p-value using background distribution
        higher_count = np.sum(background_array >= similarity)
        p_value = higher_count / len(background_array)
        
        return p_value
    
    def _calculate_confidence(self, channel_similarities: Dict[str, float]) -> float:
        """Calculate confidence based on agreement across channels."""
        if not channel_similarities:
            return 0.0
        
        similarities = list(channel_similarities.values())
        
        # High confidence if channels agree
        mean_sim = np.mean(similarities)
        std_sim = np.std(similarities)
        
        # Lower standard deviation indicates better agreement
        if std_sim == 0:
            confidence = mean_sim
        else:
            confidence = mean_sim * (1 - min(std_sim, 0.5))
        
        return min(max(confidence, 0.0), 1.0)
    
    def _is_semantic_clone(self, similarity: SemanticSimilarity) -> bool:
        """Determine if similarity indicates semantic clones."""
        return (similarity.overall_similarity >= self.similarity_threshold and
                similarity.confidence >= 0.6 and
                similarity.is_significant(self.statistical_significance_threshold))
    
    def _create_semantic_clone_issue(
        self, 
        clone_group: List[SemanticChannelEncoding],
        ctx: Optional[AnalysisContext] = None
    ) -> Optional[Issue]:
        """Create an Issue from a semantic clone group."""
        if len(clone_group) < 2:
            return None
        
        # Select primary function (first one)
        primary = clone_group[0]
        others = clone_group[1:]
        
        # Calculate average similarity within group
        similarities = []
        for i in range(len(clone_group)):
            for j in range(i + 1, len(clone_group)):
                sim = self.compute_semantic_similarity(clone_group[i], clone_group[j])
                similarities.append(sim.overall_similarity)
        
        avg_similarity = np.mean(similarities) if similarities else 0.0
        
        # Build evidence
        locations = [(enc.filepath, enc.line_number) for enc in clone_group]
        function_names = [enc.function_name for enc in clone_group]
        
        # Create detailed message
        other_locations = [f"{enc.function_name} ({enc.filepath}:{enc.line_number})" 
                          for enc in others]
        message = (f"Semantic duplicate functions detected: {primary.function_name} "
                  f"is {avg_similarity:.1%} similar to {len(others)} other function(s)")
        
        return Issue(
            kind="enhanced_semantic_duplicate",
            message=message,
            severity=3 if avg_similarity > 0.9 else 2,
            file=primary.filepath,
            line=primary.line_number,
            symbol=primary.function_name,
            evidence={
                "similarity_score": avg_similarity,
                "clone_group_size": len(clone_group),
                "function_names": function_names,
                "locations": locations,
                "channel_weights": self.channel_weights,
                "detection_method": "multimodal_semantic"
            },
            suggestions=[
                f"Consider consolidating semantically similar functions: {', '.join(function_names[:3])}",
                "Extract common logic into shared utility functions",
                "Review if functions serve truly different purposes",
                f"Functions have {avg_similarity:.1%} semantic similarity across multiple channels"
            ],
            confidence=avg_similarity
        )
    
    # Advanced encoding helper methods
    
    def _encode_data_flow_complexity(self, func_node: ast.FunctionDef, features: np.ndarray) -> None:
        """Encode complex data flow patterns."""
        # Variable dependency chains, data transformation patterns, etc.
        # Implementation details for advanced data flow analysis
        pass
    
    def _encode_return_complexity(self, func_node: ast.FunctionDef, features: np.ndarray) -> None:
        """Encode complex return patterns."""
        # Multiple return paths, complex return expressions, etc.
        pass
    
    def _encode_validation_patterns(self, func_node: ast.FunctionDef, features: np.ndarray) -> None:
        """Encode input validation and error checking patterns."""
        # Parameter validation, type checking, etc.
        pass
    
    def _encode_iterator_patterns(self, func_node: ast.FunctionDef, features: np.ndarray) -> None:
        """Encode iterator and iteration patterns."""
        # Enumerate usage, zip patterns, custom iterators, etc.
        pass
    
    def _encode_complexity_patterns(self, func_node: ast.FunctionDef, features: np.ndarray) -> None:
        """Encode additional complexity measures."""
        # Halstead complexity, cognitive complexity, etc.
        pass
    
    def _contains_return(self, node: ast.AST) -> bool:
        """Check if AST node contains return statements."""
        for child in ast.walk(node):
            if isinstance(child, ast.Return):
                return True
        return False
    
    def _calculate_cyclomatic_complexity(self, func_node: ast.FunctionDef) -> int:
        """Calculate simplified cyclomatic complexity."""
        complexity = 1  # Base complexity
        
        for node in ast.walk(func_node):
            if isinstance(node, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(node, ast.ExceptHandler):
                complexity += 1
            elif isinstance(node, ast.With):
                complexity += 1
        
        return complexity