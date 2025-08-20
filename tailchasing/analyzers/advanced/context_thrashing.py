"""
Context Window Thrashing Detection for TailChasingFixer.

This module detects when LLMs reimplement similar functionality due to context
window limitations, resulting in multiple similar functions that could be
consolidated or refactored to use shared utilities.
"""

from __future__ import annotations
import ast
import difflib
import logging
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import hashlib
import statistics

from ..base import BaseAnalyzer, AnalysisContext
from ...core.issues import Issue

logger = logging.getLogger(__name__)


@dataclass
class FunctionInfo:
    """Information about a function for similarity analysis."""
    
    function_id: str
    name: str
    file_path: str
    line_number: int
    end_line_number: Optional[int] = None
    
    # Function signature
    parameters: List[str] = field(default_factory=list)
    parameter_types: List[Optional[str]] = field(default_factory=list)
    return_type: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    
    # Function content
    ast_node: Optional[ast.FunctionDef] = None
    docstring: Optional[str] = None
    body_hash: Optional[str] = None
    
    # Structural metrics
    num_statements: int = 0
    num_branches: int = 0
    num_loops: int = 0
    complexity_score: int = 0
    
    # Similarity cache
    _name_tokens: Optional[List[str]] = None
    _structure_hash: Optional[str] = None
    
    def get_line_count(self) -> int:
        """Get the number of lines in the function."""
        if self.end_line_number:
            return self.end_line_number - self.line_number + 1
        return 1
    
    def get_name_tokens(self) -> List[str]:
        """Get tokenized function name for similarity comparison."""
        if self._name_tokens is None:
            # Split camelCase and snake_case
            name = re.sub(r'([a-z])([A-Z])', r'\1_\2', self.name).lower()
            self._name_tokens = re.findall(r'[a-z]+', name)
        return self._name_tokens
    
    def get_structure_hash(self) -> str:
        """Get a hash representing the function's structure."""
        if self._structure_hash is None:
            if self.ast_node:
                structure_elements = []
                for node in ast.walk(self.ast_node):
                    structure_elements.append(type(node).__name__)
                structure_str = ''.join(sorted(structure_elements))
                self._structure_hash = hashlib.md5(structure_str.encode()).hexdigest()[:16]
            else:
                self._structure_hash = ""
        return self._structure_hash


@dataclass
class SimilarityResult:
    """Result of similarity comparison between two functions."""
    
    function1_id: str
    function2_id: str
    
    # Individual similarity metrics
    name_similarity: float = 0.0
    parameter_similarity: float = 0.0
    structure_similarity: float = 0.0
    content_similarity: float = 0.0
    
    # Combined similarity
    overall_similarity: float = 0.0
    
    # Context metrics
    line_distance: int = 0
    same_file: bool = False
    
    # Classification
    is_likely_reimplementation: bool = False
    confidence: float = 0.0
    
    def __post_init__(self):
        """Calculate overall similarity and classification."""
        # Weighted combination of similarity metrics (structure weighted higher)
        weights = {
            'name': 0.20,
            'parameter': 0.25, 
            'structure': 0.45,
            'content': 0.10
        }
        
        self.overall_similarity = (
            self.name_similarity * weights['name'] +
            self.parameter_similarity * weights['parameter'] +
            self.structure_similarity * weights['structure'] +
            self.content_similarity * weights['content']
        )
        
        # Calculate confidence based on agreement across metrics
        similarities = [self.name_similarity, self.parameter_similarity, 
                       self.structure_similarity, self.content_similarity]
        mean_sim = statistics.mean(similarities)
        std_sim = statistics.stdev(similarities) if len(similarities) > 1 else 0
        
        # Lower standard deviation indicates better agreement
        self.confidence = mean_sim * (1 - min(std_sim * 0.5, 0.3))


@dataclass
class ContextThrashingCluster:
    """A cluster of functions that appear to be reimplementations."""
    
    cluster_id: str
    functions: List[FunctionInfo] = field(default_factory=list)
    similarities: List[SimilarityResult] = field(default_factory=list)
    
    # Cluster metrics
    avg_similarity: float = 0.0
    max_line_distance: int = 0
    min_line_distance: int = 0
    context_window_evidence: float = 0.0
    
    # Pattern evidence
    is_context_thrashing: bool = False
    confidence_score: float = 0.0
    evidence_indicators: List[str] = field(default_factory=list)
    
    def add_function(self, function: FunctionInfo) -> None:
        """Add a function to the cluster."""
        if function not in self.functions:
            self.functions.append(function)
    
    def add_similarity(self, similarity: SimilarityResult) -> None:
        """Add a similarity result to the cluster."""
        self.similarities.append(similarity)
        self._recalculate_metrics()
    
    def _recalculate_metrics(self) -> None:
        """Recalculate cluster metrics."""
        if not self.similarities:
            return
        
        # Calculate average similarity
        self.avg_similarity = statistics.mean(s.overall_similarity for s in self.similarities)
        
        # Calculate line distance metrics
        line_distances = [s.line_distance for s in self.similarities]
        if line_distances:
            self.max_line_distance = max(line_distances)
            self.min_line_distance = min(line_distances)
        
        # Calculate context window evidence
        self._calculate_context_window_evidence()
    
    def _calculate_context_window_evidence(self) -> None:
        """Calculate evidence that this is context window thrashing."""
        evidence_score = 0.0
        indicators = []
        
        # High line distance suggests context window issues (stricter thresholds)
        if self.max_line_distance > 1000:
            evidence_score += 0.4
            indicators.append(f"Functions separated by {self.max_line_distance} lines (likely context window issue)")
        elif self.max_line_distance > 500:
            evidence_score += 0.2
            indicators.append(f"Functions separated by {self.max_line_distance} lines")
        
        # Multiple files with similar functions
        files = set(f.file_path for f in self.functions)
        if len(files) > 1:
            evidence_score += 0.15  # Reduced weight
            indicators.append(f"Similar functions across {len(files)} files")
        
        # High similarity but far apart (stricter threshold)
        if self.avg_similarity > 0.8 and self.max_line_distance > 500:
            evidence_score += 0.3
            indicators.append(f"High similarity ({self.avg_similarity:.1%}) with large separation")
        elif self.avg_similarity > 0.75 and self.max_line_distance > 800:
            evidence_score += 0.2
            indicators.append(f"Moderate similarity ({self.avg_similarity:.1%}) with very large separation")
        
        # Similar parameter patterns
        param_patterns = set()
        for func in self.functions:
            param_pattern = tuple(func.parameters)
            param_patterns.add(param_pattern)
        
        if len(param_patterns) <= len(self.functions) * 0.7:  # 70% or fewer unique patterns
            evidence_score += 0.15
            indicators.append("Similar parameter patterns across functions")
        
        # Name similarity suggesting reimplementation
        name_similarities = [s.name_similarity for s in self.similarities]
        if name_similarities and statistics.mean(name_similarities) > 0.6:
            evidence_score += 0.1
            indicators.append("Similar function names suggesting reimplementation")
        
        self.context_window_evidence = min(1.0, evidence_score)
        self.evidence_indicators = indicators
        
        # Determine if this is likely context thrashing (stricter criteria)
        self.is_context_thrashing = (
            self.context_window_evidence > 0.7 and
            self.avg_similarity > 0.75 and
            self.max_line_distance > 500 and
            len(self.functions) >= 2  # Ensure we have multiple functions
        )
        
        # Calculate confidence
        self.confidence_score = (
            self.avg_similarity * 0.4 +
            self.context_window_evidence * 0.4 +
            (1.0 - min(0.5, statistics.stdev([s.confidence for s in self.similarities]))) * 0.2
        )


class ContextWindowThrashingDetector(BaseAnalyzer):
    """
    Detector for context window thrashing patterns.
    
    Identifies when similar functions are implemented multiple times due to
    LLM context window limitations, suggesting opportunities for consolidation
    or shared utility extraction.
    """
    
    name = "context_window_thrashing"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(self.name)
        self.config = config or {}
        
        # Similarity thresholds (made more strict to reduce false positives)
        self.similarity_threshold = self.config.get('similarity_threshold', 0.75)  # Raised from 0.6
        self.high_similarity_threshold = self.config.get('high_similarity_threshold', 0.85)  # Raised from 0.8
        
        # Line distance thresholds (added configurable minimum)
        self.min_line_distance = self.config.get('min_line_distance', 500)  # Raised from 200
        self.context_window_distance = self.config.get('context_window_distance', 1000)  # Raised from 500
        self.max_line_distance_threshold = self.config.get('max_line_distance_threshold', 2000)  # New limit
        
        # Function filtering
        self.min_function_lines = self.config.get('min_function_lines', 3)
        self.max_function_lines = self.config.get('max_function_lines', 500)
        
        # Analysis parameters (weighted more toward structural similarity)
        self.include_cross_file = self.config.get('include_cross_file', True)
        self.name_similarity_weight = self.config.get('name_similarity_weight', 0.20)  # Reduced
        self.parameter_similarity_weight = self.config.get('parameter_similarity_weight', 0.25)  # Increased
        self.structure_similarity_weight = self.config.get('structure_similarity_weight', 0.45)  # Increased 
        self.content_similarity_weight = self.config.get('content_similarity_weight', 0.10)  # Reduced
        
        # Semantic similarity configuration
        self.require_semantic_similarity = self.config.get('require_semantic_similarity', True)
        self.semantic_similarity_threshold = self.config.get('semantic_similarity_threshold', 0.7)
        self.exclude_simple_functions = self.config.get('exclude_simple_functions', True)
        self.min_complexity_score = self.config.get('min_complexity_score', 3)
        
        # State
        self.functions: Dict[str, FunctionInfo] = {}
        self.similarity_cache: Dict[Tuple[str, str], SimilarityResult] = {}
        
        logger.debug(f"ContextWindowThrashingDetector initialized: "
                    f"similarity_threshold={self.similarity_threshold}, "
                    f"min_line_distance={self.min_line_distance}")
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run context window thrashing detection.
        
        Args:
            ctx: Analysis context containing AST index and configuration
            
        Returns:
            List of Issue objects representing context thrashing patterns
        """
        issues = []
        
        try:
            logger.info(f"Running context window thrashing detection on {len(ctx.ast_index)} files")
            
            # Step 1: Extract all functions
            logger.debug("Extracting functions for analysis")
            self._extract_functions(ctx)
            
            if len(self.functions) < 2:
                logger.debug("Not enough functions for thrashing detection")
                return issues
            
            # Step 2: Calculate similarities between functions
            logger.debug("Calculating function similarities")
            similarities = self._calculate_all_similarities()
            
            # Step 3: Identify clusters of similar functions
            logger.debug("Clustering similar functions")
            clusters = self._cluster_similar_functions(similarities)
            
            # Step 4: Analyze clusters for context thrashing patterns
            logger.debug(f"Analyzing {len(clusters)} clusters for context thrashing")
            thrashing_clusters = []
            
            for cluster in clusters:
                if cluster.is_context_thrashing:
                    thrashing_clusters.append(cluster)
            
            # Step 5: Generate issues
            for cluster in thrashing_clusters:
                issue = self._create_context_thrashing_issue(cluster, ctx)
                if issue:
                    issues.append(issue)
            
            logger.info(f"Context thrashing detection complete: "
                       f"analyzed {len(self.functions)} functions, "
                       f"found {len(clusters)} clusters, "
                       f"detected {len(thrashing_clusters)} thrashing patterns")
            
        except Exception as e:
            logger.error(f"Error in context window thrashing detection: {e}", exc_info=True)
        
        return issues
    
    def _extract_functions(self, ctx: AnalysisContext) -> None:
        """Extract all functions from the codebase."""
        self.functions.clear()
        
        for file_path, tree in ctx.ast_index.items():
            if ctx.is_excluded(file_path):
                continue
            
            try:
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        # Skip very small or very large functions
                        estimated_lines = (getattr(node, 'end_lineno', node.lineno) - 
                                         node.lineno + 1)
                        
                        if (estimated_lines < self.min_function_lines or 
                            estimated_lines > self.max_function_lines):
                            continue
                        
                        # Skip private methods and special methods
                        if node.name.startswith('_'):
                            continue
                        
                        # Skip simple functions if configured to do so
                        if self.exclude_simple_functions:
                            # Calculate basic complexity
                            num_branches = sum(1 for n in ast.walk(node) if isinstance(n, ast.If))
                            num_loops = sum(1 for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)))
                            complexity = num_branches + num_loops + 1
                            
                            if complexity < self.min_complexity_score:
                                continue
                        
                        function_info = self._create_function_info(node, file_path)
                        self.functions[function_info.function_id] = function_info
            
            except Exception as e:
                logger.warning(f"Error extracting functions from {file_path}: {e}")
    
    def _create_function_info(self, node: ast.FunctionDef, file_path: str) -> FunctionInfo:
        """Create FunctionInfo from an AST node."""
        function_id = f"{file_path}:{node.name}:{node.lineno}"
        
        # Extract parameters
        parameters = []
        parameter_types = []
        
        for arg in node.args.args:
            parameters.append(arg.arg)
            if arg.annotation:
                parameter_types.append(ast.unparse(arg.annotation))
            else:
                parameter_types.append(None)
        
        # Extract return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)
        
        # Extract decorators
        decorators = []
        for decorator in node.decorator_list:
            decorators.append(ast.unparse(decorator))
        
        # Extract docstring
        docstring = None
        if (node.body and isinstance(node.body[0], ast.Expr) and 
            isinstance(node.body[0].value, ast.Constant) and 
            isinstance(node.body[0].value.value, str)):
            docstring = node.body[0].value.value
        
        # Calculate structural metrics
        num_statements = len(node.body)
        num_branches = sum(1 for n in ast.walk(node) if isinstance(n, ast.If))
        num_loops = sum(1 for n in ast.walk(node) if isinstance(n, (ast.For, ast.While)))
        complexity_score = num_branches + num_loops + 1
        
        # Generate body hash for content similarity
        body_hash = self._calculate_body_hash(node)
        
        return FunctionInfo(
            function_id=function_id,
            name=node.name,
            file_path=file_path,
            line_number=node.lineno,
            end_line_number=getattr(node, 'end_lineno', node.lineno),
            parameters=parameters,
            parameter_types=parameter_types,
            return_type=return_type,
            decorators=decorators,
            ast_node=node,
            docstring=docstring,
            body_hash=body_hash,
            num_statements=num_statements,
            num_branches=num_branches,
            num_loops=num_loops,
            complexity_score=complexity_score
        )
    
    def _calculate_body_hash(self, node: ast.FunctionDef) -> str:
        """Calculate a hash representing the function body structure."""
        try:
            # Create a normalized representation of the function body
            body_elements = []
            
            for stmt in node.body:
                # Skip docstring
                if (isinstance(stmt, ast.Expr) and 
                    isinstance(stmt.value, ast.Constant) and 
                    isinstance(stmt.value.value, str) and 
                    stmt == node.body[0]):
                    continue
                
                # Collect statement types and key structures
                for n in ast.walk(stmt):
                    body_elements.append(type(n).__name__)
            
            body_str = ''.join(sorted(body_elements))
            return hashlib.md5(body_str.encode()).hexdigest()[:16]
        
        except Exception as e:
            logger.debug(f"Error calculating body hash: {e}")
            return ""
    
    def _calculate_all_similarities(self) -> List[SimilarityResult]:
        """Calculate similarities between all function pairs."""
        similarities = []
        function_list = list(self.functions.values())
        
        for i, func1 in enumerate(function_list):
            for func2 in function_list[i+1:]:
                # Skip if same function
                if func1.function_id == func2.function_id:
                    continue
                
                # Calculate line distance
                if func1.file_path == func2.file_path:
                    line_distance = abs(func1.line_number - func2.line_number)
                    same_file = True
                else:
                    if not self.include_cross_file:
                        continue
                    line_distance = float('inf')  # Different files
                    same_file = False
                
                # Skip if too close (likely related functions) or too far (likely unrelated)
                if same_file and line_distance < self.min_line_distance:
                    continue
                
                # Skip if extremely far apart (likely not context thrashing)
                if same_file and line_distance > self.max_line_distance_threshold:
                    continue
                
                # Calculate similarity
                similarity = self._calculate_function_similarity(func1, func2)
                similarity.line_distance = line_distance if line_distance != float('inf') else 9999
                similarity.same_file = same_file
                
                # Only keep if above threshold
                if similarity.overall_similarity >= self.similarity_threshold:
                    similarities.append(similarity)
        
        return similarities
    
    def _calculate_function_similarity(
        self, 
        func1: FunctionInfo, 
        func2: FunctionInfo
    ) -> SimilarityResult:
        """Calculate similarity between two functions."""
        cache_key = (func1.function_id, func2.function_id)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        similarity = SimilarityResult(
            function1_id=func1.function_id,
            function2_id=func2.function_id
        )
        
        try:
            # Name similarity using difflib
            similarity.name_similarity = self._calculate_name_similarity(func1, func2)
            
            # Parameter similarity
            similarity.parameter_similarity = self._calculate_parameter_similarity(func1, func2)
            
            # Structure similarity (AST-based) - weighted higher
            similarity.structure_similarity = self._calculate_structure_similarity(func1, func2)
            
            # Apply semantic similarity check if required
            if self.require_semantic_similarity:
                semantic_sim = self._calculate_semantic_similarity(func1, func2)
                if semantic_sim < self.semantic_similarity_threshold:
                    # Significantly reduce overall similarity if semantic check fails
                    similarity.structure_similarity *= 0.5
                    similarity.content_similarity *= 0.5
            
            # Content similarity (body hash and complexity)
            similarity.content_similarity = self._calculate_content_similarity(func1, func2)
            
            # Cache the result
            self.similarity_cache[cache_key] = similarity
        
        except Exception as e:
            logger.debug(f"Error calculating similarity between {func1.name} and {func2.name}: {e}")
        
        return similarity
    
    def _calculate_name_similarity(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Calculate name similarity using token-based comparison."""
        try:
            tokens1 = func1.get_name_tokens()
            tokens2 = func2.get_name_tokens()
            
            if not tokens1 or not tokens2:
                return 0.0
            
            # Use difflib for sequence matching
            matcher = difflib.SequenceMatcher(None, tokens1, tokens2)
            token_similarity = matcher.ratio()
            
            # Also compare raw names
            name_matcher = difflib.SequenceMatcher(None, func1.name.lower(), func2.name.lower())
            raw_similarity = name_matcher.ratio()
            
            # Return weighted combination
            return token_similarity * 0.7 + raw_similarity * 0.3
        
        except Exception as e:
            logger.debug(f"Error calculating name similarity: {e}")
            return 0.0
    
    def _calculate_parameter_similarity(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Calculate parameter signature similarity."""
        try:
            # Compare parameter names
            params1 = func1.parameters
            params2 = func2.parameters
            
            if not params1 and not params2:
                return 1.0
            
            if not params1 or not params2:
                return 0.0
            
            # Calculate similarity of parameter lists
            matcher = difflib.SequenceMatcher(None, params1, params2)
            param_name_similarity = matcher.ratio()
            
            # Compare parameter count similarity
            count_similarity = 1.0 - abs(len(params1) - len(params2)) / max(len(params1), len(params2))
            
            # Compare parameter types if available
            type_similarity = 0.5  # Default if no type info
            types1 = func1.parameter_types
            types2 = func2.parameter_types
            
            if types1 and types2 and all(t is not None for t in types1 + types2):
                type_matcher = difflib.SequenceMatcher(None, types1, types2)
                type_similarity = type_matcher.ratio()
            
            # Weighted combination
            return param_name_similarity * 0.5 + count_similarity * 0.3 + type_similarity * 0.2
        
        except Exception as e:
            logger.debug(f"Error calculating parameter similarity: {e}")
            return 0.0
    
    def _calculate_structure_similarity(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Calculate AST structure similarity."""
        try:
            # Compare structure hashes
            hash1 = func1.get_structure_hash()
            hash2 = func2.get_structure_hash()
            
            if not hash1 or not hash2:
                return 0.0
            
            if hash1 == hash2:
                return 1.0
            
            # Compare structural metrics
            metrics_similarity = 0.0
            
            # Statement count similarity
            stmt_sim = 1.0 - abs(func1.num_statements - func2.num_statements) / max(func1.num_statements, func2.num_statements)
            
            # Branch count similarity
            if func1.num_branches == 0 and func2.num_branches == 0:
                branch_sim = 1.0
            elif func1.num_branches == 0 or func2.num_branches == 0:
                branch_sim = 0.0
            else:
                branch_sim = 1.0 - abs(func1.num_branches - func2.num_branches) / max(func1.num_branches, func2.num_branches)
            
            # Loop count similarity
            if func1.num_loops == 0 and func2.num_loops == 0:
                loop_sim = 1.0
            elif func1.num_loops == 0 or func2.num_loops == 0:
                loop_sim = 0.0
            else:
                loop_sim = 1.0 - abs(func1.num_loops - func2.num_loops) / max(func1.num_loops, func2.num_loops)
            
            # Complexity similarity
            complexity_sim = 1.0 - abs(func1.complexity_score - func2.complexity_score) / max(func1.complexity_score, func2.complexity_score)
            
            metrics_similarity = (stmt_sim * 0.4 + branch_sim * 0.25 + loop_sim * 0.25 + complexity_sim * 0.1)
            
            # Compare actual AST structure if available
            ast_similarity = 0.5  # Default
            if func1.ast_node and func2.ast_node:
                ast_similarity = self._compare_ast_structure(func1.ast_node, func2.ast_node)
            
            # Weight structural similarity more heavily
            return metrics_similarity * 0.7 + ast_similarity * 0.3
        
        except Exception as e:
            logger.debug(f"Error calculating structure similarity: {e}")
            return 0.0
    
    def _calculate_content_similarity(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Calculate content similarity using body hashes and other metrics."""
        try:
            # Body hash similarity
            if func1.body_hash and func2.body_hash:
                if func1.body_hash == func2.body_hash:
                    hash_similarity = 1.0
                else:
                    # Compare hash prefixes for partial similarity
                    common_prefix = 0
                    for i in range(min(len(func1.body_hash), len(func2.body_hash))):
                        if func1.body_hash[i] == func2.body_hash[i]:
                            common_prefix += 1
                        else:
                            break
                    hash_similarity = common_prefix / max(len(func1.body_hash), len(func2.body_hash))
            else:
                hash_similarity = 0.0
            
            # Docstring similarity
            docstring_similarity = 0.5  # Default
            if func1.docstring and func2.docstring:
                doc_matcher = difflib.SequenceMatcher(None, func1.docstring, func2.docstring)
                docstring_similarity = doc_matcher.ratio()
            elif not func1.docstring and not func2.docstring:
                docstring_similarity = 1.0
            
            # Line count similarity
            lines1 = func1.get_line_count()
            lines2 = func2.get_line_count()
            line_similarity = 1.0 - abs(lines1 - lines2) / max(lines1, lines2)
            
            return hash_similarity * 0.5 + docstring_similarity * 0.3 + line_similarity * 0.2
        
        except Exception as e:
            logger.debug(f"Error calculating content similarity: {e}")
            return 0.0
    
    def _calculate_semantic_similarity(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Calculate semantic similarity between functions using more sophisticated analysis."""
        try:
            # Check for similar variable usage patterns
            var_similarity = self._compare_variable_patterns(func1, func2)
            
            # Check for similar call patterns
            call_similarity = self._compare_function_call_patterns(func1, func2)
            
            # Check for similar control flow patterns
            control_similarity = self._compare_control_flow_patterns(func1, func2)
            
            # Check for similar literal usage
            literal_similarity = self._compare_literal_patterns(func1, func2)
            
            # Weighted combination emphasizing behavioral similarity
            return (
                var_similarity * 0.3 +
                call_similarity * 0.35 + 
                control_similarity * 0.25 +
                literal_similarity * 0.1
            )
            
        except Exception as e:
            logger.debug(f"Error calculating semantic similarity: {e}")
            return 0.5  # Default neutral value
    
    def _compare_variable_patterns(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Compare variable usage patterns between functions."""
        if not func1.ast_node or not func2.ast_node:
            return 0.5
        
        try:
            # Extract variable names from both functions
            vars1 = set()
            vars2 = set()
            
            for node in ast.walk(func1.ast_node):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    vars1.add(node.id)
                elif isinstance(node, ast.arg):
                    vars1.add(node.arg)
            
            for node in ast.walk(func2.ast_node):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store):
                    vars2.add(node.id)
                elif isinstance(node, ast.arg):
                    vars2.add(node.arg)
            
            if not vars1 and not vars2:
                return 1.0
            if not vars1 or not vars2:
                return 0.0
            
            # Calculate Jaccard similarity
            intersection = len(vars1.intersection(vars2))
            union = len(vars1.union(vars2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Error comparing variable patterns: {e}")
            return 0.5
    
    def _compare_function_call_patterns(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Compare function call patterns between functions."""
        if not func1.ast_node or not func2.ast_node:
            return 0.5
        
        try:
            # Extract function call names
            calls1 = []
            calls2 = []
            
            for node in ast.walk(func1.ast_node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls1.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls1.append(node.func.attr)
            
            for node in ast.walk(func2.ast_node):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls2.append(node.func.id)
                    elif isinstance(node.func, ast.Attribute):
                        calls2.append(node.func.attr)
            
            if not calls1 and not calls2:
                return 1.0
            if not calls1 or not calls2:
                return 0.0
            
            # Use sequence similarity for call patterns
            matcher = difflib.SequenceMatcher(None, calls1, calls2)
            return matcher.ratio()
            
        except Exception as e:
            logger.debug(f"Error comparing function call patterns: {e}")
            return 0.5
    
    def _compare_control_flow_patterns(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Compare control flow structures between functions."""
        if not func1.ast_node or not func2.ast_node:
            return 0.5
        
        try:
            # Extract control flow patterns
            pattern1 = []
            pattern2 = []
            
            for node in ast.walk(func1.ast_node):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    pattern1.append(type(node).__name__)
                elif isinstance(node, ast.Return):
                    pattern1.append('Return')
                elif isinstance(node, ast.Break):
                    pattern1.append('Break')
                elif isinstance(node, ast.Continue):
                    pattern1.append('Continue')
            
            for node in ast.walk(func2.ast_node):
                if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.With)):
                    pattern2.append(type(node).__name__)
                elif isinstance(node, ast.Return):
                    pattern2.append('Return')
                elif isinstance(node, ast.Break):
                    pattern2.append('Break')
                elif isinstance(node, ast.Continue):
                    pattern2.append('Continue')
            
            if not pattern1 and not pattern2:
                return 1.0
            if not pattern1 or not pattern2:
                return 0.0
            
            # Compare patterns
            matcher = difflib.SequenceMatcher(None, pattern1, pattern2)
            return matcher.ratio()
            
        except Exception as e:
            logger.debug(f"Error comparing control flow patterns: {e}")
            return 0.5
    
    def _compare_literal_patterns(self, func1: FunctionInfo, func2: FunctionInfo) -> float:
        """Compare literal usage patterns between functions."""
        if not func1.ast_node or not func2.ast_node:
            return 0.5
        
        try:
            # Extract literals (excluding strings which might be very specific)
            literals1 = []
            literals2 = []
            
            for node in ast.walk(func1.ast_node):
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, (int, float, bool)) and not isinstance(node.value, str):
                        literals1.append(repr(node.value))
            
            for node in ast.walk(func2.ast_node):
                if isinstance(node, ast.Constant):
                    if isinstance(node.value, (int, float, bool)) and not isinstance(node.value, str):
                        literals2.append(repr(node.value))
            
            if not literals1 and not literals2:
                return 1.0
            if not literals1 or not literals2:
                return 0.0
            
            # Compare literal usage
            set1 = set(literals1)
            set2 = set(literals2)
            
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            
            return intersection / union if union > 0 else 0.0
            
        except Exception as e:
            logger.debug(f"Error comparing literal patterns: {e}")
            return 0.5
    
    def _compare_ast_structure(self, node1: ast.FunctionDef, node2: ast.FunctionDef) -> float:
        """Compare AST structure of two function nodes."""
        try:
            # Get flattened node type sequences
            sequence1 = [type(node).__name__ for node in ast.walk(node1)]
            sequence2 = [type(node).__name__ for node in ast.walk(node2)]
            
            # Use sequence matcher
            matcher = difflib.SequenceMatcher(None, sequence1, sequence2)
            return matcher.ratio()
        
        except Exception as e:
            logger.debug(f"Error comparing AST structure: {e}")
            return 0.0
    
    def _cluster_similar_functions(self, similarities: List[SimilarityResult]) -> List[ContextThrashingCluster]:
        """Cluster similar functions into groups."""
        clusters = []
        processed_functions = set()
        
        # Sort similarities by overall similarity (highest first)
        similarities.sort(key=lambda s: s.overall_similarity, reverse=True)
        
        for similarity in similarities:
            func1_id = similarity.function1_id
            func2_id = similarity.function2_id
            
            # Skip if both functions are already processed
            if func1_id in processed_functions and func2_id in processed_functions:
                continue
            
            # Find existing cluster for either function
            target_cluster = None
            for cluster in clusters:
                cluster_func_ids = {f.function_id for f in cluster.functions}
                if func1_id in cluster_func_ids or func2_id in cluster_func_ids:
                    target_cluster = cluster
                    break
            
            # Create new cluster if none found
            if target_cluster is None:
                cluster_id = f"cluster_{len(clusters) + 1}"
                target_cluster = ContextThrashingCluster(cluster_id=cluster_id)
                clusters.append(target_cluster)
            
            # Add functions to cluster
            if func1_id not in {f.function_id for f in target_cluster.functions}:
                target_cluster.add_function(self.functions[func1_id])
                processed_functions.add(func1_id)
            
            if func2_id not in {f.function_id for f in target_cluster.functions}:
                target_cluster.add_function(self.functions[func2_id])
                processed_functions.add(func2_id)
            
            # Add similarity to cluster
            target_cluster.add_similarity(similarity)
        
        # Filter out clusters with only one function
        clusters = [c for c in clusters if len(c.functions) > 1]
        
        return clusters
    
    def _create_context_thrashing_issue(
        self, 
        cluster: ContextThrashingCluster, 
        ctx: AnalysisContext
    ) -> Optional[Issue]:
        """Create an Issue object for a context thrashing cluster."""
        try:
            if not cluster.functions:
                return None
            
            primary_function = cluster.functions[0]
            
            # Build description
            function_names = [f.name for f in cluster.functions]
            unique_files = len(set(f.file_path for f in cluster.functions))
            
            description = (
                f"Context window thrashing detected: {len(cluster.functions)} similar functions "
                f"with {cluster.avg_similarity:.1%} average similarity, "
                f"separated by up to {cluster.max_line_distance} lines"
            )
            
            # Build evidence
            evidence = {
                'cluster_id': cluster.cluster_id,
                'function_count': len(cluster.functions),
                'function_names': function_names,
                'average_similarity': cluster.avg_similarity,
                'max_line_distance': cluster.max_line_distance,
                'min_line_distance': cluster.min_line_distance,
                'files_affected': unique_files,
                'context_window_evidence': cluster.context_window_evidence,
                'confidence_score': cluster.confidence_score,
                'evidence_indicators': cluster.evidence_indicators,
                'function_locations': [
                    {
                        'name': f.name,
                        'file': f.file_path,
                        'line': f.line_number,
                        'lines': f.get_line_count()
                    }
                    for f in cluster.functions
                ]
            }
            
            # Add similarity matrix
            evidence['similarity_matrix'] = {}
            for sim in cluster.similarities:
                key = f"{self.functions[sim.function1_id].name} <-> {self.functions[sim.function2_id].name}"
                evidence['similarity_matrix'][key] = {
                    'overall': sim.overall_similarity,
                    'name': sim.name_similarity,
                    'parameters': sim.parameter_similarity,
                    'structure': sim.structure_similarity,
                    'content': sim.content_similarity,
                    'line_distance': sim.line_distance
                }
            
            # Generate actionable suggestions
            suggestions = self._generate_suggestions(cluster)
            
            # Determine severity
            severity = 2  # Warning
            if cluster.avg_similarity > 0.8 and cluster.max_line_distance > self.context_window_distance:
                severity = 3  # High
            if cluster.avg_similarity > 0.9 and len(cluster.functions) > 3:
                severity = 4  # Critical
            
            return Issue(
                kind="context_window_thrashing",
                message=description,
                severity=severity,
                file=primary_function.file_path,
                line=primary_function.line_number,
                symbol=primary_function.name,
                confidence=cluster.confidence_score,
                evidence=evidence,
                suggestions=suggestions
            )
        
        except Exception as e:
            logger.error(f"Error creating context thrashing issue: {e}", exc_info=True)
            return None
    
    def _generate_suggestions(self, cluster: ContextThrashingCluster) -> List[str]:
        """Generate actionable suggestions for resolving context thrashing."""
        suggestions = []
        
        function_names = [f.name for f in cluster.functions]
        unique_files = len(set(f.file_path for f in cluster.functions))
        
        # Basic consolidation suggestions
        if cluster.avg_similarity > 0.8:
            suggestions.append(f"Consider consolidating highly similar functions: {', '.join(function_names[:3])}")
        
        # Extract common functionality
        if len(cluster.functions) > 2:
            suggestions.append(f"Extract common logic from {len(cluster.functions)} similar functions into a shared utility")
        
        # Parameter harmonization
        param_counts = [len(f.parameters) for f in cluster.functions]
        if len(set(param_counts)) > 1:
            suggestions.append("Harmonize parameter signatures to enable consolidation")
        
        # Cross-file suggestions
        if unique_files > 1:
            suggestions.append(f"Functions span {unique_files} files - consider creating a shared module")
        
        # Context window specific suggestions
        if cluster.max_line_distance > self.context_window_distance:
            suggestions.append("Large separation suggests context window limitations - review implementation strategy")
        
        # Specific refactoring suggestions
        if cluster.avg_similarity > 0.7:
            suggestions.append("Refactor to use strategy pattern or template method to reduce duplication")
        
        # Documentation and naming
        name_similarities = []
        for sim in cluster.similarities:
            name_similarities.append(sim.name_similarity)
        
        if name_similarities and statistics.mean(name_similarities) > 0.6:
            suggestions.append("Similar naming patterns suggest intentional reimplementation - review necessity")
        
        # Specific metrics-based suggestions
        if cluster.context_window_evidence > 0.8:
            suggestions.append("High context window evidence - consider breaking down into smaller, reusable components")
        
        return suggestions


# Alias for backward compatibility
ContextThrashingAnalyzer = ContextWindowThrashingDetector


# Public API exports
__all__ = [
    # Main analyzer class
    'ContextWindowThrashingDetector',
    'ContextThrashingAnalyzer',  # Backward compatibility alias
    
    # Data structures 
    'FunctionInfo',
    'SimilarityResult',
    'ContextThrashingCluster',
]
