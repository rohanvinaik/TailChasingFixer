"""
Adaptive algorithm selection for optimized code analysis.

This module implements intelligent selection between different analysis strategies
based on code complexity, pattern familiarity, and influence scores. Inspired by
the strategic use of "slower" algorithms like Bellman-Ford for limited exploration.
"""

import ast
import time
import random
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from enum import Enum
import numpy as np

from ..semantic.encoder import encode_function
from ..utils.logging_setup import get_logger
from .influence_detector import InfluenceDetector


class AnalysisStrategy(Enum):
    """Available analysis strategies."""
    FAST = "fast"      # Pattern matching only
    DEEP = "deep"      # Full semantic analysis
    HYBRID = "hybrid"  # Scout with DEEP, apply FAST


@dataclass
class AnalysisResult:
    """Result from adaptive analysis."""
    strategy_used: AnalysisStrategy
    patterns_found: List[Dict[str, Any]]
    execution_time: float
    confidence: float
    samples_analyzed: int = 0
    total_nodes: int = 0
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PatternCache:
    """Cache for familiar patterns."""
    pattern_signature: str
    occurrence_count: int = 0
    last_seen: float = 0.0
    avg_analysis_time: float = 0.0
    success_rate: float = 0.0
    recommended_strategy: Optional[AnalysisStrategy] = None


class AdaptiveAlgorithmSelector:
    """
    Selects optimal analysis algorithm based on code context.
    
    Uses different strategies:
    - FAST: Pattern matching for simple, familiar patterns
    - DEEP: Full semantic analysis for complex patterns
    - HYBRID: Scout with DEEP, then apply FAST with guidance
    
    Inspired by selective use of Bellman-Ford for limited steps.
    """
    
    # Thresholds for strategy selection
    COMPLEXITY_THRESHOLD = 10  # Cyclomatic complexity
    INFLUENCE_THRESHOLD = 0.6   # For HYBRID strategy
    FAMILIARITY_THRESHOLD = 3   # Times seen before
    SAMPLE_RATIO = 0.1          # Sample 10% for scouting
    MAX_SAMPLES = 50            # Maximum samples for deep analysis
    SCOUT_DEPTH = 2             # Limited exploration depth
    
    def __init__(self, 
                 influence_detector: Optional[InfluenceDetector] = None,
                 enable_caching: bool = True):
        """
        Initialize the adaptive selector.
        
        Args:
            influence_detector: Optional influence detector for scoring
            enable_caching: Whether to cache pattern familiarity
        """
        self.logger = get_logger("adaptive_selector")
        self.influence_detector = influence_detector or InfluenceDetector()
        self.enable_caching = enable_caching
        
        # Pattern familiarity cache
        self.pattern_cache: Dict[str, PatternCache] = {}
        
        # Performance metrics
        self.performance_metrics = defaultdict(lambda: {
            'count': 0,
            'total_time': 0.0,
            'avg_time': 0.0,
            'patterns_found': 0
        })
        
        # Semantic analysis cache
        self._semantic_cache: Dict[str, np.ndarray] = {}
        
    def analyze_with_adaptive_algorithms(self, 
                                        code_section: Union[ast.AST, List[ast.AST]],
                                        context: Optional[Dict[str, Any]] = None) -> AnalysisResult:
        """
        Analyze code section with adaptively chosen algorithm.
        
        Args:
            code_section: AST node(s) to analyze
            context: Additional context for analysis
            
        Returns:
            AnalysisResult with patterns and metrics
        """
        start_time = time.time()
        
        # Normalize input
        if not isinstance(code_section, list):
            code_section = [code_section]
        
        # Determine optimal strategy
        strategy = self._select_strategy(code_section, context)
        
        self.logger.debug(f"Selected strategy: {strategy.value} for {len(code_section)} nodes")
        
        # Execute chosen strategy
        if strategy == AnalysisStrategy.FAST:
            result = self._fast_pattern_matching(code_section)
            
        elif strategy == AnalysisStrategy.DEEP:
            result = self._deep_semantic_analysis(
                code_section,
                max_iterations=3  # Limited iterations
            )
            
        else:  # HYBRID
            # Scout with deep analysis
            influential_patterns = self.scout_with_deep_analysis(
                code_section,
                scout_depth=self.SCOUT_DEPTH
            )
            
            # Apply fast analysis with guidance
            result = self._guided_fast_analysis(
                code_section,
                guide_patterns=influential_patterns
            )
        
        # Record metrics
        execution_time = time.time() - start_time
        result.execution_time = execution_time
        result.strategy_used = strategy
        result.total_nodes = len(code_section)
        
        self._update_metrics(strategy, execution_time, len(result.patterns_found))
        
        self.logger.info(
            f"Analysis complete: {strategy.value} strategy, "
            f"{len(result.patterns_found)} patterns, {execution_time:.3f}s"
        )
        
        return result
    
    def _select_strategy(self, 
                        code_section: List[ast.AST],
                        context: Optional[Dict[str, Any]]) -> AnalysisStrategy:
        """
        Select optimal analysis strategy based on code characteristics.
        
        Selection criteria:
        1. Pattern familiarity → FAST
        2. High complexity → DEEP
        3. High influence → HYBRID
        4. Default → FAST
        """
        # Check pattern familiarity
        if self._is_familiar_pattern(code_section):
            self.logger.debug("Familiar pattern detected → FAST strategy")
            return AnalysisStrategy.FAST
        
        # Calculate complexity
        complexity = self._calculate_complexity(code_section)
        if complexity > self.COMPLEXITY_THRESHOLD:
            self.logger.debug(f"High complexity ({complexity}) → DEEP strategy")
            return AnalysisStrategy.DEEP
        
        # Check influence score
        influence = self._calculate_influence(code_section, context)
        if influence > self.INFLUENCE_THRESHOLD:
            self.logger.debug(f"High influence ({influence:.2f}) → HYBRID strategy")
            return AnalysisStrategy.HYBRID
        
        # Default to FAST
        return AnalysisStrategy.FAST
    
    def _is_familiar_pattern(self, code_section: List[ast.AST]) -> bool:
        """Check if pattern has been seen before."""
        if not self.enable_caching:
            return False
        
        for node in code_section[:3]:  # Check first few nodes
            pattern_sig = self._generate_pattern_signature(node)
            if pattern_sig in self.pattern_cache:
                cache_entry = self.pattern_cache[pattern_sig]
                if cache_entry.occurrence_count >= self.FAMILIARITY_THRESHOLD:
                    return True
        
        return False
    
    def _calculate_complexity(self, code_section: List[ast.AST]) -> int:
        """
        Calculate cyclomatic complexity of code section.
        
        Simplified McCabe complexity calculation.
        """
        complexity = 1  # Base complexity
        
        for node in code_section:
            for child in ast.walk(node):
                # Decision points increase complexity
                if isinstance(child, (ast.If, ast.While, ast.For)):
                    complexity += 1
                elif isinstance(child, ast.BoolOp):
                    # Each boolean operator adds a path
                    complexity += len(child.values) - 1
                elif isinstance(child, ast.ExceptHandler):
                    complexity += 1
                elif isinstance(child, ast.With):
                    complexity += 1
                elif isinstance(child, ast.Assert):
                    complexity += 1
        
        return complexity
    
    def _calculate_influence(self, 
                            code_section: List[ast.AST],
                            context: Optional[Dict[str, Any]]) -> float:
        """Calculate influence score for code section."""
        if not code_section:
            return 0.0
        
        total_influence = 0.0
        
        for node in code_section[:5]:  # Sample first few
            # Use influence detector if available
            node_id = f"temp:{id(node)}"
            influence = self.influence_detector.scout_influence_bellman_ford_style(
                node_id, node, max_steps=2
            )
            total_influence += influence
        
        # Normalize
        avg_influence = total_influence / min(5, len(code_section))
        
        # Adjust based on context
        if context:
            if context.get('is_public_api', False):
                avg_influence *= 1.5
            if context.get('has_tests', False):
                avg_influence *= 0.8
        
        return min(avg_influence / 100.0, 1.0)  # Normalize to 0-1
    
    def _fast_pattern_matching(self, code_section: List[ast.AST]) -> AnalysisResult:
        """
        Fast pattern matching using simple heuristics.
        
        This is like Dijkstra - fast but may miss complex patterns.
        """
        patterns_found = []
        
        for node in code_section:
            # Quick pattern checks
            pattern = self._match_simple_patterns(node)
            if pattern:
                patterns_found.append(pattern)
                
                # Update cache
                if self.enable_caching:
                    self._update_pattern_cache(pattern['signature'], AnalysisStrategy.FAST)
        
        return AnalysisResult(
            strategy_used=AnalysisStrategy.FAST,
            patterns_found=patterns_found,
            execution_time=0.0,
            confidence=0.7,  # Lower confidence for fast matching
            total_nodes=len(code_section)
        )
    
    def _match_simple_patterns(self, node: ast.AST) -> Optional[Dict[str, Any]]:
        """Match common simple patterns."""
        pattern = None
        
        if isinstance(node, ast.FunctionDef):
            # Check for common anti-patterns
            if self._is_empty_function(node):
                pattern = {
                    'type': 'empty_function',
                    'name': node.name,
                    'signature': self._generate_pattern_signature(node)
                }
            elif self._is_getter_setter(node):
                pattern = {
                    'type': 'getter_setter',
                    'name': node.name,
                    'signature': self._generate_pattern_signature(node)
                }
            elif self._has_duplicate_logic(node):
                pattern = {
                    'type': 'duplicate_logic',
                    'name': node.name,
                    'signature': self._generate_pattern_signature(node)
                }
        
        elif isinstance(node, ast.ClassDef):
            if self._is_empty_class(node):
                pattern = {
                    'type': 'empty_class',
                    'name': node.name,
                    'signature': self._generate_pattern_signature(node)
                }
        
        return pattern
    
    def _is_empty_function(self, node: ast.FunctionDef) -> bool:
        """Check if function is empty or trivial."""
        if not node.body:
            return True
        
        # Check for pass or ellipsis only
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Pass):
                return True
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Ellipsis):
                return True
        
        return False
    
    def _is_getter_setter(self, node: ast.FunctionDef) -> bool:
        """Check if function is a simple getter/setter."""
        name = node.name.lower()
        
        # Check name pattern
        if not (name.startswith(('get_', 'set_')) or 
                name.startswith(('is_', 'has_'))):
            return False
        
        # Check for simple body
        if len(node.body) == 1:
            stmt = node.body[0]
            if isinstance(stmt, ast.Return):
                return True
            if isinstance(stmt, ast.Assign):
                return True
        
        return False
    
    def _has_duplicate_logic(self, node: ast.FunctionDef) -> bool:
        """Quick check for potential duplicate logic."""
        # This is a simplified check
        sig = self._generate_pattern_signature(node)
        
        if sig in self.pattern_cache:
            return self.pattern_cache[sig].occurrence_count > 1
        
        return False
    
    def _is_empty_class(self, node: ast.ClassDef) -> bool:
        """Check if class is empty or trivial."""
        if not node.body:
            return True
        
        # Check for pass only
        if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
            return True
        
        return False
    
    def _deep_semantic_analysis(self, 
                               code_section: List[ast.AST],
                               max_iterations: int = 3) -> AnalysisResult:
        """
        Deep semantic analysis using hypervectors.
        
        This is like Bellman-Ford - thorough but expensive.
        Limited iterations prevent excessive computation.
        """
        patterns_found = []
        samples_analyzed = 0
        
        # Limit analysis to prevent excessive computation
        nodes_to_analyze = code_section[:max_iterations * 10]
        
        for node in nodes_to_analyze:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Expensive hypervector encoding
                vector = self._get_or_compute_vector(node)
                
                if vector is not None:
                    # Find semantic patterns
                    similar_patterns = self._find_semantic_patterns(vector, node)
                    patterns_found.extend(similar_patterns)
                    samples_analyzed += 1
                
                # Limit iterations
                if samples_analyzed >= max_iterations * 3:
                    break
        
        return AnalysisResult(
            strategy_used=AnalysisStrategy.DEEP,
            patterns_found=patterns_found,
            execution_time=0.0,
            confidence=0.9,  # Higher confidence for deep analysis
            samples_analyzed=samples_analyzed,
            total_nodes=len(code_section)
        )
    
    def _get_or_compute_vector(self, node: ast.AST) -> Optional[np.ndarray]:
        """Get or compute semantic vector for node."""
        cache_key = ast.dump(node)[:100]  # Use truncated dump as key
        
        if cache_key in self._semantic_cache:
            return self._semantic_cache[cache_key]
        
        try:
            vector = encode_function(node)
            self._semantic_cache[cache_key] = vector
            return vector
        except Exception as e:
            self.logger.debug(f"Failed to encode node: {e}")
            return None
    
    def _find_semantic_patterns(self, 
                               vector: np.ndarray,
                               node: ast.AST) -> List[Dict[str, Any]]:
        """Find patterns using semantic similarity."""
        patterns = []
        
        # Compare with cached vectors
        for cached_key, cached_vector in list(self._semantic_cache.items())[:20]:
            similarity = self._calculate_similarity(vector, cached_vector)
            
            if similarity > 0.8 and cached_key != ast.dump(node)[:100]:
                patterns.append({
                    'type': 'semantic_duplicate',
                    'similarity': similarity,
                    'signature': self._generate_pattern_signature(node)
                })
        
        return patterns
    
    def _calculate_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity between vectors."""
        dot_product = np.dot(v1, v2)
        norm_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        
        if norm_product == 0:
            return 0.0
        
        return dot_product / norm_product
    
    def scout_with_deep_analysis(self, 
                                code_section: List[ast.AST],
                                scout_depth: int = 2) -> List[Dict[str, Any]]:
        """
        Scout with expensive analysis on samples only.
        
        Like using Bellman-Ford for just a few steps to find key patterns.
        
        Args:
            code_section: Nodes to analyze
            scout_depth: Maximum scouting iterations
            
        Returns:
            List of influential patterns found
        """
        self.logger.debug(f"Scouting {len(code_section)} nodes with depth {scout_depth}")
        
        scouting_results = []
        
        # Select representative samples
        sample_size = min(
            max(1, int(len(code_section) * self.SAMPLE_RATIO)),
            self.MAX_SAMPLES
        )
        samples = self._select_representative_samples(code_section, sample_size)
        
        # Run deep analysis on samples only
        for i, sample in enumerate(samples):
            if i >= scout_depth * 3:  # Limit total samples
                break
            
            # Deep but limited analysis
            if isinstance(sample, (ast.FunctionDef, ast.AsyncFunctionDef)):
                vector = self._get_or_compute_vector(sample)
                
                if vector is not None:
                    # Calculate influence
                    node_id = f"scout:{id(sample)}"
                    influence = self.influence_detector.scout_influence_bellman_ford_style(
                        node_id, sample, max_steps=scout_depth
                    )
                    
                    if influence > self.INFLUENCE_THRESHOLD * 100:
                        pattern = {
                            'type': 'influential_pattern',
                            'influence_score': influence,
                            'vector': vector,
                            'signature': self._generate_pattern_signature(sample),
                            'node': sample
                        }
                        scouting_results.append(pattern)
        
        self.logger.debug(f"Scouting found {len(scouting_results)} influential patterns")
        
        return scouting_results
    
    def _select_representative_samples(self, 
                                      code_section: List[ast.AST],
                                      sample_size: int) -> List[ast.AST]:
        """
        Select representative samples from code section.
        
        Uses stratified sampling to get diverse samples.
        """
        if len(code_section) <= sample_size:
            return code_section
        
        # Stratify by node type
        by_type = defaultdict(list)
        for node in code_section:
            node_type = type(node).__name__
            by_type[node_type].append(node)
        
        samples = []
        
        # Sample proportionally from each type
        for node_type, nodes in by_type.items():
            type_sample_size = max(1, int(sample_size * len(nodes) / len(code_section)))
            
            if len(nodes) <= type_sample_size:
                samples.extend(nodes)
            else:
                # Random sampling within type
                samples.extend(random.sample(nodes, type_sample_size))
        
        # Ensure we don't exceed sample_size
        if len(samples) > sample_size:
            samples = random.sample(samples, sample_size)
        
        return samples
    
    def _guided_fast_analysis(self,
                            code_section: List[ast.AST],
                            guide_patterns: List[Dict[str, Any]]) -> AnalysisResult:
        """
        Fast analysis guided by scouting results.
        
        Uses patterns found during scouting to accelerate analysis.
        """
        patterns_found = []
        
        # Extract guidance vectors and signatures
        guide_vectors = []
        guide_signatures = set()
        
        for pattern in guide_patterns:
            if 'vector' in pattern:
                guide_vectors.append(pattern['vector'])
            if 'signature' in pattern:
                guide_signatures.add(pattern['signature'])
        
        # Fast pass with guidance
        for node in code_section:
            # Quick signature check
            sig = self._generate_pattern_signature(node)
            
            # Check against guide signatures
            if sig in guide_signatures:
                patterns_found.append({
                    'type': 'guided_match',
                    'signature': sig,
                    'confidence': 0.95
                })
                continue
            
            # If node is function, check semantic similarity to guides
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and guide_vectors:
                # Only compute vector if we have guides
                vector = self._get_or_compute_vector(node)
                
                if vector is not None:
                    for guide_vector in guide_vectors:
                        similarity = self._calculate_similarity(vector, guide_vector)
                        
                        if similarity > 0.7:
                            patterns_found.append({
                                'type': 'semantic_guided',
                                'signature': sig,
                                'similarity': similarity,
                                'confidence': similarity
                            })
                            break
        
        return AnalysisResult(
            strategy_used=AnalysisStrategy.HYBRID,
            patterns_found=patterns_found,
            execution_time=0.0,
            confidence=0.85,  # Good confidence from hybrid approach
            samples_analyzed=len(guide_patterns),
            total_nodes=len(code_section),
            metrics={'guide_patterns': len(guide_patterns)}
        )
    
    def _generate_pattern_signature(self, node: ast.AST) -> str:
        """Generate a signature for pattern identification."""
        parts = [type(node).__name__]
        
        if hasattr(node, 'name'):
            # Normalize name for pattern matching
            name = getattr(node, 'name', '')
            if name.startswith('get_'):
                parts.append('getter')
            elif name.startswith('set_'):
                parts.append('setter')
            else:
                parts.append('method')
        
        # Add structural elements
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(f"args:{len(node.args.args)}")
            parts.append(f"body:{len(node.body)}")
            
            # Add control flow signature
            has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
            has_if = any(isinstance(n, ast.If) for n in ast.walk(node))
            has_try = any(isinstance(n, ast.Try) for n in ast.walk(node))
            
            if has_loop:
                parts.append('loop')
            if has_if:
                parts.append('conditional')
            if has_try:
                parts.append('exception')
        
        return ':'.join(parts)
    
    def _update_pattern_cache(self, 
                            pattern_signature: str,
                            strategy: AnalysisStrategy) -> None:
        """Update pattern familiarity cache."""
        if pattern_signature not in self.pattern_cache:
            self.pattern_cache[pattern_signature] = PatternCache(
                pattern_signature=pattern_signature
            )
        
        cache_entry = self.pattern_cache[pattern_signature]
        cache_entry.occurrence_count += 1
        cache_entry.last_seen = time.time()
        cache_entry.recommended_strategy = strategy
    
    def _update_metrics(self, 
                       strategy: AnalysisStrategy,
                       execution_time: float,
                       patterns_found: int) -> None:
        """Update performance metrics."""
        metrics = self.performance_metrics[strategy.value]
        metrics['count'] += 1
        metrics['total_time'] += execution_time
        metrics['avg_time'] = metrics['total_time'] / metrics['count']
        metrics['patterns_found'] += patterns_found
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get summary of performance metrics."""
        summary = {
            'strategies': {},
            'cache_size': len(self.pattern_cache),
            'semantic_cache_size': len(self._semantic_cache)
        }
        
        for strategy in AnalysisStrategy:
            metrics = self.performance_metrics[strategy.value]
            if metrics['count'] > 0:
                summary['strategies'][strategy.value] = {
                    'uses': metrics['count'],
                    'avg_time': round(metrics['avg_time'], 3),
                    'total_patterns': metrics['patterns_found'],
                    'patterns_per_use': round(
                        metrics['patterns_found'] / metrics['count'], 1
                    )
                }
        
        return summary
    
    def recommend_strategy(self, 
                          code_stats: Dict[str, Any]) -> AnalysisStrategy:
        """
        Recommend analysis strategy based on code statistics.
        
        Args:
            code_stats: Statistics about the code to analyze
            
        Returns:
            Recommended strategy
        """
        # High complexity → DEEP
        if code_stats.get('cyclomatic_complexity', 0) > self.COMPLEXITY_THRESHOLD:
            return AnalysisStrategy.DEEP
        
        # Many duplicates → HYBRID (scout then fast)
        if code_stats.get('duplicate_ratio', 0) > 0.3:
            return AnalysisStrategy.HYBRID
        
        # Small codebase → DEEP (can afford it)
        if code_stats.get('total_functions', 0) < 50:
            return AnalysisStrategy.DEEP
        
        # Large codebase → FAST or HYBRID
        if code_stats.get('total_functions', 0) > 500:
            # Use HYBRID for critical paths
            if code_stats.get('is_critical_path', False):
                return AnalysisStrategy.HYBRID
            else:
                return AnalysisStrategy.FAST
        
        # Default to FAST
        return AnalysisStrategy.FAST