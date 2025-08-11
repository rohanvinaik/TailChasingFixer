"""
Optimized TailChasingFixer orchestrator that breaks traditional sequential analysis.

This module coordinates all optimization components to provide revolutionary 
performance improvements over traditional dependency-ordered analysis:

1. Cluster-based analysis (no sorting) - O(n)
2. Influence detection with sampling - O(n) with limited exploration  
3. Non-linear navigation (jump to influential nodes) - O(k*m)
4. Pattern frontier exploration - O(c) where c=cluster count
5. Adaptive algorithm selection - Context-driven strategy choice
6. Fix prioritization by influence - O(i log i) cascade-aware ordering

Key innovation: Breaks the "sorting barrier" by analyzing high-impact patterns first,
regardless of their position in dependency graphs.
"""

import ast
import time
import asyncio
from typing import Dict, List, Set, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from enum import Enum
import numpy as np

from ..core.issues import Issue
from ..utils.logging_setup import get_logger
from .cluster_engine import ClusterBasedAnalyzer, CodeCluster
from .influence_detector import InfluenceDetector, InfluentialNode  
from .nonlinear_navigator import NonLinearNavigator, NavigationStrategy, FixCluster
from .adaptive_selector import AdaptiveAlgorithmSelector, AnalysisStrategy


class OptimizationPhase(Enum):
    """Phases of optimized analysis."""
    CLUSTERING = "clustering"
    INFLUENCE_DETECTION = "influence_detection" 
    NAVIGATION = "navigation"
    FRONTIER_EXPLORATION = "frontier_exploration"
    ADAPTIVE_ANALYSIS = "adaptive_analysis"
    FIX_PRIORITIZATION = "fix_prioritization"


class PerformanceMode(Enum):
    """Performance optimization modes."""
    SPEED = "speed"          # Maximum speed, minimal accuracy trade-offs
    BALANCED = "balanced"    # Balance speed and accuracy
    THOROUGH = "thorough"    # Comprehensive analysis, some speed trade-off
    LEGACY = "legacy"        # Traditional sequential analysis for comparison


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance tracking."""
    phase_times: Dict[str, float] = field(default_factory=dict)
    total_time: float = 0.0
    nodes_analyzed: int = 0
    patterns_found: int = 0
    clusters_created: int = 0
    influential_nodes: int = 0
    speedup_factor: float = 1.0
    memory_usage: float = 0.0
    cache_hit_rate: float = 0.0
    
    def add_phase_time(self, phase: OptimizationPhase, duration: float):
        """Add timing for a phase."""
        self.phase_times[phase.value] = duration
        self.total_time += duration
    
    def calculate_speedup(self, baseline_time: float):
        """Calculate speedup factor compared to baseline."""
        if baseline_time > 0:
            self.speedup_factor = baseline_time / max(self.total_time, 0.001)
        return self.speedup_factor


@dataclass
class OptimizationConfig:
    """Configuration for optimization behavior."""
    mode: PerformanceMode = PerformanceMode.BALANCED
    max_clusters: int = 50
    influence_threshold: float = 0.6
    sample_ratio: float = 0.1
    max_frontier_patterns: int = 100
    enable_caching: bool = True
    navigation_strategy: NavigationStrategy = NavigationStrategy.HYBRID_SEARCH
    analysis_strategy: AnalysisStrategy = AnalysisStrategy.HYBRID
    parallel_processing: bool = True
    memory_limit_mb: int = 512


@dataclass
class AnalysisResult:
    """Result from optimized analysis."""
    issues: List[Issue]
    fix_clusters: List[FixCluster] 
    influential_patterns: List[InfluentialNode]
    clusters: Dict[str, CodeCluster]
    metrics: OptimizationMetrics
    recommendations: List[str] = field(default_factory=list)


class PatternFrontierExplorer:
    """
    Explore the "frontier" of code patterns more efficiently by
    sampling and clustering rather than exhaustive search.
    
    Inspired by: Grouping frontier nodes into clusters and examining
    one node from each cluster to avoid O(n²) comparisons.
    """
    
    def __init__(self, max_patterns: int = 100):
        """Initialize frontier explorer."""
        self.logger = get_logger("frontier_explorer")
        self.max_patterns = max_patterns
        self.cluster_cache: Dict[str, List[Dict]] = {}
        self.pattern_signatures: Set[str] = set()
        
    def explore_pattern_frontier(self, 
                               codebase: Dict[str, ast.AST],
                               known_patterns: List[Dict]) -> List[Dict]:
        """
        Find new patterns by exploring the frontier strategically,
        not exhaustively.
        
        Args:
            codebase: Dict mapping file paths to AST trees
            known_patterns: Already discovered patterns
            
        Returns:
            List of newly discovered patterns
        """
        self.logger.debug(f"Exploring pattern frontier with {len(known_patterns)} known patterns")
        
        # Define the frontier (boundary between known and unknown patterns)
        frontier = self.identify_pattern_frontier(codebase, known_patterns)
        
        if not frontier:
            return []
        
        # Cluster frontier patterns by similarity
        frontier_clusters = self.cluster_frontier_patterns(frontier)
        
        # Sample one representative from each cluster (not all patterns)
        representatives = self.select_cluster_representatives(frontier_clusters)
        
        # Analyze only representatives to find new patterns
        new_patterns = []
        for rep in representatives:
            # Quick analysis of representative
            pattern = self.analyze_representative(rep)
            
            if pattern and self.is_novel_pattern(pattern):
                # Extrapolate to entire cluster
                cluster_patterns = self.extrapolate_to_cluster(
                    pattern, 
                    frontier_clusters.get(rep.get('cluster_id', ''), [])
                )
                new_patterns.extend(cluster_patterns[:10])  # Limit per cluster
        
        self.logger.info(f"Discovered {len(new_patterns)} new patterns from {len(representatives)} representatives")
        
        return new_patterns[:self.max_patterns]
    
    def identify_pattern_frontier(self, 
                                codebase: Dict[str, ast.AST],
                                known_patterns: List[Dict]) -> List[Dict]:
        """Identify patterns on the frontier of known space."""
        known_signatures = {p.get('signature', '') for p in known_patterns}
        self.pattern_signatures.update(known_signatures)
        
        frontier_patterns = []
        
        for file_path, ast_tree in codebase.items():
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    signature = self._generate_pattern_signature(node)
                    
                    # Pattern is on frontier if it's similar but not identical to known patterns
                    if self._is_frontier_pattern(signature, known_signatures):
                        frontier_patterns.append({
                            'node': node,
                            'file_path': file_path,
                            'signature': signature,
                            'name': getattr(node, 'name', f'line{node.lineno}')
                        })
        
        return frontier_patterns[:self.max_patterns * 2]  # Limit frontier size
    
    def _is_frontier_pattern(self, signature: str, known_signatures: Set[str]) -> bool:
        """Check if pattern is on the frontier (similar but not identical)."""
        if signature in known_signatures:
            return False  # Already known
        
        # Check similarity to known patterns
        signature_parts = set(signature.split(':'))
        
        for known_sig in known_signatures:
            known_parts = set(known_sig.split(':'))
            
            # Calculate Jaccard similarity
            intersection = len(signature_parts & known_parts)
            union = len(signature_parts | known_parts)
            
            if union > 0:
                similarity = intersection / union
                # On frontier if moderately similar (0.3-0.8)
                if 0.3 <= similarity <= 0.8:
                    return True
        
        return False
    
    def cluster_frontier_patterns(self, frontier: List[Dict]) -> Dict[str, List[Dict]]:
        """
        Group frontier patterns into clusters to avoid examining all.
        This is how the algorithm avoids the sorting barrier.
        """
        if not frontier:
            return {}
        
        clusters = {}
        
        # Use fast clustering based on signature similarity
        for pattern in frontier:
            # Find nearest cluster (don't sort all distances)
            nearest_cluster = self.find_nearest_cluster_fast(
                pattern, 
                clusters,
                max_comparisons=5  # Don't compare to all clusters
            )
            
            if nearest_cluster:
                clusters[nearest_cluster].append(pattern)
                pattern['cluster_id'] = nearest_cluster
            else:
                # Create new cluster
                new_cluster_id = self.generate_cluster_id(pattern)
                clusters[new_cluster_id] = [pattern]
                pattern['cluster_id'] = new_cluster_id
        
        self.logger.debug(f"Created {len(clusters)} frontier clusters from {len(frontier)} patterns")
        
        return clusters
    
    def find_nearest_cluster_fast(self, 
                                 pattern: Dict,
                                 clusters: Dict[str, List[Dict]],
                                 max_comparisons: int = 5) -> Optional[str]:
        """Find nearest cluster using limited comparisons."""
        pattern_sig = pattern.get('signature', '')
        pattern_parts = set(pattern_sig.split(':'))
        
        best_cluster = None
        best_similarity = 0.5  # Minimum threshold
        comparisons = 0
        
        for cluster_id, cluster_patterns in clusters.items():
            if comparisons >= max_comparisons:
                break
            
            # Compare to cluster representative (first pattern)
            if cluster_patterns:
                rep_sig = cluster_patterns[0].get('signature', '')
                rep_parts = set(rep_sig.split(':'))
                
                intersection = len(pattern_parts & rep_parts)
                union = len(pattern_parts | rep_parts)
                
                if union > 0:
                    similarity = intersection / union
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_cluster = cluster_id
            
            comparisons += 1
        
        return best_cluster
    
    def generate_cluster_id(self, pattern: Dict) -> str:
        """Generate cluster ID based on pattern characteristics."""
        node = pattern.get('node')
        if not node:
            return f"cluster_{len(self.cluster_cache)}"
        
        # Use node type and key characteristics
        parts = [type(node).__name__]
        
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            if name.startswith(('get_', 'fetch_')):
                parts.append('retrieval')
            elif name.startswith(('set_', 'update_')):
                parts.append('modification')
            elif name.startswith(('validate_', 'check_')):
                parts.append('validation')
            elif 'error' in name or 'exception' in name:
                parts.append('error_handling')
            else:
                parts.append('general')
        
        elif isinstance(node, ast.ClassDef):
            if len(node.body) > 10:
                parts.append('large')
            elif any(isinstance(child, ast.FunctionDef) and child.name.startswith('__')
                    for child in node.body):
                parts.append('protocol')
            else:
                parts.append('standard')
        
        return '_'.join(parts)
    
    def select_cluster_representatives(self, 
                                    frontier_clusters: Dict[str, List[Dict]]) -> List[Dict]:
        """Select one representative from each cluster for analysis."""
        representatives = []
        
        for cluster_id, cluster_patterns in frontier_clusters.items():
            if not cluster_patterns:
                continue
            
            # Select most "central" pattern as representative
            # For now, use first pattern, but could use centroid
            representative = cluster_patterns[0]
            representative['represents_count'] = len(cluster_patterns)
            representatives.append(representative)
        
        return representatives
    
    def analyze_representative(self, representative: Dict) -> Optional[Dict]:
        """Quickly analyze a representative pattern."""
        node = representative.get('node')
        if not node:
            return None
        
        # Quick structural analysis
        pattern_info = {
            'type': type(node).__name__,
            'signature': representative.get('signature'),
            'file_path': representative.get('file_path'),
            'name': representative.get('name'),
            'cluster_size': representative.get('represents_count', 1)
        }
        
        # Add complexity metrics
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            pattern_info.update({
                'arg_count': len(node.args.args),
                'body_size': len(node.body),
                'has_loops': any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node)),
                'has_exceptions': any(isinstance(n, ast.Try) for n in ast.walk(node)),
                'complexity_score': self._calculate_complexity(node)
            })
        
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            pattern_info.update({
                'method_count': len(methods),
                'inheritance_depth': len(node.bases),
                'is_abstract': any(self._is_abstract_method(m) for m in methods)
            })
        
        return pattern_info
    
    def _is_abstract_method(self, method: ast.FunctionDef) -> bool:
        """Check if method is abstract (just pass/ellipsis/raise NotImplementedError)."""
        if len(method.body) == 1:
            stmt = method.body[0]
            if isinstance(stmt, ast.Pass):
                return True
            elif isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Ellipsis):
                return True
            elif (isinstance(stmt, ast.Raise) and 
                  isinstance(stmt.exc, ast.Call) and
                  isinstance(stmt.exc.func, ast.Name) and
                  stmt.exc.func.id == 'NotImplementedError'):
                return True
        return False
    
    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        return complexity
    
    def is_novel_pattern(self, pattern: Dict) -> bool:
        """Check if pattern represents something genuinely new."""
        signature = pattern.get('signature', '')
        if signature in self.pattern_signatures:
            return False
        
        # Novel if complexity above threshold or unique structural characteristics
        complexity = pattern.get('complexity_score', 0)
        cluster_size = pattern.get('cluster_size', 1)
        
        # Novel if complex, represents many patterns, or has unique characteristics
        return (complexity > 5 or 
                cluster_size > 3 or
                pattern.get('has_exceptions', False) or
                pattern.get('is_abstract', False))
    
    def extrapolate_to_cluster(self, 
                             pattern: Dict, 
                             cluster_patterns: List[Dict]) -> List[Dict]:
        """Extrapolate representative analysis to entire cluster."""
        if not cluster_patterns:
            return [pattern]
        
        # Apply pattern analysis to all cluster members
        extrapolated = []
        
        for cluster_member in cluster_patterns:
            # Create pattern based on representative but with member-specific details
            member_pattern = pattern.copy()
            member_pattern.update({
                'file_path': cluster_member.get('file_path'),
                'name': cluster_member.get('name'),
                'signature': cluster_member.get('signature'),
                'node': cluster_member.get('node')
            })
            extrapolated.append(member_pattern)
        
        return extrapolated
    
    def _generate_pattern_signature(self, node: ast.AST) -> str:
        """Generate signature for pattern matching."""
        parts = [type(node).__name__]
        
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(f"args:{len(node.args.args)}")
            parts.append(f"body:{len(node.body)}")
            
            # Control flow signature
            has_try = any(isinstance(n, ast.Try) for n in ast.walk(node))
            has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
            has_if = any(isinstance(n, ast.If) for n in ast.walk(node))
            
            if has_try:
                parts.append("exception")
            if has_loop:
                parts.append("iteration") 
            if has_if:
                parts.append("conditional")
                
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            parts.append(f"methods:{len(methods)}")
            parts.append(f"bases:{len(node.bases)}")
        
        return ":".join(parts)


class OptimizedTailChasingFixer:
    """
    Main orchestrator for optimized tail-chasing analysis.
    
    Coordinates all optimization components to provide revolutionary 
    performance improvements over traditional dependency-ordered analysis.
    """
    
    def __init__(self, config: Optional[OptimizationConfig] = None):
        """
        Initialize optimized fixer.
        
        Args:
            config: Configuration for optimization behavior
        """
        self.config = config or OptimizationConfig()
        self.logger = get_logger("optimized_fixer")
        
        # Initialize optimization components
        self.cluster_analyzer = ClusterBasedAnalyzer()
        self.influence_detector = InfluenceDetector()
        self.navigator = NonLinearNavigator(self.influence_detector)
        self.adaptive_selector = AdaptiveAlgorithmSelector(
            self.influence_detector,
            enable_caching=self.config.enable_caching
        )
        self.frontier_explorer = PatternFrontierExplorer(
            max_patterns=self.config.max_frontier_patterns
        )
        
        # Performance tracking
        self.metrics = OptimizationMetrics()
        self.legacy_analyzer = None  # For comparison benchmarking
        
    def analyze_codebase_optimized(self, codebase: Dict[str, ast.AST]) -> AnalysisResult:
        """
        Main optimized analysis algorithm that breaks traditional sequential patterns.
        
        Revolutionary 6-phase approach:
        1. Cluster without sorting - O(n)
        2. Influence detection with sampling - O(n) with limited exploration
        3. Non-linear navigation - O(k*m) where k=influential nodes
        4. Pattern frontier exploration - O(c) where c=cluster count  
        5. Adaptive algorithm selection - Context-driven strategy
        6. Fix prioritization by influence - O(i log i) cascade-aware
        
        Args:
            codebase: Dict mapping file paths to AST trees
            
        Returns:
            AnalysisResult with issues, clusters, metrics, and recommendations
        """
        self.logger.info(f"Starting optimized analysis of {len(codebase)} files")
        total_start_time = time.time()
        
        # Phase 1: Cluster without sorting - O(n) 
        phase_start = time.time()
        clusters = self._phase1_clustering(codebase)
        self.metrics.add_phase_time(OptimizationPhase.CLUSTERING, time.time() - phase_start)
        self.metrics.clusters_created = len(clusters)
        
        # Phase 2: Find influential patterns - O(n) with sampling
        phase_start = time.time()  
        influential = self._phase2_influence_detection(clusters)
        self.metrics.add_phase_time(OptimizationPhase.INFLUENCE_DETECTION, time.time() - phase_start)
        self.metrics.influential_nodes = len(influential)
        
        # Phase 3: Navigate non-linearly - O(k*m)
        phase_start = time.time()
        initial_issues = self._phase3_nonlinear_navigation(codebase, influential)
        self.metrics.add_phase_time(OptimizationPhase.NAVIGATION, time.time() - phase_start)
        
        # Phase 4: Explore pattern frontier - O(c)
        phase_start = time.time()
        new_patterns = self._phase4_frontier_exploration(codebase, initial_issues)
        self.metrics.add_phase_time(OptimizationPhase.FRONTIER_EXPLORATION, time.time() - phase_start)
        
        # Phase 5: Adaptive analysis - Context-driven
        phase_start = time.time()
        enhanced_issues = self._phase5_adaptive_analysis(codebase, initial_issues, new_patterns)
        self.metrics.add_phase_time(OptimizationPhase.ADAPTIVE_ANALYSIS, time.time() - phase_start)
        self.metrics.patterns_found = len(enhanced_issues)
        
        # Phase 6: Prioritize fixes by influence - O(i log i)
        phase_start = time.time()
        fix_clusters = self._phase6_fix_prioritization(enhanced_issues)
        self.metrics.add_phase_time(OptimizationPhase.FIX_PRIORITIZATION, time.time() - phase_start)
        
        # Calculate final metrics
        self.metrics.total_time = time.time() - total_start_time
        self.metrics.nodes_analyzed = sum(len(c.members) for c in clusters.values())
        
        # Generate recommendations
        recommendations = self._generate_recommendations(clusters, influential, fix_clusters)
        
        result = AnalysisResult(
            issues=enhanced_issues,
            fix_clusters=fix_clusters,
            influential_patterns=influential,
            clusters=clusters,
            metrics=self.metrics,
            recommendations=recommendations
        )
        
        self.logger.info(
            f"Optimized analysis complete: {len(enhanced_issues)} issues, "
            f"{len(fix_clusters)} fix clusters, {self.metrics.total_time:.3f}s"
        )
        
        return result
    
    def _phase1_clustering(self, codebase: Dict[str, ast.AST]) -> Dict[str, Any]:
        """Phase 1: Create semantic clusters without dependency sorting."""
        self.logger.debug("Phase 1: Clustering without sorting")
        
        clusters = self.cluster_analyzer.create_semantic_clusters(codebase)
        
        self.logger.debug(f"Created {len(clusters)} semantic clusters")
        
        return clusters
    
    def _phase2_influence_detection(self, clusters: Dict[str, Any]) -> List[InfluentialNode]:
        """Phase 2: Find influential patterns using sampling."""
        self.logger.debug("Phase 2: Influence detection with sampling")
        
        # Find influential patterns from clusters
        influential = self.cluster_analyzer.find_influential_patterns(clusters)
        
        # Additional influence analysis using influence detector
        enhanced_influential = []
        for pattern in influential[:20]:  # Limit to top 20
            # Convert to InfluentialNode format
            node = InfluentialNode(
                node_id=pattern.pattern_id,
                node_type="pattern",
                influence_score=pattern.influence_metric,
                pattern_signature=pattern.pattern_id
            )
            enhanced_influential.append(node)
        
        self.logger.debug(f"Identified {len(enhanced_influential)} influential patterns")
        
        return enhanced_influential
    
    def _phase3_nonlinear_navigation(self, 
                                   codebase: Dict[str, ast.AST],
                                   influential: List[InfluentialNode]) -> List[Issue]:
        """Phase 3: Navigate codebase non-linearly starting from influential nodes."""
        self.logger.debug("Phase 3: Non-linear navigation")
        
        # Navigate by influence priority
        navigation_plan = self.navigator.navigate_by_influence(
            codebase,
            self.config.navigation_strategy
        )
        
        # Convert navigation results to initial issues
        initial_issues = []
        
        # Process high-priority nodes first
        for node in navigation_plan.ordered_nodes[:50]:  # Top 50 nodes
            # Quick analysis to identify obvious issues
            issues = self._quick_analyze_node(node.ast_node, node.file_path)
            initial_issues.extend(issues)
        
        self.logger.debug(f"Navigation found {len(initial_issues)} initial issues")
        
        return initial_issues
    
    def _quick_analyze_node(self, node: ast.AST, file_path: str) -> List[Issue]:
        """Quickly analyze a node for obvious issues."""
        issues = []
        
        if isinstance(node, ast.FunctionDef):
            # Check for empty function
            if (len(node.body) == 1 and 
                isinstance(node.body[0], ast.Pass)):
                issues.append(Issue(
                    kind="phantom_function",
                    message=f"Empty function '{node.name}' may be placeholder",
                    file=file_path,
                    line=node.lineno,
                    severity=2,
                    confidence=0.8
                ))
            
            # Check for bare except
            for child in ast.walk(node):
                if (isinstance(child, ast.ExceptHandler) and 
                    child.type is None):
                    issues.append(Issue(
                        kind="bare_except",
                        message="Bare except clause catches all exceptions",
                        file=file_path,
                        line=getattr(child, 'lineno', node.lineno),
                        severity=3,
                        confidence=0.9
                    ))
        
        return issues
    
    def _phase4_frontier_exploration(self, 
                                   codebase: Dict[str, ast.AST],
                                   initial_issues: List[Issue]) -> List[Dict]:
        """Phase 4: Explore pattern frontier strategically."""
        self.logger.debug("Phase 4: Pattern frontier exploration")
        
        # Convert initial issues to pattern format
        known_patterns = []
        for issue in initial_issues:
            known_patterns.append({
                'signature': f"{issue.kind}:{issue.file}:{issue.line}",
                'type': issue.kind
            })
        
        # Explore frontier
        new_patterns = self.frontier_explorer.explore_pattern_frontier(
            codebase,
            known_patterns
        )
        
        self.logger.debug(f"Frontier exploration found {len(new_patterns)} new patterns")
        
        return new_patterns
    
    def _phase5_adaptive_analysis(self, 
                                codebase: Dict[str, ast.AST],
                                initial_issues: List[Issue],
                                new_patterns: List[Dict]) -> List[Issue]:
        """Phase 5: Apply adaptive algorithm selection for detailed analysis."""
        self.logger.debug("Phase 5: Adaptive analysis")
        
        all_issues = initial_issues.copy()
        
        # Analyze new patterns with adaptive algorithm selection
        pattern_nodes = []
        for pattern in new_patterns:
            node = pattern.get('node')
            if node:
                pattern_nodes.append(node)
        
        if pattern_nodes:
            # Apply adaptive analysis
            analysis_result = self.adaptive_selector.analyze_with_adaptive_algorithms(
                pattern_nodes
            )
            
            # Convert analysis results to issues
            for i, pattern_result in enumerate(analysis_result.patterns_found):
                pattern_dict = new_patterns[i] if i < len(new_patterns) else {}
                
                issue = Issue(
                    kind=pattern_result.get('type', 'pattern_detected'),
                    message=f"Pattern detected: {pattern_result.get('type', 'unknown')}",
                    file=pattern_dict.get('file_path', 'unknown'),
                    line=getattr(pattern_dict.get('node'), 'lineno', 0),
                    severity=2,
                    confidence=pattern_result.get('confidence', 0.7)
                )
                all_issues.append(issue)
        
        self.logger.debug(f"Adaptive analysis produced {len(all_issues)} total issues")
        
        return all_issues
    
    def _phase6_fix_prioritization(self, issues: List[Issue]) -> List[FixCluster]:
        """Phase 6: Prioritize fixes by influence and cascade potential."""
        self.logger.debug("Phase 6: Fix prioritization by influence")
        
        # Use influence-based fix prioritizer
        fix_prioritizer = self.navigator.influence_detector  # Has prioritizer methods
        
        # Create fix clusters prioritized by influence
        try:
            # Use the navigator's fix prioritizer
            from .nonlinear_navigator import InfluenceBasedFixPrioritizer
            prioritizer = InfluenceBasedFixPrioritizer(self.influence_detector)
            fix_clusters = prioritizer.prioritize_fixes_by_influence(issues)
        except Exception as e:
            self.logger.warning(f"Fix prioritization failed: {e}, using fallback")
            # Fallback: simple priority by severity
            fix_clusters = self._fallback_fix_prioritization(issues)
        
        self.logger.debug(f"Created {len(fix_clusters)} fix clusters")
        
        return fix_clusters
    
    def _fallback_fix_prioritization(self, issues: List[Issue]) -> List[FixCluster]:
        """Fallback fix prioritization when main algorithm fails."""
        from .nonlinear_navigator import FixCluster
        
        # Group by severity and create simple clusters
        severity_groups = defaultdict(list)
        for issue in issues:
            severity_groups[issue.severity].append(issue)
        
        clusters = []
        cluster_id = 0
        
        for severity in sorted(severity_groups.keys(), reverse=True):
            group_issues = severity_groups[severity]
            
            # Create cluster for this severity group
            if group_issues:
                primary = group_issues[0]
                dependents = group_issues[1:]
                
                cluster = FixCluster(
                    cluster_id=f"fallback_cluster_{cluster_id}",
                    primary_issue=primary,
                    dependent_issues=dependents,
                    influence_score=severity * 10,
                    cascade_potential=len(dependents),
                    fix_complexity=1.0 + len(dependents) * 0.2,
                    estimated_benefit=severity / 5.0
                )
                clusters.append(cluster)
                cluster_id += 1
        
        return clusters
    
    def _generate_recommendations(self, 
                                clusters: Dict[str, Any],
                                influential: List[InfluentialNode],
                                fix_clusters: List[FixCluster]) -> List[str]:
        """Generate recommendations based on analysis results."""
        recommendations = []
        
        # Always provide basic recommendation
        recommendations.append(
            f"Analysis completed with {len(clusters)} clusters in {self.metrics.total_time:.3f}s. "
            f"Optimized approach processed {self.metrics.nodes_analyzed} nodes efficiently."
        )
        
        # Cluster recommendations
        if len(clusters) > 20:
            recommendations.append(
                f"Large number of clusters ({len(clusters)}) suggests high code diversity. "
                "Consider refactoring common patterns."
            )
        elif len(clusters) > 0:
            recommendations.append(
                f"Found {len(clusters)} semantic clusters. This indicates good code organization."
            )
        
        # Influential pattern recommendations
        if influential:
            top_influential = influential[0]
            recommendations.append(
                f"Most influential pattern: {top_influential.node_id} "
                f"(score: {top_influential.influence_score:.1f}). "
                "Focus optimization efforts here for maximum impact."
            )
        else:
            recommendations.append(
                "No highly influential patterns detected. Code appears well-distributed."
            )
        
        # Fix prioritization recommendations
        if fix_clusters:
            high_cascade_clusters = [c for c in fix_clusters 
                                   if c.cascade_potential > 3]
            if high_cascade_clusters:
                recommendations.append(
                    f"{len(high_cascade_clusters)} fix clusters have high cascade potential. "
                    "Fixing these will resolve multiple issues simultaneously."
                )
        
        # Performance recommendations
        if self.metrics.total_time > 30:
            recommendations.append(
                "Analysis took longer than 30 seconds. Consider using SPEED mode "
                "for faster analysis with minimal accuracy trade-off."
            )
        elif self.metrics.total_time < 1:
            recommendations.append(
                "Analysis completed very quickly. Consider using THOROUGH mode "
                "for more comprehensive issue detection."
            )
        
        return recommendations
    
    async def analyze_codebase_parallel(self, codebase: Dict[str, ast.AST]) -> AnalysisResult:
        """Parallel version of optimized analysis using asyncio."""
        if not self.config.parallel_processing:
            return self.analyze_codebase_optimized(codebase)
        
        self.logger.info("Starting parallel optimized analysis")
        
        # Run phases that can be parallelized
        tasks = []
        
        # Phase 1: Clustering (sequential - builds foundation)
        clusters = self._phase1_clustering(codebase)
        
        # Phase 2 & 3: Influence detection and navigation (can run in parallel)
        tasks.append(self._async_phase2_influence_detection(clusters))
        
        # Wait for influence detection
        influential = await tasks[0]
        
        # Phase 3: Navigation (depends on influence)
        initial_issues = self._phase3_nonlinear_navigation(codebase, influential)
        
        # Phase 4 & 5: Frontier exploration and adaptive analysis (parallel)
        tasks = [
            self._async_phase4_frontier_exploration(codebase, initial_issues),
            self._async_phase5_adaptive_analysis(codebase, initial_issues, [])
        ]
        
        new_patterns, _ = await asyncio.gather(*tasks)
        
        # Phase 5: Complete adaptive analysis with new patterns
        enhanced_issues = self._phase5_adaptive_analysis(codebase, initial_issues, new_patterns)
        
        # Phase 6: Fix prioritization (sequential)
        fix_clusters = self._phase6_fix_prioritization(enhanced_issues)
        
        # Generate results
        recommendations = self._generate_recommendations(clusters, influential, fix_clusters)
        
        return AnalysisResult(
            issues=enhanced_issues,
            fix_clusters=fix_clusters,
            influential_patterns=influential,
            clusters=clusters,
            metrics=self.metrics,
            recommendations=recommendations
        )
    
    async def _async_phase2_influence_detection(self, clusters: Dict[str, Any]) -> List[InfluentialNode]:
        """Async version of influence detection."""
        return self._phase2_influence_detection(clusters)
    
    async def _async_phase4_frontier_exploration(self, 
                                               codebase: Dict[str, ast.AST],
                                               initial_issues: List[Issue]) -> List[Dict]:
        """Async version of frontier exploration."""
        return self._phase4_frontier_exploration(codebase, initial_issues)
    
    async def _async_phase5_adaptive_analysis(self, 
                                            codebase: Dict[str, ast.AST],
                                            initial_issues: List[Issue],
                                            new_patterns: List[Dict]) -> List[Issue]:
        """Async version of adaptive analysis.""" 
        return self._phase5_adaptive_analysis(codebase, initial_issues, new_patterns)
    
    def benchmark_against_legacy(self, codebase: Dict[str, ast.AST]) -> Dict[str, Any]:
        """Benchmark optimized analysis against legacy sequential approach."""
        self.logger.info("Benchmarking optimized vs legacy analysis")
        
        # Run optimized analysis
        optimized_start = time.time()
        optimized_result = self.analyze_codebase_optimized(codebase)
        optimized_time = time.time() - optimized_start
        
        # Simulate legacy analysis (sequential file processing)
        legacy_start = time.time()
        legacy_issues = self._simulate_legacy_analysis(codebase)
        legacy_time = time.time() - legacy_start
        
        # Calculate metrics
        speedup = legacy_time / max(optimized_time, 0.001)
        self.metrics.calculate_speedup(legacy_time)
        
        benchmark_results = {
            'optimized_time': optimized_time,
            'legacy_time': legacy_time,
            'speedup_factor': speedup,
            'optimized_issues': len(optimized_result.issues),
            'legacy_issues': len(legacy_issues),
            'accuracy_ratio': len(optimized_result.issues) / max(len(legacy_issues), 1),
            'memory_optimization': self.metrics.memory_usage,
            'cache_efficiency': self.metrics.cache_hit_rate
        }
        
        self.logger.info(
            f"Benchmark complete: {speedup:.1f}x speedup, "
            f"{len(optimized_result.issues)}/{len(legacy_issues)} issues found"
        )
        
        return benchmark_results
    
    def _simulate_legacy_analysis(self, codebase: Dict[str, ast.AST]) -> List[Issue]:
        """Simulate traditional sequential analysis for comparison."""
        issues = []
        
        # Process files in alphabetical order (typical legacy approach)
        for file_path in sorted(codebase.keys()):
            ast_tree = codebase[file_path]
            
            # Simple sequential traversal
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.FunctionDef):
                    # Basic issue detection
                    if len(node.body) == 1 and isinstance(node.body[0], ast.Pass):
                        issues.append(Issue(
                            kind="empty_function",
                            message=f"Empty function: {node.name}",
                            file=file_path,
                            line=node.lineno,
                            severity=1
                        ))
                    
                    # Check for bare except
                    for child in ast.walk(node):
                        if isinstance(child, ast.ExceptHandler) and child.type is None:
                            issues.append(Issue(
                                kind="bare_except",
                                message="Bare except clause",
                                file=file_path,
                                line=getattr(child, 'lineno', node.lineno),
                                severity=3
                            ))
        
        return issues
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Get detailed performance report."""
        return {
            'phase_times': self.metrics.phase_times,
            'total_time': self.metrics.total_time,
            'nodes_analyzed': self.metrics.nodes_analyzed,
            'patterns_found': self.metrics.patterns_found,
            'clusters_created': self.metrics.clusters_created,
            'influential_nodes': self.metrics.influential_nodes,
            'speedup_factor': self.metrics.speedup_factor,
            'memory_usage_mb': self.metrics.memory_usage,
            'cache_hit_rate': self.metrics.cache_hit_rate,
            'configuration': {
                'mode': self.config.mode.value,
                'max_clusters': self.config.max_clusters,
                'influence_threshold': self.config.influence_threshold,
                'navigation_strategy': self.config.navigation_strategy.value,
                'analysis_strategy': self.config.analysis_strategy.value,
                'parallel_processing': self.config.parallel_processing
            }
        }