"""
Cluster-based code analysis engine that breaks the sorting barrier.

Instead of analyzing code in dependency order (which requires expensive sorting),
this engine groups related code into semantic clusters and analyzes strategically.

Inspired by: "Breaking the sorting barrier" - analyzing without full ordering.
"""

import ast
import re
from typing import Dict, List, Tuple, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass
import numpy as np

from ..semantic.encoder import encode_function
from ..semantic.index import SemanticIndex
from ..core.issues import Issue
from ..utils.logging_setup import get_logger


@dataclass
class CodeCluster:
    """Represents a semantic cluster of related code."""
    cluster_id: str
    pattern_type: str
    members: List[Tuple[str, ast.AST]]  # (file_path, ast_node) pairs
    centroid_vector: Optional[np.ndarray] = None
    influence_score: float = 0.0
    avg_similarity: float = 0.0


@dataclass
class InfluentialPattern:
    """A code pattern that influences many others."""
    pattern_id: str
    pattern_type: str
    examples: List[Tuple[str, ast.AST]]
    affected_clusters: Set[str]
    influence_metric: float


class ClusterBasedAnalyzer:
    """
    Analyzes code using semantic clustering instead of dependency ordering.
    
    This breaks the "sorting barrier" by avoiding the need to sort code
    by dependencies before analysis. Instead, we cluster semantically
    similar code and analyze strategically.
    """
    
    # Pattern prefixes for semantic clustering
    PATTERN_PREFIXES = {
        'data_handlers': ['get_', 'fetch_', 'load_', 'read_', 'retrieve_', 'query_'],
        'validators': ['check_', 'validate_', 'verify_', 'ensure_', 'assert_', 'is_', 'has_'],
        'transformers': ['convert_', 'transform_', 'process_', 'parse_', 'format_', 'encode_', 'decode_'],
        'controllers': ['handle_', 'manage_', 'control_', 'dispatch_', 'route_', 'execute_'],
        'utilities': ['util_', 'helper_', 'common_', 'shared_', 'base_', 'core_'],
        'initializers': ['init_', 'setup_', 'configure_', 'register_', 'bootstrap_'],
        'cleaners': ['clean_', 'clear_', 'remove_', 'delete_', 'purge_', 'reset_'],
        'builders': ['build_', 'create_', 'make_', 'generate_', 'construct_', 'assemble_'],
        'analyzers': ['analyze_', 'inspect_', 'examine_', 'scan_', 'detect_', 'find_'],
        'writers': ['write_', 'save_', 'store_', 'persist_', 'export_', 'dump_'],
    }
    
    # Structural patterns for clustering
    STRUCTURAL_PATTERNS = {
        'error_handler': 'try/except with logging',
        'decorator': 'function that returns function',
        'context_manager': 'class with __enter__/__exit__',
        'singleton': 'class with single instance pattern',
        'factory': 'function/class that creates objects',
        'iterator': 'class with __iter__/__next__',
        'property_accessor': 'getter/setter pattern',
        'callback': 'function passed as argument',
        'recursive': 'function that calls itself',
        'generator': 'function with yield',
    }
    
    def __init__(self, semantic_index: Optional[SemanticIndex] = None):
        """
        Initialize the cluster-based analyzer.
        
        Args:
            semantic_index: Optional pre-initialized semantic index for hypervector similarity
        """
        self.logger = get_logger("cluster_engine")
        self.semantic_index = semantic_index
        self.clusters: Dict[str, CodeCluster] = {}
        self.influential_patterns: List[InfluentialPattern] = []
        self._vector_cache: Dict[str, np.ndarray] = {}
        
    def analyze_without_sorting(self, codebase: Dict[str, ast.AST]) -> List[Issue]:
        """
        Analyze codebase without dependency sorting - the main entry point.
        
        This breaks the "sorting barrier" by using semantic clustering
        instead of dependency ordering.
        
        Args:
            codebase: Dict mapping file paths to AST trees
            
        Returns:
            List of detected issues
        """
        self.logger.info(f"Starting cluster-based analysis of {len(codebase)} files")
        
        # Step 1: Create semantic clusters in O(n) time
        clusters = self.create_semantic_clusters(codebase)
        self.logger.info(f"Created {len(clusters)} semantic clusters")
        
        # Step 2: Identify influential patterns (no sorting needed)
        influential_code = self.find_influential_patterns(clusters)
        self.logger.info(f"Identified {len(influential_code)} influential patterns")
        
        # Step 3: Analyze influential patterns first
        priority_issues = self.analyze_influential_first(influential_code)
        self.logger.info(f"Found {len(priority_issues)} priority issues")
        
        # Step 4: Adaptive analysis of remaining clusters
        remaining_issues = self.adaptive_cluster_analysis(
            clusters,
            learned_patterns=self._extract_patterns(priority_issues)
        )
        self.logger.info(f"Found {len(remaining_issues)} additional issues")
        
        # Step 5: Merge and deduplicate results
        return self.merge_results(priority_issues, remaining_issues)
    
    def create_semantic_clusters(self, codebase: Dict[str, ast.AST]) -> Dict[str, CodeCluster]:
        """
        Group code by semantic patterns WITHOUT sorting by dependencies.
        
        This is the key innovation - we cluster by semantic similarity,
        not by import order or dependency structure.
        
        Time complexity: O(n) where n = number of files
        
        Args:
            codebase: Dict mapping file paths to AST trees
            
        Returns:
            Dict of cluster_id -> CodeCluster
        """
        clusters = defaultdict(lambda: CodeCluster(
            cluster_id="",
            pattern_type="",
            members=[]
        ))
        
        # Single pass through codebase - O(n)
        for file_path, ast_tree in codebase.items():
            # Extract all functions and classes
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    # Determine cluster membership based on patterns
                    cluster_id = self._identify_cluster_membership(node, file_path)
                    
                    # Add to appropriate cluster
                    if cluster_id not in clusters:
                        clusters[cluster_id] = CodeCluster(
                            cluster_id=cluster_id,
                            pattern_type=self._get_pattern_type(cluster_id),
                            members=[]
                        )
                    
                    clusters[cluster_id].members.append((file_path, node))
        
        # Calculate cluster statistics (still O(n) total)
        for cluster_id, cluster in clusters.items():
            self._calculate_cluster_stats(cluster)
        
        self.clusters = dict(clusters)
        return self.clusters
    
    def _identify_cluster_membership(self, node: ast.AST, file_path: str) -> str:
        """
        Determine which cluster a code node belongs to.
        
        Uses pattern matching, NOT dependency analysis.
        """
        # Get node name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = node.name.lower()
            
            # Check prefix patterns
            for pattern_type, prefixes in self.PATTERN_PREFIXES.items():
                for prefix in prefixes:
                    if name.startswith(prefix):
                        return f"{pattern_type}_{prefix}"
            
            # Check structural patterns
            structure_type = self._identify_structure_pattern(node)
            if structure_type:
                return f"structural_{structure_type}"
            
            # Check behavioral patterns
            behavior_type = self._identify_behavioral_pattern(node)
            if behavior_type:
                return f"behavioral_{behavior_type}"
        
        elif isinstance(node, ast.ClassDef):
            # Class-specific clustering
            if self._is_exception_class(node):
                return "error_definitions"
            elif self._is_dataclass(node):
                return "data_models"
            elif self._has_pattern_methods(node, ['__enter__', '__exit__']):
                return "context_managers"
            elif self._has_pattern_methods(node, ['__iter__', '__next__']):
                return "iterators"
        
        # Default cluster based on file path
        if 'test' in file_path.lower():
            return "test_code"
        elif 'util' in file_path.lower():
            return "utilities_general"
        else:
            return "uncategorized"
    
    def _identify_structure_pattern(self, node: ast.FunctionDef) -> Optional[str]:
        """Identify structural patterns in function."""
        # Check for error handling pattern
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                return "error_handler"
            elif isinstance(child, ast.Yield):
                return "generator"
        
        # Check for decorator pattern (returns function)
        for stmt in node.body:
            if isinstance(stmt, ast.Return):
                if isinstance(stmt.value, (ast.FunctionDef, ast.Lambda)):
                    return "decorator"
        
        # Check for recursive pattern
        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id == node.name:
                    return "recursive"
        
        return None
    
    def _identify_behavioral_pattern(self, node: ast.FunctionDef) -> Optional[str]:
        """Identify behavioral patterns in function."""
        # Count different types of operations
        ops_count = defaultdict(int)
        
        for child in ast.walk(node):
            if isinstance(child, ast.If):
                ops_count['conditional'] += 1
            elif isinstance(child, ast.For):
                ops_count['iteration'] += 1
            elif isinstance(child, ast.While):
                ops_count['loop'] += 1
            elif isinstance(child, ast.Call):
                ops_count['call'] += 1
            elif isinstance(child, ast.Return):
                ops_count['return'] += 1
        
        # Classify based on dominant operation
        if ops_count['conditional'] > 3:
            return "decision_heavy"
        elif ops_count['iteration'] > 2:
            return "iteration_heavy"
        elif ops_count['call'] > 5:
            return "orchestrator"
        elif ops_count['return'] > 2:
            return "multi_exit"
        
        return None
    
    def _is_exception_class(self, node: ast.ClassDef) -> bool:
        """Check if class is an exception."""
        for base in node.bases:
            if isinstance(base, ast.Name):
                if 'error' in base.id.lower() or 'exception' in base.id.lower():
                    return True
        return False
    
    def _is_dataclass(self, node: ast.ClassDef) -> bool:
        """Check if class is a dataclass."""
        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == 'dataclass':
                return True
        return False
    
    def _has_pattern_methods(self, node: ast.ClassDef, methods: List[str]) -> bool:
        """Check if class has specific methods."""
        class_methods = {n.name for n in node.body if isinstance(n, ast.FunctionDef)}
        return all(m in class_methods for m in methods)
    
    def _get_pattern_type(self, cluster_id: str) -> str:
        """Extract pattern type from cluster ID."""
        parts = cluster_id.split('_', 1)
        return parts[0] if parts else "unknown"
    
    def _calculate_cluster_stats(self, cluster: CodeCluster) -> None:
        """Calculate statistics for a cluster."""
        if not cluster.members:
            return
        
        # Calculate average size
        total_size = 0
        for _, node in cluster.members:
            total_size += len(list(ast.walk(node)))
        
        cluster.avg_size = total_size / len(cluster.members)
        
        # If semantic index available, calculate centroid
        if self.semantic_index:
            vectors = []
            for file_path, node in cluster.members[:10]:  # Sample for efficiency
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    vector = self._get_or_compute_vector(file_path, node)
                    if vector is not None:
                        vectors.append(vector)
            
            if vectors:
                cluster.centroid_vector = np.mean(vectors, axis=0)
                
                # Calculate average intra-cluster similarity
                similarities = []
                for v in vectors:
                    sim = np.dot(v, cluster.centroid_vector) / (
                        np.linalg.norm(v) * np.linalg.norm(cluster.centroid_vector)
                    )
                    similarities.append(sim)
                cluster.avg_similarity = np.mean(similarities)
    
    def _get_or_compute_vector(self, file_path: str, node: ast.AST) -> Optional[np.ndarray]:
        """Get or compute hypervector for a node."""
        cache_key = f"{file_path}:{node.lineno}"
        
        if cache_key in self._vector_cache:
            return self._vector_cache[cache_key]
        
        try:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                vector = encode_function(node)
                self._vector_cache[cache_key] = vector
                return vector
        except Exception as e:
            self.logger.debug(f"Failed to encode function: {e}")
        
        return None
    
    def find_influential_patterns(self, clusters: Dict[str, CodeCluster]) -> List[InfluentialPattern]:
        """
        Find code patterns that influence many others.
        
        Uses Bellman-Ford-style limited exploration (not full sorting).
        """
        influential = []
        
        for cluster_id, cluster in clusters.items():
            if len(cluster.members) < 3:
                continue  # Skip small clusters
            
            # Scout for influence using limited steps
            influence_score = self._scout_influence_bellman_ford_style(
                cluster,
                max_steps=3  # Limited exploration
            )
            
            if influence_score > 0.5:  # Threshold for influence
                pattern = InfluentialPattern(
                    pattern_id=cluster_id,
                    pattern_type=cluster.pattern_type,
                    examples=cluster.members[:5],  # Keep top examples
                    affected_clusters=self._find_affected_clusters(cluster_id),
                    influence_metric=influence_score
                )
                influential.append(pattern)
        
        # Sort by influence (small sort, not full codebase)
        influential.sort(key=lambda p: p.influence_metric, reverse=True)
        
        self.influential_patterns = influential[:20]  # Keep top 20
        return self.influential_patterns
    
    def _scout_influence_bellman_ford_style(self, cluster: CodeCluster, max_steps: int) -> float:
        """
        Scout influence using limited exploration (Bellman-Ford style).
        
        Instead of full analysis, we take a few steps to estimate influence.
        """
        influence = 0.0
        
        # Factor 1: Size (more members = more influential)
        size_factor = min(len(cluster.members) / 10.0, 1.0)
        influence += size_factor * 0.3
        
        # Factor 2: Pattern type (some patterns are inherently influential)
        influential_types = {'utilities', 'validators', 'error_handler', 'data_handlers'}
        if any(t in cluster.pattern_type for t in influential_types):
            influence += 0.3
        
        # Factor 3: Structural complexity (complex patterns influence more)
        avg_complexity = 0
        for _, node in cluster.members[:max_steps]:  # Limited sampling
            complexity = len(list(ast.walk(node)))
            avg_complexity += complexity
        
        if cluster.members:
            avg_complexity /= min(len(cluster.members), max_steps)
            complexity_factor = min(avg_complexity / 50.0, 1.0)
            influence += complexity_factor * 0.2
        
        # Factor 4: Semantic centrality (if available)
        if cluster.avg_similarity > 0:
            influence += cluster.avg_similarity * 0.2
        
        return influence
    
    def _find_affected_clusters(self, cluster_id: str) -> Set[str]:
        """Find clusters potentially affected by this one."""
        affected = set()
        
        pattern_type = self._get_pattern_type(cluster_id)
        
        # Define influence relationships
        influence_map = {
            'validators': {'data_handlers', 'transformers', 'controllers'},
            'error_handler': {'controllers', 'data_handlers', 'writers'},
            'utilities': set(self.clusters.keys()),  # Affects everything
            'data_models': {'validators', 'transformers', 'builders'},
            'initializers': {'controllers', 'builders', 'data_handlers'},
        }
        
        affected = influence_map.get(pattern_type, set())
        
        # Filter to existing clusters
        return affected & set(self.clusters.keys())
    
    def analyze_influential_first(self, influential_patterns: List[InfluentialPattern]) -> List[Issue]:
        """
        Analyze influential patterns first (they affect many others).
        """
        issues = []
        
        for pattern in influential_patterns:
            # Analyze pattern for common issues
            pattern_issues = self._analyze_pattern(pattern)
            issues.extend(pattern_issues)
            
            # Learn from these issues for adaptive analysis
            self._update_learning_cache(pattern, pattern_issues)
        
        return issues
    
    def _analyze_pattern(self, pattern: InfluentialPattern) -> List[Issue]:
        """Analyze a specific pattern for issues."""
        issues = []
        
        # Check for duplicate implementations within pattern
        if len(pattern.examples) > 1:
            duplicates = self._find_duplicates_in_pattern(pattern)
            issues.extend(duplicates)
        
        # Check for anti-patterns
        for file_path, node in pattern.examples:
            anti_patterns = self._check_anti_patterns(node, pattern.pattern_type)
            for issue in anti_patterns:
                issue.file = file_path
                issues.append(issue)
        
        return issues
    
    def _find_duplicates_in_pattern(self, pattern: InfluentialPattern) -> List[Issue]:
        """Find duplicate implementations within a pattern."""
        issues = []
        
        # Compare pairs for structural similarity
        for i, (file1, node1) in enumerate(pattern.examples):
            for file2, node2 in pattern.examples[i+1:]:
                if self._are_structurally_similar(node1, node2):
                    issue = Issue(
                        kind="semantic_duplicate",
                        message=f"Duplicate {pattern.pattern_type} pattern",
                        severity=3,
                        file=file1,
                        line=node1.lineno if hasattr(node1, 'lineno') else 0,
                        evidence={
                            "pattern_type": pattern.pattern_type,
                            "duplicate_file": file2,
                            "cluster": pattern.pattern_id
                        }
                    )
                    issues.append(issue)
        
        return issues
    
    def _are_structurally_similar(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Check if two nodes are structurally similar."""
        # Simple structural comparison
        dump1 = ast.dump(node1, annotate_fields=False, include_attributes=False)
        dump2 = ast.dump(node2, annotate_fields=False, include_attributes=False)
        
        # Normalize for comparison
        dump1 = re.sub(r"Name\(id='[^']+'\)", "Name(id='VAR')", dump1)
        dump2 = re.sub(r"Name\(id='[^']+'\)", "Name(id='VAR')", dump2)
        
        # Calculate similarity
        from difflib import SequenceMatcher
        similarity = SequenceMatcher(None, dump1, dump2).ratio()
        
        return similarity > 0.8
    
    def _check_anti_patterns(self, node: ast.AST, pattern_type: str) -> List[Issue]:
        """Check for anti-patterns specific to pattern type."""
        issues = []
        
        if pattern_type == "error_handler":
            # Check for empty except blocks
            for child in ast.walk(node):
                if isinstance(child, ast.ExceptHandler):
                    if len(child.body) == 1 and isinstance(child.body[0], ast.Pass):
                        issues.append(Issue(
                            kind="empty_except",
                            message="Empty except block suppresses errors",
                            severity=3,
                            line=child.lineno if hasattr(child, 'lineno') else 0
                        ))
        
        elif "validator" in pattern_type:
            # Check for missing return statements
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            if not returns and isinstance(node, ast.FunctionDef):
                issues.append(Issue(
                    kind="missing_return",
                    message="Validator function missing return statement",
                    severity=2,
                    line=node.lineno if hasattr(node, 'lineno') else 0
                ))
        
        return issues
    
    def adaptive_cluster_analysis(self, clusters: Dict[str, CodeCluster],
                                 learned_patterns: Dict[str, Any]) -> List[Issue]:
        """
        Adaptively analyze remaining clusters using learned patterns.
        """
        issues = []
        
        for cluster_id, cluster in clusters.items():
            # Skip if already analyzed as influential
            if any(p.pattern_id == cluster_id for p in self.influential_patterns):
                continue
            
            # Apply learned patterns
            cluster_issues = self._apply_learned_patterns(cluster, learned_patterns)
            issues.extend(cluster_issues)
            
            # Quick scan for obvious issues
            quick_issues = self._quick_scan_cluster(cluster)
            issues.extend(quick_issues)
        
        return issues
    
    def _apply_learned_patterns(self, cluster: CodeCluster, 
                                learned_patterns: Dict[str, Any]) -> List[Issue]:
        """Apply patterns learned from influential code."""
        issues = []
        
        # Check if cluster matches any problematic patterns
        for pattern_key, pattern_data in learned_patterns.items():
            if self._matches_learned_pattern(cluster, pattern_data):
                for file_path, node in cluster.members:
                    issue = Issue(
                        kind="learned_pattern",
                        message=f"Matches problematic pattern: {pattern_key}",
                        severity=2,
                        file=file_path,
                        line=node.lineno if hasattr(node, 'lineno') else 0,
                        evidence={
                            "pattern": pattern_key,
                            "cluster": cluster.cluster_id
                        }
                    )
                    issues.append(issue)
        
        return issues
    
    def _matches_learned_pattern(self, cluster: CodeCluster, pattern_data: Dict) -> bool:
        """Check if cluster matches a learned pattern."""
        # Simple matching based on pattern type
        return cluster.pattern_type == pattern_data.get('type', '')
    
    def _quick_scan_cluster(self, cluster: CodeCluster) -> List[Issue]:
        """Quick scan for obvious issues without deep analysis."""
        issues = []
        
        # Check for oversized clusters (too many similar functions)
        if len(cluster.members) > 20:
            issue = Issue(
                kind="pattern_proliferation",
                message=f"Too many {cluster.pattern_type} functions ({len(cluster.members)})",
                severity=2,
                file=cluster.members[0][0] if cluster.members else "",
                evidence={
                    "count": len(cluster.members),
                    "pattern": cluster.pattern_type
                }
            )
            issues.append(issue)
        
        return issues
    
    def _extract_patterns(self, issues: List[Issue]) -> Dict[str, Any]:
        """Extract patterns from issues for learning."""
        patterns = defaultdict(dict)
        
        for issue in issues:
            pattern_key = f"{issue.kind}_{issue.evidence.get('pattern_type', '')}"
            patterns[pattern_key]['count'] = patterns[pattern_key].get('count', 0) + 1
            patterns[pattern_key]['severity'] = max(
                patterns[pattern_key].get('severity', 0),
                issue.severity
            )
            patterns[pattern_key]['type'] = issue.evidence.get('pattern_type', '')
        
        return dict(patterns)
    
    def _update_learning_cache(self, pattern: InfluentialPattern, issues: List[Issue]) -> None:
        """Update learning cache with pattern analysis results."""
        # Store results for adaptive analysis
        cache_key = pattern.pattern_id
        self._learning_cache = getattr(self, '_learning_cache', {})
        self._learning_cache[cache_key] = {
            'issues': len(issues),
            'pattern_type': pattern.pattern_type,
            'influence': pattern.influence_metric
        }
    
    def merge_results(self, priority_issues: List[Issue], 
                     remaining_issues: List[Issue]) -> List[Issue]:
        """Merge and deduplicate issues from different analysis phases."""
        all_issues = priority_issues + remaining_issues
        
        # Deduplicate based on file+line+kind
        seen = set()
        unique_issues = []
        
        for issue in all_issues:
            key = (issue.file, issue.line, issue.kind)
            if key not in seen:
                seen.add(key)
                unique_issues.append(issue)
        
        # Sort by severity
        unique_issues.sort(key=lambda i: i.severity, reverse=True)
        
        return unique_issues
    
    def get_cluster_summary(self) -> Dict[str, Any]:
        """Get summary statistics about clusters."""
        return {
            'total_clusters': len(self.clusters),
            'cluster_sizes': {
                cid: len(c.members) 
                for cid, c in self.clusters.items()
            },
            'influential_patterns': len(self.influential_patterns),
            'avg_cluster_similarity': np.mean([
                c.avg_similarity 
                for c in self.clusters.values() 
                if c.avg_similarity > 0
            ]) if self.clusters else 0,
            'pattern_distribution': defaultdict(int, {
                self._get_pattern_type(cid): len(c.members)
                for cid, c in self.clusters.items()
            })
        }