"""
Non-linear navigation for breaking traditional import-chain traversal patterns.

This module implements navigation strategies that prioritize influential nodes
over linear dependency following, enabling more efficient analysis of complex
codebases by jumping to "major intersections" first.

Key differences from traditional navigation:
- Traditional: Follow imports A→B→C→D linearly
- NonLinear: Jump A→D if D is influential, analyze B,C later
- Uses priority queue based on influence, not distance
- Starts from most influential node, not main.py or entry points
"""

import ast
import heapq
import time
import networkx as nx
from typing import Dict, List, Set, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum

from ..core.issues import Issue
from ..utils.logging_setup import get_logger
from .influence_detector import InfluenceDetector


class NavigationStrategy(Enum):
    """Available navigation strategies."""
    INFLUENCE_FIRST = "influence_first"    # Start with highest influence
    CLUSTER_HOPPING = "cluster_hopping"    # Jump between influential clusters
    CASCADE_AWARE = "cascade_aware"        # Prioritize cascade-fixing potential
    HYBRID_SEARCH = "hybrid_search"        # Combine multiple strategies


@dataclass
class NavigationNode:
    """Node in the navigation priority queue."""
    node_id: str
    file_path: str
    ast_node: ast.AST
    influence_score: float
    priority: float
    cluster_memberships: Set[str] = field(default_factory=set)
    error_correlation: float = 0.0
    is_pattern_template: bool = False
    analysis_cost: float = 0.0
    visited_at: Optional[float] = None

    def __lt__(self, other):
        """For priority queue ordering (higher priority = lower value)."""
        return self.priority > other.priority  # Reversed for max-heap behavior


@dataclass
class NavigationPlan:
    """Complete navigation plan for codebase analysis."""
    ordered_nodes: List[NavigationNode]
    skip_reasons: Dict[str, str]  # node_id -> reason for skipping
    cluster_order: List[str]
    estimated_time: float
    influence_coverage: float


@dataclass
class FixCluster:
    """Cluster of related issues that should be fixed together."""
    cluster_id: str
    primary_issue: Issue
    dependent_issues: List[Issue]
    influence_score: float
    cascade_potential: float
    fix_complexity: float
    estimated_benefit: float


class NonLinearNavigator:
    """
    Navigator that breaks traditional linear traversal patterns.
    
    Instead of following import chains sequentially, jumps to nodes
    based on influence and priority, analyzing high-impact code first.
    """
    
    # Priority adjustment factors
    CLUSTER_MEMBERSHIP_BOOST = 0.20      # +20% for multi-cluster nodes
    ERROR_CORRELATION_BOOST = 0.30       # +30% for error-prone nodes
    PATTERN_TEMPLATE_BOOST = 0.25        # +25% for pattern templates
    SIMILAR_PATTERN_PENALTY = -0.50      # -50% for already-analyzed patterns
    LOW_COMPLEXITY_PENALTY = -0.20       # -20% for low complexity nodes
    
    def __init__(self, influence_detector: Optional[InfluenceDetector] = None):
        """
        Initialize the non-linear navigator.
        
        Args:
            influence_detector: Optional influence detector for enhanced navigation
        """
        self.logger = get_logger("nonlinear_navigator")
        self.influence_detector = influence_detector or InfluenceDetector()
        
        # Navigation state
        self.priority_queue: List[NavigationNode] = []
        self.visited_patterns: Set[str] = set()
        self.visited_nodes: Set[str] = set()
        self.cluster_cache: Dict[str, Set[str]] = {}
        
        # Analysis tracking
        self.navigation_metrics = defaultdict(float)
        self.pattern_signatures: Dict[str, str] = {}
        self.influence_cache: Dict[str, float] = {}
    
    def navigate_by_influence(self, 
                            codebase: Dict[str, ast.AST],
                            strategy: NavigationStrategy = NavigationStrategy.INFLUENCE_FIRST) -> NavigationPlan:
        """
        Navigate codebase by influence priority, not import chains.
        
        Args:
            codebase: Dict mapping file paths to AST trees
            strategy: Navigation strategy to use
            
        Returns:
            NavigationPlan with ordered nodes for analysis
        """
        self.logger.info(f"Starting non-linear navigation with {strategy.value} strategy")
        
        start_time = time.time()
        
        # Build influence-based priority queue
        self._build_influence_queue(codebase)
        
        # Apply chosen navigation strategy
        if strategy == NavigationStrategy.INFLUENCE_FIRST:
            ordered_nodes = self._navigate_influence_first()
        elif strategy == NavigationStrategy.CLUSTER_HOPPING:
            ordered_nodes = self._navigate_cluster_hopping()
        elif strategy == NavigationStrategy.CASCADE_AWARE:
            ordered_nodes = self._navigate_cascade_aware()
        else:  # HYBRID_SEARCH
            ordered_nodes = self._navigate_hybrid_search()
        
        # Calculate navigation metrics
        elapsed_time = time.time() - start_time
        plan = self._create_navigation_plan(ordered_nodes, elapsed_time)
        
        self.logger.info(
            f"Navigation complete: {len(ordered_nodes)} nodes ordered, "
            f"{plan.influence_coverage:.1%} influence coverage, {elapsed_time:.3f}s"
        )
        
        return plan
    
    def _build_influence_queue(self, codebase: Dict[str, ast.AST]) -> None:
        """Build priority queue based on node influence scores."""
        self.priority_queue.clear()
        
        # Extract all analyzable nodes
        all_nodes = []
        for file_path, ast_tree in codebase.items():
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    node_id = f"{file_path}:{getattr(node, 'name', f'line{node.lineno}')}"
                    all_nodes.append((node_id, file_path, node))
        
        self.logger.debug(f"Found {len(all_nodes)} analyzable nodes")
        
        # Calculate influence and priority for each node
        for node_id, file_path, ast_node in all_nodes:
            # Get base influence score
            influence_score = self._get_cached_influence(node_id, ast_node)
            
            # Calculate adjusted priority
            priority = self.calculate_influence_priority(
                node_id, file_path, ast_node, influence_score
            )
            
            # Create navigation node
            nav_node = NavigationNode(
                node_id=node_id,
                file_path=file_path,
                ast_node=ast_node,
                influence_score=influence_score,
                priority=priority
            )
            
            # Add to priority queue
            heapq.heappush(self.priority_queue, nav_node)
    
    def calculate_influence_priority(self, 
                                   node_id: str,
                                   file_path: str, 
                                   node: ast.AST,
                                   base_influence: float) -> float:
        """
        Calculate adjusted priority based on various factors.
        
        Priority boosts:
        - Node is in multiple clusters (+20%)
        - Node has high error correlation (+30%)
        - Node is a pattern template (+25%)
        
        Priority penalties:
        - Already analyzed similar pattern (-50%)
        - Low complexity (-20%)
        
        Args:
            node_id: Unique node identifier
            file_path: File containing the node
            node: AST node
            base_influence: Base influence score
            
        Returns:
            Adjusted priority score
        """
        adjusted_priority = base_influence
        
        # Check cluster memberships
        cluster_count = len(self._get_node_clusters(node_id, node))
        if cluster_count > 1:
            adjusted_priority *= (1.0 + self.CLUSTER_MEMBERSHIP_BOOST)
            self.logger.debug(f"{node_id} multi-cluster boost: {cluster_count} clusters")
        
        # Check error correlation
        error_correlation = self._calculate_error_correlation(node)
        if error_correlation > 0.7:
            adjusted_priority *= (1.0 + self.ERROR_CORRELATION_BOOST)
            self.logger.debug(f"{node_id} error correlation boost: {error_correlation:.2f}")
        
        # Check if it's a pattern template
        if self._is_pattern_template(node):
            adjusted_priority *= (1.0 + self.PATTERN_TEMPLATE_BOOST)
            self.logger.debug(f"{node_id} pattern template boost")
        
        # Penalty for similar patterns already analyzed
        pattern_sig = self._generate_pattern_signature(node)
        if pattern_sig in self.visited_patterns:
            adjusted_priority *= (1.0 + self.SIMILAR_PATTERN_PENALTY)
            self.logger.debug(f"{node_id} similar pattern penalty")
        
        # Penalty for low complexity
        complexity = self._estimate_complexity(node)
        if complexity < 3:
            adjusted_priority *= (1.0 + self.LOW_COMPLEXITY_PENALTY)
            self.logger.debug(f"{node_id} low complexity penalty: {complexity}")
        
        return adjusted_priority
    
    def _get_cached_influence(self, node_id: str, node: ast.AST) -> float:
        """Get cached influence score or compute it."""
        if node_id not in self.influence_cache:
            self.influence_cache[node_id] = self.influence_detector.scout_influence_bellman_ford_style(
                node_id, node, max_steps=3
            )
        return self.influence_cache[node_id]
    
    def _get_node_clusters(self, node_id: str, node: ast.AST) -> Set[str]:
        """Get cluster memberships for a node."""
        if node_id not in self.cluster_cache:
            clusters = set()
            
            # Check for various cluster types based on node characteristics
            if isinstance(node, ast.FunctionDef):
                name = node.name.lower()
                if name.startswith(('get_', 'fetch_')):
                    clusters.add('data_retrieval')
                elif name.startswith(('set_', 'update_')):
                    clusters.add('data_modification')
                elif name.startswith(('validate_', 'check_')):
                    clusters.add('validation')
                elif 'error' in name or 'exception' in name:
                    clusters.add('error_handling')
                
                # Check for structural patterns
                if any(isinstance(n, ast.Try) for n in ast.walk(node)):
                    clusters.add('exception_handling')
                if any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node)):
                    clusters.add('iteration_heavy')
                if len([n for n in ast.walk(node) if isinstance(n, ast.If)]) > 3:
                    clusters.add('decision_heavy')
            
            elif isinstance(node, ast.ClassDef):
                if any(isinstance(n, ast.FunctionDef) and n.name.startswith('__') 
                       for n in node.body):
                    clusters.add('protocol_implementation')
                if len(node.body) > 10:
                    clusters.add('large_class')
            
            self.cluster_cache[node_id] = clusters
        
        return self.cluster_cache[node_id]
    
    def _calculate_error_correlation(self, node: ast.AST) -> float:
        """Calculate correlation with error-prone patterns."""
        error_indicators = 0
        total_indicators = 0
        
        # Check for error-prone patterns
        for child in ast.walk(node):
            total_indicators += 1
            
            # Exception handling without specific catch
            if isinstance(child, ast.ExceptHandler) and not child.type:
                error_indicators += 2
            
            # Global variable usage
            elif isinstance(child, ast.Global):
                error_indicators += 1
            
            # Complex nested conditions
            elif isinstance(child, ast.If):
                depth = self._get_nesting_depth(child)
                if depth > 3:
                    error_indicators += 1
            
            # Mutable default arguments
            elif (isinstance(child, ast.FunctionDef) and 
                  any(isinstance(default, (ast.List, ast.Dict)) 
                      for default in child.args.defaults)):
                error_indicators += 2
            
            # String concatenation in loops
            elif (isinstance(child, (ast.For, ast.While)) and
                  any(isinstance(n, ast.Add) for n in ast.walk(child))):
                error_indicators += 1
        
        if total_indicators == 0:
            return 0.0
        
        return min(error_indicators / total_indicators, 1.0)
    
    def _is_pattern_template(self, node: ast.AST) -> bool:
        """Check if node represents a pattern template."""
        if not isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            return False
        
        # Check for template indicators
        name = getattr(node, 'name', '').lower()
        
        # Common template patterns
        template_patterns = [
            'template', 'base', 'abstract', 'interface', 'protocol',
            'factory', 'builder', 'strategy', 'visitor', 'observer'
        ]
        
        if any(pattern in name for pattern in template_patterns):
            return True
        
        # Check for abstract methods (functions with just pass/ellipsis)
        if isinstance(node, ast.ClassDef):
            abstract_methods = 0
            total_methods = 0
            
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    total_methods += 1
                    if (len(child.body) == 1 and 
                        isinstance(child.body[0], (ast.Pass, ast.Expr)) and
                        (isinstance(child.body[0], ast.Pass) or
                         isinstance(child.body[0].value, ast.Ellipsis))):
                        abstract_methods += 1
            
            if total_methods > 0 and abstract_methods / total_methods > 0.5:
                return True
        
        return False
    
    def _generate_pattern_signature(self, node: ast.AST) -> str:
        """Generate signature for pattern matching."""
        parts = [type(node).__name__]
        
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Function structure signature
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
                
            # Return pattern
            returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
            parts.append(f"returns:{len(returns)}")
            
        elif isinstance(node, ast.ClassDef):
            # Class structure signature
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            parts.append(f"methods:{len(methods)}")
            
            # Inheritance signature
            parts.append(f"bases:{len(node.bases)}")
        
        return ":".join(parts)
    
    def _estimate_complexity(self, node: ast.AST) -> int:
        """Estimate cyclomatic complexity of node."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
        
        return complexity
    
    def _get_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth."""
        def calculate_depth(n, current_depth=0):
            max_depth = current_depth
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                    child_depth = calculate_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
            return max_depth
        
        return calculate_depth(node)
    
    def _navigate_influence_first(self) -> List[NavigationNode]:
        """Navigate by processing highest influence nodes first."""
        ordered_nodes = []
        
        while self.priority_queue:
            node = heapq.heappop(self.priority_queue)
            
            # Skip if already processed similar pattern
            pattern_sig = self._generate_pattern_signature(node.ast_node)
            if pattern_sig in self.visited_patterns:
                continue
            
            # Mark as visited
            node.visited_at = time.time()
            ordered_nodes.append(node)
            self.visited_patterns.add(pattern_sig)
            self.visited_nodes.add(node.node_id)
            
            self.logger.debug(
                f"Selected {node.node_id} (influence: {node.influence_score:.1f}, "
                f"priority: {node.priority:.1f})"
            )
        
        return ordered_nodes
    
    def _navigate_cluster_hopping(self) -> List[NavigationNode]:
        """Navigate by hopping between influential clusters."""
        ordered_nodes = []
        processed_clusters = set()
        
        while self.priority_queue:
            # Find highest priority node from unprocessed cluster
            best_node = None
            best_cluster = None
            
            # Look through queue for node from unprocessed cluster
            temp_queue = []
            while self.priority_queue:
                node = heapq.heappop(self.priority_queue)
                temp_queue.append(node)
                
                node_clusters = self._get_node_clusters(node.node_id, node.ast_node)
                unprocessed_clusters = node_clusters - processed_clusters
                
                if unprocessed_clusters and not best_node:
                    best_node = node
                    best_cluster = next(iter(unprocessed_clusters))
                    break
            
            # Restore queue
            for node in temp_queue:
                if node != best_node:
                    heapq.heappush(self.priority_queue, node)
            
            if not best_node:
                break
            
            # Process the selected node
            best_node.visited_at = time.time()
            ordered_nodes.append(best_node)
            processed_clusters.add(best_cluster)
            
            pattern_sig = self._generate_pattern_signature(best_node.ast_node)
            self.visited_patterns.add(pattern_sig)
            
            self.logger.debug(
                f"Cluster hop to {best_node.node_id} (cluster: {best_cluster})"
            )
        
        return ordered_nodes
    
    def _navigate_cascade_aware(self) -> List[NavigationNode]:
        """Navigate prioritizing nodes with high cascade fix potential."""
        ordered_nodes = []
        
        # Build dependency graph for cascade analysis
        cascade_graph = self._build_cascade_graph()
        
        while self.priority_queue:
            # Find node with highest cascade potential
            best_node = None
            best_cascade_score = 0
            temp_queue = []
            
            while self.priority_queue:
                node = heapq.heappop(self.priority_queue)
                temp_queue.append(node)
                
                cascade_score = self._calculate_cascade_potential(node, cascade_graph)
                
                if cascade_score > best_cascade_score:
                    if best_node:
                        heapq.heappush(self.priority_queue, best_node)
                    best_node = node
                    best_cascade_score = cascade_score
                else:
                    heapq.heappush(self.priority_queue, node)
                
                # Limit search to avoid excessive computation
                if len(temp_queue) > 20:
                    break
            
            if not best_node:
                break
            
            # Process selected node
            best_node.visited_at = time.time()
            ordered_nodes.append(best_node)
            
            pattern_sig = self._generate_pattern_signature(best_node.ast_node)
            self.visited_patterns.add(pattern_sig)
            
            self.logger.debug(
                f"Cascade-aware selection: {best_node.node_id} "
                f"(cascade score: {best_cascade_score:.2f})"
            )
        
        return ordered_nodes
    
    def _navigate_hybrid_search(self) -> List[NavigationNode]:
        """Navigate using hybrid approach combining strategies."""
        ordered_nodes = []
        
        # Phase 1: Process top 25% by pure influence
        phase1_count = max(1, len(self.priority_queue) // 4)
        for _ in range(min(phase1_count, len(self.priority_queue))):
            if not self.priority_queue:
                break
            node = heapq.heappop(self.priority_queue)
            node.visited_at = time.time()
            ordered_nodes.append(node)
            
            pattern_sig = self._generate_pattern_signature(node.ast_node)
            self.visited_patterns.add(pattern_sig)
        
        # Phase 2: Cluster hopping for next 50%
        phase2_nodes = self._navigate_cluster_hopping()
        ordered_nodes.extend(phase2_nodes[:len(self.priority_queue) // 2])
        
        # Phase 3: Cascade-aware for remaining
        remaining_nodes = self._navigate_cascade_aware()
        ordered_nodes.extend(remaining_nodes)
        
        return ordered_nodes
    
    def _build_cascade_graph(self) -> nx.DiGraph:
        """Build graph for cascade analysis."""
        graph = nx.DiGraph()
        
        # Add all nodes from priority queue
        for node in self.priority_queue:
            graph.add_node(node.node_id, node_data=node)
        
        # Add edges based on potential dependencies
        # (Simplified - in real implementation would analyze call patterns)
        for node in self.priority_queue:
            # Look for function calls within the node
            for child in ast.walk(node.ast_node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        called_name = child.func.id
                        # Try to find matching node
                        for other_node in self.priority_queue:
                            if (hasattr(other_node.ast_node, 'name') and
                                other_node.ast_node.name == called_name):
                                graph.add_edge(node.node_id, other_node.node_id)
        
        return graph
    
    def _calculate_cascade_potential(self, node: NavigationNode, cascade_graph: nx.DiGraph) -> float:
        """Calculate potential for fixing cascading issues."""
        if node.node_id not in cascade_graph:
            return 0.0
        
        # Count downstream dependencies
        try:
            downstream = len(nx.descendants(cascade_graph, node.node_id))
        except (nx.NetworkXError, KeyError, AttributeError) as e:
            self.logger.debug(f"Failed to calculate downstream dependencies for {node.node_id}: {e}")
            downstream = 0
        except Exception as e:
            self.logger.warning(f"Unexpected error calculating downstream dependencies for {node.node_id}: {e}")
            downstream = 0
        
        # Count upstream dependencies  
        try:
            upstream = len(nx.ancestors(cascade_graph, node.node_id))
        except (nx.NetworkXError, KeyError, AttributeError) as e:
            self.logger.debug(f"Failed to calculate upstream dependencies for {node.node_id}: {e}")
            upstream = 0
        except Exception as e:
            self.logger.warning(f"Unexpected error calculating upstream dependencies for {node.node_id}: {e}")
            upstream = 0
        
        # Weight downstream more heavily (fixing this affects more)
        cascade_score = downstream * 2.0 + upstream * 1.0
        
        # Boost if node is in multiple clusters
        cluster_count = len(self._get_node_clusters(node.node_id, node.ast_node))
        cascade_score *= (1.0 + cluster_count * 0.1)
        
        return cascade_score
    
    def _create_navigation_plan(self, ordered_nodes: List[NavigationNode], elapsed_time: float) -> NavigationPlan:
        """Create comprehensive navigation plan."""
        # Calculate influence coverage
        total_influence = sum(node.influence_score for node in ordered_nodes)
        max_possible_influence = sum(self.influence_cache.values()) if self.influence_cache else 1
        influence_coverage = total_influence / max_possible_influence if max_possible_influence > 0 else 0
        
        # Determine cluster order
        cluster_order = []
        seen_clusters = set()
        for node in ordered_nodes:
            node_clusters = self._get_node_clusters(node.node_id, node.ast_node)
            for cluster in node_clusters:
                if cluster not in seen_clusters:
                    cluster_order.append(cluster)
                    seen_clusters.add(cluster)
        
        # Identify skipped patterns
        skip_reasons = {}
        for pattern in self.pattern_signatures:
            if pattern not in self.visited_patterns:
                skip_reasons[pattern] = "Similar pattern already analyzed"
        
        return NavigationPlan(
            ordered_nodes=ordered_nodes,
            skip_reasons=skip_reasons,
            cluster_order=cluster_order,
            estimated_time=elapsed_time,
            influence_coverage=influence_coverage
        )


class InfluenceBasedFixPrioritizer:
    """
    Identify and fix the most "influential" issues first - 
    those that cause or affect many other issues.
    
    Inspired by: Finding nodes that are like "intersections of major thoroughfares"
    """
    
    def __init__(self, influence_detector: Optional[InfluenceDetector] = None):
        """Initialize the fix prioritizer."""
        self.logger = get_logger("fix_prioritizer")
        self.influence_detector = influence_detector or InfluenceDetector()
        self.issue_graph = nx.DiGraph()
        
    def prioritize_fixes_by_influence(self, all_issues: List[Issue]) -> List[FixCluster]:
        """
        Don't fix issues in order of severity or occurrence.
        Fix the influential ones that will cascade to fix others.
        
        Args:
            all_issues: All detected issues
            
        Returns:
            List of fix clusters ordered by cascade potential
        """
        self.logger.info(f"Prioritizing {len(all_issues)} issues by influence")
        
        # Build influence graph
        influence_graph = self.build_issue_influence_graph(all_issues)
        
        # Find "major intersections" - issues that affect many paths
        influential_issues = self.find_issue_intersections(influence_graph)
        
        # Group issues into fix clusters
        fix_clusters = self.create_fix_clusters(influential_issues, all_issues)
        
        # Order clusters by cascade potential, not severity
        prioritized_plan = self.order_by_cascade_potential(fix_clusters)
        
        self.logger.info(f"Created {len(prioritized_plan)} fix clusters")
        
        return prioritized_plan
    
    def build_issue_influence_graph(self, issues: List[Issue]) -> nx.DiGraph:
        """Build graph showing how issues influence each other."""
        graph = nx.DiGraph()
        
        # Add all issues as nodes
        for issue in issues:
            issue_id = f"{issue.file}:{issue.line}"
            graph.add_node(issue_id, issue_data=issue)
        
        # Add edges based on issue relationships
        for i, issue1 in enumerate(issues):
            issue1_id = f"{issue1.file}:{issue1.line}"
            
            for j, issue2 in enumerate(issues):
                if i == j:
                    continue
                
                issue2_id = f"{issue2.file}:{issue2.line}"
                
                # Check if issue1 could cause issue2
                if self._issues_related(issue1, issue2):
                    graph.add_edge(issue1_id, issue2_id)
        
        return graph
    
    def _issues_related(self, issue1: Issue, issue2: Issue) -> bool:
        """Check if two issues are related/connected."""
        # Same file - likely related
        if issue1.file == issue2.file:
            return True
        
        # Same function name in different files
        if (hasattr(issue1, 'symbol') and hasattr(issue2, 'symbol') and
            issue1.symbol == issue2.symbol):
            return True
        
        # Same issue type - could be pattern
        if issue1.kind == issue2.kind:
            return True
        
        # Import-related issues
        if ('import' in issue1.kind.lower() or 'import' in issue2.kind.lower()):
            return True
        
        return False
    
    def find_issue_intersections(self, influence_graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Find issues that, when fixed, will eliminate many other issues.
        These are our "major thoroughfares" in the issue graph.
        
        Args:
            influence_graph: Graph of issue relationships
            
        Returns:
            List of influential issue intersections
        """
        intersections = []
        
        for issue_node in influence_graph.nodes():
            # Count how many issue paths pass through this node
            paths_through = self.count_paths_through_node(
                influence_graph, 
                issue_node
            )
            
            # Count downstream effects
            cascade_potential = self.calculate_cascade_potential(
                influence_graph,
                issue_node
            )
            
            # Minimum threshold for intersection
            threshold = max(2, len(influence_graph.nodes()) * 0.1)
            
            if paths_through > threshold or cascade_potential > threshold:
                intersections.append({
                    'issue_id': issue_node,
                    'issue': influence_graph.nodes[issue_node]['issue_data'],
                    'influence_score': paths_through * cascade_potential,
                    'paths_through': paths_through,
                    'cascade_potential': cascade_potential,
                    'fixing_this_eliminates': self.find_dependent_issues(
                        influence_graph, 
                        issue_node
                    )
                })
        
        return sorted(intersections, key=lambda x: x['influence_score'], reverse=True)
    
    def count_paths_through_node(self, graph: nx.DiGraph, node: str) -> int:
        """Count how many paths pass through a given node."""
        paths_count = 0
        
        # Get predecessors and successors
        predecessors = list(graph.predecessors(node))
        successors = list(graph.successors(node))
        
        # Each combination of predecessor->node->successor is a path through
        paths_count = len(predecessors) * len(successors)
        
        # Also count direct in-degree (things that depend on this)
        paths_count += graph.in_degree(node)
        
        return paths_count
    
    def calculate_cascade_potential(self, graph: nx.DiGraph, node: str) -> float:
        """Calculate potential cascade effect of fixing this issue."""
        if node not in graph:
            return 0.0
        
        # Count all descendants (issues that would be affected)
        try:
            descendants = nx.descendants(graph, node)
            cascade_score = len(descendants)
        except (nx.NetworkXError, KeyError, AttributeError) as e:
            self.logger.debug(f"Failed to calculate issue descendants for {node}: {e}")
            cascade_score = 0
        except Exception as e:
            self.logger.warning(f"Unexpected error calculating issue cascade for {node}: {e}")
            cascade_score = 0
        
        # Weight by issue severity
        issue = graph.nodes[node]['issue_data']
        severity_weight = getattr(issue, 'severity', 1) / 5.0  # Normalize to 0-1
        
        return cascade_score * severity_weight
    
    def find_dependent_issues(self, graph: nx.DiGraph, node: str) -> List[str]:
        """Find all issues that depend on this one."""
        try:
            return list(nx.descendants(graph, node))
        except (nx.NetworkXError, KeyError, AttributeError) as e:
            self.logger.debug(f"Failed to find dependent issues for {node}: {e}")
            return []
        except Exception as e:
            self.logger.warning(f"Unexpected error finding dependent issues for {node}: {e}")
            return []
    
    def create_fix_clusters(self, 
                           influential_issues: List[Dict[str, Any]],
                           all_issues: List[Issue]) -> List[FixCluster]:
        """Group issues into clusters that should be fixed together."""
        clusters = []
        processed_issues = set()
        
        for intersection in influential_issues[:20]:  # Top 20 intersections
            if intersection['issue_id'] in processed_issues:
                continue
            
            primary_issue = intersection['issue']
            dependent_issue_ids = intersection['fixing_this_eliminates']
            
            # Find actual issue objects for dependents
            dependent_issues = []
            for issue in all_issues:
                issue_id = f"{issue.file}:{issue.line}"
                if issue_id in dependent_issue_ids:
                    dependent_issues.append(issue)
                    processed_issues.add(issue_id)
            
            # Create fix cluster
            cluster = FixCluster(
                cluster_id=f"cluster_{len(clusters)}",
                primary_issue=primary_issue,
                dependent_issues=dependent_issues,
                influence_score=intersection['influence_score'],
                cascade_potential=intersection['cascade_potential'],
                fix_complexity=self._estimate_fix_complexity(primary_issue, dependent_issues),
                estimated_benefit=self._estimate_fix_benefit(primary_issue, dependent_issues)
            )
            
            clusters.append(cluster)
            processed_issues.add(intersection['issue_id'])
        
        return clusters
    
    def _estimate_fix_complexity(self, primary_issue: Issue, dependent_issues: List[Issue]) -> float:
        """Estimate complexity of fixing this cluster."""
        base_complexity = 1.0
        
        # More dependent issues = higher complexity
        complexity = base_complexity + len(dependent_issues) * 0.2
        
        # Cross-file fixes are more complex
        files_involved = len(set([primary_issue.file] + 
                                [issue.file for issue in dependent_issues]))
        complexity += files_involved * 0.3
        
        return complexity
    
    def _estimate_fix_benefit(self, primary_issue: Issue, dependent_issues: List[Issue]) -> float:
        """Estimate benefit of fixing this cluster."""
        # Base benefit from primary issue severity
        benefit = getattr(primary_issue, 'severity', 3)
        
        # Additional benefit from each dependent issue
        for dep_issue in dependent_issues:
            benefit += getattr(dep_issue, 'severity', 2) * 0.5
        
        # Normalize to 0-1 scale
        return min(benefit / 20.0, 1.0)
    
    def order_by_cascade_potential(self, clusters: List[FixCluster]) -> List[FixCluster]:
        """Order clusters by their cascade fixing potential."""
        # Sort by benefit/complexity ratio and cascade potential
        def cluster_priority(cluster):
            efficiency = cluster.estimated_benefit / max(cluster.fix_complexity, 0.1)
            return efficiency * cluster.cascade_potential
        
        return sorted(clusters, key=cluster_priority, reverse=True)