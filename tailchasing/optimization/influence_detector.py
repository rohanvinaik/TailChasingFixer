"""
Influence detection for identifying critical code patterns.

This module implements algorithms to find "major intersections" in code -
patterns that many other parts depend on or duplicate. Uses a Bellman-Ford
inspired scouting approach to efficiently estimate influence without full analysis.
"""

import ast
import networkx as nx
from typing import Dict, List, Set, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field

from ..core.symbols import SymbolTable
from ..utils.logging_setup import get_logger


@dataclass
class InfluentialNode:
    """Represents a node with high influence in the codebase."""
    node_id: str  # file:name or file:line
    node_type: str  # function, class, pattern
    influence_score: float
    direct_calls: int = 0
    indirect_calls: int = 0
    duplicate_count: int = 0
    affected_files: Set[str] = field(default_factory=set)
    cascading_risk: float = 0.0
    pattern_signature: Optional[str] = None


@dataclass
class LayerInfo:
    """Information about a conceptual layer in the codebase."""
    name: str
    nodes: List[Tuple[str, ast.AST]]  # (file_path, node) pairs
    influence_score: float = 0.0
    cross_layer_dependencies: int = 0
    internal_cohesion: float = 0.0


class InfluenceDetector:
    """
    Detects influential patterns in code using graph-based analysis.
    
    Identifies "major intersections" - code that many paths pass through,
    using Bellman-Ford inspired limited exploration.
    """
    
    # Weights for influence calculation
    INFLUENCE_WEIGHTS = {
        'direct_calls': 2.0,      # Direct usage/calls
        'indirect_calls': 1.0,    # Indirect dependencies
        'duplicate_count': 3.0,   # Pattern duplication
        'error_propagation': 4.0, # Error source potential
        'cross_layer': 2.5,       # Cross-layer dependencies
        'api_boundary': 3.0,      # Public API exposure
    }
    
    def __init__(self, symbol_table: Optional[SymbolTable] = None):
        """
        Initialize the influence detector.
        
        Args:
            symbol_table: Optional symbol table for enhanced analysis
        """
        self.logger = get_logger("influence_detector")
        self.symbol_table = symbol_table
        self.call_graph = nx.DiGraph()
        self.pattern_graph = nx.Graph()  # For pattern similarity
        self.influential_nodes: List[InfluentialNode] = []
        self._node_cache: Dict[str, Any] = {}
        
    def find_influential_patterns(self, 
                                 codebase: Dict[str, ast.AST],
                                 top_percent: float = 0.1) -> List[InfluentialNode]:
        """
        Find the most influential patterns in the codebase.
        
        These are the "major intersections" that many code paths pass through.
        
        Args:
            codebase: Dict mapping file paths to AST trees
            top_percent: Return top X% most influential (default 10%)
            
        Returns:
            List of influential nodes sorted by influence score
        """
        self.logger.info(f"Finding influential patterns in {len(codebase)} files")
        
        # Build call graph from codebase
        self._build_call_graph(codebase)
        
        # Scout influence for all nodes
        all_nodes = []
        for file_path, ast_tree in codebase.items():
            nodes = self._extract_nodes(file_path, ast_tree)
            for node_id, node in nodes:
                influence_score = self.scout_influence_bellman_ford_style(
                    node_id, 
                    node,
                    max_steps=3
                )
                
                influential_node = InfluentialNode(
                    node_id=node_id,
                    node_type=self._get_node_type(node),
                    influence_score=influence_score
                )
                
                # Collect additional metrics
                self._collect_node_metrics(influential_node, node_id, node)
                all_nodes.append(influential_node)
        
        # Sort by influence and return top percent
        all_nodes.sort(key=lambda n: n.influence_score, reverse=True)
        
        top_count = max(1, int(len(all_nodes) * top_percent))
        self.influential_nodes = all_nodes[:top_count]
        
        self.logger.info(f"Identified {len(self.influential_nodes)} influential patterns")
        
        return self.influential_nodes
    
    def scout_influence_bellman_ford_style(self,
                                          node_id: str,
                                          node: ast.AST,
                                          max_steps: int = 3) -> float:
        """
        Scout influence using Bellman-Ford inspired limited exploration.
        
        Instead of full analysis, explore max_steps in the call graph
        to estimate influence efficiently.
        
        Args:
            node_id: Unique identifier for the node
            node: AST node to analyze
            max_steps: Maximum exploration depth (default 3)
            
        Returns:
            Influence score as float
        """
        # Initialize influence components
        direct_calls = 0
        indirect_calls = 0
        duplicate_count = 0
        
        # Direct calls (in-degree in call graph)
        if self.call_graph.has_node(node_id):
            direct_calls = self.call_graph.in_degree(node_id)
        
        # Limited BFS for indirect influence (Bellman-Ford style)
        if self.call_graph.has_node(node_id):
            visited = set()
            current_level = {node_id}
            
            for step in range(1, max_steps + 1):
                next_level = set()
                for current in current_level:
                    if current in visited:
                        continue
                    visited.add(current)
                    
                    # Count predecessors (who calls this)
                    predecessors = self.call_graph.predecessors(current)
                    for pred in predecessors:
                        if pred not in visited:
                            next_level.add(pred)
                            indirect_calls += 1 / step  # Weight by distance
                
                current_level = next_level
                
                if not current_level:
                    break  # No more nodes to explore
        
        # Check for duplicate patterns
        pattern_sig = self._get_pattern_signature(node)
        if pattern_sig:
            duplicate_count = self._count_pattern_duplicates(pattern_sig)
        
        # Calculate weighted influence score
        influence = (
            self.INFLUENCE_WEIGHTS['direct_calls'] * direct_calls +
            self.INFLUENCE_WEIGHTS['indirect_calls'] * indirect_calls +
            self.INFLUENCE_WEIGHTS['duplicate_count'] * duplicate_count
        )
        
        # Normalize to 0-100 scale
        influence = min(100, influence)
        
        return influence
    
    def _build_call_graph(self, codebase: Dict[str, ast.AST]) -> None:
        """Build a call graph from the codebase."""
        self.call_graph.clear()
        
        for file_path, ast_tree in codebase.items():
            # Add nodes for all functions and classes
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    node_id = f"{file_path}:{node.name}"
                    self.call_graph.add_node(node_id, file=file_path, name=node.name)
                    
                    # Find calls within this node
                    for child in ast.walk(node):
                        if isinstance(child, ast.Call):
                            called_name = self._extract_call_name(child)
                            if called_name:
                                # Try to resolve to a node ID
                                called_id = self._resolve_call(called_name, file_path)
                                if called_id and called_id != node_id:
                                    self.call_graph.add_edge(node_id, called_id)
    
    def _extract_call_name(self, call_node: ast.Call) -> Optional[str]:
        """Extract the name being called."""
        if isinstance(call_node.func, ast.Name):
            return call_node.func.id
        elif isinstance(call_node.func, ast.Attribute):
            return call_node.func.attr
        return None
    
    def _resolve_call(self, name: str, current_file: str) -> Optional[str]:
        """Try to resolve a call name to a node ID."""
        # First check current file
        current_file_id = f"{current_file}:{name}"
        if self.call_graph.has_node(current_file_id):
            return current_file_id
        
        # Check symbol table if available
        if self.symbol_table:
            # Look for the symbol in the symbol table
            for file, symbols in getattr(self.symbol_table, 'functions', {}).items():
                if name in symbols:
                    return f"{file}:{name}"
        
        # Check all nodes (expensive, but for small codebases OK)
        for node in self.call_graph.nodes():
            if node.endswith(f":{name}"):
                return node
        
        return None
    
    def _extract_nodes(self, file_path: str, ast_tree: ast.AST) -> List[Tuple[str, ast.AST]]:
        """Extract all analyzable nodes from an AST."""
        nodes = []
        
        for node in ast.walk(ast_tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                node_id = f"{file_path}:{node.name}"
                nodes.append((node_id, node))
            elif isinstance(node, ast.ClassDef):
                node_id = f"{file_path}:{node.name}"
                nodes.append((node_id, node))
        
        return nodes
    
    def _get_node_type(self, node: ast.AST) -> str:
        """Get the type of a node."""
        if isinstance(node, ast.FunctionDef):
            return "function"
        elif isinstance(node, ast.AsyncFunctionDef):
            return "async_function"
        elif isinstance(node, ast.ClassDef):
            return "class"
        else:
            return "unknown"
    
    def _get_pattern_signature(self, node: ast.AST) -> Optional[str]:
        """Generate a pattern signature for duplicate detection."""
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return None
        
        # Create signature from structure
        signature_parts = []
        
        # Add parameter count
        if hasattr(node, 'args'):
            signature_parts.append(f"args:{len(node.args.args)}")
        
        # Add control flow pattern
        has_try = any(isinstance(n, ast.Try) for n in ast.walk(node))
        has_loop = any(isinstance(n, (ast.For, ast.While)) for n in ast.walk(node))
        has_if = any(isinstance(n, ast.If) for n in ast.walk(node))
        
        if has_try:
            signature_parts.append("try")
        if has_loop:
            signature_parts.append("loop")
        if has_if:
            signature_parts.append("conditional")
        
        # Add return pattern
        returns = [n for n in ast.walk(node) if isinstance(n, ast.Return)]
        signature_parts.append(f"returns:{len(returns)}")
        
        return ":".join(signature_parts)
    
    def _count_pattern_duplicates(self, pattern_signature: str) -> int:
        """Count how many times a pattern appears."""
        if not hasattr(self, '_pattern_counts'):
            self._pattern_counts = defaultdict(int)
            
            # Build pattern counts
            for node_id in self.call_graph.nodes():
                node_data = self.call_graph.nodes[node_id]
                if 'pattern' in node_data:
                    self._pattern_counts[node_data['pattern']] += 1
        
        return self._pattern_counts.get(pattern_signature, 0)
    
    def _collect_node_metrics(self, influential_node: InfluentialNode,
                             node_id: str, node: ast.AST) -> None:
        """Collect additional metrics for an influential node."""
        # Direct and indirect calls
        if self.call_graph.has_node(node_id):
            influential_node.direct_calls = self.call_graph.in_degree(node_id)
            
            # Count indirect calls within 3 hops
            indirect = 0
            visited = {node_id}
            current = {node_id}
            
            for _ in range(3):
                next_level = set()
                for n in current:
                    for pred in self.call_graph.predecessors(n):
                        if pred not in visited:
                            visited.add(pred)
                            next_level.add(pred)
                            indirect += 1
                current = next_level
            
            influential_node.indirect_calls = indirect
        
        # Pattern signature and duplicates
        pattern_sig = self._get_pattern_signature(node)
        if pattern_sig:
            influential_node.pattern_signature = pattern_sig
            influential_node.duplicate_count = self._count_pattern_duplicates(pattern_sig)
        
        # Affected files
        if self.call_graph.has_node(node_id):
            for pred in self.call_graph.predecessors(node_id):
                if ':' in pred:
                    file_path = pred.split(':')[0]
                    influential_node.affected_files.add(file_path)
        
        # Cascading risk (if it's an error handler or critical path)
        influential_node.cascading_risk = self._estimate_cascading_risk(node)
    
    def _estimate_cascading_risk(self, node: ast.AST) -> float:
        """Estimate risk of cascading failures."""
        risk = 0.0
        
        # Error handlers have high cascading risk
        if any(isinstance(n, ast.Try) for n in ast.walk(node)):
            risk += 0.3
        
        # Functions that raise exceptions
        if any(isinstance(n, ast.Raise) for n in ast.walk(node)):
            risk += 0.2
        
        # Functions with many conditionals (complex logic)
        if_count = sum(1 for n in ast.walk(node) if isinstance(n, ast.If))
        if if_count > 3:
            risk += 0.2
        
        # Functions that modify global state
        global_count = sum(1 for n in ast.walk(node) 
                          if isinstance(n, ast.Global))
        if global_count > 0:
            risk += 0.3
        
        return min(1.0, risk)


class LayeredIssueDetector:
    """
    Detects issues by slicing codebase into conceptual layers.
    
    Processes layers strategically based on influence, not strict order,
    breaking the "sorting barrier" of dependency-based analysis.
    """
    
    # Layer classification patterns
    LAYER_PATTERNS = {
        'infrastructure': ['util', 'helper', 'config', 'log', 'cache', 'db'],
        'domain_logic': ['model', 'entity', 'domain', 'business', 'core'],
        'application': ['service', 'handler', 'controller', 'manager'],
        'presentation': ['view', 'ui', 'api', 'rest', 'graphql', 'route'],
    }
    
    def __init__(self, influence_detector: Optional[InfluenceDetector] = None):
        """
        Initialize the layered detector.
        
        Args:
            influence_detector: Optional influence detector for enhanced analysis
        """
        self.logger = get_logger("layered_detector")
        self.influence_detector = influence_detector or InfluenceDetector()
        self.layers: Dict[str, LayerInfo] = {}
        self.processed_patterns: Set[str] = set()
        self.cross_layer_graph = nx.DiGraph()
        
    def detect_issues_in_layers(self, codebase: Dict[str, ast.AST]) -> List[Any]:
        """
        Detect issues by processing code in conceptual layers.
        
        Layers are processed strategically based on influence,
        not in strict dependency order.
        
        Args:
            codebase: Dict mapping file paths to AST trees
            
        Returns:
            List of detected issues
        """
        self.logger.info(f"Starting layered analysis of {len(codebase)} files")
        
        # Create layers based on abstraction level
        self.layers = self._create_layers(codebase)
        self.logger.info(f"Created {len(self.layers)} conceptual layers")
        
        # Build cross-layer dependency graph
        self._build_cross_layer_graph()
        
        all_issues = []
        processed_patterns = set()
        
        # Process layers strategically (by influence, not order)
        for layer_name, layer_content in self.iterate_layers_strategically(self.layers):
            self.logger.debug(f"Processing layer: {layer_name}")
            
            # Filter out already-processed patterns
            unique_content = self.filter_processed_patterns(
                layer_content,
                processed_patterns
            )
            
            # Cross-layer analysis
            layer_issues = self.cross_layer_analysis(
                current_layer=unique_content,
                other_layers=self.layers,
                learned_patterns=processed_patterns
            )
            
            all_issues.extend(layer_issues)
            processed_patterns.update(self.extract_patterns(layer_issues))
        
        self.logger.info(f"Detected {len(all_issues)} total issues across layers")
        
        return all_issues
    
    def _create_layers(self, codebase: Dict[str, ast.AST]) -> Dict[str, LayerInfo]:
        """Create conceptual layers from codebase."""
        layers = {
            'infrastructure': LayerInfo(name='infrastructure', nodes=[]),
            'domain_logic': LayerInfo(name='domain_logic', nodes=[]),
            'application': LayerInfo(name='application', nodes=[]),
            'presentation': LayerInfo(name='presentation', nodes=[]),
            'unknown': LayerInfo(name='unknown', nodes=[]),
        }
        
        for file_path, ast_tree in codebase.items():
            # Classify file into layer
            layer_name = self._classify_file_layer(file_path)
            
            # Extract nodes from file
            for node in ast.walk(ast_tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    layers[layer_name].nodes.append((file_path, node))
        
        # Calculate layer metrics
        for layer in layers.values():
            self._calculate_layer_metrics(layer)
        
        return layers
    
    def _classify_file_layer(self, file_path: str) -> str:
        """Classify a file into a conceptual layer."""
        file_lower = file_path.lower()
        
        for layer_name, patterns in self.LAYER_PATTERNS.items():
            for pattern in patterns:
                if pattern in file_lower:
                    return layer_name
        
        # Default classification based on imports or content
        return 'unknown'
    
    def _calculate_layer_metrics(self, layer: LayerInfo) -> None:
        """Calculate metrics for a layer."""
        if not layer.nodes:
            return
        
        # Calculate influence score
        total_influence = 0.0
        for file_path, node in layer.nodes[:10]:  # Sample for efficiency
            node_id = f"{file_path}:{getattr(node, 'name', 'unknown')}"
            influence = self.influence_detector.scout_influence_bellman_ford_style(
                node_id, node, max_steps=2
            )
            total_influence += influence
        
        layer.influence_score = total_influence / min(10, len(layer.nodes))
        
        # Calculate internal cohesion (how connected nodes are within layer)
        internal_edges = 0
        total_possible = len(layer.nodes) * (len(layer.nodes) - 1)
        
        if total_possible > 0:
            # Sample connections for efficiency
            for i, (file1, node1) in enumerate(layer.nodes[:5]):
                for file2, node2 in layer.nodes[i+1:i+6]:
                    if self._nodes_connected(node1, node2):
                        internal_edges += 1
            
            layer.internal_cohesion = internal_edges / min(25, total_possible)
    
    def _nodes_connected(self, node1: ast.AST, node2: ast.AST) -> bool:
        """Check if two nodes are connected (simplified)."""
        # Check if node1 calls node2 or vice versa
        name1 = getattr(node1, 'name', None)
        name2 = getattr(node2, 'name', None)
        
        if name1 and name2:
            # Check for calls in node1 to node2
            for child in ast.walk(node1):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name) and child.func.id == name2:
                        return True
        
        return False
    
    def _build_cross_layer_graph(self) -> None:
        """Build graph of cross-layer dependencies."""
        self.cross_layer_graph.clear()
        
        for layer_name in self.layers:
            self.cross_layer_graph.add_node(layer_name)
        
        # Add edges for cross-layer dependencies
        for layer1_name, layer1 in self.layers.items():
            for layer2_name, layer2 in self.layers.items():
                if layer1_name != layer2_name:
                    # Check for dependencies (simplified)
                    if self._has_dependency(layer1, layer2):
                        self.cross_layer_graph.add_edge(layer1_name, layer2_name)
    
    def _has_dependency(self, layer1: LayerInfo, layer2: LayerInfo) -> bool:
        """Check if layer1 depends on layer2 (simplified)."""
        # Sample check - in real implementation would be more thorough
        layer2_names = {getattr(node, 'name', None) 
                       for _, node in layer2.nodes 
                       if hasattr(node, 'name')}
        
        for _, node in layer1.nodes[:5]:  # Sample
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        if child.func.id in layer2_names:
                            return True
        
        return False
    
    def iterate_layers_strategically(self, layers: Dict[str, LayerInfo]):
        """
        Iterate layers by influence, not dependency order.
        
        This breaks the sorting barrier by processing most influential
        layers first, regardless of their position in the stack.
        """
        # Calculate influence for each layer
        influence_scores = {}
        for layer_name, layer in layers.items():
            influence_scores[layer_name] = self.scout_layer_influence(layer)
        
        # Sort by influence (highest first)
        sorted_layers = sorted(
            layers.items(),
            key=lambda x: influence_scores[x[0]],
            reverse=True
        )
        
        self.logger.debug(f"Layer processing order: {[name for name, _ in sorted_layers]}")
        
        for layer_name, layer in sorted_layers:
            yield layer_name, layer
    
    def scout_layer_influence(self, layer: LayerInfo) -> float:
        """Scout influence of a layer using limited exploration."""
        # Use pre-calculated influence score if available
        if layer.influence_score > 0:
            return layer.influence_score
        
        # Calculate based on layer characteristics
        influence = 0.0
        
        # Size factor
        size_factor = min(len(layer.nodes) / 20.0, 1.0)
        influence += size_factor * 0.3
        
        # Layer type factor (some layers are inherently influential)
        if layer.name == 'infrastructure':
            influence += 0.4  # Infrastructure affects everything
        elif layer.name == 'domain_logic':
            influence += 0.3  # Core business logic is important
        elif layer.name == 'application':
            influence += 0.2
        elif layer.name == 'presentation':
            influence += 0.1
        
        # Cross-layer dependency factor
        if self.cross_layer_graph.has_node(layer.name):
            out_degree = self.cross_layer_graph.out_degree(layer.name)
            influence += min(out_degree / 3.0, 0.3)
        
        return influence
    
    def filter_processed_patterns(self, 
                                 layer_content: LayerInfo,
                                 processed_patterns: Set[str]) -> LayerInfo:
        """Filter out already-processed patterns."""
        filtered_nodes = []
        
        for file_path, node in layer_content.nodes:
            # Generate pattern signature
            pattern_sig = self._generate_pattern_signature(node)
            
            if pattern_sig not in processed_patterns:
                filtered_nodes.append((file_path, node))
        
        # Create new LayerInfo with filtered nodes
        filtered_layer = LayerInfo(
            name=layer_content.name,
            nodes=filtered_nodes,
            influence_score=layer_content.influence_score,
            cross_layer_dependencies=layer_content.cross_layer_dependencies,
            internal_cohesion=layer_content.internal_cohesion
        )
        
        return filtered_layer
    
    def _generate_pattern_signature(self, node: ast.AST) -> str:
        """Generate a signature for pattern detection."""
        parts = []
        
        # Node type
        parts.append(type(node).__name__)
        
        # Basic structure
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(f"params:{len(node.args.args)}")
            parts.append(f"body:{len(node.body)}")
        elif isinstance(node, ast.ClassDef):
            methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
            parts.append(f"methods:{len(methods)}")
        
        return ":".join(parts)
    
    def cross_layer_analysis(self,
                            current_layer: LayerInfo,
                            other_layers: Dict[str, LayerInfo],
                            learned_patterns: Set[str]) -> List[Any]:
        """
        Analyze current layer using knowledge from other layers.
        """
        issues = []
        
        # Check for cross-layer anti-patterns
        for file_path, node in current_layer.nodes:
            # Check if violates layer boundaries
            violation = self._check_layer_violation(
                node, current_layer.name, other_layers
            )
            if violation:
                issues.append(violation)
            
            # Check for duplicates across layers
            duplicate = self._check_cross_layer_duplicate(
                node, current_layer.name, other_layers
            )
            if duplicate:
                issues.append(duplicate)
        
        return issues
    
    def _check_layer_violation(self, node: ast.AST, 
                              current_layer: str,
                              other_layers: Dict[str, LayerInfo]) -> Optional[Dict]:
        """Check if node violates layer boundaries."""
        # Simplified check - presentation shouldn't directly access infrastructure
        if current_layer == 'presentation':
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    if isinstance(child.func, ast.Name):
                        # Check if calling infrastructure directly
                        for _, infra_node in other_layers.get('infrastructure', LayerInfo('', [])).nodes:
                            if hasattr(infra_node, 'name') and child.func.id == infra_node.name:
                                return {
                                    'type': 'layer_violation',
                                    'message': f"Presentation layer directly accessing infrastructure",
                                    'severity': 3
                                }
        return None
    
    def _check_cross_layer_duplicate(self, node: ast.AST,
                                    current_layer: str,
                                    other_layers: Dict[str, LayerInfo]) -> Optional[Dict]:
        """Check for duplicate implementations across layers."""
        node_sig = self._generate_pattern_signature(node)
        
        for layer_name, layer in other_layers.items():
            if layer_name != current_layer:
                for _, other_node in layer.nodes:
                    other_sig = self._generate_pattern_signature(other_node)
                    if node_sig == other_sig:
                        return {
                            'type': 'cross_layer_duplicate',
                            'message': f"Duplicate pattern across {current_layer} and {layer_name}",
                            'severity': 2
                        }
        return None
    
    def extract_patterns(self, issues: List[Any]) -> Set[str]:
        """Extract patterns from detected issues."""
        patterns = set()
        
        for issue in issues:
            if isinstance(issue, dict):
                # Generate pattern from issue type
                pattern = f"{issue.get('type', 'unknown')}:{issue.get('severity', 0)}"
                patterns.add(pattern)
        
        return patterns