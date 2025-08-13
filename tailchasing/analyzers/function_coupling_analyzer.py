"""
FunctionCouplingAnalyzer: Analyzes coupling between functions in code.

This analyzer identifies regions of code that are likely to interact and cause 
tail-chasing patterns through coupling analysis that combines multiple distance 
metrics to detect tightly coupled functions.
"""

import ast
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Set, Any, Union
from pathlib import Path
import networkx as nx
from collections import defaultdict
import logging
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings('ignore', category=RuntimeWarning, module='scipy.spatial.distance')
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
import seaborn as sns

from .base import Analyzer, AnalysisContext
from ..core.issues import Issue


logger = logging.getLogger(__name__)


@dataclass
class CodeElement:
    """Represents a code element (function, class, etc.) for chromatin analysis."""
    file_path: str
    name: str
    node_type: str  # 'function', 'class', 'method'
    line_start: int
    line_end: int
    ast_node: ast.AST
    module_path: str
    class_name: Optional[str] = None
    
    def __hash__(self) -> int:
        return hash((self.file_path, self.name, self.line_start, self.class_name))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, CodeElement):
            return False
        return (
            self.file_path == other.file_path and
            self.name == other.name and
            self.line_start == other.line_start and
            self.class_name == other.class_name
        )


@dataclass
class DistanceWeights:
    """Weights for combining different distance metrics."""
    token: float = 0.3
    ast: float = 0.25
    module: float = 0.25
    git: float = 0.2


@dataclass
class ContactParameters:
    """Parameters for the contact probability function."""
    kappa: float = 1.0          # Contact strength constant
    alpha: float = 1.5          # Power law exponent
    epsilon: float = 0.1        # Regularization parameter
    tad_penalty: float = 0.7    # Penalty for cross-compartment contacts


@dataclass
class TAD:
    """Represents a Topologically Associating Domain (code compartment)."""
    tad_id: str
    name: str
    modules: Set[str]
    elements: List[CodeElement]
    boundary_strength: float
    import_density: float
    cohesion_score: float
    parent_tad: Optional['TAD'] = None
    child_tads: List['TAD'] = None
    
    def __post_init__(self):
        if self.child_tads is None:
            self.child_tads = []


@dataclass 
class LoopAnchor:
    """Represents a loop anchor point (like CTCF/cohesin binding sites)."""
    anchor_id: str
    element1: CodeElement
    element2: CodeElement
    contact_strength: float
    anchor_type: str  # 'import', 'inheritance', 'call'
    reciprocal: bool
    distance: float


@dataclass
class ContactMatrix:
    """Hi-C style contact matrix representation."""
    matrix: np.ndarray
    elements: List[CodeElement]
    tad_boundaries: List[int]
    loop_anchors: List[LoopAnchor]
    element_to_index: Dict[CodeElement, int]
    tad_map: Dict[str, TAD]


class PolymerDistanceCalculator:
    """Calculates various distance metrics between code elements using polymer physics concepts."""
    
    def __init__(self, weights: Optional[DistanceWeights] = None):
        """
        Initialize the distance calculator.
        
        Args:
            weights: Distance metric weights. Uses default if None.
        """
        self.weights = weights or DistanceWeights()
        self._ast_distances: Dict[Tuple[int, int], float] = {}
        self._module_distances: Dict[Tuple[str, str], float] = {}
        self._git_distances: Dict[Tuple[CodeElement, CodeElement], float] = {}
    
    def tok_dist(self, elem1: CodeElement, elem2: CodeElement) -> float:
        """
        Calculate token/line distance between two code elements.
        
        For elements in the same file, returns the line difference.
        For elements in different files, returns a large constant.
        
        Args:
            elem1: First code element
            elem2: Second code element
            
        Returns:
            Token distance as a float
        """
        if elem1.file_path == elem2.file_path:
            # Same file: use line distance
            return abs(elem1.line_start - elem2.line_start)
        else:
            # Different files: large constant distance
            return 10000.0
    
    def ast_dist(self, elem1: CodeElement, elem2: CodeElement) -> float:
        """
        Calculate AST tree distance via Lowest Common Ancestor (LCA).
        
        Args:
            elem1: First code element
            elem2: Second code element
            
        Returns:
            AST distance as a float
        """
        # Cache key for memoization
        cache_key = (id(elem1.ast_node), id(elem2.ast_node))
        if cache_key in self._ast_distances:
            return self._ast_distances[cache_key]
        
        if elem1.file_path != elem2.file_path:
            # Different files: maximum distance
            distance = 1000.0
        else:
            # Same file: calculate LCA distance
            distance = self._calculate_lca_distance(elem1.ast_node, elem2.ast_node)
        
        self._ast_distances[cache_key] = distance
        return distance
    
    def mod_dist(self, elem1: CodeElement, elem2: CodeElement, 
                 import_graph: nx.DiGraph) -> float:
        """
        Calculate module distance via shortest path in import graph.
        
        Args:
            elem1: First code element
            elem2: Second code element
            import_graph: Import dependency graph
            
        Returns:
            Module distance as a float
        """
        cache_key = (elem1.module_path, elem2.module_path)
        if cache_key in self._module_distances:
            return self._module_distances[cache_key]
        
        if elem1.module_path == elem2.module_path:
            # Same module
            distance = 0.0
        else:
            try:
                # Find shortest path in import graph
                path_length = nx.shortest_path_length(
                    import_graph, elem1.module_path, elem2.module_path
                )
                distance = float(path_length)
            except (nx.NetworkXNoPath, nx.NodeNotFound):
                # No path found: use package hierarchy distance
                distance = self._package_hierarchy_distance(
                    elem1.module_path, elem2.module_path
                )
        
        self._module_distances[cache_key] = distance
        return distance
    
    def git_dist(self, elem1: CodeElement, elem2: CodeElement,
                 coedit_matrix: np.ndarray, element_to_index: Dict[CodeElement, int]) -> float:
        """
        Calculate git distance based on co-edit correlation.
        
        Args:
            elem1: First code element
            elem2: Second code element
            coedit_matrix: Co-edit correlation matrix
            element_to_index: Mapping from elements to matrix indices
            
        Returns:
            Git distance as 1 - correlation
        """
        cache_key = (elem1, elem2)
        if cache_key in self._git_distances:
            return self._git_distances[cache_key]
        
        if elem1 not in element_to_index or elem2 not in element_to_index:
            # Elements not in co-edit matrix: assume no correlation
            distance = 1.0
        else:
            idx1 = element_to_index[elem1]
            idx2 = element_to_index[elem2]
            
            # Git distance = 1 - correlation
            correlation = coedit_matrix[idx1, idx2]
            distance = 1.0 - max(0.0, correlation)  # Ensure non-negative
        
        self._git_distances[cache_key] = distance
        return distance
    
    def _calculate_lca_distance(self, node1: ast.AST, node2: ast.AST) -> float:
        """
        Calculate distance between two AST nodes via their Lowest Common Ancestor.
        
        Args:
            node1: First AST node
            node2: Second AST node
            
        Returns:
            LCA distance as a float
        """
        if node1 is node2:
            return 0.0
        
        # For simplicity, use a heuristic based on node types and positions
        # In a full implementation, this would traverse the AST to find the LCA
        
        # If nodes have line numbers, use that as a proxy
        line1 = getattr(node1, 'lineno', 0)
        line2 = getattr(node2, 'lineno', 0)
        
        if line1 and line2:
            # Heuristic: closer lines are more likely to share ancestors
            line_distance = abs(line1 - line2)
            return min(100.0, line_distance * 0.5)  # Cap at 100
        
        # Fallback: moderate distance
        return 50.0
    
    def _package_hierarchy_distance(self, module1: str, module2: str) -> float:
        """
        Calculate distance based on package hierarchy (dot-separated paths).
        
        Args:
            module1: First module path
            module2: Second module path
            
        Returns:
            Package hierarchy distance
        """
        parts1 = module1.split('.')
        parts2 = module2.split('.')
        
        # Find common prefix
        common_len = 0
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common_len += 1
            else:
                break
        
        # Distance = sum of unique parts
        unique_parts = len(parts1) + len(parts2) - 2 * common_len
        return float(unique_parts)


class FunctionCouplingAnalyzer(Analyzer):
    """
    Analyzer that detects function coupling in code architecture.
    
    This analyzer identifies regions of code that are tightly coupled and 
    potentially cause tail-chasing patterns through excessive interdependencies.
    """
    
    name = "function_coupling"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the function coupling analyzer.
        
        Args:
            config: Configuration dictionary
        """
        super().__init__(config)
        
        # Load configuration
        coupling_config = (config or {}).get('function_coupling', {})
        
        # Distance weights
        weight_config = coupling_config.get('weights', {})
        self.weights = DistanceWeights(
            token=weight_config.get('token', 0.3),
            ast=weight_config.get('ast', 0.25),
            module=weight_config.get('module', 0.25),
            git=weight_config.get('git', 0.2)
        )
        
        # Contact parameters
        contact_config = coupling_config.get('contact_params', {})
        self.contact_params = ContactParameters(
            kappa=contact_config.get('kappa', 1.0),
            alpha=contact_config.get('alpha', 1.5),
            epsilon=contact_config.get('epsilon', 0.1),
            tad_penalty=contact_config.get('tad_penalty', 0.7)
        )
        
        # Thresholds
        self.contact_threshold = coupling_config.get('contact_threshold', 0.3)
        self.thrash_threshold = coupling_config.get('thrash_threshold', 0.2)
        
        # Initialize calculator
        self.distance_calculator = PolymerDistanceCalculator(self.weights)
        
        # Cache for expensive operations
        self._elements: List[CodeElement] = []
        self._import_graph: Optional[nx.DiGraph] = None
        self._coedit_matrix: Optional[np.ndarray] = None
        self._element_to_index: Dict[CodeElement, int] = {}
        
        # TAD-specific caches
        self._tads: Dict[str, TAD] = {}
        self._contact_matrix: Optional[ContactMatrix] = None
        self._loop_anchors: List[LoopAnchor] = []
    
    def _safe_node_span(self, node: ast.AST) -> tuple[int, int]:
        """Return (line_start, line_end) for a node, tolerating nodes missing lineno.
        Falls back to children's lines or 1 if unavailable."""
        start = getattr(node, 'lineno', None)
        end = getattr(node, 'end_lineno', None)
        try:
            body = getattr(node, 'body', []) or []
        except Exception:
            body = []
        if start is None:
            if body:
                start = getattr(body[0], 'lineno', None)
        if end is None:
            if body:
                last = body[-1]
                end = getattr(last, 'end_lineno', getattr(last, 'lineno', None))
        if start is None:
            start = 1
        if end is None:
            end = start
        return int(start), int(end)
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run chromatin contact analysis.
        
        Args:
            ctx: Analysis context
            
        Returns:
            List of detected issues
        """
        logger.info("Starting function coupling analysis")
        
        # Extract code elements
        self._elements = self._extract_code_elements(ctx)
        if len(self._elements) < 2:
            logger.info("Not enough code elements for coupling analysis")
            return []
        
        logger.info(f"Analyzing {len(self._elements)} code elements")
        
        # Build import graph
        self._import_graph = self._build_import_graph(ctx)
        
        # Build co-edit matrix
        self._coedit_matrix, self._element_to_index = self._build_coedit_matrix(ctx)
        
        # Detect TADs and build full contact matrix
        self._tads = self.detect_tads(ctx)
        self._contact_matrix = self.build_contact_matrix()
        self._loop_anchors = self.identify_loop_anchors()
        
        # Calculate contact probabilities and thrash risks
        issues = []
        contact_matrix = self._contact_matrix.matrix
        
        # Find high-risk contacts
        for i in range(len(self._elements)):
            for j in range(i + 1, len(self._elements)):
                elem1, elem2 = self._elements[i], self._elements[j]
                
                contact_prob = contact_matrix[i, j]
                if contact_prob > self.contact_threshold:
                    # Calculate thrash risk
                    similarity = self._calculate_similarity(elem1, elem2)
                    thrash_risk = contact_prob * similarity
                    
                    if thrash_risk > self.thrash_threshold:
                        issue = self._create_chromatin_issue(
                            elem1, elem2, contact_prob, similarity, thrash_risk
                        )
                        issues.append(issue)
        
        logger.info(f"Found {len(issues)} function coupling issues")
        return issues
    
    def polymer_distance(self, elem1: CodeElement, elem2: CodeElement) -> float:
        """
        Calculate combined polymer distance between two code elements.
        
        D(i,j) = w_t * d_tokens + w_a * d_AST + w_m * d_module + w_g * d_git
        
        Args:
            elem1: First code element
            elem2: Second code element
            
        Returns:
            Combined polymer distance
        """
        # Calculate individual distances
        d_tokens = self.distance_calculator.tok_dist(elem1, elem2)
        d_ast = self.distance_calculator.ast_dist(elem1, elem2)
        d_module = self.distance_calculator.mod_dist(elem1, elem2, self._import_graph)
        d_git = self.distance_calculator.git_dist(
            elem1, elem2, self._coedit_matrix, self._element_to_index
        )
        
        # Normalize distances to [0, 1] range
        d_tokens_norm = min(1.0, d_tokens / 1000.0)  # Normalize by max expected
        d_ast_norm = min(1.0, d_ast / 100.0)
        d_module_norm = min(1.0, d_module / 10.0)
        d_git_norm = d_git  # Already in [0, 1]
        
        # Weighted combination
        total_distance = (
            self.weights.token * d_tokens_norm +
            self.weights.ast * d_ast_norm +
            self.weights.module * d_module_norm +
            self.weights.git * d_git_norm
        )
        
        return total_distance
    
    def contact_probability(self, elem1: CodeElement, elem2: CodeElement) -> float:
        """
        Calculate contact probability using polymer physics model.
        
        P(i,j) = κ * (D(i,j) + ε)^(-α) * γ_TAD(i,j)
        
        Args:
            elem1: First code element
            elem2: Second code element
            
        Returns:
            Contact probability
        """
        # Calculate polymer distance
        distance = self.polymer_distance(elem1, elem2)
        
        # Power law contact probability
        base_prob = self.contact_params.kappa * np.power(
            distance + self.contact_params.epsilon,
            -self.contact_params.alpha
        )
        
        # Normalize to reasonable range [0, 1]
        base_prob = min(1.0, base_prob)
        
        # Apply TAD penalty for cross-compartment contacts
        gamma_tad = self._tad_factor(elem1, elem2)
        
        return base_prob * gamma_tad
    
    def thrash_risk(self, elem1: CodeElement, elem2: CodeElement) -> float:
        """
        Calculate thrash risk score.
        
        R(i,j) = P(i,j) * similarity(i,j)
        
        Args:
            elem1: First code element
            elem2: Second code element
            
        Returns:
            Thrash risk score
        """
        contact_prob = self.contact_probability(elem1, elem2)
        similarity = self._calculate_similarity(elem1, elem2)
        return contact_prob * similarity
    
    def _extract_code_elements(self, ctx: AnalysisContext) -> List[CodeElement]:
        """Extract code elements from the AST index."""
        elements = []
        
        for file_path, tree in ctx.ast_index.items():
            module_path = self._file_to_module_path(file_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    # Check if it's a method (inside a class)
                    class_name = None
                    for parent in ast.walk(tree):
                        if isinstance(parent, ast.ClassDef):
                            for child in ast.walk(parent):
                                if child is node:
                                    class_name = parent.name
                                    break
                            if class_name:
                                break
                    
                    line_start, line_end = self._safe_node_span(node)
                    element = CodeElement(
                        file_path=file_path,
                        name=node.name,
                        node_type='method' if class_name else 'function',
                        line_start=line_start,
                        line_end=line_end,
                        ast_node=node,
                        module_path=module_path,
                        class_name=class_name
                    )
                    elements.append(element)
                
                elif isinstance(node, ast.ClassDef):
                    line_start, line_end = self._safe_node_span(node)
                    element = CodeElement(
                        file_path=file_path,
                        name=node.name,
                        node_type='class',
                        line_start=line_start,
                        line_end=line_end,
                        ast_node=node,
                        module_path=module_path
                    )
                    elements.append(element)
        
        return elements
    
    def _build_import_graph(self, ctx: AnalysisContext) -> nx.DiGraph:
        """Build import dependency graph."""
        graph = nx.DiGraph()
        
        for file_path, tree in ctx.ast_index.items():
            module_path = self._file_to_module_path(file_path)
            graph.add_node(module_path)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        graph.add_edge(module_path, alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        graph.add_edge(module_path, node.module)
        
        return graph
    
    def _build_coedit_matrix(self, ctx: AnalysisContext) -> Tuple[np.ndarray, Dict[CodeElement, int]]:
        """Build co-edit correlation matrix (simplified version)."""
        n_elements = len(self._elements)
        coedit_matrix = np.eye(n_elements)  # Identity matrix as placeholder
        
        element_to_index = {elem: i for i, elem in enumerate(self._elements)}
        
        # In a real implementation, this would analyze git history
        # For now, we use a heuristic based on file proximity
        for i, elem1 in enumerate(self._elements):
            for j, elem2 in enumerate(self._elements):
                if i != j:
                    if elem1.file_path == elem2.file_path:
                        # Same file: high correlation
                        coedit_matrix[i, j] = 0.8
                    elif elem1.module_path == elem2.module_path:
                        # Same module: moderate correlation
                        coedit_matrix[i, j] = 0.4
                    else:
                        # Different modules: low correlation
                        coedit_matrix[i, j] = 0.1
        
        return coedit_matrix, element_to_index
    
    def _calculate_contact_matrix(self) -> np.ndarray:
        """Calculate contact probability matrix for all element pairs."""
        n_elements = len(self._elements)
        contact_matrix = np.zeros((n_elements, n_elements))
        
        for i in range(n_elements):
            for j in range(i, n_elements):
                if i == j:
                    contact_matrix[i, j] = 1.0  # Self-contact
                else:
                    prob = self.contact_probability(self._elements[i], self._elements[j])
                    contact_matrix[i, j] = prob
                    contact_matrix[j, i] = prob  # Symmetric matrix
        
        return contact_matrix
    
    def _tad_factor(self, elem1: CodeElement, elem2: CodeElement) -> float:
        """
        Calculate TAD (Topologically Associating Domain) factor.
        
        Returns penalty for cross-compartment contacts.
        """
        if self._tads:
            # Use detected TADs for more accurate penalty calculation
            return self.tad_penalty(elem1.module_path, elem2.module_path, self._tads)
        else:
            # Fallback: elements in different packages are in different TADs
            if self._get_package(elem1.module_path) != self._get_package(elem2.module_path):
                return self.contact_params.tad_penalty
            return 1.0
    
    def _get_package(self, module_path: str) -> str:
        """Get the top-level package from a module path."""
        return module_path.split('.')[0]
    
    def _calculate_similarity(self, elem1: CodeElement, elem2: CodeElement) -> float:
        """Calculate similarity between two code elements."""
        # Name similarity
        name_sim = 1.0 if elem1.name == elem2.name else 0.0
        
        # Type similarity
        type_sim = 1.0 if elem1.node_type == elem2.node_type else 0.5
        
        # Size similarity (based on line count)
        size1 = elem1.line_end - elem1.line_start + 1
        size2 = elem2.line_end - elem2.line_start + 1
        size_sim = 1.0 - abs(size1 - size2) / max(size1, size2, 1)
        
        # Weighted combination
        return 0.5 * name_sim + 0.3 * type_sim + 0.2 * size_sim
    
    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to module path."""
        path = Path(file_path)
        if path.suffix == '.py':
            # Remove .py extension and convert path separators to dots
            module_parts = path.with_suffix('').parts
            # Remove common prefixes like 'src', 'lib', etc.
            if module_parts and module_parts[0] in ['src', 'lib', 'tailchasing']:
                module_parts = module_parts[1:]
            return '.'.join(module_parts)
        return str(path)
    
    def _create_chromatin_issue(self, elem1: CodeElement, elem2: CodeElement,
                               contact_prob: float, similarity: float,
                               thrash_risk: float) -> Issue:
        """Create an issue for high chromatin contact risk."""
        # Determine severity based on thrash risk
        if thrash_risk > 0.3:
            severity = 4
        elif thrash_risk > 0.15:
            severity = 3
        else:
            severity = 2
        
        # Create descriptive message
        elem1_desc = f"{elem1.class_name}.{elem1.name}" if elem1.class_name else elem1.name
        elem2_desc = f"{elem2.class_name}.{elem2.name}" if elem2.class_name else elem2.name
        
        message = f"High coupling risk between {elem1_desc} and {elem2_desc}"
        
        # Evidence for debugging and analysis
        evidence = {
            'element1': {
                'name': elem1_desc,
                'file': elem1.file_path,
                'line': elem1.line_start,
                'type': elem1.node_type
            },
            'element2': {
                'name': elem2_desc,
                'file': elem2.file_path,
                'line': elem2.line_start,
                'type': elem2.node_type
            },
            'contact_probability': float(contact_prob),
            'similarity': float(similarity),
            'thrash_risk': float(thrash_risk),
            'polymer_distance': float(self.polymer_distance(elem1, elem2))
        }
        
        return Issue(
            kind="function_coupling_risk",
            message=message,
            severity=severity,
            file=elem1.file_path,
            line=elem1.line_start,
            evidence=evidence
        )
    
    # ==================== TAD Detection Methods ====================
    
    def detect_tads(self, ctx: AnalysisContext) -> Dict[str, TAD]:
        """
        Detect Topologically Associating Domains (code compartments).
        
        Uses package structure, import density, and co-edit clustering
        to identify natural code compartments.
        
        Args:
            ctx: Analysis context
            
        Returns:
            Dictionary mapping TAD IDs to TAD objects
        """
        logger.info("Detecting TADs (code compartments)")
        
        # Step 1: Identify package-based TADs
        package_tads = self._detect_package_tads()
        
        # Step 2: Refine with import density analysis
        import_refined_tads = self._refine_tads_by_import_density(package_tads)
        
        # Step 3: Further refine with co-edit clustering
        final_tads = self._refine_tads_by_coedit_clustering(import_refined_tads, ctx)
        
        # Step 4: Calculate TAD properties
        for tad in final_tads.values():
            tad.boundary_strength = self._calculate_boundary_strength(tad)
            tad.import_density = self._calculate_import_density(tad)
            tad.cohesion_score = self._calculate_cohesion_score(tad)
        
        logger.info(f"Detected {len(final_tads)} TADs")
        return final_tads
    
    def tad_penalty(self, module_i: str, module_j: str, tad_map: Dict[str, TAD]) -> float:
        """
        Calculate TAD penalty for cross-TAD interactions.
        
        Args:
            module_i: First module path
            module_j: Second module path  
            tad_map: Mapping of TAD IDs to TAD objects
            
        Returns:
            Penalty factor (γ < 1 for cross-TAD interactions)
        """
        # Find TADs containing these modules
        tad_i = self._find_module_tad(module_i, tad_map)
        tad_j = self._find_module_tad(module_j, tad_map)
        
        if tad_i is None or tad_j is None:
            return self.contact_params.tad_penalty  # Default penalty
        
        if tad_i.tad_id == tad_j.tad_id:
            return 1.0  # Same TAD, no penalty
        
        # Cross-TAD penalty based on boundary strength
        boundary_strength = max(tad_i.boundary_strength, tad_j.boundary_strength)
        penalty = self.contact_params.tad_penalty * (1.0 - boundary_strength * 0.3)
        
        return max(0.1, penalty)  # Minimum penalty
    
    def build_contact_matrix(self) -> ContactMatrix:
        """
        Build Hi-C style contact matrix showing TAD structure.
        
        Creates a contact matrix with:
        - High contact within TADs (darker regions)
        - Lower contact across TAD boundaries (lighter regions)  
        - Import-driven loop anchors (high contact points)
        
        Returns:
            ContactMatrix object with visualization data
        """
        logger.info("Building Hi-C style contact matrix")
        
        n_elements = len(self._elements)
        contact_matrix = np.zeros((n_elements, n_elements))
        
        # Calculate contact probabilities for all pairs
        for i in range(n_elements):
            for j in range(i, n_elements):
                if i == j:
                    contact_matrix[i, j] = 1.0  # Self-contact
                else:
                    prob = self.contact_probability(self._elements[i], self._elements[j])
                    contact_matrix[i, j] = prob
                    contact_matrix[j, i] = prob
        
        # Identify TAD boundaries
        tad_boundaries = self._identify_tad_boundaries()
        
        # Create ContactMatrix object
        matrix_obj = ContactMatrix(
            matrix=contact_matrix,
            elements=self._elements.copy(),
            tad_boundaries=tad_boundaries,
            loop_anchors=self._loop_anchors.copy(),
            element_to_index=self._element_to_index.copy(),
            tad_map=self._tads.copy()
        )
        
        logger.info(f"Built contact matrix ({n_elements}x{n_elements}) with {len(tad_boundaries)} TAD boundaries")
        return matrix_obj
    
    def identify_loop_anchors(self) -> List[LoopAnchor]:
        """
        Identify loop anchor points (like CTCF/cohesin binding sites).
        
        Finds reciprocally imported symbols that create long-range contacts
        across TAD boundaries, similar to chromatin loop formation.
        
        Returns:
            List of LoopAnchor objects
        """
        logger.info("Identifying loop anchors")
        
        anchors = []
        
        # Find import-based anchors
        import_anchors = self._find_import_anchors()
        anchors.extend(import_anchors)
        
        # Find inheritance-based anchors
        inheritance_anchors = self._find_inheritance_anchors()
        anchors.extend(inheritance_anchors)
        
        # Find call-based anchors (function calls across TADs)
        call_anchors = self._find_call_anchors()
        anchors.extend(call_anchors)
        
        # Sort by contact strength
        anchors.sort(key=lambda a: a.contact_strength, reverse=True)
        
        logger.info(f"Identified {len(anchors)} loop anchors")
        return anchors
    
    # ==================== Helper Methods for TAD Detection ====================
    
    def _detect_package_tads(self) -> Dict[str, TAD]:
        """Detect initial TADs based on package structure."""
        package_groups = defaultdict(list)
        
        # Group elements by package
        for element in self._elements:
            package = self._get_package(element.module_path)
            package_groups[package].append(element)
        
        tads = {}
        for package, elements in package_groups.items():
            if len(elements) >= 2:  # Minimum size for a TAD
                modules = {elem.module_path for elem in elements}
                tad = TAD(
                    tad_id=f"TAD_{package}",
                    name=package,
                    modules=modules,
                    elements=elements,
                    boundary_strength=0.5,  # Will be refined later
                    import_density=0.0,     # Will be calculated later
                    cohesion_score=0.0      # Will be calculated later
                )
                tads[tad.tad_id] = tad
        
        return tads
    
    def _refine_tads_by_import_density(self, tads: Dict[str, TAD]) -> Dict[str, TAD]:
        """Refine TADs based on import density patterns."""
        refined_tads = {}
        
        for tad_id, tad in tads.items():
            # Calculate internal vs external import ratios
            internal_imports = self._count_internal_imports(tad)
            external_imports = self._count_external_imports(tad)
            
            density_ratio = internal_imports / max(1, internal_imports + external_imports)
            
            if density_ratio > 0.6:  # High internal cohesion
                refined_tads[tad_id] = tad
            else:
                # Split into sub-TADs or merge with neighbors
                sub_tads = self._split_tad_by_modules(tad)
                for sub_tad in sub_tads:
                    refined_tads[sub_tad.tad_id] = sub_tad
        
        return refined_tads
    
    def _refine_tads_by_coedit_clustering(self, tads: Dict[str, TAD], ctx: AnalysisContext) -> Dict[str, TAD]:
        """Refine TADs using co-edit clustering from git history."""
        if self._coedit_matrix is None:
            return tads
        
        refined_tads = {}
        
        for tad_id, tad in tads.items():
            # Get co-edit patterns for elements in this TAD
            tad_elements = [elem for elem in tad.elements if elem in self._element_to_index]
            
            if len(tad_elements) < 3:
                refined_tads[tad_id] = tad
                continue
            
            # Extract submatrix for this TAD
            indices = [self._element_to_index[elem] for elem in tad_elements]
            submatrix = self._coedit_matrix[np.ix_(indices, indices)]
            
            # Perform hierarchical clustering
            try:
                # Convert similarity to distance
                distance_matrix = 1.0 - submatrix
                
                # Check if matrix has sufficient variance for clustering
                # If all values are too similar, skip clustering
                matrix_std = np.std(distance_matrix)
                if matrix_std < 1e-10:
                    # Matrix is essentially uniform, no meaningful clusters
                    refined_tads[tad_id] = tad
                    continue
                
                # Ensure distance matrix is valid (no negative values, diagonal is 0)
                np.fill_diagonal(distance_matrix, 0)
                distance_matrix = np.maximum(distance_matrix, 0)
                
                # Add small noise to prevent singular matrices that cause warnings
                # This is a common technique to avoid numerical issues in clustering
                distance_matrix += np.random.RandomState(42).rand(*distance_matrix.shape) * 1e-10
                
                # Use scipy's squareform to convert to condensed form
                # This handles edge cases better than using pdist directly
                condensed_distances = squareform(distance_matrix, checks=False)
                
                if len(condensed_distances) > 0 and not np.isnan(condensed_distances).any():
                    # Suppress warnings for the linkage calculation
                    import warnings
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', category=RuntimeWarning)
                        linkage_matrix = linkage(condensed_distances, method='average')  # Use average instead of ward
                    clusters = fcluster(linkage_matrix, t=0.7, criterion='distance')
                    
                    # Split TAD based on clusters
                    cluster_groups = defaultdict(list)
                    for idx, cluster_id in enumerate(clusters):
                        cluster_groups[cluster_id].append(tad_elements[idx])
                    
                    for cluster_id, cluster_elements in cluster_groups.items():
                        if len(cluster_elements) >= 2:
                            sub_tad = TAD(
                                tad_id=f"{tad_id}_C{cluster_id}",
                                name=f"{tad.name}_cluster_{cluster_id}",
                                modules={elem.module_path for elem in cluster_elements},
                                elements=cluster_elements,
                                boundary_strength=0.5,
                                import_density=0.0,
                                cohesion_score=0.0,
                                parent_tad=tad
                            )
                            refined_tads[sub_tad.tad_id] = sub_tad
                else:
                    refined_tads[tad_id] = tad
            except Exception as e:
                # If clustering fails, keep original TAD
                logger.debug(f"Clustering failed for TAD {tad_id}: {e}")
                refined_tads[tad_id] = tad
        
        return refined_tads
    
    def _calculate_boundary_strength(self, tad: TAD) -> float:
        """Calculate the strength of TAD boundaries."""
        if not tad.elements or len(tad.elements) < 2:
            return 0.0
        
        # Boundary strength based on import insulation
        internal_contacts = 0
        external_contacts = 0
        
        for elem1 in tad.elements:
            for elem2 in self._elements:
                if elem2 in tad.elements:
                    internal_contacts += 1
                else:
                    # Check if there's an import relationship
                    if self._has_import_relationship(elem1, elem2):
                        external_contacts += 1
        
        total_contacts = internal_contacts + external_contacts
        if total_contacts == 0:
            return 0.5
        
        # Higher ratio of internal to external contacts = stronger boundary
        return internal_contacts / total_contacts
    
    def _calculate_import_density(self, tad: TAD) -> float:
        """Calculate import density within the TAD."""
        if not tad.modules:
            return 0.0
        
        total_imports = 0
        internal_imports = 0
        
        for module in tad.modules:
            if module in self._import_graph:
                neighbors = list(self._import_graph.successors(module))
                total_imports += len(neighbors)
                
                for neighbor in neighbors:
                    if neighbor in tad.modules:
                        internal_imports += 1
        
        return internal_imports / max(1, total_imports)
    
    def _calculate_cohesion_score(self, tad: TAD) -> float:
        """Calculate cohesion score for the TAD."""
        if len(tad.elements) < 2:
            return 0.0
        
        # Average pairwise contact probability within TAD
        total_prob = 0.0
        pairs = 0
        
        for i, elem1 in enumerate(tad.elements):
            for elem2 in tad.elements[i+1:]:
                prob = self.contact_probability(elem1, elem2)
                total_prob += prob
                pairs += 1
        
        return total_prob / max(1, pairs)
    
    def _find_module_tad(self, module_path: str, tad_map: Dict[str, TAD]) -> Optional[TAD]:
        """Find which TAD contains a given module."""
        for tad in tad_map.values():
            if module_path in tad.modules:
                return tad
        return None
    
    def _identify_tad_boundaries(self) -> List[int]:
        """Identify positions of TAD boundaries in the element list."""
        if not self._tads:
            return []
        
        boundaries = []
        current_tad = None
        
        for i, element in enumerate(self._elements):
            element_tad = self._find_module_tad(element.module_path, self._tads)
            
            if element_tad != current_tad and current_tad is not None:
                boundaries.append(i)
            current_tad = element_tad
        
        return boundaries
    
    def _find_import_anchors(self) -> List[LoopAnchor]:
        """Find import-based loop anchors."""
        anchors = []
        
        if not self._import_graph:
            return anchors
        
        # Find reciprocal imports
        for elem1 in self._elements:
            for elem2 in self._elements:
                if elem1 == elem2:
                    continue
                
                # Check for reciprocal import relationship
                has_forward = self._import_graph.has_edge(elem1.module_path, elem2.module_path)
                has_reverse = self._import_graph.has_edge(elem2.module_path, elem1.module_path)
                
                if has_forward and has_reverse:
                    # Calculate contact strength
                    contact_strength = self.contact_probability(elem1, elem2)
                    distance = self.polymer_distance(elem1, elem2)
                    
                    anchor = LoopAnchor(
                        anchor_id=f"import_{elem1.name}_{elem2.name}",
                        element1=elem1,
                        element2=elem2,
                        contact_strength=contact_strength,
                        anchor_type='import',
                        reciprocal=True,
                        distance=distance
                    )
                    anchors.append(anchor)
        
        return anchors
    
    def _find_inheritance_anchors(self) -> List[LoopAnchor]:
        """Find inheritance-based loop anchors."""
        anchors = []
        
        # Group elements by type
        classes = [elem for elem in self._elements if elem.node_type == 'class']
        methods = [elem for elem in self._elements if elem.node_type == 'method']
        
        # Find class-method relationships across TADs
        for class_elem in classes:
            for method_elem in methods:
                if method_elem.class_name == class_elem.name:
                    # Check if they're in different TADs
                    class_tad = self._find_module_tad(class_elem.module_path, self._tads)
                    method_tad = self._find_module_tad(method_elem.module_path, self._tads)
                    
                    if class_tad and method_tad and class_tad.tad_id != method_tad.tad_id:
                        contact_strength = self.contact_probability(class_elem, method_elem)
                        distance = self.polymer_distance(class_elem, method_elem)
                        
                        anchor = LoopAnchor(
                            anchor_id=f"inheritance_{class_elem.name}_{method_elem.name}",
                            element1=class_elem,
                            element2=method_elem,
                            contact_strength=contact_strength,
                            anchor_type='inheritance',
                            reciprocal=False,
                            distance=distance
                        )
                        anchors.append(anchor)
        
        return anchors
    
    def get_source_for(self, module_path: str) -> str:
        """Best-effort retrieval of module source for get_source_segment fallback.
        
        Args:
            module_path: Path to the module file
            
        Returns:
            Module source text or empty string on error
        """
        try:
            with open(module_path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except Exception:
            return ""
    
    def safe_unparse(self, node: ast.AST, *, source_text: str = "", max_len: int = 20000) -> str:
        """Safely convert an AST node to a string.
        
        Order: unparse -> get_source_segment -> dump -> "".
        Also avoids CPython unparser edge-cases for f-strings (JoinedStr).
        
        Args:
            node: AST node to unparse
            source_text: Optional source text for fallback extraction
            max_len: Maximum length of returned string
            
        Returns:
            String representation of the node (truncated if too long)
        """
        if node is None:
            return ""
        
        # Avoid the JoinedStr edge-case that can raise ValueError in CPython's unparser
        if isinstance(node, ast.JoinedStr):
            return ""
        
        try:
            s = ast.unparse(ast.fix_missing_locations(node))
        except (ValueError, TypeError, AttributeError):
            if source_text:
                try:
                    seg = ast.get_source_segment(source_text, node)
                    if seg:
                        s = seg
                    else:
                        s = ast.dump(node, include_attributes=False)
                except Exception:
                    s = ast.dump(node, include_attributes=False)
            else:
                try:
                    s = ast.dump(node, include_attributes=False)
                except Exception:
                    s = ""
        
        # Keep memory/logs sane
        if isinstance(s, str) and len(s) > max_len:
            return s[:max_len] + " …<truncated>…"
        return s
    
    def _find_call_anchors(self) -> List[LoopAnchor]:
        """Find function call-based loop anchors."""
        anchors = []
        
        # This is a simplified version - in practice would analyze AST for function calls
        functions = [elem for elem in self._elements if elem.node_type in ['function', 'method']]
        
        for elem1 in functions:
            for elem2 in functions:
                if elem1 == elem2:
                    continue
                
                # Heuristic: functions with similar names might call each other
                # Get function bodies as strings (safely)
                src1 = self.get_source_for(elem1.module_path) if getattr(elem1, "module_path", None) else ""
                src2 = self.get_source_for(elem2.module_path) if getattr(elem2, "module_path", None) else ""
                elem1_body = self.safe_unparse(elem1.ast_node, source_text=src1) if getattr(elem1, "ast_node", None) else ""
                elem2_body = self.safe_unparse(elem2.ast_node, source_text=src2) if getattr(elem2, "ast_node", None) else ""
                
                if (elem1.name in elem2_body or elem2.name in elem1_body):
                    elem1_tad = self._find_module_tad(elem1.module_path, self._tads)
                    elem2_tad = self._find_module_tad(elem2.module_path, self._tads)
                    
                    if elem1_tad and elem2_tad and elem1_tad.tad_id != elem2_tad.tad_id:
                        contact_strength = self.contact_probability(elem1, elem2)
                        distance = self.polymer_distance(elem1, elem2)
                        
                        anchor = LoopAnchor(
                            anchor_id=f"call_{elem1.name}_{elem2.name}",
                            element1=elem1,
                            element2=elem2,
                            contact_strength=contact_strength,
                            anchor_type='call',
                            reciprocal=False,
                            distance=distance
                        )
                        anchors.append(anchor)
        
        return anchors
    
    def _count_internal_imports(self, tad: TAD) -> int:
        """Count imports within the TAD."""
        count = 0
        for module in tad.modules:
            if module in self._import_graph:
                for neighbor in self._import_graph.successors(module):
                    if neighbor in tad.modules:
                        count += 1
        return count
    
    def _count_external_imports(self, tad: TAD) -> int:
        """Count imports from the TAD to external modules."""
        count = 0
        for module in tad.modules:
            if module in self._import_graph:
                for neighbor in self._import_graph.successors(module):
                    if neighbor not in tad.modules:
                        count += 1
        return count
    
    def _split_tad_by_modules(self, tad: TAD) -> List[TAD]:
        """Split a TAD into smaller TADs by module groups."""
        # Group elements by module
        module_groups = defaultdict(list)
        for element in tad.elements:
            module_groups[element.module_path].append(element)
        
        sub_tads = []
        for i, (module, elements) in enumerate(module_groups.items()):
            if len(elements) >= 2:
                sub_tad = TAD(
                    tad_id=f"{tad.tad_id}_M{i}",
                    name=f"{tad.name}_module_{i}",
                    modules={module},
                    elements=elements,
                    boundary_strength=0.5,
                    import_density=0.0,
                    cohesion_score=0.0,
                    parent_tad=tad
                )
                sub_tads.append(sub_tad)
        
        return sub_tads if sub_tads else [tad]
    
    def _has_import_relationship(self, elem1: CodeElement, elem2: CodeElement) -> bool:
        """Check if two elements have an import relationship."""
        if not self._import_graph:
            return False
        
        return (self._import_graph.has_edge(elem1.module_path, elem2.module_path) or
                self._import_graph.has_edge(elem2.module_path, elem1.module_path))
    
    # ==================== Visualization Methods ====================
    
    def visualize_contact_matrix(self, output_path: str = "contact_matrix.png", 
                                figsize: Tuple[int, int] = (12, 10)) -> None:
        """
        Generate Hi-C style visualization of the contact matrix.
        
        Args:
            output_path: Path to save the visualization
            figsize: Figure size tuple
        """
        if self._contact_matrix is None:
            logger.warning("Contact matrix not built. Run analysis first.")
            return
        
        try:
            fig, ax = plt.subplots(figsize=figsize)
            
            # Create heatmap
            sns.heatmap(
                self._contact_matrix.matrix,
                cmap='Reds',
                square=True,
                cbar_kws={'label': 'Contact Probability'},
                ax=ax
            )
            
            # Add TAD boundaries
            for boundary in self._contact_matrix.tad_boundaries:
                ax.axhline(boundary, color='blue', linewidth=2, alpha=0.7)
                ax.axvline(boundary, color='blue', linewidth=2, alpha=0.7)
            
            # Mark loop anchors
            for anchor in self._contact_matrix.loop_anchors[:10]:  # Top 10 anchors
                if (anchor.element1 in self._element_to_index and 
                    anchor.element2 in self._element_to_index):
                    i = self._element_to_index[anchor.element1]
                    j = self._element_to_index[anchor.element2]
                    ax.plot(i, j, 'wo', markersize=8, alpha=0.8)
                    ax.plot(j, i, 'wo', markersize=8, alpha=0.8)
            
            ax.set_title('Hi-C Style Contact Matrix\n(Blue lines: TAD boundaries, White dots: Loop anchors)')
            ax.set_xlabel('Code Elements')
            ax.set_ylabel('Code Elements')
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Contact matrix visualization saved to {output_path}")
            
        except ImportError:
            logger.warning("matplotlib/seaborn not available for visualization")
        except Exception as e:
            logger.error(f"Failed to create visualization: {e}")
    
    def get_tad_summary(self) -> str:
        """Generate a summary report of detected TADs."""
        if not self._tads:
            return "No TADs detected. Run analysis first."
        
        report_lines = [
            "TAD DETECTION SUMMARY",
            "=" * 50,
            f"Total TADs detected: {len(self._tads)}",
            ""
        ]
        
        # Sort TADs by size
        sorted_tads = sorted(self._tads.values(), key=lambda t: len(t.elements), reverse=True)
        
        for tad in sorted_tads:
            report_lines.extend([
                f"TAD: {tad.name} ({tad.tad_id})",
                f"  Elements: {len(tad.elements)}",
                f"  Modules: {len(tad.modules)}",
                f"  Boundary strength: {tad.boundary_strength:.3f}",
                f"  Import density: {tad.import_density:.3f}",
                f"  Cohesion score: {tad.cohesion_score:.3f}",
                f"  Modules: {', '.join(sorted(tad.modules)[:3])}{'...' if len(tad.modules) > 3 else ''}",
                ""
            ])
        
        if self._loop_anchors:
            report_lines.extend([
                f"Loop anchors detected: {len(self._loop_anchors)}",
                "Top 5 anchors:"
            ])
            
            for anchor in self._loop_anchors[:5]:
                report_lines.append(
                    f"  {anchor.anchor_type}: {anchor.element1.name} <-> {anchor.element2.name} "
                    f"(strength: {anchor.contact_strength:.3f})"
                )
        
        return "\n".join(report_lines)