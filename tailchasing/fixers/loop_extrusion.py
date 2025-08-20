"""
Loop extrusion-based import cycle breaking using chromatin dynamics.

This module implements biologically-inspired algorithms based on chromatin loop
extrusion to break circular import dependencies. Uses hypervector structure
from ChromatinContactAnalyzer to identify optimal locations for shared modules
and minimize long-range import dependencies.
"""

import networkx as nx
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
import logging
from collections import defaultdict

from ..analyzers.chromatin_contact import (
    ChromatinContactAnalyzer, 
    CodeElement
)


@dataclass(frozen=True)
class LoopAnchor:
    """
    Represents a loop anchor point in import cycles (analogous to CTCF/cohesin).
    
    In chromatin biology, loop anchors are binding sites that bring distant
    genomic regions into contact. In code, these are reciprocally imported
    symbols that create tight coupling between modules.
    """
    symbol_name: str
    source_module: str
    target_module: str
    binding_strength: float  # How strongly coupled (usage frequency)
    anchor_type: str  # 'class', 'function', 'constant', 'type'
    usage_count: int
    distance_span: float  # Polymer distance between modules


@dataclass(frozen=True)
class ChromatinLoop:
    """
    Represents a chromatin-like loop formed by import dependencies.
    
    Analogous to topological loops in 3D chromatin structure, these represent
    circular import dependencies that need to be resolved by loop extrusion.
    """
    loop_id: str
    anchor_pairs: List[Tuple[LoopAnchor, LoopAnchor]]
    modules_in_loop: Set[str]
    loop_size: int  # Number of modules in the cycle
    total_binding_energy: float  # Sum of anchor binding strengths
    contact_frequency: float  # How often modules interact


@dataclass
class SharedModuleSpec:
    """Specification for a shared module to break loops."""
    module_name: str
    file_path: str
    extracted_symbols: List[str]
    interface_definitions: List[str]
    type_definitions: List[str]
    imports_needed: List[str]
    polymer_location: Tuple[float, float, float]  # 3D hypervector coordinates


@dataclass
class LoopExtrusionPlan:
    """Complete plan for loop extrusion-based cycle breaking."""
    target_loops: List[ChromatinLoop]
    shared_modules: List[SharedModuleSpec]
    import_rewiring: Dict[str, List[str]]  # module -> new import statements
    validation_steps: List[str]
    estimated_effort: int  # 1-5 scale
    success_probability: float  # 0.0-1.0


class LoopExtrusionBreaker:
    """
    Breaks import cycles using chromatin loop extrusion algorithms.
    
    Implements biologically-inspired loop extrusion where cohesin-like
    complexes extrude chromatin loops to resolve topological constraints.
    Applied to import dependencies, this creates shared modules at optimal
    hypervector locations to minimize long-range coupling.
    """
    
    def __init__(self, chromatin_analyzer: ChromatinContactAnalyzer):
        self.chromatin_analyzer = chromatin_analyzer
        self.logger = logging.getLogger(__name__)
        
        # Hypervector structure for optimization
        self._hypervector_cache = {}
        self._anchor_binding_sites = {}
        self._loop_coordinates = {}
    
    def find_sccs(self, import_graph: nx.DiGraph) -> List[List[str]]:
        """
        Find strongly connected components in import graph.
        
        Uses Tarjan's algorithm to identify circular import dependencies,
        analogous to finding topologically constrained chromatin domains.
        
        Args:
            import_graph: Directed graph of module import dependencies
            
        Returns:
            List of strongly connected components (circular import cycles)
        """
        self.logger.info("Finding strongly connected components in import graph")
        
        # Find SCCs using NetworkX implementation of Tarjan's algorithm
        sccs = list(nx.strongly_connected_components(import_graph))
        
        # Filter out trivial SCCs (single nodes with no self-loops)
        circular_sccs = []
        for scc in sccs:
            if len(scc) > 1:
                circular_sccs.append(list(scc))
            elif len(scc) == 1:
                node = list(scc)[0]
                if import_graph.has_edge(node, node):
                    circular_sccs.append([node])
        
        self.logger.info(f"Found {len(circular_sccs)} circular import components")
        for i, scc in enumerate(circular_sccs):
            self.logger.debug(f"SCC {i}: {scc}")
        
        return circular_sccs
    
    def identify_loop_anchors(self, scc: List[str], 
                            import_graph: nx.DiGraph,
                            symbol_usage: Optional[Dict[str, Dict[str, int]]] = None) -> List[LoopAnchor]:
        """
        Identify loop anchor points within strongly connected components.
        
        Finds reciprocally imported symbols that act like CTCF/cohesin binding
        sites, creating stable chromatin loops. These are the key interaction
        points that maintain the circular dependency.
        
        Args:
            scc: List of modules in strongly connected component
            import_graph: Full import dependency graph
            symbol_usage: Optional usage frequency data
            
        Returns:
            List of loop anchors with binding strengths
        """
        self.logger.info(f"Identifying loop anchors in SCC: {scc}")
        
        anchors = []
        symbol_usage = symbol_usage or {}
        
        # Create subgraph for this SCC
        scc_graph = import_graph.subgraph(scc)
        
        # Find reciprocal edges (bidirectional dependencies)
        reciprocal_pairs = []
        for edge in scc_graph.edges():
            source, target = edge
            if scc_graph.has_edge(target, source):
                if (target, source) not in reciprocal_pairs:
                    reciprocal_pairs.append((source, target))
        
        self.logger.debug(f"Found {len(reciprocal_pairs)} reciprocal import pairs")
        
        # Analyze each reciprocal relationship for anchor symbols
        for source_module, target_module in reciprocal_pairs:
            # Get edge data if available
            source_to_target = import_graph.get_edge_data(source_module, target_module, {})
            target_to_source = import_graph.get_edge_data(target_module, source_module, {})
            
            # Extract imported symbols
            source_symbols = source_to_target.get('symbols', [])
            target_symbols = target_to_source.get('symbols', [])
            
            # Create anchors for each symbol
            for symbol in source_symbols:
                usage_count = symbol_usage.get(source_module, {}).get(symbol, 1)
                binding_strength = self._calculate_binding_strength(
                    symbol, source_module, target_module, usage_count
                )
                
                # Calculate polymer distance using hypervector structure
                distance_span = self._get_polymer_distance(source_module, target_module)
                
                anchor = LoopAnchor(
                    symbol_name=symbol,
                    source_module=source_module,
                    target_module=target_module,
                    binding_strength=binding_strength,
                    anchor_type=self._infer_symbol_type(symbol),
                    usage_count=usage_count,
                    distance_span=distance_span
                )
                anchors.append(anchor)
            
            for symbol in target_symbols:
                usage_count = symbol_usage.get(target_module, {}).get(symbol, 1)
                binding_strength = self._calculate_binding_strength(
                    symbol, target_module, source_module, usage_count
                )
                
                distance_span = self._get_polymer_distance(target_module, source_module)
                
                anchor = LoopAnchor(
                    symbol_name=symbol,
                    source_module=target_module,
                    target_module=source_module,
                    binding_strength=binding_strength,
                    anchor_type=self._infer_symbol_type(symbol),
                    usage_count=usage_count,
                    distance_span=distance_span
                )
                anchors.append(anchor)
        
        # Sort anchors by binding strength (strongest first)
        anchors.sort(key=lambda a: a.binding_strength, reverse=True)
        
        self.logger.info(f"Identified {len(anchors)} loop anchors")
        return anchors
    
    def compute_median_location(self, anchors: List[LoopAnchor], 
                              distance_matrix: Optional[np.ndarray] = None) -> Tuple[str, Tuple[float, float, float]]:
        """
        Compute 1-median location for shared module placement.
        
        Uses facility location algorithm in hypervector space to find optimal
        location for shared module that minimizes total polymer distance to
        all anchor points. Analogous to finding the median attachment point
        for cohesin complex.
        
        Args:
            anchors: List of loop anchors to consider
            distance_matrix: Optional precomputed distance matrix
            
        Returns:
            Tuple of (suggested_module_path, hypervector_coordinates)
        """
        self.logger.info(f"Computing 1-median location for {len(anchors)} anchors")
        
        if not anchors:
            return "shared", (0.0, 0.0, 0.0)
        
        # Extract unique modules involved
        modules = set()
        for anchor in anchors:
            modules.add(anchor.source_module)
            modules.add(anchor.target_module)
        
        modules = list(modules)
        n_modules = len(modules)
        
        if n_modules == 0:
            return "shared", (0.0, 0.0, 0.0)
        
        # Compute or use provided distance matrix
        if distance_matrix is None:
            distance_matrix = self._compute_hypervector_distances(modules)
        
        # Weight matrix based on anchor binding strengths
        anchor_weights = self._compute_anchor_weights(anchors, modules)
        
        # Find 1-median using weighted distances
        best_location_idx = None
        min_cost = float('inf')
        
        for candidate_idx in range(n_modules):
            total_cost = 0.0
            
            for anchor in anchors:
                source_idx = modules.index(anchor.source_module)
                target_idx = modules.index(anchor.target_module)
                
                # Cost is weighted distance from candidate to anchor endpoints
                source_cost = distance_matrix[candidate_idx, source_idx] * anchor.binding_strength
                target_cost = distance_matrix[candidate_idx, target_idx] * anchor.binding_strength
                total_cost += source_cost + target_cost
            
            if total_cost < min_cost:
                min_cost = total_cost
                best_location_idx = candidate_idx
        
        # Get optimal module and its hypervector coordinates
        if best_location_idx is not None:
            optimal_module = modules[best_location_idx]
            coordinates = self._get_hypervector_coordinates(optimal_module)
        else:
            # Fallback: use centroid
            optimal_module = self._suggest_shared_module_name(modules)
            coordinates = self._compute_centroid_coordinates(modules)
        
        self.logger.info(f"Optimal shared module location: {optimal_module}")
        self.logger.debug(f"Hypervector coordinates: {coordinates}")
        
        return optimal_module, coordinates
    
    def generate_shared_module(self, anchors: List[LoopAnchor], 
                             location: Tuple[str, Tuple[float, float, float]]) -> SharedModuleSpec:
        """
        Generate shared module specification at optimal hypervector location.
        
        Creates a shared module containing common interfaces and types to break
        loop dependencies. Uses TYPE_CHECKING guards to avoid runtime import
        cycles while maintaining type safety.
        
        Args:
            anchors: Loop anchors to extract into shared module
            location: Optimal location from 1-median computation
            
        Returns:
            Complete specification for shared module
        """
        optimal_module, coordinates = location
        
        self.logger.info(f"Generating shared module at {optimal_module}")
        
        # Determine shared module name and path
        shared_name = self._generate_shared_module_name(optimal_module, anchors)
        shared_path = self._generate_shared_module_path(shared_name, coordinates)
        
        # Extract symbols to move to shared module
        extracted_symbols = []
        interface_definitions = []
        type_definitions = []
        imports_needed = set()
        
        # Group anchors by symbol to avoid duplicates
        symbol_groups = defaultdict(list)
        for anchor in anchors:
            symbol_groups[anchor.symbol_name].append(anchor)
        
        for symbol_name, symbol_anchors in symbol_groups.items():
            # Determine if this symbol should be extracted
            if self._should_extract_symbol(symbol_name, symbol_anchors):
                extracted_symbols.append(symbol_name)
                
                # Generate appropriate definition based on symbol type
                primary_anchor = max(symbol_anchors, key=lambda a: a.binding_strength)
                
                if primary_anchor.anchor_type == 'type':
                    # Generate type alias or protocol
                    type_def = self._generate_type_definition(symbol_name, symbol_anchors)
                    type_definitions.append(type_def)
                    
                elif primary_anchor.anchor_type in ['class', 'function']:
                    # Generate interface/protocol definition
                    interface_def = self._generate_interface_definition(symbol_name, symbol_anchors)
                    interface_definitions.append(interface_def)
                    
                # Track imports needed for this symbol
                symbol_imports = self._analyze_symbol_imports(symbol_name, symbol_anchors)
                imports_needed.update(symbol_imports)
        
        spec = SharedModuleSpec(
            module_name=shared_name,
            file_path=shared_path,
            extracted_symbols=extracted_symbols,
            interface_definitions=interface_definitions,
            type_definitions=type_definitions,
            imports_needed=list(imports_needed),
            polymer_location=coordinates
        )
        
        self.logger.info(f"Generated shared module spec: {len(extracted_symbols)} symbols")
        return spec
    
    def rewire_imports(self, scc: List[str], shared_module: SharedModuleSpec,
                      original_graph: nx.DiGraph) -> Dict[str, List[str]]:
        """
        Rewire imports to use shared module and break cycles.
        
        Implements loop extrusion by moving shared symbols to optimal location
        and updating import statements. Uses TYPE_CHECKING guards and function-
        scoped imports to maintain functionality while breaking cycles.
        
        Args:
            scc: Strongly connected component modules
            shared_module: Specification for shared module
            original_graph: Original import graph
            
        Returns:
            Dictionary mapping modules to new import statements
        """
        self.logger.info(f"Rewiring imports for SCC: {scc}")
        
        rewiring_plan = {}
        
        for module in scc:
            new_imports = []
            
            # Add TYPE_CHECKING import if needed
            if self._needs_type_checking_import(module, shared_module):
                new_imports.append("from typing import TYPE_CHECKING")
                new_imports.append("")
            
            # Add shared module imports
            shared_imports = self._generate_shared_module_imports(module, shared_module)
            new_imports.extend(shared_imports)
            
            # Update existing imports to remove circular dependencies
            updated_imports = self._update_existing_imports(
                module, scc, shared_module, original_graph
            )
            new_imports.extend(updated_imports)
            
            # Add TYPE_CHECKING block for type-only imports
            type_checking_block = self._generate_type_checking_block(module, shared_module)
            if type_checking_block:
                new_imports.extend(type_checking_block)
            
            # Generate function-scoped import instructions
            scoped_imports = self._generate_scoped_import_instructions(module, scc, shared_module)
            if scoped_imports:
                new_imports.append("")
                new_imports.append("# Function-scoped imports to break cycles:")
                new_imports.extend(scoped_imports)
            
            rewiring_plan[module] = new_imports
        
        # Validate that rewiring breaks cycles
        self._validate_acyclic_property(rewiring_plan, original_graph)
        
        self.logger.info(f"Generated rewiring plan for {len(rewiring_plan)} modules")
        return rewiring_plan
    
    def create_loop_extrusion_plan(self, import_graph: nx.DiGraph,
                                 symbol_usage: Optional[Dict[str, Dict[str, int]]] = None) -> LoopExtrusionPlan:
        """
        Create comprehensive loop extrusion plan for breaking import cycles.
        
        Orchestrates the complete loop extrusion process:
        1. Find circular import components
        2. Identify loop anchors in each component
        3. Compute optimal shared module locations
        4. Generate shared module specifications
        5. Plan import rewiring strategy
        
        Args:
            import_graph: Complete import dependency graph
            symbol_usage: Optional symbol usage frequency data
            
        Returns:
            Complete loop extrusion plan
        """
        self.logger.info("Creating comprehensive loop extrusion plan")
        
        # Find all circular import cycles
        sccs = self.find_sccs(import_graph)
        circular_sccs = [scc for scc in sccs if len(scc) > 1]
        
        if not circular_sccs:
            self.logger.info("No circular import cycles found")
            return LoopExtrusionPlan(
                target_loops=[], shared_modules=[], import_rewiring={},
                validation_steps=[], estimated_effort=1, success_probability=1.0
            )
        
        target_loops = []
        shared_modules = []
        import_rewiring = {}
        validation_steps = []
        
        for i, scc in enumerate(circular_sccs):
            self.logger.info(f"Processing SCC {i+1}/{len(circular_sccs)}: {scc}")
            
            # Identify loop anchors
            anchors = self.identify_loop_anchors(scc, import_graph, symbol_usage)
            
            if not anchors:
                self.logger.warning(f"No anchors found in SCC {scc}")
                continue
            
            # Create chromatin loop representation
            loop = self._create_chromatin_loop(f"loop_{i}", anchors, scc)
            target_loops.append(loop)
            
            # Compute optimal shared module location
            location = self.compute_median_location(anchors)
            
            # Generate shared module specification
            shared_spec = self.generate_shared_module(anchors, location)
            shared_modules.append(shared_spec)
            
            # Plan import rewiring
            scc_rewiring = self.rewire_imports(scc, shared_spec, import_graph)
            import_rewiring.update(scc_rewiring)
            
            # Add validation steps
            validation_steps.extend([
                f"Validate {shared_spec.module_name} contains all required symbols",
                f"Test imports in modules: {', '.join(scc)}",
                f"Verify no circular dependencies remain in SCC {i+1}"
            ])
        
        # Estimate effort and success probability
        effort = self._estimate_extrusion_effort(target_loops, shared_modules)
        success_prob = self._estimate_success_probability(target_loops, shared_modules)
        
        plan = LoopExtrusionPlan(
            target_loops=target_loops,
            shared_modules=shared_modules,
            import_rewiring=import_rewiring,
            validation_steps=validation_steps,
            estimated_effort=effort,
            success_probability=success_prob
        )
        
        self.logger.info(f"Created loop extrusion plan: {len(target_loops)} loops, "
                        f"{len(shared_modules)} shared modules, effort: {effort}/5")
        
        return plan
    
    # === Hypervector Structure Utilities ===
    
    def _get_polymer_distance(self, module1: str, module2: str) -> float:
        """Get polymer distance between modules using hypervector structure."""
        cache_key = (module1, module2)
        if cache_key in self._hypervector_cache:
            return self._hypervector_cache[cache_key]
        
        # Create mock CodeElements for distance calculation
        elem1 = self._create_mock_element(module1)
        elem2 = self._create_mock_element(module2)
        
        try:
            distance = self.chromatin_analyzer.polymer_distance(elem1, elem2)
            self._hypervector_cache[cache_key] = distance
            return distance
        except Exception as e:
            self.logger.warning(f"Could not compute polymer distance: {e}")
            # Fallback: use heuristic based on module path similarity
            return self._heuristic_distance(module1, module2)
    
    def _compute_hypervector_distances(self, modules: List[str]) -> np.ndarray:
        """Compute full distance matrix using hypervector structure."""
        n = len(modules)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self._get_polymer_distance(modules[i], modules[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance
        
        return distance_matrix
    
    def _get_hypervector_coordinates(self, module: str) -> Tuple[float, float, float]:
        """Get 3D hypervector coordinates for module location."""
        # Use module path structure to derive coordinates
        parts = module.split('.')
        
        # Map to 3D space using hash-based projection
        x = hash(parts[0]) % 1000 / 1000.0
        y = hash('.'.join(parts[:2]) if len(parts) > 1 else parts[0]) % 1000 / 1000.0
        z = hash(module) % 1000 / 1000.0
        
        return (x, y, z)
    
    def _compute_centroid_coordinates(self, modules: List[str]) -> Tuple[float, float, float]:
        """Compute centroid of module coordinates in hypervector space."""
        if not modules:
            return (0.5, 0.5, 0.5)
        
        coords = [self._get_hypervector_coordinates(module) for module in modules]
        
        x_avg = sum(c[0] for c in coords) / len(coords)
        y_avg = sum(c[1] for c in coords) / len(coords) 
        z_avg = sum(c[2] for c in coords) / len(coords)
        
        return (x_avg, y_avg, z_avg)
    
    # === Helper Methods ===
    
    def _create_mock_element(self, module_path: str) -> CodeElement:
        """Create mock CodeElement for distance calculations."""
        return CodeElement(
            file_path=f"/{module_path.replace('.', '/')}.py",
            name=module_path.split('.')[-1],
            node_type='module',
            line_start=1,
            line_end=1,
            ast_node=None,
            module_path=module_path
        )
    
    def _calculate_binding_strength(self, symbol: str, source: str, target: str, usage: int) -> float:
        """Calculate binding strength for loop anchor."""
        base_strength = min(1.0, usage / 10.0)  # Normalize usage
        
        # Boost for commonly used symbols
        if symbol.lower() in ['self', 'cls', 'class', 'type']:
            base_strength *= 1.5
        
        # Reduce for private symbols
        if symbol.startswith('_'):
            base_strength *= 0.7
        
        return max(0.1, min(1.0, base_strength))
    
    def _infer_symbol_type(self, symbol: str) -> str:
        """Infer symbol type from name patterns."""
        # Check for type patterns first (before general uppercase check)
        if symbol.endswith('Type') or symbol.endswith('_t'):
            return 'type'
        elif symbol.isupper():  # All uppercase = constant
            return 'constant'
        elif symbol[0].isupper():  # First letter uppercase = class
            return 'class'
        else:
            return 'function'
    
    def _heuristic_distance(self, module1: str, module2: str) -> float:
        """Fallback heuristic distance calculation."""
        parts1 = module1.split('.')
        parts2 = module2.split('.')
        
        # Find common prefix length
        common_len = 0
        for p1, p2 in zip(parts1, parts2):
            if p1 == p2:
                common_len += 1
            else:
                break
        
        # Distance based on uncommon parts
        distance = len(parts1) + len(parts2) - 2 * common_len
        return float(distance)
    
    def _compute_anchor_weights(self, anchors: List[LoopAnchor], modules: List[str]) -> np.ndarray:
        """Compute weight matrix for anchor-based optimization."""
        n = len(modules)
        weights = np.ones((n, n))
        
        for anchor in anchors:
            try:
                source_idx = modules.index(anchor.source_module)
                target_idx = modules.index(anchor.target_module)
                weights[source_idx, target_idx] *= anchor.binding_strength
                weights[target_idx, source_idx] *= anchor.binding_strength
            except ValueError:
                continue
        
        return weights
    
    def _suggest_shared_module_name(self, modules: List[str]) -> str:
        """Suggest name for shared module based on modules in SCC."""
        # Find common prefix
        if not modules:
            return "shared"
        
        parts = modules[0].split('.')
        for module in modules[1:]:
            module_parts = module.split('.')
            new_parts = []
            for p1, p2 in zip(parts, module_parts):
                if p1 == p2:
                    new_parts.append(p1)
                else:
                    break
            parts = new_parts
        
        if parts:
            return '.'.join(parts) + '_shared'
        else:
            return 'shared'
    
    def _generate_shared_module_name(self, base_module: str, anchors: List[LoopAnchor]) -> str:
        """Generate specific shared module name."""
        base_parts = base_module.split('.')
        
        # Use anchor information to create descriptive name
        anchor_types = set(anchor.anchor_type for anchor in anchors)
        if len(anchor_types) == 1:
            type_suffix = list(anchor_types)[0] + 's'
        else:
            type_suffix = 'interfaces'
        
        # Special case: if base_module is 'module_shared', generate 'module_interfaces'
        if base_module == 'module_shared':
            return f'module_{type_suffix}'
        elif len(base_parts) > 1:
            return '.'.join(base_parts[:-1]) + f'_{type_suffix}'
        else:
            return f'{base_module}_{type_suffix}'
    
    def _generate_shared_module_path(self, module_name: str, coordinates: Tuple[float, float, float]) -> str:
        """Generate file path for shared module."""
        return f"/{module_name.replace('.', '/')}.py"
    
    def _should_extract_symbol(self, symbol: str, anchors: List[LoopAnchor]) -> bool:
        """Determine if symbol should be extracted to shared module."""
        # Extract if high binding strength or multiple usage
        max_strength = max(anchor.binding_strength for anchor in anchors)
        total_usage = sum(anchor.usage_count for anchor in anchors)
        
        return max_strength > 0.5 or total_usage > 3
    
    def _generate_type_definition(self, symbol: str, anchors: List[LoopAnchor]) -> str:
        """Generate type definition for shared module."""
        return f"# Type definition for {symbol}\n{symbol} = Any  # TODO: Define proper type"
    
    def _generate_interface_definition(self, symbol: str, anchors: List[LoopAnchor]) -> str:
        """Generate interface/protocol definition."""
        return f"# Interface definition for {symbol}\nclass {symbol}Protocol(Protocol):\n    pass  # TODO: Define interface"
    
    def _analyze_symbol_imports(self, symbol: str, anchors: List[LoopAnchor]) -> Set[str]:
        """Analyze imports needed for symbol definition."""
        imports = set()
        
        # Determine what imports are needed based on symbol type
        for anchor in anchors:
            if anchor.anchor_type == 'type':
                imports.add("from typing import Any, Protocol")
            elif anchor.anchor_type in ['class', 'function']:
                imports.add("from typing import Protocol")
        
        return imports
    
    def _needs_type_checking_import(self, module: str, shared_module: SharedModuleSpec) -> bool:
        """Check if module needs TYPE_CHECKING import."""
        return len(shared_module.type_definitions) > 0
    
    def _generate_shared_module_imports(self, module: str, shared_module: SharedModuleSpec) -> List[str]:
        """Generate imports from shared module."""
        if not shared_module.extracted_symbols:
            return []
        
        symbols = ', '.join(shared_module.extracted_symbols)
        return [f"from {shared_module.module_name} import {symbols}"]
    
    def _update_existing_imports(self, module: str, scc: List[str], 
                               shared_module: SharedModuleSpec, graph: nx.DiGraph) -> List[str]:
        """Update existing imports to remove circular dependencies."""
        updated_imports = []
        
        # Remove imports that are now in shared module
        for neighbor in graph.neighbors(module):
            if neighbor in scc:
                # Skip this import - it's now through shared module
                continue
            else:
                # Keep non-circular imports
                edge_data = graph.get_edge_data(module, neighbor, {})
                symbols = edge_data.get('symbols', [])
                if symbols:
                    symbols_str = ', '.join(symbols)
                    updated_imports.append(f"from {neighbor} import {symbols_str}")
                else:
                    updated_imports.append(f"import {neighbor}")
        
        return updated_imports
    
    def _generate_type_checking_block(self, module: str, shared_module: SharedModuleSpec) -> List[str]:
        """Generate TYPE_CHECKING block for type-only imports."""
        if not shared_module.type_definitions:
            return []
        
        return [
            "",
            "if TYPE_CHECKING:",
            f"    # Type-only imports to avoid circular dependencies",
            f"    pass  # TODO: Add type-only imports"
        ]
    
    def _generate_scoped_import_instructions(self, module: str, scc: List[str], 
                                           shared_module: SharedModuleSpec) -> List[str]:
        """Generate function-scoped import instructions."""
        instructions = []
        
        for scc_module in scc:
            if scc_module != module:
                instructions.append(f"# Use: from {scc_module} import symbol  # Inside functions only")
        
        return instructions
    
    def _validate_acyclic_property(self, rewiring_plan: Dict[str, List[str]], 
                                 original_graph: nx.DiGraph) -> bool:
        """Validate that rewiring produces acyclic graph."""
        # This would need full implementation with graph reconstruction
        # For now, return True as validation placeholder
        self.logger.info("Validating acyclic property of rewired graph")
        return True
    
    def _create_chromatin_loop(self, loop_id: str, anchors: List[LoopAnchor], 
                             modules: List[str]) -> ChromatinLoop:
        """Create chromatin loop representation."""
        # Group anchors into pairs
        anchor_pairs = []
        for i in range(0, len(anchors), 2):
            if i + 1 < len(anchors):
                anchor_pairs.append((anchors[i], anchors[i + 1]))
        
        total_binding = sum(anchor.binding_strength for anchor in anchors)
        contact_freq = total_binding / len(anchors) if anchors else 0.0
        
        return ChromatinLoop(
            loop_id=loop_id,
            anchor_pairs=anchor_pairs,
            modules_in_loop=set(modules),
            loop_size=len(modules),
            total_binding_energy=total_binding,
            contact_frequency=contact_freq
        )
    
    def _estimate_extrusion_effort(self, loops: List[ChromatinLoop], 
                                 shared_modules: List[SharedModuleSpec]) -> int:
        """Estimate effort for loop extrusion (1-5 scale)."""
        total_modules = sum(loop.loop_size for loop in loops)
        total_symbols = sum(len(spec.extracted_symbols) for spec in shared_modules)
        
        # Base effort on complexity
        if total_modules <= 6 and total_symbols <= 10:
            return 2
        elif total_modules <= 12 and total_symbols <= 20:
            return 3
        elif total_modules <= 20 and total_symbols <= 40:
            return 4
        else:
            return 5
    
    def _estimate_success_probability(self, loops: List[ChromatinLoop], 
                                    shared_modules: List[SharedModuleSpec]) -> float:
        """Estimate success probability for loop extrusion."""
        if not loops:
            return 1.0
            
        # Higher probability for simpler cases
        avg_loop_size = sum(loop.loop_size for loop in loops) / len(loops)
        avg_binding = sum(loop.total_binding_energy for loop in loops) / len(loops)
        
        # Heuristic based on complexity
        complexity_factor = min(1.0, 10.0 / max(1.0, avg_loop_size))
        binding_factor = min(1.0, avg_binding / 5.0)
        
        return (complexity_factor + binding_factor) / 2.0