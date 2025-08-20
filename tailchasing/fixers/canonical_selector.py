"""
Facility-location algorithms for canonical definition selection.

This module implements 1-median and Fermat-Weber point algorithms to select
optimal canonical definitions from clusters of duplicate functions, using
polymer distance metrics inspired by chromatin dynamics.
"""

import ast
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import logging

from ..analyzers.chromatin_contact import ChromatinContactAnalyzer, CodeElement
from ..core.issues import Issue


@dataclass
class CanonicalDefinition:
    """Represents a selected canonical definition."""
    element: CodeElement
    confidence_score: float
    total_distance: float
    cluster_coverage: float
    migration_complexity: int


@dataclass
class AliasingPlaybook:
    """Migration plan for canonicalization."""
    canonical: CanonicalDefinition
    shadow_elements: List[CodeElement]
    alias_statements: List[str]
    import_rewiring: Dict[str, str]  # old_import -> new_import
    migration_steps: List[str]
    rollback_plan: List[str]
    estimated_effort: int  # 1-5 scale


@dataclass
class FacilityLocation:
    """Represents a facility location solution."""
    facility: CodeElement
    served_clients: List[CodeElement]
    total_cost: float
    max_distance: float
    facility_weight: float


class FacilityLocationSelector:
    """
    Selects canonical definitions using facility location algorithms.
    
    Implements 1-median and Fermat-Weber point calculations to find optimal
    canonical definitions that minimize total polymer distance to all
    duplicate instances in a cluster.
    """
    
    def __init__(self, chromatin_analyzer: ChromatinContactAnalyzer):
        self.chromatin_analyzer = chromatin_analyzer
        self.logger = logging.getLogger(__name__)
    
    def choose_canonical(self, cluster: List[CodeElement], 
                        distance_matrix: Optional[np.ndarray] = None) -> Tuple[CanonicalDefinition, AliasingPlaybook]:
        """
        Choose canonical definition using 1-median algorithm.
        
        Args:
            cluster: List of duplicate code elements to canonicalize
            distance_matrix: Optional precomputed distance matrix
            
        Returns:
            Tuple of (canonical_definition, aliasing_playbook)
        """
        if len(cluster) < 2:
            raise ValueError("Cluster must contain at least 2 elements")
        
        self.logger.info(f"Selecting canonical from cluster of {len(cluster)} elements")
        
        # Compute distance matrix if not provided
        if distance_matrix is None:
            distance_matrix = self._compute_distance_matrix(cluster)
        
        # Find 1-median solution
        facility_solution = self._solve_1_median(cluster, distance_matrix)
        
        # Create canonical definition
        canonical = CanonicalDefinition(
            element=facility_solution.facility,
            confidence_score=self._calculate_confidence(facility_solution),
            total_distance=facility_solution.total_cost,
            cluster_coverage=self._calculate_coverage(facility_solution, cluster),
            migration_complexity=self._estimate_migration_complexity(facility_solution)
        )
        
        # Generate aliasing playbook
        shadows = [elem for elem in cluster if elem != facility_solution.facility]
        playbook = self.generate_aliasing_playbook(canonical, shadows)
        
        self.logger.info(f"Selected canonical: {canonical.element.name} in {canonical.element.file_path}")
        self.logger.info(f"Confidence: {canonical.confidence_score:.3f}, Migration complexity: {canonical.migration_complexity}")
        
        return canonical, playbook
    
    def choose_multiple_facilities(self, cluster: List[CodeElement], 
                                 num_facilities: int) -> List[FacilityLocation]:
        """
        Choose multiple canonical definitions using k-median algorithm.
        
        Args:
            cluster: List of duplicate code elements
            num_facilities: Number of canonical definitions to select
            
        Returns:
            List of facility location solutions
        """
        if num_facilities >= len(cluster):
            raise ValueError("Number of facilities must be less than cluster size")
        
        distance_matrix = self._compute_distance_matrix(cluster)
        
        # Use greedy k-median approximation
        facilities = []
        remaining_clients = set(range(len(cluster)))
        
        for k in range(num_facilities):
            best_facility = None
            best_cost = float('inf')
            
            for candidate_idx in remaining_clients:
                # Calculate cost if this candidate becomes a facility
                total_cost = 0
                served_clients = []
                
                for client_idx in remaining_clients:
                    if client_idx == candidate_idx:
                        continue
                    
                    # Find minimum distance to any existing facility or this candidate
                    min_dist = distance_matrix[client_idx, candidate_idx]
                    for facility in facilities:
                        facility_idx = cluster.index(facility.facility)
                        min_dist = min(min_dist, distance_matrix[client_idx, facility_idx])
                    
                    if min_dist == distance_matrix[client_idx, candidate_idx]:
                        served_clients.append(client_idx)
                        total_cost += min_dist
                
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_facility = candidate_idx
            
            # Add best facility
            facility_element = cluster[best_facility]
            served_elements = [cluster[i] for i in served_clients if i != best_facility]
            
            facility = FacilityLocation(
                facility=facility_element,
                served_clients=served_elements,
                total_cost=best_cost,
                max_distance=max([distance_matrix[best_facility, i] for i in served_clients] or [0]),
                facility_weight=self._calculate_facility_weight(facility_element)
            )
            
            facilities.append(facility)
            remaining_clients.discard(best_facility)
        
        return facilities
    
    def generate_aliasing_playbook(self, canonical: CanonicalDefinition, 
                                 shadows: List[CodeElement]) -> AliasingPlaybook:
        """
        Generate comprehensive migration playbook for canonicalization.
        
        Args:
            canonical: Selected canonical definition
            shadows: Shadow/duplicate definitions to be aliased
            
        Returns:
            Complete aliasing playbook with migration steps
        """
        self.logger.info(f"Generating aliasing playbook for {len(shadows)} shadow definitions")
        
        # Generate alias statements
        alias_statements = []
        import_rewiring = {}
        migration_steps = []
        rollback_plan = []
        
        canonical_module = canonical.element.module_path
        canonical_name = canonical.element.name
        
        for shadow in shadows:
            shadow_module = shadow.module_path
            shadow_name = shadow.name
            
            # Create alias statement
            alias_stmt = f"# Legacy alias for backwards compatibility\n"
            alias_stmt += f"{shadow_name} = {canonical_name}"
            alias_statements.append(alias_stmt)
            
            # Import rewiring
            old_import = f"from {shadow_module} import {shadow_name}"
            new_import = f"from {canonical_module} import {canonical_name}"
            import_rewiring[old_import] = new_import
            
            # Migration steps
            migration_steps.extend([
                f"1. Review usage of {shadow_name} in {shadow.file_path}",
                f"2. Add alias: {shadow_name} = {canonical_name}",
                f"3. Update imports to use {canonical_module}.{canonical_name}",
                f"4. Test functionality with alias in place",
                f"5. Gradually migrate callers to use canonical name",
                f"6. Remove alias after migration complete"
            ])
            
            # Rollback plan
            rollback_plan.extend([
                f"git checkout -- {shadow.file_path}  # Restore original {shadow_name}",
                f"# Revert import changes for {shadow_name}",
                f"# Remove canonical import if added"
            ])
        
        # Calculate estimated effort
        effort_factors = [
            len(shadows),  # Number of duplicates
            len(import_rewiring),  # Import complexity
            canonical.migration_complexity,  # Migration complexity
            self._count_potential_usage(shadows)  # Usage frequency
        ]
        estimated_effort = min(5, max(1, sum(effort_factors) // 3))
        
        return AliasingPlaybook(
            canonical=canonical,
            shadow_elements=shadows,
            alias_statements=alias_statements,
            import_rewiring=import_rewiring,
            migration_steps=migration_steps,
            rollback_plan=rollback_plan,
            estimated_effort=estimated_effort
        )
    
    def _compute_distance_matrix(self, cluster: List[CodeElement]) -> np.ndarray:
        """Compute pairwise polymer distance matrix for cluster elements."""
        n = len(cluster)
        distance_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                distance = self.chromatin_analyzer.polymer_distance(cluster[i], cluster[j])
                distance_matrix[i, j] = distance
                distance_matrix[j, i] = distance  # Symmetric
        
        return distance_matrix
    
    def _solve_1_median(self, cluster: List[CodeElement], 
                       distance_matrix: np.ndarray) -> FacilityLocation:
        """
        Solve 1-median problem using exhaustive search.
        
        For small clusters, exhaustive search is optimal. For larger clusters,
        this could be replaced with approximation algorithms.
        """
        n = len(cluster)
        best_facility = None
        best_cost = float('inf')
        
        for facility_idx in range(n):
            # Calculate total cost if this element is the facility
            total_cost = 0
            max_distance = 0
            served_clients = []
            
            for client_idx in range(n):
                if client_idx != facility_idx:
                    dist = distance_matrix[facility_idx, client_idx]
                    total_cost += dist
                    max_distance = max(max_distance, dist)
                    served_clients.append(cluster[client_idx])
            
            if total_cost < best_cost:
                best_cost = total_cost
                best_facility = facility_idx
        
        return FacilityLocation(
            facility=cluster[best_facility],
            served_clients=[cluster[i] for i in range(n) if i != best_facility],
            total_cost=best_cost,
            max_distance=max([distance_matrix[best_facility, i] 
                           for i in range(n) if i != best_facility] or [0]),
            facility_weight=self._calculate_facility_weight(cluster[best_facility])
        )
    
    def _calculate_confidence(self, facility_solution: FacilityLocation) -> float:
        """Calculate confidence score for canonical selection."""
        # Factors that increase confidence:
        # 1. Low total distance cost
        # 2. High facility weight (centrality, usage)
        # 3. Balanced max distance (no outliers)
        
        # Normalize costs (simple heuristic)
        normalized_cost = min(1.0, facility_solution.total_cost / 10.0)
        cost_score = 1.0 - normalized_cost
        
        weight_score = min(1.0, facility_solution.facility_weight)
        
        # Balance score - penalize high max distance relative to average
        avg_distance = facility_solution.total_cost / max(1, len(facility_solution.served_clients))
        balance_score = 1.0 - min(1.0, facility_solution.max_distance / max(0.1, avg_distance * 2))
        
        confidence = (cost_score * 0.5 + weight_score * 0.3 + balance_score * 0.2)
        return max(0.0, min(1.0, confidence))
    
    def _calculate_coverage(self, facility_solution: FacilityLocation, 
                          full_cluster: List[CodeElement]) -> float:
        """Calculate what fraction of cluster is well-covered by this facility."""
        total_elements = len(full_cluster)
        well_covered = 1  # The facility itself
        
        # Count elements within reasonable distance
        reasonable_distance_threshold = facility_solution.total_cost / len(facility_solution.served_clients) if facility_solution.served_clients else 0
        
        for client in facility_solution.served_clients:
            client_idx = full_cluster.index(client)
            facility_idx = full_cluster.index(facility_solution.facility)
            
            # This would need distance matrix access - simplified for now
            # In practice, you'd pass this info or recompute
            well_covered += 1  # Simplified - assume all served clients are well-covered
        
        return well_covered / total_elements
    
    def _estimate_migration_complexity(self, facility_solution: FacilityLocation) -> int:
        """Estimate migration complexity on 1-5 scale."""
        factors = []
        
        # Number of clients to migrate
        factors.append(min(5, len(facility_solution.served_clients)))
        
        # Spread of distances (high spread = more complex)
        if facility_solution.served_clients and facility_solution.max_distance > 0:
            avg_distance = facility_solution.total_cost / len(facility_solution.served_clients)
            spread_factor = min(5, int(facility_solution.max_distance / avg_distance))
            factors.append(spread_factor)
        
        # File diversity (more files = more complex)
        unique_files = len(set(client.file_path for client in facility_solution.served_clients))
        unique_files += 1  # Include facility file
        factors.append(min(5, unique_files))
        
        return max(1, int(np.mean(factors)))
    
    def _calculate_facility_weight(self, element: CodeElement) -> float:
        """
        Calculate facility weight based on element characteristics.
        
        Higher weight indicates better suitability as canonical definition.
        """
        weight = 0.5  # Base weight
        
        # Prefer elements in main/core modules
        if any(term in element.module_path.lower() 
               for term in ['main', 'core', 'base', 'common', 'util']):
            weight += 0.2
        
        # Prefer elements in __init__.py or package roots
        if element.file_path.endswith('__init__.py'):
            weight += 0.3
        
        # Prefer functions with documentation
        if hasattr(element, 'ast_node') and element.ast_node:
            if (isinstance(element.ast_node, ast.FunctionDef) and 
                element.ast_node.body and 
                isinstance(element.ast_node.body[0], ast.Expr) and
                isinstance(element.ast_node.body[0].value, ast.Constant) and
                isinstance(element.ast_node.body[0].value.value, str)):
                weight += 0.2  # Has docstring
        
        # Prefer shorter, more generic names
        name_length_penalty = min(0.2, len(element.name) / 50)
        weight -= name_length_penalty
        
        return max(0.0, min(1.0, weight))
    
    def _count_potential_usage(self, shadows: List[CodeElement]) -> int:
        """Estimate potential usage count for migration planning."""
        # Simplified heuristic - in practice could analyze call graphs
        return len(shadows) * 3  # Assume average 3 usages per shadow function


class DuplicateClusterProcessor:
    """
    Processes duplicate function clusters from DuplicateFunctionAnalyzer.
    
    Integrates semantic duplicate detection with polymer distance weighting
    to produce ranked canonicalization suggestions.
    """
    
    def __init__(self, chromatin_analyzer: ChromatinContactAnalyzer):
        self.chromatin_analyzer = chromatin_analyzer
        self.facility_selector = FacilityLocationSelector(chromatin_analyzer)
        self.logger = logging.getLogger(__name__)
    
    def process_duplicate_issues(self, issues: List[Issue]) -> List[Tuple[CanonicalDefinition, AliasingPlaybook]]:
        """
        Process duplicate function issues and generate canonicalization suggestions.
        
        Args:
            issues: List of issues from DuplicateFunctionAnalyzer
            
        Returns:
            List of (canonical_definition, aliasing_playbook) tuples, ranked by priority
        """
        canonicalization_results = []
        
        # Group issues by cluster (functions with same semantic signature)
        clusters = self._group_issues_by_cluster(issues)
        
        self.logger.info(f"Processing {len(clusters)} duplicate clusters")
        
        for cluster_id, cluster_issues in clusters.items():
            try:
                # Extract CodeElements from issues
                cluster_elements = self._extract_code_elements(cluster_issues)
                
                if len(cluster_elements) < 2:
                    continue
                
                # Choose canonical definition
                canonical, playbook = self.facility_selector.choose_canonical(cluster_elements)
                
                # Add priority based on cluster characteristics
                canonical.priority = self._calculate_priority(cluster_elements, canonical)
                
                canonicalization_results.append((canonical, playbook))
                
                self.logger.info(f"Processed cluster {cluster_id}: canonical = {canonical.element.name}")
                
            except Exception as e:
                self.logger.error(f"Failed to process cluster {cluster_id}: {e}")
                continue
        
        # Sort by priority (higher priority first)
        canonicalization_results.sort(key=lambda x: x[0].priority, reverse=True)
        
        return canonicalization_results
    
    def _group_issues_by_cluster(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group duplicate issues by semantic cluster."""
        clusters = {}
        
        for issue in issues:
            if issue.kind != "duplicate_function":
                continue
            
            # Use function name + approximate signature as cluster key
            # In practice, this would use semantic similarity
            evidence = issue.evidence or {}
            func_name = evidence.get('function_name', 'unknown')
            signature = evidence.get('signature', '')
            
            cluster_key = f"{func_name}_{hash(signature) % 10000}"
            
            if cluster_key not in clusters:
                clusters[cluster_key] = []
            
            clusters[cluster_key].append(issue)
        
        return clusters
    
    def _extract_code_elements(self, cluster_issues: List[Issue]) -> List[CodeElement]:
        """Extract CodeElement objects from duplicate issues."""
        elements = []
        
        for issue in cluster_issues:
            try:
                # Extract location information from issue
                file_path = issue.file
                evidence = issue.evidence or {}
                
                # Create minimal CodeElement for facility location
                # In practice, this would be extracted from AST analysis
                element = CodeElement(
                    file_path=file_path,
                    name=evidence.get('function_name', 'unknown'),
                    node_type='function',
                    line_start=issue.line or 1,
                    line_end=issue.line or 1,
                    ast_node=None,  # Would be populated from analysis
                    module_path=self._file_to_module_path(file_path)
                )
                
                elements.append(element)
                
            except Exception as e:
                self.logger.warning(f"Could not extract element from issue: {e}")
                continue
        
        return elements
    
    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to Python module path."""
        path = Path(file_path)
        
        # Remove .py extension
        if path.suffix == '.py':
            path = path.with_suffix('')
        
        # Convert path separators to dots
        parts = path.parts
        
        # Find reasonable module root (remove common prefixes)
        for i, part in enumerate(parts):
            if part in ['src', 'lib', 'tailchasing']:
                parts = parts[i+1:]
                break
        
        return '.'.join(parts)
    
    def _calculate_priority(self, cluster_elements: List[CodeElement], 
                          canonical: CanonicalDefinition) -> float:
        """Calculate priority for canonicalization (0.0 to 1.0)."""
        # Factors that increase priority:
        # 1. Cluster size (more duplicates = higher priority)
        # 2. High confidence in canonical selection
        # 3. Low migration complexity
        # 4. Elements spread across multiple files
        
        cluster_size_score = min(1.0, len(cluster_elements) / 10.0)
        confidence_score = canonical.confidence_score
        complexity_penalty = (5 - canonical.migration_complexity) / 4.0
        
        # File diversity score
        unique_files = len(set(elem.file_path for elem in cluster_elements))
        diversity_score = min(1.0, unique_files / len(cluster_elements))
        
        priority = (
            cluster_size_score * 0.3 +
            confidence_score * 0.3 +
            complexity_penalty * 0.2 +
            diversity_score * 0.2
        )
        
        return max(0.0, min(1.0, priority))