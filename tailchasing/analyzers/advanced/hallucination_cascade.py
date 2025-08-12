"""
Hallucination Cascade Detection System for TailChasingFixer.

This module detects when LLMs create entire fictional subsystems to satisfy errors.
It analyzes dependency graphs, temporal patterns, and isolation metrics to identify
clusters of interdependent classes that were likely created together without
sufficient external integration.
"""

from __future__ import annotations
import ast
import logging
import os
import subprocess
import re
from collections import defaultdict, Counter
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
import hashlib
import statistics

try:
    import networkx as nx
except ImportError:
    # Fallback for environments without networkx
    nx = None

from ..base import BaseAnalyzer, AnalysisContext
from ...core.issues import Issue
from ...core.utils import safe_get_lineno, safe_get_end_lineno
from .pattern_types import TailChasingPattern, PatternEvidence, PatternSeverity

logger = logging.getLogger(__name__)


@dataclass
class ComponentInfo:
    """Information about a code component (class, function, module)."""
    
    name: str
    file_path: str
    component_type: str  # 'class', 'function', 'module'
    line_number: int
    end_line_number: Optional[int] = None
    
    # Dependencies
    dependencies: Set[str] = field(default_factory=set)
    dependents: Set[str] = field(default_factory=set)
    
    # Temporal information
    creation_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    commit_hash: Optional[str] = None
    
    # Metrics
    external_references: int = 0
    internal_references: int = 0
    complexity_score: float = 0.0
    
    # Classification
    is_abstract: bool = False
    is_interface: bool = False
    has_implementation: bool = True
    
    def get_isolation_ratio(self) -> float:
        """Calculate how isolated this component is from external code."""
        total_refs = self.external_references + self.internal_references
        if total_refs == 0:
            return 1.0  # Completely isolated
        return self.internal_references / total_refs
    
    def get_reference_density(self) -> float:
        """Calculate reference density relative to component size."""
        base_size = max(1, (self.end_line_number or self.line_number) - self.line_number + 1)
        return (self.external_references + self.internal_references) / base_size


@dataclass
class HallucinationCluster:
    """Represents a cluster of components that may be a hallucination cascade."""
    
    cluster_id: str
    components: List[ComponentInfo] = field(default_factory=list)
    
    # Graph metrics
    internal_edges: int = 0
    external_edges: int = 0
    cluster_density: float = 0.0
    
    # Temporal metrics
    creation_timespan: Optional[timedelta] = None
    creation_burst: bool = False
    temporal_consistency: float = 0.0
    
    # Isolation metrics
    isolation_score: float = 0.0
    external_reference_ratio: float = 0.0
    
    # Confidence metrics
    confidence_score: float = 0.0
    hallucination_indicators: List[str] = field(default_factory=list)
    
    # Evidence
    supporting_evidence: Dict[str, Any] = field(default_factory=dict)
    
    def add_component(self, component: ComponentInfo) -> None:
        """Add a component to the cluster."""
        if component not in self.components:
            self.components.append(component)
    
    def calculate_metrics(self) -> None:
        """Calculate all cluster metrics."""
        if not self.components:
            return
        
        # Calculate isolation metrics
        total_external = sum(comp.external_references for comp in self.components)
        total_internal = sum(comp.internal_references for comp in self.components)
        total_refs = total_external + total_internal
        
        self.external_reference_ratio = total_external / total_refs if total_refs > 0 else 0.0
        self.isolation_score = 1.0 - self.external_reference_ratio
        
        # Calculate temporal metrics
        creation_dates = [comp.creation_date for comp in self.components if comp.creation_date]
        if creation_dates:
            earliest = min(creation_dates)
            latest = max(creation_dates)
            self.creation_timespan = latest - earliest
            
            # Consider it a "burst" if all components created within 24 hours
            self.creation_burst = self.creation_timespan <= timedelta(hours=24)
            
            # Calculate temporal consistency (how close together components were created)
            if len(creation_dates) > 1:
                time_diffs = []
                for i in range(len(creation_dates) - 1):
                    diff = abs((creation_dates[i+1] - creation_dates[i]).total_seconds())
                    time_diffs.append(diff)
                
                # Normalize temporal consistency (0-1, higher = more consistent timing)
                avg_diff = statistics.mean(time_diffs)
                max_reasonable_diff = 86400  # 24 hours in seconds
                self.temporal_consistency = max(0.0, 1.0 - (avg_diff / max_reasonable_diff))
        
        # Calculate cluster density (internal connections vs possible connections)
        num_components = len(self.components)
        max_possible_edges = num_components * (num_components - 1) // 2
        if max_possible_edges > 0:
            self.cluster_density = self.internal_edges / max_possible_edges
    
    def is_likely_hallucination(self, threshold: float = 0.7) -> bool:
        """Determine if cluster is likely a hallucination cascade."""
        return self.confidence_score >= threshold


class HallucinationCascadeDetector(BaseAnalyzer):
    """
    Detector for hallucination cascades - fictional subsystems created by LLMs.
    
    A hallucination cascade occurs when:
    1. Multiple related components are created together
    2. They primarily reference each other (high internal coupling)
    3. They have minimal external references (isolation)
    4. They were created in a short timeframe (temporal clustering)
    5. They often implement fictional or unnecessary abstractions
    """
    
    name = "hallucination_cascade"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        
        # Configuration parameters
        self.min_cluster_size = self.config.get('min_cluster_size', 3)
        self.max_cluster_size = self.config.get('max_cluster_size', 20)
        self.isolation_threshold = self.config.get('isolation_threshold', 0.6)
        self.temporal_threshold_hours = self.config.get('temporal_threshold_hours', 48)
        self.confidence_threshold = self.config.get('confidence_threshold', 0.7)
        self.density_threshold = self.config.get('density_threshold', 0.3)
        
        # Analysis parameters
        self.analyze_git_history = self.config.get('analyze_git_history', True)
        self.git_lookback_days = self.config.get('git_lookback_days', 30)
        self.include_functions = self.config.get('include_functions', True)
        self.include_classes = self.config.get('include_classes', True)
        
        # State
        self.dependency_graph: Optional[nx.DiGraph] = None
        self.components: Dict[str, ComponentInfo] = {}
        self.git_available = False
        
        # Check if networkx is available
        if nx is None:
            logger.warning("NetworkX not available - using fallback graph implementation")
        
        # Check git availability
        self._check_git_availability()
        
        logger.debug(f"HallucinationCascadeDetector initialized: "
                    f"min_cluster={self.min_cluster_size}, "
                    f"isolation_threshold={self.isolation_threshold}")
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run hallucination cascade detection.
        
        Args:
            ctx: Analysis context containing AST index and configuration
            
        Returns:
            List of Issue objects representing detected hallucination cascades
        """
        issues = []
        
        try:
            logger.info(f"Running hallucination cascade detection on {len(ctx.ast_index)} files")
            
            # Step 1: Extract components and build dependency graph
            logger.debug("Extracting components and building dependency graph")
            self._extract_components(ctx)
            self._build_dependency_graph()
            
            if len(self.components) < self.min_cluster_size:
                logger.debug(f"Not enough components ({len(self.components)}) for cascade detection")
                return issues
            
            # Step 2: Analyze git history for temporal patterns
            if self.analyze_git_history and self.git_available:
                logger.debug("Analyzing git history for temporal patterns")
                self._analyze_git_history(ctx)
            
            # Step 3: Detect clusters of interdependent components
            logger.debug("Detecting component clusters")
            clusters = self._detect_component_clusters()
            
            # Step 4: Analyze each cluster for hallucination indicators
            logger.debug(f"Analyzing {len(clusters)} clusters for hallucination patterns")
            hallucination_clusters = []
            
            for cluster in clusters:
                self._analyze_cluster_isolation(cluster)
                self._analyze_cluster_temporal_patterns(cluster)
                self._calculate_cluster_confidence(cluster)
                
                if cluster.is_likely_hallucination(self.confidence_threshold):
                    hallucination_clusters.append(cluster)
            
            # Step 5: Generate issues for detected hallucination cascades
            for cluster in hallucination_clusters:
                issue = self._create_hallucination_issue(cluster, ctx)
                if issue:
                    issues.append(issue)
            
            logger.info(f"Hallucination cascade detection complete: "
                       f"analyzed {len(self.components)} components, "
                       f"found {len(clusters)} clusters, "
                       f"detected {len(hallucination_clusters)} hallucination cascades")
            
        except Exception as e:
            logger.error(f"Error in hallucination cascade detection: {e}", exc_info=True)
        
        return issues
    
    def _extract_components(self, ctx: AnalysisContext) -> None:
        """Extract all components (classes, functions) from the codebase."""
        self.components.clear()
        
        for file_path, tree in ctx.ast_index.items():
            if ctx.is_excluded(file_path):
                continue
            
            try:
                # Extract classes
                if self.include_classes:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            component = self._create_component_info(node, file_path, 'class')
                            self.components[component.name] = component
                
                # Extract functions
                if self.include_functions:
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            # Skip methods (functions inside classes)
                            if not self._is_method(node, tree):
                                component = self._create_component_info(node, file_path, 'function')
                                self.components[component.name] = component
            
            except Exception as e:
                logger.warning(f"Error extracting components from {file_path}: {e}")
    
    def _create_component_info(
        self, 
        node: ast.AST, 
        file_path: str, 
        component_type: str
    ) -> ComponentInfo:
        """Create ComponentInfo from an AST node."""
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            name = f"{Path(file_path).stem}.{node.name}"
            
            component = ComponentInfo(
                name=name,
                file_path=file_path,
                component_type=component_type,
                line_number=safe_get_lineno(node),
                end_line_number=safe_get_end_lineno(node, safe_get_lineno(node))
            )
            
            # Analyze the component for additional information
            self._analyze_component_details(component, node)
            
            return component
        
        raise ValueError(f"Unsupported node type for component creation: {type(node)}")
    
    def _analyze_component_details(self, component: ComponentInfo, node: ast.AST) -> None:
        """Analyze component details from AST node."""
        try:
            # Check if it's abstract or interface-like
            if isinstance(node, ast.ClassDef):
                # Look for abstract methods or interface patterns
                methods = [n for n in node.body if isinstance(n, ast.FunctionDef)]
                abstract_methods = 0
                
                for method in methods:
                    # Check for abstract method indicators
                    if self._is_abstract_method(method):
                        abstract_methods += 1
                
                component.is_abstract = abstract_methods > 0
                component.is_interface = abstract_methods == len(methods) and len(methods) > 0
            
            # Calculate complexity score (simplified)
            complexity = 0
            for child in ast.walk(node):
                if isinstance(child, (ast.If, ast.For, ast.While, ast.Try)):
                    complexity += 1
                elif isinstance(child, ast.FunctionDef):
                    complexity += 2
            
            component.complexity_score = complexity
            
            # Check for implementation
            has_real_implementation = False
            for child in ast.walk(node):
                if isinstance(child, ast.Return) and child.value:
                    has_real_implementation = True
                    break
                elif isinstance(child, ast.Assign):
                    has_real_implementation = True
                    break
            
            component.has_implementation = has_real_implementation
        
        except Exception as e:
            logger.debug(f"Error analyzing component details: {e}")
    
    def _is_method(self, func_node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method inside a class."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if func_node in node.body:
                    return True
        return False
    
    def _is_abstract_method(self, method: ast.FunctionDef) -> bool:
        """Check if a method appears to be abstract."""
        # Look for common abstract method patterns
        for stmt in method.body:
            if isinstance(stmt, ast.Raise):
                if isinstance(stmt.exc, ast.Call):
                    if isinstance(stmt.exc.func, ast.Name):
                        if stmt.exc.func.id == "NotImplementedError":
                            return True
            elif isinstance(stmt, ast.Expr):
                if isinstance(stmt.value, ast.Constant):
                    if stmt.value.value == "..." or stmt.value.value is ...:
                        return True
            elif isinstance(stmt, ast.Pass):
                # A method with only pass might be abstract
                if len(method.body) == 1:
                    return True
        
        return False
    
    def _build_dependency_graph(self) -> None:
        """Build dependency graph between components."""
        if nx is not None:
            self.dependency_graph = nx.DiGraph()
        else:
            # Fallback: we'll track dependencies manually
            pass
        
        # Add all components as nodes
        for component_name, component in self.components.items():
            if nx is not None:
                self.dependency_graph.add_node(component_name, component=component)
        
        # Analyze dependencies between components
        for component_name, component in self.components.items():
            dependencies = self._find_component_dependencies(component)
            
            for dep_name in dependencies:
                if dep_name in self.components:
                    # Internal dependency
                    component.dependencies.add(dep_name)
                    self.components[dep_name].dependents.add(component_name)
                    component.internal_references += 1
                    
                    if nx is not None:
                        self.dependency_graph.add_edge(component_name, dep_name)
                else:
                    # External dependency
                    component.external_references += 1
    
    def _find_component_dependencies(self, component: ComponentInfo) -> Set[str]:
        """Find dependencies for a component by analyzing its source code."""
        dependencies = set()
        
        try:
            with open(component.file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=component.file_path)
            
            # Find the specific node for this component
            target_node = None
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if node.name in component.name and safe_get_lineno(node) == component.line_number:
                        target_node = node
                        break
            
            if target_node:
                # Look for references to other components
                for node in ast.walk(target_node):
                    if isinstance(node, ast.Name):
                        # Check if this name refers to another component
                        for comp_name in self.components:
                            if node.id in comp_name:
                                dependencies.add(comp_name)
                    elif isinstance(node, ast.Attribute):
                        # Handle attribute access like module.Class
                        if isinstance(node.value, ast.Name):
                            full_name = f"{node.value.id}.{node.attr}"
                            for comp_name in self.components:
                                if full_name in comp_name:
                                    dependencies.add(comp_name)
        
        except Exception as e:
            logger.debug(f"Error finding dependencies for {component.name}: {e}")
        
        return dependencies
    
    def _check_git_availability(self) -> None:
        """Check if git is available for history analysis."""
        try:
            result = subprocess.run(['git', '--version'], 
                                  capture_output=True, text=True)
            self.git_available = result.returncode == 0
        except FileNotFoundError:
            self.git_available = False
        
        logger.debug(f"Git available: {self.git_available}")
    
    def _analyze_git_history(self, ctx: AnalysisContext) -> None:
        """Analyze git history to determine component creation patterns."""
        if not self.git_available:
            return
        
        try:
            # Get recent commits
            since_date = (datetime.now() - timedelta(days=self.git_lookback_days)).strftime('%Y-%m-%d')
            
            cmd = [
                'git', 'log', '--since', since_date, '--name-only', 
                '--pretty=format:%H|%ai|%s', '--'
            ] + [comp.file_path for comp in self.components.values()]
            
            result = subprocess.run(cmd, capture_output=True, text=True, 
                                  cwd=str(ctx.root_dir))
            
            if result.returncode == 0:
                self._parse_git_history(result.stdout)
            else:
                logger.warning("Git history analysis failed")
        
        except Exception as e:
            logger.warning(f"Error analyzing git history: {e}")
    
    def _parse_git_history(self, git_output: str) -> None:
        """Parse git log output to extract creation timestamps."""
        lines = git_output.strip().split('\n')
        current_commit = None
        current_date = None
        
        for line in lines:
            if '|' in line and len(line.split('|')) >= 2:
                # Commit line: hash|date|message
                parts = line.split('|', 2)
                current_commit = parts[0]
                try:
                    # Parse git date format
                    date_str = parts[1].strip()
                    current_date = datetime.fromisoformat(date_str.replace(' +', '+'))
                except Exception as e:
                    logger.debug(f"Error parsing date {parts[1]}: {e}")
                    current_date = None
            elif line.strip() and current_date:
                # File line
                file_path = line.strip()
                # Update component creation dates
                for component in self.components.values():
                    if file_path == component.file_path:
                        if component.creation_date is None or current_date < component.creation_date:
                            component.creation_date = current_date
                            component.commit_hash = current_commit
    
    def _detect_component_clusters(self) -> List[HallucinationCluster]:
        """Detect clusters of strongly interconnected components."""
        clusters = []
        
        if nx is not None:
            # Use NetworkX for sophisticated clustering
            clusters.extend(self._detect_clusters_networkx())
        else:
            # Fallback clustering algorithm
            clusters.extend(self._detect_clusters_fallback())
        
        return clusters
    
    def _detect_clusters_networkx(self) -> List[HallucinationCluster]:
        """Detect clusters using NetworkX graph algorithms."""
        clusters = []
        
        if not self.dependency_graph:
            return clusters
        
        try:
            # Find strongly connected components
            strong_components = list(nx.strongly_connected_components(self.dependency_graph))
            
            # Also find weakly connected components for broader analysis
            weak_components = list(nx.weakly_connected_components(self.dependency_graph))
            
            # Process each component group
            for component_set in strong_components + weak_components:
                if len(component_set) >= self.min_cluster_size:
                    cluster = self._create_cluster_from_components(component_set)
                    if cluster:
                        clusters.append(cluster)
            
            # Additional clustering based on community detection
            try:
                # Convert to undirected graph for community detection
                undirected = self.dependency_graph.to_undirected()
                communities = nx.community.greedy_modularity_communities(undirected)
                
                for community in communities:
                    if len(community) >= self.min_cluster_size:
                        cluster = self._create_cluster_from_components(community)
                        if cluster and not self._is_duplicate_cluster(cluster, clusters):
                            clusters.append(cluster)
            
            except Exception as e:
                logger.debug(f"Community detection failed: {e}")
        
        except Exception as e:
            logger.warning(f"Error in NetworkX clustering: {e}")
        
        return clusters
    
    def _detect_clusters_fallback(self) -> List[HallucinationCluster]:
        """Fallback clustering algorithm when NetworkX is not available."""
        clusters = []
        processed = set()
        
        for component_name, component in self.components.items():
            if component_name in processed:
                continue
            
            # Start a new cluster with this component
            cluster_components = {component_name}
            to_process = [component_name]
            
            # Expand cluster by following dependencies
            while to_process:
                current = to_process.pop()
                current_comp = self.components[current]
                
                # Add all dependencies and dependents
                for dep in current_comp.dependencies:
                    if dep not in cluster_components and dep not in processed:
                        cluster_components.add(dep)
                        to_process.append(dep)
                
                for dep in current_comp.dependents:
                    if dep not in cluster_components and dep not in processed:
                        cluster_components.add(dep)
                        to_process.append(dep)
            
            # Create cluster if it meets size requirements
            if len(cluster_components) >= self.min_cluster_size:
                cluster = self._create_cluster_from_components(cluster_components)
                if cluster:
                    clusters.append(cluster)
            
            # Mark all components as processed
            processed.update(cluster_components)
        
        return clusters
    
    def _create_cluster_from_components(self, component_names: Set[str]) -> Optional[HallucinationCluster]:
        """Create a HallucinationCluster from a set of component names."""
        if len(component_names) < self.min_cluster_size:
            return None
        
        cluster_id = hashlib.md5('_'.join(sorted(component_names)).encode()).hexdigest()[:8]
        cluster = HallucinationCluster(cluster_id=cluster_id)
        
        # Add components to cluster
        for comp_name in component_names:
            if comp_name in self.components:
                cluster.add_component(self.components[comp_name])
        
        # Calculate internal and external edges
        internal_edges = 0
        external_edges = 0
        
        for component in cluster.components:
            for dep in component.dependencies:
                if dep in component_names:
                    internal_edges += 1
                else:
                    external_edges += 1
        
        cluster.internal_edges = internal_edges
        cluster.external_edges = external_edges
        
        return cluster
    
    def _is_duplicate_cluster(self, new_cluster: HallucinationCluster, existing_clusters: List[HallucinationCluster]) -> bool:
        """Check if a cluster is a duplicate of existing clusters."""
        new_names = {comp.name for comp in new_cluster.components}
        
        for existing in existing_clusters:
            existing_names = {comp.name for comp in existing.components}
            
            # Consider it duplicate if 80% overlap
            overlap = len(new_names & existing_names)
            min_size = min(len(new_names), len(existing_names))
            
            if overlap / min_size >= 0.8:
                return True
        
        return False
    
    def _analyze_cluster_isolation(self, cluster: HallucinationCluster) -> None:
        """Analyze how isolated a cluster is from external code."""
        cluster.calculate_metrics()
        
        # Additional isolation analysis
        total_components = len(cluster.components)
        isolated_components = 0
        
        for component in cluster.components:
            if component.get_isolation_ratio() > self.isolation_threshold:
                isolated_components += 1
        
        # Update isolation score based on component-level isolation
        component_isolation_ratio = isolated_components / total_components
        cluster.isolation_score = (cluster.isolation_score + component_isolation_ratio) / 2
        
        # Add evidence for isolation
        cluster.supporting_evidence['isolation_metrics'] = {
            'cluster_isolation_score': cluster.isolation_score,
            'external_reference_ratio': cluster.external_reference_ratio,
            'isolated_components': isolated_components,
            'total_components': total_components,
            'component_isolation_ratio': component_isolation_ratio
        }
    
    def _analyze_cluster_temporal_patterns(self, cluster: HallucinationCluster) -> None:
        """Analyze temporal creation patterns of cluster components."""
        creation_dates = [comp.creation_date for comp in cluster.components if comp.creation_date]
        
        if not creation_dates:
            return
        
        cluster.calculate_metrics()  # This updates temporal metrics
        
        # Additional temporal analysis
        temporal_indicators = []
        
        if cluster.creation_burst:
            temporal_indicators.append("Components created within 24-hour burst")
        
        if cluster.temporal_consistency > 0.7:
            temporal_indicators.append("Highly consistent creation timing")
        
        if cluster.creation_timespan and cluster.creation_timespan <= timedelta(hours=self.temporal_threshold_hours):
            temporal_indicators.append(f"All components created within {self.temporal_threshold_hours} hours")
        
        # Analyze commit patterns
        commit_hashes = [comp.commit_hash for comp in cluster.components if comp.commit_hash]
        unique_commits = len(set(commit_hashes))
        
        if unique_commits == 1:
            temporal_indicators.append("All components created in single commit")
        elif unique_commits <= len(cluster.components) // 2:
            temporal_indicators.append("Components created in few commits")
        
        cluster.supporting_evidence['temporal_analysis'] = {
            'creation_burst': cluster.creation_burst,
            'temporal_consistency': cluster.temporal_consistency,
            'creation_timespan_hours': cluster.creation_timespan.total_seconds() / 3600 if cluster.creation_timespan else None,
            'unique_commits': unique_commits,
            'temporal_indicators': temporal_indicators
        }
    
    def _calculate_cluster_confidence(self, cluster: HallucinationCluster) -> None:
        """Calculate confidence score for hallucination cascade detection."""
        confidence_factors = {}
        
        # Factor 1: Isolation score (0-1, higher = more isolated)
        isolation_weight = 0.3
        confidence_factors['isolation'] = cluster.isolation_score * isolation_weight
        
        # Factor 2: Temporal clustering (0-1, higher = more clustered in time)
        temporal_weight = 0.25
        temporal_score = 0.0
        
        if cluster.creation_burst:
            temporal_score += 0.4
        if cluster.temporal_consistency > 0.7:
            temporal_score += 0.3
        if cluster.creation_timespan and cluster.creation_timespan <= timedelta(hours=self.temporal_threshold_hours):
            temporal_score += 0.3
        
        confidence_factors['temporal'] = min(1.0, temporal_score) * temporal_weight
        
        # Factor 3: Internal connectivity vs external connectivity
        connectivity_weight = 0.2
        total_edges = cluster.internal_edges + cluster.external_edges
        if total_edges > 0:
            internal_ratio = cluster.internal_edges / total_edges
            confidence_factors['connectivity'] = internal_ratio * connectivity_weight
        else:
            confidence_factors['connectivity'] = 0.0
        
        # Factor 4: Cluster size (moderate size clusters are more suspicious)
        size_weight = 0.1
        size_score = 0.0
        cluster_size = len(cluster.components)
        
        if self.min_cluster_size <= cluster_size <= 8:
            size_score = 1.0  # Ideal suspicious size
        elif cluster_size <= 15:
            size_score = 0.7
        else:
            size_score = 0.3  # Very large clusters less likely to be pure hallucination
        
        confidence_factors['size'] = size_score * size_weight
        
        # Factor 5: Abstract/interface ratio (high abstraction can indicate over-engineering)
        abstraction_weight = 0.15
        abstract_components = sum(1 for comp in cluster.components if comp.is_abstract or comp.is_interface)
        abstraction_ratio = abstract_components / len(cluster.components)
        confidence_factors['abstraction'] = min(1.0, abstraction_ratio * 2) * abstraction_weight
        
        # Calculate final confidence score
        cluster.confidence_score = sum(confidence_factors.values())
        
        # Generate hallucination indicators
        indicators = []
        
        if cluster.isolation_score > 0.7:
            indicators.append(f"High isolation: {cluster.isolation_score:.2f}")
        
        if cluster.creation_burst:
            indicators.append("Components created in rapid burst")
        
        if cluster.temporal_consistency > 0.8:
            indicators.append("Suspiciously consistent creation timing")
        
        if cluster.external_reference_ratio < 0.2:
            indicators.append(f"Very low external references: {cluster.external_reference_ratio:.2f}")
        
        if abstraction_ratio > 0.5:
            indicators.append(f"High abstraction ratio: {abstraction_ratio:.2f}")
        
        if cluster.cluster_density > 0.7:
            indicators.append("High internal connectivity")
        
        cluster.hallucination_indicators = indicators
        
        # Store confidence breakdown
        cluster.supporting_evidence['confidence_breakdown'] = confidence_factors
        
        logger.debug(f"Cluster {cluster.cluster_id} confidence: {cluster.confidence_score:.3f}, "
                    f"indicators: {len(indicators)}")
    
    def _create_hallucination_issue(self, cluster: HallucinationCluster, ctx: AnalysisContext) -> Optional[Issue]:
        """Create an Issue object for a detected hallucination cascade."""
        try:
            primary_component = cluster.components[0]
            
            # Build detailed description
            description = (
                f"Hallucination cascade detected: {len(cluster.components)} interdependent components "
                f"with {cluster.confidence_score:.1%} confidence"
            )
            
            # Collect component details
            component_names = [comp.name for comp in cluster.components]
            component_files = list(set(comp.file_path for comp in cluster.components))
            
            # Build evidence dictionary
            evidence = {
                'cluster_id': cluster.cluster_id,
                'component_count': len(cluster.components),
                'component_names': component_names,
                'affected_files': component_files,
                'confidence_score': cluster.confidence_score,
                'isolation_score': cluster.isolation_score,
                'external_reference_ratio': cluster.external_reference_ratio,
                'internal_edges': cluster.internal_edges,
                'external_edges': cluster.external_edges,
                'cluster_density': cluster.cluster_density,
                'hallucination_indicators': cluster.hallucination_indicators,
                'creation_burst': cluster.creation_burst,
                'temporal_consistency': cluster.temporal_consistency
            }
            
            # Add temporal information if available
            if cluster.creation_timespan:
                evidence['creation_timespan_hours'] = cluster.creation_timespan.total_seconds() / 3600
            
            # Add supporting evidence
            evidence.update(cluster.supporting_evidence)
            
            # Generate suggestions
            suggestions = [
                f"Review the necessity of the {len(cluster.components)}-component subsystem",
                "Consider whether existing functionality could be reused instead",
                "Evaluate if the abstraction level is appropriate for the use case",
                "Check if components have sufficient external integration"
            ]
            
            # Add specific suggestions based on indicators
            if cluster.isolation_score > 0.8:
                suggestions.append("Components are highly isolated - consider adding external integration points")
            
            if cluster.creation_burst:
                suggestions.append("All components were created rapidly - review for over-engineering")
            
            if any("abstraction" in indicator for indicator in cluster.hallucination_indicators):
                suggestions.append("High abstraction ratio detected - consider simplifying the design")
            
            # Determine severity based on confidence
            severity = 3  # High
            if cluster.confidence_score > 0.85:
                severity = 4  # Critical
            elif cluster.confidence_score < 0.7:
                severity = 2  # Medium
            
            issue = Issue(
                kind="hallucination_cascade",
                message=description,
                severity=severity,
                file=primary_component.file_path,
                line=primary_component.line_number,
                symbol=primary_component.name,
                confidence=cluster.confidence_score,
                evidence=evidence,
                suggestions=suggestions
            )
            
            return issue
        
        except Exception as e:
            logger.error(f"Error creating hallucination issue: {e}", exc_info=True)
            return None


# Alias for backward compatibility
HallucinationCascadeAnalyzer = HallucinationCascadeDetector
