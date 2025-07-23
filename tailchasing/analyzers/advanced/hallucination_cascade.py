"""
Hallucination Cascade Detection

Detects when LLMs create entire fictional subsystems to satisfy errors.
Pattern: Error → Create ClassA → Error in ClassA → Create ClassB → etc.
"""

import ast
from typing import Dict, List, Set, Optional, Tuple
from datetime import datetime
from collections import defaultdict
import networkx as nx

from ..base import AnalysisContext
from ...core.issues import Issue


class HallucinationCascadeAnalyzer:
    """Detects when LLMs create entire fictional subsystems."""
    
    name = "hallucination_cascade"
    
    def __init__(self):
        self.creation_time_threshold_days = 2
        self.min_cluster_size = 3
        self.external_ref_threshold = 0.2
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Analyze codebase for hallucination cascades."""
        issues = []
        
        # Build dependency graph
        dep_graph = self._build_dependency_graph(ctx)
        
        # Get creation times (would need git integration in real implementation)
        creation_times = self._estimate_creation_times(ctx)
        
        # Find suspicious clusters
        for component in nx.weakly_connected_components(dep_graph):
            if len(component) >= self.min_cluster_size:
                cascade_issue = self._analyze_component(
                    component, dep_graph, creation_times, ctx
                )
                if cascade_issue:
                    issues.append(cascade_issue)
        
        return issues
    
    def _build_dependency_graph(self, ctx: AnalysisContext) -> nx.DiGraph:
        """Build a dependency graph of classes and functions."""
        graph = nx.DiGraph()
        
        # First pass: add all nodes
        for filepath, tree in ctx.ast_index.items():
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    graph.add_node(
                        node.name,
                        type='class',
                        file=filepath,
                        line=node.lineno
                    )
                elif isinstance(node, ast.FunctionDef):
                    # Only add module-level functions
                    if isinstance(node, ast.FunctionDef) and not any(
                        isinstance(parent, ast.ClassDef) 
                        for parent in ast.walk(tree)
                        if parent != node and hasattr(parent, 'body') and 
                        isinstance(parent.body, list) and node in parent.body
                    ):
                        graph.add_node(
                            node.name,
                            type='function',
                            file=filepath,
                            line=node.lineno
                        )
        
        # Second pass: add edges (dependencies)
        for filepath, tree in ctx.ast_index.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    deps = self._find_dependencies(node)
                    for dep in deps:
                        if dep in graph:
                            graph.add_edge(node.name, dep)
        
        return graph
    
    def _find_dependencies(self, node: ast.AST) -> Set[str]:
        """Find all dependencies of a class or function."""
        deps = set()
        
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Name) and isinstance(subnode.ctx, ast.Load):
                deps.add(subnode.id)
            elif isinstance(subnode, ast.Attribute):
                # Handle chained attributes
                base = subnode
                while isinstance(base, ast.Attribute):
                    base = base.value
                if isinstance(base, ast.Name):
                    deps.add(base.id)
        
        return deps
    
    def _estimate_creation_times(self, ctx: AnalysisContext) -> Dict[str, datetime]:
        """Estimate creation times for entities."""
        # In a real implementation, this would use git history
        # For now, return mock data showing recent creation
        times = {}
        base_time = datetime.now()
        
        for filepath, tree in ctx.ast_index.items():
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    # Mock: assume all created recently for demo
                    times[node.name] = base_time
        
        return times
    
    def _analyze_component(self,
                          component: Set[str],
                          graph: nx.DiGraph,
                          creation_times: Dict[str, datetime],
                          ctx: AnalysisContext) -> Optional[Issue]:
        """Analyze a connected component for hallucination patterns."""
        
        # Check creation time clustering
        times = [creation_times.get(n, datetime.now()) for n in component]
        if not times:
            return None
            
        time_span = (max(times) - min(times)).days
        
        if time_span > self.creation_time_threshold_days:
            return None
        
        # Check isolation from rest of codebase
        external_refs = 0
        internal_refs = 0
        
        for node in component:
            for pred in graph.predecessors(node):
                if pred in component:
                    internal_refs += 1
                else:
                    external_refs += 1
        
        total_refs = external_refs + internal_refs
        if total_refs == 0:
            return None
            
        external_ratio = external_refs / total_refs
        
        if external_ratio < self.external_ref_threshold:
            # Found a suspicious cluster
            files = set()
            locations = []
            
            for node in component:
                node_data = graph.nodes[node]
                files.add(node_data['file'])
                locations.append((node_data['file'], node_data['line']))
            
            return Issue(
                kind="hallucination_cascade",
                message=f"Detected {len(component)} interdependent entities created together "
                       f"with minimal external references ({external_ratio:.0%}). "
                       f"Likely hallucinated subsystem.",
                severity=4,
                file=list(files)[0] if len(files) == 1 else None,
                evidence={
                    'entities': list(component),
                    'file_count': len(files),
                    'external_ref_ratio': external_ratio,
                    'time_span_days': time_span,
                    'locations': locations[:5]  # Limit for readability
                },
                suggestions=[
                    f"Review if {', '.join(list(component)[:3])}{'...' if len(component) > 3 else ''} "
                    f"are actually needed",
                    "Check if existing functionality could be reused instead",
                    "Consider if this represents a misunderstanding of requirements"
                ]
            )
        
        return None
