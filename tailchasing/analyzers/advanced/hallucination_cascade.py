"""
Hallucination cascade analyzer for detecting fictional subsystems.
"""

import ast
import networkx as nx
from datetime import datetime
from typing import List, Dict, Optional
from .base_advanced import PatternDetectionAnalyzer
from ..base import Analyzer
from ...core.issues import Issue


class HallucinationCascadeAnalyzer(PatternDetectionAnalyzer, Analyzer):
    """Detect when LLM creates entire fictional subsystems."""
    
    name = "hallucination_cascade"
    
    def _initialize_specific_config(self):
        """Initialize hallucination cascade specific configuration."""
        super()._initialize_specific_config()
        self.set_config('min_cascade_size', 3)  # Minimum components for a cascade
        self.set_config('max_time_span_days', 2)  # Maximum time span for related creation
        self.set_threshold('external_ref', 0.2)  # Maximum external reference ratio
    
    @property
    def min_cascade_size(self):
        """Get minimum cascade size configuration."""
        return self.get_config('min_cascade_size', 3)
    
    @property
    def max_time_span_days(self):
        """Get maximum time span days configuration."""
        return self.get_config('max_time_span_days', 2)
    
    @property
    def external_ref_threshold(self):
        """Get external reference threshold."""
        return self.get_threshold('external_ref', 0.2)
    
    def run(self, ctx) -> List[Issue]:
        """Run hallucination cascade analysis."""
        # Build dependency graph
        dep_graph = nx.DiGraph()
        creation_times = {}
        
        # Extract classes and their dependencies
        for filepath, tree in ctx.ast_index.items():
            self._extract_dependencies(tree, filepath, dep_graph, creation_times, ctx)
        
        # Find suspicious clusters
        return self._find_cascades(dep_graph, creation_times)
    
    def _extract_dependencies(self, tree: ast.AST, filepath: str, 
                            dep_graph: nx.DiGraph, creation_times: Dict, ctx) -> None:
        """Extract class dependencies from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                full_name = f"{filepath}:{class_name}"
                
                dep_graph.add_node(full_name, 
                                 file=filepath, 
                                 line=node.lineno,
                                 class_name=class_name)
                
                # Find dependencies within this class
                for subnode in ast.walk(node):
                    if isinstance(subnode, ast.Name) and subnode.id != class_name:
                        # Check if this name refers to another class in our codebase
                        if self._is_internal_class(subnode.id, ctx):
                            dep_full_name = self._find_class_location(subnode.id, ctx)
                            if dep_full_name and dep_full_name != full_name:
                                dep_graph.add_edge(full_name, dep_full_name)
                
                # Mock creation time (in real implementation, use git history)
                creation_times[full_name] = datetime.now()
    
    def _is_internal_class(self, class_name: str, ctx) -> bool:
        """Check if a class name refers to an internal class."""
        # Simple check - look for the class in our symbol table
        return class_name in ctx.symbol_table.classes
    
    def _find_class_location(self, class_name: str, ctx) -> Optional[str]:
        """Find the full location of a class."""
        if class_name in ctx.symbol_table.classes:
            entries = ctx.symbol_table.classes[class_name]
            if entries:
                entry = entries[0]  # Take first occurrence
                return f"{entry['file']}:{class_name}"
        return None
    
    def _find_cascades(self, dep_graph: nx.DiGraph, creation_times: Dict) -> List[Issue]:
        """Find hallucination cascades in the dependency graph."""
        issues = []
        
        # Find weakly connected components
        for component in nx.weakly_connected_components(dep_graph):
            if len(component) >= self.min_cascade_size:
                # Check if this is a suspicious cascade
                if self._is_suspicious_cascade(component, dep_graph, creation_times):
                    issues.append(self._create_cascade_issue(component, dep_graph))
        
        return issues
    
    def _is_suspicious_cascade(self, component: set, dep_graph: nx.DiGraph, 
                             creation_times: Dict) -> bool:
        """Determine if a component is a suspicious hallucination cascade."""
        # Check time span (all created recently and together)
        times = [creation_times.get(node, datetime.now()) for node in component]
        if times:
            time_span = (max(times) - min(times)).days
            if time_span > self.max_time_span_days:
                return False
        
        # Check external references
        external_refs = 0
        for node in component:
            for pred in dep_graph.predecessors(node):
                if pred not in component:
                    external_refs += 1
        
        external_ref_ratio = external_refs / len(component) if component else 0
        return external_ref_ratio < self.external_ref_threshold
    
    def _create_cascade_issue(self, component: set, dep_graph: nx.DiGraph) -> Issue:
        """Create an issue for a detected hallucination cascade."""
        # Get the first node for location info
        first_node = next(iter(component))
        node_data = dep_graph.nodes[first_node]
        
        class_names = [dep_graph.nodes[node]['class_name'] for node in component]
        
        return Issue(
            kind="hallucination_cascade",
            message=f"Detected {len(component)} interdependent classes created together with minimal external references",
            severity=4,
            file=node_data['file'],
            line=node_data['line'],
            evidence={
                "components": class_names,
                "component_count": len(component),
                "files": list(set(dep_graph.nodes[node]['file'] for node in component))
            },
            suggestions=[
                f"Review if {', '.join(class_names[:3])}{'...' if len(class_names) > 3 else ''} are actually needed",
                "Check if existing functionality could be used instead",
                "Consider consolidating related classes",
                "Verify that this subsystem serves a real purpose"
            ]
        )
