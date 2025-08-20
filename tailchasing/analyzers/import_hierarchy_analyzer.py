"""
Import Hierarchy Analyzer - Suggests proper import structure and detects import issues.

This analyzer examines import patterns to identify problematic structures and
suggest improvements for better module organization.
"""

import ast
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass

from ..core.issues import Issue
from .base import BaseAnalyzer


@dataclass
class ImportLayer:
    """Represents a layer in the import hierarchy."""
    
    layer_name: str
    modules: Set[str]
    imports_from_layers: Dict[str, Set[str]]  # layer -> modules imported
    exports_to_layers: Dict[str, Set[str]]  # layer -> modules exported


@dataclass 
class ImportViolation:
    """Represents a violation of proper import hierarchy."""
    
    from_module: str
    to_module: str
    from_layer: str
    to_layer: str
    violation_type: str  # 'backward', 'skip_layer', 'circular', 'cross_layer'
    severity: str


class ImportHierarchyAnalyzer(BaseAnalyzer):
    """
    Analyzes import hierarchy and suggests proper layered architecture.
    
    Detects:
    1. Backward dependencies (lower layers importing from higher)
    2. Layer skipping (importing from non-adjacent layers)
    3. Cross-layer imports (horizontal dependencies)
    4. Import clustering issues
    5. God modules (imported by too many others)
    6. Orphan modules (no imports or exports)
    """
    
    # Define standard layer hierarchy (lowest to highest)
    STANDARD_LAYERS = [
        'utils',      # Utilities, helpers
        'core',       # Core domain logic
        'models',     # Data models
        'services',   # Business logic services  
        'api',        # API layer
        'ui',         # UI/presentation layer
    ]
    
    def __init__(self):
        super().__init__()
        self.import_graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
        self.module_layers: Dict[str, str] = {}
        self.layer_hierarchy: Dict[str, int] = {}
        self.violations: List[ImportViolation] = []
        
    def analyze(self, context) -> List[Issue]:
        """Analyze import hierarchy and structure."""
        issues = []
        
        # Build import graph
        self._build_import_graph(context)
        
        # Detect layers from module paths
        self._detect_layers(context)
        
        # Analyze layer violations
        violations = self._analyze_layer_violations()
        
        # Detect god modules
        god_modules = self._detect_god_modules()
        
        # Detect orphan modules
        orphan_modules = self._detect_orphan_modules()
        
        # Detect import cycles at module level
        module_cycles = self._detect_module_cycles()
        
        # Generate issues
        issues.extend(self._generate_violation_issues(violations))
        issues.extend(self._generate_god_module_issues(god_modules))
        issues.extend(self._generate_orphan_module_issues(orphan_modules))
        issues.extend(self._generate_cycle_issues(module_cycles))
        
        # Add hierarchy suggestion if violations found
        if violations:
            issues.append(self._generate_hierarchy_suggestion())
        
        return issues
    
    def _build_import_graph(self, context):
        """Build complete import graph from AST."""
        for file_path, ast_tree in context.ast_index.items():
            module_name = self._path_to_module(file_path)
            
            for node in ast.walk(ast_tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name
                        self.import_graph[module_name].add(imported)
                        self.reverse_graph[imported].add(module_name)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imported = node.module
                        self.import_graph[module_name].add(imported)
                        self.reverse_graph[imported].add(module_name)
    
    def _path_to_module(self, file_path: str) -> str:
        """Convert file path to module name."""
        path = Path(file_path)
        parts = []
        
        # Walk up to find package root
        for part in path.parts[:-1]:  # Exclude filename
            if part not in ('.', '..', ''):
                parts.append(part)
        
        # Add file without extension
        if path.stem != '__init__':
            parts.append(path.stem)
            
        return '.'.join(parts)
    
    def _detect_layers(self, context):
        """Detect layer structure from module organization."""
        # Build layer hierarchy with indices
        for i, layer in enumerate(self.STANDARD_LAYERS):
            self.layer_hierarchy[layer] = i
        
        # Assign modules to layers based on path components
        for file_path in context.ast_index.keys():
            module_name = self._path_to_module(file_path)
            layer = self._detect_module_layer(module_name)
            if layer:
                self.module_layers[module_name] = layer
    
    def _detect_module_layer(self, module_name: str) -> Optional[str]:
        """Detect which layer a module belongs to."""
        parts = module_name.split('.')
        
        # Check each part for layer keywords
        for part in parts:
            part_lower = part.lower()
            
            # Direct layer matches
            if part_lower in self.STANDARD_LAYERS:
                return part_lower
            
            # Common variations
            layer_mappings = {
                'util': 'utils',
                'helper': 'utils',
                'common': 'utils',
                'domain': 'core',
                'entity': 'models',
                'schema': 'models',
                'service': 'services',
                'controller': 'api',
                'route': 'api',
                'router': 'api',
                'view': 'ui',
                'component': 'ui',
            }
            
            for keyword, layer in layer_mappings.items():
                if keyword in part_lower:
                    return layer
        
        return None
    
    def _analyze_layer_violations(self) -> List[ImportViolation]:
        """Analyze violations of layer hierarchy."""
        violations = []
        
        for from_module, imports in self.import_graph.items():
            from_layer = self.module_layers.get(from_module)
            if not from_layer:
                continue
                
            from_level = self.layer_hierarchy.get(from_layer, -1)
            
            for to_module in imports:
                to_layer = self.module_layers.get(to_module)
                if not to_layer:
                    continue
                    
                to_level = self.layer_hierarchy.get(to_layer, -1)
                
                # Check for violations
                if from_level < to_level:
                    # Lower layer importing from higher - backward dependency
                    violations.append(ImportViolation(
                        from_module=from_module,
                        to_module=to_module,
                        from_layer=from_layer,
                        to_layer=to_layer,
                        violation_type='backward',
                        severity='HIGH'
                    ))
                elif from_level == to_level and from_layer != to_layer:
                    # Same level but different layer - cross-layer
                    violations.append(ImportViolation(
                        from_module=from_module,
                        to_module=to_module,
                        from_layer=from_layer,
                        to_layer=to_layer,
                        violation_type='cross_layer',
                        severity='MEDIUM'
                    ))
                elif abs(from_level - to_level) > 1:
                    # Skipping layers
                    violations.append(ImportViolation(
                        from_module=from_module,
                        to_module=to_module,
                        from_layer=from_layer,
                        to_layer=to_layer,
                        violation_type='skip_layer',
                        severity='LOW'
                    ))
        
        return violations
    
    def _detect_god_modules(self) -> List[Tuple[str, int]]:
        """Detect modules imported by too many others."""
        god_modules = []
        threshold = 10  # Modules imported by more than 10 others
        
        for module, importers in self.reverse_graph.items():
            if len(importers) > threshold:
                god_modules.append((module, len(importers)))
        
        return sorted(god_modules, key=lambda x: x[1], reverse=True)
    
    def _detect_orphan_modules(self) -> List[str]:
        """Detect modules with no imports or exports."""
        all_modules = set(self.import_graph.keys()) | set(self.reverse_graph.keys())
        orphans = []
        
        for module in all_modules:
            imports = self.import_graph.get(module, set())
            importers = self.reverse_graph.get(module, set())
            
            # Skip main/script files
            if module.endswith('__main__') or module.endswith('_script'):
                continue
                
            if not imports and not importers:
                orphans.append(module)
        
        return orphans
    
    def _detect_module_cycles(self) -> List[List[str]]:
        """Detect cycles at module level using DFS."""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(module: str, path: List[str]) -> None:
            visited.add(module)
            rec_stack.add(module)
            path.append(module)
            
            for neighbor in self.import_graph.get(module, []):
                if neighbor not in visited:
                    dfs(neighbor, path[:])
                elif neighbor in rec_stack:
                    # Found cycle
                    cycle_start = path.index(neighbor)
                    cycle = path[cycle_start:] + [neighbor]
                    if len(cycle) > 2:  # Ignore self-imports
                        cycles.append(cycle)
            
            rec_stack.remove(module)
        
        for module in self.import_graph:
            if module not in visited:
                dfs(module, [])
        
        return cycles
    
    def _generate_violation_issues(self, violations: List[ImportViolation]) -> List[Issue]:
        """Generate issues from layer violations."""
        issues = []
        
        # Group violations by type
        by_type = defaultdict(list)
        for v in violations:
            by_type[v.violation_type].append(v)
        
        # Backward dependencies (most severe)
        for v in by_type['backward']:
            issues.append(Issue(
                type="import_hierarchy_violation",
                severity="HIGH",
                message=f"Backward dependency: {v.from_layer} imports from {v.to_layer}",
                file_path=f"<module:{v.from_module}>",
                line_number=0,
                details={
                    "from_module": v.from_module,
                    "to_module": v.to_module,
                    "from_layer": v.from_layer,
                    "to_layer": v.to_layer,
                    "violation_type": "backward_dependency",
                    "recommendation": f"Move functionality to {v.from_layer} or create abstraction"
                }
            ))
        
        # Layer skipping
        if by_type['skip_layer']:
            issues.append(Issue(
                type="import_layer_skipping",
                severity="MEDIUM",
                message=f"Found {len(by_type['skip_layer'])} layer-skipping imports",
                file_path="<global>",
                line_number=0,
                details={
                    "count": len(by_type['skip_layer']),
                    "examples": [
                        f"{v.from_module} -> {v.to_module}" 
                        for v in by_type['skip_layer'][:5]
                    ],
                    "recommendation": "Use intermediate layers for proper abstraction"
                }
            ))
        
        return issues
    
    def _generate_god_module_issues(self, god_modules: List[Tuple[str, int]]) -> List[Issue]:
        """Generate issues for god modules."""
        issues = []
        
        for module, import_count in god_modules[:5]:  # Top 5
            issues.append(Issue(
                type="god_module",
                severity="MEDIUM",
                message=f"Module '{module}' imported by {import_count} others",
                file_path=f"<module:{module}>",
                line_number=0,
                details={
                    "import_count": import_count,
                    "importers_sample": list(self.reverse_graph[module])[:10],
                    "recommendation": "Consider splitting into smaller, focused modules"
                }
            ))
        
        return issues
    
    def _generate_orphan_module_issues(self, orphans: List[str]) -> List[Issue]:
        """Generate issues for orphan modules."""
        if len(orphans) > 5:
            return [Issue(
                type="orphan_modules",
                severity="LOW",
                message=f"Found {len(orphans)} orphan modules with no imports/exports",
                file_path="<global>",
                line_number=0,
                details={
                    "count": len(orphans),
                    "modules": orphans[:10],
                    "recommendation": "Remove unused modules or integrate into codebase"
                }
            )]
        return []
    
    def _generate_cycle_issues(self, cycles: List[List[str]]) -> List[Issue]:
        """Generate issues for module cycles."""
        issues = []
        
        for cycle in cycles[:5]:  # Report top 5 cycles
            issues.append(Issue(
                type="module_cycle",
                severity="HIGH",
                message=f"Module cycle detected: {' -> '.join(cycle[:3])}...",
                file_path="<global>",
                line_number=0,
                details={
                    "cycle": cycle,
                    "length": len(cycle),
                    "recommendation": "Break cycle by extracting common functionality"
                }
            ))
        
        return issues
    
    def _generate_hierarchy_suggestion(self) -> Issue:
        """Generate overall hierarchy improvement suggestion."""
        return Issue(
            type="import_hierarchy_suggestion",
            severity="INFO",
            message="Import hierarchy can be improved",
            file_path="<global>",
            line_number=0,
            details={
                "suggested_hierarchy": self.STANDARD_LAYERS,
                "current_violations": len(self.violations),
                "recommendation": "Refactor imports to follow layered architecture",
                "benefits": [
                    "Clearer dependency flow",
                    "Better testability",
                    "Easier maintenance",
                    "Prevent circular dependencies"
                ]
            }
        )