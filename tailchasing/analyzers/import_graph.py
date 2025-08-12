"""Import graph analyzer for detecting circular imports."""

import ast
from collections import defaultdict, deque
from typing import Dict, Set, List, Tuple, Optional, Iterable
from pathlib import Path

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


class ImportGraphAnalyzer(BaseAnalyzer):
    """Analyzes import dependencies and detects circular imports."""
    
    name = "import_graph"
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Detect circular imports in the codebase."""
        # Build the import graph
        graph = self._build_import_graph(ctx)
        
        # Find all cycles
        cycles = self._find_cycles(graph)
        
        # Create issues for cycles
        for cycle in cycles:
            yield self._create_cycle_issue(cycle, ctx)
            
    def _build_import_graph(self, ctx: AnalysisContext) -> Dict[str, Set[str]]:
        """Build a graph of module dependencies."""
        graph = defaultdict(set)
        
        for file, tree in ctx.ast_index.items():
            module_name = self._file_to_module(file, ctx.root_dir)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imported = alias.name
                        graph[module_name].add(imported)
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        if node.level == 0:
                            # Absolute import
                            imported = node.module
                        else:
                            # Relative import
                            imported = self._resolve_relative_import(
                                module_name, node.module, node.level
                            )
                            
                        if imported:
                            graph[module_name].add(imported)
                            
        return dict(graph)
        
    def _file_to_module(self, file_path: str, root_dir: Path) -> str:
        """Convert file path to module name."""
        try:
            path = Path(file_path)
            rel_path = path.relative_to(root_dir)
            
            # Remove .py extension
            if rel_path.suffix == '.py':
                rel_path = rel_path.with_suffix('')
                
            # Convert path to module name
            parts = rel_path.parts
            
            # Handle __init__.py
            if parts[-1] == '__init__':
                parts = parts[:-1]
                
            return '.'.join(parts)
        except ValueError:
            # File is outside root_dir
            return Path(file_path).stem
            
    def _resolve_relative_import(
        self,
        current_module: str,
        target: Optional[str],
        level: int
    ) -> Optional[str]:
        """Resolve a relative import to an absolute module name."""
        if level == 0:
            return target
            
        parts = current_module.split('.')
        
        # Go up 'level' directories
        if level > len(parts):
            return None
            
        base_parts = parts[:-level] if level <= len(parts) else []
        
        if target:
            base_parts.append(target)
            
        return '.'.join(base_parts) if base_parts else None
        
    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find all cycles in the import graph using DFS."""
        visited = set()
        rec_stack = []
        cycles = []
        
        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.append(node)
            
            for neighbor in graph.get(node, []):
                if neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = rec_stack.index(neighbor)
                    cycle = rec_stack[cycle_start:] + [neighbor]
                    cycles.append(cycle)
                elif neighbor not in visited:
                    dfs(neighbor, path + [neighbor])
                    
            rec_stack.pop()
            
        # Start DFS from each unvisited node
        for node in graph:
            if node not in visited:
                dfs(node, [node])
                
        # Deduplicate cycles (same cycle might be found from different starting points)
        unique_cycles = []
        seen = set()
        
        for cycle in cycles:
            # Normalize cycle by rotating to start with smallest element
            if not cycle:
                continue
                
            min_idx = cycle.index(min(cycle))
            normalized = tuple(cycle[min_idx:] + cycle[:min_idx])
            
            if normalized not in seen:
                seen.add(normalized)
                unique_cycles.append(list(normalized[:-1]))  # Remove duplicate last element
                
        return unique_cycles
        
    def _create_cycle_issue(self, cycle: List[str], ctx: AnalysisContext) -> Issue:
        """Create an issue for a circular import."""
        # Calculate severity based on cycle length
        severity = min(2 + len(cycle) // 3, 4)
        
        # Build a readable cycle description
        cycle_str = ' → '.join(cycle + [cycle[0]])
        
        # Find the actual import locations for better reporting
        import_locations = self._find_import_locations(cycle, ctx)
        
        issue = Issue(
            kind="circular_import",
            message=f"Circular import detected: {cycle_str}",
            severity=severity,
            evidence={
                "cycle": cycle,
                "length": len(cycle),
                "import_locations": import_locations
            },
            suggestions=[
                "Consider refactoring shared code into a separate module",
                "Use import statements inside functions instead of at module level",
                "Review if all imports are necessary"
            ],
            confidence=1.0  # Cycles are definitive
        )
        
        # Set file/line to the first import in the cycle if available
        if import_locations:
            first_loc = import_locations[0]
            issue.file = first_loc["file"]
            issue.line = first_loc["line"]
            
        return issue
        
    def _find_import_locations(
        self,
        cycle: List[str],
        ctx: AnalysisContext
    ) -> List[Dict[str, any]]:
        """Find the actual file locations of imports in the cycle."""
        locations = []
        
        for i, module in enumerate(cycle):
            next_module = cycle[(i + 1) % len(cycle)]
            
            # Find the file for this module
            for file, tree in ctx.ast_index.items():
                if self._file_to_module(file, ctx.root_dir) == module:
                    # Find the import of next_module
                    for node in ast.walk(tree):
                        if isinstance(node, ast.Import):
                            for alias in node.names:
                                if alias.name == next_module:
                                    locations.append({
                                        "file": file,
                                        "line": safe_get_lineno(node),
                                        "from_module": module,
                                        "to_module": next_module
                                    })
                                    
                        elif isinstance(node, ast.ImportFrom):
                            if node.module == next_module:
                                locations.append({
                                    "file": file,
                                    "line": safe_get_lineno(node),
                                    "from_module": module,
                                    "to_module": next_module
                                })
                                
        return locations
