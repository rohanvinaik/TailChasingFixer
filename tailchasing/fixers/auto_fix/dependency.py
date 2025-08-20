"""
Dependency analysis for the auto-fix engine.

Analyzes dependencies between fixes to determine safe execution order and
prevent conflicts during automated fixing.
"""

import ast
import logging
from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, List, Set

from .base import FixAction

logger = logging.getLogger(__name__)


class DependencyAnalyzer:
    """Analyzes dependencies between fixes to determine safe execution order."""
    
    def __init__(self):
        """Initialize dependency analyzer with empty graph."""
        self.graph: Dict[str, Set[str]] = defaultdict(set)
        self.reverse_graph: Dict[str, Set[str]] = defaultdict(set)
    
    def add_dependency(self, fix_id: str, depends_on: str) -> None:
        """
        Add an explicit dependency relationship.
        
        Args:
            fix_id: ID of fix that has dependency
            depends_on: ID of fix that must be applied first
        """
        self.graph[fix_id].add(depends_on)
        self.reverse_graph[depends_on].add(fix_id)
    
    def analyze_file_dependencies(self, fixes: List[FixAction]) -> Dict[str, Set[str]]:
        """
        Analyze file-level dependencies between fixes based on import relationships.
        
        Args:
            fixes: List of fix actions to analyze
            
        Returns:
            Dictionary mapping fix IDs to their dependencies
        """
        file_dependencies = defaultdict(set)
        
        # Group fixes by file
        fixes_by_file = defaultdict(list)
        for fix in fixes:
            if fix.issue.file:
                fixes_by_file[fix.issue.file].append(fix)
        
        # Analyze import relationships
        for file_path, file_fixes in fixes_by_file.items():
            if not Path(file_path).exists():
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    tree = ast.parse(f.read())
                
                # Find import statements
                imports = set()
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports.add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports.add(node.module)
                
                # Check if any imports correspond to files with fixes
                for imported_module in imports:
                    # Convert module name to potential file paths
                    module_path = imported_module.replace('.', '/')
                    potential_paths = [
                        f"{module_path}.py",
                        f"{module_path}/__init__.py"
                    ]
                    
                    for potential_path in potential_paths:
                        if potential_path in fixes_by_file:
                            # File depends on the imported file
                            for fix in file_fixes:
                                for imported_fix in fixes_by_file[potential_path]:
                                    file_dependencies[fix.action_id].add(imported_fix.action_id)
                                    
            except Exception as e:
                logger.warning(f"Failed to analyze dependencies for {file_path}: {e}")
        
        return file_dependencies
    
    def analyze_symbol_dependencies(self, fixes: List[FixAction]) -> Dict[str, Set[str]]:
        """
        Analyze symbol-level dependencies between fixes.
        
        Args:
            fixes: List of fix actions to analyze
            
        Returns:
            Dictionary mapping fix IDs to their symbol dependencies
        """
        symbol_dependencies = defaultdict(set)
        
        # Build symbol definition and usage maps
        symbol_definitions = {}  # symbol -> fix_id that defines it
        symbol_usages = defaultdict(set)  # symbol -> set of fix_ids that use it
        
        for fix in fixes:
            # Simplified analysis - in practice would need more sophisticated AST analysis
            if fix.issue.symbol:
                symbol = fix.issue.symbol
                
                # Fixes that define/modify symbols
                if fix.action_type in ['replace', 'insert']:
                    symbol_definitions[symbol] = fix.action_id
                
                # Fixes that use symbols  
                elif fix.action_type in ['move', 'delete']:
                    symbol_usages[symbol].add(fix.action_id)
        
        # Build dependencies: usages depend on definitions
        for symbol, using_fixes in symbol_usages.items():
            if symbol in symbol_definitions:
                defining_fix = symbol_definitions[symbol]
                for using_fix in using_fixes:
                    if using_fix != defining_fix:
                        symbol_dependencies[using_fix].add(defining_fix)
        
        return symbol_dependencies
    
    def topological_sort(self, fixes: List[FixAction]) -> List[str]:
        """
        Perform topological sort to determine safe execution order.
        
        Args:
            fixes: List of fix actions to sort
            
        Returns:
            List of fix IDs in safe execution order
        """
        # Build complete dependency graph
        file_deps = self.analyze_file_dependencies(fixes)
        symbol_deps = self.analyze_symbol_dependencies(fixes)
        
        # Combine all dependencies
        all_dependencies = defaultdict(set)
        for fix_id, deps in file_deps.items():
            all_dependencies[fix_id].update(deps)
        for fix_id, deps in symbol_deps.items():
            all_dependencies[fix_id].update(deps)
        
        # Add explicit dependencies
        for fix in fixes:
            for dep in fix.dependencies:
                all_dependencies[fix.action_id].add(dep)
        
        # Kahn's algorithm for topological sorting
        in_degree = defaultdict(int)
        graph = defaultdict(set)
        
        # Build graph and calculate in-degrees
        all_fix_ids = {fix.action_id for fix in fixes}
        for fix_id in all_fix_ids:
            for dep in all_dependencies[fix_id]:
                if dep in all_fix_ids:
                    graph[dep].add(fix_id)
                    in_degree[fix_id] += 1
        
        # Initialize queue with nodes having no dependencies
        queue = deque([fix_id for fix_id in all_fix_ids if in_degree[fix_id] == 0])
        result = []
        
        while queue:
            current = queue.popleft()
            result.append(current)
            
            # Remove current node and update in-degrees
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # Check for circular dependencies
        if len(result) != len(all_fix_ids):
            remaining = all_fix_ids - set(result)
            logger.warning(f"Circular dependencies detected for fixes: {remaining}")
            # Add remaining fixes in default order
            remaining_fixes = [f for f in fixes if f.action_id in remaining]
            result.extend([f.action_id for f in remaining_fixes])
        
        return result
    
    def detect_conflicts(self, fixes: List[FixAction]) -> List[tuple[str, str, str]]:
        """
        Detect potential conflicts between fixes.
        
        Args:
            fixes: List of fix actions to check for conflicts
            
        Returns:
            List of (fix1_id, fix2_id, conflict_reason) tuples
        """
        conflicts = []
        
        # Group fixes by file and line
        line_fixes = defaultdict(list)
        for fix in fixes:
            if fix.line_number is not None:
                key = (fix.target_file, fix.line_number)
                line_fixes[key].append(fix)
        
        # Check for line-level conflicts
        for (file_path, line_num), file_fixes in line_fixes.items():
            if len(file_fixes) > 1:
                # Multiple fixes on same line
                for i, fix1 in enumerate(file_fixes):
                    for fix2 in file_fixes[i+1:]:
                        conflicts.append((
                            fix1.action_id, 
                            fix2.action_id, 
                            f"Both modify line {line_num} in {file_path}"
                        ))
        
        # Check for symbol conflicts
        symbol_fixes = defaultdict(list)
        for fix in fixes:
            if fix.issue.symbol:
                symbol_fixes[fix.issue.symbol].append(fix)
        
        for symbol, fixes_list in symbol_fixes.items():
            if len(fixes_list) > 1:
                # Multiple fixes for same symbol
                for i, fix1 in enumerate(fixes_list):
                    for fix2 in fixes_list[i+1:]:
                        if fix1.action_type == 'delete' and fix2.action_type == 'replace':
                            conflicts.append((
                                fix1.action_id,
                                fix2.action_id, 
                                f"Conflict: deleting and replacing symbol '{symbol}'"
                            ))
        
        return conflicts


__all__ = ['DependencyAnalyzer']