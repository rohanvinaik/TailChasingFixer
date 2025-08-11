#!/usr/bin/env python3
"""
Check for potential circular imports in the codebase.

This script analyzes import relationships to detect circular dependencies
that could cause runtime import errors.
"""

import ast
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict, deque


class ImportAnalyzer:
    """Analyze import relationships and detect circular dependencies."""
    
    def __init__(self, root_path: Path):
        self.root_path = root_path
        self.imports: Dict[str, Set[str]] = defaultdict(set)
        self.module_paths: Dict[str, Path] = {}
        
    def get_module_name(self, file_path: Path) -> str:
        """Convert file path to module name."""
        try:
            rel_path = file_path.relative_to(self.root_path)
            
            # Remove .py extension
            if rel_path.suffix == '.py':
                rel_path = rel_path.with_suffix('')
            
            # Convert path separators to dots
            parts = list(rel_path.parts)
            
            # Handle __init__.py files
            if parts[-1] == '__init__':
                parts = parts[:-1]
            
            return '.'.join(parts)
        except ValueError:
            # File is outside root path
            return str(file_path)
    
    def extract_imports(self, file_path: Path) -> Set[str]:
        """Extract all imports from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content)
            imports = set()
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.add(alias.name.split('.')[0])
                        
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        # Handle relative imports
                        if node.level > 0:
                            # Relative import - resolve relative to current module
                            current_module = self.get_module_name(file_path)
                            current_parts = current_module.split('.')
                            
                            # Go up 'level' number of packages
                            if node.level <= len(current_parts):
                                base_parts = current_parts[:-node.level] if node.level > 0 else current_parts
                                if node.module:
                                    full_module = '.'.join(base_parts + [node.module])
                                else:
                                    full_module = '.'.join(base_parts)
                                imports.add(full_module.split('.')[0])
                        else:
                            # Absolute import
                            imports.add(node.module.split('.')[0])
            
            return imports
            
        except Exception as e:
            print(f"Warning: Failed to analyze {file_path}: {e}", file=sys.stderr)
            return set()
    
    def build_import_graph(self) -> None:
        """Build the complete import dependency graph."""
        # Find all Python files
        python_files = [
            f for f in self.root_path.rglob("*.py")
            if not any(part in str(f) for part in ['.git', '__pycache__', '.pytest_cache', 'node_modules'])
        ]
        
        # Build module name mapping
        for file_path in python_files:
            module_name = self.get_module_name(file_path)
            self.module_paths[module_name] = file_path
        
        # Extract imports for each module
        for file_path in python_files:
            module_name = self.get_module_name(file_path)
            imports = self.extract_imports(file_path)
            
            # Filter to only include imports that are part of our codebase
            local_imports = {
                imp for imp in imports 
                if imp in self.module_paths or any(imp.startswith(mod + '.') for mod in self.module_paths)
            }
            
            self.imports[module_name] = local_imports
    
    def find_circular_imports(self) -> List[List[str]]:
        """Find all circular import chains."""
        cycles = []
        visited = set()
        rec_stack = set()
        path = []
        
        def dfs(module: str) -> bool:
            if module in rec_stack:
                # Found a cycle - extract it
                cycle_start = path.index(module)
                cycle = path[cycle_start:] + [module]
                cycles.append(cycle)
                return True
                
            if module in visited:
                return False
            
            visited.add(module)
            rec_stack.add(module)
            path.append(module)
            
            # Visit all dependencies
            for dependency in self.imports.get(module, set()):
                if dependency in self.module_paths:  # Only check local modules
                    dfs(dependency)
            
            rec_stack.remove(module)
            path.pop()
            return False
        
        # Check all modules
        for module in self.module_paths.keys():
            if module not in visited:
                dfs(module)
        
        return cycles
    
    def find_strongly_connected_components(self) -> List[List[str]]:
        """Find strongly connected components (groups of mutually dependent modules)."""
        # Tarjan's algorithm for finding SCCs
        index_counter = [0]
        stack = []
        lowlinks = {}
        index = {}
        on_stack = {}
        result = []
        
        def strongconnect(module: str):
            index[module] = index_counter[0]
            lowlinks[module] = index_counter[0]
            index_counter[0] += 1
            stack.append(module)
            on_stack[module] = True
            
            for dependency in self.imports.get(module, set()):
                if dependency not in self.module_paths:
                    continue
                    
                if dependency not in index:
                    strongconnect(dependency)
                    lowlinks[module] = min(lowlinks[module], lowlinks[dependency])
                elif on_stack[dependency]:
                    lowlinks[module] = min(lowlinks[module], index[dependency])
            
            if lowlinks[module] == index[module]:
                component = []
                while True:
                    w = stack.pop()
                    on_stack[w] = False
                    component.append(w)
                    if w == module:
                        break
                if len(component) > 1:  # Only interested in non-trivial SCCs
                    result.append(component)
        
        for module in self.module_paths.keys():
            if module not in index:
                strongconnect(module)
        
        return result


def main():
    """Main entry point."""
    root_path = Path(__file__).parent.parent
    
    print("🔍 Checking for circular imports...")
    
    analyzer = ImportAnalyzer(root_path)
    analyzer.build_import_graph()
    
    # Find circular imports
    cycles = analyzer.find_circular_imports()
    sccs = analyzer.find_strongly_connected_components()
    
    if not cycles and not sccs:
        print("✅ No circular imports detected")
        sys.exit(0)
    
    exit_code = 0
    
    if cycles:
        print(f"❌ Found {len(cycles)} circular import chains:")
        exit_code = 1
        
        for i, cycle in enumerate(cycles, 1):
            print(f"\n  Cycle {i}:")
            for j, module in enumerate(cycle):
                if j < len(cycle) - 1:
                    print(f"    {module} →")
                else:
                    print(f"    {module}")
            
            # Show file paths for the cycle
            print("    Files involved:")
            unique_modules = list(dict.fromkeys(cycle[:-1]))  # Remove duplicate at end
            for module in unique_modules:
                if module in analyzer.module_paths:
                    rel_path = analyzer.module_paths[module].relative_to(root_path)
                    print(f"      {rel_path}")
    
    if sccs:
        print(f"\n🔗 Found {len(sccs)} strongly connected components:")
        exit_code = 1
        
        for i, scc in enumerate(sccs, 1):
            print(f"\n  Component {i} ({len(scc)} modules):")
            for module in scc:
                if module in analyzer.module_paths:
                    rel_path = analyzer.module_paths[module].relative_to(root_path)
                    print(f"    {rel_path}")
    
    if exit_code != 0:
        print("\n💡 To fix circular imports:")
        print("  1. Move common code to a separate module")
        print("  2. Use late imports (import inside functions)")
        print("  3. Restructure module dependencies")
        print("  4. Use dependency injection patterns")
    
    sys.exit(exit_code)


if __name__ == "__main__":
    main()