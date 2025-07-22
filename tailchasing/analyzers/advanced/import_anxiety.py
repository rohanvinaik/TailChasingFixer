"""
Import Anxiety Pattern Detection

Detects when LLM adds excessive "defensive" imports.
Pattern: Error → Add import → Add related imports "just in case"
"""

import ast
from typing import Dict, List, Set, Tuple
from collections import defaultdict

from ..base import AnalysisContext
from ...core.issues import Issue


class ImportAnxietyAnalyzer:
    """Detects defensive over-importing patterns."""
    
    name = "import_anxiety"
    
    def __init__(self):
        self.min_imports_threshold = 5
        self.unused_ratio_threshold = 0.66  # More than 2/3 unused
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Analyze imports for anxiety patterns."""
        issues = []
        
        for filepath, tree in ctx.ast_index.items():
            # Extract imports and usage
            imports = self._extract_imports(tree)
            usage = self._extract_usage(tree)
            
            # Analyze each module's imports
            for module, imported_items in imports.items():
                file_issue = self._analyze_module_imports(
                    module, imported_items, usage, filepath
                )
                if file_issue:
                    issues.append(file_issue)
        
        return issues
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract all imports from the AST."""
        imports = defaultdict(set)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module_name = alias.name
                    imports[module_name].add(alias.asname or module_name)
            
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        if alias.name != '*':
                            imports[node.module].add(alias.name)
        
        return dict(imports)
    
    def _extract_usage(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract all symbol usage from the AST."""
        usage = defaultdict(set)
        
        # Skip import statements themselves
        import_nodes = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                import_nodes.add(node)
        
        # Find all name usage
        for node in ast.walk(tree):
            # Skip if this is part of an import statement
            skip = False
            for import_node in import_nodes:
                if node in ast.walk(import_node):
                    skip = True
                    break
            
            if skip:
                continue
            
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                usage['direct'].add(node.id)
            
            elif isinstance(node, ast.Attribute):
                # Track attribute access
                base = node
                parts = []
                while isinstance(base, ast.Attribute):
                    parts.append(base.attr)
                    base = base.value
                
                if isinstance(base, ast.Name):
                    module = base.id
                    if parts:
                        # Record that we used something from this module
                        usage[module].update(parts)
        
        return dict(usage)
    
    def _analyze_module_imports(self,
                              module: str,
                              imported_items: Set[str],
                              usage: Dict[str, Set[str]],
                              filepath: str) -> Issue:
        """Analyze imports from a specific module."""
        
        # Skip if too few imports to be suspicious
        if len(imported_items) < self.min_imports_threshold:
            return None
        
        # Find which imports are actually used
        used_items = usage.get(module, set())
        unused_items = imported_items - used_items
        
        # Also check direct usage (for 'import module' style)
        for item in imported_items:
            if item in usage.get('direct', set()):
                used_items.add(item)
                unused_items.discard(item)
        
        # Calculate unused ratio
        unused_ratio = len(unused_items) / len(imported_items) if imported_items else 0
        
        if unused_ratio > self.unused_ratio_threshold:
            # Detect import pattern
            pattern = self._detect_import_pattern(imported_items)
            
            # Find import line numbers
            import_lines = self._find_import_lines(filepath, module)
            
            return Issue(
                kind="import_anxiety",
                message=f"Importing {len(imported_items)} items from {module} but only using "
                       f"{len(used_items)} ({unused_ratio:.0%} unused). Pattern: {pattern}",
                severity=1,
                file=filepath,
                line=import_lines[0] if import_lines else None,
                evidence={
                    'module': module,
                    'imported_count': len(imported_items),
                    'used_count': len(used_items),
                    'unused_ratio': unused_ratio,
                    'pattern': pattern,
                    'unused_items': sorted(list(unused_items))[:10],  # Limit for readability
                    'used_items': sorted(list(used_items))
                },
                suggestions=[
                    f"Remove unused imports: {', '.join(sorted(list(unused_items))[:5])}"
                    f"{'...' if len(unused_items) > 5 else ''}",
                    "Only import what you actually use",
                    "Consider using more specific imports instead of importing many related items"
                ]
            )
        
        return None
    
    def _detect_import_pattern(self, items: Set[str]) -> str:
        """Detect patterns in import names."""
        if not items:
            return "empty"
        
        # Check for error/exception pattern
        error_related = sum(1 for item in items 
                          if 'error' in item.lower() or 'exception' in item.lower())
        if error_related > len(items) * 0.7:
            return "error_handling_anxiety"
        
        # Check for class import spree (all title case)
        title_case = sum(1 for item in items if item[0].isupper())
        if title_case == len(items):
            return "class_import_spree"
        
        # Check for utility function spree (all lowercase)
        lower_case = sum(1 for item in items if item[0].islower())
        if lower_case == len(items) and len(items) > 10:
            return "utility_function_spree"
        
        # Check for "everything" pattern
        if len(items) > 20:
            return "import_everything"
        
        # Check for similar prefixes (importing all variants)
        prefixes = defaultdict(int)
        for item in items:
            if '_' in item:
                prefix = item.split('_')[0]
                prefixes[prefix] += 1
        
        max_prefix_count = max(prefixes.values()) if prefixes else 0
        if max_prefix_count > len(items) * 0.5:
            return "variant_import_pattern"
        
        return "mixed_imports"
    
    def _find_import_lines(self, filepath: str, module: str) -> List[int]:
        """Find line numbers where a module is imported."""
        # This is a simplified version - in real implementation would use the AST
        return [1]  # Placeholder
