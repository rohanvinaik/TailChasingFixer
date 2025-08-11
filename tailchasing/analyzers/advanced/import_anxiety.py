"""
Import anxiety analyzer for detecting defensive over-importing patterns.
"""

import ast
from collections import defaultdict
from typing import List, Dict, Set
from .base_advanced import PatternDetectionAnalyzer
from ..base import Analyzer
from ...core.issues import Issue


class ImportAnxietyAnalyzer(PatternDetectionAnalyzer, Analyzer):
    """Detect defensive over-importing patterns."""
    
    name = "import_anxiety"
    
    def _initialize_specific_config(self):
        """Initialize import anxiety specific configuration."""
        super()._initialize_specific_config()
        self.set_config('min_imports', 5)  # Minimum imports to consider anxiety
        self.set_threshold('unused_ratio', 2.0)  # Unused:used ratio threshold
    
    @property
    def min_imports(self):
        """Get minimum imports configuration."""
        return self.get_config('min_imports', 5)
    
    @property
    def unused_ratio_threshold(self):
        """Get unused ratio threshold."""
        return self.get_threshold('unused_ratio', 2.0)
    
    def run(self, ctx) -> List[Issue]:
        """Run import anxiety analysis."""
        issues = []
        
        for filepath, tree in ctx.ast_index.items():
            file_issues = self._analyze_file_imports(filepath, tree)
            issues.extend(file_issues)
        
        return issues
    
    def _analyze_file_imports(self, filepath: str, tree: ast.AST) -> List[Issue]:
        """Analyze imports in a single file."""
        issues = []
        
        # Extract imports and usage
        imports = self._extract_imports(tree)
        usage = self._extract_usage(tree)
        
        # Analyze each module's import pattern
        for module, imported_items in imports.items():
            if len(imported_items) >= self.min_imports:
                used_items = usage.get(module, set())
                unused_items = imported_items - used_items
                
                if len(unused_items) > len(used_items) * self.unused_ratio_threshold:
                    import_pattern = self._detect_import_pattern(imported_items)
                    
                    issues.append(Issue(
                        kind="import_anxiety",
                        message=f"Importing {len(imported_items)} items from {module} but only using {len(used_items)}",
                        severity=1,
                        file=filepath,
                        line=0,  # Would need more sophisticated line tracking
                        evidence={
                            "module": module,
                            "imported_count": len(imported_items),
                            "used_count": len(used_items),
                            "unused_items": list(unused_items)[:10],  # Limit for readability
                            "pattern": import_pattern,
                            "unused_ratio": len(unused_items) / max(len(used_items), 1)
                        },
                        suggestions=[
                            f"Remove unused imports: {', '.join(list(unused_items)[:5])}{'...' if len(unused_items) > 5 else ''}",
                            "Import only what you need",
                            "Use qualified imports where appropriate",
                            "Consider using 'from module import specific_item' instead of wildcard imports"
                        ]
                    ))
        
        return issues
    
    def _extract_imports(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract all imports from the AST."""
        imports = defaultdict(set)
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    # For "import module", we consider the module name as imported
                    imports[alias.name].add(alias.asname or alias.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name == '*':
                        # Wildcard import - flag as potential anxiety
                        imports[node.module].add('*')
                    else:
                        imports[node.module].add(alias.name)
        
        return dict(imports)
    
    def _extract_usage(self, tree: ast.AST) -> Dict[str, Set[str]]:
        """Extract usage of imported items."""
        usage = defaultdict(set)
        
        # This is simplified - a full implementation would need to track
        # qualified names and handle aliased imports properly
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                # Simple heuristic: if we see a name being used, mark it as used
                # This would need to be more sophisticated in a real implementation
                for module in usage.keys():
                    if node.id in usage[module]:
                        usage[module].add(node.id)
            elif isinstance(node, ast.Attribute):
                # Handle qualified access like "module.function"
                if isinstance(node.value, ast.Name):
                    module_name = node.value.id
                    usage[module_name].add(node.attr)
        
        return dict(usage)
    
    def _detect_import_pattern(self, items: Set[str]) -> str:
        """Detect patterns in import names."""
        items_list = list(items)
        
        # Check for wildcard imports
        if '*' in items:
            return "wildcard_import"
        
        # Check for error/exception anxiety
        error_items = [item for item in items_list 
                      if 'error' in item.lower() or 'exception' in item.lower()]
        if len(error_items) > len(items_list) * 0.5:
            return "error_handling_anxiety"
        
        # Check for class import spree
        if all(item.istitle() for item in items_list if item != '*'):
            return "class_import_spree"
        
        # Check for "import everything" pattern
        if len(items) > 15:
            return "import_everything"
        
        # Check for related functionality grouping
        common_prefixes = self._find_common_prefixes(items_list)
        if len(common_prefixes) == 1 and len(common_prefixes[0]) > 3:
            return f"grouped_functionality_{common_prefixes[0]}"
        
        return "mixed_imports"
    
    def _find_common_prefixes(self, items: List[str]) -> List[str]:
        """Find common prefixes in import names."""
        if not items:
            return []
        
        prefixes = []
        for length in range(3, 8):  # Check prefixes of length 3-7
            prefix_groups = defaultdict(list)
            for item in items:
                if len(item) >= length:
                    prefix = item[:length].lower()
                    prefix_groups[prefix].append(item)
            
            # If more than half the items share a prefix, it's significant
            for prefix, group in prefix_groups.items():
                if len(group) > len(items) * 0.5:
                    prefixes.append(prefix)
        
        return prefixes
