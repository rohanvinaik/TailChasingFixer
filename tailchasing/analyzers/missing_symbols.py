"""Missing symbol analyzer - detects references to non-existent functions/classes."""

import ast
from typing import Set, Dict, List, Any, Optional
from collections import defaultdict
import difflib
import re

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue


class MissingSymbolAnalyzer(BaseAnalyzer):
    """Detects references to non-existent symbols (hallucinated imports/calls)."""
    
    name = "missing_symbols"
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Find all references to missing symbols."""
        # First, collect all defined symbols and their locations
        defined_symbols = self._collect_defined_symbols(ctx)
        
        # Then, find all symbol references
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            # Find missing references in this file
            missing_refs = self._find_missing_references(
                file, tree, defined_symbols, ctx
            )
            
            for ref in missing_refs:
                yield ref
                
    def _collect_defined_symbols(self, ctx: AnalysisContext) -> Dict[str, List[Dict[str, Any]]]:
        """Collect all defined symbols across the codebase."""
        symbols = defaultdict(list)
        
        # Add built-in symbols
        for builtin in dir(__builtins__):
            symbols[builtin].append({
                "file": "<builtin>",
                "kind": "builtin",
                "line": 0
            })
            
        # Collect from all files
        for file, tree in ctx.ast_index.items():
            visitor = SymbolCollector(file)
            visitor.visit(tree)
            
            for name, locations in visitor.symbols.items():
                symbols[name].extend(locations)
                
        # Also add symbols from the symbol table if available
        if hasattr(ctx.symbol_table, 'symbols'):
            for name, symbol_list in ctx.symbol_table.symbols.items():
                for symbol in symbol_list:
                    symbols[name].append({
                        "file": symbol.file,
                        "kind": symbol.kind,
                        "line": symbol.line
                    })
                    
        return dict(symbols)
        
    def _find_missing_references(
        self,
        file: str,
        tree: ast.AST,
        defined_symbols: Dict[str, List[Dict[str, Any]]],
        ctx: AnalysisContext
    ) -> List[Issue]:
        """Find missing symbol references in a file."""
        issues = []
        
        # Track imports in this file
        local_imports = self._get_local_imports(tree)
        
        # Find all name references
        visitor = ReferenceVisitor(file)
        visitor.visit(tree)
        
        for ref in visitor.references:
            symbol_name = ref["name"]
            
            # Skip if it's defined locally or imported
            if symbol_name in local_imports:
                continue
                
            # Check if the symbol exists anywhere
            if symbol_name not in defined_symbols:
                # Try to find similar symbols for suggestions
                suggestions = self._find_similar_symbols(
                    symbol_name, defined_symbols.keys()
                )
                
                issue = Issue(
                    kind="missing_symbol",
                    message=f"Reference to undefined symbol '{symbol_name}'",
                    severity=2,
                    file=file,
                    line=ref["line"],
                    column=ref.get("column"),
                    symbol=symbol_name,
                    evidence={
                        "context": ref.get("context", ""),
                        "node_type": ref["node_type"]
                    },
                    suggestions=suggestions,
                    confidence=self._calculate_confidence(ref, ctx)
                )
                
                issues.append(issue)
                
        # Check for hallucinated imports
        issues.extend(self._check_hallucinated_imports(file, tree, ctx))
        
        return issues
        
    def _get_local_imports(self, tree: ast.AST) -> Set[str]:
        """Get all names imported in this file."""
        imports = set()
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    name = alias.asname or alias.name.split('.')[0]
                    imports.add(name)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    if alias.name != "*":
                        name = alias.asname or alias.name
                        imports.add(name)
                        
        return imports
        
    def _check_hallucinated_imports(
        self,
        file: str,
        tree: ast.AST,
        ctx: AnalysisContext
    ) -> List[Issue]:
        """Check for imports of non-existent modules."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module:
                    # Check if the module exists
                    if not self._module_exists(node.module, ctx):
                        issue = Issue(
                            kind="hallucinated_import",
                            message=f"Import from non-existent module '{node.module}'",
                            severity=3,
                            file=file,
                            line=node.lineno,
                            symbol=node.module,
                            evidence={
                                "imported_names": [alias.name for alias in node.names],
                                "level": node.level
                            },
                            confidence=0.9
                        )
                        issues.append(issue)
                        
        return issues
        
    def _module_exists(self, module_name: str, ctx: AnalysisContext) -> bool:
        """Check if a module exists in the project."""
        # Convert module name to potential file paths
        parts = module_name.split('.')
        
        # Check for .py file
        py_path = ctx.root_dir / Path(*parts).with_suffix('.py')
        if py_path.exists():
            return True
            
        # Check for package directory with __init__.py
        pkg_path = ctx.root_dir / Path(*parts) / '__init__.py'
        if pkg_path.exists():
            return True
            
        # Could also check standard library and installed packages
        # For now, assume external imports are valid
        return True
        
    def _find_similar_symbols(
        self,
        symbol: str,
        all_symbols: Iterable[str],
        max_results: int = 3
    ) -> List[str]:
        """Find symbols similar to the given one."""
        # Use difflib to find close matches
        close_matches = difflib.get_close_matches(
            symbol, all_symbols, n=max_results, cutoff=0.6
        )
        
        suggestions = []
        for match in close_matches:
            suggestions.append(f"Did you mean '{match}'?")
            
        # Also check for case mismatches
        lower_symbol = symbol.lower()
        for existing in all_symbols:
            if existing.lower() == lower_symbol and existing != symbol:
                suggestions.append(f"Did you mean '{existing}' (case mismatch)?")
                break
                
        return suggestions[:max_results]
        
    def _calculate_confidence(self, ref: Dict[str, Any], ctx: AnalysisContext) -> float:
        """Calculate confidence score for a missing symbol issue."""
        confidence = 0.8
        
        # Higher confidence for function calls
        if ref["node_type"] == "call":
            confidence = 0.9
            
        # Lower confidence if in a string or comment context
        if ref.get("in_string", False):
            confidence = 0.3
            
        # Lower confidence for test files
        if "test" in ref.get("file", "").lower():
            confidence *= 0.7
            
        return confidence


class SymbolCollector(ast.NodeVisitor):
    """Collects all defined symbols in a file."""
    
    def __init__(self, file: str):
        self.file = file
        self.symbols: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_class = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Record function definitions."""
        name = node.name
        if self.current_class:
            name = f"{self.current_class}.{name}"
            
        self.symbols[node.name].append({
            "file": self.file,
            "kind": "function",
            "line": node.lineno,
            "full_name": name
        })
        
        self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Record async function definitions."""
        self.visit_FunctionDef(node)
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Record class definitions."""
        self.symbols[node.name].append({
            "file": self.file,
            "kind": "class",
            "line": node.lineno
        })
        
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_Assign(self, node: ast.Assign):
        """Record variable assignments."""
        for target in node.targets:
            if isinstance(target, ast.Name):
                self.symbols[target.id].append({
                    "file": self.file,
                    "kind": "variable",
                    "line": node.lineno
                })
                
        self.generic_visit(node)


class ReferenceVisitor(ast.NodeVisitor):
    """Collects all symbol references in a file."""
    
    def __init__(self, file: str):
        self.file = file
        self.references: List[Dict[str, Any]] = []
        self.current_function = None
        self.imported_names: Set[str] = set()
        
    def visit_Import(self, node: ast.Import):
        """Track imported names."""
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.imported_names.add(name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track names from imports."""
        for alias in node.names:
            if alias.name != "*":
                name = alias.asname or alias.name
                self.imported_names.add(name)
        self.generic_visit(node)
        
    def visit_Name(self, node: ast.Name):
        """Record name references."""
        if isinstance(node.ctx, ast.Load):
            # Skip if it's an imported name
            if node.id not in self.imported_names:
                self.references.append({
                    "name": node.id,
                    "line": node.lineno,
                    "column": node.col_offset,
                    "node_type": "name",
                    "context": self.current_function or "<module>",
                    "file": self.file
                })
                
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        """Record function calls."""
        if isinstance(node.func, ast.Name):
            if node.func.id not in self.imported_names:
                self.references.append({
                    "name": node.func.id,
                    "line": node.lineno,
                    "column": node.col_offset,
                    "node_type": "call",
                    "context": self.current_function or "<module>",
                    "file": self.file
                })
        elif isinstance(node.func, ast.Attribute):
            # For method calls, we might want to check the attribute name
            # This is more complex as we need type information
            pass
            
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track current function context."""
        old_function = self.current_function
        self.current_function = node.name
        self.generic_visit(node)
        self.current_function = old_function
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Track current async function context."""
        self.visit_FunctionDef(node)
