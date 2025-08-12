"""Missing symbol analyzer - detects references to non-existent functions/classes."""

import ast
from typing import Set, Dict, List, Any, Optional, Iterable
from collections import defaultdict
from pathlib import Path
import difflib
import re

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno, safe_get_col_offset


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
        import builtins
        for builtin in dir(builtins):
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
        visitor = ReferenceVisitor(file, defined_symbols)
        visitor.visit(tree)
        
        for ref in visitor.missing_references:
            symbol_name = ref["name"]
            
            # Skip normalized variable names from structural hashing
            if re.match(r'^(VAR|ARG|NUM|STR|BOOL|CONST)\d*$', symbol_name):
                continue
                
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
                    # Skip relative imports within the same package
                    if node.module.startswith('.') or node.level > 0:
                        continue
                        
                    # Check if the module exists
                    if not self._module_exists(node.module, ctx):
                        issue = Issue(
                            kind="hallucinated_import",
                            message=f"Import from non-existent module '{node.module}'",
                            severity=3,
                            file=file,
                            line=safe_get_lineno(node),
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
            "line": safe_get_lineno(node, 0),
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
            "line": safe_get_lineno(node)
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
                    "line": safe_get_lineno(node)
                })
                
        self.generic_visit(node)


class ReferenceVisitor(ast.NodeVisitor):
    """Collects all symbol references in a file."""
    
    def __init__(self, file: str, defined_symbols: Dict[str, List[Dict[str, Any]]]):
        self.file = file
        self.defined_symbols = defined_symbols
        self.missing_references: List[Dict[str, Any]] = []
        self.current_function = None
        self.imported_names: Set[str] = set()
        self.local_scopes: List[Set[str]] = [set()]  # Stack of local scopes
        
    def visit_Import(self, node: ast.Import):
        """Track imported names."""
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.imported_names.add(name)
            self.local_scopes[-1].add(name)
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track names from imports."""
        for alias in node.names:
            if alias.name != "*":
                name = alias.asname or alias.name
                self.imported_names.add(name)
                self.local_scopes[-1].add(name)
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track current function context and parameters."""
        old_function = self.current_function
        self.current_function = node.name
        
        # Create new scope with parameters
        new_scope = set()
        for arg in node.args.args:
            new_scope.add(arg.arg)
        # Add special parameters
        if node.args.vararg:
            new_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            new_scope.add(node.args.kwarg.arg)
            
        self.local_scopes.append(new_scope)
        self.generic_visit(node)
        self.local_scopes.pop()
        
        self.current_function = old_function
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Track current async function context."""
        self.visit_FunctionDef(node)
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class definitions."""
        self.local_scopes[-1].add(node.name)
        self.generic_visit(node)
        
    def visit_Name(self, node: ast.Name):
        """Record name references."""
        if isinstance(node.ctx, ast.Load):
            # Check if name is defined locally
            name_defined = False
            
            # Check local scopes
            for scope in self.local_scopes:
                if node.id in scope:
                    name_defined = True
                    break
                    
            # Check global definitions
            if not name_defined and node.id in self.defined_symbols:
                name_defined = True
                
            # If not defined, record as missing
            if not name_defined:
                self.missing_references.append({
                    "name": node.id,
                    "line": safe_get_lineno(node, 0),
                    "column": safe_get_col_offset(node),
                    "node_type": "name",
                    "context": self.current_function or "<module>",
                    "file": self.file
                })
        elif isinstance(node.ctx, ast.Store):
            # Add to current scope
            self.local_scopes[-1].add(node.id)
                
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call):
        """Record function calls."""
        if isinstance(node.func, ast.Name):
            # Check if function is defined
            func_defined = False
            
            # Check local scopes
            for scope in self.local_scopes:
                if node.func.id in scope:
                    func_defined = True
                    break
                    
            # Check global definitions
            if not func_defined and node.func.id in self.defined_symbols:
                func_defined = True
                
            # If not defined, record as missing
            if not func_defined:
                self.missing_references.append({
                    "name": node.func.id,
                    "line": safe_get_lineno(node, 0),
                    "column": safe_get_col_offset(node),
                    "node_type": "call",
                    "context": self.current_function or "<module>",
                    "file": self.file
                })
        
        # Visit arguments but not the func name again
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
        
    def visit_For(self, node: ast.For):
        """Handle for loop variables."""
        # Visit the iterator first
        self.visit(node.iter)
        
        # Then add the target to scope before visiting body
        if isinstance(node.target, ast.Name):
            self.local_scopes[-1].add(node.target.id)
        
        # Visit the target and body
        self.visit(node.target)
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
            
    def visit_ListComp(self, node: ast.ListComp):
        """Handle list comprehension scope."""
        self._visit_comprehension(node)
        
    def visit_SetComp(self, node: ast.SetComp):
        """Handle set comprehension scope."""
        self._visit_comprehension(node)
        
    def visit_DictComp(self, node: ast.DictComp):
        """Handle dict comprehension scope."""
        self._visit_comprehension(node)
        
    def visit_GeneratorExp(self, node: ast.GeneratorExp):
        """Handle generator expression scope."""
        self._visit_comprehension(node)
        
    def _visit_comprehension(self, node):
        """Handle comprehension scopes."""
        # Create new scope for comprehension
        new_scope = set()
        self.local_scopes.append(new_scope)
        
        # Visit generators and add variables to scope
        for generator in node.generators:
            self.visit(generator.iter)
            if isinstance(generator.target, ast.Name):
                new_scope.add(generator.target.id)
            self.visit(generator.target)
            for if_clause in generator.ifs:
                self.visit(if_clause)
                
        # Visit the element(s)
        if hasattr(node, 'elt'):
            self.visit(node.elt)
        elif hasattr(node, 'key'):
            self.visit(node.key)
            self.visit(node.value)
            
        self.local_scopes.pop()
