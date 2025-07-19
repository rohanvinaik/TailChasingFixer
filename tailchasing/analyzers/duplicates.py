"""Duplicate function analyzer using structural hashing."""

import ast
import hashlib
from typing import Dict, List, Set, Tuple, Optional
from collections import defaultdict

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue


class DuplicateFunctionAnalyzer(BaseAnalyzer):
    """Detects structurally duplicate functions across the codebase."""
    
    name = "duplicates"
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Find duplicate functions."""
        # Collect all functions with their structural hashes
        function_map = self._collect_functions(ctx)
        
        # Group by hash to find duplicates
        hash_groups = defaultdict(list)
        for func_info in function_map.values():
            hash_groups[func_info["hash"]].append(func_info)
            
        # Create issues for duplicate groups
        for hash_val, group in hash_groups.items():
            if len(group) > 1:
                yield self._create_duplicate_issue(group, ctx)
                
    def _collect_functions(self, ctx: AnalysisContext) -> Dict[str, Dict]:
        """Collect all functions with their structural hashes."""
        functions = {}
        
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            visitor = FunctionCollector(file)
            visitor.visit(tree)
            
            for func_info in visitor.functions:
                # Generate structural hash
                func_info["hash"] = self._structural_hash(func_info["node"])
                
                # Use full qualified name as key
                key = f"{file}:{func_info['full_name']}"
                functions[key] = func_info
                
        return functions
        
    def _structural_hash(self, node: ast.FunctionDef) -> str:
        """Generate a structural hash of a function.
        
        This normalizes variable names and constants to detect
        structurally identical functions.
        """
        # Clone the node to avoid modifying the original
        cloned = ast.copy_location(
            ast.FunctionDef(
                name="FUNC",
                args=self._normalize_arguments(node.args),
                body=[self._normalize_node(stmt) for stmt in node.body],
                decorator_list=[],
                returns=None,
                type_comment=None
            ),
            node
        )
        
        # Generate hash from the normalized AST
        dump = ast.dump(cloned, annotate_fields=False, include_attributes=False)
        return hashlib.sha256(dump.encode()).hexdigest()[:16]
        
    def _normalize_arguments(self, args: ast.arguments) -> ast.arguments:
        """Normalize function arguments."""
        return ast.arguments(
            args=[ast.arg(arg=f"ARG{i}", annotation=None) 
                  for i in range(len(args.args))],
            kwonlyargs=[],
            kw_defaults=[],
            defaults=[],
            posonlyargs=[],
            vararg=None,
            kwarg=None
        )
        
    def _normalize_node(self, node: ast.AST) -> ast.AST:
        """Normalize an AST node for structural comparison."""
        class Normalizer(ast.NodeTransformer):
            def __init__(self):
                self.var_map = {}
                self.var_counter = 0
                
            def visit_Name(self, node: ast.Name):
                # Normalize variable names
                if isinstance(node.ctx, ast.Store):
                    if node.id not in self.var_map:
                        self.var_map[node.id] = f"VAR{self.var_counter}"
                        self.var_counter += 1
                        
                new_id = self.var_map.get(node.id, "VAR")
                return ast.copy_location(
                    ast.Name(id=new_id, ctx=node.ctx),
                    node
                )
                
            def visit_Constant(self, node: ast.Constant):
                # Normalize constants by type
                if isinstance(node.value, str):
                    value = "STR"
                elif isinstance(node.value, (int, float)):
                    value = "NUM"
                elif isinstance(node.value, bool):
                    value = "BOOL"
                else:
                    value = "CONST"
                    
                return ast.copy_location(
                    ast.Constant(value=value),
                    node
                )
                
            def visit_Call(self, node: ast.Call):
                # Normalize function calls but preserve the function name
                # This helps detect similar structure even with different APIs
                return ast.copy_location(
                    ast.Call(
                        func=self.visit(node.func),
                        args=[self.visit(arg) for arg in node.args],
                        keywords=[]
                    ),
                    node
                )
                
        normalizer = Normalizer()
        return normalizer.visit(node)
        
    def _create_duplicate_issue(self, group: List[Dict], ctx: AnalysisContext) -> Issue:
        """Create an issue for a group of duplicate functions."""
        # Sort by file and function name for consistent reporting
        group.sort(key=lambda x: (x["file"], x["name"]))
        
        # Extract unique files
        files = list(set(func["file"] for func in group))
        
        # Build message
        if len(files) == 1:
            # Duplicates within same file
            names = [func["name"] for func in group]
            message = f"Duplicate functions in {files[0]}: {', '.join(names)}"
        else:
            # Duplicates across files
            func_list = [f"{func['name']} ({func['file']})" for func in group]
            message = f"Structurally identical functions: {', '.join(func_list)}"
            
        # Calculate severity based on function size and count
        base_severity = 2
        if len(group) > 2:
            base_severity += 1
        if group[0].get("size", 0) > 20:  # Large duplicated functions are worse
            base_severity += 1
            
        issue = Issue(
            kind="duplicate_function",
            message=message,
            severity=min(base_severity, 4),
            file=group[0]["file"],
            line=group[0]["line"],
            evidence={
                "hash": group[0]["hash"],
                "count": len(group),
                "functions": [
                    {
                        "name": func["name"],
                        "file": func["file"],
                        "line": func["line"],
                        "size": func.get("size", 0)
                    }
                    for func in group
                ]
            },
            suggestions=[
                "Extract common functionality into a shared function",
                "Consider creating a base class if functions are in related classes",
                "Remove redundant implementations"
            ],
            confidence=0.95  # High confidence in structural matching
        )
        
        return issue


class FunctionCollector(ast.NodeVisitor):
    """Collects function information from an AST."""
    
    def __init__(self, file: str):
        self.file = file
        self.functions = []
        self.current_class = None
        
    def visit_ClassDef(self, node: ast.ClassDef):
        """Track class context."""
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Collect function information."""
        # Determine full name
        if self.current_class:
            full_name = f"{self.current_class}.{node.name}"
        else:
            full_name = node.name
            
        # Calculate function size (number of statements)
        size = self._count_statements(node)
        
        self.functions.append({
            "name": node.name,
            "full_name": full_name,
            "file": self.file,
            "line": node.lineno,
            "size": size,
            "node": node,
            "class": self.current_class
        })
        
        # Don't visit nested functions for now
        # self.generic_visit(node)
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Treat async functions the same as regular functions."""
        self.visit_FunctionDef(node)
        
    def _count_statements(self, node: ast.FunctionDef) -> int:
        """Count the number of statements in a function."""
        count = 0
        for stmt in ast.walk(node):
            if isinstance(stmt, ast.stmt) and not isinstance(stmt, ast.FunctionDef):
                count += 1
        return count
