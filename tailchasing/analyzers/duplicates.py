"""Duplicate function analyzer using structural hashing."""

import ast
import hashlib
from typing import Dict, List, Tuple, Iterable
from collections import defaultdict

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


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
    
    def _normalize_name_map(self, node: ast.AST, 
                           preserve_keywords: bool = True) -> Tuple[ast.AST, str]:
        """
        Normalize all identifiers in an AST to stable placeholders by scope.
        
        This enables comparing code blocks across files regardless of naming.
        Uses scope-aware renaming:
        - Function arguments → ARG0..ARGn
        - Local variables → VAR0..VARn (by first-seen order)
        - Class attributes → ATTR0..ATTRn
        - Function calls → FUNC0..FUNCn (except builtins)
        - Imports → IMPORT0..IMPORTn
        
        Returns:
            Tuple of (normalized_ast, hash_string)
        """
        
        class ScopeAwareNormalizer(ast.NodeTransformer):
            """Normalizes identifiers with scope awareness."""
            
            PYTHON_BUILTINS = {
                'print', 'len', 'range', 'str', 'int', 'float', 'bool', 
                'list', 'dict', 'set', 'tuple', 'type', 'isinstance',
                'hasattr', 'getattr', 'setattr', 'delattr', 'super',
                'open', 'file', 'input', 'sorted', 'reversed', 'enumerate',
                'zip', 'map', 'filter', 'sum', 'min', 'max', 'abs', 'all',
                'any', 'round', 'pow', 'divmod', 'id', 'object', 'None',
                'True', 'False', 'NotImplemented', 'Ellipsis', '__debug__',
                'quit', 'exit', 'copyright', 'credits', 'license'
            }
            
            PYTHON_KEYWORDS = {
                'and', 'or', 'not', 'if', 'elif', 'else', 'for', 'while',
                'break', 'continue', 'def', 'class', 'return', 'yield',
                'import', 'from', 'as', 'try', 'except', 'finally', 'raise',
                'with', 'assert', 'lambda', 'pass', 'del', 'global', 'nonlocal',
                'in', 'is', 'async', 'await'
            }
            
            def __init__(self, preserve_keywords: bool = True):
                self.preserve_keywords = preserve_keywords
                self.scopes = [{}]  # Stack of scope mappings
                self.arg_counter = 0
                self.var_counter = 0
                self.attr_counter = 0
                self.func_counter = 0
                self.import_counter = 0
                self.class_counter = 0
                
            def push_scope(self):
                """Enter a new scope."""
                self.scopes.append({})
                
            def pop_scope(self):
                """Exit current scope."""
                if len(self.scopes) > 1:
                    self.scopes.pop()
                    
            def get_mapping(self, name: str, context: str = 'var') -> str:
                """Get normalized name for an identifier."""
                # Check if it's a builtin or keyword
                if self.preserve_keywords:
                    if name in self.PYTHON_BUILTINS:
                        return name
                    if name in self.PYTHON_KEYWORDS:
                        return name
                
                # Check all scopes from innermost to outermost
                for scope in reversed(self.scopes):
                    if name in scope:
                        return scope[name]
                
                # Create new mapping based on context
                if context == 'arg':
                    mapped = f"ARG{self.arg_counter}"
                    self.arg_counter += 1
                elif context == 'attr':
                    mapped = f"ATTR{self.attr_counter}"
                    self.attr_counter += 1
                elif context == 'func':
                    mapped = f"FUNC{self.func_counter}"
                    self.func_counter += 1
                elif context == 'import':
                    mapped = f"IMPORT{self.import_counter}"
                    self.import_counter += 1
                elif context == 'class':
                    mapped = f"CLASS{self.class_counter}"
                    self.class_counter += 1
                else:  # var
                    mapped = f"VAR{self.var_counter}"
                    self.var_counter += 1
                
                # Store in current scope
                self.scopes[-1][name] = mapped
                return mapped
            
            def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.FunctionDef:
                """Handle function definitions with new scope."""
                self.push_scope()
                
                # Map function name in parent scope
                func_name = "FUNC"  # Always normalize to FUNC for the definition
                
                # Map arguments first
                new_args = []
                for arg in node.args.args:
                    old_name = arg.arg
                    new_name = self.get_mapping(old_name, 'arg')
                    new_args.append(ast.arg(arg=new_name, annotation=None))
                
                # Create new arguments object
                new_arguments = ast.arguments(
                    args=new_args,
                    kwonlyargs=[],
                    kw_defaults=[],
                    defaults=[],
                    posonlyargs=[],
                    vararg=None,
                    kwarg=None
                )
                
                # Visit body
                new_body = [self.visit(stmt) for stmt in node.body]
                
                self.pop_scope()
                
                return ast.FunctionDef(
                    name=func_name,
                    args=new_arguments,
                    body=new_body,
                    decorator_list=[],
                    returns=None,
                    type_comment=None
                )
            
            def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AsyncFunctionDef:
                """Handle async function definitions."""
                # Reuse FunctionDef logic
                result = self.visit_FunctionDef(node)
                return ast.AsyncFunctionDef(
                    name=result.name,
                    args=result.args,
                    body=result.body,
                    decorator_list=result.decorator_list,
                    returns=result.returns,
                    type_comment=result.type_comment
                )
            
            def visit_ClassDef(self, node: ast.ClassDef) -> ast.ClassDef:
                """Handle class definitions with new scope."""
                self.push_scope()
                
                class_name = self.get_mapping(node.name, 'class')
                new_body = [self.visit(stmt) for stmt in node.body]
                
                self.pop_scope()
                
                return ast.ClassDef(
                    name=class_name,
                    bases=[],
                    keywords=[],
                    body=new_body,
                    decorator_list=[]
                )
            
            def visit_Name(self, node: ast.Name) -> ast.Name:
                """Handle variable names."""
                # Determine context
                if isinstance(node.ctx, ast.Store):
                    # First assignment creates the mapping
                    new_name = self.get_mapping(node.id, 'var')
                else:
                    # Load or Del - use existing mapping
                    new_name = self.get_mapping(node.id, 'var')
                
                return ast.Name(id=new_name, ctx=node.ctx)
            
            def visit_Attribute(self, node: ast.Attribute) -> ast.Attribute:
                """Handle attribute access."""
                # Visit the value part
                new_value = self.visit(node.value)
                
                # Normalize attribute name
                if node.attr.startswith('__') and node.attr.endswith('__'):
                    # Keep magic methods
                    new_attr = node.attr
                else:
                    new_attr = f"ATTR_{node.attr}"
                
                return ast.Attribute(
                    value=new_value,
                    attr=new_attr,
                    ctx=node.ctx
                )
            
            def visit_Call(self, node: ast.Call) -> ast.Call:
                """Handle function calls."""
                # Check if it's a builtin
                if isinstance(node.func, ast.Name):
                    if node.func.id in self.PYTHON_BUILTINS:
                        new_func = node.func
                    else:
                        new_func = self.visit(node.func)
                else:
                    new_func = self.visit(node.func)
                
                new_args = [self.visit(arg) for arg in node.args]
                new_keywords = []
                
                return ast.Call(
                    func=new_func,
                    args=new_args,
                    keywords=new_keywords
                )
            
            def visit_Import(self, node: ast.Import) -> ast.Import:
                """Handle imports."""
                new_names = []
                for alias in node.names:
                    name = self.get_mapping(alias.name, 'import')
                    asname = self.get_mapping(alias.asname, 'import') if alias.asname else None
                    new_names.append(ast.alias(name=name, asname=asname))
                return ast.Import(names=new_names)
            
            def visit_ImportFrom(self, node: ast.ImportFrom) -> ast.ImportFrom:
                """Handle from imports."""
                module = self.get_mapping(node.module, 'import') if node.module else None
                new_names = []
                for alias in node.names:
                    name = alias.name if alias.name == "*" else self.get_mapping(alias.name, 'import')
                    asname = self.get_mapping(alias.asname, 'import') if alias.asname else None
                    new_names.append(ast.alias(name=name, asname=asname))
                return ast.ImportFrom(module=module, names=new_names, level=node.level or 0)
            
            def visit_Constant(self, node: ast.Constant) -> ast.Name:
                """Normalize constants to type placeholders."""
                if isinstance(node.value, str):
                    return ast.Name(id="STR", ctx=ast.Load())
                elif isinstance(node.value, (int, float, complex)):
                    return ast.Name(id="NUM", ctx=ast.Load())
                elif isinstance(node.value, bool):
                    return ast.Name(id="BOOL", ctx=ast.Load())
                elif node.value is None:
                    return ast.Name(id="NONE", ctx=ast.Load())
                else:
                    return ast.Name(id="CONST", ctx=ast.Load())
        
        # Normalize the AST
        normalizer = ScopeAwareNormalizer(preserve_keywords)
        normalized = normalizer.visit(ast.copy_location(node, node))
        
        # Generate stable hash
        ast_dump = ast.dump(normalized, annotate_fields=False, include_attributes=False)
        hash_value = hashlib.sha256(ast_dump.encode()).hexdigest()
        
        return normalized, hash_value
        
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
            "line": safe_get_lineno(node),
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
