"""
Enhanced missing symbol analyzer with stub generation capabilities.

This analyzer collects callsite information and generates typed skeleton
functions for missing symbols based on their usage patterns.
"""

import ast
import re
from typing import Set, Dict, List, Any, Optional, Iterable, Tuple, Union
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import difflib

from .base import BaseAnalyzer, AnalysisContext
from ..core.issues import Issue
from ..core.utils import safe_get_lineno


@dataclass
class CallSite:
    """Information about a function call site."""
    file: str
    line: int
    column: int
    context: str  # Function or module where call occurs
    args: List[ast.expr] = field(default_factory=list)
    keywords: List[ast.keyword] = field(default_factory=list)
    in_assignment: bool = False
    assignment_target: Optional[str] = None
    in_condition: bool = False
    in_return: bool = False
    comparison_op: Optional[str] = None
    
    
@dataclass
class ParameterInfo:
    """Inferred parameter information."""
    name: str
    position: int
    has_default: bool = False
    default_value: Any = None
    inferred_type: Optional[str] = None
    is_keyword_only: bool = False
    is_vararg: bool = False
    is_kwarg: bool = False
    

@dataclass
class InferredSignature:
    """Inferred function signature from usage."""
    name: str
    parameters: List[ParameterInfo]
    return_type: Optional[str] = None
    is_method: bool = False
    is_async: bool = False
    docstring: Optional[str] = None
    confidence: float = 0.8


class EnhancedMissingSymbolAnalyzer(BaseAnalyzer):
    """
    Enhanced analyzer that collects callsite information and generates stubs.
    
    Features:
    - Collects all callsites of missing symbols
    - Infers parameter count, names, and types
    - Infers return types from usage context
    - Generates typed skeleton functions
    """
    
    name = "enhanced_missing_symbols"
    
    def __init__(self):
        super().__init__()
        self.missing_symbols: Dict[str, List[CallSite]] = defaultdict(list)
        self.defined_symbols: Dict[str, List[Dict[str, Any]]] = {}
        self.inferred_signatures: Dict[str, InferredSignature] = {}
        
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Find missing symbols and collect callsite information."""
        # Collect defined symbols
        self.defined_symbols = self._collect_defined_symbols(ctx)
        
        # Find missing references and collect callsites
        for file, tree in ctx.ast_index.items():
            if ctx.is_excluded(file):
                continue
                
            visitor = EnhancedReferenceVisitor(file, self.defined_symbols)
            visitor.visit(tree)
            
            # Collect callsites for missing symbols
            for call_info in visitor.missing_calls:
                symbol_name = call_info["name"]
                callsite = self._create_callsite(call_info)
                self.missing_symbols[symbol_name].append(callsite)
            
            # Generate issues for missing references
            for ref in visitor.missing_references:
                yield self._create_issue(ref, ctx)
        
        # Infer signatures from collected callsites
        for symbol_name, callsites in self.missing_symbols.items():
            signature = self._infer_signature(symbol_name, callsites)
            self.inferred_signatures[symbol_name] = signature
            
    def _collect_defined_symbols(self, ctx: AnalysisContext) -> Dict[str, List[Dict[str, Any]]]:
        """Collect all defined symbols across the codebase."""
        symbols = defaultdict(list)
        
        # Add built-ins
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
                
        return dict(symbols)
        
    def _create_callsite(self, call_info: Dict[str, Any]) -> CallSite:
        """Create a CallSite object from call information."""
        return CallSite(
            file=call_info["file"],
            line=call_info["line"],
            column=call_info["column"],
            context=call_info.get("context", "<module>"),
            args=call_info.get("args", []),
            keywords=call_info.get("keywords", []),
            in_assignment=call_info.get("in_assignment", False),
            assignment_target=call_info.get("assignment_target"),
            in_condition=call_info.get("in_condition", False),
            in_return=call_info.get("in_return", False),
            comparison_op=call_info.get("comparison_op")
        )
        
    def _create_issue(self, ref: Dict[str, Any], ctx: AnalysisContext) -> Issue:
        """Create an issue for a missing symbol reference."""
        symbol_name = ref["name"]
        
        # Skip normalized names
        if re.match(r'^(VAR|ARG|NUM|STR|BOOL|CONST)\d*$', symbol_name):
            return None
            
        # Get suggestions including stub generation
        suggestions = self._get_suggestions(symbol_name)
        
        # Add stub generation suggestion if we have callsites
        if symbol_name in self.missing_symbols:
            suggestions.insert(0, f"Generate stub using: tailchasing --generate-missing-stubs")
            
        return Issue(
            kind="enhanced_missing_symbol",
            message=f"Reference to undefined symbol '{symbol_name}'",
            severity=2,
            file=ref["file"],
            line=ref["line"],
            column=ref.get("column"),
            symbol=symbol_name,
            evidence={
                "context": ref.get("context", ""),
                "node_type": ref["node_type"],
                "callsite_count": len(self.missing_symbols.get(symbol_name, []))
            },
            suggestions=suggestions,
            confidence=0.85
        )
        
    def _get_suggestions(self, symbol: str) -> List[str]:
        """Get suggestions for a missing symbol."""
        suggestions = []
        
        # Find similar symbols
        all_symbols = self.defined_symbols.keys()
        close_matches = difflib.get_close_matches(
            symbol, all_symbols, n=3, cutoff=0.6
        )
        
        for match in close_matches:
            suggestions.append(f"Did you mean '{match}'?")
            
        return suggestions
        
    def _infer_signature(self, symbol_name: str, callsites: List[CallSite]) -> InferredSignature:
        """Infer function signature from callsites."""
        if not callsites:
            return InferredSignature(name=symbol_name, parameters=[])
            
        # Determine if it's a method (called with self/cls)
        is_method = self._is_likely_method(symbol_name, callsites)
        
        # Infer parameters
        parameters = self._infer_parameters(callsites, is_method)
        
        # Infer return type
        return_type = self._infer_return_type(callsites)
        
        # Check if async
        is_async = any('async' in cs.context.lower() for cs in callsites)
        
        # Generate docstring
        docstring = self._generate_docstring(symbol_name, parameters, return_type)
        
        return InferredSignature(
            name=symbol_name,
            parameters=parameters,
            return_type=return_type,
            is_method=is_method,
            is_async=is_async,
            docstring=docstring,
            confidence=min(0.9, 0.5 + 0.1 * len(callsites))  # More callsites = higher confidence
        )
        
    def _is_likely_method(self, name: str, callsites: List[CallSite]) -> bool:
        """Determine if function is likely a method."""
        # Check naming patterns
        if name.startswith('_') and not name.startswith('__'):
            return True
            
        # Check if called in class context
        class_contexts = sum(1 for cs in callsites if 'self' in cs.context or 'cls' in cs.context)
        return class_contexts > len(callsites) / 2
        
    def _infer_parameters(self, callsites: List[CallSite], is_method: bool) -> List[ParameterInfo]:
        """Infer parameters from callsites."""
        # Track parameter usage across callsites
        max_positional = 0
        keyword_params = set()
        param_types: Dict[Union[int, str], Set[str]] = defaultdict(set)
        
        for callsite in callsites:
            # Count positional arguments
            max_positional = max(max_positional, len(callsite.args))
            
            # Collect keyword arguments
            for kw in callsite.keywords:
                if kw.arg:  # Skip **kwargs
                    keyword_params.add(kw.arg)
                    
            # Try to infer types from arguments
            for i, arg in enumerate(callsite.args):
                inferred_type = self._infer_arg_type(arg)
                if inferred_type:
                    param_types[i].add(inferred_type)
                    
            for kw in callsite.keywords:
                if kw.arg:
                    inferred_type = self._infer_arg_type(kw.value)
                    if inferred_type:
                        param_types[kw.arg].add(inferred_type)
        
        # Build parameter list
        parameters = []
        
        # Add self/cls for methods
        param_offset = 0
        if is_method:
            parameters.append(ParameterInfo(
                name="self",
                position=0,
                inferred_type=None
            ))
            param_offset = 1
            
        # Add positional parameters
        for i in range(max_positional):
            param_name = f"arg{i + 1}"
            
            # Try to get a better name from keyword usage
            if i in param_types and len(param_types[i]) == 1:
                type_hint = list(param_types[i])[0]
                if type_hint in ['str', 'int', 'float', 'bool']:
                    param_name = f"{type_hint[:3]}_param{i + 1}"
                    
            # Determine type
            types = param_types.get(i, set())
            inferred_type = self._consolidate_types(types) if types else None
            
            parameters.append(ParameterInfo(
                name=param_name,
                position=i + param_offset,
                inferred_type=inferred_type
            ))
            
        # Add keyword-only parameters
        for kw_name in sorted(keyword_params):
            types = param_types.get(kw_name, set())
            inferred_type = self._consolidate_types(types) if types else None
            
            parameters.append(ParameterInfo(
                name=kw_name,
                position=len(parameters),
                is_keyword_only=True,
                inferred_type=inferred_type
            ))
            
        return parameters
        
    def _infer_arg_type(self, arg: ast.expr) -> Optional[str]:
        """Infer type of an argument expression."""
        if isinstance(arg, ast.Constant):
            if isinstance(arg.value, str):
                return "str"
            elif isinstance(arg.value, int):
                return "int"
            elif isinstance(arg.value, float):
                return "float"
            elif isinstance(arg.value, bool):
                return "bool"
            elif arg.value is None:
                return "None"
                
        elif isinstance(arg, ast.List):
            return "List"
        elif isinstance(arg, ast.Dict):
            return "Dict"
        elif isinstance(arg, ast.Set):
            return "Set"
        elif isinstance(arg, ast.Tuple):
            return "Tuple"
        elif isinstance(arg, ast.Name):
            # Could potentially track variable types
            return None
        elif isinstance(arg, ast.Call):
            if isinstance(arg.func, ast.Name):
                # Some common constructors
                if arg.func.id in ['list', 'dict', 'set', 'tuple', 'str', 'int', 'float', 'bool']:
                    return arg.func.id.capitalize()
                    
        return None
        
    def _consolidate_types(self, types: Set[str]) -> str:
        """Consolidate multiple type hints into one."""
        if not types:
            return "Any"
        elif len(types) == 1:
            return list(types)[0]
        elif "None" in types:
            other_types = types - {"None"}
            if len(other_types) == 1:
                return f"Optional[{list(other_types)[0]}]"
            else:
                return f"Optional[Union[{', '.join(sorted(other_types))}]]"
        else:
            return f"Union[{', '.join(sorted(types))}]"
            
    def _infer_return_type(self, callsites: List[CallSite]) -> Optional[str]:
        """Infer return type from usage context."""
        return_types = set()
        
        for callsite in callsites:
            if callsite.in_assignment:
                # Could track assignment target usage
                pass
            elif callsite.in_condition:
                return_types.add("bool")
            elif callsite.in_return:
                # Being returned suggests it has a value
                return_types.add("Any")
            elif callsite.comparison_op:
                if callsite.comparison_op in ['==', '!=', 'is', 'is not']:
                    # Comparison suggests it returns something
                    return_types.add("Any")
                elif callsite.comparison_op in ['<', '<=', '>', '>=']:
                    # Numeric comparison
                    return_types.add("Union[int, float]")
                    
        if not return_types:
            return None
        elif len(return_types) == 1:
            return list(return_types)[0]
        else:
            return self._consolidate_types(return_types)
            
    def _generate_docstring(self, name: str, parameters: List[ParameterInfo], 
                           return_type: Optional[str]) -> str:
        """Generate a docstring for the inferred function."""
        lines = [f"Stub for missing function '{name}'.", ""]
        
        if parameters:
            lines.append("Args:")
            for param in parameters:
                if param.name not in ['self', 'cls']:
                    type_hint = f" ({param.inferred_type})" if param.inferred_type else ""
                    lines.append(f"    {param.name}{type_hint}: TODO: Add description")
                    
        if return_type:
            lines.append("")
            lines.append("Returns:")
            lines.append(f"    {return_type}: TODO: Add description")
            
        return "\n".join(lines)
        
    def generate_stub_file(self, output_path: Path) -> str:
        """Generate a Python file with stub implementations."""
        lines = [
            '"""',
            'Auto-generated stub functions for missing symbols.',
            'Generated by TailChasingFixer.',
            '"""',
            '',
            'from typing import Any, Optional, List, Dict, Set, Tuple, Union',
            '',
            ''
        ]
        
        # Group by likely modules (based on naming)
        standalone_funcs = []
        class_methods = defaultdict(list)
        
        for name, signature in self.inferred_signatures.items():
            if '.' in name:
                class_name, method_name = name.rsplit('.', 1)
                signature.name = method_name  # Use short name
                class_methods[class_name].append(signature)
            else:
                standalone_funcs.append(signature)
                
        # Generate standalone functions
        if standalone_funcs:
            lines.append("# Standalone functions")
            lines.append("")
            
            for sig in sorted(standalone_funcs, key=lambda s: s.name):
                lines.extend(self._generate_function_stub(sig))
                lines.append("")
                lines.append("")
                
        # Generate class methods
        if class_methods:
            lines.append("# Class methods")
            lines.append("")
            
            for class_name in sorted(class_methods.keys()):
                lines.append(f"class {class_name}:")
                lines.append(f'    """Stub class for {class_name}."""')
                lines.append("")
                
                for sig in sorted(class_methods[class_name], key=lambda s: s.name):
                    method_lines = self._generate_function_stub(sig, indent="    ")
                    lines.extend(method_lines)
                    lines.append("")
                    
                lines.append("")
                
        return "\n".join(lines)
        
    def _generate_function_stub(self, signature: InferredSignature, indent: str = "") -> List[str]:
        """Generate lines for a function stub."""
        lines = []
        
        # Build parameter list
        params = []
        for param in signature.parameters:
            if param.inferred_type:
                param_str = f"{param.name}: {param.inferred_type}"
            else:
                param_str = param.name
                
            if param.has_default:
                param_str += f" = {param.default_value!r}"
            elif param.is_keyword_only:
                # Add after * for keyword-only
                pass
                
            params.append(param_str)
            
        param_list = ", ".join(params)
        
        # Build function signature
        if signature.is_async:
            def_line = f"{indent}async def {signature.name}({param_list})"
        else:
            def_line = f"{indent}def {signature.name}({param_list})"
            
        if signature.return_type:
            def_line += f" -> {signature.return_type}"
        def_line += ":"
        
        lines.append(def_line)
        
        # Add docstring
        if signature.docstring:
            lines.append(f'{indent}    """')
            for doc_line in signature.docstring.split('\n'):
                lines.append(f"{indent}    {doc_line}")
            lines.append(f'{indent}    """')
        else:
            lines.append(f'{indent}    """Stub implementation for {signature.name}."""')
            
        # Add implementation
        lines.append(f"{indent}    # TODO: Implement this function")
        lines.append(f"{indent}    # Confidence: {signature.confidence:.0%}")
        lines.append(f"{indent}    raise NotImplementedError(")
        lines.append(f'{indent}        "Function {signature.name} is not yet implemented"')
        lines.append(f"{indent}    )")
        
        return lines


class EnhancedReferenceVisitor(ast.NodeVisitor):
    """Enhanced visitor that collects detailed callsite information."""
    
    def __init__(self, file: str, defined_symbols: Dict[str, List[Dict[str, Any]]]):
        self.file = file
        self.defined_symbols = defined_symbols
        self.missing_references: List[Dict[str, Any]] = []
        self.missing_calls: List[Dict[str, Any]] = []
        self.current_function = None
        self.local_scopes: List[Set[str]] = [set()]
        
        # Context tracking
        self.in_assignment = False
        self.assignment_target = None
        self.in_return = False
        self.in_condition = False
        self.comparison_op = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Track function context and parameters."""
        old_function = self.current_function
        self.current_function = node.name
        
        # Create new scope with parameters
        new_scope = set()
        for arg in node.args.args:
            new_scope.add(arg.arg)
        if node.args.vararg:
            new_scope.add(node.args.vararg.arg)
        if node.args.kwarg:
            new_scope.add(node.args.kwarg.arg)
            
        self.local_scopes.append(new_scope)
        self.generic_visit(node)
        self.local_scopes.pop()
        
        self.current_function = old_function
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Track async function context."""
        self.visit_FunctionDef(node)
        
    def visit_Assign(self, node: ast.Assign):
        """Track assignment context."""
        # Mark that we're in assignment
        old_in_assignment = self.in_assignment
        old_target = self.assignment_target
        
        self.in_assignment = True
        if node.targets and isinstance(node.targets[0], ast.Name):
            self.assignment_target = node.targets[0].id
            self.local_scopes[-1].add(node.targets[0].id)
            
        # Visit the value
        self.visit(node.value)
        
        self.in_assignment = old_in_assignment
        self.assignment_target = old_target
        
        # Visit targets
        for target in node.targets:
            if not isinstance(target, ast.Name):
                self.visit(target)
                
    def visit_Return(self, node: ast.Return):
        """Track return context."""
        if node.value:
            old_in_return = self.in_return
            self.in_return = True
            self.visit(node.value)
            self.in_return = old_in_return
            
    def visit_If(self, node: ast.If):
        """Track condition context."""
        old_in_condition = self.in_condition
        self.in_condition = True
        self.visit(node.test)
        self.in_condition = old_in_condition
        
        for stmt in node.body:
            self.visit(stmt)
        for stmt in node.orelse:
            self.visit(stmt)
            
    def visit_Compare(self, node: ast.Compare):
        """Track comparison operations."""
        if node.ops:
            op = node.ops[0]
            if isinstance(op, ast.Eq):
                self.comparison_op = "=="
            elif isinstance(op, ast.NotEq):
                self.comparison_op = "!="
            elif isinstance(op, ast.Lt):
                self.comparison_op = "<"
            elif isinstance(op, ast.LtE):
                self.comparison_op = "<="
            elif isinstance(op, ast.Gt):
                self.comparison_op = ">"
            elif isinstance(op, ast.GtE):
                self.comparison_op = ">="
            elif isinstance(op, ast.Is):
                self.comparison_op = "is"
            elif isinstance(op, ast.IsNot):
                self.comparison_op = "is not"
                
        self.generic_visit(node)
        self.comparison_op = None
        
    def visit_Call(self, node: ast.Call):
        """Record function calls with detailed context."""
        if isinstance(node.func, ast.Name):
            # Check if function is defined
            func_defined = self._is_defined(node.func.id)
            
            if not func_defined:
                call_info = {
                    "name": node.func.id,
                    "line": safe_get_lineno(node),
                    "column": node.col_offset,
                    "file": self.file,
                    "context": self.current_function or "<module>",
                    "args": node.args,
                    "keywords": node.keywords,
                    "in_assignment": self.in_assignment,
                    "assignment_target": self.assignment_target,
                    "in_return": self.in_return,
                    "in_condition": self.in_condition,
                    "comparison_op": self.comparison_op
                }
                self.missing_calls.append(call_info)
                
                # Also add to missing references
                self.missing_references.append({
                    "name": node.func.id,
                    "line": safe_get_lineno(node),
                    "column": node.col_offset,
                    "node_type": "call",
                    "context": self.current_function or "<module>",
                    "file": self.file
                })
        
        # Visit arguments
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            self.visit(keyword.value)
            
    def visit_Name(self, node: ast.Name):
        """Record name references."""
        if isinstance(node.ctx, ast.Load):
            if not self._is_defined(node.id):
                self.missing_references.append({
                    "name": node.id,
                    "line": safe_get_lineno(node),
                    "column": node.col_offset,
                    "node_type": "name",
                    "context": self.current_function or "<module>",
                    "file": self.file
                })
        elif isinstance(node.ctx, ast.Store):
            self.local_scopes[-1].add(node.id)
            
        self.generic_visit(node)
        
    def _is_defined(self, name: str) -> bool:
        """Check if a name is defined."""
        # Check local scopes
        for scope in self.local_scopes:
            if name in scope:
                return True
                
        # Check global definitions
        return name in self.defined_symbols


class SymbolCollector(ast.NodeVisitor):
    """Collects all defined symbols in a file."""
    
    def __init__(self, file: str):
        self.file = file
        self.symbols: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.current_class = None
        
    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Record function definitions."""
        self.symbols[node.name].append({
            "file": self.file,
            "kind": "function",
            "line": safe_get_lineno(node)
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