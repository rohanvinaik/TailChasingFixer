"""Symbol table and code analysis utilities."""

import ast
from typing import Dict, List, Set, Optional, Any, TypedDict, Union, cast
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


class SymbolReference(TypedDict):
    """Type definition for symbol references."""
    file: str
    line: int
    context: str


class FunctionSymbolInfo(TypedDict):
    """Type definition for function symbol information."""
    file: str
    lineno: int
    node: Optional[ast.FunctionDef]
    args: List[str]
    name: str


class ClassSymbolInfo(TypedDict):
    """Type definition for class symbol information."""
    file: str
    lineno: int
    node: Optional[ast.ClassDef]
    name: str


@dataclass
class Symbol:
    """Represents a symbol (function, class, variable) in the code."""
    name: str
    kind: str  # "function", "class", "variable", "import"
    file: str
    line: int
    end_line: Optional[int] = None
    args: List[str] = field(default_factory=list)
    node: Optional[Union[ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.AST]] = None
    references: List[SymbolReference] = field(default_factory=list)
    evidence: Optional[Dict[str, Any]] = None
    
    def add_reference(self, file: str, line: int, context: str = "") -> None:
        """Add a reference to this symbol."""
        ref: SymbolReference = {
            "file": file,
            "line": line,
            "context": context
        }
        self.references.append(ref)


class SymbolTable:
    """Global symbol table for the codebase."""
    
    def __init__(self) -> None:
        self.symbols: Dict[str, List[Symbol]] = {}
        self.modules: Dict[str, Set[str]] = {}  # module -> exported names
        self.file_symbols: Dict[str, List[Symbol]] = {}  # file -> symbols defined
        
    @property
    def functions(self) -> Dict[str, List[FunctionSymbolInfo]]:
        """Get all function symbols grouped by name."""
        result: Dict[str, List[FunctionSymbolInfo]] = {}
        for name, symbols in self.symbols.items():
            func_symbols: List[FunctionSymbolInfo] = []
            for sym in symbols:
                if sym.kind == "function" and isinstance(sym.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_info: FunctionSymbolInfo = {
                        "file": sym.file,
                        "lineno": sym.line,
                        "node": cast(Optional[ast.FunctionDef], sym.node),
                        "args": sym.args,
                        "name": sym.name
                    }
                    func_symbols.append(func_info)
            if func_symbols:
                result[name] = func_symbols
        return result
    
    @property
    def classes(self) -> Dict[str, List[ClassSymbolInfo]]:
        """Get all class symbols grouped by name."""
        result: Dict[str, List[ClassSymbolInfo]] = {}
        for name, symbols in self.symbols.items():
            class_symbols: List[ClassSymbolInfo] = []
            for sym in symbols:
                if sym.kind == "class" and isinstance(sym.node, ast.ClassDef):
                    class_info: ClassSymbolInfo = {
                        "file": sym.file,
                        "lineno": sym.line,
                        "node": cast(Optional[ast.ClassDef], sym.node),
                        "name": sym.name
                    }
                    class_symbols.append(class_info)
            if class_symbols:
                result[name] = class_symbols
        return result
        
    def add_symbol(self, symbol: Symbol) -> None:
        """Add a symbol to the table."""
        self.symbols.setdefault(symbol.name, []).append(symbol)
        self.file_symbols.setdefault(symbol.file, []).append(symbol)
        
    def get_symbol(self, name: str, file: Optional[str] = None) -> Optional[Symbol]:
        """Get a symbol by name, optionally filtered by file."""
        symbols = self.symbols.get(name, [])
        if not symbols:
            return None
            
        if file:
            # Try to find symbol in the same file first
            for sym in symbols:
                if sym.file == file:
                    return sym
                    
        # Return the first one found
        return symbols[0] if symbols else None
        
    def get_all_symbols(self, name: str) -> List[Symbol]:
        """Get all symbols with the given name."""
        return self.symbols.get(name, [])
        
    def has_symbol(self, name: str) -> bool:
        """Check if a symbol exists."""
        return name in self.symbols
        
    def ingest_file(self, file: str, tree: ast.AST, source: str) -> None:
        """Ingest symbols from a file."""
        visitor = SymbolVisitor(self, file, source)
        visitor.visit(tree)
    
    # Compatibility method for older code
    def ingest(self, file: str, tree: ast.AST, source: str) -> None:
        """Legacy method name for backwards compatibility."""
        self.ingest_file(file, tree, source)
        
    class UndefinedReference(TypedDict):
        """Type definition for undefined references."""
        file: str
        line: int
        referenced_in: str
    
    def find_undefined_references(self) -> Dict[str, List[UndefinedReference]]:
        """Find all references to undefined symbols."""
        undefined: Dict[str, List[SymbolTable.UndefinedReference]] = {}
        
        for symbols in self.symbols.values():
            for symbol in symbols:
                for ref in symbol.references:
                    ref_name = ref.get("context", "")
                    if ref_name and not self.has_symbol(ref_name):
                        undef_ref: SymbolTable.UndefinedReference = {
                            "file": ref["file"],
                            "line": ref["line"],
                            "referenced_in": symbol.name
                        }
                        undefined.setdefault(ref_name, []).append(undef_ref)
                        
        return undefined


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols and references."""
    
    def __init__(self, symbol_table: SymbolTable, file: str, source: str) -> None:
        self.symbol_table = symbol_table
        self.file = file
        self.source_lines = source.splitlines()
        self.current_class: Optional[str] = None
        self.current_function: Optional[str] = None
        self.imported_names: Set[str] = set()
        
    def visit_Import(self, node: ast.Import) -> None:
        """Handle import statements."""
        for alias in node.names:
            name = alias.asname or alias.name.split('.')[0]
            self.imported_names.add(name)
            
            symbol = Symbol(
                name=name,
                kind="import",
                file=self.file,
                line=node.lineno,
                node=node
            )
            self.symbol_table.add_symbol(symbol)
            
        self.generic_visit(node)
        
    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Handle from...import statements."""
        module = node.module or ""
        
        for alias in node.names:
            if alias.name == "*":
                # Can't track star imports precisely
                continue
                
            name = alias.asname or alias.name
            self.imported_names.add(name)
            
            symbol = Symbol(
                name=name,
                kind="import",
                file=self.file,
                line=node.lineno,
                node=node
            )
            symbol.evidence = {"module": module, "level": node.level}
            self.symbol_table.add_symbol(symbol)
            
        self.generic_visit(node)
        
    def visit_FunctionDef(self, node: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> None:
        """Handle function definitions."""
        # Determine full name
        if self.current_class:
            full_name = f"{self.current_class}.{node.name}"
        else:
            full_name = node.name
            
        # Extract arguments
        args = [arg.arg for arg in node.args.args]
        
        symbol = Symbol(
            name=full_name,
            kind="function",
            file=self.file,
            line=node.lineno,
            end_line=node.end_lineno,
            args=args,
            node=node
        )
        
        self.symbol_table.add_symbol(symbol)
        
        # Visit function body
        old_function = self.current_function
        self.current_function = full_name
        self.generic_visit(node)
        self.current_function = old_function
        
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Handle async function definitions."""
        # Treat async functions the same as regular functions
        self.visit_FunctionDef(node)
        
    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Handle class definitions."""
        symbol = Symbol(
            name=node.name,
            kind="class",
            file=self.file,
            line=node.lineno,
            end_line=node.end_lineno,
            node=node
        )
        
        # Extract base classes
        bases = []
        for base in node.bases:
            if isinstance(base, ast.Name):
                bases.append(base.id)
            elif isinstance(base, ast.Attribute):
                bases.append(ast.unparse(base))
                
        if bases:
            symbol.evidence = {"bases": bases}
            
        self.symbol_table.add_symbol(symbol)
        
        # Visit class body
        old_class = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = old_class
        
    def visit_Name(self, node: ast.Name) -> None:
        """Handle name references."""
        if isinstance(node.ctx, ast.Load):
            # This is a reference to a name
            if self.current_function:
                symbol = self.symbol_table.get_symbol(self.current_function)
                if symbol:
                    symbol.add_reference(
                        file=self.file,
                        line=node.lineno,
                        context=node.id
                    )
                    
        self.generic_visit(node)
        
    def visit_Call(self, node: ast.Call) -> None:
        """Handle function calls."""
        # Extract the function name being called
        if isinstance(node.func, ast.Name):
            func_name = node.func.id
        elif isinstance(node.func, ast.Attribute):
            func_name = node.func.attr
            # Could also track the full dotted name
        else:
            func_name = None
            
        if func_name and self.current_function:
            symbol = self.symbol_table.get_symbol(self.current_function)
            if symbol:
                symbol.add_reference(
                    file=self.file,
                    line=node.lineno,
                    context=func_name
                )
                
        self.generic_visit(node)


def extract_module_exports(file_path: str, tree: ast.Module) -> Set[str]:
    """Extract names that a module exports.
    
    This includes:
    - Top-level function and class definitions
    - Names in __all__ if defined
    
    Args:
        file_path: Path to the module file
        tree: AST of the module
        
    Returns:
        Set of exported names
    """
    exports = set()
    
    # Check for __all__ definition
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id == "__all__":
                    # Extract names from __all__
                    if isinstance(node.value, ast.List):
                        for elt in node.value.elts:
                            if isinstance(elt, ast.Constant) and isinstance(elt.value, str):
                                exports.add(elt.value)
                    return exports  # __all__ overrides everything else
                    
    # If no __all__, collect top-level definitions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if not node.name.startswith('_'):  # Skip private names
                exports.add(node.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and not target.id.startswith('_'):
                    exports.add(target.id)
                    
    return exports
