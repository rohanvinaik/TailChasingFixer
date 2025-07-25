"""File loading and AST parsing utilities."""

import ast
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import logging

logger = logging.getLogger(__name__)


def collect_files(
    root: Path,
    include: Optional[List[str]] = None,
    exclude: Optional[List[str]] = None
) -> List[Path]:
    """Collect Python files from the given root directory.
    
    Args:
        root: Root directory to search
        include: List of paths to include (relative to root)
        exclude: List of paths to exclude (relative to root)
        
    Returns:
        List of Python file paths
    """
    include = include or ["."]
    exclude = set(exclude or [])
    files = []
    
    for inc in include:
        base = root / inc
        if not base.exists():
            logger.warning(f"Include path does not exist: {base}")
            continue
            
        if base.is_file() and base.suffix == ".py":
            files.append(base)
        else:
            for p in base.rglob("*.py"):
                # Check if path should be excluded
                rel_path = p.relative_to(root)
                excluded = False
                
                for ex in exclude:
                    ex_path = Path(ex)
                    try:
                        # Check if the file is under an excluded directory
                        rel_path.relative_to(ex_path)
                        excluded = True
                        break
                    except ValueError:
                        # Not under this exclude path
                        pass
                        
                if not excluded:
                    files.append(p)
                    
    return sorted(set(files))


def parse_file(path: Path) -> Optional[ast.AST]:
    """Parse a single Python file.
    
    Args:
        path: Path to the Python file
        
    Returns:
        AST node or None if parsing failed
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return ast.parse(text, filename=str(path))
    except SyntaxError as e:
        logger.warning(f"Syntax error in {path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to parse {path}: {e}")
        return None


def parse_files(paths: List[Path]) -> Dict[str, ast.AST]:
    """Parse multiple Python files.
    
    Args:
        paths: List of file paths
        
    Returns:
        Dictionary mapping file paths to AST nodes
    """
    result = {}
    
    for p in paths:
        tree = parse_file(p)
        if tree is not None:
            result[str(p)] = tree
        
    return result


def get_source_lines(path: Path) -> List[str]:
    """Get source lines from a file.
    
    Args:
        path: Path to the file
        
    Returns:
        List of source lines
    """
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
        return text.splitlines()
    except Exception as e:
        logger.error(f"Failed to read {path}: {e}")
        return []


def extract_docstring(node: ast.AST) -> Optional[str]:
    """Extract docstring from a function or class node.
    
    Args:
        node: AST node (FunctionDef or ClassDef)
        
    Returns:
        Docstring or None
    """
    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return None
        
    if (node.body and 
        isinstance(node.body[0], ast.Expr) and
        isinstance(node.body[0].value, ast.Constant) and
        isinstance(node.body[0].value.value, str)):
        return node.body[0].value.value
        
    return None


def get_imports(tree: ast.AST) -> List[Dict[str, Any]]:
    """Extract all imports from an AST.
    
    Args:
        tree: AST node
        
    Returns:
        List of import information dictionaries
    """
    imports = []
    
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append({
                    "type": "import",
                    "module": alias.name,
                    "name": alias.asname or alias.name,
                    "lineno": node.lineno,
                })
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for alias in node.names:
                imports.append({
                    "type": "from",
                    "module": module,
                    "name": alias.name,
                    "asname": alias.asname,
                    "lineno": node.lineno,
                    "level": node.level,  # for relative imports
                })
                
    return imports


def find_undefined_names(tree: ast.AST) -> Set[str]:
    """Find potentially undefined names in an AST.
    
    This is a simple heuristic that doesn't handle all cases perfectly
    but works well enough for detecting obvious issues.
    
    Args:
        tree: AST node
        
    Returns:
        Set of potentially undefined names
    """
    defined = set()
    used = set()
    
    # Built-in names that are always available
    builtins = set(dir(__builtins__))
    
    class NameCollector(ast.NodeVisitor):
        def __init__(self):
            self.scopes = [set()]  # Stack of scopes
            
        def visit_FunctionDef(self, node):
            # Add function name to current scope
            self.scopes[-1].add(node.name)
            # Create new scope for function body
            self.scopes.append({arg.arg for arg in node.args.args})
            self.generic_visit(node)
            self.scopes.pop()
            
        def visit_AsyncFunctionDef(self, node):
            self.visit_FunctionDef(node)
            
        def visit_ClassDef(self, node):
            self.scopes[-1].add(node.name)
            self.scopes.append(set())
            self.generic_visit(node)
            self.scopes.pop()
            
        def visit_Name(self, node):
            if isinstance(node.ctx, ast.Store):
                self.scopes[-1].add(node.id)
            elif isinstance(node.ctx, ast.Load):
                used.add(node.id)
            self.generic_visit(node)
            
        def visit_Import(self, node):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                self.scopes[-1].add(name)
            self.generic_visit(node)
            
        def visit_ImportFrom(self, node):
            for alias in node.names:
                if alias.name == "*":
                    # Can't track star imports
                    pass
                else:
                    name = alias.asname or alias.name
                    self.scopes[-1].add(name)
            self.generic_visit(node)
            
        def get_all_defined(self):
            all_defined = set()
            for scope in self.scopes:
                all_defined.update(scope)
            return all_defined
    
    collector = NameCollector()
    collector.visit(tree)
    defined = collector.get_all_defined()
    
    # Find undefined names
    undefined = used - defined - builtins
    
    return undefined
