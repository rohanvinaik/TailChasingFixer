"""Utility functions for the tail-chasing detector."""

import ast
from pathlib import Path
from typing import Optional, Union


def find_project_root(start_path: Union[str, Path]) -> Path:
    """Find the project root by looking for common markers.
    
    Args:
        start_path: Path to start searching from
        
    Returns:
        Project root path
    """
    path = Path(start_path).resolve()
    
    # Markers that indicate project root
    markers = [
        ".git",
        ".tailchasing.yml",
        ".tailchasing.yaml",
        "pyproject.toml",
        "setup.py",
        "setup.cfg",
        "requirements.txt"
    ]
    
    # Search up the directory tree
    current = path if path.is_dir() else path.parent
    
    while current != current.parent:
        for marker in markers:
            if (current / marker).exists():
                return current
        current = current.parent
        
    # If no markers found, use the original path
    return path if path.is_dir() else path.parent


def format_file_path(file_path: Union[str, Path], root: Optional[Path] = None) -> str:
    """Format a file path for display.
    
    Args:
        file_path: Path to format
        root: Optional root to make path relative to
        
    Returns:
        Formatted path string
    """
    path = Path(file_path)
    
    if root:
        try:
            return str(path.relative_to(root))
        except ValueError:
            pass
            
    return str(path)


def is_python_file(path: Path) -> bool:
    """Check if a path is a Python file.
    
    Args:
        path: Path to check
        
    Returns:
        True if Python file
    """
    return path.is_file() and path.suffix == ".py"


def safe_read_file(path: Path, encoding: str = "utf-8") -> Optional[str]:
    """Safely read a file's contents.
    
    Args:
        path: Path to read
        encoding: File encoding
        
    Returns:
        File contents or None if error
    """
    try:
        return path.read_text(encoding=encoding, errors="ignore")
    except Exception:
        return None


def calculate_edit_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein edit distance between two strings.
    
    Args:
        s1: First string
        s2: Second string
        
    Returns:
        Edit distance
    """
    if len(s1) < len(s2):
        return calculate_edit_distance(s2, s1)
        
    if len(s2) == 0:
        return len(s1)
        
    previous_row = range(len(s2) + 1)
    
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        
        for j, c2 in enumerate(s2):
            # Cost of insertions, deletions, or substitutions
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            
            current_row.append(min(insertions, deletions, substitutions))
            
        previous_row = current_row
        
    return previous_row[-1]


def normalize_module_name(name: str) -> str:
    """Normalize a module name for comparison.
    
    Args:
        name: Module name
        
    Returns:
        Normalized name
    """
    # Remove common prefixes/suffixes
    name = name.strip()
    
    # Handle relative imports
    while name.startswith("."):
        name = name[1:]
        
    # Normalize separators
    name = name.replace("-", "_")
    
    return name.lower()


def safe_get_lineno(node: ast.AST, default: int = 1) -> int:
    """Safely get line number from AST node.
    
    Args:
        node: AST node
        default: Default value if lineno is not available
        
    Returns:
        Line number or default value
    """
    return getattr(node, 'lineno', default)


def safe_get_end_lineno(node: ast.AST, default: Optional[int] = None) -> Optional[int]:
    """Safely get end line number from AST node.
    
    Args:
        node: AST node
        default: Default value if end_lineno is not available
        
    Returns:
        End line number or default value
    """
    return getattr(node, 'end_lineno', default)


def safe_get_col_offset(node: ast.AST, default: int = 0) -> int:
    """Safely get column offset from AST node.
    
    Args:
        node: AST node
        default: Default value if col_offset is not available
        
    Returns:
        Column offset or default value
    """
    return getattr(node, 'col_offset', default)


def safe_get_end_col_offset(node: ast.AST, default: Optional[int] = None) -> Optional[int]:
    """Safely get end column offset from AST node.
    
    Args:
        node: AST node
        default: Default value if end_col_offset is not available
        
    Returns:
        End column offset or default value
    """
    return getattr(node, 'end_col_offset', default)
