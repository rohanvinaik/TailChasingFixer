"""Common functions shared across analyzers and fixers.

This module contains functions that are duplicated across multiple files
in the codebase. By extracting them here, we reduce code duplication and
ensure consistency.
"""

from typing import List, Dict, Any, TYPE_CHECKING
from pathlib import Path
import ast

if TYPE_CHECKING:
    from ..core.issues import Issue


def is_excluded(path: str, root_dir: Path, config: Dict[str, Any]) -> bool:
    """Check if a path should be excluded from analysis.
    
    Determines whether a file or directory should be excluded based on
    the exclusion patterns defined in the configuration.
    
    Args:
        path: File or directory path to check
        root_dir: Root directory of the project
        config: Configuration dictionary
        
    Returns:
        True if the path should be excluded, False otherwise
    """
    exclude_patterns: List[str] = config.get("paths", {}).get("exclude", [])
    path_obj = Path(path)
    
    for pattern in exclude_patterns:
        try:
            path_obj.relative_to(root_dir / pattern)
            return True
        except ValueError:
            pass
            
    return False


def get_source_lines(file: str, source_cache: Dict[str, List[str]]) -> List[str]:
    """Get source code lines for a file with caching.
    
    Retrieves the source code lines for a file, using the cache if
    available to avoid repeated file I/O operations.
    
    Args:
        file: Path to the source file
        source_cache: Cache dictionary to store/retrieve lines
        
    Returns:
        List of source code lines, empty list if file cannot be read
    """
    if file not in source_cache:
        try:
            path = Path(file)
            source_cache[file] = path.read_text().splitlines()
        except Exception:
            source_cache[file] = []
            
    return source_cache[file]


def should_ignore_issue(issue_kind: str, config: Dict[str, Any]) -> bool:
    """Check if an issue type should be ignored.
    
    Determines whether issues of a specific type should be ignored
    based on the configuration settings.
    
    Args:
        issue_kind: The type/kind of issue (e.g., 'phantom_function')
        config: Configuration dictionary
        
    Returns:
        True if this issue type should be ignored, False otherwise
    """
    ignored: List[str] = config.get("ignore_issue_types", [])
    return issue_kind in ignored


def is_placeholder_allowed(symbol: str, config: Dict[str, Any]) -> bool:
    """Check if a placeholder function/class is explicitly allowed.
    
    Some placeholder implementations may be intentional (e.g., abstract
    methods, protocol definitions). This method checks if a specific
    symbol is in the allowed list.
    
    Args:
        symbol: Name of the function or class to check
        config: Configuration dictionary
        
    Returns:
        True if the placeholder is explicitly allowed, False otherwise
    """
    allowed: List[str] = config.get("placeholders", {}).get("allow", [])
    return symbol in allowed


def get_analyzer_cache(analyzer_name: str, cache: Dict[str, Any]) -> Dict[str, Any]:
    """Get or create a cache dictionary for a specific analyzer.
    
    Each analyzer can have its own cache space to store computed values
    that may be reused across multiple files or analysis passes.
    
    Args:
        analyzer_name: Name of the analyzer requesting cache
        cache: Main cache dictionary
        
    Returns:
        Dictionary for analyzer-specific caching
    """
    if analyzer_name not in cache:
        cache[analyzer_name] = {}
    return cache[analyzer_name]


def get_file_metadata(file_path: str, ast_index: Dict[str, ast.AST], 
                     source_cache: Dict[str, List[str]]) -> Dict[str, Any]:
    """Get metadata about a specific file.
    
    Retrieves various metadata about a file including size, complexity,
    and other metrics that may be useful for analysis.
    
    Args:
        file_path: Path to the file
        ast_index: AST index mapping paths to trees
        source_cache: Source code cache
        
    Returns:
        Dictionary containing file metadata
    """
    if file_path not in ast_index:
        return {}
    
    tree = ast_index[file_path]
    lines = get_source_lines(file_path, source_cache)
    
    # Count various elements
    function_count = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))
    class_count = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
    
    return {
        'line_count': len(lines),
        'function_count': function_count,
        'class_count': class_count,
        'has_main': any('if __name__' in line for line in lines),
        'is_test': 'test' in Path(file_path).name.lower()
    }


def get_confidence(base_confidence: float, modifiers: Dict[str, float]) -> float:
    """Calculate confidence score with modifiers.
    
    Adjusts a base confidence score based on various contextual factors
    such as presence of docstrings, whether it's a test file, or if
    there are pragma comments.
    
    Args:
        base_confidence: Initial confidence score (0.0 to 1.0)
        modifiers: Dictionary of modifier names to multipliers
        
    Returns:
        Adjusted confidence score clamped to [0.0, 1.0]
    """
    confidence: float = base_confidence
    
    for key, modifier in modifiers.items():
        if key == "has_docstring":
            confidence *= modifier
        elif key == "is_test_file":
            confidence *= modifier
        elif key == "has_pragma":
            confidence *= modifier
            
    return min(max(confidence, 0.0), 1.0)


def filter_by_severity(issues: List['Issue'], min_severity: int = 1) -> List['Issue']:
    """Filter issues by minimum severity level.
    
    Args:
        issues: List of issues to filter
        min_severity: Minimum severity level to include (1-5)
        
    Returns:
        Filtered list of issues meeting severity threshold
    """
    return [issue for issue in issues if issue.severity >= min_severity]


def group_by_file(issues: List['Issue']) -> Dict[str, List['Issue']]:
    """Group issues by file path for organized reporting.
    
    Args:
        issues: List of issues to group
        
    Returns:
        Dictionary mapping file paths to lists of issues
    """
    grouped: Dict[str, List['Issue']] = {}
    for issue in issues:
        if issue.file not in grouped:
            grouped[issue.file] = []
        grouped[issue.file].append(issue)
    return grouped