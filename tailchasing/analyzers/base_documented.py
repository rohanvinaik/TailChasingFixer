"""
Base analyzer interface and context for TailChasing Fixer.

This module provides the foundation for all analyzers in the system,
including the analysis context that carries information between components
and the base protocol that all analyzers must implement.
"""

from __future__ import annotations
from typing import Iterable, Protocol, List, Dict, Any, Optional, TYPE_CHECKING, runtime_checkable
from dataclasses import dataclass
from pathlib import Path
import ast

from ..core.issues import Issue
from ..shared import common_functions

if TYPE_CHECKING:
    from ..core.symbols import SymbolTable


@runtime_checkable
class Analyzer(Protocol):
    """Protocol defining the interface for all analyzers.
    
    All analyzers in the TailChasing Fixer system must implement this protocol
    to be compatible with the analysis engine. This ensures consistent behavior
    and allows for easy extension with new analyzer types.
    
    Attributes:
        name: Unique identifier for the analyzer
        
    Examples:
        >>> from tailchasing.analyzers import PhantomAnalyzer
        >>> analyzer = PhantomAnalyzer()
        >>> isinstance(analyzer, Analyzer)
        True
    """
    
    name: str
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Run the analyzer and yield detected issues.
        
        This is the main entry point for analyzer execution. Analyzers
        should yield issues as they are detected for memory efficiency.
        
        Args:
            ctx: Analysis context containing configuration, AST, and symbols
            
        Yields:
            Issue: Detected issues one at a time
            
        Note:
            Analyzers should use generators to yield issues rather than
            returning a complete list for better memory efficiency with
            large codebases.
        """
        ...


@dataclass
class AnalysisContext:
    """Context object carrying information between analysis components.
    
    The AnalysisContext is passed to all analyzers and contains everything
    they need to perform their analysis, including configuration, parsed AST,
    symbol tables, and caching facilities.
    
    Attributes:
        config: Configuration dictionary from .tailchasing.yml or defaults
        root_dir: Root directory of the codebase being analyzed
        file_paths: List of Python files to analyze
        ast_index: Mapping of file paths to parsed AST trees
        symbol_table: Global symbol table with all defined symbols
        source_cache: Cache of source code lines for each file
        cache: General purpose cache for analyzer-specific data
        
    Examples:
        >>> from pathlib import Path
        >>> ctx = AnalysisContext(
        ...     config={'threshold': 0.8},
        ...     root_dir=Path('/project'),
        ...     file_paths=[Path('/project/main.py')],
        ...     ast_index={'/project/main.py': ast.parse('def foo(): pass')},
        ...     symbol_table=SymbolTable(),
        ...     source_cache={},
        ...     cache={}
        ... )
        >>> ctx.is_excluded('/project/tests/test_main.py')
        False
    """
    
    config: Dict[str, Any]
    root_dir: Path
    file_paths: List[Path]
    ast_index: Dict[str, ast.AST]
    symbol_table: 'SymbolTable'  # Forward reference to avoid circular import
    source_cache: Dict[str, List[str]]
    cache: Dict[str, Any]  # General purpose cache for analyzers
    
    def is_excluded(self, path: str) -> bool:
        """Check if a path should be excluded from analysis.
        
        Determines whether a file or directory should be excluded based on
        the exclusion patterns defined in the configuration.
        
        Args:
            path: File or directory path to check
            
        Returns:
            True if the path should be excluded, False otherwise
            
        Examples:
            >>> ctx.config = {'paths': {'exclude': ['tests', '__pycache__']}}
            >>> ctx.is_excluded('project/tests/test_main.py')
            True
            >>> ctx.is_excluded('project/src/main.py')
            False
        """
        return common_functions.is_excluded(path, self.root_dir, self.config)
        
    def get_source_lines(self, file: str) -> List[str]:
        """Get source code lines for a file with caching.
        
        Retrieves the source code lines for a file, using the cache if
        available to avoid repeated file I/O operations.
        
        Args:
            file: Path to the source file
            
        Returns:
            List of source code lines, empty list if file cannot be read
            
        Examples:
            >>> lines = ctx.get_source_lines('/project/main.py')
            >>> len(lines)
            42
            >>> lines[0]
            '#!/usr/bin/env python3'
            
        Note:
            The source cache is shared across all analyzers, improving
            performance when multiple analyzers need to access the same file.
        """
        return common_functions.get_source_lines(file, self.source_cache)
        
    def should_ignore_issue(self, issue_kind: str) -> bool:
        """Check if an issue type should be ignored.
        
        Determines whether issues of a specific type should be ignored
        based on the configuration settings.
        
        Args:
            issue_kind: The type/kind of issue (e.g., 'phantom_function')
            
        Returns:
            True if this issue type should be ignored, False otherwise
            
        Examples:
            >>> ctx.config = {'ignore_issue_types': ['missing_symbol']}
            >>> ctx.should_ignore_issue('missing_symbol')
            True
            >>> ctx.should_ignore_issue('phantom_function')
            False
        """
        return common_functions.should_ignore_issue(issue_kind, self.config)
        
    def is_placeholder_allowed(self, symbol: str) -> bool:
        """Check if a placeholder function/class is explicitly allowed.
        
        Some placeholder implementations may be intentional (e.g., abstract
        methods, protocol definitions). This method checks if a specific
        symbol is in the allowed list.
        
        Args:
            symbol: Name of the function or class to check
            
        Returns:
            True if the placeholder is explicitly allowed, False otherwise
            
        Examples:
            >>> ctx.config = {'placeholders': {'allow': ['AbstractBase.method']}}
            >>> ctx.is_placeholder_allowed('AbstractBase.method')
            True
            >>> ctx.is_placeholder_allowed('ConcreteClass.method')
            False
        """
        return common_functions.is_placeholder_allowed(symbol, self.config)
    
    def get_analyzer_cache(self, analyzer_name: str) -> Dict[str, Any]:
        """Get or create a cache dictionary for a specific analyzer.
        
        Each analyzer can have its own cache space to store computed values
        that may be reused across multiple files or analysis passes.
        
        Args:
            analyzer_name: Name of the analyzer requesting cache
            
        Returns:
            Dictionary for analyzer-specific caching
            
        Examples:
            >>> cache = ctx.get_analyzer_cache('semantic_analyzer')
            >>> cache['computed_vectors'] = vectors
            >>> # Later access:
            >>> vectors = ctx.get_analyzer_cache('semantic_analyzer')['computed_vectors']
        """
        return common_functions.get_analyzer_cache(analyzer_name, self.cache)
    
    def get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Get metadata about a specific file.
        
        Retrieves various metadata about a file including size, complexity,
        and other metrics that may be useful for analysis.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary containing file metadata
            
        Examples:
            >>> metadata = ctx.get_file_metadata('/project/main.py')
            >>> metadata['line_count']
            150
            >>> metadata['function_count']
            12
        """
        return common_functions.get_file_metadata(file_path, self.ast_index, self.source_cache)


class BaseAnalyzer:
    """Base class for analyzers with common functionality.
    
    This class provides common functionality that most analyzers need,
    including issue collection, confidence calculation, and utility methods.
    Concrete analyzers should inherit from this class and implement the
    run() method.
    
    Attributes:
        name: Unique identifier for the analyzer
        issues: List of issues detected during analysis
        
    Examples:
        >>> class MyAnalyzer(BaseAnalyzer):
        ...     name = "my_analyzer"
        ...     
        ...     def run(self, ctx):
        ...         # Analysis logic here
        ...         if self.detect_issue():
        ...             self.add_issue(Issue(...))
        ...         return self.issues
    """
    
    name: str = "base"
    
    def __init__(self) -> None:
        """Initialize the analyzer with an empty issues list."""
        self.issues: List[Issue] = []
        
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Run the analyzer on the provided context.
        
        This method must be implemented by subclasses to perform the
        actual analysis logic.
        
        Args:
            ctx: Analysis context containing all necessary information
            
        Returns:
            Iterable of detected issues
            
        Raises:
            NotImplementedError: If called on base class directly
        """
        raise NotImplementedError("Subclasses must implement run()")
        
    def add_issue(self, issue: Issue) -> None:
        """Add a detected issue to the results.
        
        This is a convenience method for collecting issues during analysis.
        
        Args:
            issue: The issue to add to results
            
        Examples:
            >>> analyzer = BaseAnalyzer()
            >>> issue = Issue(
            ...     kind='phantom_function',
            ...     message='Empty function detected',
            ...     file='main.py',
            ...     line=10
            ... )
            >>> analyzer.add_issue(issue)
            >>> len(analyzer.issues)
            1
        """
        self.issues.append(issue)
        
    def get_confidence(self, base_confidence: float, 
                       modifiers: Dict[str, float]) -> float:
        """Calculate confidence score with modifiers.
        
        Adjusts a base confidence score based on various contextual factors
        such as presence of docstrings, whether it's a test file, or if
        there are pragma comments.
        
        Args:
            base_confidence: Initial confidence score (0.0 to 1.0)
            modifiers: Dictionary of modifier names to multipliers
            
        Returns:
            Adjusted confidence score clamped to [0.0, 1.0]
            
        Examples:
            >>> analyzer = BaseAnalyzer()
            >>> confidence = analyzer.get_confidence(
            ...     0.8,
            ...     {'has_docstring': 1.1, 'is_test_file': 0.7}
            ... )
            >>> round(confidence, 2)
            0.62
            
        Note:
            Common modifiers include:
            - has_docstring: Increase confidence if documented
            - is_test_file: Decrease confidence in test files
            - has_pragma: Decrease if explicitly marked to ignore
        """
        return common_functions.get_confidence(base_confidence, modifiers)
    
    def filter_by_severity(self, issues: List[Issue], 
                          min_severity: int = 1) -> List[Issue]:
        """Filter issues by minimum severity level.
        
        Args:
            issues: List of issues to filter
            min_severity: Minimum severity level to include (1-5)
            
        Returns:
            Filtered list of issues meeting severity threshold
            
        Examples:
            >>> issues = [
            ...     Issue(kind='test', severity=1),
            ...     Issue(kind='test', severity=3),
            ...     Issue(kind='test', severity=5)
            ... ]
            >>> filtered = analyzer.filter_by_severity(issues, min_severity=3)
            >>> len(filtered)
            2
        """
        return common_functions.filter_by_severity(issues, min_severity)
    
    def group_by_file(self, issues: List[Issue]) -> Dict[str, List[Issue]]:
        """Group issues by file path for organized reporting.
        
        Args:
            issues: List of issues to group
            
        Returns:
            Dictionary mapping file paths to lists of issues
            
        Examples:
            >>> issues = [
            ...     Issue(file='a.py', line=1),
            ...     Issue(file='b.py', line=5),
            ...     Issue(file='a.py', line=10)
            ... ]
            >>> grouped = analyzer.group_by_file(issues)
            >>> len(grouped['a.py'])
            2
        """
        return common_functions.group_by_file(issues)