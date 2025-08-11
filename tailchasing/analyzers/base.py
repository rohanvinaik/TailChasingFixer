"""Base analyzer interface and context."""

from __future__ import annotations
from typing import Iterable, Protocol, List, Dict, Any, TYPE_CHECKING, runtime_checkable
from dataclasses import dataclass
from pathlib import Path
import ast

from ..core.issues import Issue
from ..shared import common_functions

if TYPE_CHECKING:
    from ..core.symbols import SymbolTable


@runtime_checkable
class Analyzer(Protocol):
    """Protocol for analyzers."""
    
    name: str
    
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Run the analyzer and yield issues."""
        ...


@dataclass
class AnalysisContext:
    """Context provided to analyzers."""
    
    config: Dict[str, Any]
    root_dir: Path
    file_paths: List[Path]
    ast_index: Dict[str, ast.AST]
    symbol_table: 'SymbolTable'  # Forward reference to avoid circular import
    source_cache: Dict[str, List[str]]
    cache: Dict[str, Any]  # General purpose cache for analyzers
    
    def is_excluded(self, path: str) -> bool:
        """Check if a path should be excluded based on config."""
        return common_functions.is_excluded(path, self.root_dir, self.config)
        
    def get_source_lines(self, file: str) -> List[str]:
        """Get source lines for a file (cached)."""
        return common_functions.get_source_lines(file, self.source_cache)
        
    def should_ignore_issue(self, issue_kind: str) -> bool:
        """Check if an issue type should be ignored."""
        return common_functions.should_ignore_issue(issue_kind, self.config)
        
    def is_placeholder_allowed(self, symbol: str) -> bool:
        """Check if a placeholder is explicitly allowed."""
        return common_functions.is_placeholder_allowed(symbol, self.config)


class BaseAnalyzer:
    """Base class for analyzers with common functionality."""
    
    name: str = "base"
    
    def __init__(self) -> None:
        self.issues: List[Issue] = []
        
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Run the analyzer."""
        raise NotImplementedError("Subclasses must implement run()")
        
    def add_issue(self, issue: Issue) -> None:
        """Add an issue to the results."""
        self.issues.append(issue)
        
    def get_confidence(self, base_confidence: float, modifiers: Dict[str, float]) -> float:
        """Calculate confidence score with modifiers."""
        return common_functions.get_confidence(base_confidence, modifiers)
