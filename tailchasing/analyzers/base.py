"""Base analyzer interface and context."""

from __future__ import annotations
from typing import Iterable, Protocol, List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import ast

from ..core.issues import Issue


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
    symbol_table: Any  # Avoid circular import
    source_cache: Dict[str, List[str]]
    
    def is_excluded(self, path: str) -> bool:
        """Check if a path should be excluded based on config."""
        exclude_patterns = self.config.get("paths", {}).get("exclude", [])
        path_obj = Path(path)
        
        for pattern in exclude_patterns:
            try:
                path_obj.relative_to(self.root_dir / pattern)
                return True
            except ValueError:
                pass
                
        return False
        
    def get_source_lines(self, file: str) -> List[str]:
        """Get source lines for a file (cached)."""
        if file not in self.source_cache:
            try:
                path = Path(file)
                self.source_cache[file] = path.read_text().splitlines()
            except Exception:
                self.source_cache[file] = []
                
        return self.source_cache[file]
        
    def should_ignore_issue(self, issue_kind: str) -> bool:
        """Check if an issue type should be ignored."""
        ignored = self.config.get("ignore_issue_types", [])
        return issue_kind in ignored
        
    def is_placeholder_allowed(self, symbol: str) -> bool:
        """Check if a placeholder is explicitly allowed."""
        allowed = self.config.get("placeholders", {}).get("allow", [])
        return symbol in allowed


class BaseAnalyzer:
    """Base class for analyzers with common functionality."""
    
    name: str = "base"
    
    def __init__(self):
        self.issues: List[Issue] = []
        
    def run(self, ctx: AnalysisContext) -> Iterable[Issue]:
        """Run the analyzer."""
        raise NotImplementedError("Subclasses must implement run()")
        
    def add_issue(self, issue: Issue) -> None:
        """Add an issue to the results."""
        self.issues.append(issue)
        
    def get_confidence(self, base_confidence: float, modifiers: Dict[str, float]) -> float:
        """Calculate confidence score with modifiers."""
        confidence = base_confidence
        
        for key, modifier in modifiers.items():
            if key == "has_docstring":
                confidence *= modifier
            elif key == "is_test_file":
                confidence *= modifier
            elif key == "has_pragma":
                confidence *= modifier
                
        return min(max(confidence, 0.0), 1.0)
