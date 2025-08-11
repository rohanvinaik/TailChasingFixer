"""Core issue tracking data structures."""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, TypedDict, Literal, Set, Iterator, Union
from enum import Enum


class IssueSeverity(Enum):
    """Issue severity levels."""
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4


IssueKind = Literal[
    "circular_import",
    "duplicate_function",
    "semantic_duplicate_function",
    "phantom_function",
    "missing_symbol",
    "wrapper_abstraction",
    "prototype_fragmentation",
    "hallucination_cascade",
    "context_window_thrashing",
    "import_anxiety",
    "semantic_stagnant_placeholder",
    "enhanced_semantic_duplicate"
]


class IssueDict(TypedDict, total=False):
    """Type definition for issue dictionary representation."""
    kind: str
    message: str
    severity: int
    file: Optional[str]
    line: Optional[int]
    end_line: Optional[int]
    column: Optional[int]
    end_column: Optional[int]
    symbol: Optional[str]
    evidence: Dict[str, Any]
    suggestions: List[str]
    confidence: float


@dataclass
class Issue:
    """Represents a detected tail-chasing issue."""
    
    kind: str  # Could be IssueKind but keeping str for flexibility
    message: str
    severity: int
    file: Optional[str] = None
    line: Optional[int] = None
    end_line: Optional[int] = None
    column: Optional[int] = None
    end_column: Optional[int] = None
    symbol: Optional[str] = None
    evidence: Dict[str, Any] = field(default_factory=dict)
    suggestions: List[str] = field(default_factory=list)
    confidence: float = 1.0  # 0.0 to 1.0

    def to_dict(self) -> IssueDict:
        """Convert to dictionary for JSON serialization."""
        result: IssueDict = {
            "kind": self.kind,
            "message": self.message,
            "severity": self.severity,
            "file": self.file,
            "line": self.line,
            "end_line": self.end_line,
            "column": self.column,
            "end_column": self.end_column,
            "symbol": self.symbol,
            "evidence": self.evidence,
            "suggestions": self.suggestions,
            "confidence": self.confidence,
        }
        return result

    @classmethod
    def from_dict(cls, data: Union[Dict[str, Any], IssueDict]) -> Issue:
        """Create from dictionary."""
        return cls(**data)

    def __hash__(self) -> int:
        """Make hashable for deduplication."""
        return hash((self.kind, self.file, self.line, self.symbol))

    def __eq__(self, other: object) -> bool:
        """Equality comparison."""
        if not isinstance(other, Issue):
            return False
        return (
            self.kind == other.kind
            and self.file == other.file
            and self.line == other.line
            and self.symbol == other.symbol
        )


@dataclass
class IssueCollection:
    """Collection of issues with convenience methods."""
    
    issues: List[Issue] = field(default_factory=list)
    _seen_hashes: Set[int] = field(default_factory=set, init=False, repr=False)

    def add(self, issue: Issue) -> None:
        """Add an issue to the collection."""
        self.issues.append(issue)

    def filter_by_kind(self, kind: str) -> List[Issue]:
        """Get all issues of a specific kind."""
        return [i for i in self.issues if i.kind == kind]

    def filter_by_file(self, file: str) -> List[Issue]:
        """Get all issues in a specific file."""
        return [i for i in self.issues if i.file == file]

    def filter_by_severity(self, min_severity: int) -> List[Issue]:
        """Get all issues with at least the given severity."""
        return [i for i in self.issues if i.severity >= min_severity]

    def deduplicate(self) -> None:
        """Remove duplicate issues."""
        seen: Set[Issue] = set()
        unique: List[Issue] = []
        for issue in self.issues:
            if issue not in seen:
                seen.add(issue)
                unique.append(issue)
        self.issues = unique
        self._seen_hashes = {hash(issue) for issue in unique}

    def sort(self, key: Literal["severity", "file", "kind"] = "severity") -> None:
        """Sort issues by the given key."""
        if key == "severity":
            self.issues.sort(key=lambda i: i.severity, reverse=True)
        elif key == "file":
            self.issues.sort(key=lambda i: (i.file or "", i.line or 0))
        elif key == "kind":
            self.issues.sort(key=lambda i: i.kind)

    def __len__(self) -> int:
        """Number of issues."""
        return len(self.issues)

    def __iter__(self) -> Iterator[Issue]:
        """Iterate over issues."""
        return iter(self.issues)
    
    def get_by_kinds(self, kinds: List[str]) -> List[Issue]:
        """Get all issues matching any of the specified kinds."""
        return [i for i in self.issues if i.kind in kinds]
    
    def get_high_confidence(self, threshold: float = 0.8) -> List[Issue]:
        """Get all issues with confidence above threshold."""
        return [i for i in self.issues if i.confidence >= threshold]
    
    def group_by_file(self) -> Dict[str, List[Issue]]:
        """Group issues by file."""
        result: Dict[str, List[Issue]] = {}
        for issue in self.issues:
            file_key = issue.file or "<no-file>"
            result.setdefault(file_key, []).append(issue)
        return result
    
    def to_dict_list(self) -> List[IssueDict]:
        """Convert all issues to dictionary list."""
        return [issue.to_dict() for issue in self.issues]
