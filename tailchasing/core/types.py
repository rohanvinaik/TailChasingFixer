"""Consolidated type definitions for the TailChasing analyzer system.

This module provides shared type definitions to eliminate duplicate type
declarations across the codebase and provide a single source of truth for
common data structures and protocols.
"""

from __future__ import annotations

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Set,
    Tuple,
    Union,
    runtime_checkable,
    TYPE_CHECKING
)

if TYPE_CHECKING:
    from ..core.issues import Issue
    from ..core.symbols import SymbolTable

# =============================================================================
# Core Data Structures
# =============================================================================

@dataclass(frozen=True)
class FunctionRecord:
    """Unified function record used across all semantic analyzers.
    
    This consolidates the multiple FunctionRecord definitions that were
    scattered across semantic modules.
    
    Attributes:
        name: Function name
        file: Source file path
        line: Line number in source file
        id: Stable identifier (e.g., fully-qualified name or file:lineno)
        source: Full module source text
        node: The ast.FunctionDef/AsyncFunctionDef node
        hv: Optional hypervector representation for semantic analysis
        metadata: Additional metadata for analysis
    """
    name: str
    file: str
    line: int
    id: str
    source: str
    node: ast.AST
    hv: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Optional convenience fields
    module: Optional[str] = None


@dataclass
class AnalysisContext:
    """Context provided to analyzers during analysis runs."""
    files: List[Path]
    symbol_table: Optional[SymbolTable] = None
    config: Optional[Dict[str, Any]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass  
class AnalysisConfig:
    """Configuration for analysis strategies."""
    enabled_analyzers: Set[str] = field(default_factory=set)
    disabled_analyzers: Set[str] = field(default_factory=set)
    risk_thresholds: Dict[str, int] = field(default_factory=dict)
    custom_patterns: Dict[str, Any] = field(default_factory=dict)
    performance_limits: Dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Type Aliases
# =============================================================================

# File system types
FilePath = Union[str, Path]
FileContent = str

# Analysis types  
IssueList = List['Issue']
SymbolName = str
ModuleName = str
QualifiedName = str

# Semantic analysis types
HypervectorDimensions = List[int]
SimilarityScore = float
SemanticSignature = Dict[str, Any]

# Configuration types
ConfigDict = Dict[str, Any]
ThresholdMap = Dict[str, Union[int, float]]
PatternMap = Dict[str, Any]

# AST types
ASTNode = ast.AST
FunctionNode = Union[ast.FunctionDef, ast.AsyncFunctionDef]
ClassNode = ast.ClassDef
ImportNode = Union[ast.Import, ast.ImportFrom]

# Analysis result types
AnalysisResults = Dict[str, Any]
IssuesByType = Dict[str, IssueList]
MetricsMap = Dict[str, Union[int, float, str]]

# Batch processing types
BatchResults = List[Tuple[FilePath, IssueList]]
ProcessingStats = Dict[str, Union[int, float]]

# LSH and MinHash types
HashSignature = List[int]
BandHash = int
LSHBucket = List[FunctionRecord]


# =============================================================================
# Protocol Definitions
# =============================================================================

@runtime_checkable
class Analyzer(Protocol):
    """Protocol for all analyzers in the system."""
    
    name: str
    
    def run(self, ctx: AnalysisContext) -> Iterable['Issue']:
        """Run the analyzer and yield issues."""
        ...


@runtime_checkable
class AnalysisStrategy(Protocol):
    """Protocol for analysis strategy implementations."""
    
    def analyze(self, context: AnalysisContext, config: AnalysisConfig) -> IssueList:
        """Perform analysis and return issues."""
        ...
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy."""
        ...


@runtime_checkable
class SemanticEncoder(Protocol):
    """Protocol for semantic encoding strategies."""
    
    def encode(self, function: FunctionRecord) -> HypervectorDimensions:
        """Encode a function into a hypervector representation."""
        ...
    
    def encode_batch(self, functions: List[FunctionRecord]) -> List[HypervectorDimensions]:
        """Encode multiple functions efficiently."""
        ...


@runtime_checkable  
class SimilarityCalculator(Protocol):
    """Protocol for similarity calculation methods."""
    
    def calculate(self, hv1: HypervectorDimensions, hv2: HypervectorDimensions) -> SimilarityScore:
        """Calculate similarity between two hypervectors."""
        ...
    
    def calculate_matrix(self, hvs: List[HypervectorDimensions]) -> List[List[SimilarityScore]]:
        """Calculate similarity matrix for multiple hypervectors."""
        ...


@runtime_checkable
class IssueReporter(Protocol):
    """Protocol for issue reporting strategies."""
    
    def report(self, issues: IssueList) -> str:
        """Generate a report from a list of issues."""
        ...
    
    def format_issue(self, issue: 'Issue') -> str:
        """Format a single issue for display."""
        ...


@runtime_checkable
class ConfigProvider(Protocol):
    """Protocol for configuration providers."""
    
    def get_config(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key."""
        ...
    
    def get_analyzer_config(self, analyzer_name: str) -> ConfigDict:
        """Get configuration for a specific analyzer."""
        ...


# =============================================================================
# Abstract Base Classes
# =============================================================================

class BaseAnalyzer(ABC):
    """Abstract base class for analyzers."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def run(self, ctx: AnalysisContext) -> Iterable['Issue']:
        """Run the analyzer and yield issues."""
        pass
    
    def should_skip_file(self, file_path: FilePath) -> bool:
        """Check if a file should be skipped during analysis."""
        return False
    
    def get_file_content(self, file_path: FilePath) -> Optional[FileContent]:
        """Read and return file content."""
        try:
            return Path(file_path).read_text(encoding='utf-8')
        except Exception:
            return None


class BaseSemanticAnalyzer(BaseAnalyzer):
    """Base class for semantic analyzers that work with FunctionRecords."""
    
    def extract_functions(self, source: FileContent, file_path: FilePath) -> List[FunctionRecord]:
        """Extract function records from source code."""
        try:
            tree = ast.parse(source)
            functions = []
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_record = FunctionRecord(
                        name=node.name,
                        file=str(file_path),
                        line=node.lineno,
                        id=f"{file_path}:{node.lineno}",
                        source=source,
                        node=node
                    )
                    functions.append(func_record)
            
            return functions
        except Exception:
            return []


# =============================================================================
# Utility Types for Specific Analyzers
# =============================================================================

@dataclass(frozen=True)
class LSHParams:
    """Parameters for LSH (Locality-Sensitive Hashing) operations."""
    num_hashes: int = 64
    bands: int = 16
    rows_per_band: int = 4
    seed: int = 0x5EED_1DEE


@dataclass(frozen=True)
class BitSignature:
    """Bit signature for MinHash operations."""
    signature: List[int]
    function_id: str


@dataclass
class SemanticCluster:
    """Cluster of semantically similar functions."""
    centroid: HypervectorDimensions
    members: List[FunctionRecord]
    similarity_threshold: SimilarityScore
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DuplicateGroup:
    """Group of duplicate or near-duplicate functions."""
    functions: List[FunctionRecord]
    similarity_score: SimilarityScore
    duplicate_type: str  # 'exact', 'semantic', 'structural'
    confidence: float = 1.0


# =============================================================================
# Compatibility and Migration Support
# =============================================================================

# Legacy type aliases for backward compatibility
LegacyFunctionRecord = FunctionRecord  # For gradual migration
LegacyAnalysisContext = AnalysisContext

# Re-export commonly used types for convenience
__all__ = [
    # Core data structures
    'FunctionRecord',
    'AnalysisContext',
    'AnalysisConfig',
    
    # Type aliases
    'FilePath',
    'FileContent', 
    'IssueList',
    'ConfigDict',
    'HypervectorDimensions',
    'SimilarityScore',
    
    # Protocols
    'Analyzer',
    'AnalysisStrategy',
    'SemanticEncoder',
    'SimilarityCalculator',
    'IssueReporter',
    'ConfigProvider',
    
    # Base classes
    'BaseAnalyzer',
    'BaseSemanticAnalyzer',
    
    # Utility types
    'LSHParams',
    'BitSignature',
    'SemanticCluster',
    'DuplicateGroup',
    
    # Legacy compatibility
    'LegacyFunctionRecord',
    'LegacyAnalysisContext',
]