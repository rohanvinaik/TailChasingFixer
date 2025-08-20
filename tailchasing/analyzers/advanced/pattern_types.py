"""
Pattern type definitions for enhanced tail-chasing detection.

This module defines the data structures and types used for advanced pattern
detection, including confidence scoring, evidence collection, and pattern
classification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Literal
from enum import Enum
import ast


class PatternConfidence(Enum):
    """Confidence levels for pattern detection."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.95


class PatternSeverity(Enum):
    """Severity levels for detected patterns."""
    INFO = 1
    WARNING = 2
    MODERATE = 3
    HIGH = 4
    CRITICAL = 5


PatternType = Literal[
    "hallucination_cascade",
    "context_window_thrashing", 
    "import_anxiety",
    "phantom_subsystem",
    "recursive_delegation",
    "abstraction_overflow",
    "interface_multiplication"
]


@dataclass
class PatternEvidence:
    """Evidence collected for a detected pattern."""
    
    # Core evidence
    files_affected: List[str] = field(default_factory=list)
    functions_involved: List[str] = field(default_factory=list)
    ast_nodes: List[ast.AST] = field(default_factory=list)
    
    # Metrics
    complexity_metrics: Dict[str, float] = field(default_factory=dict)
    similarity_scores: Dict[str, float] = field(default_factory=dict)
    
    # Temporal evidence (if git history available)
    creation_timestamps: List[str] = field(default_factory=list)
    commit_messages: List[str] = field(default_factory=list)
    author_patterns: Dict[str, int] = field(default_factory=dict)
    
    # Statistical evidence
    statistical_significance: Optional[float] = None
    p_value: Optional[float] = None
    z_score: Optional[float] = None
    
    # Pattern-specific evidence
    import_chains: List[List[str]] = field(default_factory=list)
    duplication_clusters: List[Dict[str, Any]] = field(default_factory=list)
    abstraction_layers: List[Dict[str, Any]] = field(default_factory=list)
    
    def add_file(self, filepath: str) -> None:
        """Add a file to the evidence if not already present."""
        if filepath not in self.files_affected:
            self.files_affected.append(filepath)
    
    def add_function(self, function_name: str) -> None:
        """Add a function to the evidence if not already present."""
        if function_name not in self.functions_involved:
            self.functions_involved.append(function_name)
    
    def add_similarity_score(self, comparison: str, score: float) -> None:
        """Add a similarity score for a specific comparison."""
        self.similarity_scores[comparison] = score
    
    def add_complexity_metric(self, metric_name: str, value: float) -> None:
        """Add a complexity metric."""
        self.complexity_metrics[metric_name] = value
    
    def is_statistically_significant(self, threshold: float = 0.05) -> bool:
        """Check if pattern is statistically significant."""
        return self.p_value is not None and self.p_value < threshold


@dataclass 
class TailChasingPattern:
    """
    Represents a detected tail-chasing anti-pattern with comprehensive metadata.
    
    This class encapsulates all information about a detected pattern including
    confidence scores, evidence, and remediation suggestions.
    """
    
    # Core identification
    pattern_type: PatternType
    pattern_id: str
    description: str
    
    # Location information
    primary_file: str
    primary_line: Optional[int] = None
    affected_files: List[str] = field(default_factory=list)
    
    # Confidence and severity
    confidence: float = 0.0
    severity: PatternSeverity = PatternSeverity.WARNING
    
    # Evidence and analysis
    evidence: PatternEvidence = field(default_factory=PatternEvidence)
    
    # Remediation
    suggestions: List[str] = field(default_factory=list)
    automated_fix_available: bool = False
    fix_complexity: Literal["trivial", "moderate", "complex", "manual"] = "moderate"
    
    # Metadata
    detection_timestamp: Optional[str] = None
    analyzer_version: Optional[str] = None
    
    def __post_init__(self) -> None:
        """Validate and normalize pattern data after initialization."""
        # Ensure confidence is in valid range
        self.confidence = max(0.0, min(1.0, self.confidence))
        
        # Add primary file to affected files if not present
        if self.primary_file and self.primary_file not in self.affected_files:
            self.affected_files.append(self.primary_file)
        
        # Ensure evidence references match affected files
        for file in self.affected_files:
            self.evidence.add_file(file)
    
    def add_suggestion(self, suggestion: str) -> None:
        """Add a remediation suggestion."""
        if suggestion and suggestion not in self.suggestions:
            self.suggestions.append(suggestion)
    
    def set_confidence_from_enum(self, confidence: PatternConfidence) -> None:
        """Set confidence using the PatternConfidence enum."""
        self.confidence = confidence.value
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if pattern has high confidence."""
        return self.confidence >= threshold
    
    def is_critical(self) -> bool:
        """Check if pattern is critical severity."""
        return self.severity == PatternSeverity.CRITICAL
    
    def get_impact_score(self) -> float:
        """
        Calculate overall impact score based on confidence, severity, and scope.
        
        Returns:
            float: Impact score from 0.0 to 1.0
        """
        # Base score from confidence and severity
        base_score = (self.confidence * (self.severity.value / 5.0))
        
        # Scope multiplier based on number of affected files
        scope_multiplier = min(1.0 + (len(self.affected_files) - 1) * 0.1, 2.0)
        
        # Statistical significance boost
        stat_boost = 0.1 if self.evidence.is_statistically_significant() else 0.0
        
        impact = min(1.0, base_score * scope_multiplier + stat_boost)
        return round(impact, 3)
    
    def to_issue_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary format compatible with Issue class.
        
        Returns:
            Dict containing issue data for integration with existing system
        """
        return {
            "kind": self.pattern_type,
            "message": self.description,
            "severity": self.severity.value,
            "file": self.primary_file,
            "line": self.primary_line,
            "confidence": self.confidence,
            "evidence": {
                "pattern_id": self.pattern_id,
                "affected_files": self.affected_files,
                "functions_involved": self.evidence.functions_involved,
                "complexity_metrics": self.evidence.complexity_metrics,
                "similarity_scores": self.evidence.similarity_scores,
                "statistical_significance": self.evidence.statistical_significance,
                "impact_score": self.get_impact_score(),
                "automated_fix_available": self.automated_fix_available,
                "fix_complexity": self.fix_complexity
            },
            "suggestions": self.suggestions
        }
    
    @classmethod
    def create_hallucination_cascade(
        cls,
        pattern_id: str,
        primary_file: str,
        fictional_modules: List[str],
        confidence: float = 0.8
    ) -> TailChasingPattern:
        """Factory method for hallucination cascade patterns."""
        description = f"Hallucination cascade detected: {len(fictional_modules)} fictional modules"
        
        pattern = cls(
            pattern_type="hallucination_cascade",
            pattern_id=pattern_id,
            description=description,
            primary_file=primary_file,
            confidence=confidence,
            severity=PatternSeverity.HIGH
        )
        
        # Add evidence
        for module in fictional_modules:
            pattern.evidence.add_function(module)
        
        # Default suggestions
        pattern.add_suggestion("Remove imports of non-existent modules")
        pattern.add_suggestion("Implement missing functionality or use existing libraries")
        pattern.add_suggestion("Verify all module dependencies exist")
        
        return pattern
    
    @classmethod
    def create_context_thrashing(
        cls,
        pattern_id: str,
        primary_file: str,
        duplicate_implementations: List[str],
        confidence: float = 0.9
    ) -> TailChasingPattern:
        """Factory method for context window thrashing patterns."""
        description = f"Context window thrashing: {len(duplicate_implementations)} reimplementations"
        
        pattern = cls(
            pattern_type="context_window_thrashing",
            pattern_id=pattern_id, 
            description=description,
            primary_file=primary_file,
            confidence=confidence,
            severity=PatternSeverity.MODERATE,
            automated_fix_available=True,
            fix_complexity="moderate"
        )
        
        # Add evidence
        for impl in duplicate_implementations:
            pattern.evidence.add_function(impl)
        
        # Default suggestions
        pattern.add_suggestion("Consolidate duplicate implementations")
        pattern.add_suggestion("Extract common functionality to shared utility")
        pattern.add_suggestion("Review and refactor similar functions")
        
        return pattern
    
    @classmethod 
    def create_import_anxiety(
        cls,
        pattern_id: str,
        primary_file: str,
        excessive_imports: List[str],
        confidence: float = 0.7
    ) -> TailChasingPattern:
        """Factory method for import anxiety patterns."""
        description = f"Import anxiety detected: {len(excessive_imports)} excessive imports"
        
        pattern = cls(
            pattern_type="import_anxiety",
            pattern_id=pattern_id,
            description=description,
            primary_file=primary_file,
            confidence=confidence,
            severity=PatternSeverity.WARNING,
            automated_fix_available=True,
            fix_complexity="trivial"
        )
        
        # Add evidence
        pattern.evidence.import_chains = [excessive_imports]
        
        # Default suggestions  
        pattern.add_suggestion("Remove unused imports")
        pattern.add_suggestion("Use more specific imports instead of wildcard imports")
        pattern.add_suggestion("Group related imports logically")
        
        return pattern


@dataclass
class PatternCluster:
    """Represents a cluster of related tail-chasing patterns."""
    
    cluster_id: str
    patterns: List[TailChasingPattern] = field(default_factory=list)
    cluster_type: str = "mixed"
    cohesion_score: float = 0.0
    
    def add_pattern(self, pattern: TailChasingPattern) -> None:
        """Add a pattern to the cluster."""
        if pattern not in self.patterns:
            self.patterns.append(pattern)
            self._recalculate_cohesion()
    
    def get_dominant_pattern_type(self) -> Optional[PatternType]:
        """Get the most common pattern type in cluster."""
        if not self.patterns:
            return None
        
        type_counts = {}
        for pattern in self.patterns:
            type_counts[pattern.pattern_type] = type_counts.get(pattern.pattern_type, 0) + 1
        
        return max(type_counts.keys(), key=type_counts.get)
    
    def get_total_impact(self) -> float:
        """Calculate total impact score for the cluster."""
        return sum(pattern.get_impact_score() for pattern in self.patterns)
    
    def _recalculate_cohesion(self) -> None:
        """Recalculate cluster cohesion score."""
        if len(self.patterns) < 2:
            self.cohesion_score = 1.0
            return
        
        # Simple cohesion based on file overlap
        all_files = set()
        for pattern in self.patterns:
            all_files.update(pattern.affected_files)
        
        # Count patterns that share files
        shared_pairs = 0
        total_pairs = 0
        
        for i, p1 in enumerate(self.patterns):
            for p2 in self.patterns[i+1:]:
                total_pairs += 1
                if set(p1.affected_files) & set(p2.affected_files):
                    shared_pairs += 1
        
        self.cohesion_score = shared_pairs / total_pairs if total_pairs > 0 else 0.0