"""
Base types and data structures for the auto-fix engine.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path

from ...core.issues import Issue, IssueSeverity
from ..strategies.base import RiskLevel


class FixStatus(Enum):
    """Status of fix application."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ROLLED_BACK = "rolled_back"


class FixPriority(Enum):
    """Priority levels for fix application."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class FixResult:
    """Result of applying a fix."""
    
    status: FixStatus
    message: str
    files_modified: List[str] = field(default_factory=list)
    backup_files: Dict[str, str] = field(default_factory=dict)
    execution_time: float = 0.0
    dependencies_satisfied: bool = True
    rollback_available: bool = False
    validation_passed: bool = False
    
    # Additional metadata
    lines_changed: int = 0
    confidence_score: float = 0.0
    risk_assessment: Optional[RiskLevel] = None


@dataclass
class FixPlan:
    """Plan for applying multiple fixes."""
    
    plan_id: str
    issues: List[Issue] = field(default_factory=list)
    actions: List['FixAction'] = field(default_factory=list)
    dependencies: Dict[str, Set[str]] = field(default_factory=dict)


@dataclass
class FixAction:
    """Individual fix action."""
    
    action_id: str
    issue: Issue
    strategy_name: str
    target_file: str
    action_type: str  # 'replace', 'insert', 'delete', 'move'
    
    # Action details
    content: str = ""
    line_number: Optional[int] = None
    column: Optional[int] = None
    end_line: Optional[int] = None
    end_column: Optional[int] = None
    
    # Metadata
    confidence: float = 1.0
    risk_level: RiskLevel = RiskLevel.LOW
    estimated_time: float = 0.0
    dependencies: Set[str] = field(default_factory=set)
    rollback_info: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    'FixStatus',
    'FixPriority',
    'FixResult', 
    'FixPlan',
    'FixAction'
]