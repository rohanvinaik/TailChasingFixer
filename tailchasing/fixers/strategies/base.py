"""
Base classes and common functionality for fix strategies.

This module contains the base classes, protocols, and data structures
used by all fix strategies.
"""

import ast
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Tuple, Set
from enum import Enum
import logging

from ...core.issues import Issue
from ...utils.logging_setup import get_logger


class RiskLevel(Enum):
    """Risk levels for fix strategies."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Action:
    """Represents a single action in a fix or rollback plan."""
    type: str  # "modify_file", "create_file", "delete_lines", etc.
    target: str  # File path or identifier
    content: Optional[str] = None
    line_start: Optional[int] = None
    line_end: Optional[int] = None
    backup_content: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass 
class Patch:
    """Represents a proposed fix with metadata."""
    actions: List[Action]
    description: str
    confidence: float  # 0.0 - 1.0
    risk_level: RiskLevel
    estimated_time: float  # seconds
    dependencies: List[str] = field(default_factory=list)  # Other patches this depends on
    rollback_plan: List[Action] = field(default_factory=list)
    validation_tests: List[str] = field(default_factory=list)  # Test commands to verify fix
    side_effects: List[str] = field(default_factory=list)  # Potential side effects
    
    def estimate_impact(self) -> Dict[str, Any]:
        """Estimate the impact of applying this patch."""
        lines_modified = sum(1 for action in self.actions if action.type == "modify_file")
        files_affected = len(set(action.target for action in self.actions))
        
        return {
            "files_affected": files_affected,
            "lines_modified": lines_modified,
            "risk_level": self.risk_level.name,
            "confidence": self.confidence,
            "estimated_time": self.estimated_time
        }


class FixStrategy(Protocol):
    """Protocol defining the interface for fix strategies."""
    
    name: str
    
    def can_handle(self, issue: Issue) -> bool:
        """Check if this strategy can handle the given issue."""
        ...
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix for the issue with full metadata."""
        ...
    
    def estimate_risk(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> RiskLevel:
        """Estimate the risk level of fixing this issue."""
        ...
    
    def get_dependencies(self) -> List[str]:
        """Get list of dependencies required by this strategy."""
        ...
    
    def learn_from_outcome(self, issue: Issue, patch: Patch, success: bool, feedback: str) -> None:
        """Learn from the outcome of applying this strategy."""
        ...


class BaseFixStrategy(ABC):
    """Base implementation of FixStrategy with common functionality."""
    
    # Dependency declarations - override in subclasses
    REQUIRES_ANALYZERS: Tuple[str, ...] = ()
    REQUIRES_TOOLS: Tuple[str, ...] = ()
    REQUIRES_MODELS: Tuple[str, ...] = ()
    
    def __init__(self, name: str):
        self.name = name
        self.logger = get_logger(f"strategy.{name}")
        self.success_history: List[Dict[str, Any]] = []
        self.failure_history: List[Dict[str, Any]] = []
        
    @abstractmethod
    def can_handle(self, issue: Issue) -> bool:
        """Check if this strategy can handle the given issue."""
        pass
    
    @abstractmethod
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose a fix for the issue."""
        pass
    
    def estimate_risk(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> RiskLevel:
        """Estimate the risk level of fixing this issue."""
        # Default implementation based on severity
        if issue.severity <= 2:
            return RiskLevel.LOW
        elif issue.severity <= 3:
            return RiskLevel.MEDIUM
        elif issue.severity <= 4:
            return RiskLevel.HIGH
        else:
            return RiskLevel.CRITICAL
    
    def get_dependencies(self) -> List[str]:
        """Get list of dependencies required by this strategy."""
        deps = []
        for a in getattr(self, "REQUIRES_ANALYZERS", ()):
            deps.append(f"analyzer:{a}")
        for t in getattr(self, "REQUIRES_TOOLS", ()):
            deps.append(f"tool:{t}")
        for m in getattr(self, "REQUIRES_MODELS", ()):
            deps.append(f"model:{m}")
        # Dedup
        out = sorted(set(deps))
        logging.debug("Strategy %s deps: %s", type(self).__name__, out)
        return out
    
    def learn_from_outcome(self, issue: Issue, patch: Patch, success: bool, feedback: str) -> None:
        """Learn from the outcome of applying this strategy."""
        outcome = {
            "issue_kind": issue.kind,
            "patch_description": patch.description,
            "success": success,
            "feedback": feedback,
            "confidence": patch.confidence,
            "risk_level": patch.risk_level.name
        }
        
        if success:
            self.success_history.append(outcome)
            self.logger.info(f"Strategy {self.name} succeeded for {issue.kind}")
        else:
            self.failure_history.append(outcome)
            self.logger.warning(f"Strategy {self.name} failed for {issue.kind}: {feedback}")
    
    def get_success_rate(self) -> float:
        """Calculate the success rate of this strategy."""
        total = len(self.success_history) + len(self.failure_history)
        if total == 0:
            return 0.0
        return len(self.success_history) / total
    
    def create_action(self, action_type: str, target: str, **kwargs) -> Action:
        """Helper to create an action."""
        return Action(type=action_type, target=target, **kwargs)
    
    def create_backup_action(self, original_action: Action) -> Action:
        """Create a backup/rollback action for the given action."""
        if original_action.type == "modify_file":
            return Action(
                type="restore_file",
                target=original_action.target,
                content=original_action.backup_content,
                metadata={"original_action": original_action.type}
            )
        elif original_action.type == "create_file":
            return Action(
                type="delete_file",
                target=original_action.target,
                metadata={"original_action": original_action.type}
            )
        elif original_action.type == "delete_lines":
            return Action(
                type="restore_lines",
                target=original_action.target,
                line_start=original_action.line_start,
                line_end=original_action.line_end,
                content=original_action.backup_content,
                metadata={"original_action": original_action.type}
            )
        else:
            # Generic rollback
            return Action(
                type="rollback",
                target=original_action.target,
                metadata={"original_action": original_action.type}
            )
    
    def extract_code_context(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extract relevant code context for the issue."""
        code_context = {
            "file": issue.file,
            "line": issue.line,
            "symbol": issue.symbol,
            "evidence": issue.evidence or {}
        }
        
        if context:
            # Add source code if available
            if "source_lines" in context:
                code_context["source_lines"] = context["source_lines"]
            
            # Add AST if available
            if "ast" in context:
                code_context["ast"] = context["ast"]
            
            # Add symbol table if available
            if "symbol_table" in context:
                code_context["symbol_table"] = context["symbol_table"]
        
        return code_context