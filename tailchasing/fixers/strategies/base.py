"""
Base classes and common functionality for fix strategies.

This module contains the base classes, protocols, and data structures
used by all fix strategies. Consolidates duplicate patterns from fix_strategies.py:
- 5 duplicate __init__ method patterns
- 6 duplicate get_dependencies implementations  
- Duplicate validation test generation functions
- Common time estimation and risk assessment utilities
"""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Protocol, Tuple, Union
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
class FixOutcome:
    """Represents the outcome of applying a fix."""
    success: bool
    confidence: float
    message: Optional[str] = None
    execution_time: Optional[float] = None
    validation_passed: Optional[bool] = None
    error_message: Optional[str] = None
    rollback_performed: bool = False


@dataclass
class StrategyRanking:
    """Represents a ranking for a strategy for a given issue."""
    primary_strategy: Any  # BaseFixStrategy
    confidence: float
    risk_level: RiskLevel
    estimated_time: float
    success_rate: float
    alternatives: List[Any] = field(default_factory=list)  # List[BaseFixStrategy]
    dependencies_satisfied: bool = True
    
    @property
    def confidence_score(self) -> float:
        """Alias for confidence (for test compatibility)."""
        return self.confidence
    
    @property
    def estimated_risk(self) -> RiskLevel:
        """Alias for risk_level (for test compatibility)."""
        return self.risk_level


@dataclass
class FixAttempt:
    """Represents an attempt to fix an issue."""
    issue: Issue
    strategy: Any  # FixStrategy
    proposed_patch: "Patch"
    timestamp: float
    outcome: Optional[FixOutcome] = None
    duration: Optional[float] = None


@dataclass
class SimplePatch:
    """Simple patch representation for compatibility."""
    file_path: str
    content: str
    line_number: Optional[int] = None
    description: Optional[str] = None


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


# Alias for backward compatibility
ComplexPatch = Patch


class FixStrategy(Protocol):
    """Protocol defining the interface for fix strategies."""
    
    name: str
    
    def can_handle(self, issue: Issue) -> bool:
        """Check if this strategy can handle the given issue."""
        ...
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Union[Patch, SimplePatch]]:
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


class StrategyConfigMixin:
    """
    Mixin consolidating duplicate __init__ patterns from strategy classes.
    
    Eliminates the repeated pattern of:
    - Setting up logger with strategy name
    - Initializing success/failure history
    - Setting up learned patterns dict
    
    This pattern was found duplicated across 5 strategy classes.
    """
    
    def __init__(self, name: str):
        """Initialize common strategy attributes."""
        self.name = name
        self.logger = get_logger(f"strategy.{name}")
        self.success_history: List[Dict[str, Any]] = []
        self.failure_history: List[Dict[str, Any]] = []
        self.learned_patterns: Dict[str, Any] = {}


class ValidationUtilsMixin:
    """
    Mixin providing shared validation test generation utilities.
    
    Consolidates the duplicate _generate_validation_tests implementations
    found across multiple strategy classes.
    """
    
    def _generate_validation_tests(self, issue: Issue) -> List[str]:
        """Generate basic validation test commands."""
        tests = []
        
        if issue.file:
            # Basic syntax check
            tests.append(f"python -m py_compile {issue.file}")
            
            # Import check if it's a Python file
            if issue.file.endswith('.py'):
                module_name = issue.file.replace('.py', '').replace('/', '.')
                tests.append(f"python -c \"import {module_name}\"")
        
        return tests
    
    def _extend_validation_for_symbol(self, tests: List[str], issue: Issue) -> List[str]:
        """Extend validation tests for symbol-specific checks."""
        if issue.symbol:
            tests.extend([
                f"# Test that {issue.symbol} can be called without errors",
                f"# python -c \"from {issue.file.replace('.py', '')} import {issue.symbol}; print('Import successful')\"",
                f"# TODO: Add unit tests for generated implementation of {issue.symbol}"
            ])
        return tests
    
    def _extend_validation_for_cycle(self, tests: List[str], issue: Issue) -> List[str]:
        """Extend validation tests for circular dependency checks."""
        if issue.evidence and 'cycle' in issue.evidence:
            cycle = issue.evidence['cycle']
            for module in cycle:
                tests.append(f"python -c \"import {module.replace('.py', '')}\"")
        return tests


class EstimationUtilsMixin:
    """
    Mixin providing shared time estimation utilities.
    """
    
    def _estimate_time(self, actions: List[Action]) -> float:
        """Estimate time to apply actions in seconds."""
        time_per_action = {
            "modify_file": 2.0,
            "create_file": 3.0,
            "delete_file": 1.0,
            "move_file": 2.5,
            "delete_lines": 1.5,
            "insert_lines": 2.0
        }
        
        total_time = sum(time_per_action.get(action.type, 2.0) for action in actions)
        return total_time
    
    def _calculate_complexity_score(self, actions: List[Action]) -> float:
        """Calculate complexity score based on actions."""
        base_score = len(actions)
        
        # Add complexity for different action types
        complexity_weights = {
            "modify_file": 1.0,
            "create_file": 1.5,
            "delete_file": 0.8,
            "move_file": 2.0,
            "delete_lines": 0.5,
            "insert_lines": 1.2
        }
        
        weighted_score = sum(complexity_weights.get(action.type, 1.0) for action in actions)
        return weighted_score / max(len(actions), 1)


class DependencyUtilsMixin:
    """
    Mixin providing shared dependency management utilities.
    
    Consolidates the 6 duplicate get_dependencies implementations found
    across different strategy classes.
    """
    
    def get_issue_dependencies(self, issue: Issue) -> List[str]:
        """Get dependencies specific to issue type - override in subclasses."""
        # Default dependencies based on common patterns
        dependency_map = {
            # Import-related issues have no dependencies (foundational)
            "missing_symbol": [],
            "missing_import": [],
            "import_anxiety": [],
            "unused_import": [],
            "import_error": [],
            
            # Circular dependencies have no prerequisites
            "circular_import": [],
            
            # Duplicate merging depends on imports being fixed
            "semantic_duplicate_function": ["missing_symbol", "import_error"],
            "duplicate_function": ["missing_symbol", "import_error"],
            "duplicate_class": ["missing_symbol", "import_error"],
            
            # Implementation depends on imports and merging
            "phantom_function": ["missing_symbol", "import_error", "semantic_duplicate_function"],
            "placeholder": ["missing_symbol", "import_error", "semantic_duplicate_function"],
            "todo_implementation": ["missing_symbol", "import_error", "semantic_duplicate_function"],
            "stub_function": ["missing_symbol", "import_error", "semantic_duplicate_function"],
            
            # Async fixes depend on imports
            "async_sync_mismatch": ["missing_symbol", "import_error"],
            "missing_await": ["missing_symbol", "import_error"],
            "unnecessary_await": ["missing_symbol", "import_error"],
            "async_in_sync_context": ["missing_symbol", "import_error"]
        }
        
        return dependency_map.get(issue.kind, [])


class LearningUtilsMixin:
    """
    Mixin providing shared learning and feedback utilities.
    """
    
    def learn_from_outcome(self, issue: Issue, patch: Patch, success: bool, feedback: str) -> None:
        """Learn from the outcome of applying this strategy."""
        outcome = {
            "timestamp": time.time(),
            "issue_kind": issue.kind,
            "confidence": patch.confidence,
            "risk_level": patch.risk_level.name,
            "success": success,
            "feedback": feedback,
            "actions_count": len(patch.actions)
        }
        
        if success:
            self.success_history.append(outcome)
            # Learn successful patterns
            pattern_key = f"{issue.kind}_{patch.risk_level.name}"
            if pattern_key not in self.learned_patterns:
                self.learned_patterns[pattern_key] = {
                    "success_count": 0,
                    "avg_confidence": 0.0,
                    "common_actions": []
                }
            
            pattern = self.learned_patterns[pattern_key]
            pattern["success_count"] += 1
            pattern["avg_confidence"] = (
                (pattern["avg_confidence"] * (pattern["success_count"] - 1) + patch.confidence) /
                pattern["success_count"]
            )
        else:
            self.failure_history.append(outcome)
        
        # Keep history bounded
        if len(self.success_history) > 100:
            self.success_history = self.success_history[-50:]
        if len(self.failure_history) > 100:
            self.failure_history = self.failure_history[-50:]


class BaseFixStrategy(ABC, StrategyConfigMixin, ValidationUtilsMixin, 
                     EstimationUtilsMixin, DependencyUtilsMixin, LearningUtilsMixin):
    """
    Base implementation of FixStrategy with common functionality.
    
    Consolidates duplicate patterns from fix_strategies.py:
    - 5 duplicate __init__ method patterns now use StrategyConfigMixin
    - 6 duplicate get_dependencies implementations now use DependencyUtilsMixin
    - Duplicate validation test generation now use ValidationUtilsMixin
    - Common time estimation and risk assessment use utility mixins
    """
    
    # Dependency declarations - override in subclasses
    REQUIRES_ANALYZERS: Tuple[str, ...] = ()
    REQUIRES_TOOLS: Tuple[str, ...] = ()
    REQUIRES_MODELS: Tuple[str, ...] = ()
    
    def __init__(self, name: str):
        """Initialize strategy using the consolidated mixin pattern."""
        StrategyConfigMixin.__init__(self, name)
        
    @abstractmethod
    def can_handle(self, issue: Issue) -> bool:
        """Check if this strategy can handle the given issue."""
        pass
    
    @abstractmethod
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Union[Patch, SimplePatch]]:
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