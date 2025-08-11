"""
Convergence system to prevent tail-chasing loops in fix application.

This module implements three core components:
1. ConvergenceTracker - Tracks iterations and detects loops
2. PatchValidator - Validates patches for safety and correctness  
3. FixOrchestrator - Orchestrates fix sequence with rollback capability
"""

import ast
import hashlib
import time
import difflib
import subprocess
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum
import json
import logging

from ..core.issues import Issue
from ..utils.logging_setup import get_logger


# Custom exceptions for specific error handling (no bare except blocks)
class ConvergenceError(Exception):
    """Base exception for convergence-related errors."""
    pass


class IterationLimitError(ConvergenceError):
    """Raised when iteration limit is exceeded."""
    pass


class LoopDetectedError(ConvergenceError):
    """Raised when a convergence loop is detected."""
    pass


class PatchValidationError(Exception):
    """Base exception for patch validation errors."""
    pass


class SyntaxValidationError(PatchValidationError):
    """Raised when a patch introduces syntax errors."""
    pass


class RiskThresholdError(PatchValidationError):
    """Raised when patch risk exceeds threshold."""
    pass


class OrchestrationError(Exception):
    """Base exception for orchestration errors."""
    pass


class TestFailureError(OrchestrationError):
    """Raised when tests fail after applying fixes."""
    pass


class RollbackError(OrchestrationError):
    """Raised when rollback fails."""
    pass


@dataclass
class IterationState:
    """Represents the state at a single iteration."""
    iteration: int
    timestamp: float
    issues: List[Issue]
    code_snapshot: Dict[str, str]  # file_path -> content
    error_fingerprint: str
    changes_applied: List[str]  # Descriptions of changes
    
    def compute_fingerprint(self) -> str:
        """Compute a unique fingerprint for this state."""
        issue_strs = [f"{i.kind}:{i.file}:{i.line}:{i.message}" for i in self.issues]
        code_hash = hashlib.md5(
            "|".join(f"{path}:{hash(content)}" for path, content in sorted(self.code_snapshot.items()))
            .encode()
        ).hexdigest()
        
        combined = f"{sorted(issue_strs)}:{code_hash}"
        return hashlib.sha256(combined.encode()).hexdigest()


class RiskLevel(Enum):
    """Risk levels for patches."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class PatchInfo:
    """Information about a code patch."""
    file_path: str
    original_content: str
    patched_content: str
    description: str
    issues_addressed: List[Issue]
    risk_factors: List[str] = field(default_factory=list)
    estimated_risk: RiskLevel = RiskLevel.LOW


class ConvergenceTracker:
    """
    Tracks convergence of the fix application process.
    
    Prevents infinite loops by:
    - Limiting iteration count
    - Detecting similarity in consecutive iterations
    - Caching error fingerprints to avoid repeated failures
    - Implementing exponential backoff
    """
    
    def __init__(self, max_iterations: int = 8, similarity_threshold: float = 0.05, 
                 backoff_base: float = 1.5):
        """
        Initialize convergence tracker.
        
        Args:
            max_iterations: Maximum number of fix iterations
            similarity_threshold: Minimum change required between iterations (5% default)
            backoff_base: Base for exponential backoff calculation
        """
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold
        self.backoff_base = backoff_base
        
        self.iteration_history: List[IterationState] = []
        self.error_fingerprints: Set[str] = set()
        self.failed_approaches: Dict[str, int] = {}  # approach -> failure count
        self.logger = get_logger(__name__)
        
    def should_continue(self, current_state: IterationState) -> bool:
        """
        Determine if iteration should continue.
        
        Args:
            current_state: Current iteration state
            
        Returns:
            True if should continue, False if should stop
            
        Raises:
            IterationLimitError: If max iterations exceeded
            LoopDetectedError: If loop detected
        """
        # Check iteration limit
        if current_state.iteration >= self.max_iterations:
            self.logger.warning(f"Iteration limit reached: {self.max_iterations}")
            raise IterationLimitError(f"Maximum iterations ({self.max_iterations}) exceeded")
        
        # Check for exact fingerprint match (exact loop)
        fingerprint = current_state.compute_fingerprint()
        if fingerprint in self.error_fingerprints:
            self.logger.warning(f"Loop detected via fingerprint: {fingerprint[:16]}...")
            raise LoopDetectedError(f"Detected exact state repetition at iteration {current_state.iteration}")
        
        # Check for insufficient progress
        if len(self.iteration_history) >= 2:
            if self._is_insufficient_progress(current_state):
                self.logger.warning("Insufficient progress detected between iterations")
                raise LoopDetectedError("Insufficient progress between iterations")
        
        # Record this state
        self.error_fingerprints.add(fingerprint)
        self.iteration_history.append(current_state)
        
        self.logger.info(f"Continuing iteration {current_state.iteration}")
        return True
    
    def _is_insufficient_progress(self, current: IterationState) -> bool:
        """Check if progress between iterations is insufficient."""
        if not self.iteration_history:
            return False
            
        previous = self.iteration_history[-1]
        
        # Compare issue counts
        prev_issue_count = len(previous.issues)
        curr_issue_count = len(current.issues)
        
        if prev_issue_count == 0:
            return False  # Already converged
        
        # Calculate progress ratio
        if curr_issue_count >= prev_issue_count:
            progress_ratio = 0.0  # No progress or regression
        else:
            progress_ratio = (prev_issue_count - curr_issue_count) / prev_issue_count
        
        # Check if progress is below threshold
        insufficient = progress_ratio < self.similarity_threshold
        
        if insufficient:
            self.logger.debug(f"Insufficient progress: {progress_ratio:.3f} < {self.similarity_threshold}")
        
        return insufficient
    
    def calculate_backoff_delay(self, failure_count: int) -> float:
        """Calculate exponential backoff delay."""
        delay = self.backoff_base ** failure_count
        self.logger.debug(f"Backoff delay for {failure_count} failures: {delay:.2f}s")
        return delay
    
    def record_failure(self, approach: str) -> None:
        """Record a failed approach for backoff calculation."""
        self.failed_approaches[approach] = self.failed_approaches.get(approach, 0) + 1
        self.logger.warning(f"Recorded failure for approach '{approach}': {self.failed_approaches[approach]} times")
    
    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get metrics about the convergence process."""
        if not self.iteration_history:
            return {"status": "not_started"}
        
        latest = self.iteration_history[-1]
        initial = self.iteration_history[0]
        
        return {
            "status": "in_progress",
            "iterations": len(self.iteration_history),
            "max_iterations": self.max_iterations,
            "initial_issues": len(initial.issues),
            "current_issues": len(latest.issues),
            "issues_resolved": max(0, len(initial.issues) - len(latest.issues)),
            "unique_states": len(self.error_fingerprints),
            "failed_approaches": dict(self.failed_approaches),
            "total_time": latest.timestamp - initial.timestamp if len(self.iteration_history) > 1 else 0
        }


class PatchValidator:
    """
    Validates patches for syntax correctness and risk assessment.
    
    Ensures patches don't introduce syntax errors or reintroduce issues.
    """
    
    def __init__(self, max_risk_level: RiskLevel = RiskLevel.HIGH):
        """
        Initialize patch validator.
        
        Args:
            max_risk_level: Maximum acceptable risk level
        """
        self.max_risk_level = max_risk_level
        self.logger = get_logger(__name__)
        
    def validate_patch(self, patch: PatchInfo, previous_issues: List[Issue]) -> bool:
        """
        Validate a patch for safety and correctness.
        
        Args:
            patch: Patch information
            previous_issues: Issues from previous iterations
            
        Returns:
            True if patch is valid
            
        Raises:
            SyntaxValidationError: If patch introduces syntax errors
            RiskThresholdError: If patch risk is too high
        """
        self.logger.info(f"Validating patch for {patch.file_path}")
        
        # Validate syntax
        try:
            self._validate_syntax(patch)
            self.logger.debug("Syntax validation passed")
        except SyntaxError as e:
            self.logger.error(f"Syntax validation failed: {e}")
            raise SyntaxValidationError(f"Patch introduces syntax error: {e}")
        
        # Estimate risk
        risk_level = self._estimate_risk(patch, previous_issues)
        patch.estimated_risk = risk_level
        
        if risk_level.value > self.max_risk_level.value:
            self.logger.error(f"Patch risk too high: {risk_level} > {self.max_risk_level}")
            raise RiskThresholdError(f"Patch risk ({risk_level}) exceeds threshold ({self.max_risk_level})")
        
        self.logger.info(f"Patch validation successful (risk: {risk_level})")
        return True
    
    def _validate_syntax(self, patch: PatchInfo) -> None:
        """
        Validate that patch doesn't introduce syntax errors.
        
        Args:
            patch: Patch to validate
            
        Raises:
            SyntaxError: If syntax is invalid
        """
        try:
            # Parse the patched content as Python code
            ast.parse(patch.patched_content)
        except SyntaxError as e:
            # Re-raise with more context
            raise SyntaxError(f"Invalid syntax in {patch.file_path} line {e.lineno}: {e.msg}")
    
    def _estimate_risk(self, patch: PatchInfo, previous_issues: List[Issue]) -> RiskLevel:
        """
        Estimate the risk level of applying this patch.
        
        Args:
            patch: Patch to assess
            previous_issues: Issues from previous iterations
            
        Returns:
            Estimated risk level
        """
        risk_factors = []
        risk_score = 0
        
        # Analyze changes
        diff_lines = list(difflib.unified_diff(
            patch.original_content.splitlines(keepends=True),
            patch.patched_content.splitlines(keepends=True),
            lineterm=""
        ))
        
        additions = sum(1 for line in diff_lines if line.startswith('+'))
        deletions = sum(1 for line in diff_lines if line.startswith('-'))
        
        # Risk factor: Large changes
        if additions + deletions > 50:
            risk_factors.append("Large change set")
            risk_score += 2
        elif additions + deletions > 20:
            risk_factors.append("Medium change set")
            risk_score += 1
        
        # Risk factor: Core functionality changes
        critical_patterns = ['__init__', 'main', 'execute', 'run', 'process']
        if any(pattern in patch.patched_content.lower() for pattern in critical_patterns):
            risk_factors.append("Core functionality modified")
            risk_score += 1
        
        # Risk factor: Import changes
        if 'import ' in patch.patched_content and 'import ' not in patch.original_content:
            risk_factors.append("New imports added")
            risk_score += 1
        
        # Risk factor: Exception handling changes
        if 'except' in patch.patched_content or 'raise' in patch.patched_content:
            risk_factors.append("Exception handling modified")
            risk_score += 1
        
        # Risk factor: Might reintroduce previous issues
        if self._might_reintroduce_issues(patch, previous_issues):
            risk_factors.append("Might reintroduce previous issues")
            risk_score += 2
        
        patch.risk_factors = risk_factors
        
        # Convert score to risk level
        if risk_score >= 5:
            return RiskLevel.CRITICAL
        elif risk_score >= 3:
            return RiskLevel.HIGH
        elif risk_score >= 1:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _might_reintroduce_issues(self, patch: PatchInfo, previous_issues: List[Issue]) -> bool:
        """Check if patch might reintroduce previous issues."""
        # Simple heuristic: if we're adding back similar patterns
        for issue in previous_issues:
            if issue.file == patch.file_path:
                # Check if we're adding back similar problematic patterns
                if (issue.kind == "phantom_function" and 
                    ("pass" in patch.patched_content or "NotImplementedError" in patch.patched_content)):
                    return True
                elif (issue.kind == "duplicate_function" and 
                      any(word in patch.patched_content for word in ["def ", "class "])):
                    return True
        
        return False


@dataclass 
class FixPlan:
    """Represents a planned sequence of fixes."""
    fixes: List[PatchInfo]
    dependency_order: List[int]  # Indices into fixes list
    rollback_points: List[int]   # Iteration numbers where rollback is safe
    estimated_duration: float
    confidence_score: float


@dataclass
class DecisionLogEntry:
    """Records a decision made by the orchestrator."""
    timestamp: float
    decision_type: str  # "apply_fix", "skip_fix", "rollback", etc.
    rationale: str
    context: Dict[str, Any]
    outcome: Optional[str] = None  # Set after decision outcome is known


class FixOrchestrator:
    """
    Orchestrates the application of fixes with dependency management and rollback capability.
    
    Plans fix sequences, validates dependencies, and rolls back on failures.
    """
    
    def __init__(self, convergence_tracker: ConvergenceTracker, 
                 patch_validator: PatchValidator, test_command: Optional[str] = None):
        """
        Initialize fix orchestrator.
        
        Args:
            convergence_tracker: Tracker for convergence monitoring
            patch_validator: Validator for patch safety
            test_command: Command to run tests for validation
        """
        self.convergence_tracker = convergence_tracker
        self.patch_validator = patch_validator
        self.test_command = test_command
        
        self.applied_fixes: List[PatchInfo] = []
        self.decision_log: List[DecisionLogEntry] = []
        self.rollback_snapshots: Dict[int, Dict[str, str]] = {}  # iteration -> file snapshots
        
        self.logger = get_logger(__name__)
    
    def execute_fix_sequence(self, fixes: List[PatchInfo], 
                           working_directory: Path) -> Tuple[bool, List[str]]:
        """
        Execute a sequence of fixes with validation and rollback capability.
        
        Args:
            fixes: List of fixes to apply
            working_directory: Directory where files are located
            
        Returns:
            (success, messages) tuple
            
        Raises:
            OrchestrationError: If orchestration fails
            TestFailureError: If tests fail
            RollbackError: If rollback fails
        """
        self.logger.info(f"Executing fix sequence with {len(fixes)} fixes")
        messages = []
        
        try:
            # Plan the fix sequence
            plan = self._create_fix_plan(fixes)
            self.logger.info(f"Created fix plan with {len(plan.fixes)} fixes")
            
            # Create initial snapshot for rollback
            self._create_rollback_snapshot(0, working_directory)
            
            # Apply fixes in dependency order
            for i, fix_index in enumerate(plan.dependency_order):
                fix = plan.fixes[fix_index]
                
                # Log decision
                self._log_decision("apply_fix", f"Applying fix {i+1}/{len(plan.dependency_order)}", {
                    "fix_description": fix.description,
                    "file": fix.file_path,
                    "risk_level": fix.estimated_risk.name
                })
                
                try:
                    # Validate patch
                    previous_issues = self._get_previous_issues()
                    self.patch_validator.validate_patch(fix, previous_issues)
                    
                    # Apply the fix
                    self._apply_single_fix(fix, working_directory)
                    self.applied_fixes.append(fix)
                    
                    messages.append(f"Applied fix: {fix.description}")
                    self.logger.info(f"Successfully applied fix {i+1}: {fix.description}")
                    
                    # Create rollback point if this is a safe checkpoint
                    if i in plan.rollback_points:
                        self._create_rollback_snapshot(i + 1, working_directory)
                        self.logger.debug(f"Created rollback snapshot at iteration {i + 1}")
                    
                    # Run tests if configured
                    if self.test_command and i % 3 == 0:  # Test every 3 fixes
                        if not self._run_tests(working_directory):
                            raise TestFailureError(f"Tests failed after applying fix {i+1}")
                
                except (PatchValidationError, TestFailureError) as e:
                    self.logger.error(f"Fix application failed: {e}")
                    # Record failure and attempt rollback
                    self.convergence_tracker.record_failure(f"fix_{fix.description}")
                    
                    self._log_decision("rollback", f"Rolling back due to: {e}", {
                        "failed_fix": fix.description,
                        "error": str(e)
                    })
                    
                    # Rollback to last safe point
                    rollback_point = self._find_rollback_point(i)
                    self._perform_rollback(rollback_point, working_directory)
                    
                    # Determine if we should continue or abort
                    if isinstance(e, TestFailureError):
                        messages.append(f"Rolled back due to test failure: {fix.description}")
                        continue  # Try next fix
                    else:
                        raise  # Validation errors are fatal
            
            # Final test run
            if self.test_command:
                if not self._run_tests(working_directory):
                    raise TestFailureError("Final tests failed after applying all fixes")
            
            self.logger.info("Fix sequence executed successfully")
            return True, messages
            
        except (TestFailureError, RollbackError) as e:
            self.logger.error(f"Fix sequence failed: {e}")
            messages.append(f"Fix sequence failed: {e}")
            return False, messages
        except Exception as e:
            self.logger.error(f"Unexpected error in fix orchestration: {e}", exc_info=True)
            raise OrchestrationError(f"Orchestration failed: {e}")
    
    def _create_fix_plan(self, fixes: List[PatchInfo]) -> FixPlan:
        """Create an execution plan for fixes."""
        # Simple dependency ordering - more sophisticated logic could be added
        dependency_order = list(range(len(fixes)))
        
        # Sort by risk (low risk first)
        dependency_order.sort(key=lambda i: fixes[i].estimated_risk.value)
        
        # Create rollback points every 3 fixes
        rollback_points = [i for i in range(0, len(fixes), 3)]
        
        return FixPlan(
            fixes=fixes,
            dependency_order=dependency_order,
            rollback_points=rollback_points,
            estimated_duration=len(fixes) * 10.0,  # Rough estimate
            confidence_score=0.8  # Default confidence
        )
    
    def _create_rollback_snapshot(self, iteration: int, working_directory: Path) -> None:
        """Create a snapshot of all files for rollback."""
        snapshot = {}
        
        # Snapshot all Python files in directory
        for file_path in working_directory.rglob("*.py"):
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    snapshot[str(file_path)] = f.read()
            except (IOError, UnicodeDecodeError) as e:
                self.logger.warning(f"Failed to snapshot {file_path}: {e}")
        
        self.rollback_snapshots[iteration] = snapshot
        self.logger.debug(f"Created snapshot with {len(snapshot)} files")
    
    def _apply_single_fix(self, fix: PatchInfo, working_directory: Path) -> None:
        """Apply a single fix to the filesystem."""
        target_path = working_directory / fix.file_path
        
        try:
            # Write the patched content
            with open(target_path, 'w', encoding='utf-8') as f:
                f.write(fix.patched_content)
            
            self.logger.debug(f"Applied fix to {target_path}")
            
        except IOError as e:
            raise OrchestrationError(f"Failed to write fix to {target_path}: {e}")
    
    def _run_tests(self, working_directory: Path) -> bool:
        """Run tests to validate changes."""
        if not self.test_command:
            return True
        
        try:
            self.logger.info(f"Running tests: {self.test_command}")
            result = subprocess.run(
                self.test_command, 
                shell=True, 
                cwd=working_directory,
                capture_output=True, 
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            success = result.returncode == 0
            if not success:
                self.logger.warning(f"Tests failed with exit code {result.returncode}")
                self.logger.debug(f"Test output: {result.stdout}")
                self.logger.debug(f"Test errors: {result.stderr}")
            
            return success
            
        except subprocess.TimeoutExpired:
            self.logger.error("Test execution timed out")
            return False
        except subprocess.SubprocessError as e:
            self.logger.error(f"Test execution failed: {e}")
            return False
    
    def _find_rollback_point(self, failed_iteration: int) -> int:
        """Find the most recent rollback point."""
        for point in reversed(sorted(self.rollback_snapshots.keys())):
            if point <= failed_iteration:
                return point
        return 0  # Rollback to beginning
    
    def _perform_rollback(self, rollback_point: int, working_directory: Path) -> None:
        """Perform rollback to a previous snapshot."""
        if rollback_point not in self.rollback_snapshots:
            raise RollbackError(f"No snapshot available for rollback point {rollback_point}")
        
        snapshot = self.rollback_snapshots[rollback_point]
        
        try:
            for file_path, content in snapshot.items():
                path = Path(file_path)
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(content)
            
            # Remove applied fixes after rollback point
            self.applied_fixes = self.applied_fixes[:rollback_point]
            
            self.logger.info(f"Rolled back to iteration {rollback_point}")
            
        except IOError as e:
            raise RollbackError(f"Failed to rollback to point {rollback_point}: {e}")
    
    def _get_previous_issues(self) -> List[Issue]:
        """Get issues from previous iterations."""
        if self.convergence_tracker.iteration_history:
            return self.convergence_tracker.iteration_history[-1].issues
        return []
    
    def _log_decision(self, decision_type: str, rationale: str, context: Dict[str, Any]) -> None:
        """Log a decision made by the orchestrator."""
        entry = DecisionLogEntry(
            timestamp=time.time(),
            decision_type=decision_type,
            rationale=rationale,
            context=context
        )
        self.decision_log.append(entry)
        self.logger.debug(f"Decision logged: {decision_type} - {rationale}")
    
    def get_decision_log(self) -> List[Dict[str, Any]]:
        """Get the decision log in serializable format."""
        return [
            {
                "timestamp": entry.timestamp,
                "decision_type": entry.decision_type,
                "rationale": entry.rationale,
                "context": entry.context,
                "outcome": entry.outcome
            }
            for entry in self.decision_log
        ]
    
    def export_session_data(self, output_path: Path) -> None:
        """Export session data for analysis."""
        session_data = {
            "convergence_metrics": self.convergence_tracker.get_convergence_metrics(),
            "applied_fixes": [
                {
                    "description": fix.description,
                    "file_path": fix.file_path,
                    "risk_level": fix.estimated_risk.name,
                    "risk_factors": fix.risk_factors,
                    "issues_addressed": len(fix.issues_addressed)
                }
                for fix in self.applied_fixes
            ],
            "decision_log": self.get_decision_log()
        }
        
        try:
            with open(output_path, 'w') as f:
                json.dump(session_data, f, indent=2)
            self.logger.info(f"Session data exported to {output_path}")
        except IOError as e:
            self.logger.error(f"Failed to export session data: {e}")


# Factory function for easy setup
def create_convergence_system(max_iterations: int = 8, 
                            max_risk_level: RiskLevel = RiskLevel.HIGH,
                            test_command: Optional[str] = None) -> Tuple[ConvergenceTracker, PatchValidator, FixOrchestrator]:
    """
    Create a complete convergence system.
    
    Args:
        max_iterations: Maximum fix iterations
        max_risk_level: Maximum acceptable patch risk
        test_command: Command to run tests
        
    Returns:
        (tracker, validator, orchestrator) tuple
    """
    tracker = ConvergenceTracker(max_iterations=max_iterations)
    validator = PatchValidator(max_risk_level=max_risk_level)
    orchestrator = FixOrchestrator(tracker, validator, test_command)
    
    return tracker, validator, orchestrator