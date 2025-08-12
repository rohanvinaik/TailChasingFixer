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
import numpy as np
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple, Any, Protocol, Union
from dataclasses import dataclass, field
from enum import Enum
import json
import logging
from collections import defaultdict

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
class ReplicationTiming:
    """Represents replication timing characteristics for a module."""
    module_path: str
    rt_score: float  # Overall replication timing score
    git_churn: float  # Git commit frequency/churn rate
    test_coverage: float  # Test coverage density
    runtime_reach: float  # Runtime reachability
    early_replication: bool  # True if high-priority for early fixing
    tad_membership: Optional[str] = None  # TAD identifier
    
    def __post_init__(self):
        """Calculate derived properties."""
        # Early replication = high churn + high coverage + high reach
        self.early_replication = (
            self.git_churn > 0.6 and 
            self.test_coverage > 0.5 and 
            self.runtime_reach > 0.4
        )


@dataclass
class IterationState:
    """Represents the state at a single iteration."""
    iteration: int
    timestamp: float
    issues: List[Issue]
    code_snapshot: Dict[str, str]  # file_path -> content
    error_fingerprint: str
    changes_applied: List[str]  # Descriptions of changes
    rt_scores: Dict[str, ReplicationTiming] = field(default_factory=dict)  # module -> RT scores
    rt_evolution: List[float] = field(default_factory=list)  # Track RT score changes over time
    
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
        
        Enhanced with RT evolution tracking and adaptive convergence detection.
        
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
        
        # Track RT evolution for adaptive convergence
        self._track_rt_evolution(current_state)
        
        # Check for insufficient progress (enhanced with RT metrics)
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
    
    def _track_rt_evolution(self, current_state: IterationState) -> None:
        """
        Track replication timing evolution across iterations.
        
        Monitors how RT scores change over time to detect convergence patterns
        and optimize future scheduling decisions.
        """
        if not current_state.rt_scores:
            return
        
        # Calculate average RT score for this iteration
        avg_rt_score = np.mean([rt.rt_score for rt in current_state.rt_scores.values()])
        current_state.rt_evolution.append(avg_rt_score)
        
        # Log RT evolution trends
        if len(self.iteration_history) > 0:
            prev_state = self.iteration_history[-1]
            if prev_state.rt_evolution:
                prev_avg = prev_state.rt_evolution[-1]
                rt_change = avg_rt_score - prev_avg
                
                self.logger.debug(f"RT evolution: {avg_rt_score:.3f} "
                                f"(change: {rt_change:+.3f})")
                
                # Detect RT score stagnation
                if len(current_state.rt_evolution) >= 3:
                    recent_scores = current_state.rt_evolution[-3:]
                    rt_variance = np.var(recent_scores)
                    
                    if rt_variance < 0.001:  # Very low variance
                        self.logger.info("RT score stagnation detected - convergence likely")
    
    def _is_insufficient_progress(self, current: IterationState) -> bool:
        """
        Check if progress between iterations is insufficient.
        
        Enhanced with RT evolution analysis for better convergence detection.
        """
        if not self.iteration_history:
            return False
            
        previous = self.iteration_history[-1]
        
        # Compare issue counts (traditional metric)
        prev_issue_count = len(previous.issues)
        curr_issue_count = len(current.issues)
        
        if prev_issue_count == 0:
            return False  # Already converged
        
        # Calculate traditional progress ratio
        if curr_issue_count >= prev_issue_count:
            progress_ratio = 0.0  # No progress or regression
        else:
            progress_ratio = (prev_issue_count - curr_issue_count) / prev_issue_count
        
        # Enhanced progress analysis with RT metrics
        rt_progress_factor = self._analyze_rt_progress(previous, current)
        
        # Combined progress metric (weighted average)
        combined_progress = 0.7 * progress_ratio + 0.3 * rt_progress_factor
        
        # Check if progress is below threshold
        insufficient = combined_progress < self.similarity_threshold
        
        if insufficient:
            self.logger.debug(f"Insufficient progress: combined={combined_progress:.3f} "
                            f"(issues={progress_ratio:.3f}, rt={rt_progress_factor:.3f}) "
                            f"< {self.similarity_threshold}")
        
        return insufficient
    
    def _analyze_rt_progress(self, previous: IterationState, current: IterationState) -> float:
        """
        Analyze replication timing progress between iterations.
        
        Returns normalized progress factor (0.0-1.0) based on RT metrics.
        """
        if not previous.rt_scores or not current.rt_scores:
            return 0.5  # Neutral if no RT data
        
        # Compare RT score distributions
        prev_scores = [rt.rt_score for rt in previous.rt_scores.values()]
        curr_scores = [rt.rt_score for rt in current.rt_scores.values()]
        
        if not prev_scores or not curr_scores:
            return 0.5
        
        # Calculate score improvements
        prev_avg = np.mean(prev_scores)
        curr_avg = np.mean(curr_scores)
        
        # Calculate early replication improvements
        prev_early = sum(1 for rt in previous.rt_scores.values() if rt.early_replication)
        curr_early = sum(1 for rt in current.rt_scores.values() if rt.early_replication)
        
        early_improvement = (curr_early - prev_early) / max(1, len(current.rt_scores))
        score_improvement = max(0, curr_avg - prev_avg) / max(0.1, prev_avg)
        
        # Combined RT progress factor
        rt_progress = 0.6 * score_improvement + 0.4 * early_improvement
        
        # Normalize to 0-1 range
        return min(1.0, max(0.0, rt_progress))

    def get_convergence_metrics(self) -> Dict[str, Any]:
        """Get metrics about the convergence process."""
        if not self.iteration_history:
            return {"status": "not_started"}
        
        latest = self.iteration_history[-1]
        initial = self.iteration_history[0]
        
        metrics = {
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
        
        # Add RT evolution metrics if available
        if latest.rt_evolution:
            metrics["rt_evolution"] = {
                "rt_scores": latest.rt_evolution,
                "rt_trend": self._calculate_rt_trend(),
                "rt_stability": self._calculate_rt_stability(),
                "avg_rt_score": latest.rt_evolution[-1] if latest.rt_evolution else 0.0
            }
        
        return metrics
    
    def _calculate_rt_trend(self) -> str:
        """Calculate overall RT trend across iterations."""
        if len(self.iteration_history) < 2:
            return "insufficient_data"
        
        all_scores = []
        for state in self.iteration_history:
            all_scores.extend(state.rt_evolution)
        
        if len(all_scores) < 2:
            return "insufficient_data"
        
        # Simple linear trend analysis
        x = np.arange(len(all_scores))
        slope = np.polyfit(x, all_scores, 1)[0]
        
        if slope > 0.01:
            return "improving"
        elif slope < -0.01:
            return "declining"
        else:
            return "stable"
    
    def _calculate_rt_stability(self) -> float:
        """Calculate RT stability metric (lower variance = more stable)."""
        if len(self.iteration_history) < 2:
            return 0.0
        
        all_scores = []
        for state in self.iteration_history[-3:]:  # Last 3 iterations
            all_scores.extend(state.rt_evolution)
        
        if len(all_scores) < 2:
            return 0.0
        
        # Return inverse of coefficient of variation (normalized stability)
        mean_score = np.mean(all_scores)
        if mean_score == 0:
            return 1.0
        
        cv = np.std(all_scores) / mean_score
        return max(0.0, 1.0 - cv)  # Higher = more stable


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


class ReplicationTimingScheduler:
    """
    Replication timing-based fix scheduler using biological DNA replication metaphors.
    
    Implements sophisticated scheduling based on module "replication timing" characteristics:
    - Early replication modules (high churn, high coverage, high reach) get priority
    - TAD-aware scheduling prevents cross-compartment fix interleaving  
    - Git history analysis provides churn metrics
    - Test coverage analysis provides stability metrics
    - Runtime reachability analysis provides impact metrics
    """
    
    def __init__(self, git_analyzer=None, chromatin_analyzer=None, 
                 lambda1: float = 0.4, lambda2: float = 0.3, lambda3: float = 0.3):
        """
        Initialize replication timing scheduler.
        
        Args:
            git_analyzer: GitChainAnalyzer for git history analysis
            chromatin_analyzer: ChromatinContactAnalyzer for TAD detection
            lambda1: Weight for git churn component
            lambda2: Weight for test coverage component  
            lambda3: Weight for runtime reach component
        """
        self.git_analyzer = git_analyzer
        self.chromatin_analyzer = chromatin_analyzer
        self.lambda1 = lambda1  # Git churn weight
        self.lambda2 = lambda2  # Test coverage weight
        self.lambda3 = lambda3  # Runtime reach weight
        
        self.logger = get_logger(__name__)
        self._rt_cache: Dict[str, ReplicationTiming] = {}
        self._tad_cache: Dict[str, str] = {}  # module -> TAD mapping
        
    def compute_replication_timing(self, module_path: str, 
                                 working_directory: Path,
                                 context: Optional[Dict[str, Any]] = None) -> ReplicationTiming:
        """
        Compute replication timing score for a module.
        
        RT(m) = λ₁*git_churn(m) + λ₂*test_coverage(m) + λ₃*runtime_reach(m)
        
        Args:
            module_path: Path to the module
            working_directory: Project root directory
            context: Additional context data
            
        Returns:
            ReplicationTiming object with computed metrics
        """
        if module_path in self._rt_cache:
            return self._rt_cache[module_path]
        
        self.logger.debug(f"Computing replication timing for {module_path}")
        
        # Compute individual components
        git_churn = self._compute_git_churn(module_path, working_directory)
        test_coverage = self._compute_test_coverage(module_path, working_directory)
        runtime_reach = self._compute_runtime_reach(module_path, working_directory, context)
        
        # Calculate overall RT score
        rt_score = (
            self.lambda1 * git_churn +
            self.lambda2 * test_coverage +
            self.lambda3 * runtime_reach
        )
        
        # Determine TAD membership
        tad_membership = self._get_tad_membership(module_path)
        
        rt = ReplicationTiming(
            module_path=module_path,
            rt_score=rt_score,
            git_churn=git_churn,
            test_coverage=test_coverage,
            runtime_reach=runtime_reach,
            early_replication=False,  # Set in __post_init__
            tad_membership=tad_membership
        )
        
        self._rt_cache[module_path] = rt
        self.logger.debug(f"RT score for {module_path}: {rt_score:.3f} "
                         f"(churn:{git_churn:.2f}, cov:{test_coverage:.2f}, reach:{runtime_reach:.2f})")
        
        return rt
    
    def prioritize_fixes(self, issues: List[Issue], 
                        working_directory: Path,
                        context: Optional[Dict[str, Any]] = None) -> List[Issue]:
        """
        Prioritize fixes based on replication timing scores.
        
        Scheduling strategy:
        1. Sort by early-RT (high churn, high coverage, high reach) first
        2. Group related fixes by TAD membership
        3. Prevent cross-TAD fix interleaving
        4. Within TADs, sort by RT score descending
        
        Args:
            issues: List of issues to prioritize
            working_directory: Project root directory
            context: Additional context data
            
        Returns:
            Prioritized list of issues
        """
        self.logger.info(f"Prioritizing {len(issues)} fixes using replication timing")
        
        # Compute RT scores for all affected modules
        rt_scores = {}
        for issue in issues:
            if issue.file:
                module_path = self._file_to_module_path(issue.file)
                if module_path not in rt_scores:
                    rt_scores[module_path] = self.compute_replication_timing(
                        module_path, working_directory, context
                    )
        
        # Group issues by TAD
        tad_groups = self._group_issues_by_tad(issues, rt_scores)
        
        # Prioritize TADs by average RT score
        tad_priorities = self._prioritize_tads(tad_groups, rt_scores)
        
        # Schedule fixes within each TAD
        prioritized_issues = []
        for tad_id in tad_priorities:
            tad_issues = tad_groups[tad_id]
            
            # Sort issues within TAD by RT score (early replication first)
            tad_issues.sort(key=lambda issue: (
                -rt_scores[self._file_to_module_path(issue.file or '')].rt_score,
                -rt_scores[self._file_to_module_path(issue.file or '')].early_replication,
                issue.confidence  # Secondary sort by confidence
            ))
            
            prioritized_issues.extend(tad_issues)
            
            self.logger.debug(f"TAD {tad_id}: scheduled {len(tad_issues)} fixes")
        
        self.logger.info(f"Completed replication timing-based prioritization")
        return prioritized_issues
    
    def _compute_git_churn(self, module_path: str, working_directory: Path) -> float:
        """
        Compute git churn rate for a module.
        
        High churn indicates active development requiring early attention.
        
        Args:
            module_path: Module path
            working_directory: Project root
            
        Returns:
            Normalized churn score (0.0-1.0)
        """
        if not self.git_analyzer:
            # Fallback: analyze git history directly
            return self._analyze_git_churn_direct(module_path, working_directory)
        
        try:
            # Use GitChainAnalyzer if available
            file_path = self._module_to_file_path(module_path)
            
            # Get recent commits affecting this file
            result = subprocess.run(
                ["git", "log", "--oneline", "--since=30.days.ago", "--", file_path],
                cwd=working_directory,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return 0.0
            
            commit_count = len([line for line in result.stdout.strip().split('\n') if line])
            
            # Normalize to 0-1 scale (30+ commits = 1.0)
            return min(1.0, commit_count / 30.0)
            
        except (subprocess.SubprocessError, FileNotFoundError):
            self.logger.warning(f"Could not analyze git churn for {module_path}")
            return 0.0
    
    def _compute_test_coverage(self, module_path: str, working_directory: Path) -> float:
        """
        Compute test coverage density for a module.
        
        High coverage indicates stable, well-tested code that can be fixed safely.
        
        Args:
            module_path: Module path
            working_directory: Project root
            
        Returns:
            Normalized coverage score (0.0-1.0)
        """
        try:
            # Look for corresponding test files
            test_patterns = [
                f"test_{module_path.split('.')[-1]}.py",
                f"{module_path.split('.')[-1]}_test.py",
                f"tests/test_{module_path.split('.')[-1]}.py"
            ]
            
            test_file_count = 0
            total_test_lines = 0
            
            for pattern in test_patterns:
                test_path = working_directory / pattern
                if test_path.exists():
                    test_file_count += 1
                    try:
                        with open(test_path, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                            # Count test function lines
                            test_lines = sum(1 for line in lines if 'def test_' in line or 'assert' in line)
                            total_test_lines += test_lines
                    except (IOError, UnicodeDecodeError):
                        continue
            
            # Estimate coverage based on test file presence and test density
            if test_file_count == 0:
                return 0.0
            
            # Base score from file count, bonus from test density
            base_score = min(0.5, test_file_count * 0.25)
            density_bonus = min(0.5, total_test_lines / 50.0)  # 50+ test lines = max bonus
            
            return base_score + density_bonus
            
        except Exception as e:
            self.logger.warning(f"Could not analyze test coverage for {module_path}: {e}")
            return 0.0
    
    def _compute_runtime_reach(self, module_path: str, working_directory: Path,
                              context: Optional[Dict[str, Any]]) -> float:
        """
        Compute runtime reachability for a module.
        
        High reach indicates modules that affect many other parts of the system.
        
        Args:
            module_path: Module path
            working_directory: Project root
            context: Additional context with import graph data
            
        Returns:
            Normalized reach score (0.0-1.0)
        """
        try:
            # Use import graph from context if available
            if context and 'import_graph' in context:
                import_graph = context['import_graph']
                
                if module_path in import_graph:
                    # Count incoming dependencies (how many modules depend on this one)
                    incoming = len([node for node in import_graph if module_path in import_graph.get(node, [])])
                    # Count outgoing dependencies (how many modules this one depends on)
                    outgoing = len(import_graph.get(module_path, []))
                    
                    # High incoming = high impact, moderate outgoing = stable
                    reach_score = (incoming * 0.7 + outgoing * 0.3) / 20.0  # Normalize
                    return min(1.0, reach_score)
            
            # Fallback: analyze imports in the file directly
            file_path = working_directory / self._module_to_file_path(module_path)
            if not file_path.exists():
                return 0.0
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Count import statements as proxy for reach
            import_count = content.count('import ') + content.count('from ')
            export_indicators = content.count('def ') + content.count('class ') + content.count('__all__')
            
            # High exports = high potential reach
            reach_score = (export_indicators * 0.6 + import_count * 0.4) / 30.0
            return min(1.0, reach_score)
            
        except Exception as e:
            self.logger.warning(f"Could not analyze runtime reach for {module_path}: {e}")
            return 0.0
    
    def _get_tad_membership(self, module_path: str) -> Optional[str]:
        """Get TAD membership for a module."""
        if module_path in self._tad_cache:
            return self._tad_cache[module_path]
        
        if not self.chromatin_analyzer or not hasattr(self.chromatin_analyzer, '_tads'):
            # Fallback: use package-based TAD
            package = '.'.join(module_path.split('.')[:-1]) if '.' in module_path else 'root'
            tad_id = f"TAD_{package}"
            self._tad_cache[module_path] = tad_id
            return tad_id
        
        # Find TAD containing this module
        for tad_id, tad in self.chromatin_analyzer._tads.items():
            if module_path in tad.modules:
                self._tad_cache[module_path] = tad_id
                return tad_id
        
        # Fallback if not found
        default_tad = "TAD_unknown"
        self._tad_cache[module_path] = default_tad
        return default_tad
    
    def _group_issues_by_tad(self, issues: List[Issue], 
                           rt_scores: Dict[str, ReplicationTiming]) -> Dict[str, List[Issue]]:
        """Group issues by TAD membership."""
        tad_groups = defaultdict(list)
        
        for issue in issues:
            if issue.file:
                module_path = self._file_to_module_path(issue.file)
                if module_path in rt_scores:
                    tad_id = rt_scores[module_path].tad_membership or "TAD_unknown"
                    tad_groups[tad_id].append(issue)
                else:
                    tad_groups["TAD_unknown"].append(issue)
            else:
                tad_groups["TAD_unknown"].append(issue)
        
        return dict(tad_groups)
    
    def _prioritize_tads(self, tad_groups: Dict[str, List[Issue]], 
                        rt_scores: Dict[str, ReplicationTiming]) -> List[str]:
        """Prioritize TADs by average RT score."""
        tad_priorities = []
        
        for tad_id, issues in tad_groups.items():
            # Calculate average RT score for this TAD
            total_score = 0.0
            count = 0
            
            for issue in issues:
                if issue.file:
                    module_path = self._file_to_module_path(issue.file)
                    if module_path in rt_scores:
                        total_score += rt_scores[module_path].rt_score
                        count += 1
            
            avg_score = total_score / count if count > 0 else 0.0
            tad_priorities.append((tad_id, avg_score))
        
        # Sort by average score (highest first)
        tad_priorities.sort(key=lambda x: x[1], reverse=True)
        
        return [tad_id for tad_id, _ in tad_priorities]
    
    def _analyze_git_churn_direct(self, module_path: str, working_directory: Path) -> float:
        """Direct git churn analysis fallback."""
        try:
            file_path = self._module_to_file_path(module_path)
            
            # Get commit count for this file in last 30 days
            result = subprocess.run(
                ["git", "rev-list", "--count", "--since=30.days.ago", "HEAD", "--", file_path],
                cwd=working_directory,
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                commit_count = int(result.stdout.strip() or '0')
                return min(1.0, commit_count / 20.0)  # Normalize
            
        except (subprocess.SubprocessError, ValueError):
            pass
        
        return 0.0
    
    def _file_to_module_path(self, file_path: str) -> str:
        """Convert file path to module path."""
        if file_path.endswith('.py'):
            module_path = file_path[:-3].replace('/', '.')
            return module_path
        return file_path
    
    def _module_to_file_path(self, module_path: str) -> str:
        """Convert module path to file path."""
        return module_path.replace('.', '/') + '.py'
    
    def get_rt_summary(self, rt_scores: Dict[str, ReplicationTiming]) -> Dict[str, Any]:
        """Get summary statistics of RT scores."""
        if not rt_scores:
            return {"status": "no_scores"}
        
        scores = [rt.rt_score for rt in rt_scores.values()]
        early_replication_count = sum(1 for rt in rt_scores.values() if rt.early_replication)
        
        tad_distribution = defaultdict(int)
        for rt in rt_scores.values():
            tad_distribution[rt.tad_membership or 'unknown'] += 1
        
        return {
            "total_modules": len(rt_scores),
            "avg_rt_score": np.mean(scores),
            "median_rt_score": np.median(scores),
            "std_rt_score": np.std(scores),
            "early_replication_modules": early_replication_count,
            "early_replication_ratio": early_replication_count / len(rt_scores),
            "tad_distribution": dict(tad_distribution),
            "score_distribution": {
                "high": sum(1 for s in scores if s > 0.7),
                "medium": sum(1 for s in scores if 0.3 <= s <= 0.7),
                "low": sum(1 for s in scores if s < 0.3)
            }
        }


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
    
    Enhanced with replication timing-based scheduling for optimal fix ordering.
    Plans fix sequences, validates dependencies, and rolls back on failures.
    """
    
    def __init__(self, convergence_tracker: ConvergenceTracker, 
                 patch_validator: PatchValidator, 
                 test_command: Optional[str] = None,
                 rt_scheduler: Optional[ReplicationTimingScheduler] = None):
        """
        Initialize fix orchestrator.
        
        Args:
            convergence_tracker: Tracker for convergence monitoring
            patch_validator: Validator for patch safety
            test_command: Command to run tests for validation
            rt_scheduler: Replication timing scheduler for enhanced ordering
        """
        self.convergence_tracker = convergence_tracker
        self.patch_validator = patch_validator
        self.test_command = test_command
        self.rt_scheduler = rt_scheduler
        
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
            # Plan the fix sequence with RT scheduling
            context = {"working_directory": working_directory}
            plan = self._create_fix_plan(fixes, working_directory, context)
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
    
    def _create_fix_plan(self, fixes: List[PatchInfo], 
                       working_directory: Path, 
                       context: Optional[Dict[str, Any]] = None) -> FixPlan:
        """
        Create an execution plan for fixes with replication timing optimization.
        
        Enhanced fix planning:
        1. Use replication timing scheduler if available
        2. Convert PatchInfo to Issue objects for RT scheduling
        3. Apply TAD-aware ordering and early replication priority
        4. Fallback to risk-based ordering if RT scheduler unavailable
        """
        if self.rt_scheduler:
            self.logger.info("Creating fix plan with replication timing scheduling")
            
            # Convert PatchInfo to Issue objects for RT scheduling
            issues = []
            for i, fix in enumerate(fixes):
                issue = Issue(
                    kind="patch_application",
                    message=fix.description,
                    severity=fix.estimated_risk.value,
                    file=fix.file_path,
                    line=1,
                    confidence=0.8
                )
                issues.append(issue)
            
            # Use RT scheduler to prioritize
            try:
                prioritized_issues = self.rt_scheduler.prioritize_fixes(
                    issues, working_directory, context
                )
                
                # Map back to original fix indices
                issue_to_fix_map = {id(issues[i]): i for i in range(len(issues))}
                dependency_order = []
                
                for prioritized_issue in prioritized_issues:
                    for i, original_issue in enumerate(issues):
                        if original_issue.file == prioritized_issue.file and \
                           original_issue.message == prioritized_issue.message:
                            dependency_order.append(i)
                            break
                
                # Ensure all fixes are included (handle any mapping issues)
                missing_indices = set(range(len(fixes))) - set(dependency_order)
                dependency_order.extend(sorted(missing_indices))
                
                self.logger.info(f"RT-based fix ordering: {len(dependency_order)} fixes prioritized")
                
            except Exception as e:
                self.logger.warning(f"RT scheduling failed, falling back to risk-based: {e}")
                dependency_order = self._create_risk_based_ordering(fixes)
        else:
            # Fallback to original risk-based ordering
            dependency_order = self._create_risk_based_ordering(fixes)
        
        # Create rollback points every 3 fixes or at TAD boundaries
        rollback_points = self._calculate_rollback_points(fixes, dependency_order, context)
        
        return FixPlan(
            fixes=fixes,
            dependency_order=dependency_order,
            rollback_points=rollback_points,
            estimated_duration=len(fixes) * 10.0,  # Rough estimate
            confidence_score=0.9 if self.rt_scheduler else 0.8
        )
    
    def _create_risk_based_ordering(self, fixes: List[PatchInfo]) -> List[int]:
        """Create risk-based fix ordering (fallback method)."""
        dependency_order = list(range(len(fixes)))
        dependency_order.sort(key=lambda i: fixes[i].estimated_risk.value)
        return dependency_order
    
    def _calculate_rollback_points(self, fixes: List[PatchInfo], 
                                 dependency_order: List[int],
                                 context: Optional[Dict[str, Any]]) -> List[int]:
        """Calculate optimal rollback points based on TAD boundaries and risk."""
        rollback_points = [0]  # Always start with rollback point at 0
        
        if self.rt_scheduler and len(dependency_order) > 3:
            # Place rollback points at TAD boundaries
            current_tad = None
            tad_transitions = []
            
            for i, fix_idx in enumerate(dependency_order):
                fix = fixes[fix_idx]
                module_path = self.rt_scheduler._file_to_module_path(fix.file_path)
                tad = self.rt_scheduler._get_tad_membership(module_path)
                
                if current_tad is not None and tad != current_tad:
                    tad_transitions.append(i)
                current_tad = tad
            
            # Add rollback points at major TAD transitions
            for transition in tad_transitions:
                if transition not in rollback_points:
                    rollback_points.append(transition)
        
        # Add regular interval rollback points (every 3-5 fixes)
        interval_points = [i for i in range(3, len(dependency_order), 4)]
        rollback_points.extend(interval_points)
        
        # Remove duplicates and sort
        rollback_points = sorted(list(set(rollback_points)))
        
        return rollback_points
    
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
                            test_command: Optional[str] = None,
                            git_analyzer=None,
                            chromatin_analyzer=None,
                            enable_rt_scheduling: bool = True) -> Tuple[ConvergenceTracker, PatchValidator, FixOrchestrator]:
    """
    Create a complete convergence system with replication timing scheduling.
    
    Args:
        max_iterations: Maximum fix iterations
        max_risk_level: Maximum acceptable patch risk
        test_command: Command to run tests
        git_analyzer: GitChainAnalyzer for git history analysis
        chromatin_analyzer: ChromatinContactAnalyzer for TAD detection
        enable_rt_scheduling: Enable replication timing-based scheduling
        
    Returns:
        (tracker, validator, orchestrator) tuple
    """
    tracker = ConvergenceTracker(max_iterations=max_iterations)
    validator = PatchValidator(max_risk_level=max_risk_level)
    
    # Create RT scheduler if enabled and analyzers available
    rt_scheduler = None
    if enable_rt_scheduling and (git_analyzer or chromatin_analyzer):
        rt_scheduler = ReplicationTimingScheduler(
            git_analyzer=git_analyzer,
            chromatin_analyzer=chromatin_analyzer
        )
    
    orchestrator = FixOrchestrator(tracker, validator, test_command, rt_scheduler)
    
    return tracker, validator, orchestrator


def create_replication_timing_scheduler(git_analyzer=None, 
                                      chromatin_analyzer=None,
                                      lambda1: float = 0.4,
                                      lambda2: float = 0.3, 
                                      lambda3: float = 0.3) -> ReplicationTimingScheduler:
    """
    Create a standalone replication timing scheduler.
    
    Args:
        git_analyzer: GitChainAnalyzer for git history analysis
        chromatin_analyzer: ChromatinContactAnalyzer for TAD detection
        lambda1: Weight for git churn component
        lambda2: Weight for test coverage component
        lambda3: Weight for runtime reach component
        
    Returns:
        Configured ReplicationTimingScheduler
    """
    return ReplicationTimingScheduler(
        git_analyzer=git_analyzer,
        chromatin_analyzer=chromatin_analyzer,
        lambda1=lambda1,
        lambda2=lambda2,
        lambda3=lambda3
    )