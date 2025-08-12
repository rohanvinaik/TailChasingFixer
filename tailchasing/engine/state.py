"""
State management for iteration tracking and convergence detection.

Provides clean separation of state tracking from business logic.
"""

import time
import hashlib
from typing import List, Set, Optional, Dict, Any, Protocol
from dataclasses import dataclass, field
from enum import Enum

from ..core.issues import Issue


class RiskLevel(Enum):
    """Risk levels for patches."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    
    def to_numeric(self) -> float:
        """Convert to numeric risk score."""
        mapping = {
            RiskLevel.LOW: 0.2,
            RiskLevel.MEDIUM: 0.5,
            RiskLevel.HIGH: 0.8,
            RiskLevel.CRITICAL: 1.0
        }
        return mapping[self]


@dataclass
class IterationState:
    """
    Represents the state at a single iteration.
    
    Immutable snapshot of system state for tracking convergence.
    """
    iteration: int
    timestamp: float
    issues: List[Issue]
    fixes_applied: List[str]
    state_hash: str
    metrics: Dict[str, float] = field(default_factory=dict)
    
    @classmethod
    def create(cls, iteration: int, issues: List[Issue], 
               fixes_applied: List[str]) -> 'IterationState':
        """
        Factory method to create iteration state.
        
        Args:
            iteration: Iteration number
            issues: Current issues
            fixes_applied: List of applied fix IDs
            
        Returns:
            New IterationState instance
        """
        # Create deterministic hash of state
        state_str = f"{iteration}:{len(issues)}:{sorted(fixes_applied)}"
        state_hash = hashlib.sha256(state_str.encode()).hexdigest()[:16]
        
        return cls(
            iteration=iteration,
            timestamp=time.time(),
            issues=issues.copy(),
            fixes_applied=fixes_applied.copy(),
            state_hash=state_hash
        )
    
    def is_equivalent_to(self, other: 'IterationState', 
                        tolerance: float = 0.95) -> bool:
        """
        Check if two states are equivalent.
        
        Args:
            other: Other state to compare
            tolerance: Similarity threshold
            
        Returns:
            True if states are equivalent
        """
        if self.state_hash == other.state_hash:
            return True
        
        # Check issue similarity
        if len(self.issues) != len(other.issues):
            return False
        
        # Compare issue types
        self_types = sorted([i.kind for i in self.issues])
        other_types = sorted([i.kind for i in other.issues])
        
        if self_types != other_types:
            return False
        
        # Compare fixes applied
        self_fixes = set(self.fixes_applied)
        other_fixes = set(other.fixes_applied)
        
        overlap = len(self_fixes & other_fixes)
        total = len(self_fixes | other_fixes)
        
        if total == 0:
            return True
        
        similarity = overlap / total
        return similarity >= tolerance


@dataclass
class PatchInfo:
    """Information about a patch to be applied."""
    patch_id: str
    file_path: str
    original_content: str
    patched_content: str
    risk_level: RiskLevel
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_diff(self) -> str:
        """Generate diff for the patch."""
        import difflib
        diff = difflib.unified_diff(
            self.original_content.splitlines(keepends=True),
            self.patched_content.splitlines(keepends=True),
            fromfile=f"a/{self.file_path}",
            tofile=f"b/{self.file_path}"
        )
        return ''.join(diff)


class StateTrackerProtocol(Protocol):
    """Protocol for state tracking implementations."""
    
    def add_state(self, state: IterationState) -> None:
        """Add a new state to tracking."""
        ...
    
    def detect_loop(self, lookback: int = 3) -> bool:
        """Detect if a loop has occurred."""
        ...
    
    def get_history(self) -> List[IterationState]:
        """Get state history."""
        ...


class ConvergenceTracker:
    """
    Tracks iteration states and detects convergence loops.
    
    Simplified version focusing on core loop detection.
    """
    
    def __init__(self, max_iterations: int = 8):
        """
        Initialize convergence tracker.
        
        Args:
            max_iterations: Maximum allowed iterations
        """
        self.max_iterations = max_iterations
        self.iteration_states: List[IterationState] = []
        self.state_hashes: Set[str] = set()
        self.current_iteration = 0
    
    def add_state(self, state: IterationState) -> None:
        """
        Add a new iteration state.
        
        Args:
            state: Iteration state to add
        """
        self.iteration_states.append(state)
        self.state_hashes.add(state.state_hash)
        self.current_iteration = state.iteration
    
    def detect_loop(self, lookback: int = 3) -> bool:
        """
        Detect if we're in a convergence loop.
        
        Args:
            lookback: Number of iterations to check
            
        Returns:
            True if loop detected
        """
        if len(self.iteration_states) < lookback:
            return False
        
        recent_states = self.iteration_states[-lookback:]
        
        # Check for exact repetition
        for i, state1 in enumerate(recent_states):
            for state2 in recent_states[i+1:]:
                if state1.is_equivalent_to(state2):
                    return True
        
        # Check for oscillation pattern
        if self._detect_oscillation(recent_states):
            return True
        
        return False
    
    def _detect_oscillation(self, states: List[IterationState]) -> bool:
        """
        Detect oscillation patterns in states.
        
        Args:
            states: Recent states to check
            
        Returns:
            True if oscillation detected
        """
        if len(states) < 3:
            return False
        
        # Check for A-B-A pattern
        if len(states) >= 3:
            if states[0].is_equivalent_to(states[2]):
                return True
        
        # Check for A-B-C-A pattern
        if len(states) >= 4:
            if states[0].is_equivalent_to(states[3]):
                return True
        
        return False
    
    def should_terminate(self) -> bool:
        """
        Check if convergence should terminate.
        
        Returns:
            True if should terminate
        """
        # Check iteration limit
        if self.current_iteration >= self.max_iterations:
            return True
        
        # Check for loops
        if self.detect_loop():
            return True
        
        # Check for convergence (no changes)
        if len(self.iteration_states) >= 2:
            last = self.iteration_states[-1]
            prev = self.iteration_states[-2]
            if len(last.issues) == 0 and len(prev.issues) == 0:
                return True
        
        return False
    
    def get_history(self) -> List[IterationState]:
        """Get iteration history."""
        return self.iteration_states.copy()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get convergence metrics."""
        return {
            'iterations': self.current_iteration,
            'states_tracked': len(self.iteration_states),
            'unique_states': len(self.state_hashes),
            'loop_detected': self.detect_loop(),
            'converged': len(self.iteration_states) > 0 and 
                        len(self.iteration_states[-1].issues) == 0
        }