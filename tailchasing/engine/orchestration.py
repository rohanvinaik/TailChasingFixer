"""
Core orchestration for fix application.

Simplified orchestrator focusing on essential coordination logic.
"""

import time
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Any, Protocol
from dataclasses import dataclass, field
import logging

from ..core.issues import Issue
from .state import IterationState, PatchInfo, ConvergenceTracker, RiskLevel
from .validation import PatchValidator
from .errors import OrchestrationError, ConvergenceError


class FixProviderProtocol(Protocol):
    """Protocol for fix providers."""
    
    def get_fix(self, issue: Issue) -> Optional[PatchInfo]:
        """Get fix for an issue."""
        ...


class BackupManagerProtocol(Protocol):
    """Protocol for backup management."""
    
    def create_backup(self, file_path: str) -> str:
        """Create backup of a file."""
        ...
    
    def restore_backup(self, backup_path: str, original_path: str) -> None:
        """Restore from backup."""
        ...


@dataclass
class FixPlan:
    """Execution plan for fixes."""
    issues: List[Issue]
    patches: List[PatchInfo]
    execution_order: List[int]  # Indices into patches list
    estimated_risk: float
    estimated_duration: float
    
    def get_ordered_patches(self) -> List[PatchInfo]:
        """Get patches in execution order."""
        return [self.patches[i] for i in self.execution_order]


@dataclass  
class ExecutionResult:
    """Result of fix execution."""
    success: bool
    patches_applied: List[str]
    patches_failed: List[str]
    rollback_performed: bool
    error: Optional[Exception] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class SimpleBackupManager:
    """Simple file backup manager."""
    
    def __init__(self, backup_dir: Optional[Path] = None):
        """
        Initialize backup manager.
        
        Args:
            backup_dir: Directory for backups (temp if not specified)
        """
        if backup_dir:
            self.backup_dir = Path(backup_dir)
            self.backup_dir.mkdir(parents=True, exist_ok=True)
        else:
            import tempfile
            self.backup_dir = Path(tempfile.mkdtemp(prefix='tailchasing_backup_'))
        
        self.backups: Dict[str, str] = {}
    
    def create_backup(self, file_path: str) -> str:
        """
        Create backup of a file.
        
        Args:
            file_path: File to backup
            
        Returns:
            Backup file path
        """
        source = Path(file_path)
        if not source.exists():
            raise FileNotFoundError(f"Cannot backup non-existent file: {file_path}")
        
        # Generate backup path
        backup_name = f"{source.name}.{int(time.time())}.bak"
        backup_path = self.backup_dir / backup_name
        
        # Copy file
        shutil.copy2(source, backup_path)
        self.backups[file_path] = str(backup_path)
        
        return str(backup_path)
    
    def restore_backup(self, backup_path: str, original_path: str) -> None:
        """
        Restore from backup.
        
        Args:
            backup_path: Backup file path
            original_path: Original file path
        """
        shutil.copy2(backup_path, original_path)
    
    def restore_all(self) -> None:
        """Restore all backups."""
        for original, backup in self.backups.items():
            if Path(backup).exists():
                self.restore_backup(backup, original)
    
    def cleanup(self) -> None:
        """Clean up backup directory."""
        if self.backup_dir.exists():
            shutil.rmtree(self.backup_dir, ignore_errors=True)


class FixOrchestrator:
    """
    Orchestrates fix application with convergence control.
    
    Simplified to focus on core orchestration logic.
    """
    
    def __init__(self,
                 max_iterations: int = 8,
                 max_risk: RiskLevel = RiskLevel.HIGH,
                 test_command: Optional[str] = None):
        """
        Initialize orchestrator.
        
        Args:
            max_iterations: Maximum convergence iterations
            max_risk: Maximum acceptable risk level
            test_command: Command to run tests
        """
        self.max_iterations = max_iterations
        self.convergence_tracker = ConvergenceTracker(max_iterations)
        self.validator = PatchValidator(max_risk, test_command)
        self.backup_manager = SimpleBackupManager()
        self.logger = logging.getLogger(__name__)
        
        # Execution state
        self.current_plan: Optional[FixPlan] = None
        self.applied_patches: List[str] = []
    
    def create_plan(self, issues: List[Issue], 
                   fix_provider: FixProviderProtocol) -> FixPlan:
        """
        Create execution plan for fixes.
        
        Args:
            issues: Issues to fix
            fix_provider: Provider for fixes
            
        Returns:
            Execution plan
        """
        patches = []
        for issue in issues:
            patch = fix_provider.get_fix(issue)
            if patch:
                patches.append(patch)
        
        # Determine execution order (simple priority-based)
        execution_order = self._determine_order(patches)
        
        # Calculate risk
        total_risk = sum(p.risk_level.to_numeric() for p in patches) / max(len(patches), 1)
        
        # Estimate duration (simple heuristic)
        estimated_duration = len(patches) * 2.0  # 2 seconds per patch
        
        plan = FixPlan(
            issues=issues,
            patches=patches,
            execution_order=execution_order,
            estimated_risk=total_risk,
            estimated_duration=estimated_duration
        )
        
        self.current_plan = plan
        return plan
    
    def _determine_order(self, patches: List[PatchInfo]) -> List[int]:
        """
        Determine execution order for patches.
        
        Args:
            patches: Patches to order
            
        Returns:
            List of indices in execution order
        """
        # Sort by risk (low risk first) and confidence (high confidence first)
        indexed_patches = list(enumerate(patches))
        indexed_patches.sort(
            key=lambda x: (x[1].risk_level.to_numeric(), -x[1].confidence)
        )
        return [i for i, _ in indexed_patches]
    
    def execute_plan(self, plan: FixPlan) -> ExecutionResult:
        """
        Execute a fix plan.
        
        Args:
            plan: Plan to execute
            
        Returns:
            Execution result
        """
        start_time = time.time()
        patches_applied = []
        patches_failed = []
        rollback_performed = False
        
        try:
            # Get ordered patches
            ordered_patches = plan.get_ordered_patches()
            
            # Apply patches
            for patch in ordered_patches:
                try:
                    # Validate patch
                    self.validator.validate(patch)
                    
                    # Create backup
                    self.backup_manager.create_backup(patch.file_path)
                    
                    # Apply patch
                    self._apply_patch(patch)
                    patches_applied.append(patch.patch_id)
                    self.applied_patches.append(patch.patch_id)
                    
                    self.logger.info(f"Applied patch {patch.patch_id}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply patch {patch.patch_id}: {e}")
                    patches_failed.append(patch.patch_id)
                    
                    # Decide whether to continue or rollback
                    if patch.risk_level == RiskLevel.CRITICAL:
                        raise OrchestrationError(
                            f"Critical patch failed: {patch.patch_id}",
                            operation='apply',
                            details={'patch_id': patch.patch_id, 'error': str(e)}
                        )
            
            # Create iteration state
            remaining_issues = [i for i in plan.issues 
                              if not any(p.metadata.get('fixes_issue') == i.id 
                                       for p in ordered_patches 
                                       if p.patch_id in patches_applied)]
            
            state = IterationState.create(
                iteration=self.convergence_tracker.current_iteration + 1,
                issues=remaining_issues,
                fixes_applied=self.applied_patches
            )
            
            # Check convergence
            self.convergence_tracker.add_state(state)
            
            if self.convergence_tracker.should_terminate():
                if self.convergence_tracker.detect_loop():
                    raise ConvergenceError(
                        "Convergence loop detected",
                        iteration=state.iteration,
                        loop_detected=True
                    )
                elif state.iteration >= self.max_iterations:
                    raise ConvergenceError(
                        f"Maximum iterations ({self.max_iterations}) exceeded",
                        iteration=state.iteration,
                        loop_detected=False
                    )
            
            # Success
            return ExecutionResult(
                success=True,
                patches_applied=patches_applied,
                patches_failed=patches_failed,
                rollback_performed=False,
                metrics={
                    'duration': time.time() - start_time,
                    'iteration': state.iteration,
                    'remaining_issues': len(remaining_issues)
                }
            )
            
        except Exception as e:
            # Rollback on failure
            self.logger.error(f"Execution failed, rolling back: {e}")
            
            try:
                self.backup_manager.restore_all()
                rollback_performed = True
            except Exception as rollback_error:
                raise OrchestrationError(
                    f"Rollback failed: {rollback_error}",
                    operation='rollback',
                    rollback_attempted=True,
                    details={'original_error': str(e), 'rollback_error': str(rollback_error)}
                )
            
            return ExecutionResult(
                success=False,
                patches_applied=patches_applied,
                patches_failed=patches_failed,
                rollback_performed=rollback_performed,
                error=e,
                metrics={'duration': time.time() - start_time}
            )
    
    def _apply_patch(self, patch: PatchInfo) -> None:
        """
        Apply a patch to a file.
        
        Args:
            patch: Patch to apply
        """
        file_path = Path(patch.file_path)
        file_path.write_text(patch.patched_content)
    
    def reset(self) -> None:
        """Reset orchestrator state."""
        self.convergence_tracker = ConvergenceTracker(self.max_iterations)
        self.applied_patches = []
        self.current_plan = None
        self.backup_manager.cleanup()
        self.backup_manager = SimpleBackupManager()
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestration metrics."""
        return {
            'convergence': self.convergence_tracker.get_metrics(),
            'patches_applied': len(self.applied_patches),
            'current_iteration': self.convergence_tracker.current_iteration
        }