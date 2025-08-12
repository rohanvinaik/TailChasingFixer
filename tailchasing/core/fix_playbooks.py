"""Playbook system for orchestrated code fixes with safety checks and rollbacks."""

import difflib
import json
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
import ast
import shutil
import hashlib


class PlaybookStage(Enum):
    """Stages of playbook execution."""
    PLANNING = "planning"
    SAFETY_CHECK = "safety_check" 
    EXECUTION = "execution"
    VALIDATION = "validation"
    ROLLBACK = "rollback"
    COMPLETED = "completed"
    FAILED = "failed"


class ChangeRisk(Enum):
    """Risk levels for code changes."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class CodeChange:
    """Represents a single code change operation."""
    file_path: str
    line_start: int
    line_end: int
    old_content: str
    new_content: str
    change_type: str  # 'replace', 'insert', 'delete'
    description: str
    risk_level: ChangeRisk = ChangeRisk.MEDIUM
    dependencies: List[str] = field(default_factory=list)
    validation_checks: List[str] = field(default_factory=list)


@dataclass
class SafetyCheck:
    """Represents a safety check to run before/after changes."""
    check_id: str
    description: str
    check_function: str  # Function name or command to run
    is_blocking: bool = True
    timeout_seconds: int = 30
    expected_result: Any = True


@dataclass
class PlaybookStep:
    """A single step in a fix playbook."""
    step_id: str
    name: str
    description: str
    changes: List[CodeChange]
    safety_checks: List[SafetyCheck]
    rollback_changes: List[CodeChange] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    is_optional: bool = False


@dataclass
class FixPlaybook:
    """Complete playbook for fixing a cluster of issues."""
    playbook_id: str
    name: str
    description: str
    cluster_id: str
    steps: List[PlaybookStep]
    risk_level: ChangeRisk
    safety_checks: List[SafetyCheck] = field(default_factory=list)
    rollback_strategy: str = "step_by_step"
    estimated_time_minutes: int = 5
    requires_review: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


class PlaybookExecution:
    """Tracks the execution state of a playbook."""
    
    def __init__(self, playbook: FixPlaybook):
        self.playbook = playbook
        self.current_stage = PlaybookStage.PLANNING
        self.completed_steps: List[str] = []
        self.failed_steps: List[str] = []
        self.execution_log: List[Dict[str, Any]] = []
        self.backup_files: Dict[str, str] = {}  # file_path -> backup_path
        self.start_time: Optional[float] = None
        self.end_time: Optional[float] = None
        self.dry_run = False
    
    def add_log_entry(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Add an entry to the execution log."""
        entry = {
            'timestamp': time.time(),
            'level': level,
            'message': message,
            'stage': self.current_stage.value,
            'details': details or {}
        }
        self.execution_log.append(entry)
    
    def get_unified_diff_preview(self) -> str:
        """Generate unified diff preview of all changes."""
        diff_lines = []
        
        for step in self.playbook.steps:
            diff_lines.extend([
                f"=== Step: {step.name} ===",
                f"Description: {step.description}",
                ""
            ])
            
            for change in step.changes:
                diff_lines.extend([
                    f"--- {change.file_path} (before)",
                    f"+++ {change.file_path} (after)",
                ])
                
                # Generate unified diff
                old_lines = change.old_content.splitlines(keepends=True)
                new_lines = change.new_content.splitlines(keepends=True)
                
                diff = difflib.unified_diff(
                    old_lines,
                    new_lines,
                    fromfile=f"{change.file_path} (before)",
                    tofile=f"{change.file_path} (after)",
                    lineterm=""
                )
                
                diff_lines.extend(list(diff)[2:])  # Skip the file headers
                diff_lines.append("")
        
        return "\n".join(diff_lines)


class SafetyCheckRunner:
    """Runs safety checks and validates code changes."""
    
    def __init__(self):
        self.check_registry: Dict[str, Callable] = {}
        self._register_builtin_checks()
    
    def _register_builtin_checks(self):
        """Register built-in safety checks."""
        self.check_registry.update({
            'syntax_valid': self._check_syntax_valid,
            'imports_resolvable': self._check_imports_resolvable,
            'tests_pass': self._check_tests_pass,
            'lint_clean': self._check_lint_clean,
            'no_security_issues': self._check_no_security_issues,
            'performance_acceptable': self._check_performance_acceptable,
        })
    
    def run_check(self, check: SafetyCheck, file_paths: List[str]) -> Tuple[bool, str]:
        """Run a single safety check."""
        try:
            if check.check_function in self.check_registry:
                # Built-in check function
                check_func = self.check_registry[check.check_function]
                result = check_func(file_paths)
                return result == check.expected_result, ""
            else:
                # External command
                result = subprocess.run(
                    check.check_function.split(),
                    capture_output=True,
                    text=True,
                    timeout=check.timeout_seconds,
                    cwd=Path.cwd()
                )
                
                success = (result.returncode == 0) == check.expected_result
                output = result.stdout + result.stderr
                return success, output
                
        except subprocess.TimeoutExpired:
            return False, f"Check timed out after {check.timeout_seconds}s"
        except Exception as e:
            return False, f"Check failed with error: {e}"
    
    def _check_syntax_valid(self, file_paths: List[str]) -> bool:
        """Check that Python files have valid syntax."""
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                ast.parse(content)
            except SyntaxError:
                return False
        
        return True
    
    def _check_imports_resolvable(self, file_paths: List[str]) -> bool:
        """Check that imports can be resolved."""
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
            
            try:
                result = subprocess.run(
                    ['python', '-m', 'py_compile', file_path],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode != 0:
                    return False
            except subprocess.TimeoutExpired:
                return False
        
        return True
    
    def _check_tests_pass(self, file_paths: List[str]) -> bool:
        """Run tests to ensure changes don't break functionality."""
        try:
            result = subprocess.run(
                ['python', '-m', 'pytest', '--tb=short', '-q'],
                capture_output=True,
                timeout=60
            )
            return result.returncode == 0
        except subprocess.TimeoutExpired:
            return False
    
    def _check_lint_clean(self, file_paths: List[str]) -> bool:
        """Check that files pass linting."""
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
            
            try:
                result = subprocess.run(
                    ['python', '-m', 'ruff', 'check', file_path],
                    capture_output=True,
                    timeout=10
                )
                if result.returncode != 0:
                    return False
            except (subprocess.TimeoutExpired, FileNotFoundError):
                # If ruff not available, skip this check
                continue
        
        return True
    
    def _check_no_security_issues(self, file_paths: List[str]) -> bool:
        """Check for obvious security issues in the code."""
        security_patterns = [
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.call\s*\(',
            r'os\.system\s*\(',
            r'__import__\s*\(',
        ]
        
        import re
        
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                for pattern in security_patterns:
                    if re.search(pattern, content):
                        return False
            except IOError:
                continue
        
        return True
    
    def _check_performance_acceptable(self, file_paths: List[str]) -> bool:
        """Basic performance check - no obvious inefficiencies."""
        inefficient_patterns = [
            r'while\s+True\s*:',  # Infinite loops
            r'for\s+\w+\s+in\s+range\s*\(\s*len\s*\(',  # Use enumerate instead
        ]
        
        import re
        
        for file_path in file_paths:
            if not file_path.endswith('.py'):
                continue
            
            try:
                with open(file_path, 'r') as f:
                    content = f.read()
                
                for pattern in inefficient_patterns:
                    if re.search(pattern, content):
                        return False
            except IOError:
                continue
        
        return True


class PlaybookEngine:
    """Main engine for executing fix playbooks."""
    
    def __init__(self):
        self.safety_runner = SafetyCheckRunner()
        self.active_executions: Dict[str, PlaybookExecution] = {}
    
    def preview_playbook(self, playbook: FixPlaybook) -> str:
        """Generate a preview of the playbook changes."""
        execution = PlaybookExecution(playbook)
        execution.dry_run = True
        
        preview_lines = [
            f"PLAYBOOK PREVIEW: {playbook.name}",
            "=" * 60,
            f"Description: {playbook.description}",
            f"Risk Level: {playbook.risk_level.value.upper()}",
            f"Estimated Time: {playbook.estimated_time_minutes} minutes",
            f"Steps: {len(playbook.steps)}",
            ""
        ]
        
        if playbook.requires_review:
            preview_lines.extend([
                "⚠️  REQUIRES MANUAL REVIEW BEFORE EXECUTION",
                ""
            ])
        
        preview_lines.extend([
            "UNIFIED DIFF PREVIEW:",
            "-" * 40,
            execution.get_unified_diff_preview(),
            "",
            "SAFETY CHECKS:",
            "-" * 20
        ])
        
        # List safety checks
        all_checks = set()
        for step in playbook.steps:
            for check in step.safety_checks:
                all_checks.add((check.check_id, check.description, check.is_blocking))
        
        for check_id, description, is_blocking in all_checks:
            status = "BLOCKING" if is_blocking else "WARNING"
            preview_lines.append(f"  [{status}] {check_id}: {description}")
        
        return "\n".join(preview_lines)
    
    def execute_playbook(self, playbook: FixPlaybook, dry_run: bool = False) -> PlaybookExecution:
        """Execute a playbook with safety checks and rollback capability."""
        execution = PlaybookExecution(playbook)
        execution.dry_run = dry_run
        execution.start_time = time.time()
        
        self.active_executions[playbook.playbook_id] = execution
        
        try:
            # Stage 1: Planning and validation
            execution.current_stage = PlaybookStage.PLANNING
            execution.add_log_entry("INFO", "Starting playbook execution", {
                "dry_run": dry_run,
                "playbook_id": playbook.playbook_id
            })
            
            if not self._validate_playbook(execution):
                execution.current_stage = PlaybookStage.FAILED
                return execution
            
            # Stage 2: Safety checks
            execution.current_stage = PlaybookStage.SAFETY_CHECK
            if not self._run_pre_execution_safety_checks(execution):
                execution.current_stage = PlaybookStage.FAILED
                return execution
            
            # Stage 3: Execution
            execution.current_stage = PlaybookStage.EXECUTION
            if not self._execute_steps(execution):
                execution.current_stage = PlaybookStage.ROLLBACK
                self._rollback_changes(execution)
                execution.current_stage = PlaybookStage.FAILED
                return execution
            
            # Stage 4: Validation
            execution.current_stage = PlaybookStage.VALIDATION
            if not self._validate_changes(execution):
                execution.current_stage = PlaybookStage.ROLLBACK
                self._rollback_changes(execution)
                execution.current_stage = PlaybookStage.FAILED
                return execution
            
            execution.current_stage = PlaybookStage.COMPLETED
            execution.add_log_entry("INFO", "Playbook execution completed successfully")
            
        except Exception as e:
            execution.add_log_entry("ERROR", f"Unexpected error during execution: {e}")
            execution.current_stage = PlaybookStage.ROLLBACK
            self._rollback_changes(execution)
            execution.current_stage = PlaybookStage.FAILED
        
        finally:
            execution.end_time = time.time()
        
        return execution
    
    def _validate_playbook(self, execution: PlaybookExecution) -> bool:
        """Validate playbook structure and dependencies."""
        playbook = execution.playbook
        
        # Check that all referenced files exist
        for step in playbook.steps:
            for change in step.changes:
                if not Path(change.file_path).exists():
                    execution.add_log_entry("ERROR", f"File not found: {change.file_path}")
                    return False
        
        # Check dependency order
        step_ids = {step.step_id for step in playbook.steps}
        for step in playbook.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    execution.add_log_entry("ERROR", f"Missing dependency: {dep} for step {step.step_id}")
                    return False
        
        execution.add_log_entry("INFO", "Playbook structure validation passed")
        return True
    
    def _run_pre_execution_safety_checks(self, execution: PlaybookExecution) -> bool:
        """Run safety checks before making any changes."""
        playbook = execution.playbook
        
        # Get all files that will be modified
        modified_files = set()
        for step in playbook.steps:
            for change in step.changes:
                modified_files.add(change.file_path)
        
        # Run global safety checks
        for check in playbook.safety_checks:
            success, output = self.safety_runner.run_check(check, list(modified_files))
            
            execution.add_log_entry(
                "INFO" if success else "WARNING",
                f"Safety check {check.check_id}: {'PASSED' if success else 'FAILED'}",
                {"output": output}
            )
            
            if not success and check.is_blocking:
                execution.add_log_entry("ERROR", f"Blocking safety check failed: {check.check_id}")
                return False
        
        return True
    
    def _execute_steps(self, execution: PlaybookExecution) -> bool:
        """Execute all playbook steps in dependency order."""
        playbook = execution.playbook
        
        # Sort steps by dependencies
        ordered_steps = self._topological_sort_steps(playbook.steps)
        
        for step in ordered_steps:
            if not self._execute_single_step(execution, step):
                return False
            
            execution.completed_steps.append(step.step_id)
        
        return True
    
    def _execute_single_step(self, execution: PlaybookExecution, step: PlaybookStep) -> bool:
        """Execute a single playbook step."""
        execution.add_log_entry("INFO", f"Executing step: {step.name}")
        
        # Create backups of files to be modified
        for change in step.changes:
            if change.file_path not in execution.backup_files:
                backup_path = self._create_backup(change.file_path)
                execution.backup_files[change.file_path] = backup_path
        
        # Run step-specific safety checks
        modified_files = [change.file_path for change in step.changes]
        for check in step.safety_checks:
            success, output = self.safety_runner.run_check(check, modified_files)
            
            if not success and check.is_blocking:
                execution.add_log_entry("ERROR", f"Step safety check failed: {check.check_id}")
                execution.failed_steps.append(step.step_id)
                return False
        
        # Apply changes
        if not execution.dry_run:
            for change in step.changes:
                if not self._apply_change(change, execution):
                    execution.failed_steps.append(step.step_id)
                    return False
        else:
            execution.add_log_entry("INFO", f"[DRY RUN] Would apply {len(step.changes)} changes")
        
        execution.add_log_entry("INFO", f"Step completed: {step.name}")
        return True
    
    def _apply_change(self, change: CodeChange, execution: PlaybookExecution) -> bool:
        """Apply a single code change."""
        try:
            with open(change.file_path, 'r') as f:
                current_content = f.read()
            
            # Verify the old content matches what we expect
            if change.change_type == 'replace':
                lines = current_content.splitlines()
                actual_old_content = '\n'.join(lines[change.line_start-1:change.line_end])
                
                if actual_old_content != change.old_content.strip():
                    execution.add_log_entry("ERROR", 
                        f"Content mismatch in {change.file_path} at lines {change.line_start}-{change.line_end}")
                    return False
                
                # Replace content
                new_lines = (lines[:change.line_start-1] + 
                           change.new_content.splitlines() + 
                           lines[change.line_end:])
                new_content = '\n'.join(new_lines)
            
            elif change.change_type == 'insert':
                lines = current_content.splitlines()
                new_lines = (lines[:change.line_start] + 
                           change.new_content.splitlines() + 
                           lines[change.line_start:])
                new_content = '\n'.join(new_lines)
            
            elif change.change_type == 'delete':
                lines = current_content.splitlines()
                new_lines = lines[:change.line_start-1] + lines[change.line_end:]
                new_content = '\n'.join(new_lines)
            
            else:
                execution.add_log_entry("ERROR", f"Unknown change type: {change.change_type}")
                return False
            
            # Write the new content
            with open(change.file_path, 'w') as f:
                f.write(new_content)
            
            execution.add_log_entry("INFO", f"Applied change to {change.file_path}: {change.description}")
            return True
        
        except Exception as e:
            execution.add_log_entry("ERROR", f"Failed to apply change to {change.file_path}: {e}")
            return False
    
    def _validate_changes(self, execution: PlaybookExecution) -> bool:
        """Validate that applied changes work correctly."""
        playbook = execution.playbook
        
        # Run post-execution safety checks
        modified_files = set()
        for step in playbook.steps:
            for change in step.changes:
                modified_files.add(change.file_path)
        
        for check in playbook.safety_checks:
            success, output = self.safety_runner.run_check(check, list(modified_files))
            
            execution.add_log_entry(
                "INFO" if success else "ERROR",
                f"Post-execution check {check.check_id}: {'PASSED' if success else 'FAILED'}",
                {"output": output}
            )
            
            if not success and check.is_blocking:
                return False
        
        return True
    
    def _rollback_changes(self, execution: PlaybookExecution):
        """Rollback all changes made during execution."""
        execution.add_log_entry("INFO", "Starting rollback of changes")
        
        # Restore from backups
        for file_path, backup_path in execution.backup_files.items():
            try:
                if not execution.dry_run:
                    shutil.copy2(backup_path, file_path)
                execution.add_log_entry("INFO", f"Rolled back {file_path}")
            except Exception as e:
                execution.add_log_entry("ERROR", f"Failed to rollback {file_path}: {e}")
    
    def _create_backup(self, file_path: str) -> str:
        """Create a backup of a file."""
        timestamp = int(time.time())
        backup_path = f"{file_path}.backup.{timestamp}"
        shutil.copy2(file_path, backup_path)
        return backup_path
    
    def _topological_sort_steps(self, steps: List[PlaybookStep]) -> List[PlaybookStep]:
        """Sort steps by their dependencies."""
        # Simple topological sort
        sorted_steps = []
        remaining_steps = steps.copy()
        step_map = {step.step_id: step for step in steps}
        
        while remaining_steps:
            # Find steps with no unmet dependencies
            ready_steps = []
            for step in remaining_steps:
                unmet_deps = [dep for dep in step.dependencies 
                             if dep not in [s.step_id for s in sorted_steps]]
                if not unmet_deps:
                    ready_steps.append(step)
            
            if not ready_steps:
                # Circular dependency - just take the first remaining step
                ready_steps = [remaining_steps[0]]
            
            # Add ready steps to sorted list
            for step in ready_steps:
                sorted_steps.append(step)
                remaining_steps.remove(step)
        
        return sorted_steps
    
    def get_execution_status(self, playbook_id: str) -> Optional[PlaybookExecution]:
        """Get the current execution status of a playbook."""
        return self.active_executions.get(playbook_id)
    
    def cleanup_execution(self, playbook_id: str):
        """Clean up execution resources and backups."""
        execution = self.active_executions.get(playbook_id)
        if not execution:
            return
        
        # Clean up backup files
        for backup_path in execution.backup_files.values():
            try:
                Path(backup_path).unlink()
            except FileNotFoundError:
                pass
        
        # Remove from active executions
        del self.active_executions[playbook_id]