"""
Safe code execution sandbox for validating patches.

Provides isolated execution environment with resource limits to safely test code changes.
"""

import os
import sys
import shutil
import tempfile
import subprocess
import resource
import signal
import time
import traceback
import threading
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import psutil
import logging

from ..utils.logging_setup import get_logger
from ..engine.convergence import PatchInfo


class ErrorType(Enum):
    """Types of sandbox execution errors."""
    SYNTAX_ERROR = "syntax_error"
    RUNTIME_ERROR = "runtime_error" 
    TIMEOUT = "timeout"
    MEMORY_LIMIT = "memory_limit"
    RESOURCE_LIMIT = "resource_limit"
    IO_ERROR = "io_error"
    SETUP_ERROR = "setup_error"
    CLEANUP_ERROR = "cleanup_error"


class SandboxError(Exception):
    """Base exception for sandbox execution errors."""
    def __init__(self, message: str, error_type: ErrorType, details: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.error_type = error_type
        self.details = details or {}


@dataclass
class ResourceLimits:
    """Resource limits for sandbox execution."""
    memory_mb: int = 512  # Memory limit in MB
    cpu_time_seconds: int = 45  # CPU time limit
    wall_time_seconds: int = 60  # Wall clock time limit
    max_file_descriptors: int = 256  # File descriptor limit
    max_processes: int = 10  # Process limit
    max_file_size_mb: int = 100  # Maximum file size in MB
    disable_network: bool = True  # Disable network access
    temp_dir_size_mb: int = 1024  # Temp directory size limit


@dataclass
class ExecutionMetrics:
    """Metrics from sandbox execution."""
    peak_memory_mb: float = 0.0
    cpu_time_seconds: float = 0.0
    wall_time_seconds: float = 0.0
    files_created: int = 0
    processes_spawned: int = 0
    return_code: int = 0


@dataclass
class SandboxResult(ABC):
    """Base class for sandbox execution results."""
    elapsed_time: float
    metrics: ExecutionMetrics
    sandbox_path: Optional[str]
    
    @abstractmethod
    def is_success(self) -> bool:
        """Return True if execution was successful."""
        pass


@dataclass
class SandboxSuccess(SandboxResult):
    """Successful sandbox execution result."""
    stdout: str
    stderr: str
    output_files: Dict[str, str] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        return True


@dataclass
class SandboxFailure(SandboxResult):
    """Failed sandbox execution result."""
    error_type: ErrorType
    error_message: str
    traceback_str: str
    partial_stdout: str = ""
    partial_stderr: str = ""
    
    def is_success(self) -> bool:
        return False


@dataclass
class SandboxTimeout(SandboxResult):
    """Timed out sandbox execution result."""
    timeout_type: str  # "cpu" or "wall"
    partial_stdout: str = ""
    partial_stderr: str = ""
    killed_at: float = 0.0
    
    def is_success(self) -> bool:
        return False


class ProcessMonitor:
    """Monitors resource usage of sandbox processes."""
    
    def __init__(self, limits: ResourceLimits):
        self.limits = limits
        self.metrics = ExecutionMetrics()
        self.monitoring = False
        self.start_time = 0.0
        self.process: Optional[psutil.Process] = None
        
    def start_monitoring(self, process: psutil.Process) -> None:
        """Start monitoring the given process."""
        self.process = process
        self.monitoring = True
        self.start_time = time.time()
        self.metrics = ExecutionMetrics()
        
        # Start monitoring thread
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()
    
    def stop_monitoring(self) -> ExecutionMetrics:
        """Stop monitoring and return final metrics."""
        self.monitoring = False
        if self.process:
            self.metrics.wall_time_seconds = time.time() - self.start_time
        return self.metrics
    
    def _monitor_loop(self) -> None:
        """Monitor resource usage in a loop."""
        try:
            while self.monitoring and self.process and self.process.is_running():
                try:
                    # Memory usage
                    memory_info = self.process.memory_info()
                    memory_mb = memory_info.rss / (1024 * 1024)
                    self.metrics.peak_memory_mb = max(self.metrics.peak_memory_mb, memory_mb)
                    
                    # CPU time
                    cpu_times = self.process.cpu_times()
                    self.metrics.cpu_time_seconds = cpu_times.user + cpu_times.system
                    
                    # Process count
                    try:
                        children = self.process.children(recursive=True)
                        self.metrics.processes_spawned = len(children) + 1
                        
                        # Check process limit
                        if self.metrics.processes_spawned > self.limits.max_processes:
                            self.process.terminate()
                            break
                            
                    except psutil.NoSuchProcess:
                        pass
                    
                    # Memory limit check
                    if memory_mb > self.limits.memory_mb:
                        self.process.terminate()
                        break
                    
                    # CPU time limit check
                    if self.metrics.cpu_time_seconds > self.limits.cpu_time_seconds:
                        self.process.terminate()
                        break
                    
                    time.sleep(0.1)  # Check every 100ms
                    
                except psutil.NoSuchProcess:
                    break
                except Exception:
                    # Continue monitoring even if we can't get some metrics
                    pass
                    
        except Exception:
            # Monitoring thread should not crash the main execution
            pass


class SandboxRunner:
    """
    Safe code execution sandbox with resource limits and isolation.
    
    Creates isolated temporary directories, applies patches to file copies,
    and executes tests with comprehensive resource monitoring and cleanup.
    """
    
    def __init__(self, limits: Optional[ResourceLimits] = None):
        """
        Initialize sandbox runner.
        
        Args:
            limits: Resource limits for execution
        """
        self.limits = limits or ResourceLimits()
        self.logger = get_logger(__name__)
        self._active_sandboxes: List[str] = []
        
    def execute_with_patches(self, 
                           source_dir: Path,
                           patches: List[PatchInfo],
                           test_command: str,
                           python_path: Optional[str] = None) -> SandboxResult:
        """
        Execute test command with patches applied in isolated sandbox.
        
        Args:
            source_dir: Source directory to copy
            patches: List of patches to apply
            test_command: Command to execute for testing
            python_path: Optional Python interpreter path
            
        Returns:
            SandboxResult indicating success, failure, or timeout
        """
        sandbox_dir = None
        start_time = time.time()
        
        try:
            # Create isolated sandbox
            sandbox_dir = self._create_sandbox(source_dir)
            self.logger.info(f"Created sandbox at {sandbox_dir}")
            
            # Apply patches
            self._apply_patches(sandbox_dir, patches)
            self.logger.debug(f"Applied {len(patches)} patches")
            
            # Execute test command
            result = self._execute_command(
                sandbox_dir, test_command, python_path
            )
            
            elapsed = time.time() - start_time
            result.elapsed_time = elapsed
            result.sandbox_path = str(sandbox_dir)
            
            self.logger.info(f"Sandbox execution completed in {elapsed:.2f}s")
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Sandbox execution failed: {e}", exc_info=True)
            
            metrics = ExecutionMetrics()
            metrics.wall_time_seconds = elapsed
            
            return SandboxFailure(
                elapsed_time=elapsed,
                metrics=metrics,
                sandbox_path=str(sandbox_dir) if sandbox_dir else None,
                error_type=ErrorType.SETUP_ERROR,
                error_message=str(e),
                traceback_str=traceback.format_exc()
            )
            
        finally:
            # Always attempt cleanup
            if sandbox_dir:
                self._cleanup_sandbox(sandbox_dir)
    
    def execute_code_snippet(self,
                           code: str,
                           test_code: Optional[str] = None,
                           python_path: Optional[str] = None) -> SandboxResult:
        """
        Execute a code snippet in isolation.
        
        Args:
            code: Python code to execute
            test_code: Optional test code to run
            python_path: Optional Python interpreter path
            
        Returns:
            SandboxResult indicating success, failure, or timeout
        """
        sandbox_dir = None
        start_time = time.time()
        
        try:
            # Create sandbox with just the code
            sandbox_dir = self._create_temp_sandbox()
            
            # Write code to file
            code_file = Path(sandbox_dir) / "code.py"
            code_file.write_text(code)
            
            # Prepare command
            if test_code:
                test_file = Path(sandbox_dir) / "test_code.py"
                test_file.write_text(test_code)
                command = f"python test_code.py"
            else:
                command = f"python code.py"
            
            # Execute
            result = self._execute_command(sandbox_dir, command, python_path)
            
            elapsed = time.time() - start_time
            result.elapsed_time = elapsed
            result.sandbox_path = str(sandbox_dir)
            
            return result
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Code snippet execution failed: {e}")
            
            metrics = ExecutionMetrics()
            metrics.wall_time_seconds = elapsed
            
            return SandboxFailure(
                elapsed_time=elapsed,
                metrics=metrics,
                sandbox_path=str(sandbox_dir) if sandbox_dir else None,
                error_type=ErrorType.SETUP_ERROR,
                error_message=str(e),
                traceback_str=traceback.format_exc()
            )
            
        finally:
            if sandbox_dir:
                self._cleanup_sandbox(sandbox_dir)
    
    def _create_sandbox(self, source_dir: Path) -> str:
        """Create isolated sandbox by copying source directory."""
        try:
            # Create temporary directory
            temp_dir = tempfile.mkdtemp(prefix="tcf_sandbox_")
            self._active_sandboxes.append(temp_dir)
            
            # Copy source directory
            sandbox_source = Path(temp_dir) / "source"
            if source_dir.is_file():
                # Single file
                sandbox_source.mkdir()
                shutil.copy2(source_dir, sandbox_source / source_dir.name)
            else:
                # Directory tree
                shutil.copytree(source_dir, sandbox_source, symlinks=False)
            
            self.logger.debug(f"Copied source to {sandbox_source}")
            return temp_dir
            
        except OSError as e:
            raise SandboxError(
                f"Failed to create sandbox: {e}",
                ErrorType.SETUP_ERROR,
                {"source_dir": str(source_dir)}
            )
    
    def _create_temp_sandbox(self) -> str:
        """Create empty temporary sandbox."""
        try:
            temp_dir = tempfile.mkdtemp(prefix="tcf_sandbox_")
            self._active_sandboxes.append(temp_dir)
            return temp_dir
        except OSError as e:
            raise SandboxError(f"Failed to create temp sandbox: {e}", ErrorType.SETUP_ERROR)
    
    def _apply_patches(self, sandbox_dir: str, patches: List[PatchInfo]) -> None:
        """Apply patches to files in sandbox."""
        source_dir = Path(sandbox_dir) / "source"
        
        for patch in patches:
            try:
                target_file = source_dir / patch.file_path
                
                # Ensure parent directories exist
                target_file.parent.mkdir(parents=True, exist_ok=True)
                
                # Write patched content
                target_file.write_text(patch.patched_content)
                
                self.logger.debug(f"Applied patch to {target_file}")
                
            except OSError as e:
                raise SandboxError(
                    f"Failed to apply patch to {patch.file_path}: {e}",
                    ErrorType.IO_ERROR,
                    {"patch": patch.description}
                )
    
    def _execute_command(self, 
                        sandbox_dir: str, 
                        command: str,
                        python_path: Optional[str] = None) -> SandboxResult:
        """Execute command in sandbox with resource monitoring."""
        
        # Prepare environment
        env = os.environ.copy()
        if python_path:
            env["PYTHONPATH"] = python_path
        
        # Disable network if requested
        if self.limits.disable_network:
            env["PYTHONHTTPSVERIFY"] = "0"
            # Note: True network isolation requires additional system-level controls
        
        # Change to source directory if it exists
        work_dir = Path(sandbox_dir)
        if (work_dir / "source").exists():
            work_dir = work_dir / "source"
        
        start_time = time.time()
        monitor = ProcessMonitor(self.limits)
        
        try:
            # Start process
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=str(work_dir),
                env=env,
                preexec_fn=self._set_resource_limits
            )
            
            # Start monitoring
            try:
                ps_process = psutil.Process(process.pid)
                monitor.start_monitoring(ps_process)
            except psutil.NoSuchProcess:
                # Process might have finished quickly
                pass
            
            # Wait for completion with timeout
            try:
                stdout, stderr = process.communicate(timeout=self.limits.wall_time_seconds)
                return_code = process.returncode
                
            except subprocess.TimeoutExpired:
                # Kill process tree
                try:
                    ps_process = psutil.Process(process.pid)
                    for child in ps_process.children(recursive=True):
                        child.terminate()
                    ps_process.terminate()
                    
                    # Wait a bit then force kill
                    time.sleep(1.0)
                    for child in ps_process.children(recursive=True):
                        if child.is_running():
                            child.kill()
                    if ps_process.is_running():
                        ps_process.kill()
                        
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
                
                # Get partial output
                try:
                    stdout, stderr = process.communicate(timeout=1.0)
                except subprocess.TimeoutExpired:
                    stdout, stderr = "", ""
                
                metrics = monitor.stop_monitoring()
                
                return SandboxTimeout(
                    elapsed_time=time.time() - start_time,
                    metrics=metrics,
                    sandbox_path=None,
                    timeout_type="wall",
                    partial_stdout=stdout or "",
                    partial_stderr=stderr or "",
                    killed_at=time.time() - start_time
                )
            
            # Stop monitoring
            metrics = monitor.stop_monitoring()
            metrics.return_code = return_code
            
            # Check for errors
            if return_code != 0:
                # Determine error type from output
                error_type = ErrorType.RUNTIME_ERROR
                if "SyntaxError" in stderr:
                    error_type = ErrorType.SYNTAX_ERROR
                elif "MemoryError" in stderr or "killed" in stderr.lower():
                    error_type = ErrorType.MEMORY_LIMIT
                
                return SandboxFailure(
                    elapsed_time=time.time() - start_time,
                    metrics=metrics,
                    sandbox_path=None,
                    error_type=error_type,
                    error_message=f"Process exited with code {return_code}",
                    traceback_str=stderr,
                    partial_stdout=stdout,
                    partial_stderr=stderr
                )
            
            # Success - collect output files
            output_files = self._collect_output_files(work_dir)
            
            return SandboxSuccess(
                elapsed_time=time.time() - start_time,
                metrics=metrics,
                sandbox_path=None,
                stdout=stdout,
                stderr=stderr,
                output_files=output_files
            )
            
        except Exception as e:
            metrics = monitor.stop_monitoring()
            
            return SandboxFailure(
                elapsed_time=time.time() - start_time,
                metrics=metrics,
                sandbox_path=None,
                error_type=ErrorType.RUNTIME_ERROR,
                error_message=str(e),
                traceback_str=traceback.format_exc()
            )
    
    def _set_resource_limits(self) -> None:
        """Set resource limits for child process (Unix only)."""
        try:
            # Memory limit
            if hasattr(resource, 'RLIMIT_AS'):
                memory_bytes = self.limits.memory_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_AS, (memory_bytes, memory_bytes))
            
            # CPU time limit  
            resource.setrlimit(resource.RLIMIT_CPU, (self.limits.cpu_time_seconds, self.limits.cpu_time_seconds))
            
            # File descriptor limit
            resource.setrlimit(resource.RLIMIT_NOFILE, (self.limits.max_file_descriptors, self.limits.max_file_descriptors))
            
            # File size limit
            if hasattr(resource, 'RLIMIT_FSIZE'):
                file_size_bytes = self.limits.max_file_size_mb * 1024 * 1024
                resource.setrlimit(resource.RLIMIT_FSIZE, (file_size_bytes, file_size_bytes))
                
        except (OSError, ValueError) as e:
            # Resource limit setting failed - log but continue
            # This is non-fatal as we have additional monitoring
            pass
    
    def _collect_output_files(self, work_dir: Path) -> Dict[str, str]:
        """Collect output files from sandbox execution."""
        output_files = {}
        
        try:
            # Look for common output file patterns
            for pattern in ["*.log", "*.out", "*.json", "*.xml", "*.txt"]:
                for file_path in work_dir.glob(pattern):
                    if file_path.is_file() and file_path.stat().st_size < 1024 * 1024:  # Max 1MB
                        try:
                            output_files[file_path.name] = file_path.read_text()
                        except (UnicodeDecodeError, OSError):
                            # Skip binary or unreadable files
                            pass
        except Exception:
            # File collection is best-effort
            pass
            
        return output_files
    
    def _cleanup_sandbox(self, sandbox_dir: str) -> None:
        """Clean up sandbox directory."""
        try:
            if sandbox_dir in self._active_sandboxes:
                self._active_sandboxes.remove(sandbox_dir)
            
            if os.path.exists(sandbox_dir):
                shutil.rmtree(sandbox_dir, ignore_errors=True)
                self.logger.debug(f"Cleaned up sandbox {sandbox_dir}")
                
        except Exception as e:
            self.logger.warning(f"Failed to cleanup sandbox {sandbox_dir}: {e}")
    
    def cleanup_all(self) -> None:
        """Clean up all active sandboxes."""
        for sandbox_dir in self._active_sandboxes[:]:  # Copy to avoid modification during iteration
            self._cleanup_sandbox(sandbox_dir)
    
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit with cleanup."""
        self.cleanup_all()


# Utility functions
def create_sandbox_runner(memory_mb: int = 512, 
                         timeout_seconds: int = 45,
                         disable_network: bool = True) -> SandboxRunner:
    """
    Create a sandbox runner with common settings.
    
    Args:
        memory_mb: Memory limit in MB
        timeout_seconds: Execution timeout
        disable_network: Whether to disable network access
        
    Returns:
        Configured SandboxRunner instance
    """
    limits = ResourceLimits(
        memory_mb=memory_mb,
        cpu_time_seconds=timeout_seconds,
        wall_time_seconds=timeout_seconds + 15,
        disable_network=disable_network
    )
    
    return SandboxRunner(limits)


def is_safe_to_execute(code: str) -> Tuple[bool, List[str]]:
    """
    Check if code appears safe to execute in sandbox.
    
    Args:
        code: Python code to check
        
    Returns:
        (is_safe, warnings) tuple
    """
    warnings = []
    
    # Check for dangerous imports
    dangerous_imports = [
        'subprocess', 'os.system', 'eval', 'exec', 'compile',
        'ctypes', '__import__', 'importlib', 'socket', 'urllib',
        'requests', 'http', 'ftplib', 'telnetlib'
    ]
    
    for danger in dangerous_imports:
        if danger in code:
            warnings.append(f"Contains potentially dangerous: {danger}")
    
    # Check for file system operations
    file_operations = ['open(', 'file(', 'shutil.', 'pathlib', 'glob.']
    for op in file_operations:
        if op in code:
            warnings.append(f"Contains file operation: {op}")
    
    # Consider safe if no high-risk patterns found
    high_risk = ['subprocess', 'os.system', 'eval', 'exec', '__import__']
    is_safe = not any(risk in code for risk in high_risk)
    
    return is_safe, warnings