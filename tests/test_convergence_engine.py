"""
Tests for the convergence engine system.

Tests the ConvergenceTracker, PatchValidator, and FixOrchestrator components.
"""

import pytest
import tempfile
import time
from pathlib import Path
from typing import List

from tailchasing.engine.convergence import (
    ConvergenceTracker, PatchValidator, FixOrchestrator,
    IterationState, PatchInfo, RiskLevel,
    IterationLimitError, LoopDetectedError, 
    SyntaxValidationError, RiskThresholdError,
    TestFailureError, RollbackError,
    create_convergence_system
)
from tailchasing.core.issues import Issue


class TestConvergenceTracker:
    """Test the ConvergenceTracker class."""
    
    @pytest.fixture
    def tracker(self):
        """Create a convergence tracker for testing."""
        return ConvergenceTracker(max_iterations=5, similarity_threshold=0.1)
    
    def test_iteration_limit_enforcement(self, tracker):
        """Test that iteration limit is enforced."""
        # Should allow iterations up to limit
        for i in range(5):
            state = IterationState(
                iteration=i,
                timestamp=time.time(),
                issues=[Issue(kind="test", message=f"issue_{i}", severity=2)],
                code_snapshot={f"file_{i}.py": f"content_{i}"},
                error_fingerprint=f"fp_{i}",
                changes_applied=[f"change_{i}"]
            )
            assert tracker.should_continue(state)
        
        # Should raise error on exceeding limit
        over_limit_state = IterationState(
            iteration=5, timestamp=time.time(), issues=[], 
            code_snapshot={}, error_fingerprint="over_limit", changes_applied=[]
        )
        
        with pytest.raises(IterationLimitError):
            tracker.should_continue(over_limit_state)
    
    def test_exact_loop_detection(self, tracker):
        """Test detection of exact state repetition."""
        # First state
        state1 = IterationState(
            iteration=0, timestamp=time.time(),
            issues=[Issue(kind="phantom", message="test", severity=2, file="test.py", line=10)],
            code_snapshot={"test.py": "def test(): pass"},
            error_fingerprint="test_fp", changes_applied=["added test function"]
        )
        assert tracker.should_continue(state1)
        
        # Different state
        state2 = IterationState(
            iteration=1, timestamp=time.time(),
            issues=[Issue(kind="different", message="other", severity=2)],
            code_snapshot={"other.py": "def other(): pass"},
            error_fingerprint="other_fp", changes_applied=["added other function"]
        )
        assert tracker.should_continue(state2)
        
        # Exact repeat of state1 (same fingerprint)
        state3 = IterationState(
            iteration=2, timestamp=time.time(),
            issues=[Issue(kind="phantom", message="test", severity=2, file="test.py", line=10)],
            code_snapshot={"test.py": "def test(): pass"},
            error_fingerprint="test_fp", changes_applied=["added test function"]
        )
        
        with pytest.raises(LoopDetectedError):
            tracker.should_continue(state3)
    
    def test_insufficient_progress_detection(self, tracker):
        """Test detection of insufficient progress between iterations."""
        # First iteration with 10 issues
        state1 = IterationState(
            iteration=0, timestamp=time.time(),
            issues=[Issue(kind="test", message=f"issue_{i}", severity=2) for i in range(10)],
            code_snapshot={"test.py": "initial"}, error_fingerprint="fp1", changes_applied=[]
        )
        assert tracker.should_continue(state1)
        
        # Second iteration with 10 issues (no progress) - should continue once
        state2 = IterationState(
            iteration=1, timestamp=time.time(),
            issues=[Issue(kind="test", message=f"issue_{i}", severity=2) for i in range(10)],
            code_snapshot={"test.py": "modified"}, error_fingerprint="fp2", changes_applied=["change1"]
        )
        assert tracker.should_continue(state2)
        
        # Third iteration still with 10 issues (still no progress)
        state3 = IterationState(
            iteration=2, timestamp=time.time(),
            issues=[Issue(kind="test", message=f"issue_{i}", severity=2) for i in range(10)],
            code_snapshot={"test.py": "modified2"}, error_fingerprint="fp3", changes_applied=["change2"]
        )
        
        with pytest.raises(LoopDetectedError):
            tracker.should_continue(state3)
    
    def test_backoff_calculation(self, tracker):
        """Test exponential backoff calculation."""
        assert tracker.calculate_backoff_delay(0) == 1.0  # 1.5^0
        assert tracker.calculate_backoff_delay(1) == 1.5  # 1.5^1
        assert tracker.calculate_backoff_delay(2) == 2.25  # 1.5^2
        assert tracker.calculate_backoff_delay(3) == 3.375  # 1.5^3
    
    def test_failure_tracking(self, tracker):
        """Test tracking of failed approaches."""
        tracker.record_failure("approach_a")
        tracker.record_failure("approach_b")
        tracker.record_failure("approach_a")  # Same approach fails again
        
        assert tracker.failed_approaches["approach_a"] == 2
        assert tracker.failed_approaches["approach_b"] == 1
    
    def test_convergence_metrics(self, tracker):
        """Test generation of convergence metrics."""
        # Initially no metrics
        metrics = tracker.get_convergence_metrics()
        assert metrics["status"] == "not_started"
        
        # Add some iterations
        for i in range(3):
            state = IterationState(
                iteration=i, timestamp=time.time(),
                issues=[Issue(kind="test", message=f"issue_{j}", severity=2) for j in range(10 - i)],
                code_snapshot={}, error_fingerprint=f"fp_{i}", changes_applied=[]
            )
            tracker.should_continue(state)
        
        metrics = tracker.get_convergence_metrics()
        assert metrics["status"] == "in_progress"
        assert metrics["iterations"] == 3
        assert metrics["initial_issues"] == 10
        assert metrics["current_issues"] == 8
        assert metrics["issues_resolved"] == 2


class TestPatchValidator:
    """Test the PatchValidator class."""
    
    @pytest.fixture
    def validator(self):
        """Create a patch validator for testing."""
        return PatchValidator(max_risk_level=RiskLevel.HIGH)
    
    def test_syntax_validation_success(self, validator):
        """Test successful syntax validation."""
        patch = PatchInfo(
            file_path="test.py",
            original_content="def old(): pass",
            patched_content="def new():\n    return 42",
            description="Replace function",
            issues_addressed=[]
        )
        
        assert validator.validate_patch(patch, []) is True
    
    def test_syntax_validation_failure(self, validator):
        """Test syntax validation failure."""
        patch = PatchInfo(
            file_path="test.py",
            original_content="def old(): pass",
            patched_content="def new(\n    # Missing closing parenthesis",
            description="Invalid syntax",
            issues_addressed=[]
        )
        
        with pytest.raises(SyntaxValidationError):
            validator.validate_patch(patch, [])
    
    def test_risk_assessment_low(self, validator):
        """Test low risk assessment."""
        patch = PatchInfo(
            file_path="test.py",
            original_content="def test(): pass",
            patched_content="def test():\n    return 'hello'",
            description="Add return statement",
            issues_addressed=[]
        )
        
        validator.validate_patch(patch, [])
        assert patch.estimated_risk == RiskLevel.LOW
    
    def test_risk_assessment_high(self, validator):
        """Test high risk assessment."""
        # Large change with core functionality
        large_content = "def main():\n" + "    print('line')\n" * 60  # 60+ lines
        patch = PatchInfo(
            file_path="main.py",
            original_content="def main(): pass",
            patched_content=large_content,
            description="Large main function change",
            issues_addressed=[]
        )
        
        validator.validate_patch(patch, [])
        assert patch.estimated_risk in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert "Large change set" in patch.risk_factors
        assert "Core functionality modified" in patch.risk_factors
    
    def test_risk_threshold_enforcement(self):
        """Test risk threshold enforcement."""
        strict_validator = PatchValidator(max_risk_level=RiskLevel.LOW)
        
        # Medium risk patch should fail with strict validator
        patch = PatchInfo(
            file_path="test.py",
            original_content="x = 1",
            patched_content="import os\nimport sys\nx = 1\ny = 2",  # New imports = medium risk
            description="Add imports",
            issues_addressed=[]
        )
        
        with pytest.raises(RiskThresholdError):
            strict_validator.validate_patch(patch, [])
    
    def test_reintroduction_detection(self, validator):
        """Test detection of potentially reintroduced issues."""
        previous_issues = [
            Issue(kind="phantom_function", message="Empty function", severity=2, 
                  file="test.py", line=5)
        ]
        
        # Patch that might reintroduce phantom function issue
        patch = PatchInfo(
            file_path="test.py",
            original_content="def working(): return 42",
            patched_content="def working(): pass",  # Back to phantom
            description="Simplify function",
            issues_addressed=[]
        )
        
        validator.validate_patch(patch, previous_issues)
        assert "Might reintroduce previous issues" in patch.risk_factors


class TestFixOrchestrator:
    """Test the FixOrchestrator class."""
    
    @pytest.fixture
    def setup_orchestrator(self):
        """Set up orchestrator with dependencies."""
        tracker = ConvergenceTracker(max_iterations=5)
        validator = PatchValidator()
        orchestrator = FixOrchestrator(tracker, validator)
        return orchestrator, tracker, validator
    
    @pytest.fixture
    def temp_workspace(self):
        """Create temporary workspace for testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test files
            (workspace / "test.py").write_text("def original(): pass\n")
            (workspace / "other.py").write_text("x = 1\n")
            
            yield workspace
    
    def test_fix_sequence_success(self, setup_orchestrator, temp_workspace):
        """Test successful fix sequence execution."""
        orchestrator, _, _ = setup_orchestrator
        
        fixes = [
            PatchInfo(
                file_path="test.py",
                original_content="def original(): pass\n",
                patched_content="def original():\n    return 42\n",
                description="Add return value",
                issues_addressed=[Issue(kind="phantom_function", message="Empty", severity=2)]
            )
        ]
        
        success, messages = orchestrator.execute_fix_sequence(fixes, temp_workspace)
        
        assert success is True
        assert len(messages) > 0
        assert "Applied fix" in messages[0]
        
        # Verify file was modified
        content = (temp_workspace / "test.py").read_text()
        assert "return 42" in content
    
    def test_syntax_error_rollback(self, setup_orchestrator, temp_workspace):
        """Test rollback on syntax error."""
        orchestrator, _, _ = setup_orchestrator
        
        original_content = (temp_workspace / "test.py").read_text()
        
        fixes = [
            PatchInfo(
                file_path="test.py",
                original_content=original_content,
                patched_content="def broken(\n    # Invalid syntax",
                description="Break syntax",
                issues_addressed=[]
            )
        ]
        
        success, messages = orchestrator.execute_fix_sequence(fixes, temp_workspace)
        
        assert success is False
        
        # Verify file was rolled back to original
        content = (temp_workspace / "test.py").read_text()
        assert content == original_content
    
    def test_rollback_snapshot_creation(self, setup_orchestrator, temp_workspace):
        """Test creation and use of rollback snapshots."""
        orchestrator, _, _ = setup_orchestrator
        
        # Create initial snapshot
        orchestrator._create_rollback_snapshot(0, temp_workspace)
        
        # Modify files
        (temp_workspace / "test.py").write_text("def modified(): pass\n")
        
        # Rollback to snapshot
        orchestrator._perform_rollback(0, temp_workspace)
        
        # Verify rollback worked
        content = (temp_workspace / "test.py").read_text()
        assert "def original(): pass" in content
    
    def test_decision_logging(self, setup_orchestrator, temp_workspace):
        """Test decision logging functionality."""
        orchestrator, _, _ = setup_orchestrator
        
        fixes = [
            PatchInfo(
                file_path="test.py",
                original_content="def original(): pass\n",
                patched_content="def original():\n    return 42\n",
                description="Test fix",
                issues_addressed=[]
            )
        ]
        
        orchestrator.execute_fix_sequence(fixes, temp_workspace)
        
        decision_log = orchestrator.get_decision_log()
        assert len(decision_log) > 0
        
        # Should have at least one apply_fix decision
        apply_decisions = [d for d in decision_log if d["decision_type"] == "apply_fix"]
        assert len(apply_decisions) > 0


class TestIntegration:
    """Integration tests for the complete convergence system."""
    
    def test_factory_function(self):
        """Test the factory function creates proper system."""
        tracker, validator, orchestrator = create_convergence_system(
            max_iterations=10,
            max_risk_level=RiskLevel.MEDIUM,
            test_command="echo 'test'"
        )
        
        assert tracker.max_iterations == 10
        assert validator.max_risk_level == RiskLevel.MEDIUM
        assert orchestrator.test_command == "echo 'test'"
        assert orchestrator.convergence_tracker is tracker
        assert orchestrator.patch_validator is validator
    
    def test_end_to_end_scenario(self):
        """Test a complete end-to-end scenario."""
        with tempfile.TemporaryDirectory() as temp_dir:
            workspace = Path(temp_dir)
            
            # Create test file with phantom function
            test_file = workspace / "example.py"
            test_file.write_text("""
def process_data(data):
    pass

def helper():
    raise NotImplementedError("TODO")
""")
            
            # Create convergence system
            tracker, validator, orchestrator = create_convergence_system(
                max_iterations=3,
                max_risk_level=RiskLevel.HIGH
            )
            
            # Create fixes
            fixes = [
                PatchInfo(
                    file_path="example.py",
                    original_content=test_file.read_text(),
                    patched_content="""
def process_data(data):
    return [item.upper() for item in data if item]

def helper():
    return "implemented"
""",
                    description="Implement phantom functions",
                    issues_addressed=[
                        Issue(kind="phantom_function", message="process_data", severity=2),
                        Issue(kind="phantom_function", message="helper", severity=2)
                    ]
                )
            ]
            
            # Execute fixes
            success, messages = orchestrator.execute_fix_sequence(fixes, workspace)
            
            assert success is True
            
            # Verify fixes were applied
            final_content = test_file.read_text()
            assert "pass" not in final_content
            assert "NotImplementedError" not in final_content
            assert "return" in final_content
            
            # Check convergence metrics
            metrics = tracker.get_convergence_metrics()
            assert metrics["status"] in ["not_started", "in_progress"]  # Depends on implementation details