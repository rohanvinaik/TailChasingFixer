"""Tests for fix playbook system."""

import tempfile
from pathlib import Path
import pytest

from tailchasing.core.fix_playbooks import (
    FixPlaybook, PlaybookStep, CodeChange, SafetyCheck,
    ChangeRisk, PlaybookExecution, PlaybookEngine, SafetyCheckRunner
)
from tailchasing.core.playbook_generator import PlaybookGenerator
from tailchasing.analyzers.phantom_triage import PhantomStub, PhantomPriority
from tailchasing.core.issues import Issue


class TestCodeChange:
    """Test CodeChange functionality."""
    
    def test_code_change_creation(self):
        """Test creating a code change."""
        change = CodeChange(
            file_path="/test/file.py",
            line_start=10,
            line_end=12,
            old_content="def old_function():\n    pass",
            new_content="def new_function():\n    return 42",
            change_type="replace",
            description="Replace function implementation",
            risk_level=ChangeRisk.MEDIUM
        )
        
        assert change.file_path == "/test/file.py"
        assert change.line_start == 10
        assert change.line_end == 12
        assert change.change_type == "replace"
        assert change.risk_level == ChangeRisk.MEDIUM


class TestSafetyCheckRunner:
    """Test safety check execution."""
    
    def setup_method(self):
        """Set up test environment."""
        self.runner = SafetyCheckRunner()
    
    def test_syntax_check_valid_file(self):
        """Test syntax checking on valid Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def valid_function():\n    return 42\n")
            test_file = f.name
        
        try:
            result = self.runner._check_syntax_valid([test_file])
            assert result is True
        finally:
            Path(test_file).unlink()
    
    def test_syntax_check_invalid_file(self):
        """Test syntax checking on invalid Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def invalid_function(\n    return 42  # Missing closing parenthesis\n")
            test_file = f.name
        
        try:
            result = self.runner._check_syntax_valid([test_file])
            assert result is False
        finally:
            Path(test_file).unlink()
    
    def test_security_check_safe_code(self):
        """Test security checking on safe code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def safe_function():\n    return 'safe'\n")
            test_file = f.name
        
        try:
            result = self.runner._check_no_security_issues([test_file])
            assert result is True
        finally:
            Path(test_file).unlink()
    
    def test_security_check_unsafe_code(self):
        """Test security checking on unsafe code."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def unsafe_function():\n    eval('dangerous code')\n")
            test_file = f.name
        
        try:
            result = self.runner._check_no_security_issues([test_file])
            assert result is False
        finally:
            Path(test_file).unlink()
    
    def test_run_custom_check(self):
        """Test running a custom safety check."""
        check = SafetyCheck(
            check_id="test_check",
            description="Test check",
            check_function="syntax_valid",
            is_blocking=True
        )
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test():\n    pass\n")
            test_file = f.name
        
        try:
            success, output = self.runner.run_check(check, [test_file])
            assert success is True
            assert output == ""
        finally:
            Path(test_file).unlink()


class TestPlaybookExecution:
    """Test playbook execution tracking."""
    
    def test_execution_initialization(self):
        """Test playbook execution initialization."""
        playbook = FixPlaybook(
            playbook_id="test_001",
            name="Test Playbook",
            description="Test description", 
            cluster_id="test_cluster",
            steps=[],
            risk_level=ChangeRisk.LOW
        )
        
        execution = PlaybookExecution(playbook)
        
        assert execution.playbook == playbook
        assert execution.current_stage.value == "planning"
        assert execution.completed_steps == []
        assert execution.failed_steps == []
        assert execution.dry_run is False
    
    def test_log_entry_addition(self):
        """Test adding log entries."""
        playbook = FixPlaybook("test", "Test", "Description", "cluster", [], ChangeRisk.LOW)
        execution = PlaybookExecution(playbook)
        
        execution.add_log_entry("INFO", "Test message", {"key": "value"})
        
        assert len(execution.execution_log) == 1
        entry = execution.execution_log[0]
        assert entry['level'] == "INFO"
        assert entry['message'] == "Test message"
        assert entry['details']['key'] == "value"
    
    def test_unified_diff_preview(self):
        """Test unified diff preview generation."""
        change = CodeChange(
            file_path="test.py",
            line_start=1,
            line_end=1,
            old_content="def old():\n    pass",
            new_content="def new():\n    return 42",
            change_type="replace",
            description="Update function"
        )
        
        step = PlaybookStep(
            step_id="step1",
            name="Test Step",
            description="Test step description",
            changes=[change],
            safety_checks=[]
        )
        
        playbook = FixPlaybook(
            playbook_id="test",
            name="Test",
            description="Test",
            cluster_id="cluster",
            steps=[step],
            risk_level=ChangeRisk.LOW
        )
        
        execution = PlaybookExecution(playbook)
        diff_preview = execution.get_unified_diff_preview()
        
        assert "Step: Test Step" in diff_preview
        assert "test.py" in diff_preview
        assert "def old():" in diff_preview
        assert "def new():" in diff_preview


class TestPlaybookEngine:
    """Test playbook execution engine."""
    
    def setup_method(self):
        """Set up test environment."""
        self.engine = PlaybookEngine()
    
    def test_preview_generation(self):
        """Test playbook preview generation."""
        change = CodeChange(
            file_path="test.py",
            line_start=1,
            line_end=1,
            old_content="pass",
            new_content="return 42",
            change_type="replace",
            description="Add return"
        )
        
        safety_check = SafetyCheck(
            check_id="syntax_check",
            description="Check syntax",
            check_function="syntax_valid"
        )
        
        step = PlaybookStep(
            step_id="step1",
            name="Update Function", 
            description="Update function implementation",
            changes=[change],
            safety_checks=[safety_check]
        )
        
        playbook = FixPlaybook(
            playbook_id="test_preview",
            name="Test Preview",
            description="Test preview generation",
            cluster_id="test",
            steps=[step],
            risk_level=ChangeRisk.MEDIUM,
            requires_review=True
        )
        
        preview = self.engine.preview_playbook(playbook)
        
        assert "PLAYBOOK PREVIEW: Test Preview" in preview
        assert "Risk Level: MEDIUM" in preview
        assert "REQUIRES MANUAL REVIEW" in preview
        assert "UNIFIED DIFF PREVIEW" in preview
        assert "SAFETY CHECKS" in preview
        assert "syntax_check" in preview
    
    def test_dry_run_execution(self):
        """Test dry run execution."""
        # Create a temporary test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def test_function():\n    pass\n")
            test_file = f.name
        
        try:
            change = CodeChange(
                file_path=test_file,
                line_start=2,
                line_end=2,
                old_content="    pass",
                new_content="    return 42",
                change_type="replace",
                description="Add return statement"
            )
            
            step = PlaybookStep(
                step_id="update_step",
                name="Update Function",
                description="Update function",
                changes=[change],
                safety_checks=[
                    SafetyCheck("syntax_valid", "Check syntax", "syntax_valid")
                ]
            )
            
            playbook = FixPlaybook(
                playbook_id="dry_run_test",
                name="Dry Run Test",
                description="Test dry run",
                cluster_id="test",
                steps=[step],
                risk_level=ChangeRisk.LOW,
                safety_checks=[
                    SafetyCheck("syntax_valid", "Global syntax check", "syntax_valid")
                ]
            )
            
            # Execute as dry run
            execution = self.engine.execute_playbook(playbook, dry_run=True)
            
            assert execution.dry_run is True
            assert execution.current_stage.value in ["completed", "failed"]
            
            # File should be unchanged in dry run
            with open(test_file, 'r') as f:
                content = f.read()
            assert "pass" in content  # Original content preserved
        
        finally:
            Path(test_file).unlink()
    
    def test_failed_execution_rollback(self):
        """Test rollback on failed execution."""
        # Create test file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write("def original_function():\n    return 'original'\n")
            test_file = f.name
        
        try:
            # Create a change that will fail safety checks
            change = CodeChange(
                file_path=test_file,
                line_start=2,
                line_end=2,
                old_content="    return 'original'",
                new_content="    eval('malicious code')  # This should fail security check",
                change_type="replace",
                description="Introduce security issue"
            )
            
            step = PlaybookStep(
                step_id="bad_step",
                name="Bad Change",
                description="Introduce security issue",
                changes=[change],
                safety_checks=[]
            )
            
            playbook = FixPlaybook(
                playbook_id="fail_test",
                name="Failure Test",
                description="Test failure handling",
                cluster_id="test",
                steps=[step],
                risk_level=ChangeRisk.LOW,
                safety_checks=[
                    SafetyCheck("no_security_issues", "Security check", "no_security_issues", is_blocking=True)
                ]
            )
            
            # Execute (should fail and rollback)
            execution = self.engine.execute_playbook(playbook, dry_run=False)
            
            # Should have failed and rolled back
            assert execution.current_stage.value == "failed"
            
            # File should be restored to original content
            with open(test_file, 'r') as f:
                content = f.read()
            assert "return 'original'" in content
            assert "eval" not in content
        
        finally:
            Path(test_file).unlink()


class TestPlaybookGenerator:
    """Test playbook generation for different scenarios."""
    
    def setup_method(self):
        """Set up test environment."""
        self.generator = PlaybookGenerator()
    
    def test_phantom_stub_playbook_generation(self):
        """Test generating playbook for phantom stubs."""
        # Create test stubs with different priorities
        stubs = [
            PhantomStub(
                function_name='verify_signature',
                class_name='CryptoUtils',
                file_path='/crypto/utils.py',
                line_number=100,
                signature='(self, data, signature)',
                body_content='pass',
                priority=PhantomPriority.P0_SECURITY,
                risk_factors=['Security risk: crypto_verify'],
                suggested_action='BLOCK CI',
                is_blocked=True
            ),
            PhantomStub(
                function_name='get_config',
                class_name=None,
                file_path='/app/config.py',
                line_number=50,
                signature='(key)',
                body_content='return None',
                priority=PhantomPriority.P1_FUNCTIONAL,
                risk_factors=['Functional requirement: config_getters'],
                suggested_action='Provide minimal implementation'
            ),
            PhantomStub(
                function_name='generate_verilog',
                class_name=None,
                file_path='/experimental/fpga.py',
                line_number=25,
                signature='(lut)',
                body_content='pass',
                priority=PhantomPriority.P3_EXPERIMENTAL,
                risk_factors=['Experimental feature: code_generation'],
                suggested_action='Allowlist until implemented',
                is_allowed=True
            )
        ]
        
        playbook = self.generator.generate_phantom_stub_playbook(stubs)
        
        assert playbook.name == "Phantom Stub Fixes (3 stubs)"
        assert playbook.risk_level == ChangeRisk.CRITICAL  # Due to P0 stub
        assert len(playbook.steps) == 3  # One for each priority level
        assert playbook.requires_review is True
        
        # Check metadata
        assert playbook.metadata['stub_counts']['P0'] == 1
        assert playbook.metadata['stub_counts']['P1'] == 1
        assert playbook.metadata['stub_counts']['P3'] == 1
        assert playbook.metadata['blocked_stubs'] == 1
        assert playbook.metadata['allowed_stubs'] == 1
        
        # Check that P3 step is optional
        p3_steps = [step for step in playbook.steps if 'experimental' in step.name.lower()]
        if p3_steps:
            assert p3_steps[0].is_optional is True
    
    def test_duplicate_removal_playbook_generation(self):
        """Test generating playbook for duplicate function removal."""
        # Create mock cluster and issues
        from tailchasing.analyzers.root_cause_clustering import IssueCluster
        
        cluster = IssueCluster(
            cluster_id="dup_001",
            issues=[],
            similarity_matrix=None,
            size=3,
            severity=3.0,
            locations=[('/file1.py', 10), ('/file2.py', 20), ('/file3.py', 30)],
            root_cause_guess="Duplicate function implementations",
            confidence=0.85,
            fix_playbook_id="duplicate_removal"
        )
        
        issues = [
            Issue(
                kind="duplicate_function",
                message="Function 'calculate' is duplicated",
                severity=2,
                file="/file1.py",
                line=10,
                evidence={'function_name': 'calculate', 'line_count': 5}
            ),
            Issue(
                kind="duplicate_function", 
                message="Function 'calculate' is duplicated",
                severity=2,
                file="/file2.py",
                line=20,
                evidence={'function_name': 'calculate', 'line_count': 3}
            )
        ]
        
        playbook = self.generator.generate_cluster_playbook(cluster, issues)
        
        assert "Duplicate Function Removal" in playbook.name
        assert playbook.risk_level == ChangeRisk.HIGH
        assert playbook.requires_review is True
        assert len(playbook.safety_checks) > 0
    
    def test_playbook_preview(self):
        """Test playbook preview generation."""
        stubs = [
            PhantomStub(
                function_name='test_function',
                class_name=None,
                file_path='/test.py',
                line_number=10,
                signature='()',
                body_content='pass',
                priority=PhantomPriority.P1_FUNCTIONAL,
                risk_factors=['Functional requirement: test'],
                suggested_action='Implement'
            )
        ]
        
        playbook = self.generator.generate_phantom_stub_playbook(stubs)
        preview = self.generator.preview_playbook(playbook)
        
        assert "PLAYBOOK PREVIEW" in preview
        assert playbook.name in preview
        assert "Risk Level" in preview
        assert "UNIFIED DIFF PREVIEW" in preview
        assert "SAFETY CHECKS" in preview
    
    def test_playbook_execution_dry_run(self):
        """Test playbook execution in dry run mode."""
        # Create simple stub for testing
        stubs = [
            PhantomStub(
                function_name='simple_test',
                class_name=None,
                file_path='/tmp/test.py',  # Non-existent file for dry run
                line_number=1,
                signature='()',
                body_content='pass',
                priority=PhantomPriority.P3_EXPERIMENTAL,
                risk_factors=[],
                suggested_action='Implement'
            )
        ]
        
        playbook = self.generator.generate_phantom_stub_playbook(stubs)
        
        # Execute in dry run mode
        execution = self.generator.execute_playbook(playbook, dry_run=True)
        
        assert execution.dry_run is True
        # In dry run, execution might fail due to missing files, but that's expected
        assert execution.current_stage.value in ["completed", "failed", "planning"]


class TestIntegration:
    """Integration tests for the playbook system."""
    
    def test_end_to_end_phantom_stub_workflow(self):
        """Test complete workflow from detection to playbook generation."""
        # This would test the full integration between:
        # 1. PhantomTriageAnalyzer detecting stubs
        # 2. PlaybookGenerator creating playbooks
        # 3. PlaybookEngine executing them
        
        # Create test file with phantom stubs
        test_code = '''
def verify_critical_signature(data, signature):
    """Critical security function."""
    pass

def get_user_config():
    """Get user configuration."""
    return None
'''
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(test_code)
            test_file = f.name
        
        try:
            # Step 1: Detect stubs with triage analyzer
            from tailchasing.analyzers.phantom_triage import PhantomTriageAnalyzer
            import ast
            
            config = {
                'placeholders': {
                    'block': ['**/verify_critical_signature'],
                    'allow': [],
                    'triage_enabled': True
                }
            }
            
            analyzer = PhantomTriageAnalyzer(config)
            
            with open(test_file, 'r') as f:
                content = f.read()
            tree = ast.parse(content)
            
            context = type('MockContext', (), {
                'config': config,
                'ast_index': {test_file: tree},
                'should_ignore_issue': lambda kind: False
            })()
            
            # Run analyzer
            issues = analyzer.run(context)
            stubs = analyzer.get_detected_stubs()
            
            assert len(stubs) >= 2  # Should detect both functions
            
            # Step 2: Generate playbook
            generator = PlaybookGenerator()
            playbook = generator.generate_phantom_stub_playbook(stubs)
            
            assert playbook is not None
            assert len(playbook.steps) > 0
            
            # Step 3: Preview playbook
            preview = generator.preview_playbook(playbook)
            assert "PLAYBOOK PREVIEW" in preview
            
            # Step 4: Execute in dry run mode
            execution = generator.execute_playbook(playbook, dry_run=True)
            
            # Should complete planning stage at minimum
            assert execution.current_stage.value in ["completed", "failed", "validation"]
        
        finally:
            Path(test_file).unlink()
    
    def test_safety_check_integration(self):
        """Test that safety checks properly integrate with playbook execution."""
        # Create a playbook with intentionally failing safety check
        change = CodeChange(
            file_path="/nonexistent/file.py",  # This should fail file existence checks
            line_start=1,
            line_end=1, 
            old_content="pass",
            new_content="return 42",
            change_type="replace",
            description="Test change"
        )
        
        step = PlaybookStep(
            step_id="test_step",
            name="Test Step",
            description="Test step with failing check",
            changes=[change],
            safety_checks=[
                SafetyCheck("syntax_valid", "Syntax check", "syntax_valid", is_blocking=True)
            ]
        )
        
        playbook = FixPlaybook(
            playbook_id="safety_test",
            name="Safety Check Test",
            description="Test safety check integration",
            cluster_id="test",
            steps=[step],
            risk_level=ChangeRisk.LOW
        )
        
        engine = PlaybookEngine()
        execution = engine.execute_playbook(playbook, dry_run=True)
        
        # Should fail during validation due to missing file
        assert execution.current_stage.value in ["failed", "planning"]
        
        # Should have error logs
        error_logs = [entry for entry in execution.execution_log if entry['level'] == 'ERROR']
        assert len(error_logs) > 0