"""
Tests for the fix planner module.
"""

import ast
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest

from tailchasing.core.fix_planner import (
    FixPlanner,
    FixAction,
    FixPlan,
    FixScriptGenerator,
    InteractiveFixReviewer
)
from tailchasing.core.issues import Issue


class TestFixAction:
    """Test the FixAction dataclass."""
    
    def test_fix_action_creation(self):
        """Test creating a fix action."""
        action = FixAction(
            action_type="remove_function",
            target_file="test.py",
            description="Remove duplicate function",
            confidence=0.8,
            metadata={"function_name": "test_func"}
        )
        
        assert action.action_type == "remove_function"
        assert action.target_file == "test.py"
        assert action.confidence == 0.8
        assert action.metadata["function_name"] == "test_func"
        
    def test_fix_action_with_content(self):
        """Test fix action with old and new content."""
        action = FixAction(
            action_type="replace_function",
            target_file="test.py",
            description="Replace stub",
            confidence=0.6,
            old_content="def test():\n    pass",
            new_content="def test():\n    return 42"
        )
        
        assert action.old_content == "def test():\n    pass"
        assert action.new_content == "def test():\n    return 42"


class TestFixPlan:
    """Test the FixPlan dataclass."""
    
    def test_fix_plan_creation(self):
        """Test creating a fix plan."""
        actions = [
            FixAction("test", "file.py", "Test action", 0.8)
        ]
        issues = [
            Issue(kind="test", message="Test issue", severity=2)
        ]
        
        plan = FixPlan(
            actions=actions,
            issues_addressed=issues,
            estimated_risk="low",
            total_confidence=0.8,
            backup_dir="/tmp/backups"
        )
        
        assert len(plan.actions) == 1
        assert len(plan.issues_addressed) == 1
        assert plan.estimated_risk == "low"
        assert plan.total_confidence == 0.8
        
    def test_fix_plan_summary(self):
        """Test generating fix plan summary."""
        actions = [
            FixAction("remove_function", "file1.py", "Remove func1", 0.8),
            FixAction("remove_function", "file2.py", "Remove func2", 0.7),
            FixAction("add_symbol", "file3.py", "Add symbol", 0.9)
        ]
        
        plan = FixPlan(
            actions=actions,
            issues_addressed=[],
            estimated_risk="medium",
            total_confidence=0.8,
            backup_dir="/tmp/backup"
        )
        
        summary = plan.get_summary()
        
        assert "Fix Plan Summary" in summary
        assert "Total actions: 3" in summary
        assert "Estimated risk: medium" in summary
        assert "remove_function: 2" in summary
        assert "add_symbol: 1" in summary
        
    def test_executable_script_generation(self):
        """Test generating executable Python script."""
        actions = [
            FixAction(
                action_type="remove_function",
                target_file="test.py",
                description="Remove duplicate",
                confidence=0.8,
                metadata={"function_name": "duplicate_func", "line_number": 10}
            )
        ]
        
        plan = FixPlan(
            actions=actions,
            issues_addressed=[],
            estimated_risk="low",
            total_confidence=0.8,
            backup_dir="/tmp/backup"
        )
        
        script = plan.get_executable_script()
        
        assert "#!/usr/bin/env python3" in script
        assert "import os" in script
        assert "import shutil" in script
        assert "def remove_function(" in script
        assert "def main():" in script
        assert 'if __name__ == "__main__"' in script


class TestFixPlanner:
    """Test the FixPlanner class."""
    
    def test_fix_planner_initialization(self):
        """Test initializing fix planner."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            planner = FixPlanner(root_dir)
            
            assert planner.root_dir == root_dir
            assert planner.backup_dir == root_dir / ".tailchasing_backups"
            assert not planner.interactive
            assert not planner.dry_run
            
    def test_create_fix_plan_empty(self):
        """Test creating fix plan with no issues."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            plan = planner.create_fix_plan([])
            
            assert len(plan.actions) == 0
            assert len(plan.issues_addressed) == 0
            assert plan.estimated_risk == "low"
            
    def test_plan_duplicate_fix(self):
        """Test planning fixes for duplicate functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            issues = [
                Issue(
                    kind="duplicate_function",
                    message="Duplicate function found",
                    severity=3,
                    file="file1.py",
                    line=10,
                    symbol="test_func",
                    evidence={"signature": "def test_func()"}
                ),
                Issue(
                    kind="duplicate_function",
                    message="Duplicate function found",
                    severity=3,
                    file="file2.py",
                    line=20,
                    symbol="test_func",
                    evidence={"signature": "def test_func()"}
                )
            ]
            
            plan = planner.create_fix_plan(issues)
            
            # Should have actions to remove duplicate and update imports
            assert len(plan.actions) > 0
            assert any(a.action_type == "remove_function" for a in plan.actions)
            assert any(a.action_type == "update_imports" for a in plan.actions)
            
    def test_plan_missing_symbol_fix(self):
        """Test planning fixes for missing symbols."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            issues = [
                Issue(
                    kind="missing_symbol",
                    message="Missing symbol 'process_data'",
                    severity=2,
                    file="main.py",
                    line=15,
                    symbol="process_data",
                    evidence={"usage_context": "process_data(items)"}
                )
            ]
            
            plan = planner.create_fix_plan(issues)
            
            assert len(plan.actions) > 0
            assert any(a.action_type == "add_symbol" for a in plan.actions)
            
    def test_plan_placeholder_fix(self):
        """Test planning fixes for placeholder functions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            issues = [
                Issue(
                    kind="placeholder",
                    message="Placeholder function detected",
                    severity=2,
                    file="validators.py",
                    line=30,
                    symbol="validate_input",
                    evidence={"pattern": "validation", "signature": "def validate_input(data)"}
                )
            ]
            
            plan = planner.create_fix_plan(issues)
            
            assert len(plan.actions) > 0
            assert any(a.action_type == "implement_placeholder" for a in plan.actions)
            
    def test_plan_circular_import_fix(self):
        """Test planning fixes for circular imports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            issues = [
                Issue(
                    kind="circular_import",
                    message="Circular import detected",
                    severity=3,
                    file="module1.py",
                    evidence={"cycle": ["module1.py", "module2.py", "module3.py"]}
                )
            ]
            
            plan = planner.create_fix_plan(issues)
            
            assert len(plan.actions) > 0
            assert any(a.action_type == "lazy_import" for a in plan.actions)
            
    def test_risk_estimation(self):
        """Test risk level estimation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            # Test low risk
            actions = [
                FixAction("add_symbol", "file.py", "Add", 0.9),
                FixAction("remove_import", "file.py", "Remove", 0.9)
            ]
            risk = planner._estimate_risk(actions)
            assert risk == "low"
            
            # Test high risk
            actions = [
                FixAction("remove_function", "file.py", "Remove", 0.5),
                FixAction("remove_function", "file2.py", "Remove", 0.5),
                FixAction("remove_function", "file3.py", "Remove", 0.5)
            ]
            risk = planner._estimate_risk(actions)
            assert risk == "high"
            
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            actions = [
                FixAction("test1", "file.py", "Test", 0.8),
                FixAction("test2", "file.py", "Test", 0.6),
                FixAction("test3", "file.py", "Test", 0.7)
            ]
            
            confidence = planner._calculate_total_confidence(actions)
            assert confidence == pytest.approx(0.7, rel=0.01)
            
    def test_action_deduplication(self):
        """Test deduplication of actions."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            actions = [
                FixAction("remove_function", "file.py", "Remove func", 0.8,
                         metadata={"function_name": "test"}),
                FixAction("remove_function", "file.py", "Remove func", 0.7,
                         metadata={"function_name": "test"}),  # Duplicate
                FixAction("add_symbol", "file.py", "Add symbol", 0.9,
                         metadata={"function_name": "other"})
            ]
            
            deduplicated = planner._deduplicate_actions(actions)
            assert len(deduplicated) == 2
            
    def test_action_ordering(self):
        """Test ordering of actions by priority."""
        with tempfile.TemporaryDirectory() as tmpdir:
            planner = FixPlanner(Path(tmpdir))
            
            actions = [
                FixAction("update_imports", "file.py", "Update", 0.8),
                FixAction("remove_import", "file.py", "Remove", 0.9),
                FixAction("add_symbol", "file.py", "Add", 0.7)
            ]
            
            ordered = planner._order_actions(actions)
            
            assert ordered[0].action_type == "remove_import"
            assert ordered[1].action_type == "add_symbol"
            assert ordered[2].action_type == "update_imports"


class TestFixScriptGenerator:
    """Test the FixScriptGenerator class."""
    
    def test_script_structure(self):
        """Test that generated script has proper structure."""
        generator = FixScriptGenerator()
        
        plan = FixPlan(
            actions=[
                FixAction("test", "file.py", "Test action", 0.8)
            ],
            issues_addressed=[],
            estimated_risk="low",
            total_confidence=0.8,
            backup_dir="/tmp/backup"
        )
        
        script = generator.generate(plan)
        
        # Check script structure
        assert script.startswith("#!/usr/bin/env python3")
        assert "import os" in script
        assert "def log(message" in script
        assert "def create_backup(" in script
        assert "def main():" in script
        assert "if __name__" in script
        
    def test_script_includes_actions(self):
        """Test that script includes all actions."""
        generator = FixScriptGenerator()
        
        actions = [
            FixAction(
                "remove_function", "test.py", "Remove", 0.8,
                metadata={"function_name": "test_func", "line_number": 10}
            ),
            FixAction(
                "add_symbol", "utils.py", "Add", 0.9,
                new_content="def helper(): pass"
            )
        ]
        
        plan = FixPlan(
            actions=actions,
            issues_addressed=[],
            estimated_risk="low",
            total_confidence=0.85,
            backup_dir="/tmp/backup"
        )
        
        script = generator.generate(plan)
        
        assert "remove_function(" in script
        assert "add_symbol(" in script
        assert "test.py" in script
        assert "utils.py" in script


class TestInteractiveFixReviewer:
    """Test the InteractiveFixReviewer class."""
    
    def test_reviewer_initialization(self):
        """Test initializing the reviewer."""
        plan = FixPlan(
            actions=[
                FixAction("test", "file.py", "Test", 0.8)
            ],
            issues_addressed=[],
            estimated_risk="low",
            total_confidence=0.8,
            backup_dir="/tmp/backup"
        )
        
        reviewer = InteractiveFixReviewer(plan)
        
        assert reviewer.plan == plan
        assert len(reviewer.approved_actions) == 0
        assert len(reviewer.rejected_actions) == 0
        
    @patch('builtins.input')
    def test_review_approve_action(self, mock_input):
        """Test approving an action during review."""
        mock_input.side_effect = ['y', 'q']  # Approve first, then quit
        
        plan = FixPlan(
            actions=[
                FixAction("test", "file.py", "Test action", 0.8)
            ],
            issues_addressed=[],
            estimated_risk="low",
            total_confidence=0.8,
            backup_dir="/tmp/backup"
        )
        
        reviewer = InteractiveFixReviewer(plan)
        
        with patch('builtins.print'):
            approved, rejected = reviewer.review()
        
        assert len(approved) == 1
        assert len(rejected) == 0
        
    @patch('builtins.input')
    def test_review_reject_action(self, mock_input):
        """Test rejecting an action during review."""
        mock_input.side_effect = ['n', 'q']  # Reject first, then quit
        
        plan = FixPlan(
            actions=[
                FixAction("test", "file.py", "Test action", 0.8)
            ],
            issues_addressed=[],
            estimated_risk="low",
            total_confidence=0.8,
            backup_dir="/tmp/backup"
        )
        
        reviewer = InteractiveFixReviewer(plan)
        
        with patch('builtins.print'):
            approved, rejected = reviewer.review()
        
        assert len(approved) == 0
        assert len(rejected) == 1
        
    @patch('builtins.input')
    def test_review_show_diff(self, mock_input):
        """Test showing diff during review."""
        mock_input.side_effect = ['d', 'y', 'q']  # Show diff, approve, quit
        
        action = FixAction(
            "replace_function", "file.py", "Replace", 0.8,
            old_content="def test():\n    pass",
            new_content="def test():\n    return 42"
        )
        
        plan = FixPlan(
            actions=[action],
            issues_addressed=[],
            estimated_risk="low",
            total_confidence=0.8,
            backup_dir="/tmp/backup"
        )
        
        reviewer = InteractiveFixReviewer(plan)
        
        with patch('builtins.print') as mock_print:
            approved, rejected = reviewer.review()
        
        # Check that diff was shown
        printed = ' '.join(str(call) for call in mock_print.call_args_list)
        assert "Diff:" in printed or "diff" in printed.lower()


class TestIntegration:
    """Integration tests for the fix planner."""
    
    def test_end_to_end_fix_generation(self):
        """Test complete fix generation workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root_dir = Path(tmpdir)
            
            # Create test files
            file1 = root_dir / "module1.py"
            file1.write_text("""
def duplicate_func():
    return 1

def placeholder_func():
    pass
""")
            
            file2 = root_dir / "module2.py"
            file2.write_text("""
def duplicate_func():
    return 1

# Uses missing_func which doesn't exist
result = missing_func()
""")
            
            # Create issues
            issues = [
                Issue(
                    kind="duplicate_function",
                    message="Duplicate function",
                    severity=3,
                    file=str(file1),
                    line=2,
                    symbol="duplicate_func"
                ),
                Issue(
                    kind="duplicate_function",
                    message="Duplicate function",
                    severity=3,
                    file=str(file2),
                    line=2,
                    symbol="duplicate_func"
                ),
                Issue(
                    kind="placeholder",
                    message="Placeholder detected",
                    severity=2,
                    file=str(file1),
                    line=5,
                    symbol="placeholder_func"
                ),
                Issue(
                    kind="missing_symbol",
                    message="Missing symbol",
                    severity=2,
                    file=str(file2),
                    line=5,
                    symbol="missing_func"
                )
            ]
            
            # Create fix plan
            planner = FixPlanner(root_dir)
            plan = planner.create_fix_plan(issues)
            
            # Should have actions for all issue types
            assert len(plan.actions) > 0
            assert len(plan.issues_addressed) == 4
            
            # Generate script
            script = plan.get_executable_script()
            
            # Script should be valid Python
            compile(script, "fix_plan.py", "exec")
            
            # Check script contains expected functions
            assert "def remove_function(" in script
            assert "def add_symbol(" in script
            assert "def main():" in script