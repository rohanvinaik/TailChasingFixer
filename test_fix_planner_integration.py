#!/usr/bin/env python3
"""
Test script to verify fix planner integration.
"""

import tempfile
from pathlib import Path
from tailchasing.core.fix_planner import FixPlanner, FixScriptGenerator
from tailchasing.core.issues import Issue

def test_fix_planner():
    """Test that the fix planner generates a valid script."""
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        root_dir = Path(tmpdir)
        
        # Create test file with issues
        test_file = root_dir / "test.py"
        test_file.write_text("""
def duplicate_function():
    return 1

def placeholder_function():
    pass
    
def another_duplicate():
    return 2
""")
        
        # Create another file with duplicates
        test_file2 = root_dir / "test2.py"
        test_file2.write_text("""
def duplicate_function():
    return 1
    
def another_duplicate():
    return 2
""")
        
        # Create issues
        issues = [
            Issue(
                kind="duplicate_function",
                message="Duplicate function 'duplicate_function'",
                severity=3,
                file=str(test_file),
                line=2,
                symbol="duplicate_function",
                evidence={"signature": "def duplicate_function()"}
            ),
            Issue(
                kind="duplicate_function",
                message="Duplicate function 'duplicate_function'",
                severity=3,
                file=str(test_file2),
                line=2,
                symbol="duplicate_function",
                evidence={"signature": "def duplicate_function()"}
            ),
            Issue(
                kind="placeholder",
                message="Placeholder function detected",
                severity=2,
                file=str(test_file),
                line=5,
                symbol="placeholder_function"
            ),
            Issue(
                kind="missing_symbol",
                message="Missing symbol 'process_data'",
                severity=2,
                file=str(test_file),
                line=10,
                symbol="process_data"
            )
        ]
        
        # Create fix planner
        planner = FixPlanner(
            root_dir=root_dir,
            backup_dir=root_dir / "backups",
            interactive=False,
            dry_run=True
        )
        
        # Create fix plan
        plan = planner.create_fix_plan(issues)
        
        print("Fix Plan Summary:")
        print("=" * 50)
        print(plan.get_summary())
        
        print("\nGenerated Actions:")
        print("-" * 50)
        for i, action in enumerate(plan.actions, 1):
            print(f"{i}. {action.action_type}: {action.description}")
            print(f"   Confidence: {action.confidence:.1%}")
            print(f"   Target: {action.target_file}")
            print()
        
        # Generate executable script
        script = plan.get_executable_script()
        
        # Save script
        script_path = root_dir / "fix_plan.py"
        script_path.write_text(script)
        
        print(f"Generated fix script: {script_path}")
        print(f"Script size: {len(script)} bytes")
        
        # Verify script is valid Python
        try:
            compile(script, "fix_plan.py", "exec")
            print("✓ Script is valid Python")
        except SyntaxError as e:
            print(f"✗ Script has syntax error: {e}")
            return False
        
        return True

if __name__ == "__main__":
    success = test_fix_planner()
    if success:
        print("\n✅ Fix planner integration test passed!")
    else:
        print("\n❌ Fix planner integration test failed!")
        exit(1)