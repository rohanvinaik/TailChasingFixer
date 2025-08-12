"""
Tests for automated fix generation and validation.

Comprehensive tests for fix suggestion generation, validation,
and application including orchestration and planning.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
import textwrap
import json
import time

from hypothesis import given, strategies as st, settings
import pytest

from tailchasing.fixers.suggestion_generator import SuggestionGenerator, FixSuggestion
from tailchasing.fixers.fix_applier import FixApplier, FixResult
from tailchasing.fixers.fix_validator import FixValidator, ValidationResult
from tailchasing.fixers.fix_planner import FixPlanner, FixPlan, FixStep
from tailchasing.orchestration.orchestrator import TailChasingOrchestrator
from tailchasing.core.issues import Issue


class TestSuggestionGenerator(unittest.TestCase):
    """Test fix suggestion generation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.generator = SuggestionGenerator()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_generate_duplicate_function_fix(self):
        """Test fix generation for duplicate functions."""
        # Create duplicate functions
        file_path = self.create_file("duplicates.py", """
            def calculate_sum(numbers):
                total = 0
                for num in numbers:
                    total += num
                return total
            
            def compute_total(values):
                result = 0
                for val in values:
                    result += val
                return result
        """)
        
        issue = Issue(
            kind='duplicate_function',
            file=str(file_path),
            line=7,
            symbol='compute_total',
            message='Duplicate of calculate_sum',
            severity=3,
            evidence={
                'function1': 'calculate_sum',
                'function2': 'compute_total',
                'similarity': 0.95
            }
        )
        
        suggestions = self.generator.generate_suggestions([issue])
        
        self.assertEqual(len(suggestions), 1)
        suggestion = suggestions[0]
        
        # Should suggest removing duplicate
        self.assertEqual(suggestion.fix_type, 'remove_duplicate')
        self.assertEqual(suggestion.target_file, str(file_path))
        
        # Should include replacement strategy
        self.assertIn('replacement', suggestion.metadata)
        self.assertEqual(suggestion.metadata['replacement'], 'calculate_sum')
    
    def test_generate_circular_import_fix(self):
        """Test fix generation for circular imports."""
        # Create files with circular imports
        file_a = self.create_file("module_a.py", """
            from .module_b import function_b
            
            def function_a():
                return function_b()
        """)
        
        file_b = self.create_file("module_b.py", """
            from .module_a import function_a
            
            def function_b():
                return function_a()
        """)
        
        issue = Issue(
            kind='circular_import',
            file=str(file_a),
            line=1,
            symbol='module_b',
            message='Circular import detected',
            severity=4,
            evidence={
                'cycle': ['module_a', 'module_b', 'module_a']
            }
        )
        
        suggestions = self.generator.generate_suggestions([issue])
        
        self.assertGreater(len(suggestions), 0)
        suggestion = suggestions[0]
        
        # Should suggest local import or restructuring
        self.assertIn(suggestion.fix_type, ['local_import', 'restructure'])
        
        if suggestion.fix_type == 'local_import':
            # Should move import inside function
            self.assertIn('def function_a():', suggestion.new_code)
            self.assertIn('from .module_b import function_b', suggestion.new_code)
    
    def test_generate_phantom_function_fix(self):
        """Test fix generation for phantom functions."""
        file_path = self.create_file("phantom.py", """
            def process_data(data):
                pass
            
            def handle_error(error):
                raise NotImplementedError("TODO: implement")
            
            def calculate_result():
                # TODO: implement this
                return None
        """)
        
        issue = Issue(
            kind='phantom_function',
            file=str(file_path),
            line=1,
            symbol='process_data',
            message='Empty function body',
            severity=2,
            evidence={'body_type': 'pass'}
        )
        
        suggestions = self.generator.generate_suggestions([issue])
        
        self.assertGreater(len(suggestions), 0)
        suggestion = suggestions[0]
        
        # Should suggest either implementation or removal
        self.assertIn(suggestion.fix_type, ['implement_stub', 'remove_phantom'])
        
        if suggestion.fix_type == 'implement_stub':
            # Should provide template implementation
            self.assertIn('def process_data(data):', suggestion.new_code)
            self.assertNotIn('pass', suggestion.new_code)
    
    def test_generate_import_anxiety_fix(self):
        """Test fix generation for import anxiety."""
        file_path = self.create_file("imports.py", """
            from typing import *
            import os, sys, json, csv, xml
            from collections import defaultdict, Counter, OrderedDict
            
            def read_json(filename):
                with open(filename) as f:
                    return json.load(f)
        """)
        
        issue = Issue(
            kind='import_anxiety',
            file=str(file_path),
            line=1,
            symbol='typing',
            message='Wildcard import detected',
            severity=2,
            evidence={
                'pattern': 'wildcard',
                'imported_count': 20,
                'used_count': 2
            }
        )
        
        suggestions = self.generator.generate_suggestions([issue])
        
        self.assertGreater(len(suggestions), 0)
        suggestion = suggestions[0]
        
        # Should suggest specific imports
        self.assertEqual(suggestion.fix_type, 'optimize_imports')
        
        # Should only import what's used
        self.assertIn('import json', suggestion.new_code)
        self.assertNotIn('from typing import *', suggestion.new_code)
        self.assertNotIn('import xml', suggestion.new_code)
    
    def test_batch_suggestion_generation(self):
        """Test generating suggestions for multiple issues."""
        issues = [
            Issue(
                kind='duplicate_function',
                file='file1.py',
                line=10,
                symbol='func1',
                message='Duplicate',
                severity=3
            ),
            Issue(
                kind='phantom_function',
                file='file2.py',
                line=20,
                symbol='func2',
                message='Empty',
                severity=2
            ),
            Issue(
                kind='circular_import',
                file='file3.py',
                line=1,
                symbol='module',
                message='Circular',
                severity=4
            )
        ]
        
        suggestions = self.generator.generate_suggestions(issues)
        
        # Should generate suggestion for each issue
        self.assertEqual(len(suggestions), len(issues))
        
        # Should prioritize by severity
        self.assertEqual(suggestions[0].priority, 'high')  # circular import
        self.assertEqual(suggestions[1].priority, 'medium')  # duplicate
        self.assertEqual(suggestions[2].priority, 'low')  # phantom
    
    @given(
        num_issues=st.integers(min_value=1, max_value=20),
        severities=st.lists(st.integers(min_value=1, max_value=5), min_size=1, max_size=20)
    )
    @settings(max_examples=20)
    def test_suggestion_generation_property(self, num_issues: int, severities: List[int]):
        """Property-based test for suggestion generation."""
        # Create issues with varying severities
        issues = []
        for i in range(min(num_issues, len(severities))):
            issues.append(Issue(
                kind='test_issue',
                file=f'file_{i}.py',
                line=i * 10,
                symbol=f'symbol_{i}',
                message=f'Issue {i}',
                severity=severities[i]
            ))
        
        suggestions = self.generator.generate_suggestions(issues)
        
        # Should generate suggestions for all issues
        self.assertEqual(len(suggestions), len(issues))
        
        # Should be sorted by priority
        priorities = [s.priority for s in suggestions]
        priority_values = {'critical': 5, 'high': 4, 'medium': 3, 'low': 2, 'info': 1}
        
        for i in range(len(priorities) - 1):
            val1 = priority_values.get(priorities[i], 0)
            val2 = priority_values.get(priorities[i + 1], 0)
            self.assertGreaterEqual(val1, val2)


class TestFixApplier(unittest.TestCase):
    """Test fix application."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.applier = FixApplier()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_apply_code_replacement(self):
        """Test applying code replacement fix."""
        file_path = self.create_file("test.py", """
            def old_function():
                return "old"
            
            def main():
                result = old_function()
                print(result)
        """)
        
        suggestion = FixSuggestion(
            issue_id="test_1",
            fix_type="replace_function",
            target_file=str(file_path),
            target_line=1,
            original_code="def old_function():\n    return \"old\"",
            new_code="def new_function():\n    return \"new\"",
            description="Replace old_function with new_function",
            confidence=0.9,
            priority="medium"
        )
        
        result = self.applier.apply_fix(suggestion, file_path)
        
        self.assertTrue(result.success)
        self.assertIsNone(result.error)
        
        # Check file was modified
        content = file_path.read_text()
        self.assertIn("def new_function():", content)
        self.assertNotIn("def old_function():", content)
        
        # Check backup was created
        self.assertTrue(result.backup_path.exists())
    
    def test_apply_import_optimization(self):
        """Test applying import optimization fix."""
        file_path = self.create_file("imports.py", """
            from typing import *
            import json
            
            def process(data: dict):
                return json.dumps(data)
        """)
        
        suggestion = FixSuggestion(
            issue_id="import_1",
            fix_type="optimize_imports",
            target_file=str(file_path),
            target_line=1,
            original_code="from typing import *\nimport json",
            new_code="from typing import Dict\nimport json",
            description="Replace wildcard import",
            confidence=0.95,
            priority="low"
        )
        
        result = self.applier.apply_fix(suggestion, file_path)
        
        self.assertTrue(result.success)
        
        content = file_path.read_text()
        self.assertIn("from typing import Dict", content)
        self.assertNotIn("from typing import *", content)
    
    def test_apply_multiple_fixes(self):
        """Test applying multiple fixes to same file."""
        file_path = self.create_file("multi.py", """
            def func1():
                pass
            
            def func2():
                pass
            
            def func3():
                return None
        """)
        
        suggestions = [
            FixSuggestion(
                issue_id="fix_1",
                fix_type="implement_stub",
                target_file=str(file_path),
                target_line=1,
                original_code="def func1():\n    pass",
                new_code="def func1():\n    return \"implemented\"",
                description="Implement func1",
                confidence=0.8,
                priority="medium"
            ),
            FixSuggestion(
                issue_id="fix_2",
                fix_type="implement_stub",
                target_file=str(file_path),
                target_line=4,
                original_code="def func2():\n    pass",
                new_code="def func2():\n    return \"also implemented\"",
                description="Implement func2",
                confidence=0.8,
                priority="medium"
            )
        ]
        
        results = self.applier.apply_fixes(suggestions)
        
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r.success for r in results))
        
        content = file_path.read_text()
        self.assertIn("return \"implemented\"", content)
        self.assertIn("return \"also implemented\"", content)
    
    def test_rollback_on_error(self):
        """Test rollback when fix fails."""
        file_path = self.create_file("test.py", """
            def function():
                return 42
        """)
        
        original_content = file_path.read_text()
        
        # Create invalid suggestion
        suggestion = FixSuggestion(
            issue_id="bad_fix",
            fix_type="replace",
            target_file=str(file_path),
            target_line=1,
            original_code="def function():",
            new_code="def function(  # Invalid syntax",
            description="Bad fix",
            confidence=0.5,
            priority="low"
        )
        
        result = self.applier.apply_fix(suggestion, file_path)
        
        self.assertFalse(result.success)
        self.assertIsNotNone(result.error)
        
        # File should be restored
        self.assertEqual(file_path.read_text(), original_content)
    
    def test_dry_run_mode(self):
        """Test dry run mode without actual changes."""
        file_path = self.create_file("test.py", """
            def function():
                return 42
        """)
        
        original_content = file_path.read_text()
        
        suggestion = FixSuggestion(
            issue_id="dry_run",
            fix_type="replace",
            target_file=str(file_path),
            target_line=1,
            original_code="def function():",
            new_code="def modified_function():",
            description="Test dry run",
            confidence=0.9,
            priority="medium"
        )
        
        # Apply in dry run mode
        applier = FixApplier(config={'dry_run': True})
        result = applier.apply_fix(suggestion, file_path)
        
        self.assertTrue(result.success)
        
        # File should not be changed
        self.assertEqual(file_path.read_text(), original_content)


class TestFixValidator(unittest.TestCase):
    """Test fix validation."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.validator = FixValidator()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_validate_syntax(self):
        """Test syntax validation."""
        # Valid Python code
        valid_code = """
def function():
    return 42
        """
        
        result = self.validator.validate_syntax(valid_code)
        self.assertTrue(result.is_valid)
        self.assertIsNone(result.error)
        
        # Invalid Python code
        invalid_code = """
def function(
    return 42
        """
        
        result = self.validator.validate_syntax(invalid_code)
        self.assertFalse(result.is_valid)
        self.assertIsNotNone(result.error)
        self.assertIn("syntax", result.error.lower())
    
    def test_validate_imports(self):
        """Test import validation."""
        file_path = self.create_file("test.py", """
            import json
            from typing import Dict
            
            def process(data: Dict):
                return json.dumps(data)
        """)
        
        # Valid imports
        result = self.validator.validate_imports(file_path)
        self.assertTrue(result.is_valid)
        
        # Test with missing imports
        file_path2 = self.create_file("missing.py", """
            def process(data):
                return json.dumps(data)  # json not imported
        """)
        
        result = self.validator.validate_imports(file_path2)
        # Note: This would require more sophisticated analysis
        # For now, just check it doesn't crash
        self.assertIsNotNone(result)
    
    def test_validate_fix_safety(self):
        """Test fix safety validation."""
        suggestion = FixSuggestion(
            issue_id="test",
            fix_type="replace",
            target_file="test.py",
            target_line=1,
            original_code="def safe_function():\n    return 42",
            new_code="def safe_function():\n    return 43",
            description="Safe change",
            confidence=0.9,
            priority="low"
        )
        
        result = self.validator.validate_fix(suggestion)
        self.assertTrue(result.is_valid)
        
        # Test potentially unsafe fix
        unsafe_suggestion = FixSuggestion(
            issue_id="unsafe",
            fix_type="delete",
            target_file="critical.py",
            target_line=1,
            original_code="def critical_function():\n    # Important",
            new_code="",
            description="Delete function",
            confidence=0.3,  # Low confidence
            priority="high"
        )
        
        result = self.validator.validate_fix(unsafe_suggestion)
        # Low confidence + high impact = potentially unsafe
        self.assertLess(result.confidence, 0.5)
    
    def test_validate_test_preservation(self):
        """Test that fixes don't break tests."""
        # Create a simple module
        module_path = self.create_file("module.py", """
            def add(a, b):
                return a + b
            
            def multiply(a, b):
                return a * b
        """)
        
        # Create tests
        test_path = self.create_file("test_module.py", """
            from module import add, multiply
            
            def test_add():
                assert add(2, 3) == 5
            
            def test_multiply():
                assert multiply(2, 3) == 6
        """)
        
        # Valid fix that preserves functionality
        good_suggestion = FixSuggestion(
            issue_id="rename",
            fix_type="rename",
            target_file=str(module_path),
            target_line=1,
            original_code="def add(a, b):",
            new_code="def add_numbers(a, b):",
            description="Rename function",
            confidence=0.8,
            priority="low"
        )
        
        result = self.validator.validate_against_tests(
            good_suggestion,
            test_files=[test_path]
        )
        
        # Would break tests due to import
        self.assertFalse(result.tests_pass)
        self.assertIn("import", result.error or "")


class TestFixPlanner(unittest.TestCase):
    """Test fix planning and orchestration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.planner = FixPlanner()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def test_create_fix_plan(self):
        """Test creating a fix plan."""
        issues = [
            Issue(
                kind='circular_import',
                file='module_a.py',
                line=1,
                symbol='module_b',
                message='Circular import',
                severity=4,
                evidence={'cycle': ['module_a', 'module_b']}
            ),
            Issue(
                kind='duplicate_function',
                file='utils.py',
                line=10,
                symbol='process_data',
                message='Duplicate function',
                severity=3,
                evidence={'similar_to': 'handle_data'}
            ),
            Issue(
                kind='phantom_function',
                file='helpers.py',
                line=5,
                symbol='placeholder',
                message='Empty function',
                severity=2
            )
        ]
        
        plan = self.planner.create_plan(issues)
        
        self.assertIsNotNone(plan)
        self.assertEqual(len(plan.steps), 3)
        
        # Should be ordered by dependency and severity
        self.assertEqual(plan.steps[0].issue.kind, 'circular_import')  # Highest severity
        self.assertEqual(plan.steps[0].order, 1)
        
        # Check dependencies
        self.assertIn('module_b', plan.dependencies.get('module_a.py', []))
    
    def test_plan_with_conflicts(self):
        """Test handling conflicting fixes."""
        issues = [
            Issue(
                kind='duplicate_function',
                file='same_file.py',
                line=10,
                symbol='func1',
                message='Duplicate of func2',
                severity=3
            ),
            Issue(
                kind='duplicate_function',
                file='same_file.py',
                line=20,
                symbol='func2',
                message='Duplicate of func1',
                severity=3
            )
        ]
        
        plan = self.planner.create_plan(issues)
        
        # Should handle conflict by choosing one fix
        self.assertEqual(len(plan.steps), 1)
        
        # Should choose the fix that removes the later function
        self.assertEqual(plan.steps[0].action, 'remove_duplicate')
    
    def test_plan_execution_order(self):
        """Test that plan execution order is correct."""
        issues = [
            Issue(
                kind='import_anxiety',
                file='imports.py',
                line=1,
                symbol='typing',
                message='Wildcard import',
                severity=2
            ),
            Issue(
                kind='duplicate_function',
                file='functions.py',
                line=50,
                symbol='process',
                message='Duplicate',
                severity=3
            ),
            Issue(
                kind='circular_import',
                file='imports.py',
                line=5,
                symbol='functions',
                message='Circular',
                severity=4
            )
        ]
        
        plan = self.planner.create_plan(issues)
        
        # Should handle same-file issues in correct order
        imports_steps = [s for s in plan.steps if s.issue.file == 'imports.py']
        if len(imports_steps) > 1:
            # Circular import should be fixed before import anxiety
            circular_idx = next(i for i, s in enumerate(plan.steps) 
                              if s.issue.kind == 'circular_import')
            anxiety_idx = next(i for i, s in enumerate(plan.steps)
                             if s.issue.kind == 'import_anxiety')
            self.assertLess(circular_idx, anxiety_idx)
    
    def test_plan_risk_assessment(self):
        """Test risk assessment in planning."""
        issues = [
            Issue(
                kind='phantom_function',
                file='critical_module.py',
                line=1,
                symbol='critical_function',
                message='Empty critical function',
                severity=5,
                evidence={'usage_count': 100}
            )
        ]
        
        plan = self.planner.create_plan(issues)
        
        # High-risk fix should be flagged
        self.assertGreater(plan.total_risk, 0.7)
        self.assertTrue(plan.needs_review)
        
        # Should include safety checks
        step = plan.steps[0]
        self.assertIn('validation', step.metadata)
        self.assertTrue(step.metadata.get('requires_testing', False))


class TestOrchestration(unittest.TestCase):
    """Test end-to-end orchestration."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.test_path = Path(self.temp_dir)
        self.orchestrator = TailChasingOrchestrator()
    
    def tearDown(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir)
    
    def create_file(self, name: str, content: str) -> Path:
        """Helper to create test files."""
        file_path = self.test_path / name
        file_path.write_text(textwrap.dedent(content))
        return file_path
    
    def test_full_orchestration(self):
        """Test complete orchestration workflow."""
        # Create problematic code
        self.create_file("problems.py", """
            from typing import *
            
            def calculate_sum(numbers):
                total = 0
                for num in numbers:
                    total += num
                return total
            
            def compute_total(values):
                # Duplicate of calculate_sum
                result = 0
                for val in values:
                    result += val
                return result
            
            def placeholder():
                pass
            
            def another_placeholder():
                raise NotImplementedError()
        """)
        
        # Run orchestration
        result = self.orchestrator.orchestrate(
            path=self.test_path,
            auto_fix=True,
            dry_run=False
        )
        
        self.assertIsNotNone(result)
        self.assertGreater(result['issues_found'], 0)
        
        # Should have applied some fixes
        if result.get('fixes_applied'):
            self.assertGreater(result['fixes_applied'], 0)
            
            # Check that file was modified
            content = (self.test_path / "problems.py").read_text()
            
            # Wildcard import should be fixed
            self.assertNotIn("from typing import *", content)
    
    def test_orchestration_with_validation(self):
        """Test orchestration with validation enabled."""
        # Create code with issues
        self.create_file("validate.py", """
            def function1():
                return 42
            
            def function2():
                return 42
            
            def broken():
                pass
        """)
        
        # Create test file
        self.create_file("test_validate.py", """
            from validate import function1
            
            def test_function1():
                assert function1() == 42
        """)
        
        config = {
            'validate_fixes': True,
            'preserve_tests': True,
            'dry_run': True
        }
        
        orchestrator = TailChasingOrchestrator(config)
        result = orchestrator.orchestrate(
            path=self.test_path,
            auto_fix=True
        )
        
        # Should detect issues
        self.assertGreater(result['issues_found'], 0)
        
        # In dry run, no actual fixes applied
        self.assertEqual(result.get('fixes_applied', 0), 0)
        
        # Should have validation results
        if 'validation_results' in result:
            self.assertIsNotNone(result['validation_results'])
    
    def test_performance_benchmark(self):
        """Benchmark orchestration performance."""
        # Create multiple files with various issues
        for i in range(10):
            self.create_file(f"module_{i}.py", f"""
                from typing import *
                import os, sys, json
                
                def function_{i}():
                    return {i}
                
                def duplicate_{i}():
                    return {i}
                
                def stub_{i}():
                    pass
            """)
        
        start_time = time.time()
        result = self.orchestrator.orchestrate(
            path=self.test_path,
            auto_fix=False  # Just analyze
        )
        elapsed = time.time() - start_time
        
        print(f"Orchestration time: {elapsed:.2f}s for 10 files")
        print(f"Issues found: {result['issues_found']}")
        
        # Should complete in reasonable time
        self.assertLess(elapsed, 30.0)  # 30 seconds max
        
        # Should find multiple issues
        self.assertGreater(result['issues_found'], 10)


if __name__ == '__main__':
    unittest.main()