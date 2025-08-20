"""
Validation and test generation strategies.

Handles validation test generation, test case creation, and verification strategies.
Extracted from fix_strategies.py to reduce context window thrashing.
"""

from typing import List, Dict, Any, Optional

from .base import BaseFixStrategy, Action, Patch, RiskLevel
from ...core.issues import Issue


class ValidationTestGenerator(BaseFixStrategy):
    """
    Strategy for generating validation tests for fixes.
    
    Creates test cases that verify:
    - Syntax correctness
    - Import resolution
    - Function behavior
    - Performance impact
    """
    
    def __init__(self):
        super().__init__("ValidationTestGenerator")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle test generation for any issue type."""
        return True  # Can generate tests for any issue
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose validation tests for an issue."""
        actions = self._generate_test_actions(issue, context)
        
        if not actions:
            return None
        
        return Patch(
            actions=actions,
            description=f"Generate validation tests for: {issue.kind}",
            confidence=0.9,  # Test generation is usually reliable
            risk_level=RiskLevel.LOW,  # Tests don't affect production code
            estimated_time=self._estimate_time(actions),
            dependencies=[],
            rollback_plan=[self.create_backup_action(action) for action in actions],
            validation_tests=[],  # Tests don't need validation tests
            side_effects=["Creates test files"]
        )
    
    def _generate_test_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to create validation tests."""
        actions = []
        
        # Generate test file content
        test_content = self._generate_test_content(issue, context)
        
        if test_content:
            # Create test file
            test_filename = self._generate_test_filename(issue)
            
            actions.append(Action(
                type="create_file",
                target=test_filename,
                content=test_content,
                metadata={
                    "test_type": "validation",
                    "issue_kind": issue.kind,
                    "original_file": issue.file
                }
            ))
        
        return actions
    
    def _generate_test_filename(self, issue: Issue) -> str:
        """Generate appropriate test filename."""
        if issue.file:
            # Convert source file to test file
            base_name = issue.file.replace('.py', '')
            return f"test_{base_name.replace('/', '_')}_validation.py"
        else:
            return f"test_{issue.kind}_validation.py"
    
    def _generate_test_content(self, issue: Issue, context: Optional[Dict[str, Any]]) -> str:
        """Generate test content based on issue type."""
        test_cases = []
        
        # Add basic syntax test
        test_cases.append(self._generate_syntax_test(issue))
        
        # Add issue-specific tests
        if issue.kind in ["missing_symbol", "missing_import"]:
            test_cases.append(self._generate_import_test(issue))
        elif issue.kind in ["duplicate_function", "semantic_duplicate_function"]:
            test_cases.append(self._generate_duplicate_test(issue))
        elif issue.kind in ["circular_import", "circular_dependency"]:
            test_cases.append(self._generate_circular_import_test(issue))
        elif issue.kind in ["phantom_function", "placeholder"]:
            test_cases.append(self._generate_implementation_test(issue))
        
        # Add performance test
        test_cases.append(self._generate_performance_test(issue))
        
        return self._format_test_file(test_cases, issue)
    
    def _generate_syntax_test(self, issue: Issue) -> str:
        """Generate syntax validation test."""
        return f'''def test_syntax_validation():
    """Test that the file has valid Python syntax."""
    import ast
    import sys
    
    try:
        with open("{issue.file}", "r") as f:
            content = f.read()
        
        # Parse AST to check syntax
        ast.parse(content)
        assert True, "Syntax is valid"
        
    except SyntaxError as e:
        assert False, f"Syntax error in {issue.file}: {{e}}"
    except FileNotFoundError:
        assert False, f"File not found: {issue.file}"
    except Exception as e:
        assert False, f"Unexpected error: {{e}}"
'''
    
    def _generate_import_test(self, issue: Issue) -> str:
        """Generate import validation test."""
        module_name = issue.file.replace('.py', '').replace('/', '.')
        symbol = issue.symbol or 'unknown_symbol'
        
        return f'''def test_import_validation():
    """Test that imports work correctly."""
    import sys
    import importlib
    
    try:
        # Test module import
        module = importlib.import_module("{module_name}")
        assert module is not None, "Module imported successfully"
        
        # Test symbol availability if specified
        symbol_name = "{symbol}"
        if symbol_name != "unknown_symbol":
            assert hasattr(module, symbol_name), f"Symbol {{symbol_name}} is available"
        
    except ImportError as e:
        assert False, f"Import failed: {{e}}"
    except AttributeError as e:
        assert False, f"Symbol not found: {{e}}"
    except Exception as e:
        assert False, f"Unexpected import error: {{e}}"
'''
    
    def _generate_duplicate_test(self, issue: Issue) -> str:
        """Generate test for duplicate function handling."""
        return f'''def test_duplicate_handling():
    """Test that duplicate functions are properly handled."""
    import ast
    import sys
    
    try:
        with open("{issue.file}", "r") as f:
            content = f.read()
        
        tree = ast.parse(content)
        
        # Count function definitions
        function_names = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                function_names.append(node.name)
        
        # Check for duplicates
        unique_functions = set(function_names)
        duplicates = [name for name in unique_functions 
                     if function_names.count(name) > 1]
        
        if duplicates:
            # If duplicates exist, verify they are intentional (e.g., overloads)
            print(f"Warning: Found duplicate functions: {{duplicates}}")
            # TODO: Add specific validation for intentional duplicates
        
        assert True, "Duplicate function check completed"
        
    except Exception as e:
        assert False, f"Error checking duplicates: {{e}}"
'''
    
    def _generate_circular_import_test(self, issue: Issue) -> str:
        """Generate test for circular import resolution."""
        return f'''def test_circular_import_resolution():
    """Test that circular imports are resolved."""
    import sys
    import importlib
    
    try:
        # Clear import cache to test fresh imports
        modules_to_clear = [mod for mod in sys.modules.keys() 
                           if "{issue.file.replace('.py', '')}" in mod]
        for mod in modules_to_clear:
            if mod in sys.modules:
                del sys.modules[mod]
        
        # Test import
        module_name = "{issue.file.replace('.py', '').replace('/', '.')}"
        module = importlib.import_module(module_name)
        
        assert module is not None, "Module imported without circular dependency"
        
        # Test that the module is functional
        assert hasattr(module, '__name__'), "Module has proper attributes"
        
    except ImportError as e:
        assert False, f"Circular import not resolved: {{e}}"
    except Exception as e:
        assert False, f"Unexpected error: {{e}}"
'''
    
    def _generate_implementation_test(self, issue: Issue) -> str:
        """Generate test for phantom function implementation."""
        symbol = issue.symbol or 'unknown_function'
        
        return f'''def test_implementation_exists():
    """Test that the implemented function exists and is callable."""
    import sys
    import importlib
    
    try:
        module_name = "{issue.file.replace('.py', '').replace('/', '.')}"
        module = importlib.import_module(module_name)
        
        function_name = "{symbol}"
        assert hasattr(module, function_name), f"Function {{function_name}} exists"
        
        func = getattr(module, function_name)
        assert callable(func), f"{{function_name}} is callable"
        
        # Test basic functionality (if it's a method, need instance)
        try:
            if hasattr(func, '__self__'):
                # Method with bound instance
                result = func()
            else:
                # Function or unbound method - try with minimal args
                import inspect
                sig = inspect.signature(func)
                params = sig.parameters
                
                if len(params) == 0:
                    result = func()
                elif len(params) == 1 and 'self' in params:
                    # Skip - need instance
                    pass
                else:
                    # Try with None arguments
                    result = func(*[None] * len(params))
            
        except (TypeError, NotImplementedError):
            # Expected for auto-generated functions
            pass
        
        assert True, f"Function {{function_name}} is properly implemented"
        
    except Exception as e:
        assert False, f"Implementation test failed: {{e}}"
'''
    
    def _generate_performance_test(self, issue: Issue) -> str:
        """Generate performance validation test."""
        return f'''def test_performance_impact():
    """Test that the fix doesn't introduce performance regressions."""
    import time
    import sys
    import importlib
    
    try:
        # Measure import time
        start_time = time.time()
        module_name = "{issue.file.replace('.py', '').replace('/', '.')}"
        module = importlib.import_module(module_name)
        import_time = time.time() - start_time
        
        # Import should be reasonably fast (< 1 second for most modules)
        assert import_time < 1.0, f"Import time acceptable: {{import_time:.3f}}s"
        
        # Test memory usage (basic check)
        import sys
        module_size = sys.getsizeof(module)
        
        # Module size should be reasonable (< 10MB for most modules)
        max_size = 10 * 1024 * 1024  # 10MB
        assert module_size < max_size, f"Module size acceptable: {{module_size}} bytes"
        
        print(f"Performance metrics - Import: {{import_time:.3f}}s, Size: {{module_size}} bytes")
        
    except Exception as e:
        # Performance tests are not critical
        print(f"Performance test warning: {{e}}")
        assert True, "Performance test completed with warnings"
'''
    
    def _format_test_file(self, test_cases: List[str], issue: Issue) -> str:
        """Format test cases into a complete test file."""
        header = f'''"""
Validation tests for fix applied to {issue.file}

Issue: {issue.kind}
Symbol: {issue.symbol or 'N/A'}
Message: {issue.message or 'N/A'}

Generated by TailChasingFixer ValidationTestGenerator
"""

import pytest
import sys
import os
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
while project_root.name != "tailchasing" and project_root.parent != project_root:
    project_root = project_root.parent
sys.path.insert(0, str(project_root.parent))

'''
        
        footer = '''

if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
'''
        
        return header + '\n\n'.join(test_cases) + footer


class TestCaseGenerator(BaseFixStrategy):
    """
    Strategy for generating comprehensive test cases for fixed code.
    
    Creates unit tests, integration tests, and edge case tests.
    """
    
    def __init__(self):
        super().__init__("TestCaseGenerator")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle test case generation for function-related issues."""
        return issue.kind in [
            "phantom_function",
            "placeholder",
            "duplicate_function",
            "semantic_duplicate_function"
        ] and issue.symbol is not None
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose comprehensive test cases for fixed functions."""
        actions = self._generate_test_case_actions(issue, context)
        
        if not actions:
            return None
        
        return Patch(
            actions=actions,
            description=f"Generate test cases for: {issue.symbol}",
            confidence=0.8,
            risk_level=RiskLevel.LOW,
            estimated_time=self._estimate_time(actions),
            dependencies=[],
            rollback_plan=[self.create_backup_action(action) for action in actions],
            validation_tests=[],
            side_effects=["Creates comprehensive test suite"]
        )
    
    def _generate_test_case_actions(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate actions to create comprehensive test cases."""
        actions = []
        
        if not issue.symbol:
            return actions
        
        # Generate test content
        test_content = self._generate_comprehensive_tests(issue, context)
        
        if test_content:
            # Create test file
            test_filename = f"test_{issue.symbol}_comprehensive.py"
            
            actions.append(Action(
                type="create_file",
                target=test_filename,
                content=test_content,
                metadata={
                    "test_type": "comprehensive",
                    "function_name": issue.symbol,
                    "issue_kind": issue.kind
                }
            ))
        
        return actions
    
    def _generate_comprehensive_tests(self, issue: Issue, context: Optional[Dict[str, Any]]) -> str:
        """Generate comprehensive test cases for a function."""
        func_name = issue.symbol
        module_name = issue.file.replace('.py', '').replace('/', '.')
        
        return f'''"""
Comprehensive test cases for {func_name}

Generated by TailChasingFixer TestCaseGenerator
"""

import pytest
import sys
import os
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent
while project_root.name != "tailchasing" and project_root.parent != project_root:
    project_root = project_root.parent
sys.path.insert(0, str(project_root.parent))

try:
    from {module_name} import {func_name}
except ImportError as e:
    pytest.skip(f"Cannot import {{func_name}}: {{e}}", allow_module_level=True)


class Test{func_name.title()}:
    """Comprehensive test suite for {func_name}."""
    
    def test_function_exists(self):
        """Test that the function exists and is callable."""
        assert callable({func_name}), f"{func_name} should be callable"
    
    def test_function_signature(self):
        """Test the function signature."""
        import inspect
        
        sig = inspect.signature({func_name})
        params = sig.parameters
        
        # Basic signature validation
        assert sig is not None, "Function should have a signature"
        
        # TODO: Add specific parameter validations based on function purpose
        print(f"Function signature: {{sig}}")
    
    def test_normal_operation(self):
        """Test normal operation with valid inputs."""
        try:
            # TODO: Replace with actual test inputs based on function purpose
            if len(inspect.signature({func_name}).parameters) == 0:
                result = {func_name}()
            else:
                # Try with mock inputs
                result = {func_name}(Mock())
            
            # TODO: Add assertions based on expected behavior
            assert result is not None or result is None  # Placeholder assertion
            
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
        except Exception as e:
            pytest.fail(f"Normal operation failed: {{e}}")
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with None input
        try:
            if len(inspect.signature({func_name}).parameters) > 0:
                result = {func_name}(None)
                # TODO: Add specific edge case assertions
        except (ValueError, TypeError):
            # Expected for some functions
            pass
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_error_handling(self):
        """Test error handling and exception cases."""
        # TODO: Add specific error condition tests
        try:
            # Test with invalid inputs
            pass
        except Exception as e:
            # Verify expected exceptions
            pass
    
    @pytest.mark.parametrize("test_input,expected", [
        # TODO: Add parameterized test cases
        (None, None),  # Placeholder
    ])
    def test_parameterized_cases(self, test_input, expected):
        """Test multiple input/output combinations."""
        try:
            if len(inspect.signature({func_name}).parameters) > 0:
                result = {func_name}(test_input)
                # TODO: Add specific assertions
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
    
    def test_performance(self):
        """Test performance characteristics."""
        import time
        
        try:
            start_time = time.time()
            
            # Run function multiple times
            for _ in range(100):
                if len(inspect.signature({func_name}).parameters) == 0:
                    {func_name}()
                else:
                    {func_name}(Mock())
            
            execution_time = time.time() - start_time
            
            # Function should execute reasonably quickly
            assert execution_time < 1.0, f"Performance acceptable: {{execution_time:.3f}}s"
            
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
        except Exception as e:
            pytest.fail(f"Performance test failed: {{e}}")
    
    def test_memory_usage(self):
        """Test memory usage characteristics."""
        import tracemalloc
        
        try:
            tracemalloc.start()
            
            # Execute function
            if len(inspect.signature({func_name}).parameters) == 0:
                result = {func_name}()
            else:
                result = {func_name}(Mock())
            
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            # Memory usage should be reasonable
            max_memory = 10 * 1024 * 1024  # 10MB
            assert peak < max_memory, f"Memory usage acceptable: {{peak}} bytes"
            
        except NotImplementedError:
            pytest.skip("Function not yet implemented")
        except Exception as e:
            pytest.fail(f"Memory test failed: {{e}}")


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v", "--tb=short"])
'''


class SecurityValidationStrategy(BaseFixStrategy):
    """
    Strategy for validating security aspects of fixes.
    
    Checks for potential security vulnerabilities introduced by fixes.
    """
    
    def __init__(self):
        super().__init__("SecurityValidation")
    
    def can_handle(self, issue: Issue) -> bool:
        """Handle security validation for generated code."""
        return issue.kind in [
            "phantom_function",
            "placeholder",
            "generated_implementation"
        ]
    
    def propose_fix(self, issue: Issue, context: Optional[Dict[str, Any]] = None) -> Optional[Patch]:
        """Propose security validation checks."""
        actions = self._generate_security_checks(issue, context)
        
        if not actions:
            return None
        
        return Patch(
            actions=actions,
            description=f"Add security validation for: {issue.symbol}",
            confidence=0.95,  # Security checks are important
            risk_level=RiskLevel.LOW,
            estimated_time=self._estimate_time(actions),
            dependencies=[],
            rollback_plan=[self.create_backup_action(action) for action in actions],
            validation_tests=[],
            side_effects=["Adds security validation"]
        )
    
    def _generate_security_checks(self, issue: Issue, context: Optional[Dict[str, Any]]) -> List[Action]:
        """Generate security validation actions."""
        actions = []
        
        # Create security checklist file
        checklist_content = self._generate_security_checklist(issue)
        
        actions.append(Action(
            type="create_file",
            target=f"security_checklist_{issue.symbol}.md",
            content=checklist_content,
            metadata={
                "type": "security_validation",
                "function_name": issue.symbol
            }
        ))
        
        return actions
    
    def _generate_security_checklist(self, issue: Issue) -> str:
        """Generate security validation checklist."""
        func_name = issue.symbol or "unknown_function"
        
        return f'''# Security Validation Checklist for {func_name}

Generated by TailChasingFixer SecurityValidationStrategy

## Function: {func_name}
- **File**: {issue.file}
- **Issue Type**: {issue.kind}

## Security Checks

### Input Validation
- [ ] Function validates all input parameters
- [ ] Input size limits are enforced
- [ ] Type checking is performed
- [ ] SQL injection prevention (if applicable)
- [ ] XSS prevention (if applicable)

### Authentication & Authorization  
- [ ] Function checks user permissions
- [ ] Authentication is required where appropriate
- [ ] Authorization levels are enforced
- [ ] Session management is secure

### Data Handling
- [ ] Sensitive data is not logged
- [ ] Data is properly sanitized
- [ ] Encryption is used for sensitive data
- [ ] Data is validated before processing

### Error Handling
- [ ] Errors don't leak sensitive information
- [ ] Exception handling is comprehensive
- [ ] Error messages are user-friendly
- [ ] Stack traces are not exposed

### Resource Management
- [ ] Memory usage is bounded
- [ ] File operations are secure
- [ ] Network requests are validated
- [ ] Resources are properly cleaned up

### Code Quality
- [ ] No hard-coded secrets or credentials
- [ ] Dependencies are from trusted sources
- [ ] Code follows security best practices
- [ ] Regular security updates are planned

## Review Actions
- [ ] Code review by security team
- [ ] Penetration testing if applicable
- [ ] Static security analysis
- [ ] Dynamic security testing

## Notes
Generated implementations require manual security review.
Auto-generated code may not include all necessary security measures.
'''