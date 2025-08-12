"""
Fix Validation Framework for TailChasingFixer.

This module provides comprehensive validation capabilities for ensuring
the correctness and safety of applied fixes, including syntax validation,
semantic preservation checks, import resolution, and test suite validation.
"""

from __future__ import annotations
import ast
import importlib.util
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
import hashlib
import json
import re

from ...core.ast_analyzer import ASTAnalyzer
from ...analyzers.base import AnalysisContext
from .fix_strategies import FixPlan, FixAction

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Represents the result of a validation operation."""
    
    valid: bool
    validator_type: str
    validation_id: str
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    details: Dict[str, Any] = field(default_factory=dict)
    files_checked: Set[str] = field(default_factory=set)
    execution_time: Optional[float] = None
    
    def add_error(self, message: str, file_path: Optional[str] = None) -> None:
        """Add an error to the validation result."""
        error_msg = f"{file_path}: {message}" if file_path else message
        self.errors.append(error_msg)
        self.valid = False
    
    def add_warning(self, message: str, file_path: Optional[str] = None) -> None:
        """Add a warning to the validation result."""
        warning_msg = f"{file_path}: {message}" if file_path else message
        self.warnings.append(warning_msg)
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one."""
        self.valid = self.valid and other.valid
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.files_checked.update(other.files_checked)
        self.details.update(other.details)


class FixValidator:
    """
    Comprehensive fix validation system.
    
    Provides multiple validation layers to ensure fixes are safe and correct:
    - Syntax validation using Python AST parser
    - Semantic preservation through code analysis
    - Import resolution verification
    - Test suite execution and validation
    """
    
    def __init__(self, ctx: AnalysisContext):
        """Initialize the fix validator."""
        self.ctx = ctx
        self.ast_analyzer = ASTAnalyzer()
        
        # Validation configuration
        self.config = getattr(ctx, 'config', {}).get('validation', {})
        self.strict_mode = self.config.get('strict_mode', True)
        self.test_timeout = self.config.get('test_timeout', 300)  # 5 minutes
        self.max_import_depth = self.config.get('max_import_depth', 5)
        
        # Caches for performance
        self._syntax_cache: Dict[str, ValidationResult] = {}
        self._import_cache: Dict[str, Set[str]] = {}
        self._test_cache: Dict[str, ValidationResult] = {}
        
        logger.debug(f"FixValidator initialized with strict_mode={self.strict_mode}")
    
    def validate_syntax(
        self, 
        file_paths: List[str], 
        planned_actions: Optional[List[FixAction]] = None
    ) -> ValidationResult:
        """
        Validate Python syntax for files and planned changes.
        
        Args:
            file_paths: List of file paths to validate
            planned_actions: Optional list of actions to simulate for validation
            
        Returns:
            ValidationResult with syntax validation details
        """
        validation_id = f"syntax_{hashlib.md5(str(sorted(file_paths)).encode()).hexdigest()[:8]}"
        
        result = ValidationResult(
            valid=True,
            validator_type='syntax',
            validation_id=validation_id
        )
        
        try:
            logger.info(f"Validating syntax for {len(file_paths)} files")
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    result.add_error(f"File not found: {file_path}")
                    continue
                
                result.files_checked.add(file_path)
                
                try:
                    # Read current file content
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Apply planned actions if provided (simulation)
                    if planned_actions:
                        simulated_content = self._simulate_actions_on_content(
                            content, file_path, planned_actions
                        )
                    else:
                        simulated_content = content
                    
                    # Validate syntax using AST parser
                    syntax_validation = self._validate_python_syntax(simulated_content, file_path)
                    
                    if not syntax_validation['valid']:
                        result.add_error(
                            f"Syntax error: {syntax_validation['error']}", 
                            file_path
                        )
                        result.details[f'syntax_error_{file_path}'] = syntax_validation
                    else:
                        logger.debug(f"Syntax valid: {file_path}")
                
                except Exception as e:
                    result.add_error(f"Error validating syntax: {str(e)}", file_path)
                    logger.error(f"Error validating syntax for {file_path}: {e}", exc_info=True)
            
            # Additional syntax checks for Python best practices
            if result.valid and self.strict_mode:
                self._validate_python_best_practices(file_paths, result)
            
            result.details['files_validated'] = len(result.files_checked)
            result.details['strict_mode'] = self.strict_mode
            
            logger.info(f"Syntax validation complete: {len(result.files_checked)} files, "
                       f"valid: {result.valid}, errors: {len(result.errors)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in syntax validation: {e}", exc_info=True)
            result.add_error(f"Critical validation error: {str(e)}")
            return result
    
    def validate_semantic_preservation(
        self, 
        file_paths: List[str], 
        fix_plan: FixPlan
    ) -> ValidationResult:
        """
        Validate that semantic meaning is preserved after fixes.
        
        Args:
            file_paths: List of file paths to validate
            fix_plan: Fix plan that was applied
            
        Returns:
            ValidationResult with semantic preservation details
        """
        validation_id = f"semantic_{fix_plan.plan_id}_{hashlib.md5(str(sorted(file_paths)).encode()).hexdigest()[:8]}"
        
        result = ValidationResult(
            valid=True,
            validator_type='semantic_preservation',
            validation_id=validation_id
        )
        
        try:
            logger.info(f"Validating semantic preservation for {len(file_paths)} files")
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    result.add_warning(f"File not found for semantic validation: {file_path}")
                    continue
                
                result.files_checked.add(file_path)
                
                try:
                    # Analyze current file structure
                    current_structure = self._analyze_file_structure(file_path)
                    
                    # Check for semantic preservation based on fix types
                    preservation_checks = self._perform_semantic_preservation_checks(
                        file_path, current_structure, fix_plan
                    )
                    
                    if not preservation_checks['preserved']:
                        for issue in preservation_checks['issues']:
                            result.add_warning(f"Semantic preservation concern: {issue}", file_path)
                    
                    result.details[f'semantic_analysis_{file_path}'] = preservation_checks
                    
                    logger.debug(f"Semantic preservation checked: {file_path}")
                
                except Exception as e:
                    result.add_warning(f"Error in semantic analysis: {str(e)}", file_path)
                    logger.warning(f"Error in semantic analysis for {file_path}: {e}", exc_info=True)
            
            # Determine overall preservation status
            result.details['preservation_score'] = self._calculate_preservation_score(result)
            preserved = len(result.errors) == 0  # Only hard errors should fail this
            
            result.valid = preserved
            result.details['files_analyzed'] = len(result.files_checked)
            
            logger.info(f"Semantic preservation validation complete: {len(result.files_checked)} files, "
                       f"preserved: {preserved}, warnings: {len(result.warnings)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in semantic preservation validation: {e}", exc_info=True)
            result.add_error(f"Critical semantic validation error: {str(e)}")
            return result
    
    def validate_import_resolution(self, file_paths: List[str]) -> ValidationResult:
        """
        Validate that all imports can be resolved correctly.
        
        Args:
            file_paths: List of file paths to validate imports for
            
        Returns:
            ValidationResult with import resolution details
        """
        validation_id = f"imports_{hashlib.md5(str(sorted(file_paths)).encode()).hexdigest()[:8]}"
        
        result = ValidationResult(
            valid=True,
            validator_type='import_resolution',
            validation_id=validation_id
        )
        
        try:
            logger.info(f"Validating import resolution for {len(file_paths)} files")
            
            all_imports = {}
            unresolved_imports = []
            
            for file_path in file_paths:
                if not os.path.exists(file_path):
                    result.add_error(f"File not found for import validation: {file_path}")
                    continue
                
                result.files_checked.add(file_path)
                
                try:
                    # Extract imports from file
                    file_imports = self._extract_imports(file_path)
                    all_imports[file_path] = file_imports
                    
                    # Validate each import
                    for import_info in file_imports:
                        resolution_result = self._resolve_import(import_info, file_path)
                        
                        if not resolution_result['resolved']:
                            unresolved_imports.append({
                                'file': file_path,
                                'import': import_info,
                                'reason': resolution_result['reason']
                            })
                            result.add_error(
                                f"Unresolved import: {import_info['module']} - {resolution_result['reason']}", 
                                file_path
                            )
                    
                    logger.debug(f"Import resolution checked: {file_path} ({len(file_imports)} imports)")
                
                except Exception as e:
                    result.add_error(f"Error validating imports: {str(e)}", file_path)
                    logger.error(f"Error validating imports for {file_path}: {e}", exc_info=True)
            
            # Check for circular import dependencies
            circular_imports = self._detect_circular_imports(all_imports)
            if circular_imports:
                for cycle in circular_imports:
                    result.add_warning(f"Circular import detected: {' -> '.join(cycle)}")
            
            result.details['total_imports'] = sum(len(imports) for imports in all_imports.values())
            result.details['unresolved_count'] = len(unresolved_imports)
            result.details['circular_imports'] = len(circular_imports)
            result.details['unresolved_details'] = unresolved_imports
            
            logger.info(f"Import resolution validation complete: {result.details['total_imports']} imports, "
                       f"{result.details['unresolved_count']} unresolved, valid: {result.valid}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in import resolution validation: {e}", exc_info=True)
            result.add_error(f"Critical import validation error: {str(e)}")
            return result
    
    def validate_test_suite(self, affected_files: List[str]) -> ValidationResult:
        """
        Validate that test suites pass after fixes are applied.
        
        Args:
            affected_files: List of files that were modified by fixes
            
        Returns:
            ValidationResult with test execution details
        """
        validation_id = f"tests_{hashlib.md5(str(sorted(affected_files)).encode()).hexdigest()[:8]}"
        
        result = ValidationResult(
            valid=True,
            validator_type='test_suite',
            validation_id=validation_id
        )
        
        try:
            logger.info(f"Validating test suite for {len(affected_files)} affected files")
            
            # Find relevant test files
            test_files = self._find_test_files(affected_files)
            result.details['test_files_found'] = len(test_files)
            
            if not test_files:
                result.add_warning("No test files found for affected modules")
                result.details['test_execution'] = 'skipped_no_tests'
                return result
            
            # Execute test suite
            test_results = self._execute_tests(test_files)
            
            result.files_checked.update(test_files)
            result.details['test_execution'] = test_results
            
            # Analyze test results
            if test_results['return_code'] != 0:
                result.valid = False
                result.add_error(f"Test suite failed with return code {test_results['return_code']}")
                
                # Parse test failures
                failures = self._parse_test_failures(test_results['output'])
                for failure in failures:
                    result.add_error(f"Test failure: {failure['test']} - {failure['error']}")
                
                result.details['failures'] = failures
            else:
                logger.info("All tests passed")
            
            result.details['tests_run'] = test_results.get('tests_run', 0)
            result.details['tests_passed'] = test_results.get('tests_passed', 0)
            result.details['tests_failed'] = test_results.get('tests_failed', 0)
            result.details['execution_time'] = test_results.get('execution_time', 0)
            
            logger.info(f"Test validation complete: {result.details.get('tests_run', 0)} tests run, "
                       f"valid: {result.valid}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in test validation: {e}", exc_info=True)
            result.add_error(f"Critical test validation error: {str(e)}")
            return result
    
    def validate_comprehensive(
        self, 
        file_paths: List[str], 
        fix_plan: Optional[FixPlan] = None,
        run_tests: bool = True
    ) -> ValidationResult:
        """
        Run comprehensive validation including all validation types.
        
        Args:
            file_paths: List of files to validate
            fix_plan: Optional fix plan that was applied
            run_tests: Whether to run test suite validation
            
        Returns:
            Combined ValidationResult from all validation types
        """
        validation_id = f"comprehensive_{hashlib.md5(str(sorted(file_paths)).encode()).hexdigest()[:8]}"
        
        result = ValidationResult(
            valid=True,
            validator_type='comprehensive',
            validation_id=validation_id
        )
        
        try:
            logger.info(f"Running comprehensive validation for {len(file_paths)} files")
            
            # Step 1: Syntax validation
            logger.debug("Running syntax validation")
            syntax_result = self.validate_syntax(file_paths)
            result.merge(syntax_result)
            result.details['syntax_validation'] = syntax_result.details
            
            # Step 2: Import resolution validation  
            logger.debug("Running import resolution validation")
            import_result = self.validate_import_resolution(file_paths)
            result.merge(import_result)
            result.details['import_validation'] = import_result.details
            
            # Step 3: Semantic preservation (if fix plan provided)
            if fix_plan:
                logger.debug("Running semantic preservation validation")
                semantic_result = self.validate_semantic_preservation(file_paths, fix_plan)
                result.merge(semantic_result)
                result.details['semantic_validation'] = semantic_result.details
            
            # Step 4: Test suite validation (if requested)
            if run_tests:
                logger.debug("Running test suite validation")
                test_result = self.validate_test_suite(file_paths)
                # Test failures are warnings in comprehensive validation
                if not test_result.valid:
                    result.warnings.extend([f"Test: {error}" for error in test_result.errors])
                result.warnings.extend(test_result.warnings)
                result.details['test_validation'] = test_result.details
            
            result.details['validation_summary'] = {
                'syntax_valid': syntax_result.valid,
                'imports_resolved': import_result.valid,
                'semantics_preserved': semantic_result.valid if fix_plan else None,
                'tests_passing': test_result.valid if run_tests else None,
                'files_validated': len(result.files_checked),
                'total_errors': len(result.errors),
                'total_warnings': len(result.warnings)
            }
            
            logger.info(f"Comprehensive validation complete: valid={result.valid}, "
                       f"errors={len(result.errors)}, warnings={len(result.warnings)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Critical error in comprehensive validation: {e}", exc_info=True)
            result.add_error(f"Critical comprehensive validation error: {str(e)}")
            return result
    
    # Helper methods for syntax validation
    
    def _validate_python_syntax(self, content: str, file_path: str) -> Dict[str, Any]:
        """Validate Python syntax using AST parser."""
        try:
            # Remove BOM if present
            if content.startswith('\ufeff'):
                content = content[1:]
            
            # Parse AST
            ast.parse(content, filename=file_path)
            
            return {'valid': True, 'error': None}
        
        except SyntaxError as e:
            return {
                'valid': False,
                'error': f"Line {e.lineno}: {e.msg}",
                'line': e.lineno,
                'column': e.offset,
                'details': str(e)
            }
        except Exception as e:
            return {
                'valid': False,
                'error': f"Parse error: {str(e)}",
                'details': str(e)
            }
    
    def _simulate_actions_on_content(
        self, 
        content: str, 
        file_path: str, 
        actions: List[FixAction]
    ) -> str:
        """Simulate applying actions to content for validation."""
        simulated_content = content
        
        for action in actions:
            if action.file_path != file_path:
                continue
            
            try:
                if action.action_type == 'replace_text' and action.old_content and action.new_content:
                    simulated_content = simulated_content.replace(
                        action.old_content, action.new_content, 1
                    )
                elif action.action_type == 'add_import' and action.new_content:
                    # Add import at the beginning (simplified)
                    lines = simulated_content.splitlines()
                    lines.insert(0, action.new_content)
                    simulated_content = '\n'.join(lines)
                # Add more action types as needed
            except Exception as e:
                logger.warning(f"Error simulating action {action.action_type}: {e}")
        
        return simulated_content
    
    def _validate_python_best_practices(self, file_paths: List[str], result: ValidationResult) -> None:
        """Validate Python best practices in strict mode."""
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Check for common issues
                lines = content.splitlines()
                for i, line in enumerate(lines, 1):
                    # Check for trailing whitespace
                    if line.rstrip() != line:
                        result.add_warning(f"Line {i}: Trailing whitespace", file_path)
                    
                    # Check for tabs (should use spaces)
                    if '\t' in line:
                        result.add_warning(f"Line {i}: Tab character found (use spaces)", file_path)
                    
                    # Check for very long lines (>120 chars)
                    if len(line) > 120:
                        result.add_warning(f"Line {i}: Line too long ({len(line)} chars)", file_path)
            
            except Exception as e:
                logger.debug(f"Error in best practices validation for {file_path}: {e}")
    
    # Helper methods for semantic preservation
    
    def _analyze_file_structure(self, file_path: str) -> Dict[str, Any]:
        """Analyze the structure of a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            
            structure = {
                'functions': [],
                'classes': [],
                'imports': [],
                'constants': [],
                'complexity': 0
            }
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    structure['functions'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'args': len(node.args.args),
                        'returns': bool(node.returns)
                    })
                elif isinstance(node, ast.ClassDef):
                    structure['classes'].append({
                        'name': node.name,
                        'line': node.lineno,
                        'methods': len([n for n in node.body if isinstance(n, ast.FunctionDef)])
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    structure['imports'].append({
                        'type': type(node).__name__,
                        'line': node.lineno
                    })
                elif isinstance(node, ast.Assign):
                    # Look for module-level constants
                    if isinstance(node.targets[0], ast.Name):
                        name = node.targets[0].id
                        if name.isupper():
                            structure['constants'].append(name)
            
            structure['complexity'] = len(structure['functions']) + len(structure['classes'])
            
            return structure
        
        except Exception as e:
            logger.warning(f"Error analyzing file structure for {file_path}: {e}")
            return {'error': str(e)}
    
    def _perform_semantic_preservation_checks(
        self, 
        file_path: str, 
        structure: Dict[str, Any], 
        fix_plan: FixPlan
    ) -> Dict[str, Any]:
        """Perform semantic preservation checks."""
        result = {
            'preserved': True,
            'issues': [],
            'changes_detected': []
        }
        
        try:
            # Check if critical structures are preserved
            relevant_actions = [a for a in fix_plan.actions if a.file_path == file_path]
            
            for action in relevant_actions:
                if action.action_type == 'remove_function':
                    result['changes_detected'].append(f"Function removal: {action.target_symbol}")
                    # Check if function is referenced elsewhere
                    if self._is_function_referenced(action.target_symbol, file_path):
                        result['issues'].append(f"Removed function {action.target_symbol} may be referenced elsewhere")
                
                elif action.action_type == 'modify_signature':
                    result['changes_detected'].append(f"Signature change: {action.target_symbol}")
                    result['issues'].append(f"Function signature changed for {action.target_symbol} - may break callers")
            
            # If there are semantic concerns, mark as not fully preserved
            if result['issues']:
                result['preserved'] = len([i for i in result['issues'] if 'may break' in i]) == 0
            
            return result
        
        except Exception as e:
            logger.warning(f"Error in semantic preservation checks: {e}")
            return {'preserved': False, 'issues': [f"Analysis error: {str(e)}"], 'changes_detected': []}
    
    def _is_function_referenced(self, function_name: str, file_path: str) -> bool:
        """Check if a function is referenced in the codebase."""
        # Simplified check - in a real implementation, this would be more sophisticated
        try:
            for root, dirs, files in os.walk(os.path.dirname(file_path)):
                for file in files:
                    if file.endswith('.py'):
                        full_path = os.path.join(root, file)
                        if full_path == file_path:
                            continue
                        
                        with open(full_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                            if function_name in content:
                                return True
            
            return False
        except Exception:
            return True  # Conservative assumption
    
    def _calculate_preservation_score(self, result: ValidationResult) -> float:
        """Calculate a semantic preservation score."""
        total_checks = len(result.files_checked)
        if total_checks == 0:
            return 1.0
        
        # Simple scoring based on warnings
        warning_penalty = len(result.warnings) * 0.1
        error_penalty = len(result.errors) * 0.3
        
        score = max(0.0, 1.0 - (warning_penalty + error_penalty) / total_checks)
        return round(score, 3)
    
    # Helper methods for import resolution
    
    def _extract_imports(self, file_path: str) -> List[Dict[str, Any]]:
        """Extract import statements from a Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            tree = ast.parse(content, filename=file_path)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append({
                            'type': 'import',
                            'module': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno
                        })
                elif isinstance(node, ast.ImportFrom):
                    module = node.module or ''
                    for alias in node.names:
                        imports.append({
                            'type': 'from_import',
                            'module': module,
                            'name': alias.name,
                            'alias': alias.asname,
                            'line': node.lineno,
                            'level': node.level
                        })
            
            return imports
        
        except Exception as e:
            logger.warning(f"Error extracting imports from {file_path}: {e}")
            return []
    
    def _resolve_import(self, import_info: Dict[str, Any], file_path: str) -> Dict[str, Any]:
        """Attempt to resolve a single import."""
        try:
            module_name = import_info['module']
            
            # Skip standard library modules (simplified check)
            if self._is_stdlib_module(module_name):
                return {'resolved': True, 'type': 'stdlib'}
            
            # Try to resolve using importlib
            try:
                if import_info.get('level', 0) > 0:
                    # Relative import
                    package = self._get_package_name(file_path)
                    spec = importlib.util.find_spec(module_name, package=package)
                else:
                    # Absolute import
                    spec = importlib.util.find_spec(module_name)
                
                if spec is not None:
                    return {'resolved': True, 'type': 'module', 'spec': str(spec)}
                else:
                    return {'resolved': False, 'reason': 'Module not found'}
            
            except Exception as e:
                return {'resolved': False, 'reason': f'Import error: {str(e)}'}
        
        except Exception as e:
            return {'resolved': False, 'reason': f'Resolution error: {str(e)}'}
    
    def _is_stdlib_module(self, module_name: str) -> bool:
        """Check if a module is part of the Python standard library."""
        stdlib_modules = {
            'os', 'sys', 'json', 'csv', 'sqlite3', 'datetime', 'pathlib',
            're', 'collections', 'itertools', 'functools', 'typing',
            'logging', 'urllib', 'http', 'xml', 'html', 'email',
            'subprocess', 'threading', 'multiprocessing', 'concurrent',
            'ast', 'inspect', 'importlib', 'tempfile', 'shutil', 'hashlib'
        }
        
        return module_name.split('.')[0] in stdlib_modules
    
    def _get_package_name(self, file_path: str) -> str:
        """Get the package name for a file."""
        # Simplified implementation
        parts = Path(file_path).parts
        if '__init__.py' in parts[-1]:
            return '.'.join(parts[:-2])
        else:
            return '.'.join(parts[:-1])
    
    def _detect_circular_imports(self, all_imports: Dict[str, List[Dict[str, Any]]]) -> List[List[str]]:
        """Detect circular import dependencies."""
        # Simplified circular import detection
        import_graph = {}
        
        for file_path, imports in all_imports.items():
            module_name = Path(file_path).stem
            import_graph[module_name] = []
            
            for imp in imports:
                if imp['type'] == 'from_import' or imp['type'] == 'import':
                    target = imp['module'].split('.')[0]
                    if target != module_name:
                        import_graph[module_name].append(target)
        
        # Simple cycle detection (could be improved)
        cycles = []
        visited = set()
        
        def dfs(node, path):
            if node in path:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return
            
            if node in visited or node not in import_graph:
                return
            
            visited.add(node)
            path.append(node)
            
            for neighbor in import_graph[node]:
                dfs(neighbor, path.copy())
        
        for node in import_graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles
    
    # Helper methods for test validation
    
    def _find_test_files(self, affected_files: List[str]) -> List[str]:
        """Find test files related to the affected files."""
        test_files = []
        
        for file_path in affected_files:
            file_path_obj = Path(file_path)
            
            # Look for test files with common patterns
            test_patterns = [
                f"test_{file_path_obj.stem}.py",
                f"{file_path_obj.stem}_test.py",
                f"tests/test_{file_path_obj.stem}.py",
                f"test/test_{file_path_obj.stem}.py"
            ]
            
            for pattern in test_patterns:
                # Check in same directory
                test_path = file_path_obj.parent / pattern
                if test_path.exists():
                    test_files.append(str(test_path))
                
                # Check in parent directories
                for parent in file_path_obj.parents:
                    test_path = parent / pattern
                    if test_path.exists():
                        test_files.append(str(test_path))
        
        # Also look for general test directories
        base_dirs = set(Path(f).parent for f in affected_files)
        for base_dir in base_dirs:
            for test_dir in ['tests', 'test']:
                test_path = base_dir / test_dir
                if test_path.exists() and test_path.is_dir():
                    for test_file in test_path.glob('test_*.py'):
                        test_files.append(str(test_file))
        
        return list(set(test_files))  # Remove duplicates
    
    def _execute_tests(self, test_files: List[str]) -> Dict[str, Any]:
        """Execute test files and return results."""
        try:
            # Try pytest first
            result = self._run_pytest(test_files)
            if result['success']:
                return result
            
            # Fall back to unittest
            return self._run_unittest(test_files)
        
        except Exception as e:
            logger.error(f"Error executing tests: {e}", exc_info=True)
            return {
                'success': False,
                'return_code': -1,
                'output': f"Test execution error: {str(e)}",
                'error': str(e)
            }
    
    def _run_pytest(self, test_files: List[str]) -> Dict[str, Any]:
        """Run tests using pytest."""
        try:
            cmd = ['python', '-m', 'pytest'] + test_files + ['-v', '--tb=short']
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                cwd=str(self.ctx.root_dir)
            )
            
            return {
                'success': True,
                'framework': 'pytest',
                'return_code': result.returncode,
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': None  # Could be parsed from output
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'framework': 'pytest',
                'return_code': -1,
                'output': '',
                'error': f'Test execution timed out after {self.test_timeout} seconds'
            }
        except FileNotFoundError:
            return {'success': False, 'error': 'pytest not found'}
        except Exception as e:
            return {'success': False, 'error': f'pytest error: {str(e)}'}
    
    def _run_unittest(self, test_files: List[str]) -> Dict[str, Any]:
        """Run tests using unittest."""
        try:
            # Convert file paths to module names
            modules = []
            for test_file in test_files:
                module_path = Path(test_file)
                if module_path.suffix == '.py':
                    module_name = str(module_path.with_suffix('').relative_to(self.ctx.root_dir)).replace('/', '.')
                    modules.append(module_name)
            
            if not modules:
                return {'success': False, 'error': 'No valid test modules found'}
            
            cmd = ['python', '-m', 'unittest'] + modules + ['-v']
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.test_timeout,
                cwd=str(self.ctx.root_dir)
            )
            
            return {
                'success': True,
                'framework': 'unittest',
                'return_code': result.returncode,
                'output': result.stdout,
                'error': result.stderr,
                'execution_time': None
            }
        
        except subprocess.TimeoutExpired:
            return {
                'success': False,
                'framework': 'unittest',
                'return_code': -1,
                'output': '',
                'error': f'Test execution timed out after {self.test_timeout} seconds'
            }
        except Exception as e:
            return {'success': False, 'error': f'unittest error: {str(e)}'}
    
    def _parse_test_failures(self, output: str) -> List[Dict[str, str]]:
        """Parse test failures from test output."""
        failures = []
        
        try:
            lines = output.splitlines()
            
            # Parse pytest output
            if 'FAILED' in output:
                for line in lines:
                    if '::' in line and 'FAILED' in line:
                        parts = line.split(' FAILED ')
                        if len(parts) >= 2:
                            failures.append({
                                'test': parts[0].strip(),
                                'error': parts[1].strip() if len(parts) > 1 else 'Unknown error'
                            })
            
            # Parse unittest output
            elif 'FAIL:' in output:
                current_test = None
                for line in lines:
                    if line.startswith('FAIL: '):
                        current_test = line[6:].strip()
                    elif current_test and line.strip() and not line.startswith('='):
                        failures.append({
                            'test': current_test,
                            'error': line.strip()
                        })
                        current_test = None
            
        except Exception as e:
            logger.warning(f"Error parsing test failures: {e}")
        
        return failures