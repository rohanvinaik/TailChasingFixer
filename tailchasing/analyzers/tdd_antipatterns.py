"""
Detect LLM-generated test anti-patterns that indicate superficial understanding.
"""

import ast
from typing import List, Dict, Set, Optional
from dataclasses import dataclass
from ..core.issues import Issue
from ..core.utils import safe_get_lineno
from .base import AnalysisContext


@dataclass
class TestPattern:
    """Represents a detected test anti-pattern."""
    pattern_type: str
    test_name: str
    function_name: Optional[str]
    confidence: float
    explanation: str


class TDDAntipatternAnalyzer:
    """Detect test-driven development anti-patterns in LLM-generated tests."""
    
    name = "tdd_antipatterns"
    
    def __init__(self):
        self.test_patterns = []
        self.implementation_map = {}
        
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Analyze test files for LLM-induced anti-patterns."""
        issues = []
        
        # First, build a map of implementations
        self._build_implementation_map(ctx)
        
        # Then analyze test files
        for file, tree in ctx.ast_index.items():
            if self._is_test_file(file):
                test_issues = self._analyze_test_file(file, tree, ctx)
                issues.extend(test_issues)
        
        return issues
    
    def _is_test_file(self, filepath: str) -> bool:
        """Check if file is a test file."""
        return any(pattern in filepath for pattern in ['test_', '_test.py', '/tests/', '/test/'])
    
    def _build_implementation_map(self, ctx: AnalysisContext):
        """Build map of function implementations for comparison."""
        for file, tree in ctx.ast_index.items():
            if not self._is_test_file(file):
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        self.implementation_map[node.name] = {
                            'file': file,
                            'node': node,
                            'body_dump': ast.dump(node.body)
                        }
    
    def _analyze_test_file(self, file: str, tree: ast.AST, ctx: AnalysisContext) -> List[Issue]:
        """Analyze a test file for anti-patterns."""
        issues = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith('test_'):
                # Check for mirror tests
                mirror_issue = self._check_mirror_test(node, file)
                if mirror_issue:
                    issues.append(mirror_issue)
                
                # Check for overly brittle assertions
                brittle_issue = self._check_brittle_assertions(node, file)
                if brittle_issue:
                    issues.append(brittle_issue)
                
                # Check for redundant tests
                redundant_issue = self._check_redundant_test(node, file, tree)
                if redundant_issue:
                    issues.append(redundant_issue)
                
                # Check for missing edge cases
                edge_case_issue = self._check_missing_edge_cases(node, file)
                if edge_case_issue:
                    issues.append(edge_case_issue)
        
        return issues
    
    def _check_mirror_test(self, test_node: ast.FunctionDef, file: str) -> Optional[Issue]:
        """
        Check if test simply mirrors implementation without real validation.
        Pattern: Test that reproduces exact implementation logic.
        """
        # Extract what function is being tested
        tested_func = self._extract_tested_function(test_node)
        if not tested_func or tested_func not in self.implementation_map:
            return None
        
        impl = self.implementation_map[tested_func]
        
        # Check if test logic mirrors implementation
        test_calls = self._extract_calls(test_node)
        impl_calls = self._extract_calls(impl['node'])
        
        # If test reproduces same sequence of operations
        similarity = self._calculate_call_similarity(test_calls, impl_calls)
        
        if similarity > 0.8:  # High similarity indicates mirroring
            return Issue(
                kind="mirror_test",
                message=f"Test '{test_node.name}' appears to mirror implementation of '{tested_func}' rather than validating behavior",
                severity=2,
                file=file,
                line=safe_get_lineno(test_node),
                symbol=test_node.name,
                evidence={
                    'tested_function': tested_func,
                    'similarity_score': similarity
                },
                suggestions=[
                    "Focus on testing expected behavior and edge cases",
                    "Test the contract/interface, not the implementation details",
                    "Add assertions for error conditions and boundary values"
                ]
            )
        
        return None
    
    def _check_brittle_assertions(self, test_node: ast.FunctionDef, file: str) -> Optional[Issue]:
        """
        Check for overly specific assertions that break with minor changes.
        Pattern: Exact string matching, hardcoded values, implementation details.
        """
        brittle_patterns = []
        
        for node in ast.walk(test_node):
            # Check for exact string equality on complex outputs
            if isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                if node.func.attr in ['assertEqual', 'assertEquals']:
                    if len(node.args) >= 2:
                        arg = node.args[1]
                        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                            if len(str(arg.value)) > 100:  # Long exact string match
                                brittle_patterns.append("exact_long_string")
                        elif isinstance(arg, ast.Dict) and len(arg.keys) > 5:
                            brittle_patterns.append("exact_complex_dict")
                
                # Check for assertions on internal state
                if node.func.attr == 'assertEqual' and len(node.args) >= 1:
                    first_arg = node.args[0]
                    if isinstance(first_arg, ast.Attribute):
                        if first_arg.attr.startswith('_'):  # Private attribute
                            brittle_patterns.append("private_attribute_assertion")
        
        if brittle_patterns:
            return Issue(
                kind="brittle_test_assertions",
                message=f"Test '{test_node.name}' contains brittle assertions that may break easily",
                severity=2,
                file=file,
                line=safe_get_lineno(test_node),
                symbol=test_node.name,
                evidence={'patterns': list(set(brittle_patterns))},
                suggestions=[
                    "Use more flexible assertions (assertIn, assertAlmostEqual, etc.)",
                    "Test behavior rather than exact implementation details",
                    "Avoid testing private attributes directly"
                ]
            )
        
        return None
    
    def _check_redundant_test(self, test_node: ast.FunctionDef, file: str, tree: ast.AST) -> Optional[Issue]:
        """
        Check if test is redundant given existing test coverage.
        Pattern: Multiple tests testing the same behavior with minor variations.
        """
        # Find similar tests in the same file
        similar_tests = []
        test_signature = self._get_test_signature(test_node)
        
        for other_node in ast.walk(tree):
            if isinstance(other_node, ast.FunctionDef) and other_node.name.startswith('test_'):
                if other_node.name != test_node.name:
                    other_signature = self._get_test_signature(other_node)
                    similarity = self._calculate_test_similarity(test_signature, other_signature)
                    
                    if similarity > 0.85:  # Very similar tests
                        similar_tests.append((other_node.name, similarity))
        
        if similar_tests:
            return Issue(
                kind="redundant_test",
                message=f"Test '{test_node.name}' appears redundant with existing tests",
                severity=1,
                file=file,
                line=safe_get_lineno(test_node),
                symbol=test_node.name,
                evidence={'similar_tests': similar_tests},
                suggestions=[
                    "Consolidate similar tests using parameterized testing",
                    "Focus on testing different aspects or edge cases",
                    "Remove redundant test coverage"
                ]
            )
        
        return None
    
    def _check_missing_edge_cases(self, test_node: ast.FunctionDef, file: str) -> Optional[Issue]:
        """
        Check if test is missing important edge cases.
        Pattern: Only happy path testing, no error conditions, no boundary values.
        """
        has_error_handling = False
        has_boundary_tests = False
        has_none_checks = False
        
        for node in ast.walk(test_node):
            # Check for exception handling tests
            if isinstance(node, ast.Call) and hasattr(node.func, 'attr'):
                if node.func.attr in ['assertRaises', 'assertRaisesRegex', 'raises']:
                    has_error_handling = True
            
            # Check for None/empty value tests
            if isinstance(node, ast.Constant) and node.value is None:
                has_none_checks = True
            
            # Check for boundary value tests (0, -1, empty lists, etc.)
            if isinstance(node, ast.Constant):
                if node.value in [0, -1, "", [], {}, float('inf'), float('-inf')]:
                    has_boundary_tests = True
        
        missing_coverage = []
        if not has_error_handling:
            missing_coverage.append("error_conditions")
        if not has_boundary_tests:
            missing_coverage.append("boundary_values")
        if not has_none_checks:
            missing_coverage.append("null_handling")
        
        if len(missing_coverage) >= 2:  # Missing multiple types of edge cases
            return Issue(
                kind="incomplete_test_coverage",
                message=f"Test '{test_node.name}' lacks edge case coverage",
                severity=2,
                file=file,
                line=safe_get_lineno(test_node),
                symbol=test_node.name,
                evidence={'missing': missing_coverage},
                suggestions=[
                    "Add tests for error conditions and exceptions",
                    "Test boundary values (0, -1, empty collections)",
                    "Include None/null handling tests",
                    "Test invalid input scenarios"
                ]
            )
        
        return None
    
    def _extract_tested_function(self, test_node: ast.FunctionDef) -> Optional[str]:
        """Extract the name of the function being tested."""
        # Simple heuristic: test_function_name -> function_name
        if test_node.name.startswith('test_'):
            func_name = test_node.name[5:]  # Remove 'test_' prefix
            # Handle common patterns like test_function_name_with_condition
            if '_' in func_name:
                parts = func_name.split('_')
                # Try to find matching function name in implementation map
                for i in range(len(parts), 0, -1):
                    candidate = '_'.join(parts[:i])
                    if candidate in self.implementation_map:
                        return candidate
            return func_name
        return None
    
    def _extract_calls(self, node: ast.AST) -> List[str]:
        """Extract all function calls from a node."""
        calls = []
        for subnode in ast.walk(node):
            if isinstance(subnode, ast.Call):
                if isinstance(subnode.func, ast.Name):
                    calls.append(subnode.func.id)
                elif isinstance(subnode.func, ast.Attribute):
                    calls.append(subnode.func.attr)
        return calls
    
    def _calculate_call_similarity(self, calls1: List[str], calls2: List[str]) -> float:
        """Calculate similarity between two call sequences."""
        if not calls1 or not calls2:
            return 0.0
        
        # Use set intersection for now (could be more sophisticated)
        common = len(set(calls1) & set(calls2))
        total = max(len(calls1), len(calls2))
        
        return common / total if total > 0 else 0.0
    
    def _get_test_signature(self, test_node: ast.FunctionDef) -> Dict:
        """Extract test signature for similarity comparison."""
        return {
            'calls': self._extract_calls(test_node),
            'assertions': [
                node.func.attr for node in ast.walk(test_node)
                if isinstance(node, ast.Call) and hasattr(node.func, 'attr')
                and node.func.attr.startswith('assert')
            ],
            'constants': [
                node.value for node in ast.walk(test_node)
                if isinstance(node, ast.Constant) and isinstance(node.value, (int, str, float))
            ][:10]  # Limit to avoid noise
        }
    
    def _calculate_test_similarity(self, sig1: Dict, sig2: Dict) -> float:
        """Calculate similarity between two test signatures."""
        # Compare different aspects
        call_sim = self._calculate_call_similarity(sig1['calls'], sig2['calls'])
        
        assertion_sim = len(set(sig1['assertions']) & set(sig2['assertions'])) / \
                       max(len(sig1['assertions']), len(sig2['assertions']), 1)
        
        const_sim = len(set(sig1['constants']) & set(sig2['constants'])) / \
                   max(len(sig1['constants']), len(sig2['constants']), 1)
        
        # Weighted average
        return 0.4 * call_sim + 0.4 * assertion_sim + 0.2 * const_sim
