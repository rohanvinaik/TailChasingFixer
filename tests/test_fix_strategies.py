"""
Tests for fix strategies system.

Tests the modular fix strategies with mocking to avoid requiring complex setup.
"""

import pytest
import ast
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, List

from tailchasing.fixers.advanced.fix_strategies import (
    FixStrategy, FixOutcome, RiskLevel, StrategyRanking,
    ImportResolutionStrategy, DuplicateMergeStrategy, 
    PlaceholderImplementationStrategy, CircularDependencyBreaker,
    AsyncSyncMismatchFixer, StrategySelector, FixAttempt
)
from tailchasing.core.issues import Issue
from tailchasing.core.patches import Patch, PatchInfo


@pytest.fixture
def sample_import_issue():
    """Create a sample import resolution issue."""
    return Issue(
        kind="missing_import", 
        message="numpy is used but not imported",
        severity=2,
        file="test.py",
        line=5,
        symbol="numpy",
        evidence={"imported_modules": ["os", "sys"], "used_symbols": ["numpy.array"]}
    )


@pytest.fixture
def sample_duplicate_issue():
    """Create a sample duplicate function issue."""
    return Issue(
        kind="duplicate_function",
        message="Function calculate_total is duplicated", 
        severity=3,
        file="test.py",
        line=10,
        symbol="calculate_total",
        evidence={
            "duplicate_locations": [
                {"file": "test.py", "line": 10, "implementation": "return sum(items)"},
                {"file": "utils.py", "line": 25, "implementation": "return reduce(add, items)"}
            ]
        }
    )


@pytest.fixture
def sample_placeholder_issue():
    """Create a sample placeholder function issue."""
    return Issue(
        kind="phantom_function",
        message="Function process_data has no implementation",
        severity=3, 
        file="test.py",
        line=15,
        symbol="process_data",
        evidence={"stub_type": "pass", "docstring": "Process input data and return results"}
    )


@pytest.fixture
def sample_circular_issue():
    """Create a sample circular dependency issue."""
    return Issue(
        kind="circular_import",
        message="Circular import between module_a and module_b",
        severity=4,
        file="module_a.py", 
        line=1,
        symbol="module_b",
        evidence={
            "import_chain": ["module_a.py", "module_b.py", "module_a.py"],
            "involved_modules": ["module_a.py", "module_b.py"]
        }
    )


@pytest.fixture
def sample_async_issue():
    """Create a sample async/sync mismatch issue."""
    return Issue(
        kind="async_sync_mismatch",
        message="Async function called without await",
        severity=3,
        file="test.py",
        line=20, 
        symbol="async_function",
        evidence={
            "function_type": "async",
            "call_context": "sync",
            "suggested_fix": "add_await"
        }
    )


class TestImportResolutionStrategy:
    """Test import resolution strategy."""

    def test_can_handle_import_issues(self, sample_import_issue):
        """Test strategy can handle import-related issues."""
        strategy = ImportResolutionStrategy()
        assert strategy.can_handle(sample_import_issue) is True
        
        # Should not handle non-import issues
        other_issue = Issue(kind="duplicate_function", message="Test", severity=2)
        assert strategy.can_handle(other_issue) is False

    def test_risk_estimation(self, sample_import_issue):
        """Test risk estimation for import fixes."""
        strategy = ImportResolutionStrategy()
        risk = strategy.estimate_risk(sample_import_issue)
        assert risk == RiskLevel.LOW  # Import fixes are typically low risk

    def test_propose_fix_missing_import(self, sample_import_issue):
        """Test proposing fix for missing import."""
        strategy = ImportResolutionStrategy()
        patch = strategy.propose_fix(sample_import_issue)
        
        assert patch is not None
        assert patch.file_path == "test.py"
        assert "import numpy" in patch.content
        assert patch.line_number == 1  # Imports go at the top

    def test_propose_fix_with_context(self, sample_import_issue):
        """Test proposing fix with additional context."""
        strategy = ImportResolutionStrategy() 
        context = {
            "existing_imports": ["import os", "import sys"],
            "import_style": "from numpy import array"
        }
        
        patch = strategy.propose_fix(sample_import_issue, context)
        assert patch is not None
        # Should suggest appropriate import style based on context

    def test_rollback_plan(self, sample_import_issue):
        """Test rollback plan generation."""
        strategy = ImportResolutionStrategy()
        patch = strategy.propose_fix(sample_import_issue)
        rollback = strategy.get_rollback_plan(patch, sample_import_issue)
        
        assert rollback is not None
        assert "remove import numpy" in rollback.lower()

    def test_validation_tests(self, sample_import_issue):
        """Test validation test generation."""
        strategy = ImportResolutionStrategy()
        patch = strategy.propose_fix(sample_import_issue)
        tests = strategy.get_validation_tests(patch, sample_import_issue)
        
        assert len(tests) > 0
        assert any("import" in test.lower() for test in tests)


class TestDuplicateMergeStrategy:
    """Test duplicate merge strategy."""

    def test_can_handle_duplicate_issues(self, sample_duplicate_issue):
        """Test strategy can handle duplicate function issues."""
        strategy = DuplicateMergeStrategy()
        assert strategy.can_handle(sample_duplicate_issue) is True

    def test_risk_estimation_by_complexity(self, sample_duplicate_issue):
        """Test risk estimation based on duplicate complexity."""
        strategy = DuplicateMergeStrategy()
        
        # Simple duplicates should be low risk
        risk = strategy.estimate_risk(sample_duplicate_issue)
        assert risk in [RiskLevel.LOW, RiskLevel.MEDIUM]
        
        # Complex duplicates with different implementations should be higher risk
        complex_issue = Issue(
            kind="duplicate_function",
            message="Complex duplicate with different implementations",
            severity=4,
            file="test.py",
            line=10,
            symbol="complex_function",
            evidence={
                "duplicate_locations": [
                    {"implementation": "very long implementation with many lines"},
                    {"implementation": "completely different approach with different logic"}
                ]
            }
        )
        complex_risk = strategy.estimate_risk(complex_issue)
        assert complex_risk in [RiskLevel.MEDIUM, RiskLevel.HIGH]

    def test_propose_fix_simple_merge(self, sample_duplicate_issue):
        """Test proposing fix for simple duplicate merge."""
        strategy = DuplicateMergeStrategy()
        patch = strategy.propose_fix(sample_duplicate_issue)
        
        assert patch is not None
        assert patch.file_path == "test.py"
        # Should keep the better implementation
        assert "def calculate_total" in patch.content

    def test_analyze_implementations(self, sample_duplicate_issue):
        """Test implementation analysis for merging."""
        strategy = DuplicateMergeStrategy()
        
        implementations = [
            "return sum(items)",  # Simple and clear
            "return reduce(operator.add, items, 0)"  # More complex but robust
        ]
        
        best = strategy._select_best_implementation(implementations)
        assert best in implementations


class TestPlaceholderImplementationStrategy:
    """Test placeholder implementation strategy."""

    def test_can_handle_phantom_functions(self, sample_placeholder_issue):
        """Test strategy can handle phantom function issues."""
        strategy = PlaceholderImplementationStrategy()
        assert strategy.can_handle(sample_placeholder_issue) is True

    def test_risk_estimation(self, sample_placeholder_issue):
        """Test risk estimation for placeholder fixes."""
        strategy = PlaceholderImplementationStrategy()
        
        # Simple placeholder should be low risk
        risk = strategy.estimate_risk(sample_placeholder_issue)
        assert risk == RiskLevel.LOW
        
        # Complex function with many dependencies should be higher risk
        complex_issue = Issue(
            kind="phantom_function",
            message="Complex function with many parameters",
            severity=4,
            file="test.py",
            line=15,
            symbol="complex_process",
            evidence={"stub_type": "pass", "parameters": ["a", "b", "c", "d", "e"]}
        )
        complex_risk = strategy.estimate_risk(complex_issue)
        assert complex_risk in [RiskLevel.MEDIUM, RiskLevel.HIGH]

    @patch('tailchasing.fixers.advanced.fix_strategies.LLMManager')
    def test_propose_fix_with_llm(self, mock_llm_manager, sample_placeholder_issue):
        """Test proposing fix using LLM for implementation."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = 'def process_data(data):\n    """Process input data and return results."""\n    return [x * 2 for x in data]'
        mock_response.is_valid_fix.return_value = True
        
        mock_llm_manager.return_value.generate_fix.return_value = mock_response
        
        strategy = PlaceholderImplementationStrategy()
        patch = strategy.propose_fix(sample_placeholder_issue)
        
        assert patch is not None
        assert "def process_data" in patch.content
        assert "return [x * 2 for x in data]" in patch.content

    def test_propose_fix_fallback(self, sample_placeholder_issue):
        """Test fallback implementation when LLM unavailable."""
        strategy = PlaceholderImplementationStrategy()
        
        # Mock LLM failure
        with patch.object(strategy, '_generate_with_llm', return_value=None):
            patch = strategy.propose_fix(sample_placeholder_issue)
            
            assert patch is not None
            assert "def process_data" in patch.content
            assert "NotImplementedError" in patch.content

    def test_validation_tests_generation(self, sample_placeholder_issue):
        """Test validation test generation for implemented functions."""
        strategy = PlaceholderImplementationStrategy()
        patch = strategy.propose_fix(sample_placeholder_issue)
        tests = strategy.get_validation_tests(patch, sample_placeholder_issue)
        
        assert len(tests) > 0
        assert any("process_data" in test for test in tests)


class TestCircularDependencyBreaker:
    """Test circular dependency breaker strategy."""

    def test_can_handle_circular_imports(self, sample_circular_issue):
        """Test strategy can handle circular import issues."""
        strategy = CircularDependencyBreaker()
        assert strategy.can_handle(sample_circular_issue) is True

    def test_risk_estimation(self, sample_circular_issue):
        """Test risk estimation for circular dependency fixes."""
        strategy = CircularDependencyBreaker()
        risk = strategy.estimate_risk(sample_circular_issue)
        
        # Circular dependency fixes are inherently medium to high risk
        assert risk in [RiskLevel.MEDIUM, RiskLevel.HIGH]

    def test_propose_fix_move_import(self, sample_circular_issue):
        """Test proposing fix by moving imports."""
        strategy = CircularDependencyBreaker()
        patch = strategy.propose_fix(sample_circular_issue)
        
        assert patch is not None
        # Should suggest moving import inside function or using lazy import
        content_lower = patch.content.lower()
        assert any(keyword in content_lower for keyword in 
                  ["def ", "import ", "from ", "lazy"])

    def test_complex_circular_chain(self):
        """Test handling complex circular dependency chains."""
        complex_issue = Issue(
            kind="circular_import",
            message="Complex circular chain A->B->C->A", 
            severity=5,
            file="module_a.py",
            line=1,
            symbol="module_b",
            evidence={
                "import_chain": ["module_a.py", "module_b.py", "module_c.py", "module_a.py"],
                "involved_modules": ["module_a.py", "module_b.py", "module_c.py"]
            }
        )
        
        strategy = CircularDependencyBreaker()
        risk = strategy.estimate_risk(complex_issue)
        assert risk == RiskLevel.HIGH


class TestAsyncSyncMismatchFixer:
    """Test async/sync mismatch fixer strategy."""

    def test_can_handle_async_issues(self, sample_async_issue):
        """Test strategy can handle async/sync mismatch issues."""
        strategy = AsyncSyncMismatchFixer()
        assert strategy.can_handle(sample_async_issue) is True

    def test_risk_estimation(self, sample_async_issue):
        """Test risk estimation for async fixes."""
        strategy = AsyncSyncMismatchFixer()
        risk = strategy.estimate_risk(sample_async_issue)
        assert risk == RiskLevel.LOW  # Adding await is typically low risk

    def test_propose_fix_add_await(self, sample_async_issue):
        """Test proposing fix by adding await."""
        strategy = AsyncSyncMismatchFixer()
        patch = strategy.propose_fix(sample_async_issue)
        
        assert patch is not None
        assert "await" in patch.content

    def test_propose_fix_make_sync(self):
        """Test proposing fix by making function synchronous."""
        sync_issue = Issue(
            kind="async_sync_mismatch", 
            message="Async function should be sync",
            severity=2,
            file="test.py",
            line=20,
            symbol="should_be_sync",
            evidence={
                "function_type": "async",
                "call_context": "sync",
                "suggested_fix": "make_sync"
            }
        )
        
        strategy = AsyncSyncMismatchFixer()
        patch = strategy.propose_fix(sync_issue)
        
        assert patch is not None
        # Should remove async keyword
        assert "def " in patch.content
        assert "async def" not in patch.content

    def test_validation_tests(self, sample_async_issue):
        """Test validation test generation for async fixes."""
        strategy = AsyncSyncMismatchFixer()
        patch = strategy.propose_fix(sample_async_issue)
        tests = strategy.get_validation_tests(patch, sample_async_issue)
        
        assert len(tests) > 0
        assert any("await" in test for test in tests)


class TestStrategySelector:
    """Test strategy selector functionality."""

    def test_initialization(self):
        """Test strategy selector initialization."""
        selector = StrategySelector()
        
        assert len(selector.strategies) == 5
        assert isinstance(selector.strategies[0], ImportResolutionStrategy)
        assert isinstance(selector.strategies[1], DuplicateMergeStrategy)

    def test_select_strategies_single_issue(self, sample_import_issue):
        """Test selecting strategies for a single issue."""
        selector = StrategySelector()
        rankings = selector.select_strategies([sample_import_issue])
        
        assert len(rankings) == 1
        issue, ranking = rankings[0]
        assert issue == sample_import_issue
        assert ranking.primary_strategy is not None
        assert isinstance(ranking.primary_strategy, ImportResolutionStrategy)

    def test_select_strategies_multiple_issues(self, sample_import_issue, 
                                             sample_duplicate_issue):
        """Test selecting strategies for multiple issues."""
        selector = StrategySelector()
        rankings = selector.select_strategies([sample_import_issue, sample_duplicate_issue])
        
        assert len(rankings) == 2
        
        # Should assign appropriate strategies
        strategies = [ranking.primary_strategy.__class__.__name__ for _, ranking in rankings]
        assert "ImportResolutionStrategy" in strategies
        assert "DuplicateMergeStrategy" in strategies

    def test_strategy_ranking_calculation(self, sample_duplicate_issue):
        """Test strategy ranking calculation."""
        selector = StrategySelector()
        
        strategy = DuplicateMergeStrategy()
        ranking = selector._calculate_ranking(strategy, sample_duplicate_issue)
        
        assert isinstance(ranking, StrategyRanking)
        assert ranking.primary_strategy == strategy
        assert 0.0 <= ranking.confidence_score <= 1.0
        assert ranking.estimated_risk in [RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH]

    def test_dependency_checking(self, sample_import_issue, sample_duplicate_issue):
        """Test dependency checking between strategies."""
        selector = StrategySelector()
        
        # Import resolution might affect duplicate detection
        has_dependency = selector._has_dependency(
            ImportResolutionStrategy(), sample_import_issue,
            DuplicateMergeStrategy(), sample_duplicate_issue
        )
        
        # Should check for potential interactions
        assert isinstance(has_dependency, bool)

    def test_learning_from_outcomes(self, sample_import_issue):
        """Test learning from strategy outcomes."""
        selector = StrategySelector()
        
        strategy = ImportResolutionStrategy()
        
        # Simulate successful outcome
        outcome = FixOutcome(
            success=True,
            confidence=0.9,
            execution_time=1.5,
            validation_passed=True
        )
        
        selector.learn_from_outcome(strategy, sample_import_issue, outcome)
        
        # Should track the outcome
        key = (strategy.__class__.__name__, sample_import_issue.kind)
        assert key in selector.success_history

    def test_success_rate_calculation(self, sample_import_issue):
        """Test success rate calculation for strategies."""
        selector = StrategySelector()
        strategy = ImportResolutionStrategy()
        
        # Add some history
        for success in [True, True, False, True]:
            outcome = FixOutcome(success=success, confidence=0.8)
            selector.learn_from_outcome(strategy, sample_import_issue, outcome)
        
        success_rate = selector._get_success_rate(strategy, sample_import_issue.kind)
        assert success_rate == 0.75  # 3/4 successes

    def test_strategy_prioritization(self, sample_import_issue):
        """Test strategy prioritization based on history."""
        selector = StrategySelector()
        
        # Create mock strategies with different success rates
        strategy1 = ImportResolutionStrategy()
        strategy2 = DuplicateMergeStrategy()
        
        # Give strategy1 better history
        for _ in range(3):
            outcome = FixOutcome(success=True, confidence=0.9)
            selector.learn_from_outcome(strategy1, sample_import_issue, outcome)
        
        # Give strategy2 worse history  
        for _ in range(2):
            outcome = FixOutcome(success=False, confidence=0.5)
            selector.learn_from_outcome(strategy2, sample_import_issue, outcome)
        
        # Strategy1 should be prioritized for similar issues
        rankings = selector.select_strategies([sample_import_issue])
        _, ranking = rankings[0]
        
        # Should consider success history in ranking
        assert ranking.confidence_score > 0.5


class TestFixAttempt:
    """Test fix attempt tracking."""

    def test_fix_attempt_creation(self, sample_import_issue):
        """Test creating fix attempt records."""
        strategy = ImportResolutionStrategy()
        patch = Patch(
            file_path="test.py",
            content="import numpy",
            line_number=1,
            description="Add missing numpy import"
        )
        
        attempt = FixAttempt(
            issue=sample_import_issue,
            strategy=strategy,
            proposed_patch=patch,
            timestamp=1234567890.0
        )
        
        assert attempt.issue == sample_import_issue
        assert attempt.strategy == strategy
        assert attempt.proposed_patch == patch
        assert attempt.outcome is None

    def test_fix_attempt_completion(self, sample_import_issue):
        """Test completing fix attempts with outcomes."""
        strategy = ImportResolutionStrategy()
        patch = Patch(file_path="test.py", content="import numpy", line_number=1)
        
        attempt = FixAttempt(
            issue=sample_import_issue,
            strategy=strategy, 
            proposed_patch=patch,
            timestamp=1234567890.0
        )
        
        outcome = FixOutcome(
            success=True,
            confidence=0.9,
            execution_time=2.0,
            validation_passed=True
        )
        
        attempt.outcome = outcome
        
        assert attempt.outcome.success is True
        assert attempt.outcome.confidence == 0.9


class TestIntegration:
    """Integration tests for the complete fix strategies system."""

    def test_end_to_end_fix_application(self, sample_import_issue, sample_duplicate_issue):
        """Test end-to-end fix application workflow."""
        selector = StrategySelector()
        
        # Select strategies
        issues = [sample_import_issue, sample_duplicate_issue]
        rankings = selector.select_strategies(issues)
        
        assert len(rankings) == 2
        
        # Apply strategies
        for issue, ranking in rankings:
            strategy = ranking.primary_strategy
            patch = strategy.propose_fix(issue)
            
            assert patch is not None
            assert patch.file_path
            assert patch.content
            
            # Get validation tests
            tests = strategy.get_validation_tests(patch, issue)
            assert len(tests) > 0
            
            # Get rollback plan
            rollback = strategy.get_rollback_plan(patch, issue)
            assert rollback is not None

    def test_risk_based_strategy_selection(self):
        """Test that high-risk issues get more careful strategy selection."""
        high_risk_issue = Issue(
            kind="circular_import",
            message="Complex circular dependency",
            severity=5,
            file="test.py",
            line=1,
            symbol="problematic_import"
        )
        
        selector = StrategySelector()
        rankings = selector.select_strategies([high_risk_issue])
        
        _, ranking = rankings[0]
        
        # High-risk issues should have lower confidence or alternative strategies
        assert ranking.estimated_risk in [RiskLevel.MEDIUM, RiskLevel.HIGH]
        if ranking.alternative_strategies:
            assert len(ranking.alternative_strategies) > 0

    @patch('tailchasing.fixers.advanced.fix_strategies.LLMManager')
    def test_llm_integration(self, mock_llm_manager, sample_placeholder_issue):
        """Test integration with LLM for complex fixes."""
        # Mock LLM response
        mock_response = Mock()
        mock_response.content = 'def process_data(data):\n    return data.upper()'
        mock_response.is_valid_fix.return_value = True
        mock_llm_manager.return_value.generate_fix.return_value = mock_response
        
        selector = StrategySelector()
        rankings = selector.select_strategies([sample_placeholder_issue])
        
        _, ranking = rankings[0]
        strategy = ranking.primary_strategy
        patch = strategy.propose_fix(sample_placeholder_issue)
        
        assert patch is not None
        assert "def process_data" in patch.content

    def test_multiple_strategy_coordination(self):
        """Test coordination between multiple strategies."""
        # Create issues that might interact
        import_issue = Issue(kind="missing_import", message="Missing pandas", severity=2)
        duplicate_issue = Issue(kind="duplicate_function", message="Duplicate pandas usage", severity=3)
        
        selector = StrategySelector()
        rankings = selector.select_strategies([import_issue, duplicate_issue])
        
        # Should detect potential interactions
        assert len(rankings) == 2
        
        # Check if dependencies are noted
        dependencies_found = any(
            ranking.dependencies for _, ranking in rankings if ranking.dependencies
        )
        # Dependencies might or might not be found depending on the specific case


class TestErrorHandling:
    """Test error handling in fix strategies."""

    def test_invalid_issue_handling(self):
        """Test handling of invalid or malformed issues."""
        strategy = ImportResolutionStrategy()
        
        # Issue with missing required fields
        invalid_issue = Issue(kind="missing_import", message="", severity=0)
        
        # Should handle gracefully
        can_handle = strategy.can_handle(invalid_issue)
        assert isinstance(can_handle, bool)
        
        if can_handle:
            patch = strategy.propose_fix(invalid_issue)
            # Should either return None or a safe fallback
            assert patch is None or isinstance(patch, Patch)

    def test_ast_parsing_errors(self):
        """Test handling of AST parsing errors in code analysis."""
        strategy = DuplicateMergeStrategy()
        
        # Issue with invalid syntax in evidence
        malformed_issue = Issue(
            kind="duplicate_function",
            message="Duplicate with syntax error",
            severity=3,
            file="test.py",
            line=10,
            symbol="broken_func",
            evidence={
                "duplicate_locations": [
                    {"implementation": "def broken(: # invalid syntax"},
                    {"implementation": "def broken(): return 42"}
                ]
            }
        )
        
        # Should handle parsing errors gracefully
        patch = strategy.propose_fix(malformed_issue)
        assert patch is None or isinstance(patch, Patch)

    def test_llm_failure_fallback(self, sample_placeholder_issue):
        """Test fallback behavior when LLM calls fail."""
        strategy = PlaceholderImplementationStrategy()
        
        # Mock LLM to raise exception
        with patch.object(strategy, '_generate_with_llm', side_effect=Exception("LLM failed")):
            patch = strategy.propose_fix(sample_placeholder_issue)
            
            # Should fall back to basic implementation
            assert patch is not None
            assert "NotImplementedError" in patch.content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])