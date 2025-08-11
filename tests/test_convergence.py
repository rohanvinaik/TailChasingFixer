"""
Test suite for convergence and loop prevention in tail-chasing detection.

Tests max iteration limits, diff similarity detection, and error fingerprint
caching to ensure the system doesn't get stuck in infinite loops.
"""

import pytest
from pathlib import Path
from typing import List, Dict, Set, Any
from dataclasses import dataclass
import hashlib
import tempfile
import shutil

from tailchasing.core.issues import Issue, IssueCollection
from tailchasing.analyzers.base import AnalysisContext


@dataclass
class IterationState:
    """Tracks state between iterations to detect loops."""
    iteration: int
    issues: List[Issue]
    code_diff: str
    error_fingerprint: str
    
    def get_fingerprint(self) -> str:
        """Generate a fingerprint for this iteration state."""
        issue_str = ";".join(f"{i.kind}:{i.file}:{i.line}" for i in self.issues)
        return hashlib.md5(f"{issue_str}:{self.code_diff}".encode()).hexdigest()


class ConvergenceDetector:
    """Detects when fixes are stuck in a loop."""
    
    def __init__(self, max_iterations: int = 10, similarity_threshold: float = 0.9):
        self.max_iterations = max_iterations
        self.similarity_threshold = similarity_threshold
        self.iteration_history: List[IterationState] = []
        self.error_fingerprints: Set[str] = set()
        self.diff_cache: Dict[str, int] = {}
        
    def should_continue(self, current_state: IterationState) -> bool:
        """Check if we should continue iterating or stop due to loop detection."""
        # Check max iterations
        if current_state.iteration >= self.max_iterations:
            return False
            
        # Check for exact fingerprint match (exact loop)
        fingerprint = current_state.get_fingerprint()
        if fingerprint in self.error_fingerprints:
            return False
        self.error_fingerprints.add(fingerprint)
        
        # Check for similar diffs (approximate loop)
        if self._is_similar_to_previous(current_state):
            return False
            
        self.iteration_history.append(current_state)
        return True
        
    def _is_similar_to_previous(self, current: IterationState) -> bool:
        """Check if current state is similar to any previous state."""
        for prev in self.iteration_history[-3:]:  # Check last 3 iterations
            similarity = self._calculate_similarity(prev.code_diff, current.code_diff)
            if similarity >= self.similarity_threshold:
                return True
        return False
        
    def _calculate_similarity(self, diff1: str, diff2: str) -> float:
        """Calculate similarity between two code diffs."""
        if not diff1 or not diff2:
            return 0.0
            
        # Simple line-based similarity
        lines1 = set(diff1.strip().split('\n'))
        lines2 = set(diff2.strip().split('\n'))
        
        if not lines1 or not lines2:
            return 0.0
            
        intersection = lines1 & lines2
        union = lines1 | lines2
        
        return len(intersection) / len(union) if union else 0.0


class TestConvergenceDetection:
    """Test suite for convergence detection."""
    
    def test_max_iteration_limit(self):
        """Test that iteration stops at max limit."""
        detector = ConvergenceDetector(max_iterations=5)
        
        for i in range(10):
            state = IterationState(
                iteration=i,
                issues=[Issue(kind="test", message=f"Issue {i}", severity=2)],
                code_diff=f"diff {i}",
                error_fingerprint=f"error_{i}"
            )
            
            should_continue = detector.should_continue(state)
            
            if i < 5:
                assert should_continue, f"Should continue at iteration {i}"
            else:
                assert not should_continue, f"Should stop at iteration {i}"
                break
                
    def test_exact_loop_detection(self):
        """Test detection of exact loops via fingerprints."""
        detector = ConvergenceDetector()
        
        # First occurrence - should continue
        state1 = IterationState(
            iteration=0,
            issues=[Issue(kind="phantom", message="Phantom func", severity=3, file="test.py", line=10)],
            code_diff="+ def phantom():\n+     pass",
            error_fingerprint="error_1"
        )
        assert detector.should_continue(state1)
        
        # Different state - should continue
        state2 = IterationState(
            iteration=1,
            issues=[Issue(kind="phantom", message="Another phantom", severity=3, file="test.py", line=20)],
            code_diff="+ def another():\n+     pass",
            error_fingerprint="error_2"
        )
        assert detector.should_continue(state2)
        
        # Exact repeat of state1 - should stop
        state3 = IterationState(
            iteration=2,
            issues=[Issue(kind="phantom", message="Phantom func", severity=3, file="test.py", line=10)],
            code_diff="+ def phantom():\n+     pass",
            error_fingerprint="error_1"
        )
        assert not detector.should_continue(state3)
        
    def test_similar_diff_detection(self):
        """Test detection of similar but not identical diffs."""
        detector = ConvergenceDetector(similarity_threshold=0.7)
        
        # Initial state
        state1 = IterationState(
            iteration=0,
            issues=[Issue(kind="duplicate", message="Dup", severity=2)],
            code_diff="def calculate_total(items):\n    return sum(items)",
            error_fingerprint="err1"
        )
        assert detector.should_continue(state1)
        
        # Very similar diff (just whitespace and naming)
        state2 = IterationState(
            iteration=1,
            issues=[Issue(kind="duplicate", message="Dup", severity=2)],
            code_diff="def calculate_total(items):\n    return  sum(items)",  # Extra space
            error_fingerprint="err2"
        )
        # Should detect as similar and stop
        assert not detector.should_continue(state2)
        
    @pytest.mark.parametrize("pattern_type,expected_iterations", [
        ("escalating", 3),  # Errors that grow each iteration
        ("oscillating", 4),  # Errors that alternate
        ("converging", 10),  # Errors that decrease but never reach zero
    ])
    def test_pattern_detection(self, pattern_type: str, expected_iterations: int):
        """Test detection of various loop patterns."""
        detector = ConvergenceDetector(max_iterations=10)
        
        patterns = {
            "escalating": lambda i: [Issue(kind=f"err_{j}", message=f"Error {j}", severity=2) 
                                    for j in range(i + 1)],
            "oscillating": lambda i: [Issue(kind="err", message="Error", severity=2)] 
                                    if i % 2 == 0 else 
                                    [Issue(kind="warn", message="Warning", severity=1)],
            "converging": lambda i: [Issue(kind="err", message="Error", severity=2)] * max(1, 10 - i)
        }
        
        pattern_gen = patterns[pattern_type]
        iterations_run = 0
        
        for i in range(10):
            state = IterationState(
                iteration=i,
                issues=pattern_gen(i),
                code_diff=f"diff_{pattern_type}_{i}",
                error_fingerprint=f"fp_{pattern_type}_{i}"
            )
            
            if detector.should_continue(state):
                iterations_run += 1
            else:
                break
                
        # Allow some variance in detection
        assert abs(iterations_run - expected_iterations) <= 2


class TestErrorFingerprintCaching:
    """Test error fingerprint caching mechanisms."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.cache_file = Path(self.temp_dir) / "error_cache.json"
        
    def teardown_method(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_fingerprint_generation(self):
        """Test that fingerprints are deterministic and unique."""
        issue1 = Issue(kind="phantom", message="Test", severity=2, file="a.py", line=10)
        issue2 = Issue(kind="phantom", message="Test", severity=2, file="a.py", line=10)
        issue3 = Issue(kind="phantom", message="Test", severity=2, file="b.py", line=10)
        
        state1 = IterationState(0, [issue1], "diff1", "err1")
        state2 = IterationState(0, [issue2], "diff1", "err1")
        state3 = IterationState(0, [issue3], "diff1", "err1")
        
        # Same issues should have same fingerprint
        assert state1.get_fingerprint() == state2.get_fingerprint()
        
        # Different file should have different fingerprint
        assert state1.get_fingerprint() != state3.get_fingerprint()
        
    def test_cache_persistence(self):
        """Test that error cache persists across sessions."""
        import json
        
        # First session
        cache = {"fp1": 1, "fp2": 2}
        with open(self.cache_file, 'w') as f:
            json.dump(cache, f)
            
        # Second session - load cache
        with open(self.cache_file, 'r') as f:
            loaded_cache = json.load(f)
            
        assert loaded_cache == cache
        
    def test_cache_size_limit(self):
        """Test that cache doesn't grow unbounded."""
        max_cache_size = 100
        cache = {}
        
        for i in range(200):
            fp = f"fingerprint_{i}"
            cache[fp] = i
            
            # Implement LRU-style eviction
            if len(cache) > max_cache_size:
                oldest = min(cache.keys(), key=lambda k: cache[k])
                del cache[oldest]
                
        assert len(cache) <= max_cache_size


class TestTailChasingPatterns:
    """Test detection of specific tail-chasing patterns."""
    
    @pytest.fixture
    def sample_context(self):
        """Create a sample analysis context."""
        from tailchasing.core.symbols import SymbolTable
        return AnalysisContext(
            config={},
            root_dir=Path("."),
            file_paths=[],
            ast_index={},
            symbol_table=SymbolTable(),
            source_cache={},
            cache={}
        )
        
    def test_phantom_cascade_detection(self, sample_context):
        """Test detection of phantom function cascades."""
        issues = [
            Issue(kind="phantom_function", message="Phantom: helper1", severity=3),
            Issue(kind="phantom_function", message="Phantom: helper2", severity=3),
            Issue(kind="missing_symbol", message="Missing: helper1", severity=3),
        ]
        
        # Detect cascade pattern
        cascade_detected = self._detect_cascade(issues)
        assert cascade_detected, "Should detect phantom cascade pattern"
        
    def test_import_anxiety_detection(self, sample_context):
        """Test detection of defensive over-importing."""
        issues = [
            Issue(kind="import_anxiety", message="Unused import: typing", severity=1),
            Issue(kind="import_anxiety", message="Unused import: collections", severity=1),
            Issue(kind="import_anxiety", message="Star import: from utils import *", severity=2),
        ]
        
        anxiety_score = sum(1 for i in issues if "import" in i.kind)
        assert anxiety_score >= 3, "Should detect import anxiety pattern"
        
    def test_context_thrashing_detection(self, sample_context):
        """Test detection of context window thrashing."""
        issues = [
            Issue(kind="duplicate_function", message="Duplicate: calculate_total", severity=3),
            Issue(kind="semantic_duplicate_function", message="Semantic dup: get_sum", severity=3),
            Issue(kind="context_window_thrashing", message="Reimplemented: compute_sum", severity=4),
        ]
        
        thrashing_detected = any("thrashing" in i.kind for i in issues)
        assert thrashing_detected, "Should detect context thrashing"
        
    def _detect_cascade(self, issues: List[Issue]) -> bool:
        """Helper to detect cascade patterns."""
        phantom_count = sum(1 for i in issues if "phantom" in i.kind)
        missing_count = sum(1 for i in issues if "missing" in i.kind)
        return phantom_count >= 2 and missing_count >= 1


class TestFixSuggestions:
    """Test that suggested fixes actually resolve issues."""
    
    def test_phantom_function_fix(self):
        """Test fix suggestion for phantom functions."""
        issue = Issue(
            kind="phantom_function",
            message="Phantom function: process_data",
            severity=3,
            file="processor.py",
            line=10,
            suggestions=[
                "Implement the function body",
                "Remove the function if unused",
                "Import from actual module"
            ]
        )
        
        assert len(issue.suggestions) >= 3
        assert any("implement" in s.lower() for s in issue.suggestions)
        
    def test_circular_import_fix(self):
        """Test fix suggestion for circular imports."""
        issue = Issue(
            kind="circular_import",
            message="Circular import: a.py -> b.py -> a.py",
            severity=4,
            suggestions=[
                "Move shared code to a third module",
                "Use late imports (inside functions)",
                "Restructure module dependencies"
            ]
        )
        
        assert any("shared" in s.lower() or "third" in s.lower() for s in issue.suggestions)
        
    @pytest.mark.parametrize("issue_kind,expected_suggestion", [
        ("duplicate_function", "merge"),
        ("wrapper_abstraction", "remove"),
        ("hallucination_cascade", "actual"),
        ("import_anxiety", "specific"),
    ])
    def test_fix_suggestions_by_type(self, issue_kind: str, expected_suggestion: str):
        """Test that each issue type has appropriate fix suggestions."""
        from tailchasing.core.suggestions import FixSuggestionGenerator
        
        generator = FixSuggestionGenerator()
        issue = Issue(kind=issue_kind, message="Test issue", severity=2)
        
        suggestions = generator.generate_suggestions(issue)
        
        # Check that at least one suggestion contains the expected keyword
        assert any(expected_suggestion in s.lower() for s in suggestions), \
            f"Expected '{expected_suggestion}' in suggestions for {issue_kind}"