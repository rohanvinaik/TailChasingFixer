"""Tests for root cause clustering functionality."""

import ast
import pytest
from pathlib import Path
import tempfile
import textwrap

from tailchasing.core.issues import Issue
from tailchasing.analyzers.root_cause_clustering import (
    RootCauseClusterer,
    ASTNormalizer,
    ASTHasher,
    IssueCluster,
    FixPlaybook
)


class TestASTNormalizer:
    """Test AST normalization functionality."""
    
    def test_variable_normalization(self):
        """Test that variable names are normalized to placeholders."""
        code = "x = 5; y = x + 1"
        tree = ast.parse(code)
        normalizer = ASTNormalizer()
        normalized = normalizer.visit(tree)
        
        # Check that variables are normalized
        names = [node.id for node in ast.walk(normalized) if isinstance(node, ast.Name)]
        assert all(name.startswith("VAR_") for name in names)
        assert len(set(names)) == 2  # Two unique variables
    
    def test_literal_normalization(self):
        """Test that literals are replaced with type placeholders."""
        code = 'x = "hello"; y = 42; z = True'
        tree = ast.parse(code)
        normalizer = ASTNormalizer()
        normalized = normalizer.visit(tree)
        
        # Check literals are normalized
        constants = [node.value for node in ast.walk(normalized) if isinstance(node, ast.Constant)]
        assert "STR_LITERAL" in constants
        assert "NUM_LITERAL" in constants
        assert "BOOL_LITERAL" in constants
    
    def test_function_normalization(self):
        """Test that function names are normalized."""
        code = "def my_function(): pass"
        tree = ast.parse(code)
        normalizer = ASTNormalizer(preserve_structure=True)
        normalized = normalizer.visit(tree)
        
        # Check function name is normalized
        func_defs = [node for node in ast.walk(normalized) if isinstance(node, ast.FunctionDef)]
        assert len(func_defs) == 1
        assert func_defs[0].name == "FUNC"


class TestASTHasher:
    """Test AST hashing functionality."""
    
    def test_exact_hash_identical_code(self):
        """Test that identical normalized code produces same hash."""
        code1 = "x = 5; y = x + 1"
        code2 = "a = 10; b = a + 1"  # Different vars/values, same structure
        
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        hash1 = ASTHasher.exact_hash(tree1)
        hash2 = ASTHasher.exact_hash(tree2)
        
        assert hash1 == hash2  # Should be same after normalization
    
    def test_exact_hash_different_code(self):
        """Test that different code produces different hashes."""
        code1 = "x = 5"
        code2 = "x = 5; y = 10"
        
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        hash1 = ASTHasher.exact_hash(tree1)
        hash2 = ASTHasher.exact_hash(tree2)
        
        assert hash1 != hash2
    
    def test_sexpr_conversion(self):
        """Test S-expression conversion."""
        code = "x = 5"
        tree = ast.parse(code)
        sexpr = ASTHasher.ast_to_sexpr(tree)
        
        assert sexpr.startswith("(Module")
        assert "Assign" in sexpr
    
    def test_structural_hash(self):
        """Test structural hashing ignores all values."""
        code1 = "def foo(): return 1"
        code2 = "def bar(): return 2"
        
        tree1 = ast.parse(code1)
        tree2 = ast.parse(code2)
        
        hash1 = ASTHasher.structural_hash(tree1)
        hash2 = ASTHasher.structural_hash(tree2)
        
        assert hash1 == hash2  # Same structure


class TestRootCauseClusterer:
    """Test root cause clustering functionality."""
    
    def create_test_issues(self) -> list:
        """Create test issues for clustering."""
        return [
            Issue(
                kind="duplicate_function",
                message="Duplicate function found",
                severity=2,
                file="test1.py",
                line=10
            ),
            Issue(
                kind="duplicate_function",
                message="Duplicate function found",
                severity=2,
                file="test2.py",
                line=20
            ),
            Issue(
                kind="circular_import",
                message="Circular import detected",
                severity=3,
                file="module1.py",
                line=1
            ),
            Issue(
                kind="circular_import",
                message="Circular import detected",
                severity=3,
                file="module2.py",
                line=1
            ),
            Issue(
                kind="phantom_function",
                message="Stub implementation",
                severity=1,
                file="stubs.py",
                line=15
            )
        ]
    
    def test_cluster_creation(self):
        """Test that clusters are created correctly."""
        issues = self.create_test_issues()
        clusterer = RootCauseClusterer(similarity_threshold=0.7)
        
        # Note: Without actual file content, clustering will be limited
        # This tests the API and basic functionality
        clusters = clusterer.cluster(issues)
        
        assert isinstance(clusters, list)
        for cluster in clusters:
            assert isinstance(cluster, IssueCluster)
            assert cluster.cluster_id
            assert cluster.representative_issue
            assert cluster.members
            assert cluster.root_cause_guess
            assert cluster.fix_playbook_id
    
    def test_cluster_with_files(self, tmp_path):
        """Test clustering with actual file content."""
        # Create test files with duplicate code
        file1 = tmp_path / "file1.py"
        file2 = tmp_path / "file2.py"
        
        code = textwrap.dedent("""
            def calculate(x):
                return x * 2
        """)
        
        file1.write_text(code)
        file2.write_text(code)
        
        issues = [
            Issue(
                kind="duplicate_function",
                message="Duplicate function",
                severity=2,
                file=str(file1),
                line=2
            ),
            Issue(
                kind="duplicate_function",
                message="Duplicate function",
                severity=2,
                file=str(file2),
                line=2
            )
        ]
        
        clusterer = RootCauseClusterer()
        clusters = clusterer.cluster(issues)
        
        # Should cluster the duplicate functions together
        assert len(clusters) >= 1
        if clusters:
            assert clusters[0].size >= 2
    
    def test_root_cause_guessing(self):
        """Test root cause analysis."""
        clusterer = RootCauseClusterer()
        
        # Test duplicate function root cause
        dup_issues = [
            Issue(kind="duplicate_function", message="Duplicate", file="test_utils.py", line=1, severity=2),
            Issue(kind="duplicate_function", message="Duplicate", file="test_helpers.py", line=1, severity=2)
        ]
        root_cause = clusterer._guess_root_cause(dup_issues)
        assert "test" in root_cause.lower() or "duplicate" in root_cause.lower()
        
        # Test circular import root cause
        circ_issues = [
            Issue(kind="circular_import", message="Circular", file="a.py", line=1, severity=3),
            Issue(kind="circular_import", message="Circular", file="b.py", line=1, severity=3)
        ]
        root_cause = clusterer._guess_root_cause(circ_issues)
        assert "coupling" in root_cause.lower() or "circular" in root_cause.lower()
    
    def test_confidence_calculation(self):
        """Test confidence score calculation."""
        clusterer = RootCauseClusterer()
        
        # High confidence: same type, severity, and location
        high_conf_issues = [
            Issue(kind="duplicate_function", message="Dup", severity=2, file="module/a.py", line=1),
            Issue(kind="duplicate_function", message="Dup", severity=2, file="module/b.py", line=1)
        ]
        confidence = clusterer._calculate_confidence(high_conf_issues)
        assert confidence > 0.7
        
        # Lower confidence: different types and severities
        low_conf_issues = [
            Issue(kind="duplicate_function", message="Dup", severity=2, file="a.py", line=1),
            Issue(kind="phantom_function", message="Phantom", severity=1, file="b/c.py", line=1)
        ]
        confidence = clusterer._calculate_confidence(low_conf_issues)
        assert confidence < 0.7
    
    def test_report_generation(self):
        """Test cluster report generation."""
        issues = self.create_test_issues()
        clusterer = RootCauseClusterer()
        clusters = clusterer.cluster(issues)
        
        report = clusterer.generate_report(clusters)
        
        assert "ROOT CAUSE ANALYSIS REPORT" in report
        assert "Total Clusters:" in report
        assert "Total Issues:" in report


class TestFixPlaybook:
    """Test fix playbook functionality."""
    
    def test_get_playbook(self):
        """Test retrieving playbooks by ID."""
        playbook = FixPlaybook.get_playbook("DEDUP_001")
        assert playbook is not None
        assert playbook['name'] == "Deduplicate Implementation"
        assert 'steps' in playbook
        assert 'risk' in playbook
    
    def test_suggest_playbook(self):
        """Test playbook suggestion based on cluster."""
        # Create a mock cluster with duplicate functions
        issues = [
            Issue(kind="duplicate_function", message="Dup", file="a.py", line=1, severity=2),
            Issue(kind="duplicate_function", message="Dup", file="b.py", line=1, severity=2)
        ]
        
        cluster = IssueCluster(
            cluster_id="TEST_001",
            representative_issue=issues[0],
            members=issues,
            root_cause_guess="Duplicate implementations",
            fix_playbook_id="",
            confidence=0.8,
            ast_signature="test",
            locations=[("a.py", 1), ("b.py", 1)]
        )
        
        suggested = FixPlaybook.suggest_playbook(cluster)
        assert suggested in ["duplicate_implementation", "shadow_module"]
    
    def test_all_playbooks_valid(self):
        """Test that all playbooks have required fields."""
        for playbook in FixPlaybook.PLAYBOOKS.values():
            assert 'id' in playbook
            assert 'name' in playbook
            assert 'description' in playbook
            assert 'steps' in playbook
            assert 'risk' in playbook
            assert isinstance(playbook['steps'], list)
            assert len(playbook['steps']) > 0


class TestIssueCluster:
    """Test IssueCluster functionality."""
    
    def test_cluster_properties(self):
        """Test cluster property calculations."""
        issues = [
            Issue(kind="test", message="Test", severity=2, file="a.py", line=1),
            Issue(kind="test", message="Test", severity=3, file="b.py", line=1),
            Issue(kind="test", message="Test", severity=1, file="c.py", line=1)
        ]
        
        cluster = IssueCluster(
            cluster_id="TEST",
            representative_issue=issues[1],
            members=issues,
            root_cause_guess="Test",
            fix_playbook_id="TEST",
            confidence=0.9,
            ast_signature="sig",
            locations=[("a.py", 1), ("b.py", 1), ("c.py", 1)]
        )
        
        assert cluster.size == 3
        assert cluster.severity == 3  # Max severity
    
    def test_cluster_to_dict(self):
        """Test cluster serialization."""
        issue = Issue(kind="test", severity=2, file="test.py", line=10, message="Test issue")
        cluster = IssueCluster(
            cluster_id="TEST_001",
            representative_issue=issue,
            members=[issue],
            root_cause_guess="Test root cause",
            fix_playbook_id="TEST_PB",
            confidence=0.85,
            ast_signature="abc123",
            locations=[("test.py", 10)]
        )
        
        data = cluster.to_dict()
        
        assert data['cluster_id'] == "TEST_001"
        assert data['size'] == 1
        assert data['severity'] == 2
        assert data['root_cause'] == "Test root cause"
        assert data['fix_playbook'] == "TEST_PB"
        assert data['confidence'] == 0.85
        assert data['locations'] == [("test.py", 10)]
        assert 'representative' in data