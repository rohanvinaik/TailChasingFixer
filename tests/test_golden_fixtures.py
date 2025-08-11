"""
Test suite for golden fixtures - known tail-chasing scenarios.

Tests the detection and fix suggestions for our fixture scenarios.
"""

import pytest
import ast
from pathlib import Path
from typing import List

from tailchasing.core.loader import parse_files
from tailchasing.core.symbols import SymbolTable
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.plugins import load_analyzers
from tailchasing.core.issues import Issue


@pytest.fixture
def fixture_dir():
    """Get the fixtures directory."""
    return Path(__file__).parent / "fixtures" / "tail_chase_scenarios"


@pytest.fixture
def analyzer_context(fixture_dir):
    """Create analysis context for fixtures."""
    def _create_context(filename):
        file_path = fixture_dir / filename
        files = [file_path]
        ast_index = parse_files(files)
        
        symbol_table = SymbolTable()
        for filepath, tree in ast_index.items():
            with open(filepath, 'r') as f:
                source = f.read()
            symbol_table.ingest_file(str(filepath), tree, source)
        
        config = {
            "ignore_issue_types": [],
            "scoring_weights": {
                "phantom_function": 3,
                "hallucination_cascade": 4,
                "import_anxiety": 2,
                "context_window_thrashing": 3,
                "semantic_duplicate_function": 3,
            }
        }
        
        return AnalysisContext(
            config=config,
            root_dir=fixture_dir,
            file_paths=files,
            ast_index=ast_index,
            symbol_table=symbol_table,
            source_cache={str(file_path): open(file_path).read().splitlines()},
            cache={}
        )
    
    return _create_context


class TestPhantomCascade:
    """Test detection of phantom cascade pattern."""
    
    @pytest.mark.fixture
    def test_detect_phantom_cascade(self, analyzer_context):
        """Test that phantom cascade is properly detected."""
        ctx = analyzer_context("phantom_cascade.py")
        analyzers = load_analyzers(ctx.config)
        
        all_issues = []
        for analyzer in analyzers:
            issues = list(analyzer.run(ctx))
            all_issues.extend(issues)
        
        # Should detect multiple phantom functions
        phantom_issues = [i for i in all_issues if "phantom" in i.kind]
        assert len(phantom_issues) >= 5, "Should detect multiple phantom functions"
        
        # Should detect the cascade pattern
        cascade_issues = [i for i in all_issues if "cascade" in i.kind.lower() or "hallucination" in i.kind.lower()]
        assert len(cascade_issues) >= 1, "Should detect hallucination cascade"
        
        # Check specific phantom classes are detected
        phantom_classes = ["OrderValidator", "ValidationRulesEngine", "RuleLoader", "RuleParser"]
        detected_phantoms = [i.symbol for i in phantom_issues if i.symbol]
        
        for phantom_class in phantom_classes:
            assert any(phantom_class in str(p) for p in detected_phantoms), \
                f"Should detect {phantom_class} as phantom"
    
    def test_phantom_cascade_suggestions(self, analyzer_context):
        """Test that appropriate fix suggestions are generated."""
        ctx = analyzer_context("phantom_cascade.py")
        analyzers = load_analyzers(ctx.config)
        
        all_issues = []
        for analyzer in analyzers:
            issues = list(analyzer.run(ctx))
            all_issues.extend(issues)
        
        phantom_issues = [i for i in all_issues if "phantom" in i.kind]
        
        # Check suggestions mention implementation or removal
        for issue in phantom_issues:
            suggestions_text = " ".join(issue.suggestions).lower()
            assert any(word in suggestions_text for word in ["implement", "remove", "actual", "real"]), \
                f"Suggestions should address phantom: {issue.suggestions}"
    
    def test_cascade_depth_detection(self, analyzer_context):
        """Test that the depth of the cascade is detected."""
        ctx = analyzer_context("phantom_cascade.py")
        
        # Count levels of phantom dependencies
        tree = list(ctx.ast_index.values())[0]
        
        phantom_classes = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check if it's a phantom (has pass/... in methods)
                is_phantom = False
                for item in node.body:
                    if isinstance(item, ast.FunctionDef):
                        if len(item.body) == 1:
                            stmt = item.body[0]
                            if isinstance(stmt, (ast.Pass, ast.Ellipsis)):
                                is_phantom = True
                            elif isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call):
                                if hasattr(stmt.exc.func, 'id') and stmt.exc.func.id == 'NotImplementedError':
                                    is_phantom = True
                
                if is_phantom:
                    phantom_classes[node.name] = node
        
        # The fixture has 3 levels of cascade
        assert len(phantom_classes) >= 8, "Should detect deep phantom cascade"


class TestImportAnxiety:
    """Test detection of import anxiety pattern."""
    
    @pytest.mark.fixture
    def test_detect_import_anxiety(self, analyzer_context):
        """Test that import anxiety is properly detected."""
        ctx = analyzer_context("import_anxiety.py")
        analyzers = load_analyzers(ctx.config)
        
        all_issues = []
        for analyzer in analyzers:
            issues = list(analyzer.run(ctx))
            all_issues.extend(issues)
        
        # Should detect import-related issues
        import_issues = [i for i in all_issues if "import" in i.kind.lower()]
        assert len(import_issues) >= 1, "Should detect import anxiety patterns"
        
        # Check for star imports
        star_import_issues = [i for i in all_issues if "*" in i.message or "star" in i.message.lower()]
        assert len(star_import_issues) >= 1, "Should detect star imports"
    
    def test_unused_import_detection(self, analyzer_context):
        """Test detection of unused imports."""
        ctx = analyzer_context("import_anxiety.py")
        
        # Parse the file to count imports vs usage
        tree = list(ctx.ast_index.values())[0]
        
        imported_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_modules.add(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imported_modules.add(node.module.split('.')[0])
        
        # The fixture imports 40+ modules but uses <5
        assert len(imported_modules) > 30, "Should have many imports"
        
        # Count actual usage (simplified check)
        used_modules = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                if node.id in imported_modules:
                    used_modules.add(node.id)
        
        unused_ratio = 1 - (len(used_modules) / len(imported_modules))
        assert unused_ratio > 0.7, "Most imports should be unused"
    
    def test_defensive_import_patterns(self, analyzer_context):
        """Test detection of defensive import patterns."""
        ctx = analyzer_context("import_anxiety.py")
        tree = list(ctx.ast_index.values())[0]
        
        # Count try/except import patterns
        defensive_imports = 0
        for node in ast.walk(tree):
            if isinstance(node, ast.Try):
                for handler in node.handlers:
                    if isinstance(handler.type, ast.Name) and handler.type.id == "ImportError":
                        defensive_imports += 1
        
        assert defensive_imports >= 4, "Should detect multiple defensive import patterns"


class TestContextThrashing:
    """Test detection of context window thrashing."""
    
    @pytest.mark.fixture  
    def test_detect_context_thrashing(self, analyzer_context):
        """Test that context thrashing is properly detected."""
        ctx = analyzer_context("context_thrash.py")
        analyzers = load_analyzers(ctx.config)
        
        all_issues = []
        for analyzer in analyzers:
            issues = list(analyzer.run(ctx))
            all_issues.extend(issues)
        
        # Should detect duplicate implementations
        duplicate_issues = [i for i in all_issues if "duplicate" in i.kind]
        assert len(duplicate_issues) >= 2, "Should detect duplicate functions"
        
        # Should detect semantic duplicates
        semantic_issues = [i for i in all_issues if "semantic" in i.kind]
        assert len(semantic_issues) >= 1, "Should detect semantic duplicates"
    
    def test_multiple_validator_detection(self, analyzer_context):
        """Test detection of multiple validator implementations."""
        ctx = analyzer_context("context_thrash.py")
        tree = list(ctx.ast_index.values())[0]
        
        # Count validator implementations
        validator_classes = []
        validator_functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                if "validat" in node.name.lower():
                    validator_classes.append(node.name)
            elif isinstance(node, ast.FunctionDef):
                if "validat" in node.name.lower() or "check" in node.name.lower():
                    validator_functions.append(node.name)
        
        # Should have multiple validator implementations
        assert len(validator_classes) >= 2, "Should have multiple validator classes"
        assert len(validator_functions) >= 3, "Should have multiple validation functions"
    
    def test_inconsistent_usage_detection(self, analyzer_context):
        """Test detection of inconsistent usage patterns."""
        ctx = analyzer_context("context_thrash.py")
        tree = list(ctx.ast_index.values())[0]
        
        # Find UserManager class
        user_manager = None
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "UserManager":
                user_manager = node
                break
        
        assert user_manager is not None, "Should find UserManager class"
        
        # Count different validators used
        validators_used = set()
        for node in ast.walk(user_manager):
            if isinstance(node, ast.Attribute):
                if hasattr(node.value, 'id') and "validat" in node.attr.lower():
                    validators_used.add(f"{node.value.id}.{node.attr}")
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and "validat" in node.func.id.lower():
                    validators_used.add(node.func.id)
        
        # Should use multiple different validators inconsistently
        assert len(validators_used) >= 2, "Should use multiple validators inconsistently"


@pytest.mark.parametrize("fixture_file,expected_patterns", [
    ("phantom_cascade.py", ["phantom_function", "hallucination_cascade"]),
    ("import_anxiety.py", ["import_anxiety", "unused_import"]),
    ("context_thrash.py", ["duplicate_function", "semantic_duplicate", "context_window_thrashing"]),
])
def test_fixture_detection(analyzer_context, fixture_file, expected_patterns):
    """Parametrized test for all fixtures."""
    ctx = analyzer_context(fixture_file)
    analyzers = load_analyzers(ctx.config)
    
    all_issues = []
    for analyzer in analyzers:
        issues = list(analyzer.run(ctx))
        all_issues.extend(issues)
    
    detected_patterns = {issue.kind for issue in all_issues}
    
    for expected in expected_patterns:
        assert any(expected in kind for kind in detected_patterns), \
            f"Should detect {expected} pattern in {fixture_file}"