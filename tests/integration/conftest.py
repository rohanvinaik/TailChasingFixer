"""
Shared fixtures and utilities for integration tests.

Provides common test fixtures, helper functions, and configuration
for the comprehensive testing framework.
"""

import ast
import tempfile
import shutil
import pytest
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass

# Import test components
from .synthetic_generator import (
    SyntheticCodebaseGenerator, 
    GenerationConfig, 
    create_small_test_codebase,
    create_medium_test_codebase
)

# Import TailChasingFixer components
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.symbols import SymbolTable
from tailchasing.core.issues import Issue, IssueSeverity
from tailchasing.catalytic.hv_encoder import HypervectorEncoder
from tailchasing.catalytic.catalytic_index import CatalyticIndex
from tailchasing.catalytic.similarity_pipeline import SimilarityPipeline


@dataclass 
class TestCodebase:
    """Container for test codebase information."""
    path: Path
    files: List[Path]
    ast_index: Dict[str, ast.AST]
    symbol_table: SymbolTable
    source_cache: Dict[str, List[str]]
    analysis_context: AnalysisContext
    generation_summary: Optional[Dict[str, Any]] = None


@pytest.fixture
def temp_dir():
    """Provide a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path)


@pytest.fixture
def small_synthetic_codebase(temp_dir):
    """Provide a small synthetic codebase for testing."""
    config = GenerationConfig(
        num_files=3,
        functions_per_file=5,
        classes_per_file=2,
        duplicate_function_rate=0.4,
        phantom_function_rate=0.3,
        context_thrashing_rate=0.2,
        hallucination_cascade_rate=0.1
    )
    
    generator = SyntheticCodebaseGenerator(config)
    codebase_path = generator.generate_codebase(temp_dir / "small_synthetic")
    
    return create_test_codebase_container(codebase_path)


@pytest.fixture
def medium_synthetic_codebase(temp_dir):
    """Provide a medium synthetic codebase for performance testing."""
    config = GenerationConfig(
        num_files=10,
        functions_per_file=15,
        classes_per_file=3,
        duplicate_function_rate=0.2,
        phantom_function_rate=0.25,
        context_thrashing_rate=0.15,
        hallucination_cascade_rate=0.05
    )
    
    generator = SyntheticCodebaseGenerator(config)
    codebase_path = generator.generate_codebase(temp_dir / "medium_synthetic")
    
    return create_test_codebase_container(codebase_path)


@pytest.fixture
def fixture_codebase():
    """Provide access to the curated test fixtures."""
    fixtures_path = Path(__file__).parent / "fixtures"
    return create_test_codebase_container(fixtures_path)


@pytest.fixture
def hypervector_encoder():
    """Provide a configured hypervector encoder."""
    return HypervectorEncoder()


@pytest.fixture
def catalytic_index(temp_dir):
    """Provide a temporary catalytic index."""
    index_dir = temp_dir / "catalytic_index"
    with CatalyticIndex(str(index_dir), mode='w') as index:
        yield index


@pytest.fixture
def similarity_pipeline(catalytic_index):
    """Provide a configured similarity pipeline."""
    return SimilarityPipeline(catalytic_index)


def create_test_codebase_container(codebase_path: Path) -> TestCodebase:
    """Create a TestCodebase container from a codebase directory."""
    codebase_path = Path(codebase_path)
    
    # Find all Python files
    python_files = list(codebase_path.rglob("*.py"))
    
    # Parse files and build indices
    ast_index = {}
    source_cache = {}
    
    for file_path in python_files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                source_cache[str(file_path)] = content.split('\n')
                
                # Parse AST
                tree = ast.parse(content, filename=str(file_path))
                ast_index[str(file_path)] = tree
                
        except (SyntaxError, UnicodeDecodeError) as e:
            print(f"Warning: Could not parse {file_path}: {e}")
            continue
    
    # Build symbol table
    symbol_table = SymbolTable()
    for file_path, tree in ast_index.items():
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                symbol_table.add_function(
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    node=node
                )
            elif isinstance(node, ast.ClassDef):
                symbol_table.add_class(
                    name=node.name,
                    file_path=file_path,
                    line_number=node.lineno,
                    node=node
                )
    
    # Create analysis context
    analysis_context = AnalysisContext(
        config={},
        root_dir=codebase_path,
        file_paths=python_files,
        ast_index=ast_index,
        symbol_table=symbol_table,
        source_cache=source_cache,
        cache={}
    )
    
    # Load generation summary if available
    generation_summary = None
    summary_file = codebase_path / "generation_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            generation_summary = json.load(f)
    
    return TestCodebase(
        path=codebase_path,
        files=python_files,
        ast_index=ast_index,
        symbol_table=symbol_table,
        source_cache=source_cache,
        analysis_context=analysis_context,
        generation_summary=generation_summary
    )


def create_mock_issues(count: int = 5, issue_types: Optional[List[str]] = None) -> List[Issue]:
    """Create mock issues for testing auto-fix functionality."""
    if issue_types is None:
        issue_types = [
            "duplicate_function",
            "phantom_function",
            "import_anxiety",
            "context_window_thrashing",
            "hallucination_cascade"
        ]
    
    issues = []
    for i in range(count):
        issue_type = issue_types[i % len(issue_types)]
        
        issue = Issue(
            kind=issue_type,
            message=f"Test {issue_type.replace('_', ' ')}: {i}",
            severity=IssueSeverity.WARNING.value,
            file=f"test_file_{i}.py",
            line=i * 10 + 1,
            symbol=f"test_function_{i}",
            evidence={
                "test_data": f"evidence_{i}",
                "pattern_type": issue_type
            }
        )
        issues.append(issue)
    
    return issues


def measure_execution_time(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time of a function."""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    execution_time = end_time - start_time
    return result, execution_time


def validate_python_syntax(file_path: Path) -> Tuple[bool, Optional[str]]:
    """Validate that a Python file has correct syntax."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        ast.parse(content, filename=str(file_path))
        return True, None
    except SyntaxError as e:
        return False, str(e)
    except Exception as e:
        return False, f"Unexpected error: {e}"


def count_pattern_occurrences(codebase: TestCodebase, pattern_indicators: Dict[str, List[str]]) -> Dict[str, int]:
    """Count occurrences of specific patterns in the codebase."""
    counts = {pattern: 0 for pattern in pattern_indicators.keys()}
    
    for file_path, tree in codebase.ast_index.items():
        source_lines = codebase.source_cache.get(file_path, [])
        
        # Count AST-based patterns
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Check for phantom patterns
                if (len(node.body) == 1 and isinstance(node.body[0], ast.Pass)):
                    counts["phantom_functions"] += 1
                elif any("TODO" in line for line in source_lines[max(0, node.lineno-1):node.end_lineno or node.lineno]):
                    counts["phantom_functions"] += 1
                elif any("NotImplementedError" in ast.dump(child) for child in ast.walk(node)):
                    counts["phantom_functions"] += 1
        
        # Count text-based patterns
        source_text = '\n'.join(source_lines)
        for pattern_name, indicators in pattern_indicators.items():
            for indicator in indicators:
                counts[pattern_name] += source_text.count(indicator)
    
    return counts


def create_test_file_with_content(temp_dir: Path, filename: str, content: str) -> Path:
    """Create a test file with specific content."""
    file_path = temp_dir / filename
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(content)
    return file_path


def get_functions_from_ast(tree: ast.AST) -> List[ast.FunctionDef]:
    """Extract all function definitions from an AST."""
    functions = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.append(node)
    return functions


def get_classes_from_ast(tree: ast.AST) -> List[ast.ClassDef]:
    """Extract all class definitions from an AST."""
    classes = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            classes.append(node)
    return classes


def compare_ast_similarity(node1: ast.AST, node2: ast.AST) -> float:
    """Calculate similarity between two AST nodes."""
    dump1 = ast.dump(node1, annotate_fields=False, include_attributes=False)
    dump2 = ast.dump(node2, annotate_fields=False, include_attributes=False)
    
    # Simple similarity calculation based on common characters
    common_chars = sum(1 for a, b in zip(dump1, dump2) if a == b)
    max_length = max(len(dump1), len(dump2))
    
    return common_chars / max_length if max_length > 0 else 0.0


class TestResultCollector:
    """Collect and analyze test results."""
    
    def __init__(self):
        self.results = []
        self.start_time = time.time()
    
    def add_result(self, test_name: str, passed: bool, execution_time: float, 
                   details: Optional[Dict[str, Any]] = None):
        """Add a test result."""
        result = {
            'test_name': test_name,
            'passed': passed,
            'execution_time': execution_time,
            'timestamp': time.time() - self.start_time,
            'details': details or {}
        }
        self.results.append(result)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of test results."""
        if not self.results:
            return {'total': 0, 'passed': 0, 'failed': 0, 'pass_rate': 0.0}
        
        total = len(self.results)
        passed = sum(1 for r in self.results if r['passed'])
        failed = total - passed
        total_time = sum(r['execution_time'] for r in self.results)
        
        return {
            'total': total,
            'passed': passed,
            'failed': failed,
            'pass_rate': passed / total,
            'total_execution_time': total_time,
            'average_execution_time': total_time / total
        }
    
    def get_failed_tests(self) -> List[Dict[str, Any]]:
        """Get list of failed tests."""
        return [r for r in self.results if not r['passed']]
    
    def save_results(self, output_path: Path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump({
                'summary': self.get_summary(),
                'detailed_results': self.results
            }, f, indent=2)


@pytest.fixture
def test_result_collector():
    """Provide a test result collector."""
    return TestResultCollector()


# Pytest configuration and markers
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance tests"
    )
    config.addinivalue_line(
        "markers", "requires_catalytic: marks tests that require catalytic hypervector system"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark slow tests
        if "performance" in item.nodeid or "benchmark" in item.nodeid:
            item.add_marker(pytest.mark.slow)
        
        # Mark integration tests
        if "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        
        # Mark tests requiring catalytic system
        if any(keyword in item.nodeid.lower() for keyword in ["catalytic", "hypervector", "similarity"]):
            item.add_marker(pytest.mark.requires_catalytic)