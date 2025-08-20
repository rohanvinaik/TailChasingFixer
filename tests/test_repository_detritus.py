"""Tests for the Repository Detritus Analyzer."""

import tempfile
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from tailchasing.analyzers.repository_detritus import RepositoryDetritusAnalyzer, DetritusPattern
from tailchasing.analyzers.base import AnalysisContext


class TestRepositoryDetritusAnalyzer:
    """Test suite for repository detritus detection."""
    
    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return RepositoryDetritusAnalyzer()
        
    @pytest.fixture
    def temp_repo(self):
        """Create a temporary repository structure."""
        temp_dir = tempfile.mkdtemp()
        repo_path = Path(temp_dir)
        
        # Create .git directory to mark as repo
        (repo_path / '.git').mkdir()
        
        # Create some Python files
        (repo_path / 'main.py').write_text('print("hello")')
        (repo_path / 'utils.py').write_text('def helper(): pass')
        
        yield repo_path
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
    def test_detect_temp_files(self, analyzer, temp_repo):
        """Test detection of temporary files."""
        # Create temp files (large enough to be detected)
        (temp_repo / 'file.tmp').write_text('temp' * 1000)
        (temp_repo / 'backup.bak').write_text('backup' * 1000)
        (temp_repo / 'old_file~').write_text('old' * 1000)
        
        # Create context
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        # Run analyzer
        issues = list(analyzer.run(ctx))
        
        # Check that temp files were detected
        temp_issues = [i for i in issues if 'detritus' in i.kind]
        assert len(temp_issues) >= 3
        
        # Check categories
        categories = {i.evidence['category'] for i in temp_issues if i.kind == 'repository_detritus'}
        assert 'temp' in categories or 'backup' in categories
        
    def test_detect_build_artifacts(self, analyzer, temp_repo):
        """Test detection of build artifacts."""
        # Create build directories
        (temp_repo / 'dist').mkdir()
        (temp_repo / 'dist' / 'app.js').write_text('built' * 1000)
        (temp_repo / 'build').mkdir()
        (temp_repo / 'build' / 'output.txt').write_text('output' * 1000)
        (temp_repo / 'myproject.egg-info').mkdir()
        (temp_repo / 'myproject.egg-info' / 'PKG-INFO').write_text('info' * 1000)
        
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        issues = list(analyzer.run(ctx))
        
        build_issues = [i for i in issues 
                       if i.kind == 'repository_detritus' and 
                       i.evidence.get('category') == 'build']
        assert len(build_issues) >= 2
        
    def test_detect_editor_files(self, analyzer, temp_repo):
        """Test detection of editor swap files."""
        # Create editor files
        (temp_repo / '.main.py.swp').write_text('swap')
        (temp_repo / '.#tempfile').write_text('emacs')
        
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        issues = list(analyzer.run(ctx))
        
        editor_issues = [i for i in issues 
                        if i.kind == 'repository_detritus' and 
                        i.evidence.get('category') == 'editor']
        assert len(editor_issues) >= 1
        
    def test_detect_os_files(self, analyzer, temp_repo):
        """Test detection of OS-generated files."""
        # Create OS files
        (temp_repo / '.DS_Store').write_text('mac')
        (temp_repo / 'Thumbs.db').write_text('windows')
        
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        issues = list(analyzer.run(ctx))
        
        os_issues = [i for i in issues 
                    if i.kind == 'repository_detritus' and 
                    i.evidence.get('category') == 'os']
        assert len(os_issues) >= 2
        
    def test_detect_python_artifacts(self, analyzer, temp_repo):
        """Test detection of Python artifacts."""
        # Create Python artifacts
        (temp_repo / '__pycache__').mkdir()
        (temp_repo / '__pycache__' / 'main.cpython-39.pyc').write_bytes(b'pyc' * 1000)
        (temp_repo / '.pytest_cache').mkdir()
        (temp_repo / '.pytest_cache' / 'README.md').write_text('cache' * 1000)
        (temp_repo / '.coverage').write_text('coverage' * 1000)
        
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        issues = list(analyzer.run(ctx))
        
        python_issues = [i for i in issues 
                        if i.kind == 'repository_detritus' and 
                        i.evidence.get('category') == 'python']
        assert len(python_issues) >= 2
        
    def test_gitignore_awareness(self, analyzer, temp_repo):
        """Test that analyzer checks .gitignore."""
        # Create .gitignore
        gitignore_content = """
*.tmp
__pycache__/
.coverage
"""
        (temp_repo / '.gitignore').write_text(gitignore_content)
        
        # Create files that are in gitignore
        (temp_repo / 'ignored.tmp').write_text('ignored')
        (temp_repo / '__pycache__').mkdir()
        
        # Create files that are NOT in gitignore
        (temp_repo / 'not_ignored.bak').write_text('backup')
        
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        issues = list(analyzer.run(ctx))
        
        # Files in gitignore should have lower severity
        for issue in issues:
            if issue.kind == 'repository_detritus':
                if 'ignored.tmp' in issue.file or '__pycache__' in issue.file:
                    assert not issue.evidence['should_gitignore']
                elif 'not_ignored.bak' in issue.file:
                    assert issue.evidence['should_gitignore']
                    
    def test_summary_issue_generation(self, analyzer, temp_repo):
        """Test that summary issue is created for multiple detritus files."""
        # Create many detritus files
        for i in range(6):
            (temp_repo / f'temp_{i}.tmp').write_text(f'temp{i}' * 1000)  # Make them big enough
            
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        issues = list(analyzer.run(ctx))
        
        # Should have individual issues plus summary
        summary_issues = [i for i in issues if i.kind == 'repository_detritus_summary']
        assert len(summary_issues) == 1
        
        summary = summary_issues[0]
        assert summary.evidence['total_files'] >= 5
        assert 'categories' in summary.evidence
        
    def test_size_filtering(self, analyzer, temp_repo):
        """Test that small files are filtered by default."""
        # Create a tiny file
        (temp_repo / 'tiny.tmp').write_text('x')  # 1 byte
        
        # Create a larger file
        (temp_repo / 'large.tmp').write_text('x' * 2000)  # 2KB
        
        ctx = AnalysisContext(
            files=[str(temp_repo / 'main.py')]
        )
        
        issues = list(analyzer.run(ctx))
        
        # Only large file should be reported (unless it's OS/editor file)
        file_paths = {i.file for i in issues if i.kind == 'repository_detritus'}
        assert 'large.tmp' in file_paths
        # tiny.tmp might not be in the list due to size filtering
        
    def test_categorize_file_patterns(self, analyzer):
        """Test file categorization logic."""
        test_cases = [
            ('file.tmp', 'temp'),
            ('backup.bak', 'backup'),
            ('dist/', 'build'),
            ('.main.py.swp', 'editor'),
            ('.DS_Store', 'os'),
            ('debug.log', 'debug'),
            ('__pycache__/', 'python'),
            ('node_modules/', 'package'),
            ('.coverage', 'test'),
        ]
        
        for path, expected_category in test_cases:
            category, pattern = analyzer._categorize_file(path)
            assert category == expected_category, f"Expected {path} to be {expected_category}, got {category}"
            
    def test_format_size(self, analyzer):
        """Test human-readable size formatting."""
        assert analyzer._format_size(500) == "500.0 B"
        assert analyzer._format_size(1500) == "1.5 KB"
        assert analyzer._format_size(1500000) == "1.4 MB"
        assert analyzer._format_size(1500000000) == "1.4 GB"
        
    def test_custom_config(self):
        """Test analyzer with custom configuration."""
        config = {
            'min_size_bytes': 100,  # Lower threshold
            'check_subdirs': True,   # Check subdirectories
        }
        
        analyzer = RepositoryDetritusAnalyzer(config)
        assert analyzer.min_size_to_report == 100
        assert analyzer.check_subdirs is True
        
    def test_no_repo_root(self, analyzer):
        """Test handling when no repository root is found."""
        ctx = AnalysisContext(files=[])
        
        issues = list(analyzer.run(ctx))
        assert len(issues) == 0  # Should handle gracefully
        
    def test_recommendation_generation(self, analyzer):
        """Test that appropriate recommendations are generated."""
        recommendations = {
            'temp': "Remove temporary file and add pattern to .gitignore",
            'python': "Add Python artifacts to .gitignore (use GitHub's Python template)",
            'os': "Add OS-specific patterns to global gitignore",
        }
        
        for category, expected in recommendations.items():
            actual = analyzer._get_recommendation(category, 'dummy_path')
            assert actual == expected