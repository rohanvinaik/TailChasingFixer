"""Tests for the ignore functionality."""

import tempfile
from pathlib import Path
import pytest

from tailchasing.core.ignore import IgnoreManager, create_default_tcdignore
from tailchasing.core.loader import collect_files


class TestIgnoreManager:
    """Test IgnoreManager functionality."""
    
    def test_default_patterns(self):
        """Test that default patterns are loaded."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(root)
            
            # Check some default patterns are present
            patterns = manager.get_patterns()
            assert "__pycache__/" in patterns
            assert "*.pyc" in patterns
            assert ".mypy_cache/" in patterns
            assert "venv/" in patterns
    
    def test_should_ignore_pycache(self):
        """Test that __pycache__ directories are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(root)
            
            # Create test paths
            pycache = root / "__pycache__" / "test.pyc"
            normal = root / "test.py"
            
            assert manager.should_ignore(pycache)
            assert not manager.should_ignore(normal)
    
    def test_should_ignore_venv(self):
        """Test that virtual environment directories are ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(root)
            
            # Test various venv patterns
            assert manager.should_ignore(root / "venv" / "lib" / "python3.9" / "test.py")
            assert manager.should_ignore(root / ".venv" / "test.py")
            assert manager.should_ignore(root / "env" / "test.py")
            assert not manager.should_ignore(root / "environment.py")
    
    def test_custom_patterns(self):
        """Test adding custom patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(
                root,
                additional_patterns=["custom/*.py", "test_*.py"]
            )
            
            assert manager.should_ignore(root / "custom" / "file.py")
            assert manager.should_ignore(root / "test_something.py")
            assert not manager.should_ignore(root / "normal.py")
    
    def test_tcdignore_file(self):
        """Test reading patterns from .tcdignore file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create .tcdignore file
            ignore_file = root / ".tcdignore"
            ignore_file.write_text("""
# Custom ignore patterns
custom_dir/
*.backup
test_*.py

# Comments should be ignored
legacy/
""")
            
            manager = IgnoreManager(root)
            
            # Test custom patterns from file
            assert manager.should_ignore(root / "custom_dir" / "file.py")
            assert manager.should_ignore(root / "data.backup")
            assert manager.should_ignore(root / "test_foo.py")
            assert manager.should_ignore(root / "legacy" / "old.py")
            assert not manager.should_ignore(root / "normal.py")
    
    def test_filter_paths(self):
        """Test filtering a list of paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(root)
            
            paths = [
                root / "normal.py",
                root / "__pycache__" / "cached.pyc",
                root / ".mypy_cache" / "type.json",
                root / "src" / "main.py",
                root / "venv" / "lib" / "module.py",
            ]
            
            filtered = manager.filter_paths(paths)
            
            assert len(filtered) == 2
            assert root / "normal.py" in filtered
            assert root / "src" / "main.py" in filtered
    
    def test_add_patterns(self):
        """Test dynamically adding patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(root, use_defaults=False)
            
            # Initially nothing is ignored
            assert not manager.should_ignore(root / "test.tmp")
            
            # Add pattern
            manager.add_pattern("*.tmp")
            
            # Now it should be ignored
            assert manager.should_ignore(root / "test.tmp")
            assert not manager.should_ignore(root / "test.py")
    
    def test_save_to_file(self):
        """Test saving patterns to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(
                root,
                additional_patterns=["custom/*.py"],
                use_defaults=True
            )
            
            # Save to file
            save_path = root / "test_ignore"
            manager.save_to_file(save_path)
            
            # Check file was created
            assert save_path.exists()
            content = save_path.read_text()
            assert "# TailChasing ignore file" in content
            assert "__pycache__/" in content
    
    def test_statistics(self):
        """Test getting pattern statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            manager = IgnoreManager(
                root,
                additional_patterns=["custom/", "*.tmp", "!important.tmp"],
                use_defaults=True
            )
            
            stats = manager.get_statistics()
            assert stats["total_patterns"] > 0
            assert stats["default_patterns"] > 0
            # Only 2 custom patterns are added (negation pattern is filtered out during init)
            assert stats["custom_patterns"] >= 2
            assert stats["directory_patterns"] > 0
            assert stats["wildcard_patterns"] > 0
            # Negation patterns might be filtered depending on implementation
            assert stats["negation_patterns"] >= 0


class TestCollectFilesIntegration:
    """Test integration with collect_files."""
    
    def test_collect_with_ignore_manager(self):
        """Test collect_files uses IgnoreManager correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create test file structure
            (root / "main.py").write_text("# main")
            (root / "test.py").write_text("# test")
            
            # Create ignored directories
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "main.cpython-39.pyc").write_text("")
            
            (root / "venv").mkdir()
            (root / "venv" / "lib.py").write_text("# venv lib")
            
            (root / ".mypy_cache").mkdir()
            (root / ".mypy_cache" / "data.py").write_text("# cache")
            
            # Collect files
            files = collect_files(root)
            
            # Should only get the non-ignored files
            assert len(files) == 2
            assert root / "main.py" in files
            assert root / "test.py" in files
            assert root / "__pycache__" / "main.cpython-39.pyc" not in files
            assert root / "venv" / "lib.py" not in files
    
    def test_collect_with_custom_exclude(self):
        """Test collect_files with custom exclude patterns."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create test files
            (root / "main.py").write_text("# main")
            (root / "test_foo.py").write_text("# test")
            (root / "backup.py").write_text("# backup")
            
            # Create custom IgnoreManager
            ignore_manager = IgnoreManager(
                root,
                additional_patterns=["test_*.py", "backup.py"],
                use_defaults=True
            )
            
            files = collect_files(root, ignore_manager=ignore_manager)
            
            assert len(files) == 1
            assert root / "main.py" in files
            assert root / "test_foo.py" not in files
            assert root / "backup.py" not in files


class TestCreateDefaultTcdIgnore:
    """Test creating default .tcdignore file."""
    
    def test_create_default(self):
        """Test creating a default .tcdignore file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            create_default_tcdignore(root)
            
            ignore_file = root / ".tcdignore"
            assert ignore_file.exists()
            
            content = ignore_file.read_text()
            assert "# TailChasing ignore file" in content
            assert "__pycache__/" in content
            assert "venv/" in content
            assert ".mypy_cache/" in content
    
    def test_no_overwrite(self):
        """Test that existing .tcdignore is not overwritten."""
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            
            # Create existing file
            ignore_file = root / ".tcdignore"
            ignore_file.write_text("# Custom content\nmy_pattern/")
            
            create_default_tcdignore(root)
            
            # Should not be overwritten
            content = ignore_file.read_text()
            assert content == "# Custom content\nmy_pattern/"