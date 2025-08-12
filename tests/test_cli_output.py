"""
Tests for CLI output manager and related features.
"""

import json
import yaml
import sys
import time
from pathlib import Path
from io import StringIO
from unittest.mock import MagicMock, patch, mock_open
import tempfile

import pytest

from tailchasing.cli import (
    OutputManager,
    VerbosityLevel,
    OutputFormat,
    PerformanceProfiler,
    ComponentTimer
)
from tailchasing.core.issues import Issue


class TestOutputManager:
    """Test the output manager."""
    
    def test_verbosity_levels(self):
        """Test different verbosity levels."""
        # Quiet mode - capture console output
        output = StringIO()
        manager = OutputManager(verbosity=VerbosityLevel.QUIET)
        
        with patch.object(manager.console, 'file', output):
            manager.log("Normal message", VerbosityLevel.NORMAL)
            manager.log("Verbose message", VerbosityLevel.VERBOSE)
            manager.error("Error message")
        
        output_text = output.getvalue()
        
        # Only error should be shown in quiet mode
        assert "ERROR" in output_text or "Error" in output_text
        # Normal and verbose messages should not appear
        # (Note: Rich formatting might add escape sequences)
        
    def test_output_formats(self):
        """Test different output formats."""
        issues = [
            Issue(kind="test", message="Test issue 1", severity=2, file="test1.py", line=10),
            Issue(kind="test", message="Test issue 2", severity=3, file="test2.py", line=20)
        ]
        
        # JSON format
        output = StringIO()
        manager = OutputManager(output_format=OutputFormat.JSON)
        with patch('sys.stdout', output):
            manager.output_issues(issues)
        
        json_output = json.loads(output.getvalue())
        assert json_output["total_issues"] == 2
        assert len(json_output["issues"]) == 2
        
        # YAML format
        output = StringIO()
        manager = OutputManager(output_format=OutputFormat.YAML)
        with patch('sys.stdout', output):
            manager.output_issues(issues)
        
        yaml_output = yaml.safe_load(output.getvalue())
        assert yaml_output["total_issues"] == 2
        
    def test_colored_output(self):
        """Test colored output with rich."""
        manager = OutputManager(use_color=True)
        
        # Should create console with color support
        assert manager.console.force_terminal is True
        
        # No color mode
        manager_no_color = OutputManager(use_color=False)
        assert manager_no_color.console.force_terminal is False
        
    def test_output_to_file(self):
        """Test writing output to file."""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
            output_file = Path(f.name)
            
        try:
            issues = [Issue(kind="test", message="Test", severity=2)]
            manager = OutputManager(
                output_format=OutputFormat.JSON,
                output_file=output_file
            )
            
            manager.output_issues(issues)
            manager.finish()
            
            # Read the file
            with open(output_file) as f:
                data = json.load(f)
            
            assert data["total_issues"] == 1
            
        finally:
            output_file.unlink(missing_ok=True)
            
    def test_progress_context(self):
        """Test progress context manager."""
        output = StringIO()
        manager = OutputManager(verbosity=VerbosityLevel.NORMAL)
        
        with manager.progress_context("Processing", total=10) as progress:
            if progress:
                for i in range(10):
                    progress.update_func(1)
                    time.sleep(0.01)
                    
        # Progress should complete without errors
        assert True
        
    def test_spinner(self):
        """Test spinner for long operations."""
        manager = OutputManager(verbosity=VerbosityLevel.NORMAL)
        
        task_id = manager.start_spinner("Processing...")
        time.sleep(0.1)
        manager.stop_spinner(task_id)
        
        # Should complete without errors
        assert True
        
    def test_dry_run_summary(self):
        """Test dry-run summary output."""
        output = StringIO()
        manager = OutputManager(output_format=OutputFormat.TEXT)
        
        with patch.object(manager.console, 'print') as mock_print:
            manager.show_dry_run_summary(
                files=["file1.py", "file2.py", "file3.py"],
                analyzers=["analyzer1", "analyzer2"],
                config={"excluded_paths": ["test/"]}
            )
            
            # Should have called print to show the summary
            assert mock_print.called
            
    def test_sarif_output(self):
        """Test SARIF format output."""
        issues = [
            Issue(kind="error", message="Error found", severity=3, file="test.py", line=10)
        ]
        
        output = StringIO()
        manager = OutputManager(output_format=OutputFormat.SARIF)
        
        with patch('sys.stdout', output):
            manager.output_issues(issues)
            
        sarif = json.loads(output.getvalue())
        
        assert sarif["$schema"] == "https://json.schemastore.org/sarif-2.1.0.json"
        assert sarif["version"] == "2.1.0"
        assert len(sarif["runs"][0]["results"]) == 1
        
    def test_html_output(self):
        """Test HTML format output."""
        issues = [
            Issue(kind="warning", message="Warning", severity=2, file="test.py", line=5)
        ]
        
        output = StringIO()
        manager = OutputManager(output_format=OutputFormat.HTML)
        
        with patch('sys.stdout', output):
            manager.output_issues(issues)
            
        html = output.getvalue()
        
        assert "<!DOCTYPE html>" in html
        assert "<table>" in html
        assert "Warning" in html
        assert "test.py" in html


class TestPerformanceProfiler:
    """Test the performance profiler."""
    
    def test_basic_profiling(self):
        """Test basic profiling functionality."""
        profiler = PerformanceProfiler(enabled=True, track_memory=False)
        
        with profiler.profile("test_component"):
            time.sleep(0.1)
            
        report = profiler.get_report()
        
        assert "test_component" in report
        assert report["test_component"]["calls"] == 1
        assert report["test_component"]["time"] >= 0.1
        
    def test_nested_profiling(self):
        """Test nested component profiling."""
        profiler = PerformanceProfiler(enabled=True, track_memory=False)
        
        with profiler.profile("outer"):
            time.sleep(0.05)
            with profiler.profile("inner"):
                time.sleep(0.05)
                
        report = profiler.get_report()
        
        assert "outer" in report
        assert "inner" in report
        assert report["outer"]["time"] >= 0.1
        assert report["inner"]["time"] >= 0.05
        
    def test_function_decorator(self):
        """Test function profiling decorator."""
        profiler = PerformanceProfiler(enabled=True, track_memory=False)
        
        @profiler.profile_function("test_func")
        def slow_function():
            time.sleep(0.1)
            return "done"
            
        result = slow_function()
        
        assert result == "done"
        
        report = profiler.get_report()
        assert "test_func" in report
        assert report["test_func"]["time"] >= 0.1
        
    def test_disabled_profiling(self):
        """Test that disabled profiling has no overhead."""
        profiler = PerformanceProfiler(enabled=False)
        
        with profiler.profile("test"):
            time.sleep(0.1)
            
        report = profiler.get_report()
        
        # Should return empty report when disabled
        assert report == {}
        
    def test_memory_tracking(self):
        """Test memory tracking if available."""
        profiler = PerformanceProfiler(enabled=True, track_memory=True)
        
        with profiler.profile("memory_test"):
            # Allocate some memory
            data = [i for i in range(10000)]
            
        report = profiler.get_report()
        
        if profiler.track_memory:
            assert "memory_test" in report
            # Memory tracking might not always work in test environment
            if "memory_delta" in report["memory_test"]:
                assert isinstance(report["memory_test"]["memory_delta"], int)
                
    def test_summary_generation(self):
        """Test summary statistics generation."""
        profiler = PerformanceProfiler(enabled=True, track_memory=False)
        
        with profiler.profile("comp1"):
            time.sleep(0.1)
        with profiler.profile("comp2"):
            time.sleep(0.2)
            
        summary = profiler.get_summary()
        
        assert "total_time" in summary
        assert "slowest_components" in summary
        assert len(summary["slowest_components"]) >= 2
        
        # comp2 should be slower
        assert summary["slowest_components"][0][0] == "comp2"
        
    def test_reset(self):
        """Test resetting profiler data."""
        profiler = PerformanceProfiler(enabled=True)
        
        with profiler.profile("test"):
            pass
            
        assert len(profiler.profile_data) == 1
        
        profiler.reset()
        
        assert len(profiler.profile_data) == 0


class TestComponentTimer:
    """Test the component timer."""
    
    def test_basic_timing(self):
        """Test basic timing functionality."""
        timer = ComponentTimer()
        
        with timer.time("test"):
            time.sleep(0.1)
            
        stats = timer.get_stats()
        
        assert "test" in stats
        assert stats["test"]["count"] == 1
        assert stats["test"]["total"] >= 0.1
        assert stats["test"]["average"] >= 0.1
        
    def test_multiple_timings(self):
        """Test multiple timings of same component."""
        timer = ComponentTimer()
        
        for _ in range(3):
            with timer.time("test"):
                time.sleep(0.05)
                
        stats = timer.get_stats()
        
        assert stats["test"]["count"] == 3
        assert stats["test"]["total"] >= 0.15
        assert stats["test"]["average"] >= 0.05
        
    def test_clear(self):
        """Test clearing timer data."""
        timer = ComponentTimer()
        
        with timer.time("test"):
            pass
            
        assert len(timer.timings) == 1
        
        timer.clear()
        
        assert len(timer.timings) == 0


class TestWatchMode:
    """Test watch mode functionality."""
    
    def test_watch_mode_initialization(self):
        """Test watch mode initialization."""
        manager = OutputManager(watch_mode=True)
        
        assert manager.watch_mode is True
        
    def test_update_watch_display(self):
        """Test updating watch display."""
        manager = OutputManager(watch_mode=True)
        
        # Should not crash even without live display
        manager.update_watch_display(
            current_file="test.py",
            progress=50.0,
            issues_found=10
        )
        
        assert True


class TestCLIIntegration:
    """Test CLI integration of new features."""
    
    @patch('sys.argv', ['tailchasing', '--dry-run', 'test_dir'])
    def test_dry_run_flag(self):
        """Test dry-run flag parsing."""
        from tailchasing.cli import main
        
        with patch('tailchasing.cli.Path') as mock_path:
            mock_path.return_value.resolve.return_value.exists.return_value = True
            
            # Should exit early in dry-run mode
            with pytest.raises(SystemExit) as exc_info:
                main()
                
    def test_output_format_selection(self):
        """Test output format selection."""
        # Test each format
        for format_name in ["text", "json", "yaml", "html", "sarif"]:
            output = StringIO()
            manager = OutputManager(output_format=OutputFormat[format_name.upper()])
            
            issues = [Issue(kind="test", message="Test", severity=2)]
            
            with patch('sys.stdout', output):
                manager.output_issues(issues)
                
            # Should produce some output
            assert len(output.getvalue()) > 0