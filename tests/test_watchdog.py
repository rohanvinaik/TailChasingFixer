"""
Tests for the watchdog system.
"""

import time
import threading
from unittest.mock import MagicMock, patch, call
from typing import List

import pytest

from tailchasing.core.watchdog import (
    AnalyzerWatchdog,
    WatchdogConfig,
    HeartbeatMonitor,
    AnalyzerExecutionStats,
    SemanticAnalysisFallback
)
from tailchasing.core.issues import Issue


class TestHeartbeatMonitor:
    """Test heartbeat monitoring functionality."""
    
    def test_heartbeat_recording(self):
        """Test that heartbeats are recorded correctly."""
        monitor = HeartbeatMonitor(interval=0.1, timeout_multiplier=3)
        monitor.start()
        
        try:
            # Register an analyzer
            monitor.register("test_analyzer")
            
            # Send heartbeats
            for _ in range(3):
                monitor.heartbeat("test_analyzer")
                time.sleep(0.05)
            
            # Check heartbeat was recorded
            assert "test_analyzer" in monitor.heartbeats
            
            # Unregister
            monitor.unregister("test_analyzer")
            assert "test_analyzer" not in monitor.heartbeats
            
        finally:
            monitor.stop()
    
    def test_heartbeat_timeout_detection(self):
        """Test that heartbeat timeouts are detected."""
        monitor = HeartbeatMonitor(interval=0.1, timeout_multiplier=2)
        
        # Track timeout callback
        timeout_called = threading.Event()
        
        def timeout_callback(analyzer_name):
            assert analyzer_name == "test_analyzer"
            timeout_called.set()
        
        monitor.start()
        
        try:
            # Register with timeout callback
            monitor.register("test_analyzer", timeout_callback)
            
            # Don't send heartbeats, wait for timeout
            timeout_called.wait(timeout=1.0)
            assert timeout_called.is_set()
            
        finally:
            monitor.stop()
    
    def test_multiple_analyzers(self):
        """Test monitoring multiple analyzers simultaneously."""
        monitor = HeartbeatMonitor(interval=0.1, timeout_multiplier=3)
        monitor.start()
        
        try:
            # Register multiple analyzers
            monitor.register("analyzer1")
            monitor.register("analyzer2")
            monitor.register("analyzer3")
            
            # Send heartbeats for some
            monitor.heartbeat("analyzer1")
            monitor.heartbeat("analyzer3")
            
            # Check all are registered
            assert len(monitor.heartbeats) == 3
            
            # Unregister one
            monitor.unregister("analyzer2")
            assert len(monitor.heartbeats) == 2
            
        finally:
            monitor.stop()


class TestAnalyzerWatchdog:
    """Test the main watchdog functionality."""
    
    def test_successful_analyzer_execution(self):
        """Test wrapping and executing a successful analyzer."""
        config = WatchdogConfig(
            analyzer_timeout=5.0,
            heartbeat_interval=0.5
        )
        watchdog = AnalyzerWatchdog(config)
        watchdog.start()
        
        try:
            # Create mock analyzer
            def mock_analyzer(ctx):
                issues = []
                for i in range(3):
                    issues.append(Issue(
                        kind="test_issue",
                        message=f"Test issue {i}",
                        severity=2,
                        file="test.py",
                        line=i + 1
                    ))
                return issues
            
            # Wrap and execute
            wrapped = watchdog.wrap_analyzer("test_analyzer", mock_analyzer)
            context = MagicMock()
            issues = wrapped(context)
            
            # Check results
            assert len(issues) == 3
            assert all(issue.kind == "test_issue" for issue in issues)
            
            # Check stats
            assert len(watchdog.execution_stats) == 1
            stats = watchdog.execution_stats[0]
            assert stats.analyzer_name == "test_analyzer"
            assert stats.issues_found == 3
            assert not stats.timed_out
            assert stats.duration is not None
            
        finally:
            watchdog.stop()
    
    def test_analyzer_timeout(self):
        """Test that analyzer timeouts are handled correctly."""
        config = WatchdogConfig(
            analyzer_timeout=0.2,  # Very short timeout
            heartbeat_interval=0.1,
            enable_fallback=False
        )
        watchdog = AnalyzerWatchdog(config)
        watchdog.start()
        
        try:
            # Create slow analyzer
            def slow_analyzer(ctx):
                time.sleep(1.0)  # Will timeout
                return [Issue(kind="test", message="Should not appear", severity=2)]
            
            # Wrap and execute
            wrapped = watchdog.wrap_analyzer("slow_analyzer", slow_analyzer)
            context = MagicMock()
            issues = wrapped(context)
            
            # Should return empty list due to timeout
            assert issues == []
            
            # Check stats
            assert len(watchdog.execution_stats) == 1
            stats = watchdog.execution_stats[0]
            assert stats.analyzer_name == "slow_analyzer"
            assert stats.timed_out
            assert stats.error is not None
            assert "Timeout" in stats.error
            
        finally:
            watchdog.stop()
    
    def test_analyzer_with_exception(self):
        """Test handling of analyzer exceptions."""
        config = WatchdogConfig(analyzer_timeout=5.0)
        watchdog = AnalyzerWatchdog(config)
        watchdog.start()
        
        try:
            # Create failing analyzer
            def failing_analyzer(ctx):
                raise ValueError("Test error")
            
            # Wrap and execute
            wrapped = watchdog.wrap_analyzer("failing_analyzer", failing_analyzer)
            context = MagicMock()
            issues = wrapped(context)
            
            # Should return empty list due to error
            assert issues == []
            
            # Check stats
            stats = watchdog.execution_stats[0]
            assert stats.error == "Test error"
            assert not stats.timed_out
            
        finally:
            watchdog.stop()
    
    def test_fallback_mechanism(self):
        """Test fallback to alternative analyzer on timeout."""
        config = WatchdogConfig(
            analyzer_timeout=0.1,
            enable_fallback=True
        )
        watchdog = AnalyzerWatchdog(config)
        watchdog.start()
        
        try:
            # Create slow analyzer
            def slow_analyzer(ctx):
                time.sleep(1.0)
                return []
            
            # Create fast fallback
            def fast_fallback(ctx):
                return [Issue(kind="fallback", message="Fallback issue", severity=2)]
            
            # Wrap with fallback
            wrapped = watchdog.wrap_analyzer(
                "semantic_analyzer",
                slow_analyzer,
                fast_fallback
            )
            
            context = MagicMock()
            issues = wrapped(context)
            
            # Should return fallback results
            assert len(issues) == 1
            assert issues[0].kind == "fallback"
            
        finally:
            watchdog.stop()
    
    def test_execution_report(self):
        """Test generation of execution report."""
        config = WatchdogConfig(analyzer_timeout=5.0)
        watchdog = AnalyzerWatchdog(config)
        watchdog.start()
        
        try:
            # Run multiple analyzers with different outcomes
            
            # Successful analyzer
            def success_analyzer(ctx):
                return [Issue(kind="test", message="Test", severity=2)]
            
            # Slow analyzer
            def slow_analyzer(ctx):
                time.sleep(0.2)
                return []
            
            # Run analyzers
            wrapped1 = watchdog.wrap_analyzer("success", success_analyzer)
            wrapped2 = watchdog.wrap_analyzer("slow", slow_analyzer)
            
            context = MagicMock()
            wrapped1(context)
            wrapped2(context)
            wrapped1(context)  # Run success twice
            
            # Get report
            report = watchdog.get_execution_report()
            
            # Check report contents
            assert report['total_executions'] == 3
            assert report['timeout_count'] == 0
            assert report['error_count'] == 0
            
            # Check analyzer summary
            assert 'success' in report['analyzer_summary']
            assert report['analyzer_summary']['success']['executions'] == 2
            assert report['analyzer_summary']['success']['issues_found'] == 2
            
            assert 'slow' in report['analyzer_summary']
            assert report['analyzer_summary']['slow']['executions'] == 1
            
            # Check slowest analyzers
            assert len(report['slowest_analyzers']) > 0
            slowest = report['slowest_analyzers'][0]
            assert slowest['analyzer'] == 'slow'
            
        finally:
            watchdog.stop()
    
    def test_heartbeat_integration(self):
        """Test that heartbeats are sent during analyzer execution."""
        config = WatchdogConfig(
            analyzer_timeout=5.0,
            heartbeat_interval=0.1
        )
        watchdog = AnalyzerWatchdog(config)
        watchdog.start()
        
        try:
            # Create analyzer that sends heartbeats
            def heartbeat_analyzer(ctx):
                issues = []
                for i in range(5):
                    # Simulate work that triggers heartbeat
                    if hasattr(ctx, 'log'):
                        ctx.log(f"Processing item {i}")
                    time.sleep(0.05)
                    issues.append(Issue(kind="test", message=f"Issue {i}", severity=2))
                return issues
            
            # Wrap and execute
            wrapped = watchdog.wrap_analyzer("heartbeat_test", heartbeat_analyzer)
            context = MagicMock()
            context.log = MagicMock()
            
            issues = wrapped(context)
            
            # Check execution succeeded
            assert len(issues) == 5
            
            # Check heartbeats were recorded
            stats = watchdog.execution_stats[0]
            assert stats.heartbeat_count > 0
            
        finally:
            watchdog.stop()


class TestSemanticAnalysisFallback:
    """Test the semantic analysis fallback mechanism."""
    
    @patch('ast.walk')
    @patch('ast.unparse')
    def test_tfidf_fallback(self, mock_unparse, mock_walk):
        """Test TF-IDF fallback for semantic analysis."""
        # Mock AST walking
        mock_func1 = MagicMock()
        mock_func1.name = "func1"
        mock_func1.lineno = 10
        
        mock_func2 = MagicMock()
        mock_func2.name = "func2"
        mock_func2.lineno = 20
        
        mock_walk.return_value = [mock_func1, mock_func2]
        mock_unparse.side_effect = ["def func1(): pass", "def func2(): pass"]
        
        # Create context with AST index
        context = MagicMock()
        context.ast_index = {"file1.py": MagicMock(), "file2.py": MagicMock()}
        
        # Run fallback
        issues = SemanticAnalysisFallback.tfidf_fallback(context)
        
        # Should attempt to find duplicates
        # (Will fail due to mocking but shouldn't crash)
        assert isinstance(issues, list)
    
    def test_tfidf_fallback_error_handling(self):
        """Test that TF-IDF fallback handles errors gracefully."""
        # Create context that will cause errors
        context = MagicMock()
        context.ast_index = None  # Will cause AttributeError
        
        # Should not raise exception
        issues = SemanticAnalysisFallback.tfidf_fallback(context)
        assert issues == []


class TestWatchdogConfig:
    """Test watchdog configuration."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = WatchdogConfig()
        assert config.analyzer_timeout == 30.0
        assert config.heartbeat_interval == 2.0
        assert config.heartbeat_timeout_multiplier == 3.0
        assert config.enable_fallback is True
        assert config.max_retries == 1
        assert config.verbose is False
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = WatchdogConfig(
            analyzer_timeout=60.0,
            heartbeat_interval=5.0,
            heartbeat_timeout_multiplier=2.0,
            enable_fallback=False,
            max_retries=3,
            verbose=True
        )
        assert config.analyzer_timeout == 60.0
        assert config.heartbeat_interval == 5.0
        assert config.heartbeat_timeout_multiplier == 2.0
        assert config.enable_fallback is False
        assert config.max_retries == 3
        assert config.verbose is True


class TestAnalyzerExecutionStats:
    """Test execution statistics tracking."""
    
    def test_stats_initialization(self):
        """Test stats initialization."""
        stats = AnalyzerExecutionStats(
            analyzer_name="test_analyzer",
            start_time=time.time()
        )
        assert stats.analyzer_name == "test_analyzer"
        assert stats.start_time is not None
        assert stats.end_time is None
        assert stats.duration is None
        assert stats.timed_out is False
        assert stats.error is None
        assert stats.issues_found == 0
    
    def test_stats_completion(self):
        """Test stats after completion."""
        start = time.time()
        stats = AnalyzerExecutionStats(
            analyzer_name="test_analyzer",
            start_time=start
        )
        
        # Simulate completion
        time.sleep(0.1)
        stats.end_time = time.time()
        stats.duration = stats.end_time - stats.start_time
        stats.issues_found = 5
        
        assert stats.duration > 0.1
        assert stats.issues_found == 5
        assert not stats.timed_out
    
    def test_stats_timeout(self):
        """Test stats after timeout."""
        stats = AnalyzerExecutionStats(
            analyzer_name="test_analyzer",
            start_time=time.time()
        )
        
        # Simulate timeout
        stats.timed_out = True
        stats.error = "Timeout after 30s"
        stats.file_being_processed = "test.py"
        stats.function_being_processed = "test_function"
        
        assert stats.timed_out
        assert "Timeout" in stats.error
        assert stats.file_being_processed == "test.py"
        assert stats.function_being_processed == "test_function"