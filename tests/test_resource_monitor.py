"""
Tests for the resource monitor module.
"""

import gc
import time
import threading
from unittest.mock import Mock, patch, MagicMock
import pytest

from tailchasing.core.resource_monitor import (
    MemoryMonitor,
    MemoryStats,
    AdaptiveConfig,
    AdaptiveProcessor,
    create_memory_monitor,
    estimate_hypervector_memory,
    get_safe_hypervector_dimensions
)


class TestMemoryStats:
    """Test the MemoryStats dataclass."""
    
    def test_memory_stats_creation(self):
        """Test creating memory stats."""
        stats = MemoryStats(
            current_mb=500.0,
            peak_mb=600.0,
            limit_mb=1000.0,
            usage_percent=50.0,
            available_mb=500.0
        )
        
        assert stats.current_mb == 500.0
        assert stats.peak_mb == 600.0
        assert stats.usage_percent == 50.0
        
    def test_memory_stats_to_dict(self):
        """Test converting memory stats to dictionary."""
        stats = MemoryStats(
            current_mb=123.456,
            peak_mb=234.567,
            limit_mb=1000.0,
            usage_percent=12.345,
            available_mb=876.544,
            gc_collections=3,
            streaming_activations=1,
            dimension_reductions=2
        )
        
        result = stats.to_dict()
        
        assert result["current_mb"] == 123.46
        assert result["peak_mb"] == 234.57
        assert result["usage_percent"] == 12.3
        assert result["gc_collections"] == 3
        assert result["streaming_activations"] == 1
        assert result["dimension_reductions"] == 2


class TestAdaptiveConfig:
    """Test the AdaptiveConfig dataclass."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = AdaptiveConfig()
        
        assert config.gc_threshold_percent == 80.0
        assert config.streaming_threshold_percent == 90.0
        assert config.dimension_scale_threshold_percent == 75.0
        assert config.min_hypervector_dims == 128
        assert config.max_hypervector_dims == 4096
        assert config.default_hypervector_dims == 1024
        assert config.enable_monitoring is True
        
    def test_custom_config(self):
        """Test custom configuration values."""
        config = AdaptiveConfig(
            gc_threshold_percent=70.0,
            streaming_threshold_percent=85.0,
            enable_monitoring=False
        )
        
        assert config.gc_threshold_percent == 70.0
        assert config.streaming_threshold_percent == 85.0
        assert config.enable_monitoring is False


class TestMemoryMonitor:
    """Test the MemoryMonitor class."""
    
    def test_monitor_initialization(self):
        """Test initializing memory monitor."""
        monitor = MemoryMonitor(memory_limit_mb=1000, verbose=True)
        
        assert monitor.memory_limit_mb == 1000
        assert monitor.verbose is True
        assert monitor.streaming_mode_active is False
        assert monitor.current_hypervector_dims == 1024
        
        monitor.cleanup()
        
    def test_monitor_with_config(self):
        """Test monitor with custom configuration."""
        config = AdaptiveConfig(
            gc_threshold_percent=75.0,
            enable_monitoring=False
        )
        monitor = MemoryMonitor(memory_limit_mb=2000, config=config)
        
        assert monitor.config.gc_threshold_percent == 75.0
        assert monitor.config.enable_monitoring is False
        assert monitor.memory_limit_mb == 2000
        
        monitor.cleanup()
        
    @patch('tailchasing.core.resource_monitor.HAS_PSUTIL', False)
    def test_monitor_without_psutil(self):
        """Test monitor behavior without psutil."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        # Should use conservative fallback
        assert monitor.config.enable_monitoring is False
        
        monitor.cleanup()
        
    def test_get_stats(self):
        """Test getting memory statistics."""
        monitor = MemoryMonitor(memory_limit_mb=1000, verbose=False)
        
        stats = monitor.get_stats()
        
        assert isinstance(stats, MemoryStats)
        assert stats.limit_mb == 1000.0
        assert stats.usage_percent >= 0
        
        monitor.cleanup()
        
    def test_force_gc(self):
        """Test forcing garbage collection."""
        monitor = MemoryMonitor(memory_limit_mb=1000, verbose=True)
        
        # Create some garbage
        garbage = [[] for _ in range(1000)]
        del garbage
        
        collected = monitor.force_gc()
        
        assert isinstance(collected, int)
        assert monitor.stats.gc_collections > 0
        
        monitor.cleanup()
        
    def test_check_memory_available(self):
        """Test checking if memory is available."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        # Should have memory available for small allocation
        assert monitor.check_memory_available(100.0) is True
        
        # Should not have memory for huge allocation
        assert monitor.check_memory_available(10000.0) is False
        
        monitor.cleanup()
        
    def test_estimate_operation_memory(self):
        """Test estimating memory for operations."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        # Small file count
        estimate = monitor.estimate_operation_memory(10)
        assert estimate > 100  # Should include overhead
        
        # Large file count
        large_estimate = monitor.estimate_operation_memory(1000)
        assert large_estimate > estimate
        
        monitor.cleanup()
        
    def test_should_use_streaming(self):
        """Test streaming mode decision."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        # Small file count should not trigger streaming
        assert monitor.should_use_streaming(10) is False
        
        # Large file count might trigger streaming
        result = monitor.should_use_streaming(10000)
        assert isinstance(result, bool)
        
        monitor.cleanup()
        
    def test_get_optimal_batch_size(self):
        """Test optimal batch size calculation."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        batch_size = monitor.get_optimal_batch_size(100)
        assert batch_size >= 1
        assert batch_size <= 100
        
        # Single file
        single_batch = monitor.get_optimal_batch_size(1)
        assert single_batch == 1
        
        monitor.cleanup()
        
    def test_get_hypervector_dimensions(self):
        """Test getting hypervector dimensions."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        dims = monitor.get_hypervector_dimensions()
        assert dims == 1024  # Default
        
        monitor.cleanup()
        
    def test_callbacks(self):
        """Test callback functionality."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        gc_called = False
        streaming_called = False
        dimension_called = False
        
        def gc_callback():
            nonlocal gc_called
            gc_called = True
            
        def streaming_callback():
            nonlocal streaming_called
            streaming_called = True
            
        def dimension_callback(dims):
            nonlocal dimension_called
            dimension_called = True
            
        monitor.set_gc_callback(gc_callback)
        monitor.set_streaming_callback(streaming_callback)
        monitor.set_dimension_callback(dimension_callback)
        
        # Trigger callbacks manually
        monitor._trigger_garbage_collection()
        monitor._trigger_streaming_mode()
        monitor._scale_hypervector_dimensions(80.0)
        
        assert gc_called
        assert streaming_called
        assert dimension_called
        
        monitor.cleanup()
        
    def test_context_manager(self):
        """Test memory monitor as context manager."""
        with MemoryMonitor(memory_limit_mb=1000) as monitor:
            assert monitor.memory_limit_mb == 1000
            
        # Should be cleaned up automatically
        
    def test_adaptive_behavior_thresholds(self):
        """Test adaptive behavior trigger thresholds."""
        config = AdaptiveConfig(
            gc_threshold_percent=60.0,
            streaming_threshold_percent=70.0,
            dimension_scale_threshold_percent=50.0
        )
        monitor = MemoryMonitor(memory_limit_mb=1000, config=config)
        
        # Test threshold calculations
        monitor._trigger_adaptive_behaviors(65.0)  # Above GC threshold
        assert monitor.stats.gc_collections > 0
        
        monitor._trigger_adaptive_behaviors(75.0)  # Above streaming threshold
        assert monitor.streaming_mode_active is True
        
        monitor.cleanup()


class TestAdaptiveProcessor:
    """Test the AdaptiveProcessor class."""
    
    def test_processor_initialization(self):
        """Test initializing adaptive processor."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        processor = AdaptiveProcessor(monitor)
        
        assert processor.monitor == monitor
        
        monitor.cleanup()
        
    def test_process_files_empty(self):
        """Test processing empty file list."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        processor = AdaptiveProcessor(monitor)
        
        def process_func(files):
            return [f"processed_{f}" for f in files]
            
        results = processor.process_files([], process_func)
        assert results == []
        
        monitor.cleanup()
        
    def test_process_files_batch_mode(self):
        """Test processing files in batch mode."""
        monitor = MemoryMonitor(memory_limit_mb=10000)  # High limit
        processor = AdaptiveProcessor(monitor)
        
        files = [f"file{i}.py" for i in range(5)]
        processed_files = []
        
        def process_func(batch):
            processed_files.extend(batch)
            return [f"processed_{f}" for f in batch]
            
        results = processor.process_files(files, process_func, min_batch_size=2)
        
        assert len(results) == 5
        assert len(processed_files) == 5
        
        monitor.cleanup()
        
    def test_process_files_streaming_mode(self):
        """Test processing files in streaming mode."""
        monitor = MemoryMonitor(memory_limit_mb=100)  # Low limit to trigger streaming
        processor = AdaptiveProcessor(monitor)
        
        # Force streaming mode
        monitor.streaming_mode_active = True
        
        files = [f"file{i}.py" for i in range(3)]
        processed_files = []
        
        def process_func(batch):
            assert len(batch) == 1  # Streaming processes one at a time
            processed_files.extend(batch)
            return [f"processed_{f}" for f in batch]
            
        results = processor.process_files(files, process_func)
        
        assert len(results) == 3
        assert len(processed_files) == 3
        
        monitor.cleanup()
        
    def test_with_memory_limit(self):
        """Test executing operation with memory limit."""
        monitor = MemoryMonitor(memory_limit_mb=10000)  # High limit
        processor = AdaptiveProcessor(monitor)
        
        def test_operation():
            return "success"
            
        result = processor.with_memory_limit(test_operation, 100.0)
        assert result == "success"
        
        monitor.cleanup()
        
    def test_with_memory_limit_insufficient(self):
        """Test operation with insufficient memory."""
        monitor = MemoryMonitor(memory_limit_mb=100)  # Low limit
        processor = AdaptiveProcessor(monitor)
        
        def test_operation():
            return "success"
            
        # Should raise MemoryError for impossible allocation
        with pytest.raises(MemoryError):
            processor.with_memory_limit(test_operation, 10000.0)
        
        monitor.cleanup()


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_create_memory_monitor(self):
        """Test creating memory monitor with utility function."""
        monitor = create_memory_monitor(
            memory_limit_mb=2000,
            verbose=True,
            gc_threshold_percent=70.0
        )
        
        assert monitor.memory_limit_mb == 2000
        assert monitor.verbose is True
        assert monitor.config.gc_threshold_percent == 70.0
        
        monitor.cleanup()
        
    def test_estimate_hypervector_memory(self):
        """Test estimating hypervector memory usage."""
        # Small vectors
        small_estimate = estimate_hypervector_memory(128, 100)
        assert small_estimate > 0
        
        # Large vectors
        large_estimate = estimate_hypervector_memory(4096, 1000)
        assert large_estimate > small_estimate
        
    def test_get_safe_hypervector_dimensions(self):
        """Test getting safe hypervector dimensions."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        # Small vector count
        dims = get_safe_hypervector_dimensions(100, monitor)
        assert dims >= 128  # Minimum
        assert dims <= 4096  # Maximum
        
        # Large vector count (should reduce dimensions)
        large_dims = get_safe_hypervector_dimensions(100000, monitor)
        assert large_dims >= 128
        assert large_dims <= dims  # Should be smaller
        
        monitor.cleanup()


class TestMemoryMonitorIntegration:
    """Integration tests for memory monitoring."""
    
    def test_real_memory_monitoring(self):
        """Test real memory monitoring (if psutil available)."""
        monitor = MemoryMonitor(memory_limit_mb=1000, verbose=False)
        
        # Get initial stats
        initial_stats = monitor.get_stats()
        
        # Allocate some memory
        data = [i for i in range(10000)]
        
        # Check stats again
        after_stats = monitor.get_stats()
        
        # Memory usage should have increased (if psutil available)
        if hasattr(monitor, '_get_current_memory_mb'):
            # At minimum, current should be non-zero
            assert after_stats.current_mb >= 0
            
        del data
        monitor.cleanup()
        
    def test_background_monitoring(self):
        """Test background monitoring thread."""
        config = AdaptiveConfig(
            enable_monitoring=True,
            check_interval_seconds=0.1
        )
        monitor = MemoryMonitor(
            memory_limit_mb=1000,
            config=config,
            verbose=False
        )
        
        # Give monitoring thread time to run
        time.sleep(0.2)
        
        # Thread should be running
        if monitor.monitor_thread:
            assert monitor.monitor_thread.is_alive()
            
        monitor.cleanup()
        
        # Thread should stop
        if monitor.monitor_thread:
            # Give it time to stop
            time.sleep(0.1)
            assert not monitor.monitor_thread.is_alive()
            
    def test_memory_pressure_simulation(self):
        """Test memory pressure simulation."""
        config = AdaptiveConfig(
            gc_threshold_percent=30.0,
            streaming_threshold_percent=40.0,
            dimension_scale_threshold_percent=25.0,
            enable_monitoring=False  # Disable background monitoring for test
        )
        monitor = MemoryMonitor(memory_limit_mb=1000, config=config)
        
        # Manually trigger different pressure levels
        monitor._trigger_adaptive_behaviors(35.0)  # Above GC threshold
        assert monitor.stats.gc_collections > 0
        
        monitor._trigger_adaptive_behaviors(45.0)  # Above streaming threshold
        assert monitor.streaming_mode_active is True
        
        # Check dimension scaling
        original_dims = monitor.current_hypervector_dims
        monitor._trigger_adaptive_behaviors(80.0)  # High pressure
        assert monitor.current_hypervector_dims <= original_dims
        
        monitor.cleanup()
        
    def test_error_handling(self):
        """Test error handling in monitoring."""
        monitor = MemoryMonitor(memory_limit_mb=1000)
        
        # Test with invalid callback
        def bad_callback():
            raise Exception("Test error")
            
        monitor.set_gc_callback(bad_callback)
        
        # Should not crash when callback fails
        monitor._trigger_garbage_collection()
        
        monitor.cleanup()


class TestMemoryMonitorWithMocks:
    """Test memory monitor with mocked psutil."""
    
    @patch('tailchasing.core.resource_monitor.HAS_PSUTIL', True)
    @patch('tailchasing.core.resource_monitor.psutil')
    def test_memory_monitoring_with_mock(self, mock_psutil):
        """Test memory monitoring with mocked psutil."""
        # Mock system memory
        mock_virtual_memory = Mock()
        mock_virtual_memory.total = 8 * 1024 * 1024 * 1024  # 8GB
        mock_psutil.virtual_memory.return_value = mock_virtual_memory
        
        # Mock process memory
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 500 * 1024 * 1024  # 500MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process
        
        # Create monitor (should auto-detect system memory)
        monitor = MemoryMonitor(memory_limit_mb=None)
        
        # Should have set limit to 80% of system memory
        expected_limit = int((8 * 1024 * 0.8))
        assert monitor.memory_limit_mb == expected_limit
        
        # Test stats retrieval
        stats = monitor.get_stats()
        assert stats.current_mb == 500.0
        
        monitor.cleanup()
        
    @patch('tailchasing.core.resource_monitor.HAS_PSUTIL', True)
    @patch('tailchasing.core.resource_monitor.psutil')
    def test_memory_threshold_triggers(self, mock_psutil):
        """Test memory threshold triggers with mocked values."""
        mock_process = Mock()
        mock_memory_info = Mock()
        
        # Start with low memory usage
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process
        
        config = AdaptiveConfig(
            gc_threshold_percent=50.0,
            streaming_threshold_percent=60.0,
            enable_monitoring=False
        )
        monitor = MemoryMonitor(memory_limit_mb=1000, config=config)
        
        # Simulate increasing memory usage
        mock_memory_info.rss = 600 * 1024 * 1024  # 600MB (60%)
        
        # Check memory should trigger streaming
        monitor._check_memory()
        assert monitor.streaming_mode_active is True
        assert monitor.stats.gc_collections > 0
        
        monitor.cleanup()