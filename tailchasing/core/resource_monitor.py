"""
Memory monitoring and adaptive processing to prevent OOM issues.
"""

import gc
import os
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from pathlib import Path
import logging

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

from ..utils.logging_setup import get_logger


logger = get_logger(__name__)


@dataclass
class MemoryStats:
    """Memory usage statistics."""
    current_mb: float
    peak_mb: float
    limit_mb: float
    usage_percent: float
    available_mb: float
    gc_collections: int = 0
    streaming_activations: int = 0
    dimension_reductions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "current_mb": round(self.current_mb, 2),
            "peak_mb": round(self.peak_mb, 2),
            "limit_mb": self.limit_mb,
            "usage_percent": round(self.usage_percent, 1),
            "available_mb": round(self.available_mb, 2),
            "gc_collections": self.gc_collections,
            "streaming_activations": self.streaming_activations,
            "dimension_reductions": self.dimension_reductions
        }


@dataclass
class AdaptiveConfig:
    """Configuration for adaptive processing."""
    gc_threshold_percent: float = 80.0
    streaming_threshold_percent: float = 90.0
    dimension_scale_threshold_percent: float = 75.0
    min_hypervector_dims: int = 128
    max_hypervector_dims: int = 4096
    default_hypervector_dims: int = 1024
    check_interval_seconds: float = 1.0
    enable_monitoring: bool = True
    enable_gc_trigger: bool = True
    enable_streaming_mode: bool = True
    enable_dimension_scaling: bool = True


class MemoryMonitor:
    """Monitors memory usage and triggers adaptive behaviors."""
    
    def __init__(
        self, 
        memory_limit_mb: Optional[int] = None,
        config: Optional[AdaptiveConfig] = None,
        verbose: bool = False
    ):
        """
        Initialize memory monitor.
        
        Args:
            memory_limit_mb: Hard memory limit in MB (None for auto-detect)
            config: Adaptive processing configuration
            verbose: Enable verbose logging
        """
        self.config = config or AdaptiveConfig()
        self.verbose = verbose
        self._shutdown = False
        self._lock = threading.Lock()
        
        # Detect system memory if not specified
        if memory_limit_mb is None:
            if HAS_PSUTIL:
                system_memory_mb = psutil.virtual_memory().total / (1024 * 1024)
                # Default to 80% of system memory
                self.memory_limit_mb = int(system_memory_mb * 0.8)
            else:
                # Conservative fallback
                self.memory_limit_mb = 4096
                logger.warning("psutil not available, using conservative 4GB memory limit")
        else:
            self.memory_limit_mb = memory_limit_mb
            
        if not HAS_PSUTIL:
            logger.warning("psutil not available - memory monitoring will be limited")
            self.config.enable_monitoring = False
            
        # Initialize statistics
        self.stats = MemoryStats(
            current_mb=0.0,
            peak_mb=0.0,
            limit_mb=float(self.memory_limit_mb),
            usage_percent=0.0,
            available_mb=float(self.memory_limit_mb)
        )
        
        # State tracking
        self.streaming_mode_active = False
        self.current_hypervector_dims = self.config.default_hypervector_dims
        
        # Callbacks for adaptive behaviors
        self.gc_callback: Optional[Callable] = None
        self.streaming_callback: Optional[Callable] = None
        self.dimension_callback: Optional[Callable[[int], None]] = None
        
        # Background monitoring thread
        self.monitor_thread: Optional[threading.Thread] = None
        if self.config.enable_monitoring:
            self._start_monitoring()
            
        logger.info(f"Memory monitor initialized with {self.memory_limit_mb}MB limit")
        
    def _start_monitoring(self):
        """Start background monitoring thread."""
        if self.monitor_thread and self.monitor_thread.is_alive():
            return
            
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
            name="MemoryMonitor"
        )
        self.monitor_thread.start()
        
    def _monitor_loop(self):
        """Background monitoring loop."""
        while not self._shutdown:
            try:
                self._check_memory()
                time.sleep(self.config.check_interval_seconds)
            except Exception as e:
                logger.error(f"Error in memory monitoring: {e}")
                time.sleep(self.config.check_interval_seconds * 2)
                
    def _check_memory(self):
        """Check current memory usage and trigger adaptive behaviors."""
        if not HAS_PSUTIL:
            return
            
        try:
            # Get current process memory
            process = psutil.Process(os.getpid())
            memory_info = process.memory_info()
            current_mb = memory_info.rss / (1024 * 1024)
            
            # Update statistics
            with self._lock:
                self.stats.current_mb = current_mb
                self.stats.peak_mb = max(self.stats.peak_mb, current_mb)
                self.stats.usage_percent = (current_mb / self.memory_limit_mb) * 100
                self.stats.available_mb = self.memory_limit_mb - current_mb
                
                if self.verbose:
                    logger.debug(f"Memory usage: {current_mb:.1f}MB ({self.stats.usage_percent:.1f}%)")
                    
            # Trigger adaptive behaviors based on thresholds
            self._trigger_adaptive_behaviors(self.stats.usage_percent)
            
        except Exception as e:
            logger.error(f"Failed to check memory usage: {e}")
            
    def _trigger_adaptive_behaviors(self, usage_percent: float):
        """Trigger adaptive behaviors based on memory usage."""
        
        # Trigger garbage collection at 80%
        if (usage_percent >= self.config.gc_threshold_percent and 
            self.config.enable_gc_trigger):
            self._trigger_garbage_collection()
            
        # Trigger streaming mode at 90%
        if (usage_percent >= self.config.streaming_threshold_percent and 
            self.config.enable_streaming_mode and
            not self.streaming_mode_active):
            self._trigger_streaming_mode()
            
        # Scale down hypervector dimensions at 75%
        if (usage_percent >= self.config.dimension_scale_threshold_percent and
            self.config.enable_dimension_scaling):
            self._scale_hypervector_dimensions(usage_percent)
            
    def _trigger_garbage_collection(self):
        """Trigger garbage collection to free memory."""
        if self.verbose:
            logger.info("Memory usage high, triggering garbage collection")
            
        before_mb = self.stats.current_mb
        
        # Force garbage collection
        collected = gc.collect()
        
        # Update stats
        with self._lock:
            self.stats.gc_collections += 1
            
        if self.gc_callback:
            try:
                self.gc_callback()
            except Exception as e:
                logger.error(f"Error in GC callback: {e}")
                
        # Log results
        after_mb = self._get_current_memory_mb()
        freed_mb = before_mb - after_mb
        if self.verbose:
            logger.info(f"GC freed {freed_mb:.1f}MB (collected {collected} objects)")
            
    def _trigger_streaming_mode(self):
        """Trigger streaming mode for memory-efficient processing."""
        if self.streaming_mode_active:
            return
            
        self.streaming_mode_active = True
        
        with self._lock:
            self.stats.streaming_activations += 1
            
        if self.verbose:
            logger.warning("Memory usage critical, activating streaming mode")
            
        if self.streaming_callback:
            try:
                self.streaming_callback()
            except Exception as e:
                logger.error(f"Error in streaming callback: {e}")
                
    def _scale_hypervector_dimensions(self, usage_percent: float):
        """Scale hypervector dimensions based on memory usage."""
        # Calculate target dimensions based on memory pressure
        pressure = (usage_percent - self.config.dimension_scale_threshold_percent) / 25.0
        pressure = max(0.0, min(1.0, pressure))  # Clamp to [0, 1]
        
        # Scale dimensions: high pressure = lower dimensions
        target_dims = int(
            self.config.max_hypervector_dims * (1.0 - pressure * 0.7)
        )
        target_dims = max(self.config.min_hypervector_dims, target_dims)
        
        # Only update if significantly different
        if abs(target_dims - self.current_hypervector_dims) > 64:
            old_dims = self.current_hypervector_dims
            self.current_hypervector_dims = target_dims
            
            with self._lock:
                self.stats.dimension_reductions += 1
                
            if self.verbose:
                logger.info(f"Scaled hypervector dimensions: {old_dims} → {target_dims}")
                
            if self.dimension_callback:
                try:
                    self.dimension_callback(target_dims)
                except Exception as e:
                    logger.error(f"Error in dimension callback: {e}")
                    
    def _get_current_memory_mb(self) -> float:
        """Get current memory usage in MB."""
        if not HAS_PSUTIL:
            return 0.0
            
        try:
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / (1024 * 1024)
        except:
            return 0.0
            
    def get_stats(self) -> MemoryStats:
        """Get current memory statistics."""
        # Update current memory before returning
        if HAS_PSUTIL:
            current_mb = self._get_current_memory_mb()
            with self._lock:
                self.stats.current_mb = current_mb
                self.stats.usage_percent = (current_mb / self.memory_limit_mb) * 100
                self.stats.available_mb = self.memory_limit_mb - current_mb
                
        return self.stats
        
    def set_gc_callback(self, callback: Callable):
        """Set callback to trigger on garbage collection."""
        self.gc_callback = callback
        
    def set_streaming_callback(self, callback: Callable):
        """Set callback to trigger on streaming mode activation."""
        self.streaming_callback = callback
        
    def set_dimension_callback(self, callback: Callable[[int], None]):
        """Set callback to trigger on dimension scaling."""
        self.dimension_callback = callback
        
    def force_gc(self) -> int:
        """Force garbage collection and return number of collected objects."""
        if self.verbose:
            logger.info("Forcing garbage collection")
            
        before_mb = self._get_current_memory_mb()
        collected = gc.collect()
        after_mb = self._get_current_memory_mb()
        
        with self._lock:
            self.stats.gc_collections += 1
            
        if self.verbose:
            logger.info(f"Forced GC freed {before_mb - after_mb:.1f}MB")
            
        return collected
        
    def check_memory_available(self, required_mb: float) -> bool:
        """Check if enough memory is available for an operation."""
        stats = self.get_stats()
        return stats.available_mb >= required_mb
        
    def estimate_operation_memory(self, file_count: int, avg_file_size_kb: float = 10) -> float:
        """Estimate memory required for processing given number of files."""
        # Rough estimation: file parsing + AST + analysis overhead
        base_overhead_mb = 100  # Base overhead
        per_file_mb = (avg_file_size_kb / 1024) * 3  # 3x file size for processing
        semantic_overhead_mb = file_count * 0.1  # Semantic analysis overhead
        
        return base_overhead_mb + (file_count * per_file_mb) + semantic_overhead_mb
        
    def should_use_streaming(self, file_count: int) -> bool:
        """Determine if streaming mode should be used based on file count and memory."""
        estimated_memory = self.estimate_operation_memory(file_count)
        current_stats = self.get_stats()
        
        # Use streaming if:
        # 1. Already in streaming mode
        # 2. Estimated memory exceeds available
        # 3. Current usage is high
        return (
            self.streaming_mode_active or
            estimated_memory > current_stats.available_mb or
            current_stats.usage_percent > self.config.streaming_threshold_percent
        )
        
    def get_optimal_batch_size(self, file_count: int, target_memory_mb: float = 500) -> int:
        """Get optimal batch size for processing files."""
        if file_count <= 1:
            return 1
            
        # Estimate memory per file
        estimated_total = self.estimate_operation_memory(file_count)
        memory_per_file = estimated_total / file_count
        
        # Calculate batch size to stay under target
        batch_size = max(1, int(target_memory_mb / memory_per_file))
        
        # Don't exceed available memory
        stats = self.get_stats()
        max_batch_by_available = max(1, int(stats.available_mb / memory_per_file))
        
        return min(batch_size, max_batch_by_available, file_count)
        
    def get_hypervector_dimensions(self) -> int:
        """Get current hypervector dimensions based on memory constraints."""
        return self.current_hypervector_dims
        
    def log_memory_summary(self):
        """Log comprehensive memory usage summary."""
        stats = self.get_stats()
        
        logger.info("Memory Usage Summary:")
        logger.info(f"  Current: {stats.current_mb:.1f}MB ({stats.usage_percent:.1f}%)")
        logger.info(f"  Peak: {stats.peak_mb:.1f}MB")
        logger.info(f"  Limit: {stats.limit_mb:.0f}MB")
        logger.info(f"  Available: {stats.available_mb:.1f}MB")
        logger.info(f"  GC Collections: {stats.gc_collections}")
        logger.info(f"  Streaming Activations: {stats.streaming_activations}")
        logger.info(f"  Dimension Reductions: {stats.dimension_reductions}")
        logger.info(f"  Current HV Dims: {self.current_hypervector_dims}")
        
        if HAS_PSUTIL:
            try:
                system_mem = psutil.virtual_memory()
                logger.info(f"  System Memory: {system_mem.percent:.1f}% used")
            except:
                pass
                
    def cleanup(self):
        """Clean up resources and stop monitoring."""
        self._shutdown = True
        
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)
            
        if self.verbose:
            self.log_memory_summary()
            
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


class AdaptiveProcessor:
    """Processes data adaptively based on memory constraints."""
    
    def __init__(self, memory_monitor: MemoryMonitor):
        """Initialize with memory monitor."""
        self.monitor = memory_monitor
        
    def process_files(
        self, 
        files: List[Path], 
        process_func: Callable[[List[Path]], Any],
        min_batch_size: int = 1
    ) -> List[Any]:
        """
        Process files adaptively based on memory constraints.
        
        Args:
            files: List of files to process
            process_func: Function to process batches of files
            min_batch_size: Minimum batch size
            
        Returns:
            List of processing results
        """
        if not files:
            return []
            
        results = []
        
        # Determine processing strategy
        if self.monitor.should_use_streaming(len(files)):
            # Streaming mode: process one file at a time
            if self.monitor.verbose:
                logger.info(f"Using streaming mode for {len(files)} files")
                
            for file_path in files:
                try:
                    # Force GC before each file in streaming mode
                    if self.monitor.stats.usage_percent > 85:
                        self.monitor.force_gc()
                        
                    result = process_func([file_path])
                    results.extend(result if isinstance(result, list) else [result])
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
                    continue
                    
        else:
            # Batch mode: process in optimal batches
            batch_size = max(
                min_batch_size,
                self.monitor.get_optimal_batch_size(len(files))
            )
            
            if self.monitor.verbose:
                logger.info(f"Using batch mode with batch size {batch_size}")
                
            for i in range(0, len(files), batch_size):
                batch = files[i:i + batch_size]
                
                try:
                    # Check memory before processing batch
                    estimated_mb = self.monitor.estimate_operation_memory(len(batch))
                    if not self.monitor.check_memory_available(estimated_mb):
                        logger.warning(f"Insufficient memory for batch of {len(batch)} files")
                        self.monitor.force_gc()
                        
                    result = process_func(batch)
                    results.extend(result if isinstance(result, list) else [result])
                    
                except Exception as e:
                    logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                    continue
                    
        return results
        
    def with_memory_limit(self, operation: Callable, memory_limit_mb: float):
        """Execute operation with memory limit checking."""
        if not self.monitor.check_memory_available(memory_limit_mb):
            logger.warning(f"Insufficient memory for operation (need {memory_limit_mb}MB)")
            self.monitor.force_gc()
            
            if not self.monitor.check_memory_available(memory_limit_mb):
                raise MemoryError(f"Cannot allocate {memory_limit_mb}MB")
                
        return operation()


# Utility functions for integration
def create_memory_monitor(
    memory_limit_mb: Optional[int] = None,
    verbose: bool = False,
    **config_kwargs
) -> MemoryMonitor:
    """Create a memory monitor with optional configuration."""
    config = AdaptiveConfig(**config_kwargs)
    return MemoryMonitor(memory_limit_mb, config, verbose)


def estimate_hypervector_memory(dimensions: int, vector_count: int) -> float:
    """Estimate memory usage for hypervector operations in MB."""
    # Rough estimation: 4 bytes per float32 element
    bytes_per_vector = dimensions * 4
    total_bytes = bytes_per_vector * vector_count
    
    # Add overhead for operations (2x for intermediate calculations)
    return (total_bytes * 2) / (1024 * 1024)


def get_safe_hypervector_dimensions(
    vector_count: int, 
    memory_monitor: MemoryMonitor,
    max_memory_mb: float = 1000
) -> int:
    """Get safe hypervector dimensions based on available memory."""
    available_mb = memory_monitor.get_stats().available_mb
    target_memory = min(max_memory_mb, available_mb * 0.5)  # Use 50% of available
    
    # Calculate maximum dimensions that fit in target memory
    max_dims = int((target_memory * 1024 * 1024) / (vector_count * 8))  # 8 bytes safety margin
    
    # Clamp to reasonable bounds
    config = memory_monitor.config
    max_dims = max(config.min_hypervector_dims, min(max_dims, config.max_hypervector_dims))
    
    return max_dims