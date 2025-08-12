"""
Performance monitoring and profiling utilities.

Provides execution time tracking, memory profiling, and bottleneck
identification for optimization.
"""

import time
import psutil
import logging
import threading
import cProfile
import pstats
import io
import tracemalloc
from typing import Dict, List, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
from datetime import datetime, timedelta
from collections import defaultdict, deque
import json
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Detailed performance metrics."""
    
    operation: str
    start_time: float
    end_time: float = 0.0
    duration: float = 0.0
    
    # Memory metrics
    memory_start: float = 0.0
    memory_end: float = 0.0
    memory_peak: float = 0.0
    memory_delta: float = 0.0
    
    # CPU metrics
    cpu_percent: float = 0.0
    cpu_time_user: float = 0.0
    cpu_time_system: float = 0.0
    
    # I/O metrics
    io_read_count: int = 0
    io_write_count: int = 0
    io_read_bytes: int = 0
    io_write_bytes: int = 0
    
    # Throughput
    items_processed: int = 0
    throughput: float = 0.0
    
    # Cache metrics
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Additional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def complete(self):
        """Mark operation as complete and calculate final metrics."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
        if self.items_processed > 0 and self.duration > 0:
            self.throughput = self.items_processed / self.duration
        
        self.memory_delta = self.memory_end - self.memory_start
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'operation': self.operation,
            'duration': round(self.duration, 3),
            'memory_delta_mb': round(self.memory_delta, 1),
            'memory_peak_mb': round(self.memory_peak, 1),
            'cpu_percent': round(self.cpu_percent, 1),
            'items_processed': self.items_processed,
            'throughput': round(self.throughput, 1) if self.throughput else 0,
            'cache_hit_rate': self.cache_hit_rate(),
            'timestamp': datetime.fromtimestamp(self.start_time).isoformat()
        }
    
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total) if total > 0 else 0.0


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system.
    """
    
    def __init__(self, enable_profiling: bool = False):
        """
        Initialize performance monitor.
        
        Args:
            enable_profiling: Enable detailed profiling
        """
        self.enable_profiling = enable_profiling
        self.metrics: List[PerformanceMetrics] = []
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        self.lock = threading.RLock()
        
        # Process monitoring
        self.process = psutil.Process()
        self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        # Profiling
        self.profiler: Optional[cProfile.Profile] = None
        self.memory_tracer_started = False
        
        # Operation statistics
        self.operation_stats: Dict[str, List[float]] = defaultdict(list)
        
        # Performance history (rolling window)
        self.history_window = 100
        self.performance_history = deque(maxlen=self.history_window)
    
    def start_operation(
        self,
        name: str,
        track_io: bool = False
    ) -> PerformanceMetrics:
        """
        Start tracking an operation.
        
        Args:
            name: Operation name
            track_io: Track I/O statistics
            
        Returns:
            PerformanceMetrics instance
        """
        # Get current resource usage
        memory_info = self.process.memory_info()
        cpu_times = self.process.cpu_times()
        
        metric = PerformanceMetrics(
            operation=name,
            start_time=time.time(),
            memory_start=memory_info.rss / 1024 / 1024,
            memory_peak=memory_info.rss / 1024 / 1024,
            cpu_time_user=cpu_times.user,
            cpu_time_system=cpu_times.system
        )
        
        if track_io:
            try:
                io_counters = self.process.io_counters()
                metric.io_read_count = io_counters.read_count
                metric.io_write_count = io_counters.write_count
                metric.io_read_bytes = io_counters.read_bytes
                metric.io_write_bytes = io_counters.write_bytes
            except:
                pass  # I/O tracking not available on all platforms
        
        with self.lock:
            self.active_operations[name] = metric
        
        return metric
    
    def end_operation(
        self,
        name: str,
        items_processed: int = 0,
        cache_hits: int = 0,
        cache_misses: int = 0
    ) -> Optional[PerformanceMetrics]:
        """
        End tracking an operation.
        
        Args:
            name: Operation name
            items_processed: Number of items processed
            cache_hits: Number of cache hits
            cache_misses: Number of cache misses
            
        Returns:
            Completed metrics or None
        """
        with self.lock:
            if name not in self.active_operations:
                return None
            
            metric = self.active_operations.pop(name)
        
        # Update final metrics
        memory_info = self.process.memory_info()
        cpu_times = self.process.cpu_times()
        
        metric.memory_end = memory_info.rss / 1024 / 1024
        metric.cpu_time_user = cpu_times.user - metric.cpu_time_user
        metric.cpu_time_system = cpu_times.system - metric.cpu_time_system
        
        # Calculate CPU percentage
        if metric.duration > 0:
            cpu_time_total = metric.cpu_time_user + metric.cpu_time_system
            metric.cpu_percent = (cpu_time_total / metric.duration) * 100
        
        # Update I/O if tracked
        if metric.io_read_count > 0:
            try:
                io_counters = self.process.io_counters()
                metric.io_read_count = io_counters.read_count - metric.io_read_count
                metric.io_write_count = io_counters.write_count - metric.io_write_count
                metric.io_read_bytes = io_counters.read_bytes - metric.io_read_bytes
                metric.io_write_bytes = io_counters.write_bytes - metric.io_write_bytes
            except:
                pass
        
        metric.items_processed = items_processed
        metric.cache_hits = cache_hits
        metric.cache_misses = cache_misses
        metric.complete()
        
        # Store metrics
        with self.lock:
            self.metrics.append(metric)
            self.operation_stats[name].append(metric.duration)
            self.performance_history.append(metric)
        
        return metric
    
    @contextmanager
    def track(
        self,
        operation: str,
        track_io: bool = False
    ):
        """
        Context manager for tracking operations.
        
        Args:
            operation: Operation name
            track_io: Track I/O statistics
            
        Yields:
            PerformanceMetrics instance
        """
        metric = self.start_operation(operation, track_io)
        
        try:
            yield metric
        finally:
            self.end_operation(
                operation,
                metric.items_processed,
                metric.cache_hits,
                metric.cache_misses
            )
    
    def start_profiling(self):
        """Start CPU profiling."""
        if not self.enable_profiling:
            return
        
        if self.profiler is None:
            self.profiler = cProfile.Profile()
            self.profiler.enable()
            logger.info("CPU profiling started")
    
    def stop_profiling(self) -> str:
        """
        Stop CPU profiling and return results.
        
        Returns:
            Profiling results as string
        """
        if self.profiler is None:
            return "Profiling not enabled"
        
        self.profiler.disable()
        
        # Generate statistics
        s = io.StringIO()
        ps = pstats.Stats(self.profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(50)  # Top 50 functions
        
        result = s.getvalue()
        self.profiler = None
        
        logger.info("CPU profiling stopped")
        return result
    
    def start_memory_profiling(self):
        """Start memory profiling."""
        if not self.memory_tracer_started:
            tracemalloc.start()
            self.memory_tracer_started = True
            logger.info("Memory profiling started")
    
    def get_memory_snapshot(self) -> Dict[str, Any]:
        """
        Get current memory usage snapshot.
        
        Returns:
            Memory usage details
        """
        if not self.memory_tracer_started:
            self.start_memory_profiling()
        
        snapshot = tracemalloc.take_snapshot()
        top_stats = snapshot.statistics('lineno')
        
        # Get top memory users
        top_users = []
        for stat in top_stats[:20]:  # Top 20
            top_users.append({
                'file': stat.traceback.format()[0] if stat.traceback else 'unknown',
                'size_mb': stat.size / 1024 / 1024,
                'count': stat.count
            })
        
        # Current memory usage
        current, peak = tracemalloc.get_traced_memory()
        
        return {
            'current_mb': current / 1024 / 1024,
            'peak_mb': peak / 1024 / 1024,
            'top_users': top_users,
            'process_memory_mb': self.process.memory_info().rss / 1024 / 1024
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        if not self.metrics:
            return {'status': 'no_metrics'}
        
        total_duration = sum(m.duration for m in self.metrics)
        total_items = sum(m.items_processed for m in self.metrics)
        
        # Operation statistics
        operation_summary = {}
        for op, durations in self.operation_stats.items():
            if durations:
                operation_summary[op] = {
                    'count': len(durations),
                    'total': sum(durations),
                    'avg': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations)
                }
        
        # Memory statistics
        memory_deltas = [m.memory_delta for m in self.metrics if m.memory_delta != 0]
        memory_peaks = [m.memory_peak for m in self.metrics]
        
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        return {
            'total_operations': len(self.metrics),
            'total_duration': round(total_duration, 2),
            'total_items_processed': total_items,
            'overall_throughput': round(total_items / total_duration, 1) if total_duration > 0 else 0,
            'memory': {
                'current_mb': round(current_memory, 1),
                'baseline_mb': round(self.baseline_memory, 1),
                'increase_mb': round(current_memory - self.baseline_memory, 1),
                'peak_mb': round(max(memory_peaks), 1) if memory_peaks else 0,
                'avg_delta_mb': round(sum(memory_deltas) / len(memory_deltas), 1) if memory_deltas else 0
            },
            'operations': operation_summary,
            'bottlenecks': self.identify_bottlenecks(),
            'recommendations': self.get_recommendations()
        }
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        if not self.metrics:
            return []
        
        bottlenecks = []
        total_duration = sum(m.duration for m in self.metrics)
        
        # Find slowest operations
        sorted_metrics = sorted(self.metrics, key=lambda m: m.duration, reverse=True)
        
        for metric in sorted_metrics[:5]:
            percentage = (metric.duration / total_duration * 100) if total_duration > 0 else 0
            
            if percentage > 5:  # More than 5% of total time
                bottleneck = {
                    'operation': metric.operation,
                    'duration': round(metric.duration, 2),
                    'percentage': round(percentage, 1),
                    'items': metric.items_processed,
                    'throughput': round(metric.throughput, 1) if metric.throughput else 0
                }
                
                # Add specific bottleneck type
                if metric.cache_hit_rate() < 0.5 and (metric.cache_hits + metric.cache_misses) > 0:
                    bottleneck['issue'] = 'low_cache_hit_rate'
                    bottleneck['cache_hit_rate'] = round(metric.cache_hit_rate(), 2)
                elif metric.memory_delta > 100:  # More than 100MB
                    bottleneck['issue'] = 'high_memory_usage'
                    bottleneck['memory_delta_mb'] = round(metric.memory_delta, 1)
                elif metric.cpu_percent > 90:
                    bottleneck['issue'] = 'high_cpu_usage'
                    bottleneck['cpu_percent'] = round(metric.cpu_percent, 1)
                
                bottlenecks.append(bottleneck)
        
        return bottlenecks
    
    def get_recommendations(self) -> List[str]:
        """Get performance improvement recommendations."""
        recommendations = []
        
        if not self.metrics:
            return recommendations
        
        # Analyze cache performance
        total_hits = sum(m.cache_hits for m in self.metrics)
        total_misses = sum(m.cache_misses for m in self.metrics)
        
        if total_hits + total_misses > 0:
            hit_rate = total_hits / (total_hits + total_misses)
            if hit_rate < 0.5:
                recommendations.append(
                    f"Low cache hit rate ({hit_rate:.1%}). Consider increasing cache size or TTL."
                )
        
        # Analyze memory usage
        current_memory = self.process.memory_info().rss / 1024 / 1024
        if current_memory > 2048:  # More than 2GB
            recommendations.append(
                f"High memory usage ({current_memory:.0f} MB). Consider enabling memory optimization."
            )
        
        # Analyze operation distribution
        operation_counts = defaultdict(int)
        for metric in self.metrics:
            operation_counts[metric.operation] += 1
        
        # Find repeated operations
        for op, count in operation_counts.items():
            if count > 100:
                recommendations.append(
                    f"Operation '{op}' executed {count} times. Consider batching or caching."
                )
        
        # Analyze throughput
        if self.metrics:
            avg_throughput = sum(m.throughput for m in self.metrics if m.throughput > 0)
            if avg_throughput > 0:
                avg_throughput /= len([m for m in self.metrics if m.throughput > 0])
                if avg_throughput < 10:  # Less than 10 items/second
                    recommendations.append(
                        f"Low average throughput ({avg_throughput:.1f} items/s). "
                        "Consider parallel processing."
                    )
        
        return recommendations
    
    def export_metrics(self, filepath: Path):
        """
        Export metrics to file.
        
        Args:
            filepath: Output file path
        """
        data = {
            'summary': self.get_summary(),
            'metrics': [m.to_dict() for m in self.metrics],
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Metrics exported to {filepath}")
    
    def reset(self):
        """Reset all metrics."""
        with self.lock:
            self.metrics.clear()
            self.active_operations.clear()
            self.operation_stats.clear()
            self.performance_history.clear()
            self.baseline_memory = self.process.memory_info().rss / 1024 / 1024
        
        if self.profiler:
            self.profiler.disable()
            self.profiler = None
        
        if self.memory_tracer_started:
            tracemalloc.stop()
            self.memory_tracer_started = False


def track_performance(
    name: Optional[str] = None,
    track_io: bool = False
):
    """
    Decorator to track function performance.
    
    Args:
        name: Operation name (defaults to function name)
        track_io: Track I/O statistics
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = name or func.__name__
            
            # Get or create monitor
            monitor = get_monitor()
            
            with monitor.track(op_name, track_io) as metric:
                result = func(*args, **kwargs)
                
                # Try to infer items processed
                if isinstance(result, (list, tuple, set)):
                    metric.items_processed = len(result)
                elif isinstance(result, dict):
                    metric.items_processed = len(result)
                
                return result
        
        return wrapper
    return decorator


class PerformanceOptimizer:
    """
    Automatic performance optimization based on monitoring.
    """
    
    def __init__(self, monitor: PerformanceMonitor):
        """
        Initialize optimizer.
        
        Args:
            monitor: Performance monitor instance
        """
        self.monitor = monitor
        self.optimizations_applied = []
    
    def analyze_and_optimize(self) -> Dict[str, Any]:
        """
        Analyze performance and apply optimizations.
        
        Returns:
            Optimization results
        """
        summary = self.monitor.get_summary()
        
        if summary.get('status') == 'no_metrics':
            return {'status': 'no_data'}
        
        optimizations = []
        
        # Check memory usage
        memory = summary.get('memory', {})
        current_mb = memory.get('current_mb', 0)
        
        if current_mb > 2048:  # More than 2GB
            # Clear caches
            from .cache import clear_all_caches
            clear_all_caches()
            
            # Force garbage collection
            import gc
            gc.collect()
            
            optimizations.append({
                'type': 'memory_cleanup',
                'reason': f'High memory usage: {current_mb:.0f} MB',
                'action': 'Cleared caches and forced garbage collection'
            })
        
        # Check cache performance
        bottlenecks = summary.get('bottlenecks', [])
        
        for bottleneck in bottlenecks:
            if bottleneck.get('issue') == 'low_cache_hit_rate':
                optimizations.append({
                    'type': 'cache_tuning',
                    'reason': f"Low cache hit rate for {bottleneck['operation']}",
                    'recommendation': 'Increase cache size or TTL'
                })
        
        # Check for repeated operations
        operations = summary.get('operations', {})
        
        for op, stats in operations.items():
            if stats['count'] > 100 and stats['avg'] > 0.1:
                optimizations.append({
                    'type': 'batching_opportunity',
                    'operation': op,
                    'reason': f"Operation executed {stats['count']} times",
                    'recommendation': 'Consider batching or parallelization'
                })
        
        self.optimizations_applied.extend(optimizations)
        
        return {
            'optimizations': optimizations,
            'total_applied': len(self.optimizations_applied),
            'current_memory_mb': self.monitor.process.memory_info().rss / 1024 / 1024
        }


# Global monitor instance
_monitor: Optional[PerformanceMonitor] = None


def get_monitor(enable_profiling: bool = False) -> PerformanceMonitor:
    """Get or create global performance monitor."""
    global _monitor
    
    if _monitor is None:
        _monitor = PerformanceMonitor(enable_profiling)
    
    return _monitor


def reset_monitor():
    """Reset global monitor."""
    if _monitor:
        _monitor.reset()