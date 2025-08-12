"""
Performance profiler for analyzing performance metrics.
"""

import time
import tracemalloc
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass, field
from contextlib import contextmanager
from functools import wraps
import threading

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None


@dataclass
class ProfileData:
    """Data collected for a profiled component."""
    name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    calls: int = 0
    memory_start: Optional[int] = None
    memory_end: Optional[int] = None
    memory_peak: Optional[int] = None
    sub_components: Dict[str, 'ProfileData'] = field(default_factory=dict)
    
    def finish(self):
        """Mark component as finished and calculate duration."""
        self.end_time = time.time()
        self.duration = self.end_time - self.start_time
        
    def add_call(self):
        """Increment call counter."""
        self.calls += 1


class PerformanceProfiler:
    """Tracks performance metrics for analysis."""
    
    def __init__(self, enabled: bool = True, track_memory: bool = True):
        """
        Initialize profiler.
        
        Args:
            enabled: Whether profiling is enabled
            track_memory: Whether to track memory usage
        """
        self.enabled = enabled
        self.track_memory = track_memory and enabled
        
        self.profile_data: Dict[str, ProfileData] = {}
        self.current_stack: List[str] = []
        self._lock = threading.Lock()
        
        # Memory tracking
        self.memory_tracker_started = False
        if self.track_memory:
            try:
                tracemalloc.start()
                self.memory_tracker_started = True
            except:
                self.track_memory = False
                
        # System resources
        self.process = None
        if HAS_PSUTIL:
            try:
                self.process = psutil.Process()
            except:
                self.process = None
        self.start_time = time.time()
        self.start_memory = self._get_memory_usage()
        
    def _get_memory_usage(self) -> int:
        """Get current memory usage in bytes."""
        if not self.process:
            return 0
        try:
            return self.process.memory_info().rss
        except:
            return 0
            
    @contextmanager
    def profile(self, component_name: str):
        """
        Context manager for profiling a component.
        
        Args:
            component_name: Name of the component to profile
        """
        if not self.enabled:
            yield
            return
            
        with self._lock:
            # Create or get profile data
            if component_name not in self.profile_data:
                self.profile_data[component_name] = ProfileData(
                    name=component_name,
                    start_time=time.time()
                )
            
            profile = self.profile_data[component_name]
            profile.add_call()
            
            if self.track_memory:
                profile.memory_start = self._get_memory_usage()
                
            self.current_stack.append(component_name)
            
        try:
            yield profile
        finally:
            with self._lock:
                if self.current_stack and self.current_stack[-1] == component_name:
                    self.current_stack.pop()
                    
                profile.finish()
                
                if self.track_memory:
                    profile.memory_end = self._get_memory_usage()
                    profile.memory_peak = max(
                        profile.memory_peak or 0,
                        profile.memory_end or 0
                    )
                    
    def profile_function(self, name: Optional[str] = None):
        """
        Decorator for profiling functions.
        
        Args:
            name: Optional name for the component (defaults to function name)
        """
        def decorator(func: Callable) -> Callable:
            component_name = name or f"{func.__module__}.{func.__name__}"
            
            @wraps(func)
            def wrapper(*args, **kwargs):
                with self.profile(component_name):
                    return func(*args, **kwargs)
                    
            return wrapper if self.enabled else func
            
        return decorator
        
    def get_report(self) -> Dict[str, Any]:
        """
        Get profiling report.
        
        Returns:
            Dictionary containing profiling data
        """
        if not self.enabled:
            return {}
            
        report = {}
        
        # Component timings
        for name, data in self.profile_data.items():
            report[name] = {
                "time": data.duration or 0,
                "calls": data.calls,
                "avg_time": (data.duration or 0) / data.calls if data.calls > 0 else 0
            }
            
            if self.track_memory and data.memory_start and data.memory_end:
                report[name]["memory_delta"] = data.memory_end - data.memory_start
                report[name]["memory_peak"] = data.memory_peak
                
        # Overall metrics
        total_time = time.time() - self.start_time
        current_memory = self._get_memory_usage()
        
        report["_total"] = {
            "time": total_time,
            "memory_start": self.start_memory,
            "memory_current": current_memory,
            "memory_delta": current_memory - self.start_memory
        }
        
        # System metrics
        if self.process:
            try:
                report["_system"] = {
                    "cpu_percent": self.process.cpu_percent(),
                    "memory_rss_mb": current_memory / 1024 / 1024,
                    "memory_percent": self.process.memory_percent(),
                    "num_threads": self.process.num_threads()
                }
            except:
                pass
            
        # Memory profiling if available
        if self.track_memory and self.memory_tracker_started:
            snapshot = tracemalloc.take_snapshot()
            top_stats = snapshot.statistics('lineno')[:10]
            
            report["_memory_top"] = []
            for stat in top_stats:
                report["_memory_top"].append({
                    "file": stat.traceback.format()[0] if stat.traceback else "unknown",
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count
                })
                
        return report
        
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summarized profiling data.
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.enabled:
            return {}
            
        report = self.get_report()
        
        # Find slowest components
        components = [(name, data["time"]) for name, data in report.items() 
                     if not name.startswith("_") and "time" in data]
        components.sort(key=lambda x: x[1], reverse=True)
        
        summary = {
            "total_time": report.get("_total", {}).get("time", 0),
            "total_memory_mb": report.get("_system", {}).get("memory_rss_mb", 0),
            "slowest_components": components[:5],
            "component_count": len(components)
        }
        
        # Calculate time breakdown
        total_component_time = sum(data["time"] for name, data in report.items() 
                                 if not name.startswith("_") and "time" in data)
        
        summary["time_breakdown"] = {}
        for name, time_val in components[:10]:
            percentage = (time_val / total_component_time * 100) if total_component_time > 0 else 0
            summary["time_breakdown"][name] = {
                "time": time_val,
                "percentage": percentage
            }
            
        return summary
        
    def reset(self):
        """Reset profiling data."""
        with self._lock:
            self.profile_data.clear()
            self.current_stack.clear()
            self.start_time = time.time()
            self.start_memory = self._get_memory_usage()
            
    def stop(self):
        """Stop profiling and clean up resources."""
        if self.memory_tracker_started:
            try:
                tracemalloc.stop()
            except:
                pass
                
    def __enter__(self):
        """Context manager entry."""
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop()


class ComponentTimer:
    """Simple timer for measuring component execution time."""
    
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        
    @contextmanager
    def time(self, component: str):
        """
        Time a component's execution.
        
        Args:
            component: Component name
        """
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self.timings.setdefault(component, []).append(duration)
            
    def get_stats(self) -> Dict[str, Dict[str, float]]:
        """
        Get timing statistics.
        
        Returns:
            Dictionary with timing statistics per component
        """
        stats = {}
        
        for component, times in self.timings.items():
            stats[component] = {
                "total": sum(times),
                "count": len(times),
                "average": sum(times) / len(times) if times else 0,
                "min": min(times) if times else 0,
                "max": max(times) if times else 0
            }
            
        return stats
        
    def clear(self):
        """Clear all timings."""
        self.timings.clear()


def profile_analyzer(profiler: Optional[PerformanceProfiler] = None):
    """
    Decorator for profiling analyzer execution.
    
    Args:
        profiler: Optional profiler instance (creates new if None)
    """
    def decorator(analyzer_class):
        """Decorate analyzer class."""
        original_run = analyzer_class.run
        
        def profiled_run(self, ctx):
            """Profiled version of run method."""
            nonlocal profiler
            if profiler is None:
                profiler = PerformanceProfiler(enabled=False)
                
            with profiler.profile(f"analyzer.{self.name}"):
                return original_run(self, ctx)
                
        analyzer_class.run = profiled_run
        return analyzer_class
        
    return decorator