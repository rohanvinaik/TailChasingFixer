"""
Performance metrics tracking execution time and resource usage.
"""

import time
import psutil
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PerformanceMetrics:
    """Metrics related to execution performance."""
    
    # Time tracking
    total_time: float = 0.0  # seconds
    step_times: List[float] = field(default_factory=list)
    average_step_time: float = 0.0
    min_step_time: float = float('inf')
    max_step_time: float = 0.0
    
    # Memory tracking
    peak_memory_mb: float = 0.0
    average_memory_mb: float = 0.0
    memory_samples: List[float] = field(default_factory=list)
    
    # CPU tracking
    peak_cpu_percent: float = 0.0
    average_cpu_percent: float = 0.0
    cpu_samples: List[float] = field(default_factory=list)
    
    # I/O tracking
    files_read: int = 0
    files_written: int = 0
    total_bytes_processed: int = 0
    
    # Performance issues
    slow_steps: int = 0  # Steps taking >2x average
    memory_spikes: int = 0  # Sudden memory increases
    
    def __init__(self):
        self.process = psutil.Process()
        self.start_time: Optional[float] = None
    
    def start_tracking(self):
        """Start performance tracking."""
        self.start_time = time.time()
        self.step_times = []
        self.memory_samples = []
        self.cpu_samples = []
        
        # Take initial measurements
        self._sample_resources()
    
    def record_step_time(self, step_duration: float):
        """Record time taken for a step."""
        self.step_times.append(step_duration)
        
        # Update min/max
        self.min_step_time = min(self.min_step_time, step_duration)
        self.max_step_time = max(self.max_step_time, step_duration)
        
        # Check for slow step
        if len(self.step_times) > 1:
            current_avg = sum(self.step_times[:-1]) / len(self.step_times[:-1])
            if step_duration > current_avg * 2:
                self.slow_steps += 1
        
        # Sample resources
        self._sample_resources()
    
    def record_file_operation(self, operation: str, bytes_count: int = 0):
        """Record file I/O operation."""
        if operation == "read":
            self.files_read += 1
        elif operation == "write":
            self.files_written += 1
        
        self.total_bytes_processed += bytes_count
    
    def _sample_resources(self):
        """Sample current resource usage."""
        try:
            # Memory usage
            memory_mb = self.process.memory_info().rss / 1024 / 1024
            self.memory_samples.append(memory_mb)
            self.peak_memory_mb = max(self.peak_memory_mb, memory_mb)
            
            # Check for memory spike
            if len(self.memory_samples) > 1:
                prev_memory = self.memory_samples[-2]
                if memory_mb > prev_memory * 1.5:  # 50% increase
                    self.memory_spikes += 1
            
            # CPU usage
            cpu_percent = self.process.cpu_percent(interval=0.1)
            self.cpu_samples.append(cpu_percent)
            self.peak_cpu_percent = max(self.peak_cpu_percent, cpu_percent)
            
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            # Process might have ended or we lack permissions
            pass
    
    def finish_tracking(self, total_time: Optional[float] = None):
        """Finish tracking and calculate final metrics."""
        if total_time:
            self.total_time = total_time
        elif self.start_time:
            self.total_time = time.time() - self.start_time
        
        # Calculate averages
        if self.step_times:
            self.average_step_time = sum(self.step_times) / len(self.step_times)
        
        if self.memory_samples:
            self.average_memory_mb = sum(self.memory_samples) / len(self.memory_samples)
        
        if self.cpu_samples:
            self.average_cpu_percent = sum(self.cpu_samples) / len(self.cpu_samples)
    
    def get_speed_score(self) -> float:
        """Calculate speed score (0-100) based on performance."""
        if self.total_time == 0:
            return 100.0
        
        base_score = 100.0
        
        # Penalize for slow execution (>30 seconds is slow for our benchmarks)
        if self.total_time > 30:
            time_penalty = min(50, (self.total_time - 30) / 30 * 50)
            base_score -= time_penalty
        
        # Penalize for slow steps
        if len(self.step_times) > 0:
            slow_step_ratio = self.slow_steps / len(self.step_times)
            base_score -= slow_step_ratio * 20
        
        # Penalize for memory issues
        base_score -= self.memory_spikes * 5
        
        # Penalize for excessive memory usage (>500MB is excessive)
        if self.peak_memory_mb > 500:
            memory_penalty = min(20, (self.peak_memory_mb - 500) / 500 * 20)
            base_score -= memory_penalty
        
        return max(0.0, base_score)
    
    def get_resource_efficiency(self) -> float:
        """Calculate resource efficiency score."""
        efficiency = 100.0
        
        # Memory efficiency
        if self.peak_memory_mb > 200:  # Base expectation is under 200MB
            memory_excess = (self.peak_memory_mb - 200) / 200
            efficiency -= min(30, memory_excess * 30)
        
        # CPU efficiency
        if self.average_cpu_percent > 50:  # Should not hog CPU
            cpu_excess = (self.average_cpu_percent - 50) / 50
            efficiency -= min(20, cpu_excess * 20)
        
        return max(0.0, efficiency)