"""
Watchdog system for analyzer timeout protection and monitoring.

This module provides a comprehensive watchdog system to prevent analyzer hangs,
track execution times, and provide fallback mechanisms for long-running operations.
"""

import time
import threading
import multiprocessing
import signal
import traceback
import logging
from queue import Queue, Empty
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import functools

# Third-party imports for fallback functionality
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    import numpy as np
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..core.issues import Issue

logger = logging.getLogger(__name__)


@dataclass
class AnalyzerExecutionStats:
    """Statistics for analyzer execution tracking."""
    analyzer_name: str
    start_time: float = 0.0
    end_time: float = 0.0
    duration: float = 0.0
    timeout_count: int = 0
    error_count: int = 0
    issues_found: int = 0
    heartbeat_count: int = 0
    last_heartbeat: float = 0.0
    status: str = "pending"  # pending, running, completed, timeout, error
    error_message: Optional[str] = None
    execution_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class WatchdogConfig:
    """Configuration for watchdog timeout and heartbeat settings."""
    analyzer_timeout: float = 30.0  # seconds
    heartbeat_interval: float = 2.0  # seconds
    heartbeat_timeout_multiplier: float = 3.0  # timeout = interval * multiplier
    enable_fallback: bool = True
    enable_threading: bool = True
    max_retries: int = 1
    verbose_logging: bool = False
    execution_report: bool = True


class HeartbeatMonitor:
    """Monitors periodic heartbeats from running analyzers."""
    
    def __init__(self, interval: float = 2.0, timeout_multiplier: float = 3.0):
        """
        Initialize heartbeat monitor.
        
        Args:
            interval: Heartbeat interval in seconds
            timeout_multiplier: Timeout = interval * multiplier
        """
        self.interval = interval
        self.timeout = interval * timeout_multiplier
        self.heartbeats: Dict[str, float] = {}
        self.callbacks: Dict[str, Callable] = {}
        self.running = False
        self._lock = threading.Lock()
        self._monitor_thread: Optional[threading.Thread] = None
        
    def start_monitoring(self):
        """Start the heartbeat monitoring thread."""
        if not self.running:
            self.running = True
            self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self._monitor_thread.start()
            logger.debug("Heartbeat monitor started")
    
    def stop_monitoring(self):
        """Stop the heartbeat monitoring thread."""
        self.running = False
        if self._monitor_thread and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        logger.debug("Heartbeat monitor stopped")
    
    def register_analyzer(self, analyzer_id: str, timeout_callback: Optional[Callable] = None):
        """Register an analyzer for heartbeat monitoring."""
        with self._lock:
            self.heartbeats[analyzer_id] = time.time()
            if timeout_callback:
                self.callbacks[analyzer_id] = timeout_callback
        logger.debug(f"Registered analyzer for heartbeat monitoring: {analyzer_id}")
    
    def unregister_analyzer(self, analyzer_id: str):
        """Unregister an analyzer from heartbeat monitoring."""
        with self._lock:
            self.heartbeats.pop(analyzer_id, None)
            self.callbacks.pop(analyzer_id, None)
        logger.debug(f"Unregistered analyzer from heartbeat monitoring: {analyzer_id}")
    
    def heartbeat(self, analyzer_id: str):
        """Record a heartbeat from an analyzer."""
        with self._lock:
            if analyzer_id in self.heartbeats:
                self.heartbeats[analyzer_id] = time.time()
    
    def _monitor_loop(self):
        """Main monitoring loop that checks for timeouts."""
        while self.running:
            current_time = time.time()
            timed_out = []
            
            with self._lock:
                for analyzer_id, last_heartbeat in self.heartbeats.items():
                    if current_time - last_heartbeat > self.timeout:
                        timed_out.append(analyzer_id)
            
            # Handle timeouts outside the lock
            for analyzer_id in timed_out:
                logger.warning(f"Heartbeat timeout detected for analyzer: {analyzer_id}")
                callback = self.callbacks.get(analyzer_id)
                if callback:
                    try:
                        callback(analyzer_id)
                    except Exception as e:
                        logger.error(f"Error in timeout callback for {analyzer_id}: {e}")
            
            time.sleep(self.interval / 2)  # Check twice per interval


class SemanticAnalysisFallback:
    """Provides TF-IDF fallback for semantic analysis when full analysis times out."""
    
    @staticmethod
    def tfidf_fallback(functions: List[Dict[str, Any]], threshold: float = 0.8) -> List[Issue]:
        """
        TF-IDF fallback for semantic duplicate detection.
        
        Args:
            functions: List of function dictionaries with 'name', 'file', 'line', 'body'
            threshold: Similarity threshold for duplicates
            
        Returns:
            List of semantic duplicate issues found
        """
        if not SKLEARN_AVAILABLE:
            logger.warning("scikit-learn not available for TF-IDF fallback")
            return []
        
        if len(functions) < 2:
            return []
        
        issues = []
        
        try:
            # Extract function bodies for TF-IDF analysis
            function_bodies = []
            for func in functions:
                body = func.get('body', '')
                if isinstance(body, str):
                    function_bodies.append(body)
                else:
                    function_bodies.append(str(body))
            
            if len(function_bodies) < 2:
                return []
            
            # Create TF-IDF vectors
            vectorizer = TfidfVectorizer(
                stop_words='english',
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            
            tfidf_matrix = vectorizer.fit_transform(function_bodies)
            
            # Calculate cosine similarity with warnings suppressed for edge cases
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                similarity_matrix = cosine_similarity(tfidf_matrix)
            
            # Find similar pairs
            for i in range(len(functions)):
                for j in range(i + 1, len(functions)):
                    similarity = similarity_matrix[i, j]
                    
                    if similarity >= threshold:
                        func1, func2 = functions[i], functions[j]
                        
                        issue = Issue(
                            kind="semantic_duplicate_function",
                            message=f"TF-IDF detected semantic similarity ({similarity:.2f}) between "
                                   f"'{func1['name']}' and '{func2['name']}'",
                            severity=2,
                            file=func1.get('file'),
                            line=func1.get('line'),
                            evidence={
                                'similarity_score': similarity,
                                'method': 'tfidf_fallback',
                                'pair': [func1, func2],
                                'same_file': func1.get('file') == func2.get('file')
                            },
                            confidence=min(0.8, similarity)  # Lower confidence than full analysis
                        )
                        issues.append(issue)
        
        except Exception as e:
            logger.error(f"Error in TF-IDF fallback analysis: {e}")
        
        logger.info(f"TF-IDF fallback found {len(issues)} semantic duplicates")
        return issues


class AnalyzerWatchdog:
    """Main watchdog to wrap and monitor analyzers with timeout protection."""
    
    def __init__(self, config: Optional[WatchdogConfig] = None):
        """
        Initialize analyzer watchdog.
        
        Args:
            config: Watchdog configuration
        """
        self.config = config or WatchdogConfig()
        self.stats: Dict[str, AnalyzerExecutionStats] = {}
        self.heartbeat_monitor = HeartbeatMonitor(
            interval=self.config.heartbeat_interval,
            timeout_multiplier=self.config.heartbeat_timeout_multiplier
        )
        self._lock = threading.Lock()
        
        # Start heartbeat monitoring
        self.heartbeat_monitor.start_monitoring()
        
    def __del__(self):
        """Cleanup when watchdog is destroyed."""
        self.shutdown()
    
    def shutdown(self):
        """Shutdown the watchdog and cleanup resources."""
        self.heartbeat_monitor.stop_monitoring()
    
    def wrap_analyzer(self, analyzer, analyzer_name: str) -> Callable:
        """
        Wrap an analyzer with timeout protection and monitoring.
        
        Args:
            analyzer: The analyzer object to wrap
            analyzer_name: Name for tracking and logging
            
        Returns:
            Wrapped analyzer function with timeout protection
        """
        @functools.wraps(analyzer.run)
        def wrapped_analyzer(*args, **kwargs):
            return self._execute_with_monitoring(
                analyzer.run, analyzer_name, *args, **kwargs
            )
        
        return wrapped_analyzer
    
    def _execute_with_monitoring(self, analyzer_func: Callable, analyzer_name: str, 
                               *args, **kwargs) -> List[Issue]:
        """
        Execute analyzer with comprehensive monitoring and timeout protection.
        
        Args:
            analyzer_func: Analyzer function to execute
            analyzer_name: Name for tracking
            *args, **kwargs: Arguments for analyzer function
            
        Returns:
            List of issues or fallback results
        """
        # Initialize stats
        stats = AnalyzerExecutionStats(analyzer_name=analyzer_name)
        stats.start_time = time.time()
        stats.status = "running"
        
        with self._lock:
            self.stats[analyzer_name] = stats
        
        # Register for heartbeat monitoring
        def timeout_callback(analyzer_id: str):
            logger.warning(f"Analyzer '{analyzer_id}' heartbeat timeout")
            with self._lock:
                if analyzer_id in self.stats:
                    self.stats[analyzer_id].timeout_count += 1
        
        self.heartbeat_monitor.register_analyzer(analyzer_name, timeout_callback)
        
        try:
            # Create heartbeat context for injection
            heartbeat_context = self._create_heartbeat_context(analyzer_name)
            
            if self.config.enable_threading:
                # Execute with thread-based timeout
                issues = self._execute_with_thread_timeout(
                    analyzer_func, analyzer_name, heartbeat_context, *args, **kwargs
                )
            else:
                # Execute directly with heartbeat injection
                issues = self._execute_with_heartbeat_injection(
                    analyzer_func, analyzer_name, heartbeat_context, *args, **kwargs
                )
            
            # Convert generator to list if needed
            if issues is not None:
                issues = list(issues)
            
            # Record successful completion
            stats.end_time = time.time()
            stats.duration = stats.end_time - stats.start_time
            stats.status = "completed"
            stats.issues_found = len(issues) if issues else 0
            
            if self.config.verbose_logging:
                logger.info(f"Analyzer '{analyzer_name}' completed in {stats.duration:.2f}s, "
                           f"found {stats.issues_found} issues")
            
            return issues
            
        except FutureTimeoutError:
            return self._handle_timeout(analyzer_name, stats, *args, **kwargs)
        except Exception as e:
            return self._handle_error(analyzer_name, stats, e)
        finally:
            self.heartbeat_monitor.unregister_analyzer(analyzer_name)
            self._finalize_stats(stats)
    
    def _create_heartbeat_context(self, analyzer_name: str) -> Dict[str, Any]:
        """Create context for heartbeat injection."""
        return {
            'heartbeat': lambda: self.heartbeat_monitor.heartbeat(analyzer_name),
            'analyzer_name': analyzer_name,
            'start_time': time.time()
        }
    
    def _execute_with_thread_timeout(self, analyzer_func: Callable, analyzer_name: str,
                                   heartbeat_context: Dict[str, Any], *args, **kwargs) -> List[Issue]:
        """Execute analyzer with thread-based timeout."""
        with ThreadPoolExecutor(max_workers=1) as executor:
            # Inject heartbeat context if analyzer supports it
            if hasattr(analyzer_func, '__self__'):
                analyzer_obj = analyzer_func.__self__
                if hasattr(analyzer_obj, '_set_heartbeat_context'):
                    analyzer_obj._set_heartbeat_context(heartbeat_context)
            
            future = executor.submit(analyzer_func, *args, **kwargs)
            
            # Monitor with periodic heartbeats
            heartbeat_thread = threading.Thread(
                target=self._heartbeat_worker,
                args=(analyzer_name, future),
                daemon=True
            )
            heartbeat_thread.start()
            
            try:
                return future.result(timeout=self.config.analyzer_timeout)
            except Exception:
                future.cancel()
                raise
    
    def _execute_with_heartbeat_injection(self, analyzer_func: Callable, analyzer_name: str,
                                        heartbeat_context: Dict[str, Any], *args, **kwargs) -> List[Issue]:
        """Execute analyzer with heartbeat context injection."""
        # Start heartbeat thread
        stop_heartbeat = threading.Event()
        heartbeat_thread = threading.Thread(
            target=self._periodic_heartbeat,
            args=(analyzer_name, stop_heartbeat),
            daemon=True
        )
        heartbeat_thread.start()
        
        try:
            # Try to inject context into analyzer
            if hasattr(analyzer_func, '__self__'):
                analyzer_obj = analyzer_func.__self__
                if hasattr(analyzer_obj, '_set_heartbeat_context'):
                    analyzer_obj._set_heartbeat_context(heartbeat_context)
            
            return analyzer_func(*args, **kwargs)
        finally:
            stop_heartbeat.set()
    
    def _heartbeat_worker(self, analyzer_name: str, future):
        """Worker thread for sending periodic heartbeats."""
        while not future.done():
            self.heartbeat_monitor.heartbeat(analyzer_name)
            with self._lock:
                if analyzer_name in self.stats:
                    self.stats[analyzer_name].heartbeat_count += 1
                    self.stats[analyzer_name].last_heartbeat = time.time()
            time.sleep(self.config.heartbeat_interval)
    
    def _periodic_heartbeat(self, analyzer_name: str, stop_event: threading.Event):
        """Send periodic heartbeats until stopped."""
        while not stop_event.is_set():
            self.heartbeat_monitor.heartbeat(analyzer_name)
            with self._lock:
                if analyzer_name in self.stats:
                    self.stats[analyzer_name].heartbeat_count += 1
                    self.stats[analyzer_name].last_heartbeat = time.time()
            stop_event.wait(self.config.heartbeat_interval)
    
    def _handle_timeout(self, analyzer_name: str, stats: AnalyzerExecutionStats,
                       *args, **kwargs) -> List[Issue]:
        """Handle analyzer timeout with fallback if available."""
        stats.timeout_count += 1
        stats.status = "timeout"
        stats.end_time = time.time()
        stats.duration = stats.end_time - stats.start_time
        
        logger.warning(f"Analyzer '{analyzer_name}' timed out after {stats.duration:.2f}s")
        
        # Try fallback for semantic analysis
        if self.config.enable_fallback and 'semantic' in analyzer_name.lower():
            try:
                logger.info(f"Attempting TF-IDF fallback for '{analyzer_name}'")
                return self._try_semantic_fallback(*args, **kwargs)
            except Exception as e:
                logger.error(f"Fallback failed for '{analyzer_name}': {e}")
        
        return []
    
    def _try_semantic_fallback(self, *args, **kwargs) -> List[Issue]:
        """Try semantic analysis fallback using TF-IDF."""
        # Extract context and attempt fallback
        if args and hasattr(args[0], 'ast_index'):
            ctx = args[0]
            functions = self._extract_functions_for_fallback(ctx)
            return SemanticAnalysisFallback.tfidf_fallback(functions)
        return []
    
    def _extract_functions_for_fallback(self, ctx) -> List[Dict[str, Any]]:
        """Extract function information for fallback analysis."""
        functions = []
        try:
            import ast
            for file_path, tree in ctx.ast_index.items():
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        try:
                            # Get function body as string
                            body = ast.unparse(node) if hasattr(ast, 'unparse') else str(node)
                        except:
                            body = str(node)
                        
                        functions.append({
                            'name': node.name,
                            'file': file_path,
                            'line': getattr(node, 'lineno', 1),
                            'body': body
                        })
        except Exception as e:
            logger.debug(f"Error extracting functions for fallback: {e}")
        
        return functions
    
    def _handle_error(self, analyzer_name: str, stats: AnalyzerExecutionStats, 
                     error: Exception) -> List[Issue]:
        """Handle analyzer execution error."""
        stats.error_count += 1
        stats.status = "error"
        stats.end_time = time.time()
        stats.duration = stats.end_time - stats.start_time
        stats.error_message = str(error)
        
        logger.error(f"Analyzer '{analyzer_name}' failed: {error}")
        if self.config.verbose_logging:
            logger.debug(f"Analyzer '{analyzer_name}' traceback:\n{traceback.format_exc()}")
        
        return []
    
    def _finalize_stats(self, stats: AnalyzerExecutionStats):
        """Finalize execution statistics."""
        execution_record = {
            'timestamp': stats.start_time,
            'duration': stats.duration,
            'status': stats.status,
            'heartbeats': stats.heartbeat_count,
            'timeouts': stats.timeout_count,
            'errors': stats.error_count,
            'issues': stats.issues_found
        }
        stats.execution_history.append(execution_record)
    
    def get_execution_report(self) -> Dict[str, Any]:
        """Generate comprehensive execution report."""
        with self._lock:
            total_analyzers = len(self.stats)
            total_duration = sum(s.duration for s in self.stats.values())
            total_timeouts = sum(s.timeout_count for s in self.stats.values())
            total_errors = sum(s.error_count for s in self.stats.values())
            total_issues = sum(s.issues_found for s in self.stats.values())
            
            # Find slowest analyzers
            slowest = sorted(
                [(name, stats.duration) for name, stats in self.stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            # Find most problematic analyzers
            problematic = sorted(
                [(name, stats.timeout_count + stats.error_count) 
                 for name, stats in self.stats.items()],
                key=lambda x: x[1],
                reverse=True
            )[:5]
            
            return {
                'summary': {
                    'total_analyzers': total_analyzers,
                    'total_duration': total_duration,
                    'average_duration': total_duration / max(total_analyzers, 1),
                    'total_timeouts': total_timeouts,
                    'total_errors': total_errors,
                    'total_issues': total_issues
                },
                'slowest_analyzers': slowest,
                'most_problematic': problematic,
                'analyzer_details': {
                    name: {
                        'duration': stats.duration,
                        'status': stats.status,
                        'timeouts': stats.timeout_count,
                        'errors': stats.error_count,
                        'issues_found': stats.issues_found,
                        'heartbeats': stats.heartbeat_count
                    }
                    for name, stats in self.stats.items()
                }
            }
    
    def clear_stats(self):
        """Clear all execution statistics."""
        with self._lock:
            self.stats.clear()
    
    @contextmanager
    def monitoring_context(self, analyzer_name: str):
        """Context manager for manual analyzer monitoring."""
        self.heartbeat_monitor.register_analyzer(analyzer_name)
        try:
            yield lambda: self.heartbeat_monitor.heartbeat(analyzer_name)
        finally:
            self.heartbeat_monitor.unregister_analyzer(analyzer_name)


# Convenience functions for easy integration
def create_watchdog(timeout: float = 30.0, heartbeat_interval: float = 2.0, 
                   enable_fallback: bool = True) -> AnalyzerWatchdog:
    """Create a configured analyzer watchdog."""
    config = WatchdogConfig(
        analyzer_timeout=timeout,
        heartbeat_interval=heartbeat_interval,
        enable_fallback=enable_fallback
    )
    return AnalyzerWatchdog(config)


def monitor_analyzer(analyzer, name: str, watchdog: Optional[AnalyzerWatchdog] = None):
    """Decorator to add watchdog monitoring to an analyzer."""
    if watchdog is None:
        watchdog = create_watchdog()
    
    return watchdog.wrap_analyzer(analyzer, name)