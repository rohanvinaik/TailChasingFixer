"""
Watchdog system to prevent analyzer hangs and track execution times.
"""

import time
import threading
import multiprocessing
import signal
import traceback
from typing import Any, Dict, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from queue import Queue, Empty
import logging

from ..core.issues import Issue


logger = logging.getLogger(__name__)


@dataclass
class AnalyzerExecutionStats:
    """Statistics for analyzer execution."""
    analyzer_name: str
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    timed_out: bool = False
    error: Optional[str] = None
    file_being_processed: Optional[str] = None
    function_being_processed: Optional[str] = None
    issues_found: int = 0
    heartbeat_count: int = 0
    last_heartbeat: Optional[float] = None


@dataclass
class WatchdogConfig:
    """Configuration for the watchdog."""
    analyzer_timeout: float = 30.0  # seconds
    heartbeat_interval: float = 2.0  # seconds
    heartbeat_timeout_multiplier: float = 3.0  # x heartbeat_interval
    enable_fallback: bool = True
    max_retries: int = 1
    verbose: bool = False


class HeartbeatMonitor:
    """Monitor heartbeats from analyzers."""
    
    def __init__(self, interval: float = 2.0, timeout_multiplier: float = 3.0):
        self.interval = interval
        self.timeout = interval * timeout_multiplier
        self.heartbeats: Dict[str, float] = {}
        self.running = False
        self.monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self.timeout_callbacks: Dict[str, Callable] = {}
        
    def start(self):
        """Start the heartbeat monitor."""
        self.running = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        
    def stop(self):
        """Stop the heartbeat monitor."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1)
            
    def register(self, analyzer_name: str, timeout_callback: Optional[Callable] = None):
        """Register an analyzer for heartbeat monitoring."""
        with self._lock:
            self.heartbeats[analyzer_name] = time.time()
            if timeout_callback:
                self.timeout_callbacks[analyzer_name] = timeout_callback
                
    def unregister(self, analyzer_name: str):
        """Unregister an analyzer from heartbeat monitoring."""
        with self._lock:
            self.heartbeats.pop(analyzer_name, None)
            self.timeout_callbacks.pop(analyzer_name, None)
            
    def heartbeat(self, analyzer_name: str):
        """Record a heartbeat from an analyzer."""
        with self._lock:
            self.heartbeats[analyzer_name] = time.time()
            
    def _monitor_loop(self):
        """Monitor heartbeats and trigger timeouts."""
        while self.running:
            current_time = time.time()
            with self._lock:
                for analyzer_name, last_heartbeat in list(self.heartbeats.items()):
                    if current_time - last_heartbeat > self.timeout:
                        logger.warning(
                            f"Analyzer '{analyzer_name}' heartbeat timeout "
                            f"({current_time - last_heartbeat:.1f}s since last heartbeat)"
                        )
                        callback = self.timeout_callbacks.get(analyzer_name)
                        if callback:
                            try:
                                callback(analyzer_name)
                            except Exception as e:
                                logger.error(f"Error in timeout callback: {e}")
            time.sleep(self.interval / 2)


class AnalyzerWatchdog:
    """Watchdog to monitor and control analyzer execution."""
    
    def __init__(self, config: Optional[WatchdogConfig] = None):
        self.config = config or WatchdogConfig()
        self.heartbeat_monitor = HeartbeatMonitor(
            interval=self.config.heartbeat_interval,
            timeout_multiplier=self.config.heartbeat_timeout_multiplier
        )
        self.execution_stats: List[AnalyzerExecutionStats] = []
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._current_stats: Dict[str, AnalyzerExecutionStats] = {}
        self._lock = threading.Lock()
        
    def start(self):
        """Start the watchdog."""
        self.heartbeat_monitor.start()
        
    def stop(self):
        """Stop the watchdog."""
        self.heartbeat_monitor.stop()
        self._executor.shutdown(wait=False)
        
    def wrap_analyzer(
        self,
        analyzer_name: str,
        analyzer_func: Callable,
        fallback_func: Optional[Callable] = None
    ) -> Callable:
        """
        Wrap an analyzer function with timeout and heartbeat monitoring.
        
        Args:
            analyzer_name: Name of the analyzer
            analyzer_func: The analyzer function to wrap
            fallback_func: Optional fallback function if analyzer times out
            
        Returns:
            Wrapped analyzer function
        """
        def wrapped(*args, **kwargs):
            return self._execute_with_monitoring(
                analyzer_name,
                analyzer_func,
                fallback_func,
                args,
                kwargs
            )
        return wrapped
        
    def _execute_with_monitoring(
        self,
        analyzer_name: str,
        analyzer_func: Callable,
        fallback_func: Optional[Callable],
        args: tuple,
        kwargs: dict
    ) -> List[Issue]:
        """Execute an analyzer with monitoring."""
        stats = AnalyzerExecutionStats(
            analyzer_name=analyzer_name,
            start_time=time.time()
        )
        
        with self._lock:
            self._current_stats[analyzer_name] = stats
            
        # Create heartbeat wrapper
        heartbeat_queue = Queue()
        
        def heartbeat_wrapper(*args, **kwargs):
            """Wrapper that sends heartbeats during execution."""
            def send_heartbeat():
                while not heartbeat_queue.empty():
                    try:
                        heartbeat_queue.get_nowait()
                        self.heartbeat_monitor.heartbeat(analyzer_name)
                        stats.heartbeat_count += 1
                        stats.last_heartbeat = time.time()
                    except Empty:
                        break
                        
            # Register with heartbeat monitor
            self.heartbeat_monitor.register(
                analyzer_name,
                lambda name: self._handle_heartbeat_timeout(name)
            )
            
            try:
                # Create a heartbeat-aware context
                if hasattr(args[0], '__dict__'):  # Check if first arg is context
                    context = args[0]
                    original_log = getattr(context, 'log', None)
                    
                    def log_with_heartbeat(message: str, level: str = "info"):
                        heartbeat_queue.put(True)
                        send_heartbeat()
                        if original_log:
                            original_log(message, level)
                            
                    # Inject heartbeat into context
                    context.log = log_with_heartbeat
                    
                # Run the analyzer
                result = analyzer_func(*args, **kwargs)
                
                # Convert generator to list, sending heartbeats
                issues = []
                if hasattr(result, '__iter__'):
                    for issue in result:
                        heartbeat_queue.put(True)
                        send_heartbeat()
                        issues.append(issue)
                        stats.issues_found += 1
                else:
                    issues = result
                    
                return issues
                
            finally:
                self.heartbeat_monitor.unregister(analyzer_name)
                
        try:
            # Execute with timeout
            future = self._executor.submit(heartbeat_wrapper, *args, **kwargs)
            issues = future.result(timeout=self.config.analyzer_timeout)
            
            # Success
            stats.end_time = time.time()
            stats.duration = stats.end_time - stats.start_time
            
            if self.config.verbose:
                logger.info(
                    f"Analyzer '{analyzer_name}' completed in {stats.duration:.2f}s "
                    f"({stats.issues_found} issues found)"
                )
                
            return issues
            
        except FutureTimeoutError:
            # Timeout occurred
            stats.timed_out = True
            stats.end_time = time.time()
            stats.duration = stats.end_time - stats.start_time
            stats.error = f"Timeout after {self.config.analyzer_timeout}s"
            
            logger.error(
                f"Analyzer '{analyzer_name}' timed out after {self.config.analyzer_timeout}s"
                f" (file: {stats.file_being_processed}, function: {stats.function_being_processed})"
            )
            
            # Try fallback if available
            if fallback_func and self.config.enable_fallback:
                logger.info(f"Attempting fallback for '{analyzer_name}'")
                try:
                    return fallback_func(*args, **kwargs)
                except Exception as e:
                    logger.error(f"Fallback also failed: {e}")
                    
            return []
            
        except Exception as e:
            # Other error
            stats.error = str(e)
            stats.end_time = time.time()
            stats.duration = stats.end_time - stats.start_time
            
            logger.error(
                f"Analyzer '{analyzer_name}' failed: {e}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            return []
            
        finally:
            self.execution_stats.append(stats)
            with self._lock:
                self._current_stats.pop(analyzer_name, None)
                
    def _handle_heartbeat_timeout(self, analyzer_name: str):
        """Handle heartbeat timeout for an analyzer."""
        stats = self._current_stats.get(analyzer_name)
        if stats:
            logger.error(
                f"Heartbeat timeout for '{analyzer_name}' - "
                f"last file: {stats.file_being_processed}, "
                f"last function: {stats.function_being_processed}"
            )
            
    def update_analyzer_context(
        self,
        analyzer_name: str,
        file_path: Optional[str] = None,
        function_name: Optional[str] = None
    ):
        """Update the context information for an analyzer."""
        with self._lock:
            stats = self._current_stats.get(analyzer_name)
            if stats:
                if file_path:
                    stats.file_being_processed = file_path
                if function_name:
                    stats.function_being_processed = function_name
                    
    def get_execution_report(self) -> Dict[str, Any]:
        """Get a report of all analyzer executions."""
        total_duration = sum(s.duration or 0 for s in self.execution_stats)
        timeout_count = sum(1 for s in self.execution_stats if s.timed_out)
        error_count = sum(1 for s in self.execution_stats if s.error and not s.timed_out)
        
        analyzer_summary = {}
        for stats in self.execution_stats:
            if stats.analyzer_name not in analyzer_summary:
                analyzer_summary[stats.analyzer_name] = {
                    'executions': 0,
                    'total_duration': 0,
                    'avg_duration': 0,
                    'timeouts': 0,
                    'errors': 0,
                    'issues_found': 0,
                    'avg_heartbeats': 0
                }
                
            summary = analyzer_summary[stats.analyzer_name]
            summary['executions'] += 1
            summary['total_duration'] += stats.duration or 0
            summary['timeouts'] += 1 if stats.timed_out else 0
            summary['errors'] += 1 if stats.error and not stats.timed_out else 0
            summary['issues_found'] += stats.issues_found
            summary['avg_heartbeats'] += stats.heartbeat_count
            
        # Calculate averages
        for summary in analyzer_summary.values():
            if summary['executions'] > 0:
                summary['avg_duration'] = summary['total_duration'] / summary['executions']
                summary['avg_heartbeats'] = summary['avg_heartbeats'] / summary['executions']
                
        return {
            'total_executions': len(self.execution_stats),
            'total_duration': total_duration,
            'timeout_count': timeout_count,
            'error_count': error_count,
            'analyzer_summary': analyzer_summary,
            'slowest_analyzers': self._get_slowest_analyzers(5),
            'most_problematic': self._get_most_problematic_analyzers(3)
        }
        
    def _get_slowest_analyzers(self, n: int = 5) -> List[Dict[str, Any]]:
        """Get the n slowest analyzer executions."""
        sorted_stats = sorted(
            [s for s in self.execution_stats if s.duration],
            key=lambda s: s.duration or 0,
            reverse=True
        )[:n]
        
        return [
            {
                'analyzer': s.analyzer_name,
                'duration': s.duration,
                'timed_out': s.timed_out,
                'file': s.file_being_processed,
                'function': s.function_being_processed
            }
            for s in sorted_stats
        ]
        
    def _get_most_problematic_analyzers(self, n: int = 3) -> List[Dict[str, Any]]:
        """Get the n most problematic analyzers (timeouts + errors)."""
        problem_count = {}
        for stats in self.execution_stats:
            if stats.timed_out or stats.error:
                problem_count[stats.analyzer_name] = problem_count.get(stats.analyzer_name, 0) + 1
                
        sorted_problems = sorted(
            problem_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:n]
        
        return [
            {'analyzer': name, 'problem_count': count}
            for name, count in sorted_problems
        ]


class SemanticAnalysisFallback:
    """Fallback mechanism for semantic analysis."""
    
    @staticmethod
    def tfidf_fallback(context, *args, **kwargs) -> List[Issue]:
        """
        Fallback to TF-IDF based analysis when semantic analysis times out.
        """
        logger.info("Falling back to TF-IDF analysis")
        
        try:
            from ..analyzers.advanced.enhanced_semantic import EnhancedSemanticAnalyzer as SemanticDuplicateAnalyzer
            from sklearn.feature_extraction.text import TfidfVectorizer
            import numpy as np
            
            # Create a simplified TF-IDF analyzer
            issues = []
            functions = []
            
            # Collect all functions
            for file_path, ast_tree in context.ast_index.items():
                import ast
                for node in ast.walk(ast_tree):
                    if isinstance(node, ast.FunctionDef):
                        # Extract function text
                        try:
                            func_text = ast.unparse(node)
                            functions.append({
                                'name': node.name,
                                'file': file_path,
                                'line': getattr(node, 'lineno', 1),
                                'text': func_text
                            })
                        except:
                            pass
                            
            if len(functions) < 2:
                return issues
                
            # Build TF-IDF matrix
            vectorizer = TfidfVectorizer(
                max_features=100,
                token_pattern=r'\b[a-zA-Z_][a-zA-Z0-9_]*\b'
            )
            
            texts = [f['text'] for f in functions]
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            # Find similar pairs
            threshold = 0.8
            for i in range(len(functions)):
                for j in range(i + 1, len(functions)):
                    similarity = (tfidf_matrix[i] * tfidf_matrix[j].T).toarray()[0, 0]
                    
                    if similarity > threshold:
                        issues.append(Issue(
                            kind="semantic_duplicate_tfidf",
                            message=f"Functions '{functions[i]['name']}' and '{functions[j]['name']}' are similar (TF-IDF: {similarity:.2f})",
                            file=functions[i]['file'],
                            line=functions[i]['line'],
                            severity=2,  # WARNING level
                            confidence=similarity,
                            evidence={
                                'similar_to': functions[j]['name'],
                                'similar_file': functions[j]['file'],
                                'similarity_score': similarity,
                                'method': 'tfidf_fallback'
                            }
                        ))
                        
            return issues
            
        except Exception as e:
            logger.error(f"TF-IDF fallback failed: {e}")
            return []