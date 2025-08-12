"""
Parallel processing utilities for performance optimization.

Provides concurrent and parallel execution for CPU and I/O bound tasks
with automatic batching and load balancing.
"""

import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed, Future
from typing import List, Dict, Any, Optional, Callable, Tuple, Iterator, TypeVar, Generic
from dataclasses import dataclass
import logging
import time
import queue
import threading
from pathlib import Path
import ast
import numpy as np
from contextlib import contextmanager

logger = logging.getLogger(__name__)

T = TypeVar('T')
R = TypeVar('R')


@dataclass
class TaskResult(Generic[T]):
    """Result of a parallel task."""
    
    task_id: str
    result: Optional[T]
    error: Optional[Exception]
    duration: float
    
    @property
    def success(self) -> bool:
        """Check if task succeeded."""
        return self.error is None


class ParallelExecutor:
    """
    Advanced parallel executor with dynamic scheduling and load balancing.
    """
    
    def __init__(
        self,
        max_workers: Optional[int] = None,
        use_processes: bool = False,
        chunk_size: Optional[int] = None
    ):
        """
        Initialize parallel executor.
        
        Args:
            max_workers: Maximum number of workers
            use_processes: Use processes instead of threads
            chunk_size: Default chunk size for batching
        """
        self.max_workers = max_workers or mp.cpu_count()
        self.use_processes = use_processes
        self.chunk_size = chunk_size
        
        self._executor: Optional[Any] = None
        self._shutdown = False
    
    def __enter__(self):
        """Context manager entry."""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()
    
    def start(self):
        """Start the executor."""
        if self._executor is None:
            if self.use_processes:
                self._executor = ProcessPoolExecutor(max_workers=self.max_workers)
            else:
                self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._shutdown = False
    
    def shutdown(self, wait: bool = True):
        """Shutdown the executor."""
        if self._executor and not self._shutdown:
            self._executor.shutdown(wait=wait)
            self._executor = None
            self._shutdown = True
    
    def map(
        self,
        func: Callable[[T], R],
        items: List[T],
        chunk_size: Optional[int] = None,
        timeout: Optional[float] = None
    ) -> List[R]:
        """
        Map function over items in parallel.
        
        Args:
            func: Function to apply
            items: Items to process
            chunk_size: Items per chunk
            timeout: Timeout per item
            
        Returns:
            List of results in original order
        """
        if not items:
            return []
        
        self.start()
        chunk_size = chunk_size or self.chunk_size or 1
        
        # Submit tasks
        futures: Dict[Future, int] = {}
        
        for i in range(0, len(items), chunk_size):
            chunk = items[i:i + chunk_size]
            if chunk_size == 1:
                future = self._executor.submit(func, chunk[0])
                futures[future] = i
            else:
                future = self._executor.submit(self._process_chunk, func, chunk)
                futures[future] = i
        
        # Collect results
        results = [None] * len(items)
        
        for future in as_completed(futures, timeout=timeout):
            idx = futures[future]
            try:
                result = future.result()
                if chunk_size == 1:
                    results[idx] = result
                else:
                    for j, r in enumerate(result):
                        results[idx + j] = r
            except Exception as e:
                logger.error(f"Task failed: {e}")
                if chunk_size == 1:
                    results[idx] = None
                else:
                    for j in range(len(items[idx:idx + chunk_size])):
                        results[idx + j] = None
        
        return results
    
    def map_async(
        self,
        func: Callable[[T], R],
        items: List[T]
    ) -> List[Future]:
        """
        Submit tasks asynchronously.
        
        Args:
            func: Function to apply
            items: Items to process
            
        Returns:
            List of futures
        """
        self.start()
        
        futures = []
        for item in items:
            future = self._executor.submit(func, item)
            futures.append(future)
        
        return futures
    
    def starmap(
        self,
        func: Callable,
        args_list: List[Tuple],
        chunk_size: Optional[int] = None
    ) -> List[Any]:
        """
        Apply function with multiple arguments.
        
        Args:
            func: Function to apply
            args_list: List of argument tuples
            chunk_size: Items per chunk
            
        Returns:
            List of results
        """
        if not args_list:
            return []
        
        # Wrapper to unpack arguments
        def wrapper(args):
            return func(*args)
        
        return self.map(wrapper, args_list, chunk_size)
    
    @staticmethod
    def _process_chunk(func: Callable, chunk: List) -> List:
        """Process a chunk of items."""
        return [func(item) for item in chunk]
    
    def map_reduce(
        self,
        map_func: Callable[[T], R],
        reduce_func: Callable[[R, R], R],
        items: List[T],
        initial: Optional[R] = None
    ) -> Optional[R]:
        """
        Parallel map-reduce operation.
        
        Args:
            map_func: Map function
            reduce_func: Reduce function
            items: Items to process
            initial: Initial value for reduction
            
        Returns:
            Reduced result
        """
        if not items:
            return initial
        
        # Map phase
        mapped = self.map(map_func, items)
        
        # Reduce phase (could be parallelized for associative operations)
        result = initial
        for value in mapped:
            if value is not None:
                if result is None:
                    result = value
                else:
                    result = reduce_func(result, value)
        
        return result


class BatchProcessor:
    """
    Process items in optimized batches with progress tracking.
    """
    
    def __init__(
        self,
        func: Callable[[List[T]], List[R]],
        batch_size: int = 100,
        parallel: bool = True,
        max_workers: int = 4
    ):
        """
        Initialize batch processor.
        
        Args:
            func: Batch processing function
            batch_size: Items per batch
            parallel: Process batches in parallel
            max_workers: Number of parallel workers
        """
        self.func = func
        self.batch_size = batch_size
        self.parallel = parallel
        self.max_workers = max_workers
        
        self.pending_items: List[T] = []
        self.results: List[R] = []
        
        if parallel:
            self.executor = ParallelExecutor(max_workers=max_workers)
    
    def add(self, item: T):
        """Add item to batch."""
        self.pending_items.append(item)
        
        if len(self.pending_items) >= self.batch_size:
            self.flush()
    
    def add_many(self, items: List[T]):
        """Add multiple items."""
        for item in items:
            self.add(item)
    
    def flush(self):
        """Process all pending items."""
        if not self.pending_items:
            return
        
        # Create batches
        batches = []
        for i in range(0, len(self.pending_items), self.batch_size):
            batch = self.pending_items[i:i + self.batch_size]
            batches.append(batch)
        
        # Process batches
        if self.parallel and len(batches) > 1:
            batch_results = self.executor.map(self.func, batches)
            for batch_result in batch_results:
                if batch_result:
                    self.results.extend(batch_result)
        else:
            for batch in batches:
                batch_result = self.func(batch)
                if batch_result:
                    self.results.extend(batch_result)
        
        self.pending_items.clear()
    
    def get_results(self) -> List[R]:
        """Get all results."""
        self.flush()
        return self.results
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.flush()
        if self.parallel:
            self.executor.shutdown()


class ParallelFileProcessor:
    """
    Specialized parallel processor for file operations.
    """
    
    def __init__(self, max_workers: Optional[int] = None):
        """Initialize file processor."""
        self.executor = ParallelExecutor(
            max_workers=max_workers,
            use_processes=False  # Use threads for I/O
        )
    
    def parse_files(
        self,
        files: List[Path],
        cache_manager: Optional[Any] = None
    ) -> Dict[Path, ast.AST]:
        """
        Parse multiple Python files in parallel.
        
        Args:
            files: List of file paths
            cache_manager: Optional cache manager
            
        Returns:
            Dictionary mapping paths to ASTs
        """
        def parse_single(filepath: Path) -> Tuple[Path, Optional[ast.AST]]:
            """Parse a single file."""
            try:
                if cache_manager:
                    tree = cache_manager.ast_cache.get_ast(str(filepath))
                else:
                    content = filepath.read_text()
                    tree = ast.parse(content, filename=str(filepath))
                return filepath, tree
            except Exception as e:
                logger.debug(f"Failed to parse {filepath}: {e}")
                return filepath, None
        
        # Parse in parallel
        results = self.executor.map(parse_single, files)
        
        # Convert to dictionary
        return {path: tree for path, tree in results if tree is not None}
    
    def read_files(
        self,
        files: List[Path],
        encoding: str = 'utf-8'
    ) -> Dict[Path, str]:
        """
        Read multiple files in parallel.
        
        Args:
            files: List of file paths
            encoding: File encoding
            
        Returns:
            Dictionary mapping paths to contents
        """
        def read_single(filepath: Path) -> Tuple[Path, Optional[str]]:
            """Read a single file."""
            try:
                content = filepath.read_text(encoding=encoding)
                return filepath, content
            except Exception as e:
                logger.debug(f"Failed to read {filepath}: {e}")
                return filepath, None
        
        results = self.executor.map(read_single, files)
        return {path: content for path, content in results if content is not None}
    
    def process_files(
        self,
        files: List[Path],
        processor: Callable[[Path], Any],
        chunk_size: int = 10
    ) -> List[TaskResult]:
        """
        Process files with a custom processor.
        
        Args:
            files: List of file paths
            processor: Processing function
            chunk_size: Files per chunk
            
        Returns:
            List of task results
        """
        def process_with_timing(filepath: Path) -> TaskResult:
            """Process file and track timing."""
            start_time = time.time()
            
            try:
                result = processor(filepath)
                duration = time.time() - start_time
                
                return TaskResult(
                    task_id=str(filepath),
                    result=result,
                    error=None,
                    duration=duration
                )
            except Exception as e:
                duration = time.time() - start_time
                
                return TaskResult(
                    task_id=str(filepath),
                    result=None,
                    error=e,
                    duration=duration
                )
        
        return self.executor.map(process_with_timing, files, chunk_size)


class ParallelSemanticAnalyzer:
    """
    Parallel semantic analysis for hypervector operations.
    """
    
    def __init__(
        self,
        encoder: Any,
        max_workers: Optional[int] = None
    ):
        """
        Initialize semantic analyzer.
        
        Args:
            encoder: Semantic encoder instance
            max_workers: Maximum parallel workers
        """
        self.encoder = encoder
        self.executor = ParallelExecutor(
            max_workers=max_workers,
            use_processes=True  # Use processes for CPU-bound work
        )
    
    def compute_hypervectors(
        self,
        functions: List[Tuple[str, str]],
        cache_manager: Optional[Any] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute hypervectors for functions in parallel.
        
        Args:
            functions: List of (function_id, code) tuples
            cache_manager: Optional cache manager
            
        Returns:
            Dictionary mapping IDs to hypervectors
        """
        # Check cache first
        to_compute = []
        results = {}
        
        if cache_manager:
            for func_id, code in functions:
                cached = cache_manager.hypervector_cache.get(func_id)
                if cached is not None:
                    results[func_id] = cached
                else:
                    to_compute.append((func_id, code))
        else:
            to_compute = functions
        
        if not to_compute:
            return results
        
        # Compute missing vectors in parallel
        def compute_single(item: Tuple[str, str]) -> Tuple[str, np.ndarray]:
            """Compute single hypervector."""
            func_id, code = item
            hv, _ = self.encoder.encode_function(code)
            return func_id, hv
        
        computed = self.executor.map(compute_single, to_compute)
        
        # Cache and return
        for func_id, hv in computed:
            if hv is not None:
                if cache_manager:
                    cache_manager.hypervector_cache.set(func_id, hv)
                results[func_id] = hv
        
        return results
    
    def compute_similarity_matrix(
        self,
        vectors: Dict[str, np.ndarray],
        threshold: float = 0.0,
        cache_manager: Optional[Any] = None
    ) -> np.ndarray:
        """
        Compute similarity matrix in parallel.
        
        Args:
            vectors: Dictionary of hypervectors
            threshold: Minimum similarity threshold
            cache_manager: Optional cache manager
            
        Returns:
            Similarity matrix
        """
        ids = list(vectors.keys())
        n = len(ids)
        matrix = np.zeros((n, n))
        
        # Set diagonal
        np.fill_diagonal(matrix, 1.0)
        
        # Prepare pairs to compute
        pairs_to_compute = []
        
        for i in range(n):
            for j in range(i + 1, n):
                id1, id2 = ids[i], ids[j]
                
                # Check cache
                if cache_manager:
                    cached = cache_manager.similarity_cache.get(id1, id2)
                    if cached is not None:
                        matrix[i, j] = matrix[j, i] = cached
                        continue
                
                pairs_to_compute.append((i, j, vectors[id1], vectors[id2]))
        
        if pairs_to_compute:
            # Compute similarities in parallel
            def compute_similarity(item: Tuple) -> Tuple[int, int, float]:
                """Compute similarity for a pair."""
                i, j, v1, v2 = item
                # Cosine similarity
                sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                return i, j, sim
            
            # Process in batches to avoid memory issues
            batch_size = 1000
            for batch_start in range(0, len(pairs_to_compute), batch_size):
                batch = pairs_to_compute[batch_start:batch_start + batch_size]
                
                similarities = self.executor.map(compute_similarity, batch)
                
                for i, j, sim in similarities:
                    if sim >= threshold:
                        matrix[i, j] = matrix[j, i] = sim
                        
                        # Cache result
                        if cache_manager:
                            cache_manager.similarity_cache.set(ids[i], ids[j], sim)
        
        return matrix


class StreamingProcessor:
    """
    Process items in a streaming fashion with backpressure.
    """
    
    def __init__(
        self,
        processor: Callable[[T], R],
        buffer_size: int = 1000,
        max_workers: int = 4
    ):
        """
        Initialize streaming processor.
        
        Args:
            processor: Processing function
            buffer_size: Maximum buffer size
            max_workers: Number of workers
        """
        self.processor = processor
        self.buffer_size = buffer_size
        self.max_workers = max_workers
        
        self.input_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        self.output_queue: queue.Queue = queue.Queue(maxsize=buffer_size)
        
        self.workers: List[threading.Thread] = []
        self.stop_event = threading.Event()
    
    def start(self):
        """Start worker threads."""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker,
                name=f"StreamWorker-{i}"
            )
            worker.daemon = True
            worker.start()
            self.workers.append(worker)
    
    def stop(self):
        """Stop worker threads."""
        self.stop_event.set()
        
        # Add sentinel values to unblock workers
        for _ in range(self.max_workers):
            self.input_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5.0)
        
        self.workers.clear()
    
    def _worker(self):
        """Worker thread function."""
        while not self.stop_event.is_set():
            try:
                item = self.input_queue.get(timeout=0.1)
                
                if item is None:  # Sentinel
                    break
                
                try:
                    result = self.processor(item)
                    self.output_queue.put(result)
                except Exception as e:
                    logger.error(f"Processing error: {e}")
                    self.output_queue.put(None)
                
            except queue.Empty:
                continue
    
    def process(self, item: T):
        """Add item for processing."""
        self.input_queue.put(item)
    
    def get_result(self, timeout: Optional[float] = None) -> Optional[R]:
        """Get processed result."""
        try:
            return self.output_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def process_stream(
        self,
        items: Iterator[T]
    ) -> Iterator[R]:
        """
        Process a stream of items.
        
        Args:
            items: Iterator of items
            
        Yields:
            Processed results
        """
        self.start()
        
        try:
            # Submit items
            submitted = 0
            for item in items:
                self.process(item)
                submitted += 1
                
                # Yield available results
                while True:
                    try:
                        result = self.output_queue.get_nowait()
                        if result is not None:
                            yield result
                    except queue.Empty:
                        break
            
            # Collect remaining results
            collected = 0
            while collected < submitted:
                result = self.get_result(timeout=10.0)
                if result is not None:
                    yield result
                collected += 1
        
        finally:
            self.stop()


@contextmanager
def parallel_context(max_workers: Optional[int] = None, use_processes: bool = False):
    """
    Context manager for parallel processing.
    
    Args:
        max_workers: Maximum workers
        use_processes: Use processes instead of threads
        
    Yields:
        ParallelExecutor instance
    """
    executor = ParallelExecutor(max_workers, use_processes)
    executor.start()
    
    try:
        yield executor
    finally:
        executor.shutdown()


def parallel_map(
    func: Callable[[T], R],
    items: List[T],
    max_workers: Optional[int] = None,
    use_processes: bool = False
) -> List[R]:
    """
    Convenience function for parallel mapping.
    
    Args:
        func: Function to apply
        items: Items to process
        max_workers: Maximum workers
        use_processes: Use processes instead of threads
        
    Returns:
        List of results
    """
    with parallel_context(max_workers, use_processes) as executor:
        return executor.map(func, items)