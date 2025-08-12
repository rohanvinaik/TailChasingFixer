"""
Integration layer for parallel semantic analysis with existing TailChasingFixer pipeline.

Provides a seamless interface between the high-performance parallel analyzer
and the existing semantic duplicate detection system.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import tempfile

from ..performance.parallel_analyzer import (
    ParallelSemanticAnalyzer,
    analyze_codebase_parallel,
    SharedMemoryConfig,
    EncodingConfig
)
from ..catalytic.hv_encoder import HypervectorEncoder
from ..core.issues import Issue, IssueSeverity
from .similarity import SimilarityAnalyzer

logger = logging.getLogger(__name__)


class ParallelSemanticDetector:
    """
    High-level interface for parallel semantic duplicate detection.
    
    Integrates with existing TailChasingFixer analyzers while providing
    high-performance parallel processing for large codebases.
    """
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 cache_dir: Optional[str] = None,
                 num_processes: Optional[int] = None):
        """
        Initialize parallel semantic detector.
        
        Args:
            config: Configuration dictionary
            cache_dir: Directory for persistent caching
            num_processes: Number of parallel processes
        """
        self.config = config or {}
        self.cache_dir = cache_dir
        
        # Semantic analysis configuration
        self.similarity_threshold = self.config.get('semantic_similarity_threshold', 0.85)
        self.z_threshold = self.config.get('z_threshold', 2.5)
        self.fdr_alpha = self.config.get('fdr_q', 0.05)
        
        # Memory and performance configuration
        memory_config = SharedMemoryConfig(
            max_vectors_in_memory=self.config.get('max_vectors_in_memory', 10000),
            vector_dimension=self.config.get('vector_dimension', 8192),
            use_memory_mapping=self.config.get('use_memory_mapping', True),
            cache_directory=cache_dir
        )
        
        encoding_config = EncodingConfig(
            dimension=self.config.get('vector_dimension', 8192),
            sparsity=self.config.get('vector_sparsity', 0.33),
            max_depth=self.config.get('max_ast_depth', 10),
            normalize_ast=self.config.get('normalize_ast', True)
        )
        
        # Initialize parallel analyzer
        self.parallel_analyzer = ParallelSemanticAnalyzer(
            num_processes=num_processes,
            cache_dir=cache_dir,
            memory_config=memory_config,
            encoding_config=encoding_config
        )
        
        # Initialize traditional analyzer for comparison/fallback
        self.similarity_analyzer = SimilarityAnalyzer(self.config)
        
        logger.info(f"Initialized ParallelSemanticDetector with {self.parallel_analyzer.num_processes} processes")
    
    def analyze_files(self, 
                     file_paths: List[str],
                     progress_callback: Optional[callable] = None) -> List[Issue]:
        """
        Analyze Python files for semantic duplicates using parallel processing.
        
        Args:
            file_paths: List of Python file paths to analyze
            progress_callback: Optional progress callback function
            
        Returns:
            List of semantic duplicate issues found
        """
        logger.info(f"Starting parallel semantic analysis of {len(file_paths)} files")
        
        try:
            # Use the convenience function for codebase analysis
            results = analyze_codebase_parallel(
                file_paths,
                similarity_threshold=self.similarity_threshold,
                num_processes=self.parallel_analyzer.num_processes,
                cache_dir=self.cache_dir
            )
            
            # Convert results to Issue objects
            issues = self._convert_results_to_issues(results, file_paths)
            
            # Log performance metrics
            stats = results.get('stats')
            if stats:
                logger.info(f"Parallel analysis completed: {stats.processed_functions} functions "
                           f"in {stats.total_time:.2f}s ({stats.functions_per_second:.1f} func/sec)")
                logger.info(f"Peak memory: {stats.peak_memory_mb:.1f} MB, "
                           f"Cache hit rate: {stats.cache_hit_rate:.2%}")
            
            return issues
            
        except Exception as e:
            logger.error(f"Parallel semantic analysis failed: {e}")
            logger.info("Falling back to traditional semantic analysis")
            return self._fallback_analysis(file_paths, progress_callback)
    
    def analyze_functions(self, 
                         functions: List[Tuple[str, Any]],
                         progress_callback: Optional[callable] = None) -> List[Issue]:
        """
        Analyze a list of function ASTs for semantic duplicates.
        
        Args:
            functions: List of (function_id, ast_node) tuples
            progress_callback: Optional progress callback function
            
        Returns:
            List of semantic duplicate issues found
        """
        logger.info(f"Analyzing {len(functions)} functions for semantic duplicates")
        
        try:
            results = self.parallel_analyzer.analyze_functions_parallel(
                functions,
                similarity_threshold=self.similarity_threshold,
                progress_callback=progress_callback
            )
            
            # Convert to issues
            issues = self._convert_similarity_results_to_issues(results['similarities'])
            
            logger.info(f"Found {len(issues)} semantic duplicate issues")
            return issues
            
        except Exception as e:
            logger.error(f"Function analysis failed: {e}")
            return []
    
    def get_performance_report(self) -> str:
        """Get detailed performance report from the parallel analyzer."""
        return self.parallel_analyzer.get_performance_report()
    
    def clear_cache(self):
        """Clear the persistent cache."""
        self.parallel_analyzer.cache.clear()
        logger.info("Cleared semantic analysis cache")
    
    def _convert_results_to_issues(self, 
                                  results: Dict[str, Any], 
                                  file_paths: List[str]) -> List[Issue]:
        """Convert parallel analysis results to Issue objects."""
        issues = []
        similarities = results.get('similarities', [])
        
        for sim in similarities:
            func1_id = sim['function1_id']
            func2_id = sim['function2_id']
            similarity_score = sim['similarity']
            confidence = sim.get('confidence', similarity_score)
            
            # Extract file information from function IDs
            file1 = self._extract_file_from_function_id(func1_id)
            file2 = self._extract_file_from_function_id(func2_id)
            
            # Create issue for semantic duplicate
            issue = Issue(
                kind="semantic_duplicate_function",
                message=f"Semantic duplicate detected with {self._extract_symbol_from_function_id(func2_id)} "
                       f"in {file2} (similarity: {similarity_score:.3f})",
                severity=IssueSeverity.WARNING.value,
                file=file1,
                line=self._extract_line_from_function_id(func1_id),
                symbol=self._extract_symbol_from_function_id(func1_id),
                evidence={
                    'duplicate_file': file2,
                    'duplicate_symbol': self._extract_symbol_from_function_id(func2_id),
                    'duplicate_line': self._extract_line_from_function_id(func2_id),
                    'similarity_score': similarity_score,
                    'confidence': confidence,
                    'analysis_method': 'parallel_hypervector',
                    'function1_id': func1_id,
                    'function2_id': func2_id
                },
                confidence=confidence
            )
            
            issues.append(issue)
        
        return issues
    
    def _convert_similarity_results_to_issues(self, similarities: List[Dict[str, Any]]) -> List[Issue]:
        """Convert similarity results directly to issues."""
        issues = []
        
        for sim in similarities:
            func1_id = sim['function1_id']
            func2_id = sim['function2_id']
            similarity_score = sim['similarity']
            confidence = sim.get('confidence', similarity_score)
            
            file1 = self._extract_file_from_function_id(func1_id)
            file2 = self._extract_file_from_function_id(func2_id)
            
            issue = Issue(
                kind="semantic_duplicate_function",
                message=f"Semantic duplicate: {similarity_score:.1%} similar to {self._extract_symbol_from_function_id(func2_id)}",
                severity=IssueSeverity.WARNING.value,
                file=file1,
                line=self._extract_line_from_function_id(func1_id),
                symbol=self._extract_symbol_from_function_id(func1_id),
                evidence={
                    'duplicate_file': file2,
                    'duplicate_symbol': self._extract_symbol_from_function_id(func2_id),
                    'duplicate_line': self._extract_line_from_function_id(func2_id),
                    'similarity_score': similarity_score,
                    'confidence': confidence,
                    'analysis_method': 'parallel_hypervector'
                },
                confidence=confidence
            )
            
            issues.append(issue)
        
        return issues
    
    def _extract_file_from_function_id(self, function_id: str) -> str:
        """Extract file path from function ID format: file:name:line"""
        return function_id.split(':')[0] if ':' in function_id else 'unknown'
    
    def _extract_symbol_from_function_id(self, function_id: str) -> str:
        """Extract symbol name from function ID."""
        parts = function_id.split(':')
        return parts[1] if len(parts) > 1 else 'unknown'
    
    def _extract_line_from_function_id(self, function_id: str) -> int:
        """Extract line number from function ID."""
        parts = function_id.split(':')
        try:
            return int(parts[2]) if len(parts) > 2 else 1
        except (ValueError, IndexError):
            return 1
    
    def _fallback_analysis(self, 
                          file_paths: List[str], 
                          progress_callback: Optional[callable] = None) -> List[Issue]:
        """Fallback to traditional analysis if parallel analysis fails."""
        # This would integrate with existing semantic analysis
        # For now, return empty list as a placeholder
        logger.warning("Fallback analysis not yet implemented")
        return []


def create_parallel_semantic_detector(config: Dict[str, Any]) -> ParallelSemanticDetector:
    """
    Factory function to create a configured parallel semantic detector.
    
    Args:
        config: Configuration dictionary containing semantic analysis settings
        
    Returns:
        Configured ParallelSemanticDetector instance
    """
    # Set up cache directory
    cache_dir = config.get('cache_dir')
    if cache_dir is None:
        cache_dir = str(Path(tempfile.gettempdir()) / "tailchasing_semantic_cache")
    
    # Determine optimal process count
    num_processes = config.get('num_processes')
    if num_processes is None:
        import multiprocessing as mp
        num_processes = mp.cpu_count()
    
    detector = ParallelSemanticDetector(
        config=config,
        cache_dir=cache_dir,
        num_processes=num_processes
    )
    
    logger.info(f"Created parallel semantic detector: {num_processes} processes, cache: {cache_dir}")
    return detector


# Integration with existing analyzer interface
class ParallelSemanticAnalyzerInterface:
    """
    Interface adapter for existing analyzer framework.
    
    Allows the parallel semantic detector to be used as a drop-in replacement
    for existing semantic analyzers.
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.detector = create_parallel_semantic_detector(config)
        self.enabled = config.get('enable_parallel_semantic', True)
    
    def analyze(self, files: List[str], **kwargs) -> List[Issue]:
        """Main analysis method compatible with existing analyzer interface."""
        if not self.enabled:
            return []
        
        progress_callback = kwargs.get('progress_callback')
        return self.detector.analyze_files(files, progress_callback)
    
    def can_analyze(self, file_path: str) -> bool:
        """Check if this analyzer can handle the given file."""
        return file_path.endswith('.py')
    
    def get_name(self) -> str:
        """Get analyzer name."""
        return "parallel_semantic_duplicate"
    
    def get_description(self) -> str:
        """Get analyzer description."""
        return "High-performance parallel semantic duplicate detection using hypervectors"
    
    def cleanup(self):
        """Cleanup resources."""
        # Close any open resources
        pass