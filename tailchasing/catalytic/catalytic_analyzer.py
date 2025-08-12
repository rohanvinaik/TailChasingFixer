"""
Catalytic analyzer for efficient duplicate detection.

Integrates the catalytic hypervector system with the TailChasingFixer framework.
"""

import ast
import tempfile
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from ..analyzers.base import Analyzer, AnalysisContext
from ..core.issues import Issue
from .hv_encoder import HypervectorEncoder, EncodingConfig
from .catalytic_index import CatalyticIndex
from .similarity_pipeline import SimilarityPipeline, QueryResult


class CatalyticDuplicateAnalyzer(Analyzer):
    """
    Analyzer using catalytic hypervector similarity for duplicate detection.
    
    Features:
    - O(N) complexity instead of O(N²)
    - Memory-mapped storage for large codebases
    - LSH-based approximate search
    - Working memory <100MB for 100k+ functions
    """
    
    name = "catalytic_duplicates"
    
    def __init__(self):
        """Initialize the analyzer."""
        self.encoder = HypervectorEncoder()
        self.index: Optional[CatalyticIndex] = None
        self.pipeline: Optional[SimilarityPipeline] = None
        self.logger = logging.getLogger(__name__)
        
        # Configuration
        self.hv_threshold = 0.88
        self.ast_threshold = 0.85
        self.min_similarity_for_issue = 0.9
        self.max_candidates = 50
        
        # Temporary directory for index
        self.temp_dir = None
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run catalytic duplicate detection.
        
        Args:
            ctx: Analysis context
            
        Returns:
            List of duplicate issues found
        """
        # Get configuration
        config = ctx.config.get('catalytic', {})
        if not config.get('enabled', True):
            return []
        
        # Update thresholds from config
        self.hv_threshold = config.get('hv_threshold', 0.88)
        self.ast_threshold = config.get('ast_threshold', 0.85)
        self.min_similarity_for_issue = config.get('min_similarity', 0.9)
        
        # Create temporary index
        self.temp_dir = tempfile.mkdtemp(prefix='catalytic_')
        self.index = CatalyticIndex(self.temp_dir, mode='w')
        self.pipeline = SimilarityPipeline(
            self.index,
            hv_threshold=self.hv_threshold,
            ast_threshold=self.ast_threshold,
            max_candidates=self.max_candidates
        )
        
        try:
            # Build index
            self._build_index(ctx)
            
            # Find duplicates
            issues = self._find_duplicates(ctx)
            
            # Log statistics
            stats = self.index.get_stats()
            self.logger.info(
                f"Catalytic analysis complete: {stats['num_functions']} functions indexed, "
                f"working memory: {stats['working_memory_mb']:.1f}MB"
            )
            
            return issues
            
        finally:
            # Clean up
            if self.index:
                self.index.close()
            if self.temp_dir and Path(self.temp_dir).exists():
                import shutil
                shutil.rmtree(self.temp_dir)
    
    def _build_index(self, ctx: AnalysisContext) -> None:
        """
        Build hypervector index from functions.
        
        Args:
            ctx: Analysis context
        """
        import time
        start_time = time.time()
        total_functions = 0
        skipped_functions = 0
        
        # Get symbol table
        functions = ctx.symbol_table.functions
        total_to_process = sum(len(entries) for entries in functions.values())
        
        self.logger.info(f"Building catalytic index for {total_to_process} function entries")
        
        try:
            for func_name, entries in functions.items():
                for entry in entries:
                    file_path = entry.get('file', '')
                    line_number = entry.get('lineno', 0)
                    node = entry.get('node')
                    
                    if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        skipped_functions += 1
                        continue
                    
                    # Build context with imports extracted from AST
                    try:
                        imports = self._extract_file_imports(ctx.ast_index.get(file_path))
                        context = {
                            'imports': imports,
                            'class_name': entry.get('class_name', None)
                        }
                        
                        # Add to index with timeout protection
                        self.pipeline.update_index(
                            func_ast=node,
                            file_path=file_path,
                            function_name=func_name,
                            line_number=line_number,
                            context=context
                        )
                        total_functions += 1
                        
                        # More frequent progress updates and timeout check
                        if total_functions % 50 == 0:
                            elapsed = time.time() - start_time
                            self.logger.info(f"Indexed {total_functions}/{total_to_process} functions ({elapsed:.1f}s elapsed)")
                            
                            # Timeout protection - abort if taking too long
                            if elapsed > 25:  # Give 25 seconds for indexing
                                self.logger.warning(f"Indexing timeout after {elapsed:.1f}s, stopping at {total_functions} functions")
                                break
                                
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to index {func_name} at {file_path}:{line_number}: {e}"
                        )
                        skipped_functions += 1
                        
                # Check overall timeout
                if time.time() - start_time > 25:
                    break
            
            elapsed = time.time() - start_time
            self.logger.info(f"Index build complete: {total_functions} indexed, {skipped_functions} skipped in {elapsed:.1f}s")
            
        except Exception as e:
            elapsed = time.time() - start_time
            self.logger.error(f"Index building failed after {elapsed:.1f}s: {e}")
            raise
    
    def _extract_file_imports(self, tree: Optional[ast.AST]) -> List[str]:
        """Extract import names from AST tree."""
        if not tree:
            return []
        
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
                for alias in node.names:
                    imports.append(alias.name)
        
        return imports
    
    def _find_duplicates(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Find duplicate functions using the index.
        
        Args:
            ctx: Analysis context
            
        Returns:
            List of duplicate issues
        """
        issues = []
        processed_pairs: Set[Tuple[str, str]] = set()
        
        # Find all high-similarity pairs
        duplicates = self.pipeline.find_duplicates(
            min_similarity=self.min_similarity_for_issue
        )
        
        # Group duplicates by equivalence classes
        duplicate_groups = self._group_duplicates(duplicates)
        
        for group in duplicate_groups:
            if len(group) < 2:
                continue
            
            # Get metadata for all functions in group
            functions_info = []
            for func_id in group:
                metadata = self.index.get_metadata(func_id)
                if metadata:
                    functions_info.append({
                        'id': func_id,
                        'file': metadata.file_path,
                        'name': metadata.function_name,
                        'line': metadata.line_number
                    })
            
            if len(functions_info) < 2:
                continue
            
            # Create issue for the group
            primary = functions_info[0]
            duplicates_list = functions_info[1:]
            
            issue = Issue(
                kind="catalytic_duplicate_group",
                message=f"Found {len(functions_info)} identical functions: "
                       f"{primary['name']} and {len(duplicates_list)} others",
                severity=self._compute_severity(len(functions_info)),
                file=primary['file'],
                line=primary['line'],
                symbol=primary['name'],
                evidence={
                    'primary': primary,
                    'duplicates': duplicates_list,
                    'count': len(functions_info),
                    'detection_method': 'catalytic_hypervector'
                },
                suggestions=self._get_suggestions(functions_info)
            )
            
            issues.append(issue)
        
        # Also check for near-duplicates with structural differences
        self._add_near_duplicate_issues(ctx, issues, processed_pairs)
        
        return issues
    
    def _group_duplicates(self, duplicates: List[Tuple[str, str, float]]) -> List[Set[str]]:
        """
        Group duplicates into equivalence classes.
        
        Args:
            duplicates: List of (id1, id2, similarity) tuples
            
        Returns:
            List of duplicate groups
        """
        # Build adjacency list
        graph = defaultdict(set)
        for id1, id2, _ in duplicates:
            graph[id1].add(id2)
            graph[id2].add(id1)
        
        # Find connected components
        visited = set()
        groups = []
        
        for node in graph:
            if node not in visited:
                # BFS to find component
                component = set()
                queue = [node]
                
                while queue:
                    current = queue.pop(0)
                    if current in visited:
                        continue
                    
                    visited.add(current)
                    component.add(current)
                    
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            queue.append(neighbor)
                
                if component:
                    groups.append(component)
        
        return groups
    
    def _add_near_duplicate_issues(self, ctx: AnalysisContext, 
                                  issues: List[Issue],
                                  processed: Set[Tuple[str, str]]) -> None:
        """
        Add issues for near-duplicates (high similarity but not exact).
        
        Args:
            ctx: Analysis context
            issues: List to append issues to
            processed: Set of already processed pairs
        """
        # Sample some functions for near-duplicate detection
        functions = ctx.symbol_table.functions
        sample_size = min(100, len(functions))
        
        if sample_size < 2:
            return
        
        # Sample functions
        import random
        sampled_funcs = random.sample(list(functions.items()), sample_size)
        
        for func_name, entries in sampled_funcs:
            for entry in entries[:1]:  # Check first entry only
                node = entry['node']
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    continue
                
                # Query for similar functions
                results = self.pipeline.query_function(node, top_k=5)
                
                for result in results:
                    if result.similarity_score >= 0.85 and not result.is_exact_match:
                        pair = tuple(sorted([f"{entry['file']}:{func_name}", 
                                           result.function_id]))
                        
                        if pair not in processed:
                            processed.add(pair)
                            
                            issue = Issue(
                                kind="catalytic_near_duplicate",
                                message=f"Near-duplicate functions: {func_name} and "
                                       f"{result.function_name} "
                                       f"(similarity: {result.similarity_score:.2f})",
                                severity=2,
                                file=entry['file'],
                                line=entry['lineno'],
                                symbol=func_name,
                                evidence={
                                    'similar_to': {
                                        'file': result.file_path,
                                        'name': result.function_name,
                                        'line': result.line_number
                                    },
                                    'hv_similarity': result.hv_similarity,
                                    'ast_similarity': result.ast_similarity,
                                    'confidence': result.confidence
                                },
                                suggestions=[
                                    "Consider merging these similar functions",
                                    "Extract common logic to a shared function",
                                    "Review if both implementations are necessary"
                                ]
                            )
                            
                            issues.append(issue)
    
    def _compute_severity(self, num_duplicates: int) -> int:
        """
        Compute severity based on number of duplicates.
        
        Args:
            num_duplicates: Number of duplicate functions
            
        Returns:
            Severity level (1-5)
        """
        if num_duplicates >= 5:
            return 5
        elif num_duplicates >= 3:
            return 4
        else:
            return 3
    
    def _get_suggestions(self, functions_info: List[Dict[str, Any]]) -> List[str]:
        """
        Generate suggestions for duplicate groups.
        
        Args:
            functions_info: Information about duplicate functions
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Check if duplicates are in same file
        files = {f['file'] for f in functions_info}
        
        if len(files) == 1:
            suggestions.append(
                "Remove duplicate implementations in the same file"
            )
        else:
            suggestions.append(
                "Extract the common function to a shared module"
            )
        
        suggestions.extend([
            f"Keep only one implementation and import it where needed",
            f"Consider if all {len(functions_info)} copies are necessary",
            "Use inheritance or composition to share functionality"
        ])
        
        return suggestions