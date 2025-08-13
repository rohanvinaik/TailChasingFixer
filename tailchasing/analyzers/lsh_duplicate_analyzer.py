"""
LSH-accelerated duplicate function analyzer.

Uses MinHash with Locality Sensitive Hashing to reduce duplicate detection
from O(n²) to O(n·k) where k << n.
"""

import ast
import logging
import os
import time
from pathlib import Path
from typing import List, Dict, Optional, Any, Set, Tuple
from collections import defaultdict

from .base import Analyzer, AnalysisContext
from ..core.issues import Issue
from ..semantic.lsh_index import (
    FunctionRecord,
    LSHParams,
    FeatureConfig,
    precluster_for_comparison,
    create_function_records
)
try:
    from ..semantic.progressive_encoder import (
        ProgressiveParams,
        progressive_refine_lsh_pairs
    )
    PROGRESSIVE_AVAILABLE = True
except ImportError:
    PROGRESSIVE_AVAILABLE = False


class LSHDuplicateAnalyzer(Analyzer):
    """
    Fast duplicate detection using MinHash LSH pre-clustering.
    
    This analyzer:
    1. Extracts shingles from AST (3-grams), imports, and call patterns
    2. Computes MinHash signatures for each function
    3. Uses LSH banding to find candidate pairs
    4. Performs detailed comparison only on candidates
    
    Complexity: O(n·k) where k is the average bucket size (typically << n)
    """
    
    name = "lsh_duplicates"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the analyzer with configuration."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # LSH parameters
        lsh_config = self.config.get('lsh', {})
        self.lsh_params = LSHParams(
            num_hashes=lsh_config.get('num_hashes', 100),
            bands=lsh_config.get('bands', 20),
            rows_per_band=lsh_config.get('rows_per_band', 5),
            seed=lsh_config.get('seed', 0x5EED_1DEE)
        )
        
        # Feature extraction config
        self.feature_config = FeatureConfig(
            use_ast_3grams=lsh_config.get('use_ast_3grams', True),
            use_imports=lsh_config.get('use_imports', True),
            use_call_patterns=lsh_config.get('use_call_patterns', True),
            max_nodes=lsh_config.get('max_nodes', 50_000)
        )
        
        # Similarity thresholds
        self.exact_match_threshold = lsh_config.get('exact_match_threshold', 0.95)
        self.semantic_match_threshold = lsh_config.get('semantic_match_threshold', 0.85)
        
        # Performance limits
        self.max_comparisons = lsh_config.get('max_comparisons', 10_000)
        self.min_function_size = lsh_config.get('min_function_size', 3)  # Min lines
        
        # Timeout configuration from environment or config
        self.timeout_seconds = float(os.getenv("TAILCHASING_ANALYZER_TIMEOUT_SEC", 
                                              lsh_config.get('timeout_seconds', 120.0)))
        
        # Progressive refinement settings
        self.use_progressive = lsh_config.get('use_progressive', False) and PROGRESSIVE_AVAILABLE
        if self.use_progressive:
            self.progressive_params = ProgressiveParams(
                min_l2_jaccard=lsh_config.get('progressive_l2_threshold', 0.35),
                min_l3_similarity=lsh_config.get('progressive_l3_threshold', 0.25)
            )
        
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run LSH-accelerated duplicate detection.
        
        Args:
            ctx: Analysis context with AST index
            
        Returns:
            List of duplicate function issues
        """
        start_time = time.time()
        
        # Check if enabled
        if not self.config.get('enabled', True):
            return []
            
        # Convert AST index to function records
        self.logger.info("Building function records from AST index...")
        function_records = self._create_function_records(ctx)
        
        if not function_records:
            self.logger.info("No functions to analyze")
            return []
            
        self.logger.info(f"Analyzing {len(function_records)} functions with LSH")
        
        # Check timeout before LSH clustering
        if self.timeout_seconds > 0 and (time.time() - start_time) > self.timeout_seconds:
            self.logger.warning(f"Timeout reached before LSH clustering ({time.time() - start_time:.1f}s)")
            return []
        
        # Run LSH pre-clustering
        self.logger.info("Running MinHash LSH pre-clustering...")
        candidate_pairs, stats = precluster_for_comparison(
            function_records,
            feat_cfg=self.feature_config,
            params=self.lsh_params
        )
        
        # Log statistics
        self.logger.info(
            f"LSH Stats: {stats.total_functions} functions, "
            f"{stats.non_singleton_buckets} non-singleton buckets, "
            f"{len(candidate_pairs)} candidate pairs"
        )
        
        # Check timeout before progressive refinement
        if self.timeout_seconds > 0 and (time.time() - start_time) > self.timeout_seconds:
            self.logger.warning(f"Timeout reached before progressive refinement ({time.time() - start_time:.1f}s)")
            return []
        
        # Apply progressive refinement if enabled
        if self.use_progressive and candidate_pairs:
            self.logger.info("Applying 3-level progressive refinement...")
            
            # Build function map for progressive encoder
            func_map = {fr.id: fr for fr in function_records}
            
            # Refine through L2/L3 progressive encoding
            refined_pairs, prog_stats = progressive_refine_lsh_pairs(
                candidate_pairs,
                func_map,
                params=self.progressive_params
            )
            
            self.logger.info(
                f"Progressive refinement: {len(candidate_pairs)} → "
                f"{prog_stats.l2_screened_pairs} (L2) → "
                f"{prog_stats.l3_screened_pairs} (L3) → "
                f"{prog_stats.final_pairs} final"
            )
            
            candidate_pairs = refined_pairs
        
        # Safety check on number of comparisons
        if len(candidate_pairs) > self.max_comparisons:
            self.logger.warning(
                f"Too many candidate pairs ({len(candidate_pairs)}), "
                f"limiting to {self.max_comparisons}"
            )
            candidate_pairs = candidate_pairs[:self.max_comparisons]
        
        # Perform detailed comparison on candidates
        issues = self._compare_candidates(candidate_pairs, function_records, ctx)
        
        elapsed = time.time() - start_time
        self.logger.info(
            f"LSH duplicate detection complete in {elapsed:.2f}s: "
            f"{len(issues)} duplicates found from {len(candidate_pairs)} candidates"
        )
        
        return issues
    
    def _create_function_records(self, ctx: AnalysisContext) -> List[FunctionRecord]:
        """
        Create function records from the analysis context.
        
        Args:
            ctx: Analysis context
            
        Returns:
            List of function records
        """
        records = []
        
        for file_path, tree in ctx.ast_index.items():
            # Get source code
            if file_path in ctx.source_cache:
                source_lines = ctx.source_cache[file_path]
                source = '\n'.join(source_lines) if isinstance(source_lines, list) else source_lines
            else:
                try:
                    source = Path(file_path).read_text(encoding='utf-8', errors='ignore')
                except Exception as e:
                    self.logger.warning(f"Could not read source for {file_path}: {e}")
                    continue
            
            # Extract functions
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip small functions
                    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                        size = node.end_lineno - node.lineno + 1
                        if size < self.min_function_size:
                            continue
                    
                    # Skip test functions if configured
                    if self._should_skip_function(node.name, file_path):
                        continue
                    
                    # Create unique ID
                    func_id = f"{file_path}::{node.name}@{node.lineno}"
                    
                    # Create record
                    record = FunctionRecord(
                        id=func_id,
                        source=source,
                        node=node,
                        file=file_path
                    )
                    records.append(record)
        
        return records
    
    def _should_skip_function(self, func_name: str, file_path: str) -> bool:
        """
        Check if a function should be skipped.
        
        Args:
            func_name: Function name
            file_path: File path
            
        Returns:
            True if function should be skipped
        """
        # Skip __init__ methods unless configured
        if func_name == '__init__' and not self.config.get('include_init', False):
            return True
        
        # Skip test functions unless configured
        if not self.config.get('include_tests', False):
            if 'test' in file_path.lower() or func_name.startswith('test_'):
                return True
        
        # Skip private functions unless configured
        if func_name.startswith('_') and not self.config.get('include_private', False):
            return True
        
        return False
    
    def _compare_candidates(
        self,
        candidate_pairs: List[Tuple[str, str]],
        function_records: List[FunctionRecord],
        ctx: AnalysisContext
    ) -> List[Issue]:
        """
        Perform detailed comparison on candidate pairs.
        
        Args:
            candidate_pairs: List of (func_id1, func_id2) pairs
            function_records: All function records
            ctx: Analysis context
            
        Returns:
            List of duplicate issues
        """
        # Build lookup index
        func_lookup = {fr.id: fr for fr in function_records}
        
        issues = []
        duplicates_found = defaultdict(set)  # Track groups of duplicates
        
        for func_id1, func_id2 in candidate_pairs:
            # Skip if already found as duplicates
            if func_id2 in duplicates_found.get(func_id1, set()):
                continue
            
            # Get function records
            fr1 = func_lookup.get(func_id1)
            fr2 = func_lookup.get(func_id2)
            
            if not fr1 or not fr2:
                continue
            
            # Perform detailed comparison
            similarity = self._compute_detailed_similarity(fr1, fr2)
            
            if similarity >= self.exact_match_threshold:
                # Track duplicate group
                duplicates_found[func_id1].add(func_id2)
                duplicates_found[func_id2].add(func_id1)
                
                # Create issue
                issue = self._create_duplicate_issue(fr1, fr2, similarity, "exact")
                issues.append(issue)
                
            elif similarity >= self.semantic_match_threshold:
                # Track duplicate group
                duplicates_found[func_id1].add(func_id2)
                duplicates_found[func_id2].add(func_id1)
                
                # Create issue
                issue = self._create_duplicate_issue(fr1, fr2, similarity, "semantic")
                issues.append(issue)
        
        return issues
    
    def _compute_detailed_similarity(
        self,
        fr1: FunctionRecord,
        fr2: FunctionRecord
    ) -> float:
        """
        Compute detailed similarity between two functions.
        
        This is where you can plug in more sophisticated comparison
        (e.g., using the hypervector system or AST comparison).
        
        Args:
            fr1: First function record
            fr2: Second function record
            
        Returns:
            Similarity score [0, 1]
        """
        # For now, use a simple AST comparison
        # You can enhance this with hypervector comparison
        
        # Get AST dumps
        ast1 = ast.dump(fr1.node)
        ast2 = ast.dump(fr2.node)
        
        # Quick exact match check
        if ast1 == ast2:
            return 1.0
        
        # Compute Jaccard similarity of AST tokens
        tokens1 = set(ast1.split())
        tokens2 = set(ast2.split())
        
        if not tokens1 or not tokens2:
            return 0.0
        
        intersection = len(tokens1 & tokens2)
        union = len(tokens1 | tokens2)
        
        return intersection / union if union > 0 else 0.0
    
    def _create_duplicate_issue(
        self,
        fr1: FunctionRecord,
        fr2: FunctionRecord,
        similarity: float,
        match_type: str
    ) -> Issue:
        """
        Create a duplicate function issue.
        
        Args:
            fr1: First function record
            fr2: Second function record
            similarity: Similarity score
            match_type: Type of match (exact/semantic)
            
        Returns:
            Issue object
        """
        # Parse function IDs
        file1, func_loc1 = fr1.id.split('::', 1)
        file2, func_loc2 = fr2.id.split('::', 1)
        func_name1, line1 = func_loc1.split('@')
        func_name2, line2 = func_loc2.split('@')
        
        # Determine severity based on match type
        if match_type == "exact":
            severity = 3  # High
            kind = "duplicate_function"
            message = f"Exact duplicate: {func_name1} identical to {func_name2} in {Path(file2).name}"
        else:
            severity = 2  # Medium
            kind = "semantic_duplicate_function"
            message = f"Semantic duplicate: {func_name1} similar to {func_name2} in {Path(file2).name} (similarity: {similarity:.2f})"
        
        return Issue(
            kind=kind,
            message=message,
            severity=severity,
            file=file1,
            line=int(line1),
            confidence=similarity,
            evidence={
                'duplicate_location': f"{file2}:{line2}",
                'function_name': func_name1,
                'duplicate_name': func_name2,
                'similarity': similarity,
                'match_type': match_type
            },
            suggestions=[
                f"Consider extracting common functionality to a shared module",
                f"Remove duplicate and import from {Path(file2).stem}"
            ]
        )