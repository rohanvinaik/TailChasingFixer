"""
Hierarchical semantic hypervector analyzer for large codebases.

This analyzer implements a 4-step hierarchical strategy:
1. Group functions by module/file proximity
2. Build fast signatures (32-64 bits) for each group
3. Generate candidate pairs via LSH within groups
4. Progressive comparison (quick → medium → full HV)

Designed for O(n·log(n)) complexity on 100k+ function codebases.
"""

import ast
import logging
import math
import os
import random
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Sequence, DefaultDict, Union

from .base import Analyzer, AnalysisContext
from ..core.issues import Issue

# Import LSH and progressive encoder components
try:
    from ..semantic.lsh_index import (
        FunctionRecord,
        LSHParams,
        FeatureConfig,
        precluster_for_comparison,
    )
    LSH_AVAILABLE = True
except ImportError:
    LSH_AVAILABLE = False
    # Fallback classes for MinHash compatibility
    @dataclass(frozen=True)
    class FunctionRecord:
        id: str
        source: str
        node: ast.AST
        file: Optional[str] = None
    
    @dataclass
    class LSHParams:
        num_hashes: int = 64
        bands: int = 16
        rows_per_band: int = 4
        seed: int = 0x5EED_1DEE

try:
    from ..semantic.progressive_encoder import (
        encode_level1,
        encode_level2,
        encode_level3,
        BitSig,
        ProgressiveParams,
    )
    PROGRESSIVE_AVAILABLE = True
except ImportError:
    PROGRESSIVE_AVAILABLE = False

# Fallback to lightweight MinHash+LSH if the semantic.index LSH isn't available
MINHASH_AVAILABLE = False
try:
    from ..semantic.minhash import MinHashIndex  # self-contained LSH
    MINHASH_AVAILABLE = True
except Exception:
    MINHASH_AVAILABLE = False

# Process-level cache for precomputed Level-1/2 signatures
_SIGNATURE_CACHE: Dict[str, Dict[str, Any]] = {
    'level1': {},  # id -> BitSig
    'level2': {},  # id -> BitSig  
    'level3': {},  # id -> BitSig
}


@dataclass
class HierarchicalStats:
    """Statistics for hierarchical analysis."""
    total_functions: int = 0
    total_modules: int = 0
    avg_module_size: float = 0.0
    l1_candidates: int = 0
    l2_candidates: int = 0
    l3_candidates: int = 0
    final_duplicates: int = 0
    time_grouping: float = 0.0
    time_signatures: float = 0.0
    time_lsh: float = 0.0
    time_refinement: float = 0.0


class SemanticHVAnalyzer(Analyzer):
    """
    Hierarchical semantic analysis using hypervector encoding.
    
    This analyzer scales to very large codebases (100k+ functions) by:
    1. Grouping functions by module proximity to reduce search space
    2. Using fast bit signatures for initial filtering within groups
    3. Applying LSH for candidate generation
    4. Progressive refinement through multiple encoding levels
    
    Complexity: O(n·log(n)) average case, O(n²/m) worst case where m is module count
    """
    
    name = "semantic_hv"
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the hierarchical analyzer."""
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Check dependencies
        if not LSH_AVAILABLE:
            if MINHASH_AVAILABLE:
                self.logger.info("LSH index not available, using MinHash LSH fallback")
            else:
                self.logger.warning("No LSH backend available, falling back to simpler analysis")
        if not PROGRESSIVE_AVAILABLE:
            self.logger.warning("Progressive encoder not available, using basic comparison")
        
        # Hierarchical grouping parameters
        self.max_group_size = config.get("max_group_size", 1000) if config else 1000
        self.min_group_size = config.get("min_group_size", 10) if config else 10
        
        # LSH parameters for within-group analysis
        self.lsh_params = LSHParams(
            num_hashes=config.get("num_hashes", 64) if config else 64,
            bands=config.get("bands", 16) if config else 16,
            rows_per_band=config.get("rows_per_band", 4) if config else 4,
        )
        
        # Progressive encoding parameters
        self.progressive_params = ProgressiveParams(
            level1_bits=32,
            level2_bits=128,
            level3_bits=1024,
            min_l2_jaccard=config.get("l2_threshold", 0.4) if config else 0.4,
            min_l3_similarity=config.get("l3_threshold", 0.7) if config else 0.7,
        )
        
        # Similarity thresholds
        self.exact_threshold = config.get("exact_threshold", 0.95) if config else 0.95
        self.semantic_threshold = config.get("semantic_threshold", 0.85) if config else 0.85
        
        # Performance limits
        self.max_comparisons = config.get("max_comparisons", 10000) if config else 10000
        # Read timeout from environment variable or config, with fallback
        self.timeout_seconds = float(os.getenv("TAILCHASING_ANALYZER_TIMEOUT_SEC", 
                                              config.get("timeout_seconds", 30.0) if config else 30.0))
        self.max_bucket_size = config.get("max_bucket_size", 100) if config else 100
        self.enable_cross_module = config.get("enable_cross_module", False) if config else False
        
        # Group timeout budget (configurable per group)
        self.group_timeout_seconds = float(os.getenv("TAILCHASING_GROUP_TIMEOUT_SEC", 8.0))
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """
        Run hierarchical semantic analysis.
        
        Args:
            ctx: Analysis context with AST index
            
        Returns:
            List of semantic duplicate issues
        """
        if not self.config.get("enabled", True):
            return []
        
        start_time = time.time()
        stats = HierarchicalStats()
        
        # Step 1: Group functions by module
        self.logger.info("Step 1: Grouping functions by module proximity...")
        t0 = time.time()
        module_groups = self._group_by_module(ctx)
        stats.time_grouping = time.time() - t0
        stats.total_modules = len(module_groups)
        stats.total_functions = sum(len(g) for g in module_groups.values())
        
        if stats.total_functions == 0:
            self.logger.info("No functions to analyze")
            return []
        
        stats.avg_module_size = stats.total_functions / max(1, stats.total_modules)
        self.logger.info(
            f"Grouped {stats.total_functions} functions into {stats.total_modules} modules "
            f"(avg size: {stats.avg_module_size:.1f})"
        )
        
        # Step 2: Build fast signatures for each group
        self.logger.info("Step 2: Building fast signatures...")
        t0 = time.time()
        group_signatures = self._build_group_signatures(module_groups)
        stats.time_signatures = time.time() - t0
        
        # Step 3: Process each group with LSH
        self.logger.info("Step 3: Running LSH within groups...")
        t0 = time.time()
        candidate_pairs = []
        
        for module_path, functions in module_groups.items():
            # Skip tiny modules entirely - no point in LSH overhead
            if len(functions) < 2:
                continue
            
            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self.timeout_seconds:
                self.logger.warning(f"Timeout reached ({elapsed:.1f}s), stopping analysis")
                break
            
            # Run LSH on this group
            group_pairs = self._process_group_with_lsh(functions, module_path)
            candidate_pairs.extend(group_pairs)
            stats.l1_candidates += len(group_pairs)
            
            # Safety check
            if len(candidate_pairs) > self.max_comparisons:
                self.logger.warning(
                    f"Too many candidates ({len(candidate_pairs)}), stopping early"
                )
                break
        
        stats.time_lsh = time.time() - t0
        self.logger.info(f"LSH generated {len(candidate_pairs)} candidate pairs")
        
        # Step 4: Progressive refinement
        self.logger.info("Step 4: Progressive refinement of candidates...")
        t0 = time.time()
        issues = self._progressive_refinement(candidate_pairs, ctx, stats)
        stats.time_refinement = time.time() - t0
        stats.final_duplicates = len(issues)
        
        # Log final statistics
        total_time = time.time() - start_time
        self.logger.info(
            f"Hierarchical analysis complete in {total_time:.2f}s:\n"
            f"  - Grouping: {stats.time_grouping:.2f}s\n"
            f"  - Signatures: {stats.time_signatures:.2f}s\n"
            f"  - LSH: {stats.time_lsh:.2f}s\n"
            f"  - Refinement: {stats.time_refinement:.2f}s\n"
            f"  - Candidates: {stats.l1_candidates} → {stats.l2_candidates} → {stats.final_duplicates}"
        )
        
        return issues
    
    def _group_by_module(self, ctx: AnalysisContext) -> Dict[str, List[FunctionRecord]]:
        """
        Group functions by module/file proximity.
        
        Uses a hierarchical clustering approach:
        1. Group by exact file
        2. Group by parent directory
        3. Split large groups if needed
        
        Args:
            ctx: Analysis context
            
        Returns:
            Dictionary mapping module path to list of functions
        """
        module_groups = defaultdict(list)
        
        for file_path, tree in ctx.ast_index.items():
            # Get source code
            if file_path in ctx.source_cache:
                source_lines = ctx.source_cache[file_path]
                source = '\n'.join(source_lines) if isinstance(source_lines, list) else source_lines
            else:
                try:
                    source = Path(file_path).read_text(encoding='utf-8', errors='ignore')
                except Exception:
                    continue
            
            # Extract functions from this file
            file_functions = []
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    # Skip small functions
                    if hasattr(node, 'end_lineno') and hasattr(node, 'lineno'):
                        size = node.end_lineno - node.lineno + 1
                        if size < 3:  # Min 3 lines
                            continue
                    
                    func_id = f"{file_path}::{node.name}@{node.lineno}"
                    record = FunctionRecord(
                        id=func_id,
                        source=source,
                        node=node,
                        file=file_path
                    )
                    file_functions.append(record)
            
            if file_functions:
                # Determine module key (use parent directory for grouping)
                path = Path(file_path)
                if len(file_functions) > self.max_group_size:
                    # Split large files into chunks
                    for i in range(0, len(file_functions), self.max_group_size):
                        chunk_key = f"{path.parent}/{path.stem}_chunk{i//self.max_group_size}"
                        module_groups[chunk_key].extend(
                            file_functions[i:i+self.max_group_size]
                        )
                else:
                    module_key = str(path.parent)
                    module_groups[module_key].extend(file_functions)
        
        # Merge small groups with nearby groups
        final_groups = {}
        small_groups = []
        
        for module_path, functions in module_groups.items():
            if len(functions) >= self.min_group_size:
                final_groups[module_path] = functions
            else:
                small_groups.append((module_path, functions))
        
        # Merge small groups
        if small_groups:
            merged = []
            for path, funcs in small_groups:
                merged.extend(funcs)
                if len(merged) >= self.min_group_size:
                    final_groups[f"merged_{len(final_groups)}"] = merged
                    merged = []
            if merged:
                final_groups["merged_final"] = merged
        
        return final_groups
    
    def _build_group_signatures(
        self, 
        module_groups: Dict[str, List[FunctionRecord]]
    ) -> Dict[str, Dict[str, BitSig]]:
        """
        Build fast signatures for all functions in each group.
        
        Args:
            module_groups: Functions grouped by module
            
        Returns:
            Nested dict: module -> function_id -> signature
        """
        if not PROGRESSIVE_AVAILABLE:
            return {}
        
        group_sigs = {}
        
        for module_path, functions in module_groups.items():
            module_sigs = {}
            for func in functions:
                # Use Level 1 encoder for fast signatures
                sig = encode_level1(func, width=32)
                module_sigs[func.id] = sig
            group_sigs[module_path] = module_sigs
        
        return group_sigs
    
    def _process_group_with_lsh(
        self, 
        functions: List[FunctionRecord],
        module_path: str
    ) -> List[Tuple[FunctionRecord, FunctionRecord]]:
        """
        Process a group of functions with LSH.
        
        Args:
            functions: Functions in this group
            module_path: Module identifier
            
        Returns:
            List of candidate pairs
        """
        if len(functions) < 2:
            return []
        
        import time
        group_start = time.time()
        GROUP_BUDGET = self.group_timeout_seconds  # Use configurable timeout
        
        # Apply intelligent sampling for large groups
        sampled_functions = functions
        if len(functions) > 500:  # Large group threshold
            self.logger.info(f"Large group in {module_path} ({len(functions)} functions), applying intelligent sampling")
            sampled_functions = self.sample_functions_intelligently(functions, max_sample=500)
            self.logger.info(f"Sampled {len(sampled_functions)} functions from {len(functions)}")
        
        # --- Run LSH pre-clustering ---
        pairs: List[Tuple[str, str]] = []

        if LSH_AVAILABLE:
            # existing path (precluster_for_comparison)
            pairs, _stats = precluster_for_comparison(
                sampled_functions,
                params=self.lsh_params,
                feat_cfg=FeatureConfig(use_ast_3grams=True, use_imports=True, use_call_patterns=True),
            )
        elif MINHASH_AVAILABLE:
            # fallback path using MinHashIndex
            idx = MinHashIndex(num_perm=64, threshold=0.5)  # Fixed 64 for speed
            idx.add_functions(sampled_functions)

            # --- FAST candidate build from LSH buckets (no per-item queries) ---
            pairs = []
            seen = set()
            MAX_BUCKET = getattr(self, "max_bucket_size", 100)
            
            # Access buckets from the index's LSH (safe: it's an internal struct we control)
            bucket_count = 0
            for id_list in idx.lsh._buckets.values():  # list[str]
                if len(id_list) < 2:
                    continue
                bucket_count += 1
                uniq = sorted(set(id_list))
                
                # slice big buckets to prevent combinatorial explosion
                for s in range(0, len(uniq), MAX_BUCKET):
                    chunk = uniq[s:s+MAX_BUCKET]
                    if len(chunk) < 2:
                        continue
                    
                    # pairwise combinations in chunk
                    for a_i in range(len(chunk)):
                        for b_i in range(a_i + 1, len(chunk)):
                            a, b = chunk[a_i], chunk[b_i]
                            key = (a, b) if a <= b else (b, a)
                            if key in seen:
                                continue
                            seen.add(key)
                            pairs.append(key)
            
            self.logger.info(f"LSH buckets: {bucket_count}, candidate pairs: {len(pairs)} after dedup")
        else:
            # last resort: no LSH at all
            self.logger.warning("No LSH backend available; skipping LSH pre-clustering for this group")
            pairs = []
        
        # Check time budget after candidate generation
        if time.time() - group_start > GROUP_BUDGET:
            self.logger.info(f"Group {module_path} exceeded budget; skipping detailed screening")
            return []
        
        # Convert ID pairs back to function records
        func_map = {f.id: f for f in sampled_functions}
        result = []
        
        for id1, id2 in pairs:
            f1 = func_map.get(id1)
            f2 = func_map.get(id2)
            if f1 and f2:
                result.append((f1, f2))
        
        self.logger.debug(
            f"Module {module_path}: {len(functions)} functions → {len(result)} candidates"
        )
        
        return result
    
    def _progressive_refinement(
        self,
        candidate_pairs: List[Tuple[FunctionRecord, FunctionRecord]],
        ctx: AnalysisContext,
        stats: HierarchicalStats
    ) -> List[Issue]:
        """
        Progressively refine candidate pairs through multiple levels.
        
        Args:
            candidate_pairs: Initial candidate pairs
            ctx: Analysis context
            stats: Statistics to update
            
        Returns:
            List of duplicate issues
        """
        if not PROGRESSIVE_AVAILABLE:
            # Fallback to simple comparison
            return self._simple_comparison(candidate_pairs, ctx)
        
        issues = []
        l2_candidates = []
        
        # Level 1 gating (compute L2 only if L1 ≥ 0.35)
        L1_GATE = 0.35
        
        for f1, f2 in candidate_pairs:
            # Get or compute L1 signatures (from cache)
            if f1.id not in _SIGNATURE_CACHE['level1']:
                _SIGNATURE_CACHE['level1'][f1.id] = encode_level1(f1, width=32)
            if f2.id not in _SIGNATURE_CACHE['level1']:
                _SIGNATURE_CACHE['level1'][f2.id] = encode_level1(f2, width=32)
            
            # Check L1 similarity first - only proceed if promising
            l1_similarity = _SIGNATURE_CACHE['level1'][f1.id].jaccard(_SIGNATURE_CACHE['level1'][f2.id])
            if l1_similarity < L1_GATE:
                continue  # Skip L2 computation entirely
            
            # Get or compute L2 signatures (from cache)
            if f2.id not in _SIGNATURE_CACHE['level2']:
                _SIGNATURE_CACHE['level2'][f2.id] = encode_level2(f2, width=128)
            if f1.id not in _SIGNATURE_CACHE['level2']:
                _SIGNATURE_CACHE['level2'][f1.id] = encode_level2(f1, width=128)
            
            # Check L2 similarity
            l2_similarity = _SIGNATURE_CACHE['level2'][f1.id].jaccard(_SIGNATURE_CACHE['level2'][f2.id])
            if l2_similarity >= self.progressive_params.min_l2_jaccard:
                l2_candidates.append((f1, f2, l2_similarity))
        
        stats.l2_candidates = len(l2_candidates)
        
        # Log cache efficiency
        if self.config.get('verbose_logging', False):
            cache_sizes = {k: len(v) for k, v in _SIGNATURE_CACHE.items()}
            self.logger.debug(f"Signature cache sizes: {cache_sizes}")
        
        # Level 3 refinement (full hypervector)
        for f1, f2, l2_sim in l2_candidates:
            # Get or compute L3 signatures (from cache)
            if f1.id not in _SIGNATURE_CACHE['level3']:
                _SIGNATURE_CACHE['level3'][f1.id] = encode_level3(f1, width=1024)
            if f2.id not in _SIGNATURE_CACHE['level3']:
                _SIGNATURE_CACHE['level3'][f2.id] = encode_level3(f2, width=1024)
            
            # Check L3 similarity
            similarity = _SIGNATURE_CACHE['level3'][f1.id].jaccard(_SIGNATURE_CACHE['level3'][f2.id])
            
            if similarity >= self.exact_threshold:
                issue = self._create_issue(f1, f2, similarity, "exact")
                issues.append(issue)
            elif similarity >= self.semantic_threshold:
                issue = self._create_issue(f1, f2, similarity, "semantic")
                issues.append(issue)
        
        stats.l3_candidates = len(issues)
        
        return issues
    
    def _simple_comparison(
        self,
        candidate_pairs: List[Tuple[FunctionRecord, FunctionRecord]],
        ctx: AnalysisContext
    ) -> List[Issue]:
        """
        Fallback simple comparison when progressive encoder is not available.
        
        Args:
            candidate_pairs: Candidate pairs to compare
            ctx: Analysis context
            
        Returns:
            List of duplicate issues
        """
        issues = []
        
        for f1, f2 in candidate_pairs:
            # Simple AST comparison
            ast1 = ast.dump(f1.node)
            ast2 = ast.dump(f2.node)
            
            if ast1 == ast2:
                similarity = 1.0
                issue = self._create_issue(f1, f2, similarity, "exact")
                issues.append(issue)
            else:
                # Token-based similarity
                tokens1 = set(ast1.split())
                tokens2 = set(ast2.split())
                
                if tokens1 and tokens2:
                    intersection = len(tokens1 & tokens2)
                    union = len(tokens1 | tokens2)
                    similarity = intersection / union if union > 0 else 0.0
                    
                    if similarity >= self.semantic_threshold:
                        issue = self._create_issue(f1, f2, similarity, "semantic")
                        issues.append(issue)
        
        return issues
    
    def _create_issue(
        self,
        f1: FunctionRecord,
        f2: FunctionRecord,
        similarity: float,
        match_type: str
    ) -> Issue:
        """
        Create a duplicate issue from two function records.
        
        Args:
            f1: First function
            f2: Second function
            similarity: Similarity score
            match_type: Type of match (exact/semantic)
            
        Returns:
            Issue object
        """
        # Parse function locations
        file1, func_loc1 = f1.id.split('::', 1)
        file2, func_loc2 = f2.id.split('::', 1)
        func_name1, line1 = func_loc1.split('@')
        func_name2, line2 = func_loc2.split('@')
        
        if match_type == "exact":
            severity = 3
            kind = "semantic_hv_exact"
            message = (
                f"Exact hypervector match: {func_name1} identical to "
                f"{func_name2} in {Path(file2).name}"
            )
        else:
            severity = 2
            kind = "semantic_hv_duplicate"
            message = (
                f"Semantic duplicate (HV): {func_name1} similar to "
                f"{func_name2} in {Path(file2).name} (similarity: {similarity:.2f})"
            )
        
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
                'match_type': match_type,
                'analyzer': 'semantic_hv'
            },
            suggestions=[
                f"Consider extracting common functionality to a shared module",
                f"Remove duplicate and import from {Path(file2).stem}",
                f"Use semantic analysis tools to understand the relationship"
            ]
        )
    
    # --- Intelligent Sampling Methods ---
    
    def sample_functions_intelligently(self, functions: Sequence[FunctionRecord], max_sample: int = 500) -> List[FunctionRecord]:
        """
        Adaptive sampler biased toward likely duplicates/near-duplicates.

        Priority weights (default):
          1) Name-similar groups .......... 40%
          2) Identical signature groups ... 30%
          3) High-complexity functions .... 20%
          4) Uniform random coverage ...... 10%
        """
        if not functions:
            return []
        
        # Cheap pre-trim for very large groups before any AST-heavy scoring
        HUGE = 5000  # or use LargeCodebaseConfig.huge_codebase_threshold if available
        if len(functions) > HUGE:
            # 1) Keep at most 200 by name groups (quick)
            g_name = self.group_by_name_similarity(functions)
            pre = self.sample_from_groups(g_name, n=200, picked_ids=set(), prefer_complex=False)
            # 2) Add up to 200 by signature (quick)
            g_sig = self.group_by_signature(functions)
            pre_ids = {f.id for f in pre}
            pre += self.sample_from_groups(g_sig, n=200, picked_ids=pre_ids, prefer_complex=False)
            # 3) Fill to ~max_sample with random (no cost)
            if len(pre) < max_sample:
                pool = [f for f in functions if f.id not in {f.id for f in pre}]
                need = min(max_sample - len(pre), max(0, len(pool)))
                if need:
                    pre.extend(random.sample(pool, need))
            # From here on, operate on the pre-trimmed set only
            functions = pre

        # Compute target counts (safe rounding with leftovers filled later)
        n1 = int(round(max_sample * 0.40))
        n2 = int(round(max_sample * 0.30))
        n3 = int(round(max_sample * 0.20))
        n4 = max_sample - (n1 + n2 + n3)

        picked: List[FunctionRecord] = []
        picked_ids: Set[str] = set()

        # Priority 1: name similarity groups
        name_groups = self.group_by_name_similarity(functions)
        picked.extend(self.sample_from_groups(name_groups, n=n1, picked_ids=picked_ids))

        # Priority 2: identical signatures
        sig_groups = self.group_by_signature(functions)
        picked.extend(self.sample_from_groups(sig_groups, n=n2, picked_ids=picked_ids))

        # Priority 3: high-complexity
        complex_funcs = self.get_high_complexity_functions(functions)
        for fr in complex_funcs:
            if len(picked) >= n1 + n2 + n3:
                break
            if fr.id not in picked_ids:
                picked.append(fr)
                picked_ids.add(fr.id)

        # Priority 4: random coverage
        remaining = max_sample - len(picked)
        if remaining > 0:
            # Avoid ValueError if population < remaining
            pool = [f for f in functions if f.id not in picked_ids]
            if pool:
                k = min(remaining, len(pool))
                picked.extend(random.sample(pool, k))

        # Final cap + stable order
        return picked[:max_sample]

    # ----------------------------
    # Helpers for adaptive sampling
    # ----------------------------

    def group_by_name_similarity(self, functions: Sequence[FunctionRecord]) -> List[List[FunctionRecord]]:
        """
        Bucket functions by cheap, robust name-similarity keys.
        Strategy:
          - split snake/camel to tokens
          - build multiple keys: primary token, sorted tokens, prefix of first token
          - union groups that share a key; emit groups with size >= 2
        """
        # Build inverted index: key -> set(idx)
        key_to_idxs: DefaultDict[str, Set[int]] = defaultdict(set)
        names: List[str] = []
        for i, fr in enumerate(functions):
            name = self._func_name(fr)
            names.append(name)
            toks = self._name_tokens(name)
            if not toks:
                continue
            # primary token
            key1 = f"tok:{toks[0]}"
            key_to_idxs[key1].add(i)
            # sorted token bag
            key2 = "bag:" + "|".join(sorted(toks))
            key_to_idxs[key2].add(i)
            # prefix (first token, first 4 chars)
            pref = toks[0][:4]
            if pref:
                key3 = f"pre:{pref}"
                key_to_idxs[key3].add(i)

        # Union similar sets into groups via simple overlapping-union
        # (small N per key; cheap).
        idxs_to_group: List[Set[int]] = []
        for _, s in key_to_idxs.items():
            if len(s) < 2:
                continue
            merged = False
            for g in idxs_to_group:
                if s & g:
                    g |= s
                    merged = True
                    break
            if not merged:
                idxs_to_group.append(set(s))

        groups: List[List[FunctionRecord]] = []
        for g in idxs_to_group:
            if len(g) >= 2:
                groups.append([functions[i] for i in sorted(g)])
        return groups

    def group_by_signature(self, functions: Sequence[FunctionRecord]) -> List[List[FunctionRecord]]:
        """
        Group by identical (coarse) signatures:
          - param count (incl. kwonly, *args, **kwargs)
          - simple return type category (if annotated)
          - kwonly presence flags
        """
        sig_map: DefaultDict[str, List[FunctionRecord]] = defaultdict(list)
        for fr in functions:
            sig = self._signature_fingerprint(fr)
            sig_map[sig].append(fr)
        return [grp for grp in sig_map.values() if len(grp) >= 2]

    def get_high_complexity_functions(self, functions: Sequence[FunctionRecord]) -> List[FunctionRecord]:
        """
        Rank by crude structural complexity (depth + control-flow).
        """
        scored: List[Tuple[float, FunctionRecord]] = []
        for fr in functions:
            try:
                node = fr.node
                d = self._ast_depth_sampling(node, max_nodes=self.progressive_params.max_nodes)
                c = self._control_flow_complexity_sampling(node, max_nodes=self.progressive_params.max_nodes)
                # Weighted sum; adjust weights if desired
                score = (0.6 * math.log2(max(d, 1))) + (0.4 * math.log2(max(c, 1)))
                scored.append((score, fr))
            except Exception:
                # In weird cases, push to bottom
                scored.append((0.0, fr))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [fr for _, fr in scored]

    def sample_from_groups(
        self,
        groups: Sequence[Sequence[FunctionRecord]],
        *,
        n: int,
        picked_ids: Optional[Set[str]] = None,
        prefer_complex: bool = True,
    ) -> List[FunctionRecord]:
        """
        Take ~n items from groups, proportionally by group size, avoiding duplicates.
        If prefer_complex=True, pick higher-complexity members within each group first.
        """
        if n <= 0 or not groups:
            return []

        picked: List[FunctionRecord] = []
        seen = picked_ids if picked_ids is not None else set()

        # Proportional quotas with at least 1 for large groups
        sizes = [len(g) for g in groups]
        weights = [max(1, s - 1) for s in sizes]  # slightly favor larger groups
        W = sum(weights)
        quotas = [max(1, int(round(n * (w / W)))) for w in weights] if W else [1] * len(groups)

        # Normalize total to n
        total = sum(quotas)
        if total > n:
            # Trim extras from smallest groups first
            over = total - n
            order = sorted(range(len(groups)), key=lambda i: sizes[i])  # shrink small groups first
            for i in order:
                if over <= 0:
                    break
                dec = min(over, max(0, quotas[i] - 1))
                quotas[i] -= dec
                over -= dec
        elif total < n:
            # Distribute remainder to largest groups
            rem = n - total
            order = sorted(range(len(groups)), key=lambda i: sizes[i], reverse=True)
            for i in order:
                if rem <= 0:
                    break
                quotas[i] += 1
                rem -= 1

        # Within each group, pick items (complexity-preferred or random)
        for grp, k in zip(groups, quotas):
            if k <= 0:
                continue
            cand = list(grp)
            if prefer_complex:
                # Reuse complexity scorer
                c_sorted = self.get_high_complexity_functions(cand)
                cand = c_sorted
            else:
                random.shuffle(cand)

            for fr in cand:
                if len(picked) >= n:
                    break
                if fr.id in seen:
                    continue
                picked.append(fr)
                seen.add(fr.id)

        return picked

    # ----------------------------
    # Local primitives (name/sig/complexity)
    # ----------------------------

    _CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")

    def _func_name(self, fr: FunctionRecord) -> str:
        n = fr.node
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return n.name
        return "anonymous"

    def _name_tokens(self, name: str) -> List[str]:
        snake = name.replace("__", "_").strip("_")
        parts: List[str] = []
        for p in snake.split("_"):
            if not p:
                continue
            parts.extend(self._CAMEL_SPLIT.split(p))
        toks = [t.lower() for t in parts if t]
        # Optional light stemming: drop common verbs/prefixes to reduce noise
        stop = {"get", "set", "make", "build", "create", "compute", "calc", "init", "update", "process", "run"}
        return [t for t in toks if t not in stop]

    def _signature_fingerprint(self, fr: FunctionRecord) -> str:
        """Stable coarse signature: argc bucket + kw flags + return category."""
        n = fr.node
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            return "anon"
        a = n.args
        argc = len(a.args) + len(a.kwonlyargs)
        argc_bucket = str(argc) if argc < 6 else "6+"
        has_var = 1 if a.vararg else 0
        has_kw = 1 if a.kwarg else 0
        ret_cat = self._return_type_category_sampling(n)
        return f"argc:{argc_bucket}|var:{has_var}|kw:{has_kw}|ret:{ret_cat}"

    def _return_type_category_sampling(self, fn: Union[ast.FunctionDef, ast.AsyncFunctionDef]) -> str:
        ann = getattr(fn, "returns", None)
        if ann is None:
            return "unknown"
        txt = ast.unparse(ann) if hasattr(ast, "unparse") else type(ann).__name__
        t = txt.lower().replace(" ", "")
        if t in {"none", "nonetype"}:
            return "none"
        if "bool" in t:
            return "bool"
        if "int" in t:
            return "int"
        if "float" in t or "decimal" in t:
            return "float"
        if "str" in t or "bytes" in t:
            return "str"
        if any(k in t for k in ["list", "tuple", "set", "iter", "seq", "ndarray"]):
            return "iterable"
        if any(k in t for k in ["dict", "mapping"]):
            return "mapping"
        return "object"

    # Reuse the same complexity primitives as in progressive encoder; define here to avoid import coupling
    def _ast_depth_sampling(self, node: ast.AST, *, max_nodes: int) -> int:
        max_depth = 0
        stack: List[Tuple[ast.AST, int]] = [(node, 1)]
        visited = 0
        while stack:
            cur, d = stack.pop()
            max_depth = max(max_depth, d)
            visited += 1
            if visited > max_nodes:
                break
            for child in ast.iter_child_nodes(cur):
                stack.append((child, d + 1))
        return max_depth

    def _control_flow_complexity_sampling(self, node: ast.AST, *, max_nodes: int) -> int:
        score = 1
        visited = 0
        CONTROL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith, ast.BoolOp)
        # Add Match for Python 3.10+
        try:
            CONTROL = CONTROL + (ast.Match,)
        except AttributeError:
            pass  # Python < 3.10
        COMPREH = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
        for n in ast.walk(node):
            visited += 1
            if visited > max_nodes:
                break
            if isinstance(n, CONTROL) or isinstance(n, COMPREH):
                score += 1
        return score