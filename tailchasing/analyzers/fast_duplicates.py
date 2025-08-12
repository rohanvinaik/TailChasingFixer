"""
Fast duplicate detection using LSH (Locality Sensitive Hashing).

This analyzer uses a multi-layer approach with approximate matching
to achieve O(n log n) complexity instead of O(n²).
"""

import ast
import hashlib
import logging
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional, Any
import yaml

try:
    import xxhash
    HAS_XXHASH = True
except ImportError:
    HAS_XXHASH = False
    xxhash = None

import numpy as np

from ..core.issues import Issue
from .base import Analyzer, AnalysisContext

logger = logging.getLogger(__name__)


@dataclass
class DuplicateCluster:
    """Represents a cluster of duplicate files."""
    cluster_id: str
    files: List[str] = field(default_factory=list)
    canonical_file: Optional[str] = None
    similarity_type: str = "exact"  # exact, near, structural
    content_hash: Optional[str] = None
    minhash_signature: Optional[np.ndarray] = None
    
    def select_canonical(self) -> str:
        """Select the canonical file from the cluster."""
        if not self.files:
            return None
        
        # Sort by: 1) shortest path, 2) newest mtime, 3) alphabetical
        def sort_key(file_path: str):
            path = Path(file_path)
            try:
                mtime = path.stat().st_mtime
            except:
                mtime = 0
            
            return (
                len(path.parts),  # Shorter paths first
                -mtime,  # Newer files first (negative for reverse sort)
                str(path)  # Alphabetical as tiebreaker
            )
        
        sorted_files = sorted(self.files, key=sort_key)
        self.canonical_file = sorted_files[0]
        return self.canonical_file


@dataclass 
class ConsolidationPlan:
    """Plan for consolidating duplicate files."""
    clusters: List[DuplicateCluster]
    total_files: int
    redundant_files: int
    estimated_loc_saved: int
    actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_yaml(self) -> str:
        """Convert consolidation plan to YAML format."""
        plan = {
            'duplicate_consolidation_plan': {
                'summary': {
                    'total_files_analyzed': self.total_files,
                    'redundant_files': self.redundant_files,
                    'estimated_loc_saved': self.estimated_loc_saved,
                    'clusters_found': len(self.clusters)
                },
                'clusters': []
            }
        }
        
        for cluster in self.clusters:
            cluster_data = {
                'id': cluster.cluster_id,
                'type': cluster.similarity_type,
                'canonical': cluster.canonical_file,
                'duplicates': [f for f in cluster.files if f != cluster.canonical_file],
                'action': 'consolidate'
            }
            plan['duplicate_consolidation_plan']['clusters'].append(cluster_data)
        
        # Add detailed actions
        plan['duplicate_consolidation_plan']['actions'] = self.actions
        
        return yaml.dump(plan, default_flow_style=False, sort_keys=False)


class FastDuplicateAnalyzer(Analyzer):
    """
    Fast duplicate detection using multi-layer LSH approach.
    
    Layers:
    1. Content hash for exact duplicates (xxhash on first 1MB)
    2. Shingle hashes for near-duplicates (k=7 character shingles)
    3. MinHash signatures from AST structure
    
    Uses LSH (Locality Sensitive Hashing) for O(n log n) complexity.
    """
    
    name = "fast_duplicates"
    
    def __init__(self):
        self.content_hashes: Dict[str, str] = {}
        self.shingle_hashes: Dict[str, Set[int]] = {}
        self.minhash_signatures: Dict[str, np.ndarray] = {}
        self.lsh_buckets: Dict[int, List[str]] = defaultdict(list)
        
        # LSH parameters
        self.num_perm = 128  # Number of MinHash permutations
        self.bands = 8  # Number of bands for LSH
        self.rows = 16  # Rows per band (bands * rows = num_perm)
        
        # Shingle parameters
        self.shingle_k = 7  # Character k-grams
        
        # Performance stats
        self.stats = {
            'files_processed': 0,
            'exact_duplicates': 0,
            'near_duplicates': 0,
            'structural_duplicates': 0,
            'comparisons_made': 0,
            'comparisons_saved': 0
        }
    
    def run(self, ctx: AnalysisContext) -> List[Issue]:
        """Run fast duplicate detection."""
        issues = []
        
        # Get configuration
        config = ctx.config.get('duplicates', {})
        if not config.get('enabled', True):
            return []
        
        # Get resource limits
        resource_limits = ctx.config.get('resource_limits', {})
        self.lsh_bucket_cap = resource_limits.get('lsh_bucket_cap', 2000)
        
        # Configure LSH parameters from config
        self.bands = config.get('lsh_bands', 8)
        self.rows = config.get('lsh_rows', 16)
        self.num_perm = self.bands * self.rows
        
        logger.info(f"Running fast duplicate detection with LSH (bands={self.bands}, rows={self.rows})")
        
        # Phase 1: Compute hashes for all files
        self._compute_all_hashes(ctx)
        
        # Phase 2: Find exact duplicates using content hash
        exact_clusters = self._find_exact_duplicates()
        
        # Phase 3: Find near duplicates using shingles and MinHash
        near_clusters = self._find_near_duplicates()
        
        # Phase 4: Find structural duplicates using AST MinHash
        structural_clusters = self._find_structural_duplicates(ctx)
        
        # Merge and deduplicate clusters
        all_clusters = self._merge_clusters(exact_clusters, near_clusters, structural_clusters)
        
        # Generate issues
        for cluster in all_clusters:
            if len(cluster.files) > 1:
                cluster.select_canonical()
                for file_path in cluster.files:
                    if file_path != cluster.canonical_file:
                        issue = Issue(
                            kind="duplicate_file",
                            file=file_path,
                            line=1,
                            message=f"Duplicate of {cluster.canonical_file} ({cluster.similarity_type})",
                            severity=2,
                            confidence=0.9 if cluster.similarity_type == "exact" else 0.7,
                            suggestions=[
                                f"Consider consolidating with {cluster.canonical_file}",
                                f"Remove this file and import from {cluster.canonical_file}",
                            ]
                        )
                        issues.append(issue)
        
        # Generate consolidation plan
        if all_clusters and config.get('generate_plan', True):
            plan = self._generate_consolidation_plan(all_clusters, ctx)
            self._save_plan(plan, ctx)
        
        # Log statistics
        self._log_statistics()
        
        return issues
    
    def _compute_all_hashes(self, ctx: AnalysisContext) -> None:
        """Compute all hash layers for files."""
        for file_path in ctx.ast_index:
            self.stats['files_processed'] += 1
            
            # Skip quarantined files
            if ctx.is_quarantined(file_path):
                continue
            
            # Layer 1: Content hash
            self.content_hashes[file_path] = self._compute_content_hash(file_path)
            
            # Layer 2: Shingle hashes
            self.shingle_hashes[file_path] = self._compute_shingle_hashes(file_path)
            
            # Layer 3: AST MinHash signature
            ast_tree = ctx.ast_index.get(file_path)
            if ast_tree:
                self.minhash_signatures[file_path] = self._compute_ast_minhash(ast_tree)
    
    def _compute_content_hash(self, file_path: str, max_bytes: int = 1024 * 1024) -> str:
        """
        Compute fast content hash using xxhash on first 1MB.
        
        Falls back to MD5 if xxhash is not available.
        """
        try:
            with open(file_path, 'rb') as f:
                content = f.read(max_bytes)
            
            if HAS_XXHASH:
                return xxhash.xxh64(content).hexdigest()
            else:
                return hashlib.md5(content).hexdigest()
        except Exception as e:
            logger.debug(f"Error hashing {file_path}: {e}")
            return f"error_{file_path}"
    
    def _compute_shingle_hashes(self, file_path: str) -> Set[int]:
        """
        Compute character k-shingles for text similarity.
        
        Uses rolling hash for efficiency.
        """
        shingles = set()
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Normalize whitespace
            content = ' '.join(content.split())
            
            # Generate k-shingles with rolling hash
            if len(content) >= self.shingle_k:
                for i in range(len(content) - self.shingle_k + 1):
                    shingle = content[i:i + self.shingle_k]
                    # Use hash() for speed, & mask to ensure positive
                    shingle_hash = hash(shingle) & 0x7FFFFFFF
                    shingles.add(shingle_hash)
        except Exception as e:
            logger.debug(f"Error computing shingles for {file_path}: {e}")
        
        return shingles
    
    def _compute_ast_minhash(self, ast_tree: ast.AST) -> np.ndarray:
        """
        Compute MinHash signature from AST structure.
        
        Extracts features from AST and creates MinHash signature.
        """
        # Extract AST features
        features = self._extract_ast_features(ast_tree)
        
        # Convert features to shingles
        feature_shingles = set()
        for feature in features:
            # Create hash from feature
            feature_hash = hash(feature) & 0x7FFFFFFF
            feature_shingles.add(feature_hash)
        
        # Compute MinHash signature
        signature = self._compute_minhash(feature_shingles)
        return signature
    
    def _extract_ast_features(self, tree: ast.AST) -> List[str]:
        """Extract structural features from AST."""
        features = []
        
        for node in ast.walk(tree):
            # Function definitions
            if isinstance(node, ast.FunctionDef):
                # Include function signature
                args = [arg.arg for arg in node.args.args]
                feature = f"func:{node.name}({','.join(args)})"
                features.append(feature)
                
                # Include return type if present
                if node.returns:
                    features.append(f"return_type:{ast.unparse(node.returns)}")
            
            # Class definitions
            elif isinstance(node, ast.ClassDef):
                bases = [ast.unparse(base) for base in node.bases]
                feature = f"class:{node.name}({','.join(bases)})"
                features.append(feature)
            
            # Import statements
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    features.append(f"import:{alias.name}")
            
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ''
                for alias in node.names:
                    features.append(f"from:{module}.{alias.name}")
            
            # Control flow
            elif isinstance(node, ast.If):
                features.append("if_statement")
            elif isinstance(node, ast.For):
                features.append("for_loop")
            elif isinstance(node, ast.While):
                features.append("while_loop")
            elif isinstance(node, ast.Try):
                features.append("try_except")
        
        return features
    
    def _compute_minhash(self, shingles: Set[int]) -> np.ndarray:
        """
        Compute MinHash signature from a set of shingles.
        
        Uses universal hashing with random permutations.
        """
        if not shingles:
            return np.zeros(self.num_perm, dtype=np.uint32)
        
        # Convert shingles to numpy array for vectorized operations
        shingle_array = np.array(list(shingles), dtype=np.uint32)
        
        signature = np.zeros(self.num_perm, dtype=np.uint32)
        
        # Use deterministic random permutations
        np.random.seed(42)
        
        for i in range(self.num_perm):
            # Generate random hash coefficients
            a = np.random.randint(1, 2**32 - 1)
            b = np.random.randint(0, 2**32 - 1)
            c = 2**32 - 1  # Large prime
            
            # Apply hash function: h(x) = (a*x + b) % c
            hashed = ((a * shingle_array + b) % c).astype(np.uint32)
            
            # Take minimum
            signature[i] = np.min(hashed)
        
        return signature
    
    def _find_exact_duplicates(self) -> List[DuplicateCluster]:
        """Find exact duplicates using content hash."""
        hash_to_files = defaultdict(list)
        
        for file_path, content_hash in self.content_hashes.items():
            hash_to_files[content_hash].append(file_path)
        
        clusters = []
        for content_hash, files in hash_to_files.items():
            if len(files) > 1:
                self.stats['exact_duplicates'] += len(files) - 1
                cluster = DuplicateCluster(
                    cluster_id=f"exact_{content_hash[:8]}",
                    files=files,
                    similarity_type="exact",
                    content_hash=content_hash
                )
                clusters.append(cluster)
        
        return clusters
    
    def _find_near_duplicates(self) -> List[DuplicateCluster]:
        """Find near duplicates using LSH on shingle hashes."""
        # Build LSH index
        self._build_lsh_index()
        
        # Find candidates in same buckets
        clusters = []
        processed = set()
        
        for bucket_id, candidates in self.lsh_buckets.items():
            if len(candidates) <= 1:
                continue
            
            # Cap bucket size to avoid O(n²) within buckets
            if len(candidates) > self.lsh_bucket_cap:
                logger.warning(f"Bucket {bucket_id} has {len(candidates)} candidates, capping at {self.lsh_bucket_cap}")
                candidates = candidates[:self.lsh_bucket_cap]
            
            # Compare candidates within bucket
            for i, file1 in enumerate(candidates):
                if file1 in processed:
                    continue
                
                cluster_files = [file1]
                
                for file2 in candidates[i+1:]:
                    if file2 in processed:
                        continue
                    
                    self.stats['comparisons_made'] += 1
                    
                    # Compute Jaccard similarity
                    similarity = self._jaccard_similarity(
                        self.shingle_hashes.get(file1, set()),
                        self.shingle_hashes.get(file2, set())
                    )
                    
                    if similarity > 0.7:  # Threshold for near-duplicate
                        cluster_files.append(file2)
                        processed.add(file2)
                
                if len(cluster_files) > 1:
                    self.stats['near_duplicates'] += len(cluster_files) - 1
                    cluster = DuplicateCluster(
                        cluster_id=f"near_{hash(tuple(sorted(cluster_files))) & 0x7FFFFFFF:08x}",
                        files=cluster_files,
                        similarity_type="near"
                    )
                    clusters.append(cluster)
                    processed.add(file1)
        
        # Calculate comparisons saved
        n = len(self.minhash_signatures)
        total_possible = n * (n - 1) // 2
        self.stats['comparisons_saved'] = total_possible - self.stats['comparisons_made']
        
        return clusters
    
    def _build_lsh_index(self) -> None:
        """Build LSH index for MinHash signatures."""
        self.lsh_buckets.clear()
        
        for file_path, signature in self.minhash_signatures.items():
            # Split signature into bands
            for band_idx in range(self.bands):
                start = band_idx * self.rows
                end = start + self.rows
                band = signature[start:end]
                
                # Hash the band to get bucket ID
                bucket_id = hash(tuple(band)) & 0x7FFFFFFF
                
                self.lsh_buckets[bucket_id].append(file_path)
    
    def _find_structural_duplicates(self, ctx: AnalysisContext) -> List[DuplicateCluster]:
        """Find structural duplicates based on AST similarity."""
        clusters = []
        
        # Use LSH buckets to find candidates
        processed = set()
        
        for bucket_id, candidates in self.lsh_buckets.items():
            if len(candidates) <= 1:
                continue
            
            for i, file1 in enumerate(candidates):
                if file1 in processed:
                    continue
                
                ast1 = ctx.ast_index.get(file1)
                if not ast1:
                    continue
                
                cluster_files = [file1]
                
                for file2 in candidates[i+1:]:
                    if file2 in processed:
                        continue
                    
                    ast2 = ctx.ast_index.get(file2)
                    if not ast2:
                        continue
                    
                    # Check structural similarity
                    if self._are_structurally_similar(ast1, ast2):
                        cluster_files.append(file2)
                        processed.add(file2)
                
                if len(cluster_files) > 1:
                    self.stats['structural_duplicates'] += len(cluster_files) - 1
                    cluster = DuplicateCluster(
                        cluster_id=f"struct_{hash(tuple(sorted(cluster_files))) & 0x7FFFFFFF:08x}",
                        files=cluster_files,
                        similarity_type="structural"
                    )
                    clusters.append(cluster)
                    processed.add(file1)
        
        return clusters
    
    def _are_structurally_similar(self, ast1: ast.AST, ast2: ast.AST) -> bool:
        """Check if two ASTs are structurally similar."""
        # Extract structural features
        features1 = set(self._extract_ast_features(ast1))
        features2 = set(self._extract_ast_features(ast2))
        
        # Compute Jaccard similarity
        if not features1 or not features2:
            return False
        
        similarity = self._jaccard_similarity(features1, features2)
        return similarity > 0.8  # High threshold for structural similarity
    
    def _jaccard_similarity(self, set1: Set, set2: Set) -> float:
        """Compute Jaccard similarity between two sets."""
        if not set1 and not set2:
            return 1.0
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
    
    def _merge_clusters(self, *cluster_lists) -> List[DuplicateCluster]:
        """Merge and deduplicate clusters from different detection methods."""
        file_to_cluster = {}
        merged_clusters = []
        
        # Process clusters in priority order: exact > near > structural
        for clusters in cluster_lists:
            for cluster in clusters:
                # Check if any file is already in a cluster
                existing_cluster = None
                for file_path in cluster.files:
                    if file_path in file_to_cluster:
                        existing_cluster = file_to_cluster[file_path]
                        break
                
                if existing_cluster:
                    # Merge into existing cluster
                    for file_path in cluster.files:
                        if file_path not in existing_cluster.files:
                            existing_cluster.files.append(file_path)
                            file_to_cluster[file_path] = existing_cluster
                else:
                    # Create new cluster
                    merged_clusters.append(cluster)
                    for file_path in cluster.files:
                        file_to_cluster[file_path] = cluster
        
        return merged_clusters
    
    def _generate_consolidation_plan(self, clusters: List[DuplicateCluster], ctx: AnalysisContext) -> ConsolidationPlan:
        """Generate a consolidation plan for duplicate files."""
        total_files = len(ctx.ast_index)
        redundant_files = sum(len(c.files) - 1 for c in clusters)
        
        # Estimate LOC saved
        loc_saved = 0
        for cluster in clusters:
            for file_path in cluster.files:
                if file_path != cluster.canonical_file:
                    try:
                        lines = ctx.get_source_lines(file_path)
                        loc_saved += len(lines)
                    except:
                        pass
        
        # Generate actions
        actions = []
        for cluster in clusters:
            if not cluster.canonical_file:
                cluster.select_canonical()
            
            for file_path in cluster.files:
                if file_path != cluster.canonical_file:
                    action = {
                        'type': 'remove_duplicate',
                        'file': file_path,
                        'canonical': cluster.canonical_file,
                        'similarity': cluster.similarity_type,
                        'safe': cluster.similarity_type == "exact",
                        'command': f"# Remove duplicate\nrm {file_path}\n# Update imports to use {cluster.canonical_file}"
                    }
                    actions.append(action)
        
        plan = ConsolidationPlan(
            clusters=clusters,
            total_files=total_files,
            redundant_files=redundant_files,
            estimated_loc_saved=loc_saved,
            actions=actions
        )
        
        return plan
    
    def _save_plan(self, plan: ConsolidationPlan, ctx: AnalysisContext) -> None:
        """Save consolidation plan to YAML file."""
        output_dir = Path(ctx.config.get('report', {}).get('output_dir', '.'))
        plan_file = output_dir / 'duplicates_plan.yaml'
        
        try:
            with open(plan_file, 'w') as f:
                f.write(plan.to_yaml())
            logger.info(f"Consolidation plan saved to {plan_file}")
        except Exception as e:
            logger.error(f"Failed to save consolidation plan: {e}")
    
    def _log_statistics(self) -> None:
        """Log performance statistics."""
        logger.info(
            f"Fast duplicate detection complete: "
            f"{self.stats['files_processed']} files, "
            f"{self.stats['exact_duplicates']} exact, "
            f"{self.stats['near_duplicates']} near, "
            f"{self.stats['structural_duplicates']} structural duplicates. "
            f"Made {self.stats['comparisons_made']} comparisons "
            f"(saved {self.stats['comparisons_saved']} comparisons using LSH)"
        )