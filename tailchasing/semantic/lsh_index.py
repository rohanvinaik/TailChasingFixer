# tailchasing/semantic/lsh_index.py
"""
MinHash LSH implementation for efficient semantic duplicate pre-clustering.

This module provides a dependency-light implementation of MinHash with Locality
Sensitive Hashing (LSH) for reducing O(n²) comparisons to O(n·k) where k << n.
"""
from __future__ import annotations

import ast
import hashlib
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# ----------------------------
# Public API
# ----------------------------

# Import FunctionRecord from consolidated types
from ..core.types import FunctionRecord


@dataclass
class LSHParams:
    """
    Parameters for MinHash + LSH banding.

    Notes
    -----
    - num_hashes should be rows_per_band * bands.
    - Default: 100 hashes, 20 bands, 5 rows per band (20 * 5 = 100).
    """
    num_hashes: int = 100
    bands: int = 20
    rows_per_band: int = 5
    # Hash sampler seed controls reproducibility of MinHash perm functions
    seed: int = 0x5EED_1DEE
    # Jaccard of shingles is only approximated via MinHash; tune threshold in downstream stages
    jaccard_gate: float = 0.0  # gate at 0 here; actual filtering happens after candidate generation


@dataclass
class LSHIndex:
    """
    LSH index that maps band buckets -> function IDs and yields candidate pairs.

    After `build`, call `candidate_pairs()` to iterate (id_i, id_j) to compare.
    """
    params: LSHParams
    buckets: Dict[Tuple[int, str], List[str]] = field(default_factory=dict)  # (band_id, key) -> list of function ids
    signatures: Dict[str, Tuple[int, ...]] = field(default_factory=dict)  # id -> MinHash signature

    def add(self, func_id: str, signature: Sequence[int]) -> None:
        sig_t = tuple(signature)
        if len(sig_t) != self.params.num_hashes:
            raise ValueError("Signature length does not match num_hashes")
        self.signatures[func_id] = sig_t
        # Split into bands
        r = self.params.rows_per_band
        for b in range(self.params.bands):
            start = b * r
            end = start + r
            band_slice = sig_t[start:end]
            bucket_key = (b, _stable_band_key(band_slice))
            self.buckets.setdefault(bucket_key, []).append(func_id)

    def candidate_pairs(self) -> Iterator[Tuple[str, str]]:
        """
        Yield candidate pairs (i, j) with i < j that share at least one LSH bucket.
        """
        seen: Set[Tuple[str, str]] = set()
        for (_, _key), ids in self.buckets.items():
            if len(ids) < 2:
                continue
            # Deduplicate within band
            ids_sorted = sorted(ids)
            for i_idx in range(len(ids_sorted)):
                for j_idx in range(i_idx + 1, len(ids_sorted)):
                    pair = (ids_sorted[i_idx], ids_sorted[j_idx])
                    if pair not in seen:
                        seen.add(pair)
                        yield pair


# ----------------------------
# Feature Extraction
# ----------------------------

@dataclass
class FeatureConfig:
    """
    Controls which feature channels to include in the shingle set.
    """
    use_ast_3grams: bool = True
    use_imports: bool = True
    use_call_patterns: bool = True
    max_nodes: int = 50_000  # safety cap


def extract_shingles(func: FunctionRecord, cfg: FeatureConfig = FeatureConfig()) -> Set[str]:
    """
    Extract a unified shingle set from:
    - AST node-type 3-grams
    - Import signatures (module, alias)
    - Call graph patterns (callee names + simple context)

    Returns
    -------
    Set[str] of shingles suitable for MinHash.
    """
    shingles: Set[str] = set()
    try:
        root = _ensure_module_ast(func.source)
    except Exception:
        # If parsing failed upstream, fallback to walking the provided node only
        root = func.node

    # Restrict traversal to the function body to avoid cross-function leakage
    target_node = func.node

    if cfg.use_ast_3grams:
        node_types = _collect_node_types(target_node, max_nodes=cfg.max_nodes)
        for triad in _n_grams(node_types, 3):
            shingles.add(f"AST3::{triad[0]}>{triad[1]}>{triad[2]}")

    if cfg.use_imports:
        for imp_sig in _collect_import_signatures(root, within_node=target_node):
            shingles.add(f"IMP::{imp_sig}")

    if cfg.use_call_patterns:
        for call_sig in _collect_call_signatures(target_node):
            shingles.add(f"CALL::{call_sig}")

    return shingles


def _collect_node_types(node: ast.AST, *, max_nodes: int) -> List[str]:
    types: List[str] = []
    for i, n in enumerate(ast.walk(node)):
        if i >= max_nodes:
            break
        types.append(type(n).__name__)
    return types


def _n_grams(tokens: Sequence[str], n: int) -> Iterator[Tuple[str, ...]]:
    if n <= 0 or n > len(tokens):
        return iter(())
    for i in range(len(tokens) - n + 1):
        yield tuple(tokens[i:i + n])


def _collect_import_signatures(module_root: ast.AST, *, within_node: ast.AST) -> Iterator[str]:
    """
    Collect imports that are *used* inside the function of interest, if we can detect simple usage.
    Fallback: include all module-level imports referenced by name within the function.
    """
    imported: Dict[str, str] = {}  # local name -> module[:object]
    for n in ast.walk(module_root):
        if isinstance(n, ast.Import):
            for alias in n.names:
                local = alias.asname or alias.name.split(".")[0]
                imported[local] = alias.name
        elif isinstance(n, ast.ImportFrom):
            mod = n.module or ""
            for alias in n.names:
                local = alias.asname or alias.name
                imported[local] = f"{mod}.{alias.name}" if mod else alias.name

    used_names: Set[str] = set()
    for n in ast.walk(within_node):
        if isinstance(n, ast.Name):
            used_names.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            used_names.add(n.value.id)

    for local, target in imported.items():
        if local in used_names:
            yield f"{local}={target}"


def _collect_call_signatures(node: ast.AST) -> Iterator[str]:
    """
    Very lightweight call pattern signature:
    - Callee simple name (or dotted base)
    - Arity (positional count)
    - Whether there are kwargs
    """
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            name = _extract_callee_name(n.func)
            argc = len(n.args)
            has_kwargs = bool(n.keywords)
            yield f"{name}|args:{argc}|kwargs:{int(has_kwargs)}"


def _extract_callee_name(func: ast.AST) -> str:
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        parts: List[str] = []
        cur: Optional[ast.AST] = func
        # Walk dotted attribute chain
        while isinstance(cur, ast.Attribute):
            parts.append(cur.attr)
            cur = cur.value
        if isinstance(cur, ast.Name):
            parts.append(cur.id)
        parts.reverse()
        return ".".join(parts) if parts else "attr"
    return type(func).__name__


def _ensure_module_ast(source: str) -> ast.AST:
    return ast.parse(source)


# ----------------------------
# MinHash (dependency-light)
# ----------------------------

class MinHasher:
    """
    Simple MinHash implementation using multiple seeded 64-bit hash functions simulated
    via hashlib.blake2b with per-permutation salt. Good enough for LSH pre-clustering.
    """
    __slots__ = ("_seeds", "_num_hashes")

    def __init__(self, num_hashes: int, seed: int) -> None:
        self._num_hashes = num_hashes
        rnd = random.Random(seed)
        # Precompute per-permutation salt bytes to avoid re-seeding/allocs
        self._seeds: List[bytes] = []
        for _ in range(num_hashes):
            salt = rnd.getrandbits(64).to_bytes(8, "little")
            self._seeds.append(salt)

    def signature(self, shingles: Iterable[str]) -> Tuple[int, ...]:
        # Initialize mins to "infinity" (here: max uint64)
        mins = [0xFFFFFFFFFFFFFFFF] * self._num_hashes
        for sh in shingles:
            # Normalize to bytes once per shingle
            sbytes = sh.encode("utf-8", errors="ignore")
            for i, salt in enumerate(self._seeds):
                # blake2b digest with 8-byte salt; digest_size=8 -> 64-bit
                h = hashlib.blake2b(sbytes, digest_size=8, salt=salt).digest()
                hv = int.from_bytes(h, "little")
                if hv < mins[i]:
                    mins[i] = hv
        return tuple(mins)


# ----------------------------
# LSH Banding Utilities
# ----------------------------

def _stable_band_key(rows: Sequence[int]) -> str:
    """
    Stable, concise key for a band slice of the signature.
    """
    # Use a short digest over the 5-row band to compress the key length
    m = hashlib.blake2b(digest_size=8)
    for r in rows:
        m.update(r.to_bytes(8, "little"))
    return m.hexdigest()


# ----------------------------
# Orchestration
# ----------------------------

@dataclass
class PreclusterStats:
    total_functions: int = 0
    total_buckets: int = 0
    non_singleton_buckets: int = 0
    avg_bucket_size: float = 0.0

    def as_dict(self) -> Dict[str, float]:
        return {
            "total_functions": float(self.total_functions),
            "total_buckets": float(self.total_buckets),
            "non_singleton_buckets": float(self.non_singleton_buckets),
            "avg_bucket_size": float(self.avg_bucket_size),
        }


def build_lsh_index(
    functions: Iterable[FunctionRecord],
    *,
    feat_cfg: FeatureConfig = FeatureConfig(),
    params: LSHParams = LSHParams(),
) -> Tuple[LSHIndex, PreclusterStats]:
    """
    Build the LSH index for the provided functions.

    Returns
    -------
    (index, stats)
    """
    if params.bands * params.rows_per_band != params.num_hashes:
        raise ValueError("num_hashes must equal bands * rows_per_band")

    index = LSHIndex(params=params)
    hasher = MinHasher(num_hashes=params.num_hashes, seed=params.seed)

    count = 0
    for fr in functions:
        shingles = extract_shingles(fr, cfg=feat_cfg)
        sig = hasher.signature(shingles if shingles else ["__EMPTY__"])
        index.add(fr.id, sig)
        count += 1

    # Stats
    bucket_sizes = [len(v) for v in index.buckets.values()]
    non_singletons = sum(1 for s in bucket_sizes if s > 1)
    avg_size = (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0.0
    stats = PreclusterStats(
        total_functions=count,
        total_buckets=len(bucket_sizes),
        non_singleton_buckets=non_singletons,
        avg_bucket_size=avg_size,
    )
    return index, stats


def candidate_pairs_from_index(index: LSHIndex) -> Iterator[Tuple[str, str]]:
    """
    Convenience wrapper if callers don't want to touch the index object.
    """
    yield from index.candidate_pairs()


# ----------------------------
# High-Level Entry Point
# ----------------------------

def precluster_for_comparison(
    functions: Iterable[FunctionRecord],
    *,
    feat_cfg: FeatureConfig = FeatureConfig(),
    params: LSHParams = LSHParams(),
) -> Tuple[List[Tuple[str, str]], PreclusterStats]:
    """
    Main function your pipeline should call.

    Returns
    -------
    (pairs, stats)
      - pairs: list of (func_id_i, func_id_j) candidate pairs to send to detailed comparison
      - stats: basic preclustering stats for observability
    """
    index, stats = build_lsh_index(functions, feat_cfg=feat_cfg, params=params)
    pairs = list(candidate_pairs_from_index(index))
    return pairs, stats


# ----------------------------
# Integration Helper
# ----------------------------

def create_function_records(
    ast_index: Dict[str, ast.AST],
    source_cache: Optional[Dict[str, str]] = None
) -> List[FunctionRecord]:
    """
    Convert AST index to FunctionRecord objects for LSH indexing.
    
    Args:
        ast_index: Dictionary mapping file paths to AST trees
        source_cache: Optional cache of file contents
        
    Returns:
        List of FunctionRecord objects
    """
    from pathlib import Path
    
    records = []
    for file_path, tree in ast_index.items():
        # Get source code
        if source_cache and file_path in source_cache:
            source = source_cache[file_path]
        else:
            try:
                source = Path(file_path).read_text(encoding='utf-8', errors='ignore')
            except Exception:
                continue
                
        # Extract functions
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
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


# ----------------------------
# Example (for local testing / removable)
# ----------------------------

if __name__ == "__main__":
    # Minimal smoke test using two toy functions
    code = """
import math as m
from collections import defaultdict as dd

def f(x):
    y = m.sqrt(x) + m.sin(x)
    return y

def g(z):
    h = m.sqrt(z) + m.cos(z)
    return h

def unrelated(a): 
    return a + 1
"""
    tree = ast.parse(code)
    funcs: List[FunctionRecord] = []
    for n in ast.walk(tree):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fid = f"{n.name}@{n.lineno}"
            funcs.append(FunctionRecord(id=fid, source=code, node=n, file="__inline__"))

    pairs, stats = precluster_for_comparison(funcs)
    print("Pairs:", pairs)
    print("Stats:", stats.as_dict())