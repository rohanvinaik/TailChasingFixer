# tailchasing/semantic/progressive_encoder.py
"""
Progressive multi-level encoding for efficient semantic duplicate detection.

Implements a 3-level progressive signature scheme:
- Level 1 (32 bits): Coarse bucketing by name/signature patterns
- Level 2 (128 bits): Medium granularity with AST depth and control flow
- Level 3 (1024 bits): Fine-grained hypervector encoding

This provides staged refinement, reducing comparisons at each level.
"""
from __future__ import annotations

import ast
import hashlib
import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# ---------------------------------------------------------------------
# Public data model (mirrors tailchasing.semantic.lsh_index.FunctionRecord)
# ---------------------------------------------------------------------

# Import FunctionRecord from consolidated types
from ..core.types import FunctionRecord


# ---------------------------------------------------------------------
# Bit-signature helpers
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class BitSig:
    """Fixed-length bit signature (stored in a Python int)."""
    bits: int          # bit container as an int
    width: int         # number of valid bits

    def hamming(self, other: "BitSig") -> int:
        if self.width != other.width:
            raise ValueError("BitSig width mismatch")
        return int.bit_count(self.bits ^ other.bits)

    def jaccard(self, other: "BitSig") -> float:
        """Jaccard similarity on set bits (0..1)."""
        if self.width != other.width:
            raise ValueError("BitSig width mismatch")
        inter = int.bit_count(self.bits & other.bits)
        union = int.bit_count(self.bits | other.bits)
        return (inter / union) if union else 1.0

    def popcount(self) -> int:
        return int.bit_count(self.bits)

    @staticmethod
    def zero(width: int) -> "BitSig":
        return BitSig(0, width)

    def __str__(self) -> str:
        # Short debug string
        return f"BitSig<{self.width}> pc={self.popcount()}"


# ---------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------

@dataclass
class ProgressiveParams:
    # Bit widths per level
    level1_bits: int = 32
    level2_bits: int = 128
    level3_bits: int = 1024

    # Refinement gates (defaults are conservative; tune empirically)
    # Level 1 -> Level 2: bucket by exact Level-1 signature (or gate by same bucket id)
    # Level 2 -> Level 3: require Jaccard >= min_l2_jaccard
    min_l2_jaccard: float = 0.35

    # Final acceptance at Level 3 (hypervector similarity threshold)
    # If using bit-Jaccard for HV, this is fine; if using cosine, reinterpret accordingly
    min_l3_similarity: float = 0.25

    # Import/control-flow caps to avoid pathological files
    max_nodes: int = 50_000


# ---------------------------------------------------------------------
# Feature extraction – shared primitives
# ---------------------------------------------------------------------

_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")

def _ensure_module_ast(source: str) -> ast.AST:
    return ast.parse(source)

def _funcdef(node: ast.AST) -> ast.FunctionDef | ast.AsyncFunctionDef:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return node
    raise TypeError("FunctionRecord.node must be FunctionDef/AsyncFunctionDef")

def _name_tokens(name: str) -> List[str]:
    # snake_case + CamelCase split
    snake = name.replace("__", "_").strip("_")
    parts: List[str] = []
    for p in snake.split("_"):
        parts.extend(_CAMEL_SPLIT.split(p) if p else [])
    return [t.lower() for t in parts if t]

def _return_type_category(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    ann = getattr(fn, "returns", None)
    if ann is None:
        return "unknown"
    # Quick categories; expand as needed
    text = ast.unparse(ann) if hasattr(ast, "unparse") else _ann_fallback(ann)
    t = text.lower().replace(" ", "")
    # Simple buckets
    if t in {"none", "nonetype"}:
        return "none"
    if any(k in t for k in ["bool"]):
        return "bool"
    if any(k in t for k in ["int"]):
        return "int"
    if any(k in t for k in ["float", "decimal"]):
        return "float"
    if any(k in t for k in ["str", "bytes"]):
        return "str"
    if any(k in t for k in ["list", "tuple", "set", "iter", "seq", "np.ndarray"]):
        return "iterable"
    if any(k in t for k in ["dict", "mapping"]):
        return "mapping"
    return "object"

def _param_count(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> int:
    a = fn.args
    return (len(a.args) + len(a.kwonlyargs)
            + (1 if a.vararg else 0)
            + (1 if a.kwarg else 0))

def _ast_depth(node: ast.AST, *, max_nodes: int) -> int:
    # Iterative DFS to avoid recursion depth issues
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

def _control_flow_complexity(node: ast.AST, *, max_nodes: int) -> int:
    """Crude McCabe-ish proxy."""
    score = 1
    visited = 0
    CONTROL = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With, ast.AsyncWith)
    COMPREH = (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    # Add Match for Python 3.10+
    try:
        CONTROL = CONTROL + (ast.Match,)
    except AttributeError:
        pass  # Python < 3.10
    
    for n in ast.walk(node):
        visited += 1
        if visited > max_nodes:
            break
        if isinstance(n, CONTROL):
            score += 1
        elif isinstance(n, COMPREH):
            score += 1
        elif isinstance(n, ast.BoolOp):
            score += 1
    return score

def _collect_import_signatures(module_root: ast.AST, *, within_node: ast.AST) -> Set[str]:
    imported: Dict[str, str] = {}
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
    used: Set[str] = set()
    for n in ast.walk(within_node):
        if isinstance(n, ast.Name):
            used.add(n.id)
        elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
            used.add(n.value.id)
    out: Set[str] = set()
    for local, target in imported.items():
        if local in used:
            out.add(f"{local}={target}")
    return out

def _ann_fallback(ann: ast.AST) -> str:
    # Very rough; fine for skeleton
    if isinstance(ann, ast.Name):
        return ann.id
    return type(ann).__name__


# ---------------------------------------------------------------------
# Hashing primitives
# ---------------------------------------------------------------------

def _mix_into_bits(seed: int, width: int) -> int:
    """Turn a 64-bit seed into a width-bit mask via xorshift-like mixing."""
    x = seed & 0xFFFFFFFFFFFFFFFF
    x ^= (x << 13) & 0xFFFFFFFFFFFFFFFF
    x ^= (x >> 7)
    x ^= (x << 17) & 0xFFFFFFFFFFFFFFFF
    # Expand to requested width by tiling the 64-bit pattern
    if width <= 64:
        mask = (1 << width) - 1
        return x & mask
    chunks = []
    remaining = width
    cur = x
    while remaining > 0:
        chunks.append(cur)
        # evolve cur
        cur = (cur * 0x9E3779B185EBCA87) & 0xFFFFFFFFFFFFFFFF
        remaining -= 64
    acc = 0
    pos = 0
    for c in chunks:
        acc |= (c << pos)
        pos += 64
    # Trim any extra high bits
    return acc & ((1 << width) - 1)

def _hash_str(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")

def _hash_tuple(parts: Sequence[str]) -> int:
    h = hashlib.blake2b(digest_size=8)
    for p in parts:
        h.update(p.encode("utf-8"))
        h.update(b"\x00")
    return int.from_bytes(h.digest(), "little")

def _bit_sig_from_features(features: Iterable[str], width: int) -> BitSig:
    """Feature hashing into a fixed-size bit signature (set a few bits per feature)."""
    bits = 0
    for f in features:
        seed = _hash_str(f)
        # set 3 pseudo-random positions for stability
        for k in range(3):
            s = seed ^ (0x9E3779B185EBCA87 * (k + 1))
            pos = s % width
            bits |= (1 << pos)
    return BitSig(bits, width)


# ---------------------------------------------------------------------
# Level encoders
# ---------------------------------------------------------------------

def encode_level1(fr: FunctionRecord, *, width: int = 32) -> BitSig:
    """
    Level 1 (Coarse - 32 bits):
      - Function name tokens
      - Return type category
      - Parameter count (bucketed)
    """
    fn = _funcdef(fr.node)
    feats: List[str] = []

    # Name tokens
    for tok in _name_tokens(fn.name):
        feats.append(f"name:{tok}")

    # Return type category
    feats.append(f"ret:{_return_type_category(fn)}")

    # Param count bucket (0,1,2,3,4,5+)
    pc = _param_count(fn)
    pc_bucket = str(pc) if pc < 5 else "5plus"
    feats.append(f"argc:{pc_bucket}")

    return _bit_sig_from_features(feats, width)

def encode_level2(fr: FunctionRecord, *, width: int = 128, max_nodes: int = 50_000) -> BitSig:
    """
    Level 2 (Medium - 128 bits):
      - Level 1 features
      - AST depth
      - Control flow complexity
      - Import patterns (actually used in function)
    """
    fn = _funcdef(fr.node)
    feats: List[str] = []

    # Reuse level 1 feats by re-hashing its set bits as tokens (cheap & stable)
    l1 = encode_level1(fr, width=32)
    feats.append(f"l1:pc={l1.popcount()}")  # coarse carry-over

    # AST context
    depth = _ast_depth(fn, max_nodes=max_nodes)
    cflow = _control_flow_complexity(fn, max_nodes=max_nodes)
    # Bucket to reduce sensitivity
    feats.append(f"depth:{_bucket(depth)}")
    feats.append(f"cflow:{_bucket(cflow)}")

    # Import patterns (from module, used in function)
    try:
        module = _ensure_module_ast(fr.source)
    except Exception:
        module = fn  # fallback
    imps = _collect_import_signatures(module, within_node=fn)
    for imp in sorted(imps):
        feats.append(f"imp:{imp}")

    return _bit_sig_from_features(feats, width)

def encode_level3(fr: FunctionRecord, *, width: int = 1024) -> BitSig:
    """
    Level 3 (Fine - 1024 bits):
      - Full hypervector encoding (pluggable)
    
    Default implementation provides a deterministic bipolar projection over simple tokens,
    so you can smoke-test end-to-end; swap with your real HV encoder when ready.
    """
    # ------- Hook point: replace this section with your HV pipeline -------
    # Try to use the real hypervector encoder if available
    try:
        from .hv_space import HVSpace
        from .encoder import encode_function
        
        # Use the real hypervector encoder
        hv_space = HVSpace(dim=width)
        # Create a basic config for the encoder
        config = {
            'channels': ['tokens', 'structure', 'data_flow'],
            'weights': {'tokens': 0.4, 'structure': 0.3, 'data_flow': 0.3}
        }
        hv = encode_function(fr.node, fr.file or 'unknown', hv_space, config)
        
        # Convert hypervector to bit signature
        # Map positive values to 1, negative to 0
        bits = 0
        for i, val in enumerate(hv[:width]):
            if val > 0:
                bits |= (1 << i)
        return BitSig(bits, width)
        
    except ImportError:
        # Fallback to simple deterministic projection
        # Build a light feature bag: identifiers + statement types
        fn = _funcdef(fr.node)
        tokens: List[str] = []
        # identifiers
        for n in ast.walk(fn):
            if isinstance(n, ast.Name):
                tokens.append(f"id:{n.id.lower()}")
            # statement kinds as coarse structure
            stmt_types = (ast.If, ast.For, ast.AsyncFor, ast.While, ast.Try, ast.With)
            # Add Match for Python 3.10+
            try:
                stmt_types = stmt_types + (ast.Match,)
            except AttributeError:
                pass
            if isinstance(n, stmt_types):
                tokens.append(f"stmt:{type(n).__name__}")
        # Deterministic projection: each token toggles a small subset of bits
        bits = 0
        for t in tokens:
            seed = _hash_str(t)
            for k in range(5):  # a few bits per token
                pos = (seed ^ (k * 0x9E3779B185EBCA87)) % width
                bits ^= (1 << pos)  # XOR → quasi-bipolar effect
        return BitSig(bits, width)
    # ---------------------------------------------------------------------


def _bucket(x: int) -> str:
    # Log-ish buckets to stabilize
    if x <= 0:
        return "0"
    if x <= 2:
        return str(x)
    b = int(math.log2(x))  # 3–4->2, 5–8->3, 9–16->4, ...
    return f"2^~{b}"


# ---------------------------------------------------------------------
# Progressive refinement orchestration
# ---------------------------------------------------------------------

@dataclass
class ProgressiveStats:
    total_functions: int = 0
    l1_buckets: int = 0
    l2_screened_pairs: int = 0
    l3_screened_pairs: int = 0
    final_pairs: int = 0
    avg_bucket_size_l1: float = 0.0

def progressive_pairs(
    functions: Iterable[FunctionRecord],
    *,
    params: ProgressiveParams = ProgressiveParams(),
) -> Tuple[List[Tuple[str, str]], ProgressiveStats]:
    """
    Progressive flow:
      1) Group by Level-1 signature (exact bit-equality bucket).
      2) For each bucket, compute Level-2 signatures and screen by Jaccard.
      3) For L2-passed pairs, compute Level-3 signatures and screen by final similarity.
    Returns final accepted pairs + stats.
    """
    fr_list = list(functions)
    stats = ProgressiveStats(total_functions=len(fr_list))

    # --- Level 1 bucketing
    l1_map: Dict[str, List[int]] = {}  # bucket_key -> indices of fr_list
    l1_sigs: List[BitSig] = []
    for i, fr in enumerate(fr_list):
        sig = encode_level1(fr, width=params.level1_bits)
        l1_sigs.append(sig)
        key = _digest_bits(sig)
        l1_map.setdefault(key, []).append(i)

    bucket_sizes = [len(ixs) for ixs in l1_map.values()]
    stats.l1_buckets = len(bucket_sizes)
    stats.avg_bucket_size_l1 = (sum(bucket_sizes) / len(bucket_sizes)) if bucket_sizes else 0.0

    # --- Level 2 screening
    l2_sigs: Dict[int, BitSig] = {}
    l2_pairs: List[Tuple[int, int]] = []
    for _, idxs in l1_map.items():
        if len(idxs) < 2:
            continue
        # Precompute L2 for bucket members
        for i in idxs:
            if i not in l2_sigs:
                l2_sigs[i] = encode_level2(fr_list[i], width=params.level2_bits, max_nodes=params.max_nodes)
        # Pairwise inside the bucket
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                i, j = idxs[a], idxs[b]
                jacc = l2_sigs[i].jaccard(l2_sigs[j])
                stats.l2_screened_pairs += 1
                if jacc >= params.min_l2_jaccard:
                    l2_pairs.append((i, j))

    # --- Level 3 final refinement
    final: List[Tuple[str, str]] = []
    l3_cache: Dict[int, BitSig] = {}
    for i, j in l2_pairs:
        if i not in l3_cache:
            l3_cache[i] = encode_level3(fr_list[i], width=params.level3_bits)
        if j not in l3_cache:
            l3_cache[j] = encode_level3(fr_list[j], width=params.level3_bits)
        sim = l3_cache[i].jaccard(l3_cache[j])  # swap for cosine if HV is dense +/-1
        stats.l3_screened_pairs += 1
        if sim >= params.min_l3_similarity:
            final.append((fr_list[i].id, fr_list[j].id))

    stats.final_pairs = len(final)
    return final, stats


def progressive_refine_lsh_pairs(
    lsh_pairs: List[Tuple[str, str]],
    function_map: Dict[str, FunctionRecord],
    *,
    params: ProgressiveParams = ProgressiveParams(),
) -> Tuple[List[Tuple[str, str]], ProgressiveStats]:
    """
    Alternative entry point: refine already-found LSH pairs through L2/L3 only.
    
    This skips Level-1 bucketing since LSH already did coarse filtering.
    More efficient when you have LSH output.
    
    Args:
        lsh_pairs: Candidate pairs from LSH (function IDs)
        function_map: Mapping from function ID to FunctionRecord
        params: Progressive encoding parameters
        
    Returns:
        Refined pairs that pass L2 and L3 thresholds, plus stats
    """
    stats = ProgressiveStats(total_functions=len(function_map))
    stats.l1_buckets = 0  # Skipped
    
    # Prepare index mapping
    id_to_record = function_map
    
    # Level 2 screening
    l2_cache: Dict[str, BitSig] = {}
    l2_passed: List[Tuple[str, str]] = []
    
    for id1, id2 in lsh_pairs:
        # Get records
        fr1 = id_to_record.get(id1)
        fr2 = id_to_record.get(id2)
        if not fr1 or not fr2:
            continue
            
        # Compute L2 signatures
        if id1 not in l2_cache:
            l2_cache[id1] = encode_level2(fr1, width=params.level2_bits)
        if id2 not in l2_cache:
            l2_cache[id2] = encode_level2(fr2, width=params.level2_bits)
            
        # Check L2 similarity
        jacc = l2_cache[id1].jaccard(l2_cache[id2])
        stats.l2_screened_pairs += 1
        
        if jacc >= params.min_l2_jaccard:
            l2_passed.append((id1, id2))
    
    # Level 3 refinement
    l3_cache: Dict[str, BitSig] = {}
    final: List[Tuple[str, str]] = []
    
    for id1, id2 in l2_passed:
        # Get records
        fr1 = id_to_record[id1]
        fr2 = id_to_record[id2]
        
        # Compute L3 signatures
        if id1 not in l3_cache:
            l3_cache[id1] = encode_level3(fr1, width=params.level3_bits)
        if id2 not in l3_cache:
            l3_cache[id2] = encode_level3(fr2, width=params.level3_bits)
            
        # Check L3 similarity
        sim = l3_cache[id1].jaccard(l3_cache[id2])
        stats.l3_screened_pairs += 1
        
        if sim >= params.min_l3_similarity:
            final.append((id1, id2))
    
    stats.final_pairs = len(final)
    return final, stats


def _digest_bits(sig: BitSig) -> str:
    """Short stable digest for bucket keys."""
    # Note: equality of digests is stronger than needed; safe for Level-1 buckets
    b = sig.bits.to_bytes((sig.width + 7) // 8, "little", signed=False)
    return hashlib.blake2b(b, digest_size=8).hexdigest()


# ---------------------------------------------------------------------
# Minimal smoke test (safe to remove)
# ---------------------------------------------------------------------

if __name__ == "__main__":
    code = """
from math import sqrt, sin, cos

def foo(x:int)->float:
    if x > 0:
        return sqrt(x) + sin(x)
    return 0.0

def bar(z:int)->float:
    if z > 1:
        return sqrt(z) + cos(z)
    return 0.0

def baz(t:str)->int:
    for _ in range(len(t)):
        pass
    return 1
"""
    mod = ast.parse(code)
    frs: List[FunctionRecord] = []
    for n in ast.walk(mod):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            frs.append(FunctionRecord(id=f"{n.name}@{n.lineno}", source=code, node=n, file="__inline__"))

    pairs, st = progressive_pairs(frs)
    print("pairs:", pairs)
    print("stats:", st)