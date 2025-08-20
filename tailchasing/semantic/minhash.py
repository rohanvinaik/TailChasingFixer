# tailchasing/semantic/minhash.py
from __future__ import annotations

import ast
import hashlib
import math
import random
from typing import Dict, Iterable, List, Sequence, Set, Tuple

# --- Optional import: align with your existing FunctionRecord, but fall back if missing ---
# Import FunctionRecord from consolidated types
from ..core.types import FunctionRecord


# =========================
# MinHash (incremental)
# =========================

class MinHash:
    """
    Simple incremental MinHash:
      - num_perm independent 64-bit permutations simulated via salted blake2b.
      - update(token_bytes) to fold in a token.
      - digest() returns the signature as a tuple[int].
    """
    __slots__ = ("num_perm", "_salts", "_mins")

    def __init__(self, num_perm: int = 128, seed: int = 0x5151_1EAF) -> None:
        if num_perm <= 0:
            raise ValueError("num_perm must be > 0")
        self.num_perm = num_perm
        rnd = random.Random(seed)
        self._salts: List[bytes] = [rnd.getrandbits(64).to_bytes(8, "little") for _ in range(num_perm)]
        # Initialize to max uint64
        self._mins: List[int] = [0xFFFFFFFFFFFFFFFF] * num_perm

    def update(self, token: bytes) -> None:
        # Fold token through all salt "permutations"
        for i, salt in enumerate(self._salts):
            hv = hashlib.blake2b(token, digest_size=8, salt=salt).digest()
            v = int.from_bytes(hv, "little")
            if v < self._mins[i]:
                self._mins[i] = v

    def merge(self, other: "MinHash") -> None:
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm mismatch")
        self._mins = [min(a, b) for a, b in zip(self._mins, other._mins)]

    def digest(self) -> Tuple[int, ...]:
        return tuple(self._mins)

    def estimate_jaccard(self, other: "MinHash") -> float:
        if self.num_perm != other.num_perm:
            raise ValueError("num_perm mismatch")
        eq = sum(1 for a, b in zip(self._mins, other._mins) if a == b)
        return eq / float(self.num_perm)


# =========================
# LSH for MinHash signatures
# =========================

def _choose_bands_rows(num_perm: int, threshold: float) -> Tuple[int, int]:
    """
    Heuristic selection of (bands, rows) given num_perm and target threshold.
    Uses the standard curve P(match) = 1 - (1 - s^r)^b with b*r = num_perm.
    We pick rows near sqrt(num_perm) and nudge to approximate threshold.
    """
    if num_perm < 16:
        return (num_perm, 1)  # degenerate: each hash is its own band
    # start near sqrt
    r = max(1, int(round(math.sqrt(num_perm))))
    # ensure divisibility
    while r > 1 and (num_perm % r) != 0:
        r -= 1
    b = num_perm // r
    # light nudge: if threshold is high, increase rows (stricter); if low, decrease rows
    if threshold >= 0.7 and r < num_perm:
        r = min(r + 1, num_perm)
        while (num_perm % r) != 0 and r < num_perm:
            r += 1
        b = num_perm // r if r <= num_perm and (num_perm % r) == 0 else b
    elif threshold <= 0.3 and r > 1:
        r = max(1, r - 1)
        while r > 1 and (num_perm % r) != 0:
            r -= 1
        b = num_perm // r
    return (b, r)


class MinHashLSH:
    """
    Lightweight LSH for MinHash signatures with banding.
    - Insert: split signature into b bands of r rows each, bucket by short digest
    - Query: collect union of co-bucketed ids, then (optionally) verify by MinHash similarity
    """
    def __init__(self, threshold: float = 0.5, num_perm: int = 128) -> None:
        self.threshold = float(threshold)
        self.num_perm = int(num_perm)
        self.bands, self.rows = _choose_bands_rows(self.num_perm, self.threshold)
        self._buckets: Dict[Tuple[int, str], List[str]] = {}
        self._sigs: Dict[str, Tuple[int, ...]] = {}

    @staticmethod
    def _band_key(rows: Sequence[int]) -> str:
        m = hashlib.blake2b(digest_size=8)
        for v in rows:
            m.update(v.to_bytes(8, "little"))
        return m.hexdigest()

    def insert(self, key: str, minhash: MinHash) -> None:
        sig = minhash.digest()
        if len(sig) != self.num_perm:
            raise ValueError("signature length != num_perm")
        self._sigs[key] = sig
        r = self.rows
        for b in range(self.bands):
            start = b * r
            end = start + r
            band = sig[start:end]
            bk = (b, self._band_key(band))
            self._buckets.setdefault(bk, []).append(key)

    def query(self, minhash: MinHash) -> List[str]:
        """
        Return candidate keys whose MinHash similarity is likely > threshold.
        We gather all items sharing at least one band; then verify using the
        MinHash estimate to filter.
        """
        sig = minhash.digest()
        r = self.rows
        cands: Set[str] = set()
        for b in range(self.bands):
            band = sig[b * r : (b + 1) * r]
            bk = (b, self._band_key(band))
            ids = self._buckets.get(bk)
            if ids:
                cands.update(ids)
        # Verify with MinHash similarity estimate
        out: List[str] = []
        # temporary MinHash from our query sig for verification
        tmp = MinHash(num_perm=self.num_perm)
        tmp._mins = list(sig)  # type: ignore[attr-defined]
        for k in cands:
            other = MinHash(num_perm=self.num_perm)
            other._mins = list(self._sigs[k])  # type: ignore[attr-defined]
            if tmp.estimate_jaccard(other) >= self.threshold:
                out.append(k)
        return out


# =========================
# High-level MinHash index
# =========================

class MinHashIndex:
    """
    Fast approximate similarity index for functions using MinHash + LSH.

    API mirrors your request:
      - add_function(func)
      - query_similar(func, threshold=0.5) -> List[str]
    """

    def __init__(self, num_perm: int = 128, threshold: float = 0.5, seed: int = 0x5151_1EAF) -> None:
        self.num_perm = num_perm
        self.seed = seed
        self.lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)

    # --- tokenization ---

    def extract_tokens(self, func: FunctionRecord) -> Iterable[str]:
        """
        Token stream used for MinHash. Keep it cheap but semantically meaningful:
          - function name tokens (snake/camel split)
          - AST node-type 3-grams within the function body
          - identifier names used
        You can override/extend this method if you want richer tokens.
        """
        # name tokens
        name = "anonymous"
        if isinstance(func.node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            name = func.node.name
        for t in _split_name_tokens(name):
            yield f"name:{t}"

        # identifiers + statement kinds
        ids: List[str] = []
        types: List[str] = []
        for n in ast.walk(func.node):
            types.append(type(n).__name__)
            if isinstance(n, ast.Name):
                ids.append(n.id.lower())

        for ident in ids:
            yield f"id:{ident}"

        # AST type 3-grams to capture local structure
        for a, b, c in zip(types, types[1:], types[2:]):
            yield f"ast3:{a}>{b}>{c}"

    # --- construction/query ---

    def create_minhash(self, func: FunctionRecord) -> MinHash:
        mh = MinHash(num_perm=self.num_perm, seed=self.seed)
        for token in self.extract_tokens(func):
            mh.update(token.encode("utf-8"))
        return mh

    def add_function(self, func: FunctionRecord) -> None:
        minhash = self.create_minhash(func)
        self.lsh.insert(func.id, minhash)

    def add_functions(self, functions: Iterable[FunctionRecord]) -> None:
        for f in functions:
            self.add_function(f)

    def query_similar(self, func: FunctionRecord, threshold: float = 0.5) -> List[str]:
        """
        Find function IDs with estimated Jaccard similarity >= threshold.
        Temporarily adjust the LSH threshold for this query if it differs.
        """
        # If threshold differs, do a local verify pass after LSH
        qmh = self.create_minhash(func)

        # Fast candidate retrieval with current LSH threshold
        # (LSH is built with its own threshold; we still post-filter to user 'threshold')
        candidates = self.lsh.query(qmh)
        if not candidates:
            return []

        # Verify against requested threshold (could be stricter/looser)
        out: List[str] = []
        for key in candidates:
            other = MinHash(self.num_perm)
            other._mins = list(self.lsh._sigs[key])  # type: ignore[attr-defined]
            if qmh.estimate_jaccard(other) >= threshold:
                out.append(key)
        return out


# =========================
# Utilities
# =========================

_CAMEL_SPLIT = None
def _split_name_tokens(name: str) -> List[str]:
    global _CAMEL_SPLIT
    if _CAMEL_SPLIT is None:
        import re
        _CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")
    snake = name.replace("__", "_").strip("_")
    parts: List[str] = []
    for p in snake.split("_"):
        if not p:
            continue
        parts.extend(_CAMEL_SPLIT.split(p))
    toks = [t.lower() for t in parts if t]
    # Light stoplist
    stop = {"get", "set", "make", "build", "create", "compute", "calc", "init", "update", "process", "run"}
    return [t for t in toks if t not in stop]


# =========================
# Smoke test (safe to remove)
# =========================

if __name__ == "__main__":
    demo = """
from math import sqrt, sin, cos
def foo_value(x:int)->float:
    return sqrt(x) + sin(x)
def bar_value(z:int)->float:
    return sqrt(z) + cos(z)
def baz_name(s:str)->int:
    for _ in range(len(s)):
        pass
    return 1
"""
    mod = ast.parse(demo)
    funcs: List[FunctionRecord] = []
    for n in ast.walk(mod):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(FunctionRecord(id=f"{n.name}@{n.lineno}", source=demo, node=n, file="__inline__"))

    idx = MinHashIndex(num_perm=128, threshold=0.5)
    idx.add_functions(funcs)

    q = funcs[0]  # foo_value
    print("Similar to", q.id, "→", idx.query_similar(q, threshold=0.4))