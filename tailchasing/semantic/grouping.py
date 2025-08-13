# tailchasing/semantic/grouping.py
from __future__ import annotations

import ast
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# --- Optional: align with your existing FunctionRecord; fall back if absent ---
try:
    from tailchasing.semantic.lsh_index import FunctionRecord  # type: ignore
except Exception:  # pragma: no cover
    @dataclass(frozen=True)
    class FunctionRecord:
        id: str
        source: str
        node: ast.AST
        file: Optional[str] = None
        # Optional convenience fields your pipeline might already set
        module: Optional[str] = None
        name: Optional[str] = None
        params: Optional[Tuple[str, ...]] = None
        returns: Optional[str] = None
        imports: Optional[Sequence[str]] = None


# ----------------------------
# Public API
# ----------------------------

def group_functions_smart(functions: Sequence[FunctionRecord]) -> List[List[FunctionRecord]]:
    """Group functions by multiple strategies to minimize cross-group comparisons.

    Returns a list of groups (lists of FunctionRecord). Only compare within groups.
    """
    if not functions:
        return []

    groups: Dict[str, DefaultDict[Any, List[FunctionRecord]]] = {
        "by_module": defaultdict(list),
        "by_name_pattern": defaultdict(list),
        "by_signature": defaultdict(list),
        "by_complexity": defaultdict(list),
        "by_imports": defaultdict(list),
    }

    for fr in functions:
        # --- Group by module/file ---
        mod = getattr(fr, "module", None) or _infer_module(fr)
        groups["by_module"][mod].append(fr)

        # --- Group by name pattern (get_X, set_X, handle_X, etc.) ---
        fname = getattr(fr, "name", None) or _infer_func_name(fr)
        pattern = extract_name_pattern(fname)
        groups["by_name_pattern"][pattern].append(fr)

        # --- Group by signature hash (params + return category) ---
        params, ret = _infer_signature(fr)
        sig_key = (params, ret)
        groups["by_signature"][sig_key].append(fr)

        # --- Group by complexity bucket ---
        cx = calculate_complexity(fr)
        bucket = cx // 10  # coarse bucket in steps of 10
        groups["by_complexity"][bucket].append(fr)

        # --- Group by import signature ---
        imps = tuple(sorted(_infer_imports(fr)))
        groups["by_imports"][imps].append(fr)

    # Merge the strategy-specific groups into cohesive clusters.
    return merge_groups_intelligently(groups, min_votes=2)


# ----------------------------
# Strategy helpers
# ----------------------------

_NAME_PREFIX = re.compile(r"^(get|set|handle|compute|calc|process|load|save|build|create|update|fetch|read|write|parse|render|encode|decode|train|predict|infer|match|merge|map|reduce|apply|validate|normalize|extract|transform|clean|prep|init|dispatch|route|sync|flush|collect|aggregate|score|rank|sort|filter|split|join|compare|similar|distance|hash|minhash|cluster|lsh|sample|group|bucket|index|search|query|fetch)_(.+)$", re.IGNORECASE)
_CAMEL_SPLIT = re.compile(r"(?<!^)(?=[A-Z])")

def extract_name_pattern(name: str) -> str:
    """
    Normalize name to a pattern token:
      - get_user → "get_*"
      - handle_event → "handle_*"
      - camelCase variants → treated similarly
    Fallback: first token (snake/camel) + "*", else "other".
    """
    if not name:
        return "other"

    # direct snake case prefix match
    m = _NAME_PREFIX.match(name)
    if m:
        return f"{m.group(1).lower()}_*"

    # camelCase to snake-ish
    if "_" not in name and any(c.isupper() for c in name):
        parts = _CAMEL_SPLIT.split(name)
        if parts:
            p0 = parts[0].lower()
            return f"{p0}_*"

    # generic snake case: take first token
    p0 = name.split("_", 1)[0].lower()
    return f"{p0}_*" if p0 else "other"


def calculate_complexity(func: FunctionRecord, *, max_nodes: int = 50_000) -> int:
    """
    Crude structural complexity: combination of AST depth and control-flow counts.
    If AST not available, fall back to token/length heuristics.
    """
    node = getattr(func, "node", None)
    if isinstance(node, ast.AST):
        depth = _ast_depth(node, max_nodes=max_nodes)
        cflow = _control_flow_complexity(node, max_nodes=max_nodes)
        # Weighted sum; scale to a rough 0..100+ range
        return min(999, 4 * depth + 8 * cflow)

    # Fallback: use source length + punctuation heuristics
    src = getattr(func, "source", "") or ""
    lines = src.count("\n") + 1
    ops = sum(src.count(ch) for ch in ("if", "for", "while", "try", "with"))
    return min(999, lines // 2 + 10 * ops)


def merge_groups_intelligently(
    groups: Dict[str, DefaultDict[Any, List[FunctionRecord]]],
    *,
    min_votes: int = 2,
    max_group_size: int = 5000,
) -> List[List[FunctionRecord]]:
    """
    Merge strategy-specific buckets by co-membership voting.

    Algorithm (simple & effective):
      1) For each strategy bucket, add +1 vote for every intra-bucket pair.
      2) Build connected components over the graph of functions where edge weight >= min_votes.
      3) Emit components as final groups (capped by max_group_size; large ones get split).
    """
    # Map function id -> original record
    id_to_fr: Dict[str, FunctionRecord] = {}
    for strat_buckets in groups.values():
        for bucket in strat_buckets.values():
            for fr in bucket:
                id_to_fr[fr.id] = fr
    ids = list(id_to_fr.keys())
    if len(ids) < 2:
        return [[id_to_fr[i]] for i in ids] if ids else []

    # Build adjacency with vote counts
    adj: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def vote_bucket(bucket: List[FunctionRecord]) -> None:
        if len(bucket) < 2:
            return
        # Use ids to avoid object identity issues
        ids_local = [fr.id for fr in bucket]
        ids_local = list(dict.fromkeys(ids_local))  # dedupe, preserve order
        n = len(ids_local)
        for i in range(n):
            ai = ids_local[i]
            for j in range(i + 1, n):
                bj = ids_local[j]
                adj[ai][bj] += 1
                adj[bj][ai] += 1

    for strat_buckets in groups.values():
        for bucket in strat_buckets.values():
            vote_bucket(bucket)

    # Union-Find over edges that meet min_votes
    parent: Dict[str, str] = {i: i for i in ids}
    rank: Dict[str, int] = {i: 0 for i in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1

    for a, nbrs in adj.items():
        for b, votes in nbrs.items():
            if votes >= min_votes:
                union(a, b)

    # Collect components
    comp: DefaultDict[str, List[str]] = defaultdict(list)
    for i in ids:
        comp[find(i)].append(i)

    # Convert to FunctionRecord groups; cap oversized groups by simple chunking
    out: List[List[FunctionRecord]] = []
    for members in comp.values():
        if len(members) <= max_group_size:
            out.append([id_to_fr[m] for m in members])
        else:
            for k in range(0, len(members), max_group_size):
                chunk = members[k : k + max_group_size]
                out.append([id_to_fr[m] for m in chunk])

    return out


# ----------------------------
# Inference helpers
# ----------------------------

def _infer_module(fr: FunctionRecord) -> str:
    # Prefer explicit .module, else derive from file path; fallback to "__unknown__"
    mod = getattr(fr, "module", None)
    if mod:
        return mod
    fp = getattr(fr, "file", None) or ""
    if not fp:
        return "__unknown__"
    # normalize to module-ish key (no extension)
    return fp.rsplit(".", 1)[0].replace("/", ".").replace("\\", ".")

def _infer_func_name(fr: FunctionRecord) -> str:
    n = getattr(fr, "node", None)
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return n.name
    return getattr(fr, "name", None) or fr.id.rsplit(":", 1)[0]

def _infer_signature(fr: FunctionRecord) -> Tuple[Tuple[str, ...], str]:
    """Return (param_types-ish, return_category). Very coarse & robust."""
    # If pipeline supplied direct fields, use them
    supplied_params = getattr(fr, "params", None)
    supplied_ret = getattr(fr, "returns", None)
    if supplied_params is not None and supplied_ret is not None:
        return tuple(map(str, supplied_params)), str(supplied_ret)

    n = getattr(fr, "node", None)
    if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
        args = n.args
        argc = len(args.args) + len(args.kwonlyargs)
        has_var = 1 if args.vararg else 0
        has_kw = 1 if args.kwarg else 0
        # Bucket param count for coarse signature
        argc_bucket = str(argc) if argc < 6 else "6+"
        params = (f"argc:{argc_bucket}", f"var:{has_var}", f"kw:{has_kw}")

        ret = n.returns
        if ret is None:
            ret_cat = "unknown"
        else:
            try:
                txt = ast.unparse(ret)
            except Exception:
                txt = type(ret).__name__
            t = txt.lower().replace(" ", "")
            if t in {"none", "nonetype"}:
                ret_cat = "none"
            elif "bool" in t:
                ret_cat = "bool"
            elif "int" in t:
                ret_cat = "int"
            elif "float" in t or "decimal" in t:
                ret_cat = "float"
            elif "str" in t or "bytes" in t:
                ret_cat = "str"
            elif any(k in t for k in ["list", "tuple", "set", "iter", "seq", "ndarray"]):
                ret_cat = "iterable"
            elif any(k in t for k in ["dict", "mapping"]):
                ret_cat = "mapping"
            else:
                ret_cat = "object"
        return params, ret_cat

    # Fallback: unknown signature
    return ("argc:?", "var:0", "kw:0"), "unknown"

def _infer_imports(fr: FunctionRecord) -> List[str]:
    # Use provided imports if available
    imps = getattr(fr, "imports", None)
    if imps is not None:
        return list(imps)

    # Fallback: parse module-level imports referenced by the function
    try:
        mod = ast.parse(fr.source)
    except Exception:
        return []
    imported: Dict[str, str] = {}
    for n in ast.walk(mod):
        if isinstance(n, ast.Import):
            for alias in n.names:
                local = alias.asname or alias.name.split(".")[0]
                imported[local] = alias.name
        elif isinstance(n, ast.ImportFrom):
            m = n.module or ""
            for alias in n.names:
                local = alias.asname or alias.name
                imported[local] = f"{m}.{alias.name}" if m else alias.name

    used: Set[str] = set()
    if isinstance(fr.node, ast.AST):
        for n in ast.walk(fr.node):
            if isinstance(n, ast.Name):
                used.add(n.id)
            elif isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name):
                used.add(n.value.id)

    out = []
    for local, target in imported.items():
        if local in used:
            out.append(f"{local}={target}")
    return out


# ----------------------------
# Complexity primitives
# ----------------------------

def _ast_depth(node: ast.AST, *, max_nodes: int) -> int:
    max_depth = 0
    stack: List[Tuple[ast.AST, int]] = [(node, 1)]
    visited = 0
    while stack:
        cur, d = stack.pop()
        max_depth = max(max_depth, d)
        visited += 1
        if visited > max_nodes:
            break
        for ch in ast.iter_child_nodes(cur):
            stack.append((ch, d + 1))
    return max_depth

def _control_flow_complexity(node: ast.AST, *, max_nodes: int) -> int:
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


# ----------------------------
# Minimal smoke test (safe to remove)
# ----------------------------
if __name__ == "__main__":
    demo = """
import math
from collections import defaultdict as dd

def get_user(id:int)->dict:
    if id>0:
        return {'id':id}
    return {}

def set_user(u:dict)->None:
    if 'id' in u:
        u['ok']=True

def compute_score(x):
    for _ in range(3):
        x+=1
    return x

def renderView(x): return x
"""
    mod = ast.parse(demo)
    funcs: List[FunctionRecord] = []
    for n in ast.walk(mod):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            funcs.append(FunctionRecord(id=f"{n.name}@{n.lineno}", source=demo, node=n, file="demo.py", module="demo", name=n.name))
    grouped = group_functions_smart(funcs)
    for i, g in enumerate(grouped, 1):
        print(f"Group {i}: {[f.id for f in g]}")