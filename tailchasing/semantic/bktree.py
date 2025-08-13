# tailchasing/semantic/bktree.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Generic, Iterable, Iterator, List, Optional, Tuple, TypeVar

T = TypeVar("T")
DistanceFunc = Callable[[T, T], float]


@dataclass
class _BKNode(Generic[T]):
    item: T
    children: Dict[int, "_BKNode[T]"] = field(default_factory=dict)
    count: int = 1  # duplicate frequency for identical items


class BKTree(Generic[T]):
    """BK-tree for metric space similarity search.

    Notes
    -----
    - Works with any non-negative metric obeying the triangle inequality.
    - Distances are rounded to nearest int for child-edge keys (classic BK-tree layout).
      The search still uses the exact float distance for filtering.
    - Duplicate inserts increment a node counter (`count`).
    """

    def __init__(self, distance_func: DistanceFunc[T]) -> None:
        self.distance_func: DistanceFunc[T] = distance_func
        self.root: Optional[_BKNode[T]] = None
        self._size: int = 0  # total items inserted (including duplicates)

    # ----------------------------
    # Construction
    # ----------------------------
    def add(self, item: T) -> None:
        """Add an item to the tree."""
        if self.root is None:
            self.root = _BKNode(item=item)
            self._size += 1
            return

        node = self.root
        while True:
            d = self.distance_func(item, node.item)
            if d == 0:  # identical by metric
                node.count += 1
                self._size += 1
                return
            edge = int(round(d))
            child = node.children.get(edge)
            if child is None:
                node.children[edge] = _BKNode(item=item)
                self._size += 1
                return
            node = child

    def build(self, items: Iterable[T]) -> None:
        """Bulk insert items."""
        for it in items:
            self.add(it)

    # ----------------------------
    # Queries
    # ----------------------------
    def find_similar(self, item: T, threshold: float) -> List[Tuple[T, float, int]]:
        """Find all items within `threshold` distance of `item`.

        Returns a list of tuples: (matched_item, distance, count).
        `count` is the number of times the matching item was inserted.
        Results are sorted by ascending distance.
        """
        if self.root is None:
            return []

        results: List[Tuple[T, float, int]] = []
        stack: List[_BKNode[T]] = [self.root]

        while stack:
            node = stack.pop()
            d = self.distance_func(item, node.item)

            if d <= threshold:
                results.append((node.item, d, node.count))

            # Triangle inequality pruning:
            # Only explore child edges e where (d - threshold) <= e <= (d + threshold)
            lo = int(max(0, round(d - threshold)))
            hi = int(round(d + threshold))
            for edge, child in node.children.items():
                if lo <= edge <= hi:
                    stack.append(child)

        results.sort(key=lambda t: t[1])
        return results

    # ----------------------------
    # Introspection / utilities
    # ----------------------------
    def __len__(self) -> int:
        return self._size

    def is_empty(self) -> bool:
        return self.root is None

    def __bool__(self) -> bool:
        return not self.is_empty()


# ----------------------------
# Minimal smoke test
# ----------------------------
if __name__ == "__main__":
    # Example with Levenshtein distance (simple implementation for demo)
    def levenshtein(a: str, b: str) -> float:
        if a == b:
            return 0.0
        la, lb = len(a), len(b)
        if la == 0:
            return float(lb)
        if lb == 0:
            return float(la)
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev, dp[0] = dp[0], i
            for j in range(1, lb + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,      # deletion
                    dp[j - 1] + 1,  # insertion
                    prev + cost,    # substitution
                )
                prev = cur
        return float(dp[lb])

    words = ["book", "books", "cake", "boo", "cape", "boon", "cook", "cart"]
    tree = BKTree(levenshtein)
    tree.build(words)

    q = "bok"
    matches = tree.find_similar(q, threshold=2.0)
    for w, dist, cnt in matches:
        print(f"{w:>5}  d={dist:.0f}  x{cnt}")