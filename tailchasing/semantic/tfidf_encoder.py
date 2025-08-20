# tailchasing/semantic/tfidf_encoder.py
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple, Optional, Sequence, Union

# ----------------------------
# Optional sklearn backend
# ----------------------------
_SKLEARN_AVAILABLE = False
try:  # pragma: no cover
    from sklearn.feature_extraction.text import TfidfVectorizer  # type: ignore
    _SKLEARN_AVAILABLE = True
except Exception:  # pragma: no cover
    TfidfVectorizer = None  # type: ignore


# ----------------------------
# Pure-Python TF-IDF
# ----------------------------

class SimpleTFIDF:
    """Lightweight TF-IDF without sklearn dependency.

    Parameters
    ----------
    lowercase : bool
        Lowercase text before tokenization.
    token_pattern : str
        Regex used to extract tokens. Default keeps >=2 char "word" tokens.
    stop_words : Optional[set[str]]
        Tokens to drop after tokenization.
    ngram_range : Tuple[int, int]
        Inclusive n-gram range; (1,1) = unigrams only.
    use_idf : bool
    smooth_idf : bool
        If True, add 1 to document frequencies ("1 + df") for IDF smoothing.
    sublinear_tf : bool
        If True, use 1 + log(tf) instead of raw term frequency.
    norm : Optional[str]
        'l2' (default) to L2-normalize vectors; None to skip normalization.
    min_df : int
        Minimum document frequency to keep a term.
    max_df : Union[int, float]
        If int: max absolute document frequency; if float in (0,1]: max proportion.
    """

    def __init__(
        self,
        *,
        lowercase: bool = True,
        token_pattern: str = r"\b\w{2,}\b",
        stop_words: Optional[set] = None,
        ngram_range: Tuple[int, int] = (1, 1),
        use_idf: bool = True,
        smooth_idf: bool = True,
        sublinear_tf: bool = False,
        norm: Optional[str] = "l2",
        min_df: int = 1,
        max_df: Union[int, float] = 1.0,
    ) -> None:
        self.lowercase = lowercase
        self.token_pattern = re.compile(token_pattern)
        self.stop_words = stop_words or set()
        self.ngram_range = ngram_range
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.norm = norm
        self.min_df = int(min_df)
        self.max_df = max_df

        self.vocabulary: Dict[str, int] = {}   # term -> index
        self.feature_names_: List[str] = []    # index -> term
        self.idf_: List[float] = []            # index -> idf

    # -------- public API --------

    def fit(self, documents: Sequence[str]) -> "SimpleTFIDF":
        df_counter, per_doc_terms = self._scan_documents(documents)
        n_docs = max(1, len(documents))

        # df thresholding
        max_df_abs = self._resolve_max_df(n_docs)
        kept_terms = [
            t for t, df in df_counter.items()
            if df >= self.min_df and df <= max_df_abs
        ]

        kept_terms.sort()
        self.vocabulary = {t: i for i, t in enumerate(kept_terms)}
        self.feature_names_ = kept_terms

        # compute IDF
        if self.use_idf:
            self.idf_ = [0.0] * len(kept_terms)
            for t, i in self.vocabulary.items():
                df = df_counter[t]
                if self.smooth_idf:
                    # classic: log((1 + n) / (1 + df)) + 1
                    self.idf_[i] = math.log((1.0 + n_docs) / (1.0 + df)) + 1.0
                else:
                    self.idf_[i] = math.log(n_docs / max(1, df))
        else:
            self.idf_ = [1.0] * len(kept_terms)

        return self

    def transform(self, documents: Sequence[str]) -> List[Dict[int, float]]:
        """Return sparse row vectors: list of {feature_index: tfidf_weight}."""
        vectors: List[Dict[int, float]] = []
        for tokens in self._tokens_iter(documents):
            tf = Counter(t for t in tokens if t in self.vocabulary)
            vec: Dict[int, float] = {}
            for term, f in tf.items():
                j = self.vocabulary[term]
                tf_val = (1.0 + math.log(f)) if self.sublinear_tf else float(f)
                vec[j] = tf_val * self.idf_[j]
            if self.norm == "l2" and vec:
                self._l2_normalize(vec)
            vectors.append(vec)
        return vectors

    def fit_transform(self, documents: Sequence[str]) -> List[Dict[int, float]]:
        self.fit(documents)
        return self.transform(documents)

    # sklearn-ish compatibility bits
    def get_feature_names_out(self) -> List[str]:
        return list(self.feature_names_)

    # -------- utilities --------

    def tokenize(self, doc: str) -> List[str]:
        if self.lowercase:
            doc = doc.lower()
        toks = [m.group(0) for m in self.token_pattern.finditer(doc)]
        toks = [t for t in toks if t not in self.stop_words]
        n1, n2 = self.ngram_range
        if (n1, n2) == (1, 1):
            return toks
        grams: List[str] = []
        for n in range(n1, n2 + 1):
            if n == 1:
                grams.extend(toks)
            else:
                for i in range(len(toks) - n + 1):
                    grams.append(" ".join(toks[i : i + n]))
        return grams

    def _tokens_iter(self, documents: Sequence[str]) -> Iterable[List[str]]:
        for doc in documents:
            yield self.tokenize(doc or "")

    def _scan_documents(self, documents: Sequence[str]) -> Tuple[Dict[str, int], List[set]]:
        doc_freq: Dict[str, int] = defaultdict(int)
        per_doc_terms: List[set] = []
        for tokens in self._tokens_iter(documents):
            terms = set(tokens)
            per_doc_terms.append(terms)
            for t in terms:
                doc_freq[t] += 1
        return doc_freq, per_doc_terms

    def _resolve_max_df(self, n_docs: int) -> int:
        if isinstance(self.max_df, float):
            if not (0.0 < self.max_df <= 1.0):
                raise ValueError("max_df as float must be in (0, 1].")
            return int(math.floor(self.max_df * n_docs))
        return int(self.max_df)

    @staticmethod
    def _l2_normalize(vec: Dict[int, float]) -> None:
        s = math.sqrt(sum(v * v for v in vec.values()))
        if s > 0:
            inv = 1.0 / s
            for k in list(vec.keys()):
                vec[k] *= inv

    # conversion helper
    def to_dense(self, vectors: List[Dict[int, float]]) -> List[List[float]]:
        m = len(vectors)
        n = len(self.vocabulary)
        dense = [[0.0] * n for _ in range(m)]
        for i, row in enumerate(vectors):
            for j, val in row.items():
                dense[i][j] = val
        return dense


# ----------------------------
# Convenience backend selector
# ----------------------------

@dataclass
class TFIDFResult:
    """Unified output for both sklearn and SimpleTFIDF."""
    vectors: Union[List[Dict[int, float]], "scipy.sparse.csr_matrix", List[List[float]]]
    feature_names: List[str]
    is_sparse: bool

    def to_dense(self) -> List[List[float]]:
        if isinstance(self.vectors, list):
            # either list-of-dicts or list-of-lists
            if self.vectors and isinstance(self.vectors[0], dict):
                # list of sparse dicts
                n = len(self.feature_names)
                out = [[0.0] * n for _ in range(len(self.vectors))]
                for i, row in enumerate(self.vectors):
                    for j, v in row.items():
                        out[i][j] = v
                return out
            return self.vectors  # already dense list-of-lists
        # scipy sparse matrix path
        try:  # pragma: no cover
            return self.vectors.toarray().tolist()
        except Exception:
            # last resort: manual iteration
            coo = self.vectors.tocoo()
            out = [[0.0] * len(self.feature_names) for _ in range(self.vectors.shape[0])]
            for i, j, v in zip(coo.row, coo.col, coo.data):
                out[int(i)][int(j)] = float(v)
            return out


def tfidf_vectorize(
    documents: Sequence[str],
    *,
    prefer_sklearn: bool = True,
    **kwargs
) -> TFIDFResult:
    """
    Build TF-IDF representation of `documents`, preferring sklearn when available.

    Returns
    -------
    TFIDFResult with:
      - vectors: list-of-dict (simple) or scipy.sparse matrix (sklearn)
      - feature_names: list[str]
      - is_sparse: bool
    """
    if prefer_sklearn and _SKLEARN_AVAILABLE:  # pragma: no cover
        vec = TfidfVectorizer(**kwargs)
        X = vec.fit_transform(documents)
        names = list(getattr(vec, "get_feature_names_out")())
        return TFIDFResult(vectors=X, feature_names=names, is_sparse=True)

    # Fallback to pure-Python
    tf = SimpleTFIDF(**kwargs)
    sparse_rows = tf.fit_transform(documents)
    return TFIDFResult(vectors=sparse_rows, feature_names=tf.get_feature_names_out(), is_sparse=True)


# ----------------------------
# Minimal smoke test
# ----------------------------
if __name__ == "__main__":
    docs = [
        "The quick brown fox jumps over the lazy dog",
        "The quick red fox jumped over a sleeping dog",
        "Graph embeddings for code search and semantic similarity",
    ]
    res = tfidf_vectorize(docs, ngram_range=(1, 2), prefer_sklearn=False)
    print("features:", len(res.feature_names))
    dense = res.to_dense()
    print("shape:", len(dense), "x", len(dense[0]))