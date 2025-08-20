# tailchasing/core/analysis_cache.py
from __future__ import annotations

import json
import hashlib
import os
import pickle
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

SCHEMA_VERSION = 1

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def _file_fingerprint(path: str) -> Tuple[int, float, int]:
    # (size, mtime, short hash-of-first-64KB) for quick dirtiness check
    st = os.stat(path)
    size, mtime = st.st_size, st.st_mtime
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(65536))
    return (size, mtime, int(h.hexdigest()[:8], 16))

def build_repo_fingerprint(root: str, patterns: Tuple[str, ...] = (".py",)) -> str:
    acc = hashlib.sha256()
    for dirpath, _, files in os.walk(root):
        for fn in files:
            if not fn.endswith(patterns):
                continue
            p = os.path.join(dirpath, fn)
            try:
                size, mtime, short = _file_fingerprint(p)
            except OSError:
                continue
            acc.update(p.encode("utf-8"))
            acc.update(size.to_bytes(8, "little", signed=False))
            acc.update(int(mtime).to_bytes(8, "little", signed=False))
            acc.update(short.to_bytes(8, "little", signed=False))
    return acc.hexdigest()

def config_fingerprint(cfg: Dict[str, Any]) -> str:
    # deterministic hash of config relevant to analyzers
    blob = json.dumps(cfg, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return _sha256_bytes(blob)

@dataclass
class AnalysisArtifact:
    schema: int
    created_at: float
    root: str
    repo_fp: str
    config_fp: str
    issues: List[Dict[str, Any]]
    stats: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), separators=(",", ":"), ensure_ascii=False)

    @staticmethod
    def from_json(s: str) -> "AnalysisArtifact":
        obj = json.loads(s)
        return AnalysisArtifact(**obj)

class IssueCache:
    def __init__(self, cache_dir: str = ".tailchasing_cache") -> None:
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.pkl_path = os.path.join(cache_dir, "analysis_artifact.pkl")
        self.json_path = os.path.join(cache_dir, "analysis_artifact.json")

    def save(self, artifact: AnalysisArtifact) -> None:
        tmp = self.pkl_path + ".tmp"
        with open(tmp, "wb") as f:
            pickle.dump(artifact, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, self.pkl_path)
        with open(self.json_path + ".tmp", "w", encoding="utf-8") as f:
            f.write(artifact.to_json())
        os.replace(self.json_path + ".tmp", self.json_path)

    def load(self) -> Optional[AnalysisArtifact]:
        try:
            with open(self.pkl_path, "rb") as f:
                art = pickle.load(f)
            if isinstance(art, AnalysisArtifact) and art.schema == SCHEMA_VERSION:
                return art
        except Exception:
            pass
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                return AnalysisArtifact.from_json(f.read())
        except Exception:
            return None

    def is_reusable(self, root: str, cfg: Dict[str, Any]) -> Tuple[bool, Optional[AnalysisArtifact], str, str]:
        art = self.load()
        if art is None:
            return (False, None, "", "")
        repo_fp = build_repo_fingerprint(root)
        cfg_fp = config_fingerprint(cfg)
        ok = (art.schema == SCHEMA_VERSION and art.root == root and
              art.repo_fp == repo_fp and art.config_fp == cfg_fp)
        return (ok, art if ok else None, repo_fp, cfg_fp)