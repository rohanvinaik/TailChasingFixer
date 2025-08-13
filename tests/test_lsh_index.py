"""
Tests for MinHash LSH index implementation.
"""

import ast
import pytest
from typing import List

from tailchasing.semantic.lsh_index import (
    FunctionRecord,
    LSHParams,
    FeatureConfig,
    MinHasher,
    extract_shingles,
    build_lsh_index,
    precluster_for_comparison,
    create_function_records
)


class TestMinHasher:
    """Test MinHash implementation."""
    
    def test_minhash_signature_deterministic(self):
        """Test that MinHash produces deterministic signatures."""
        hasher = MinHasher(num_hashes=10, seed=42)
        
        shingles = {"a", "b", "c", "d", "e"}
        sig1 = hasher.signature(shingles)
        sig2 = hasher.signature(shingles)
        
        assert sig1 == sig2
        assert len(sig1) == 10
    
    def test_minhash_similarity(self):
        """Test that similar sets produce similar signatures."""
        hasher = MinHasher(num_hashes=100, seed=42)
        
        # Two very similar sets
        set1 = {f"token_{i}" for i in range(100)}
        set2 = {f"token_{i}" for i in range(5, 105)}  # 95% overlap
        
        sig1 = hasher.signature(set1)
        sig2 = hasher.signature(set2)
        
        # Count matching signature values
        matches = sum(1 for a, b in zip(sig1, sig2) if a == b)
        similarity = matches / len(sig1)
        
        # Should be approximately 90% similar (allowing for approximation error)
        assert 0.85 < similarity < 1.0
    
    def test_minhash_empty_set(self):
        """Test MinHash handles empty sets gracefully."""
        hasher = MinHasher(num_hashes=10, seed=42)
        
        sig = hasher.signature([])
        assert len(sig) == 10
        # All values should be max uint64
        assert all(v == 0xFFFFFFFFFFFFFFFF for v in sig)


class TestFeatureExtraction:
    """Test feature extraction from AST."""
    
    def test_extract_shingles_basic(self):
        """Test basic shingle extraction."""
        code = """
def add(a, b):
    return a + b
"""
        tree = ast.parse(code)
        func_node = tree.body[0]
        
        record = FunctionRecord(
            id="test::add@1",
            source=code,
            node=func_node,
            file="test.py"
        )
        
        shingles = extract_shingles(record)
        
        # Should contain AST 3-grams
        assert any("AST3::" in s for s in shingles)
        # Should contain call patterns (even if empty in this case)
        assert len(shingles) > 0
    
    def test_extract_shingles_with_imports(self):
        """Test shingle extraction with imports."""
        code = """
import math

def calculate(x):
    return math.sqrt(x) + math.sin(x)
"""
        tree = ast.parse(code)
        func_node = tree.body[1]  # Second item after import
        
        record = FunctionRecord(
            id="test::calculate@3",
            source=code,
            node=func_node,
            file="test.py"
        )
        
        shingles = extract_shingles(record)
        
        # Should contain import signatures
        assert any("IMP::" in s for s in shingles)
        # Should contain call patterns for math.sqrt and math.sin
        assert any("CALL::math.sqrt" in s for s in shingles)
        assert any("CALL::math.sin" in s for s in shingles)
    
    def test_extract_shingles_channels(self):
        """Test that all feature channels are represented."""
        code = """
import os
from typing import List

def process_files(files: List[str]):
    results = []
    for f in files:
        if os.path.exists(f):
            results.append(f)
    return results
"""
        tree = ast.parse(code)
        func_node = tree.body[2]  # Third item after imports
        
        record = FunctionRecord(
            id="test::process_files@4",
            source=code,
            node=func_node,
            file="test.py"
        )
        
        cfg = FeatureConfig(
            use_ast_3grams=True,
            use_imports=True,
            use_call_patterns=True
        )
        
        shingles = extract_shingles(record, cfg)
        
        # Check all channels are present
        ast_shingles = [s for s in shingles if s.startswith("AST3::")]
        imp_shingles = [s for s in shingles if s.startswith("IMP::")]
        call_shingles = [s for s in shingles if s.startswith("CALL::")]
        
        assert len(ast_shingles) > 0, "Should have AST 3-grams"
        assert len(imp_shingles) > 0, "Should have import signatures"
        assert len(call_shingles) > 0, "Should have call patterns"


class TestLSHIndex:
    """Test LSH index functionality."""
    
    def test_lsh_index_basic(self):
        """Test basic LSH index operations."""
        params = LSHParams(
            num_hashes=20,
            bands=4,
            rows_per_band=5
        )
        
        code = """
def f1(x):
    return x + 1

def f2(x):
    return x + 1

def f3(x):
    return x * 2
"""
        tree = ast.parse(code)
        
        records = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                records.append(FunctionRecord(
                    id=f"{node.name}@{node.lineno}",
                    source=code,
                    node=node,
                    file="test.py"
                ))
        
        # Build index
        index, stats = build_lsh_index(records, params=params)
        
        # Check stats
        assert stats.total_functions == 3
        assert stats.total_buckets > 0
        
        # Get candidate pairs
        pairs = list(index.candidate_pairs())
        
        # f1 and f2 are identical, should be candidates
        pair_strs = [f"{p[0]}|{p[1]}" for p in pairs]
        assert any("f1@1" in p and "f2@4" in p for p in pair_strs)
    
    def test_lsh_finds_duplicates(self):
        """Test that LSH finds duplicate functions."""
        code = """
def calculate_sum(a, b):
    result = a + b
    return result

def add_numbers(x, y):
    result = x + y
    return result

def multiply(a, b):
    return a * b
"""
        tree = ast.parse(code)
        
        records = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                records.append(FunctionRecord(
                    id=f"{node.name}@{node.lineno}",
                    source=code,
                    node=node,
                    file="test.py"
                ))
        
        # Use more aggressive parameters for testing
        params = LSHParams(
            num_hashes=50,
            bands=10,
            rows_per_band=5
        )
        
        pairs, stats = precluster_for_comparison(records, params=params)
        
        # Should find that calculate_sum and add_numbers are similar
        assert len(pairs) > 0
        assert stats.total_functions == 3
    
    def test_lsh_scalability(self):
        """Test LSH with many functions."""
        # Generate many similar and dissimilar functions
        code_parts = []
        
        # Add 50 similar functions (groups of 5)
        for group in range(10):
            for variant in range(5):
                code_parts.append(f"""
def func_{group}_{variant}(x):
    temp = x + {group}
    result = temp * 2
    return result
""")
        
        code = "\n".join(code_parts)
        tree = ast.parse(code)
        
        records = []
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                records.append(FunctionRecord(
                    id=f"{node.name}@{node.lineno}",
                    source=code,
                    node=node,
                    file="test.py"
                ))
        
        params = LSHParams(
            num_hashes=100,
            bands=20,
            rows_per_band=5
        )
        
        pairs, stats = precluster_for_comparison(records, params=params)
        
        # Should have processed all functions
        assert stats.total_functions == 50
        
        # Should find many candidate pairs within groups
        assert len(pairs) > 0
        
        # Pairs should be much less than n*(n-1)/2 = 1225
        assert len(pairs) < 500  # Should be significantly reduced


class TestIntegration:
    """Integration tests with AST index."""
    
    def test_create_function_records(self):
        """Test creating function records from AST index."""
        code = """
import math

class Calculator:
    def add(self, a, b):
        return a + b
    
    def multiply(self, a, b):
        return a * b

def standalone():
    pass
"""
        tree = ast.parse(code)
        ast_index = {"test.py": tree}
        source_cache = {"test.py": code}
        
        records = create_function_records(ast_index, source_cache)
        
        # Should find 3 functions (including methods)
        assert len(records) == 3
        
        # Check IDs are properly formatted
        ids = [r.id for r in records]
        assert all("::" in id and "@" in id for id in ids)
        
        # Check all have source and node
        assert all(r.source == code for r in records)
        assert all(isinstance(r.node, (ast.FunctionDef, ast.AsyncFunctionDef)) for r in records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])