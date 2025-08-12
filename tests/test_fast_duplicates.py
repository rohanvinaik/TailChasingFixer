"""Tests for fast duplicate detection."""

import tempfile
import ast
from pathlib import Path
import pytest
import numpy as np

from tailchasing.analyzers.fast_duplicates import (
    FastDuplicateAnalyzer, DuplicateCluster, ConsolidationPlan
)
from tailchasing.analyzers.base import AnalysisContext
from tailchasing.core.symbols import SymbolTable


class TestDuplicateCluster:
    """Test DuplicateCluster functionality."""
    
    def test_select_canonical_shortest_path(self):
        """Test that canonical selection prefers shortest path."""
        cluster = DuplicateCluster(
            cluster_id="test",
            files=[
                "/deep/nested/path/to/file.py",
                "/src/file.py",
                "/another/deep/nested/file.py"
            ]
        )
        
        canonical = cluster.select_canonical()
        assert canonical == "/src/file.py"
    
    def test_select_canonical_newest_mtime(self):
        """Test that canonical selection prefers newest files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files with different mtimes
            old_file = Path(tmpdir) / "old.py"
            old_file.write_text("old")
            
            import time
            time.sleep(0.01)
            
            new_file = Path(tmpdir) / "new.py"
            new_file.write_text("new")
            
            cluster = DuplicateCluster(
                cluster_id="test",
                files=[str(old_file), str(new_file)]
            )
            
            canonical = cluster.select_canonical()
            assert canonical == str(new_file)
    
    def test_select_canonical_empty_cluster(self):
        """Test canonical selection on empty cluster."""
        cluster = DuplicateCluster(cluster_id="test", files=[])
        canonical = cluster.select_canonical()
        assert canonical is None


class TestConsolidationPlan:
    """Test ConsolidationPlan functionality."""
    
    def test_to_yaml(self):
        """Test YAML generation for consolidation plan."""
        cluster = DuplicateCluster(
            cluster_id="test",
            files=["file1.py", "file2.py"],
            canonical_file="file1.py",
            similarity_type="exact"
        )
        
        plan = ConsolidationPlan(
            clusters=[cluster],
            total_files=10,
            redundant_files=1,
            estimated_loc_saved=100,
            actions=[{
                'type': 'remove_duplicate',
                'file': 'file2.py',
                'canonical': 'file1.py',
                'similarity': 'exact',
                'safe': True
            }]
        )
        
        yaml_str = plan.to_yaml()
        assert "duplicate_consolidation_plan" in yaml_str
        assert "total_files_analyzed: 10" in yaml_str
        assert "redundant_files: 1" in yaml_str
        assert "file1.py" in yaml_str
        assert "file2.py" in yaml_str


class TestFastDuplicateAnalyzer:
    """Test FastDuplicateAnalyzer functionality."""
    
    def test_content_hash_identical_files(self):
        """Test content hash for identical files."""
        analyzer = FastDuplicateAnalyzer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"
            
            content = "def hello():\n    print('Hello, world!')\n"
            file1.write_text(content)
            file2.write_text(content)
            
            hash1 = analyzer._compute_content_hash(str(file1))
            hash2 = analyzer._compute_content_hash(str(file2))
            
            assert hash1 == hash2
    
    def test_content_hash_different_files(self):
        """Test content hash for different files."""
        analyzer = FastDuplicateAnalyzer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file1.py"
            file2 = Path(tmpdir) / "file2.py"
            
            file1.write_text("def hello():\n    print('Hello')\n")
            file2.write_text("def goodbye():\n    print('Goodbye')\n")
            
            hash1 = analyzer._compute_content_hash(str(file1))
            hash2 = analyzer._compute_content_hash(str(file2))
            
            assert hash1 != hash2
    
    def test_shingle_hashes(self):
        """Test shingle hash computation."""
        analyzer = FastDuplicateAnalyzer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            file1 = Path(tmpdir) / "file.py"
            file1.write_text("def test_function():\n    return 42")
            
            shingles = analyzer._compute_shingle_hashes(str(file1))
            
            assert len(shingles) > 0
            assert isinstance(shingles, set)
            assert all(isinstance(s, int) for s in shingles)
    
    def test_jaccard_similarity(self):
        """Test Jaccard similarity computation."""
        analyzer = FastDuplicateAnalyzer()
        
        set1 = {1, 2, 3, 4, 5}
        set2 = {3, 4, 5, 6, 7}
        
        similarity = analyzer._jaccard_similarity(set1, set2)
        # Intersection: {3, 4, 5} = 3 elements
        # Union: {1, 2, 3, 4, 5, 6, 7} = 7 elements
        # Similarity: 3/7 ≈ 0.428
        assert 0.42 < similarity < 0.43
        
        # Test identical sets
        assert analyzer._jaccard_similarity(set1, set1) == 1.0
        
        # Test disjoint sets
        assert analyzer._jaccard_similarity({1, 2}, {3, 4}) == 0.0
        
        # Test empty sets
        assert analyzer._jaccard_similarity(set(), set()) == 1.0
    
    def test_ast_feature_extraction(self):
        """Test AST feature extraction."""
        analyzer = FastDuplicateAnalyzer()
        
        code = """
def my_function(x, y):
    return x + y

class MyClass:
    def __init__(self):
        pass

import os
from pathlib import Path
"""
        tree = ast.parse(code)
        features = analyzer._extract_ast_features(tree)
        
        assert "func:my_function(x,y)" in features
        assert "class:MyClass()" in features
        assert "import:os" in features
        assert "from:pathlib.Path" in features
    
    def test_minhash_computation(self):
        """Test MinHash signature computation."""
        analyzer = FastDuplicateAnalyzer()
        analyzer.num_perm = 16  # Smaller for testing
        
        shingles = {1, 2, 3, 4, 5}
        signature = analyzer._compute_minhash(shingles)
        
        assert len(signature) == 16
        assert signature.dtype == np.uint32
        
        # Test empty shingles
        empty_sig = analyzer._compute_minhash(set())
        assert len(empty_sig) == 16
        assert np.all(empty_sig == 0)
    
    def test_lsh_bucket_building(self):
        """Test LSH bucket construction."""
        analyzer = FastDuplicateAnalyzer()
        analyzer.bands = 2
        analyzer.rows = 4
        analyzer.num_perm = 8
        
        # Add some signatures
        sig1 = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.uint32)
        sig2 = np.array([1, 2, 3, 4, 9, 10, 11, 12], dtype=np.uint32)
        sig3 = np.array([9, 10, 11, 12, 13, 14, 15, 16], dtype=np.uint32)
        
        analyzer.minhash_signatures = {
            "file1.py": sig1,
            "file2.py": sig2,
            "file3.py": sig3
        }
        
        analyzer._build_lsh_index()
        
        # Files 1 and 2 should share at least one bucket (same first band)
        buckets_with_file1 = []
        buckets_with_file2 = []
        
        for bucket_id, files in analyzer.lsh_buckets.items():
            if "file1.py" in files:
                buckets_with_file1.append(bucket_id)
            if "file2.py" in files:
                buckets_with_file2.append(bucket_id)
        
        # Should have at least one common bucket
        common_buckets = set(buckets_with_file1) & set(buckets_with_file2)
        assert len(common_buckets) > 0
    
    def test_find_exact_duplicates(self):
        """Test exact duplicate detection."""
        analyzer = FastDuplicateAnalyzer()
        
        analyzer.content_hashes = {
            "file1.py": "hash_abc",
            "file2.py": "hash_abc",
            "file3.py": "hash_xyz",
            "file4.py": "hash_abc"
        }
        
        clusters = analyzer._find_exact_duplicates()
        
        assert len(clusters) == 1
        assert len(clusters[0].files) == 3
        assert set(clusters[0].files) == {"file1.py", "file2.py", "file4.py"}
        assert clusters[0].similarity_type == "exact"
    
    def test_merge_clusters(self):
        """Test cluster merging."""
        analyzer = FastDuplicateAnalyzer()
        
        cluster1 = DuplicateCluster(
            cluster_id="1",
            files=["file1.py", "file2.py"]
        )
        
        cluster2 = DuplicateCluster(
            cluster_id="2",
            files=["file2.py", "file3.py"]
        )
        
        cluster3 = DuplicateCluster(
            cluster_id="3",
            files=["file4.py", "file5.py"]
        )
        
        merged = analyzer._merge_clusters([cluster1, cluster2], [cluster3])
        
        # Clusters 1 and 2 should merge (both contain file2.py)
        assert len(merged) == 2
        
        # Find the merged cluster
        for cluster in merged:
            if "file1.py" in cluster.files:
                assert set(cluster.files) == {"file1.py", "file2.py", "file3.py"}
            else:
                assert set(cluster.files) == {"file4.py", "file5.py"}
    
    def test_integration_with_context(self):
        """Test integration with AnalysisContext."""
        analyzer = FastDuplicateAnalyzer()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create test files
            file1 = Path(tmpdir) / "duplicate1.py"
            file2 = Path(tmpdir) / "duplicate2.py"
            file3 = Path(tmpdir) / "unique.py"
            
            duplicate_content = "def test():\n    return 42"
            file1.write_text(duplicate_content)
            file2.write_text(duplicate_content)
            file3.write_text("def unique():\n    return 'unique'")
            
            # Create mock context
            ast_index = {
                str(file1): ast.parse(duplicate_content),
                str(file2): ast.parse(duplicate_content),
                str(file3): ast.parse("def unique():\n    return 'unique'")
            }
            
            ctx = AnalysisContext(
                config={
                    "duplicates": {
                        "enabled": True,
                        "use_fast_detection": True,
                        "generate_plan": False
                    },
                    "resource_limits": {
                        "lsh_bucket_cap": 100
                    }
                },
                root_dir=Path(tmpdir),
                file_paths=[file1, file2, file3],
                ast_index=ast_index,
                symbol_table=SymbolTable(),
                source_cache={},
                cache={}
            )
            
            issues = analyzer.run(ctx)
            
            # Should detect the duplicate
            assert len(issues) >= 1
            duplicate_issues = [i for i in issues if "duplicate" in i.kind.lower()]
            assert len(duplicate_issues) >= 1
    
    def test_performance_statistics(self):
        """Test that performance statistics are tracked."""
        analyzer = FastDuplicateAnalyzer()
        
        assert analyzer.stats['files_processed'] == 0
        assert analyzer.stats['comparisons_made'] == 0
        
        # Run some operations
        analyzer.content_hashes = {
            "file1.py": "hash1",
            "file2.py": "hash2"
        }
        analyzer._find_exact_duplicates()
        
        # Stats should be updated
        assert analyzer.stats['exact_duplicates'] == 0  # No duplicates
        
        # Test with actual duplicates
        analyzer.content_hashes = {
            "file1.py": "hash_same",
            "file2.py": "hash_same"
        }
        analyzer._find_exact_duplicates()
        
        assert analyzer.stats['exact_duplicates'] == 1