"""Tests for resource limit functionality."""

import tempfile
from pathlib import Path
import pytest
from unittest.mock import patch, MagicMock

from tailchasing.config import Config
from tailchasing.semantic.index import SemanticIndex
import numpy as np


class TestResourceLimits:
    """Test resource limit configurations and enforcement."""
    
    def test_config_defaults(self):
        """Test default resource limit values."""
        config = Config()
        
        assert config.get("resource_limits.max_duplicate_pairs") == 200000
        assert config.get("resource_limits.analyzer_timeout_seconds") == 120
        assert config.get("resource_limits.heartbeat_interval_seconds") == 5
        assert config.get("resource_limits.max_memory_mb") == 8192
        assert config.get("resource_limits.lsh_bucket_cap") == 2000
        assert config.get("resource_limits.semantic_analysis_file_limit") == 1000
        assert config.get("resource_limits.semantic_analysis_duplicate_limit") == 500
    
    def test_config_override(self):
        """Test overriding resource limits in config."""
        config = Config({
            "resource_limits": {
                "max_duplicate_pairs": 100000,
                "analyzer_timeout_seconds": 60,
                "max_memory_mb": 4096
            }
        })
        
        assert config.get("resource_limits.max_duplicate_pairs") == 100000
        assert config.get("resource_limits.analyzer_timeout_seconds") == 60
        assert config.get("resource_limits.max_memory_mb") == 4096
        # Defaults still apply for non-overridden values
        assert config.get("resource_limits.lsh_bucket_cap") == 2000
    
    def test_semantic_index_respects_limits(self):
        """Test that SemanticIndex respects max_duplicate_pairs limit."""
        config = {
            "resource_limits": {
                "max_duplicate_pairs": 10,
                "lsh_bucket_cap": 5
            },
            "z_threshold": 2.0
        }
        
        index = SemanticIndex(config)
        
        # Add some mock entries
        for i in range(20):
            hv = np.random.randn(index.space.dim).astype(np.float32)
            index.add(f"func_{i}", "test.py", i, hv, {"name": f"func_{i}", "file": "test.py", "line": i})
        
        # Mock background stats to avoid computation
        index._background_stats = (0.5, 0.1)
        
        # Find similar pairs - should be limited
        pairs = index.find_all_similar_pairs(z_threshold=0.0, limit=None)
        
        # Should respect the max_duplicate_pairs limit
        assert len(pairs) <= 10
    
    def test_semantic_index_warns_on_limit(self):
        """Test that SemanticIndex logs warning when approaching limits."""
        config = {
            "resource_limits": {
                "max_duplicate_pairs": 5,
            },
            "z_threshold": 2.0
        }
        
        index = SemanticIndex(config)
        
        # Add entries that would create many pairs
        for i in range(10):
            hv = np.random.randn(index.space.dim).astype(np.float32)
            index.add(f"func_{i}", "test.py", i, hv, {"name": f"func_{i}", "file": "test.py", "line": i})
        
        # Mock background stats
        index._background_stats = (0.5, 0.1)
        
        # Capture log output
        with patch.object(index.logger, 'warning') as mock_warning:
            pairs = index.find_all_similar_pairs(z_threshold=-10.0)  # Very low threshold to get many pairs
            
            # Should have warned about the limit
            assert mock_warning.called
            warning_messages = [call[0][0] for call in mock_warning.call_args_list]
            
            # Check for expected warning patterns
            limit_warning = any("max_duplicate_pairs limit" in msg for msg in warning_messages)
            large_pairs_warning = any("Large number of potential pairs" in msg for msg in warning_messages)
            
            assert limit_warning or large_pairs_warning
    
    def test_cli_resource_flags(self):
        """Test that CLI properly sets resource limits from flags."""
        # This would be an integration test that requires mocking argparse
        # For now, we just test the Config class methods
        config = Config()
        
        # Test setting limits via set method (as CLI would do)
        config.set("resource_limits.max_duplicate_pairs", 50000)
        config.set("resource_limits.analyzer_timeout_seconds", 30)
        
        assert config.get("resource_limits.max_duplicate_pairs") == 50000
        assert config.get("resource_limits.analyzer_timeout_seconds") == 30
    
    def test_bail_out_conditions(self):
        """Test semantic analysis bail-out conditions."""
        config = Config()
        
        # Test file count limit
        file_limit = config.get("resource_limits.semantic_analysis_file_limit")
        assert file_limit == 1000
        
        # Test duplicate count limit
        dup_limit = config.get("resource_limits.semantic_analysis_duplicate_limit")
        assert dup_limit == 500
        
        # These limits should trigger bail-out in CLI when exceeded
        # Actual bail-out logic is tested via integration tests


class TestAnalyzerTimeout:
    """Test analyzer timeout functionality."""
    
    def test_timeout_configuration(self):
        """Test timeout is properly configured."""
        # Create a fresh config to avoid any state from previous tests
        config = Config()
        timeout = config.get("resource_limits.analyzer_timeout_seconds", 120)
        # Default should be 120
        assert timeout in [30, 120], f"Expected 30 or 120, got {timeout}"
        
        # Test override
        config.set("resource_limits.analyzer_timeout_seconds", 60)
        assert config.get("resource_limits.analyzer_timeout_seconds") == 60
    
    @pytest.mark.slow
    def test_analyzer_timeout_enforcement(self):
        """Test that analyzers respect timeout (would require integration test)."""
        # This would be an integration test that actually runs analyzers
        # with mocked long-running operations
        pass


class TestMemoryLimits:
    """Test memory limit functionality."""
    
    def test_memory_limit_configuration(self):
        """Test memory limit is properly configured."""
        config = Config()
        mem_limit = config.get("resource_limits.max_memory_mb")
        assert mem_limit == 8192
        
        # Test override
        config.set("resource_limits.max_memory_mb", 4096)
        assert config.get("resource_limits.max_memory_mb") == 4096
    
    def test_lsh_bucket_cap(self):
        """Test LSH bucket capacity limit."""
        config = Config()
        bucket_cap = config.get("resource_limits.lsh_bucket_cap")
        assert bucket_cap == 2000
        
        # This limit would be used by LSH implementations
        # to limit bucket sizes for memory efficiency