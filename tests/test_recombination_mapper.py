"""
Test suite for recombination mapper and co-edit analysis.

Tests genetic recombination mapping applied to code evolution,
linkage disequilibrium computation, and insulator generation.
"""

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Set, Any
from dataclasses import dataclass
from unittest.mock import Mock, patch
from datetime import datetime, timedelta

from tailchasing.analyzers.recombination_mapper import (
    RecombinationMapper, InsulatorGenerator, CoEditEvent, 
    LinkageGroup, ExtractionROI, ModuleBoundary, enhance_chromatin_analyzer
)
from tailchasing.core.issues import Issue


class TestRecombinationMapper:
    """Test genetic recombination mapping for co-edit analysis."""
    
    def setup_method(self):
        """Set up test environment."""
        self.mapper = RecombinationMapper()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_build_coedit_matrix_basic(self):
        """Test basic co-edit matrix construction."""
        git_history = [
            CoEditEvent("commit1", datetime.now(), {"file1.py", "file2.py"}, "author1", "msg1", {"file1.py": 10, "file2.py": 5}),
            CoEditEvent("commit2", datetime.now(), {"file2.py", "file3.py"}, "author1", "msg2", {"file2.py": 8, "file3.py": 12}),
            CoEditEvent("commit3", datetime.now(), {"file1.py", "file3.py"}, "author1", "msg3", {"file1.py": 6, "file3.py": 7}),
        ]
        
        file_list = ["file1.py", "file2.py", "file3.py"]
        matrix = self.mapper.build_coedit_matrix(git_history, file_list)
        
        assert matrix.shape == (3, 3)
        assert matrix[0, 1] > 0  # file1-file2 co-edited 
        assert matrix[1, 2] > 0  # file2-file3 co-edited
        assert matrix[0, 2] > 0  # file1-file3 co-edited
        assert np.all(matrix >= 0)
        assert np.allclose(np.diag(matrix), 1.0)  # Diagonal should be 1.0
        
    def test_build_coedit_matrix_frequency_weighting(self):
        """Test frequency weighting in co-edit matrix."""
        git_history = [
            CoEditEvent("commit1", datetime.now(), {"file1.py", "file2.py"}, "author1", "msg1", {"file1.py": 10, "file2.py": 5}),
            CoEditEvent("commit2", datetime.now(), {"file1.py", "file2.py"}, "author1", "msg2", {"file1.py": 8, "file2.py": 7}),  # Same pair again
            CoEditEvent("commit3", datetime.now(), {"file2.py", "file3.py"}, "author1", "msg3", {"file2.py": 6, "file3.py": 9}),
        ]
        
        file_list = ["file1.py", "file2.py", "file3.py"]
        matrix = self.mapper.build_coedit_matrix(git_history, file_list)
        
        # file1-file2 appears twice, file2-file3 appears once, but normalization affects this
        # Just check that they both have reasonable frequencies
        assert matrix[0, 1] >= matrix[1, 2]  # Should be at least equal or higher
        # Matrix should be symmetric
        assert np.allclose(matrix, matrix.T)
        
    def test_compute_linkage_disequilibrium_basic(self):
        """Test basic linkage disequilibrium computation."""
        # Create simple co-edit matrix
        coedit_matrix = np.array([
            [1.0, 0.8, 0.3],
            [0.8, 1.0, 0.4], 
            [0.3, 0.4, 1.0]
        ])
        
        file_list = ["file1.py", "file2.py", "file3.py"]
        ld_matrix = self.mapper.compute_linkage_disequilibrium(coedit_matrix, file_list)
        
        assert isinstance(ld_matrix, np.ndarray)
        assert ld_matrix.shape == (3, 3)
        assert np.all(-1 <= ld_matrix)
        assert np.all(ld_matrix <= 1)
        assert np.allclose(np.diag(ld_matrix), 1.0)  # Diagonal should be 1.0
        
    def test_identify_linkage_groups_basic(self):
        """Test linkage group identification."""
        # Create co-edit matrix with clear groups
        coedit_matrix = np.array([
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.7],
            [0.1, 0.1, 0.7, 1.0]
        ])
        
        file_list = ["file1.py", "file2.py", "file3.py", "file4.py"]
        linkage_groups = self.mapper.identify_linkage_groups(coedit_matrix, file_list, threshold=0.6)
        
        assert len(linkage_groups) >= 1  # Should find at least one group
        for group in linkage_groups:
            assert isinstance(group, LinkageGroup)
            assert len(group.files) >= 2
            assert 0.0 <= group.coedit_frequency <= 1.0
            assert 0.0 <= group.recombination_rate <= 1.0
    
    def test_predict_extraction_roi_with_context(self):
        """Test extraction ROI prediction."""
        # Create linkage groups
        linkage_groups = [
            LinkageGroup(
                files={"file1.py", "file2.py"},
                coedit_frequency=0.8,
                recombination_rate=0.2,
                extraction_candidate=True,
                shared_functionality=["helper_func"]
            )
        ]
        
        ld_matrix = np.array([[1.0, 0.8, 0.2], [0.8, 1.0, 0.2], [0.2, 0.2, 1.0]])
        file_list = ["file1.py", "file2.py", "file3.py"]
        
        # Mock analysis context
        from tailchasing.analyzers.base import AnalysisContext
        from tailchasing.core.symbols import SymbolTable
        context = AnalysisContext(
            config={}, root_dir=Path("."), file_paths=[], 
            ast_index={}, symbol_table=SymbolTable(), 
            source_cache={}, cache={}
        )
        
        rois = self.mapper.predict_extraction_roi(linkage_groups, ld_matrix, file_list, context)
        
        assert isinstance(rois, list)
        # ROI generation depends on extraction benefit threshold and shared symbols
        for roi in rois:
            assert isinstance(roi, ExtractionROI)
            assert 0.0 <= roi.extraction_benefit <= 1.0
            assert 0.0 <= roi.insulation_score <= 1.0
    
    def test_extract_git_history_basic(self):
        """Test git history extraction."""
        working_dir = Path(self.temp_dir)
        
        # Mock git command that returns no history
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 1  # Git command fails
            mock_run.return_value.stderr = "not a git repository"
            
            events = self.mapper.extract_git_history(working_dir)
            assert events == []
            
        # Mock successful git command
        with patch('subprocess.run') as mock_run:
            mock_run.return_value.returncode = 0
            mock_run.return_value.stdout = "abc123|2023-01-01 10:00:00|author|commit message\nM\tfile1.py\n\n"
            
            events = self.mapper.extract_git_history(working_dir)
            assert isinstance(events, list)


class TestInsulatorGenerator:
    """Test insulator generation for module boundaries."""
    
    def setup_method(self):
        """Set up test environment."""
        self.generator = InsulatorGenerator()
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_add_module_boundaries_basic(self):
        """Test basic module boundary addition."""
        from tailchasing.analyzers.base import AnalysisContext
        from tailchasing.core.symbols import SymbolTable
        
        context = AnalysisContext(
            config={}, root_dir=Path("."), file_paths=[], 
            ast_index={}, symbol_table=SymbolTable(), 
            source_cache={}, cache={}
        )
        
        boundary = self.generator.add_module_boundaries("test_module.py", context)
        
        assert isinstance(boundary, ModuleBoundary)
        assert boundary.module_path == "test_module.py"
        assert boundary.boundary_type in ["facade", "common", "export", "simple"]
        assert 0.0 <= boundary.insulation_score <= 1.0
        assert isinstance(boundary.suggested_exports, set)
        
    def test_compute_insulation_score_basic(self):
        """Test insulation score computation."""
        from tailchasing.analyzers.base import AnalysisContext
        from tailchasing.core.symbols import SymbolTable
        
        context = AnalysisContext(
            config={}, root_dir=Path("."), file_paths=[], 
            ast_index={}, symbol_table=SymbolTable(), 
            source_cache={}, cache={}
        )
        
        score = self.generator.compute_insulation_score("test_module.py", context)
        
        assert 0.0 <= score <= 1.0
        
    def test_generate_facade_file_content(self):
        """Test facade file generation."""
        boundary = ModuleBoundary(
            module_path="test_module.py",
            boundary_type="facade",
            insulation_score=0.7,
            suggested_exports={"TestClass", "test_function"},
            facade_content="facade",
            common_extractions=set()
        )
        
        content = self.generator.generate_facade_file("test_module.py", boundary)
        
        assert isinstance(content, str)
        assert "TestClass" in content
        assert "test_function" in content
        assert "__all__" in content
        assert "facade" in content.lower()
        
    def test_generate_common_file_content(self):
        """Test common module file generation."""
        extraction_roi = ExtractionROI(
            target_files={"file1.py", "file2.py"},
            shared_symbols={"CONSTANT", "helper_function", "HelperClass"},
            extraction_benefit=0.8,
            proposed_helper_name="shared_helpers",
            insulation_score=0.6,
            boundary_suggestions=["Add exports"]
        )
        
        content = self.generator.generate_common_file(extraction_roi)
        
        assert isinstance(content, str)
        assert "CONSTANT" in content
        assert "helper_function" in content
        assert "HelperClass" in content
        assert "__all__" in content
        assert "shared_helpers" in content


class TestRecombinationMapperIntegration:
    """Integration tests for recombination mapper with other components."""
    
    def setup_method(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        
    def teardown_method(self):
        """Clean up test environment."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_enhance_chromatin_analyzer_integration(self):
        """Test integration with ChromatinContactAnalyzer."""
        # Mock ChromatinContactAnalyzer
        chromatin_analyzer = Mock()
        chromatin_analyzer.polymer_distance = Mock(return_value=1.0)
        
        mapper = RecombinationMapper()
        
        # Enhance with recombination mapping
        enhance_chromatin_analyzer(chromatin_analyzer, mapper)
        
        # Should have recombination mapper attached
        assert hasattr(chromatin_analyzer, '_recombination_mapper')
        assert chromatin_analyzer._recombination_mapper is mapper
        
    def test_end_to_end_recombination_analysis(self):
        """Test complete recombination analysis workflow."""
        # Create sample git history
        git_history = [
            CoEditEvent("commit1", datetime.now(), {"src/auth.py", "src/user.py"}, "author1", "msg1", {"src/auth.py": 20, "src/user.py": 15}),
            CoEditEvent("commit2", datetime.now(), {"src/auth.py", "src/user.py"}, "author1", "msg2", {"src/auth.py": 10, "src/user.py": 8}),
            CoEditEvent("commit3", datetime.now(), {"src/utils.py", "src/helpers.py"}, "author1", "msg3", {"src/utils.py": 12, "src/helpers.py": 18}),
        ]
        
        file_list = ["src/auth.py", "src/user.py", "src/utils.py", "src/helpers.py"]
        
        mapper = RecombinationMapper()
        generator = InsulatorGenerator()
        
        # Build co-edit matrix
        coedit_matrix = mapper.build_coedit_matrix(git_history, file_list)
        
        # Compute linkage disequilibrium
        ld_matrix = mapper.compute_linkage_disequilibrium(coedit_matrix, file_list)
        
        # Identify linkage groups
        linkage_groups = mapper.identify_linkage_groups(coedit_matrix, file_list)
        
        # Mock context for ROI prediction
        from tailchasing.analyzers.base import AnalysisContext
        from tailchasing.core.symbols import SymbolTable
        context = AnalysisContext(
            config={}, root_dir=Path("."), file_paths=[], 
            ast_index={}, symbol_table=SymbolTable(), 
            source_cache={}, cache={}
        )
        
        # Predict extraction ROI
        rois = mapper.predict_extraction_roi(linkage_groups, ld_matrix, file_list, context)
        
        # Generate boundaries
        boundary = generator.add_module_boundaries("src/auth.py", context)
        
        # Verify complete workflow
        assert coedit_matrix.shape == (4, 4)
        assert isinstance(ld_matrix, np.ndarray)
        assert isinstance(linkage_groups, list)
        assert isinstance(rois, list)
        assert isinstance(boundary, ModuleBoundary)


class TestCoEditEventProcessing:
    """Test co-edit event processing and temporal analysis."""
    
    def test_coedit_event_creation(self):
        """Test CoEditEvent dataclass creation."""
        event = CoEditEvent(
            commit_hash="abc123", 
            timestamp=datetime.now(),
            files={"file1.py", "file2.py"}, 
            author="test_author",
            message="test commit",
            lines_changed={"file1.py": 10, "file2.py": 5}
        )
        
        assert event.commit_hash == "abc123"
        assert event.files == {"file1.py", "file2.py"}
        assert event.author == "test_author"
        assert event.message == "test commit"
        assert event.lines_changed["file1.py"] == 10
        
    def test_large_commit_handling(self):
        """Test handling of large commits with many files."""
        mapper = RecombinationMapper()
        
        # Large commit touching many files
        large_commit_files = {f"module_{i}.py" for i in range(10)}
        git_history = [
            CoEditEvent(
                "large_commit", datetime.now(), large_commit_files, 
                "author1", "large refactor", 
                {f: 10 for f in large_commit_files}
            )
        ]
        
        file_list = list(large_commit_files)
        matrix = mapper.build_coedit_matrix(git_history, file_list)
        
        # All files should be linked to each other
        assert matrix.shape == (10, 10)
        
        # Diagonal should be 1.0 (files always change with themselves)
        assert np.allclose(np.diag(matrix), 1.0)
        
        # All off-diagonal elements should be positive (all files co-edited)
        mask = ~np.eye(10, dtype=bool)
        assert np.all(matrix[mask] > 0)


if __name__ == "__main__":
    pytest.main([__file__])