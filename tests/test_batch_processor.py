"""
Tests for batch processing system.
"""

import json
import time
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch, call
import tempfile
import shutil

import pytest

from tailchasing.core.batch_processor import (
    BatchProcessor,
    BatchInfo,
    ProcessingStage,
    CheckpointData,
    ProcessingStats
)
from tailchasing.core.issues import Issue, IssueCollection
from tailchasing.analyzers.base import AnalysisContext


class TestBatchProcessor:
    """Test the batch processor."""
    
    def test_group_files_by_module(self):
        """Test grouping files by module."""
        processor = BatchProcessor()
        
        # Create mock file structure
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create module structure
            (tmpdir / "module1").mkdir()
            (tmpdir / "module1" / "__init__.py").touch()
            (tmpdir / "module1" / "file1.py").touch()
            (tmpdir / "module1" / "file2.py").touch()
            
            (tmpdir / "module2").mkdir()
            (tmpdir / "module2" / "__init__.py").touch()
            (tmpdir / "module2" / "file1.py").touch()
            
            (tmpdir / "root_file.py").touch()
            
            files = [
                tmpdir / "module1" / "file1.py",
                tmpdir / "module1" / "file2.py",
                tmpdir / "module2" / "file1.py",
                tmpdir / "root_file.py"
            ]
            
            groups = processor.group_files_by_module(files)
            
            assert "module1" in groups
            assert len(groups["module1"]) == 2
            assert "module2" in groups
            assert len(groups["module2"]) == 1
            assert "root" in groups
            assert len(groups["root"]) == 1
    
    def test_create_batches(self):
        """Test batch creation."""
        processor = BatchProcessor(batch_size=2)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test files
            files = []
            for i in range(5):
                file_path = tmpdir / f"file{i}.py"
                file_path.write_text(f"# File {i}\n" * 10)
                files.append(file_path)
            
            batches = processor.create_batches(files)
            
            # Should create 3 batches (2, 2, 1)
            assert len(batches) == 3
            assert batches[0].batch_id == 0
            assert len(batches[0].files) == 2
            assert len(batches[1].files) == 2
            assert len(batches[2].files) == 1
            
            # Check batch info
            for batch in batches:
                assert batch.total_size > 0
                assert batch.avg_file_size > 0
                assert batch.module_name == "root"
    
    def test_priority_calculation(self):
        """Test batch priority calculation."""
        processor = BatchProcessor()
        
        # Test different module names
        assert processor._calculate_priority("test", 1000, 5) > 0
        assert processor._calculate_priority("core", 1000, 5) > 0
        assert processor._calculate_priority("random", 1000, 5) >= 0
        
        # Test module should have higher priority
        test_priority = processor._calculate_priority("test", 5000, 10)
        other_priority = processor._calculate_priority("other", 5000, 10)
        assert test_priority > other_priority
    
    def test_process_stage(self):
        """Test processing a single stage."""
        processor = BatchProcessor()
        
        # Create mock stage
        stage = ProcessingStage(
            name="test_stage",
            description="Test stage",
            priority=1,
            analyzer_names=["test_analyzer"],
            estimated_cost=1.0
        )
        
        # Create mock batch
        batch = BatchInfo(
            batch_id=0,
            module_name="test",
            files=["test1.py", "test2.py"],
            total_size=1000,
            avg_file_size=500
        )
        
        # Create mock context and analyzer
        context = MagicMock(spec=AnalysisContext)
        
        mock_analyzer = MagicMock()
        mock_analyzer.name = "test_analyzer"
        mock_analyzer.run.return_value = [
            Issue(kind="test", message="Test issue", severity=2, file="test1.py")
        ]
        
        # Process stage
        issues = processor.process_stage(
            stage,
            [batch],
            context,
            [mock_analyzer]
        )
        
        assert len(issues) == 1
        assert issues[0].kind == "test"
        assert stage.name in processor.stats.stage_timings
        assert processor.stats.issues_by_stage[stage.name] == 1
    
    def test_checkpoint_save_and_load(self):
        """Test checkpoint saving and loading."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            processor = BatchProcessor(checkpoint_dir=checkpoint_dir)
            
            # Mark some stages as completed
            processor.completed_stages.add("stage1")
            processor.completed_stages.add("stage2")
            processor.stats.processed_files = 100
            
            # Save checkpoint
            processor._save_checkpoint("stage2", 5)
            
            # Check checkpoint file exists
            checkpoints = list(checkpoint_dir.glob("checkpoint_*.json"))
            assert len(checkpoints) == 1
            
            # Load checkpoint
            loaded = processor._load_checkpoint("stage2")
            assert loaded is not None
            assert "stage1" in loaded.completed_stages
            assert "stage2" in loaded.completed_stages
            assert loaded.processed_files == 100
    
    def test_batch_context_creation(self):
        """Test creating filtered context for a batch."""
        processor = BatchProcessor()
        
        # Create mock context
        context = MagicMock(spec=AnalysisContext)
        context.root_dir = Path(".")
        context.config = {}
        context.ast_index = {
            "file1.py": "ast1",
            "file2.py": "ast2",
            "file3.py": "ast3"
        }
        context.symbol_table = {}
        context.source_cache = {}
        context.cache = {}
        
        # Create batch with subset of files
        batch = BatchInfo(
            batch_id=0,
            module_name="test",
            files=["file1.py", "file3.py"],
            total_size=1000,
            avg_file_size=500
        )
        
        # Create batch context
        batch_context = processor._create_batch_context(batch, context)
        
        # Should only include files from batch
        assert len(batch_context.ast_index) == 2
        assert "file1.py" in batch_context.ast_index
        assert "file3.py" in batch_context.ast_index
        assert "file2.py" not in batch_context.ast_index
    
    def test_full_processing_workflow(self):
        """Test complete processing workflow."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            
            # Create test files
            files = []
            for i in range(3):
                file_path = tmpdir / f"file{i}.py"
                file_path.write_text(f"def func{i}(): pass")
                files.append(file_path)
            
            processor = BatchProcessor(
                batch_size=2,
                show_progress=False
            )
            
            # Create mock context
            context = MagicMock(spec=AnalysisContext)
            context.root_dir = tmpdir
            context.config = {}
            context.ast_index = {str(f): f"ast{i}" for i, f in enumerate(files)}
            context.symbol_table = {}
            context.source_cache = {}
            context.cache = {}
            
            # Create mock analyzers
            mock_analyzer1 = MagicMock()
            mock_analyzer1.name = "duplicates"
            mock_analyzer1.run.return_value = [
                Issue(kind="duplicate", message="Duplicate found", severity=2)
            ]
            
            mock_analyzer2 = MagicMock()
            mock_analyzer2.name = "missing_symbols"
            mock_analyzer2.run.return_value = []
            
            # Process
            issue_collection = processor.process(
                files=files,
                context=context,
                analyzers=[mock_analyzer1, mock_analyzer2]
            )
            
            # Check results
            assert processor.stats.total_files == 3
            assert processor.stats.total_batches == 2  # 2 + 1 files
            assert len(processor.completed_stages) > 0
            
            # Should have processed both analyzers
            assert mock_analyzer1.run.called
            assert mock_analyzer2.run.called
    
    def test_resume_from_checkpoint(self):
        """Test resuming from a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            
            # Create initial processor and save checkpoint
            processor1 = BatchProcessor(checkpoint_dir=checkpoint_dir)
            processor1.completed_stages.add("quarantine_check")
            processor1.completed_stages.add("duplicate_detection")
            processor1._save_checkpoint("duplicate_detection", 10)
            
            # Create new processor and resume
            processor2 = BatchProcessor(checkpoint_dir=checkpoint_dir)
            
            # Mock context and analyzers
            context = MagicMock(spec=AnalysisContext)
            context.ast_index = {}
            
            with patch.object(processor2, '_load_checkpoint') as mock_load:
                mock_load.return_value = CheckpointData(
                    timestamp=None,
                    stage_name="duplicate_detection",
                    completed_stages=["quarantine_check", "duplicate_detection"],
                    completed_batches={},
                    issues_found={},
                    processing_stats={},
                    total_files=100,
                    processed_files=50
                )
                
                processor2.process(
                    files=[],
                    context=context,
                    analyzers=[],
                    resume_from="duplicate_detection"
                )
                
                # Should have loaded checkpoint
                assert "quarantine_check" in processor2.completed_stages
                assert "duplicate_detection" in processor2.completed_stages
                assert processor2.stats.processed_files == 50
    
    def test_skip_stages(self):
        """Test skipping specific stages."""
        processor = BatchProcessor(show_progress=False)
        
        context = MagicMock(spec=AnalysisContext)
        context.ast_index = {}
        
        # Process with stages to skip
        processor.process(
            files=[],
            context=context,
            analyzers=[],
            skip_stages=["semantic_analysis", "advanced_patterns"]
        )
        
        # Should not have processed skipped stages
        assert "semantic_analysis" not in processor.completed_stages
        assert "advanced_patterns" not in processor.completed_stages
    
    def test_statistics_tracking(self):
        """Test that statistics are properly tracked."""
        processor = BatchProcessor()
        
        # Simulate processing
        processor.stats.total_files = 100
        processor.stats.processed_files = 95
        processor.stats.quarantined_files = 5
        processor.stats.total_batches = 10
        processor.stats.processed_batches = 10
        
        processor.stats.stage_timings["duplicate_detection"] = 5.5
        processor.stats.stage_timings["semantic_analysis"] = 12.3
        
        processor.stats.issues_by_stage["duplicate_detection"] = 15
        processor.stats.issues_by_stage["semantic_analysis"] = 8
        
        # Get statistics
        stats = processor.get_statistics()
        
        assert stats.total_files == 100
        assert stats.processed_files == 95
        assert stats.quarantined_files == 5
        assert len(stats.stage_timings) == 2
        assert stats.issues_by_stage["duplicate_detection"] == 15
    
    def test_report_generation(self):
        """Test report generation."""
        processor = BatchProcessor()
        
        # Set up stats
        processor.stats.start_time = time.time() - 10
        processor.stats.end_time = time.time()
        processor.stats.total_files = 100
        processor.stats.processed_files = 95
        processor.stats.quarantined_files = 5
        processor.stats.stage_timings["test_stage"] = 5.0
        processor.stats.issues_by_stage["test_stage"] = 10
        
        # Generate report
        report = processor.generate_report()
        
        # Check report content
        assert "BATCH PROCESSING REPORT" in report
        assert "Total Duration:" in report
        assert "Files Processed: 95/100" in report
        assert "Quarantined Files: 5" in report
        assert "test_stage" in report
        assert "10" in report  # issues count
    
    def test_progress_estimation(self):
        """Test time estimation during processing."""
        processor = BatchProcessor()
        
        # Mark some stages complete
        processor.completed_stages.add("quarantine_check")
        processor.completed_stages.add("duplicate_detection")
        
        # Update time estimate
        processor._update_time_estimate()
        
        # Should have some estimate (or None if not enough data)
        # Just check it doesn't crash
        assert processor.stats.estimated_remaining is None or \
               isinstance(processor.stats.estimated_remaining, float)


class TestProcessingStage:
    """Test ProcessingStage dataclass."""
    
    def test_stage_creation(self):
        """Test creating a processing stage."""
        stage = ProcessingStage(
            name="test",
            description="Test stage",
            priority=1,
            analyzer_names=["analyzer1", "analyzer2"],
            estimated_cost=1.5,
            can_skip=True,
            depends_on=["other_stage"]
        )
        
        assert stage.name == "test"
        assert stage.priority == 1
        assert len(stage.analyzer_names) == 2
        assert stage.can_skip is True
        assert "other_stage" in stage.depends_on
    
    def test_stage_ordering(self):
        """Test that stages can be ordered by priority."""
        stage1 = ProcessingStage("s1", "Stage 1", 2, [], 1.0)
        stage2 = ProcessingStage("s2", "Stage 2", 1, [], 1.0)
        stage3 = ProcessingStage("s3", "Stage 3", 3, [], 1.0)
        
        stages = sorted([stage1, stage2, stage3], key=lambda s: s.priority)
        
        assert stages[0].name == "s2"  # Priority 1
        assert stages[1].name == "s1"  # Priority 2
        assert stages[2].name == "s3"  # Priority 3


class TestBatchInfo:
    """Test BatchInfo dataclass."""
    
    def test_batch_info_creation(self):
        """Test creating batch info."""
        batch = BatchInfo(
            batch_id=0,
            module_name="test_module",
            files=["file1.py", "file2.py"],
            total_size=2048,
            avg_file_size=1024.0,
            priority_score=5.0
        )
        
        assert batch.batch_id == 0
        assert batch.module_name == "test_module"
        assert len(batch.files) == 2
        assert batch.total_size == 2048
        assert batch.avg_file_size == 1024.0
        assert batch.priority_score == 5.0
    
    def test_batch_sorting(self):
        """Test that batches can be sorted by priority."""
        batch1 = BatchInfo(0, "m1", [], 100, 100, 3.0)
        batch2 = BatchInfo(1, "m2", [], 100, 100, 5.0)
        batch3 = BatchInfo(2, "m3", [], 100, 100, 1.0)
        
        batches = sorted([batch1, batch2, batch3], 
                        key=lambda b: b.priority_score, reverse=True)
        
        assert batches[0].batch_id == 1  # Highest priority (5.0)
        assert batches[1].batch_id == 0  # Medium priority (3.0)
        assert batches[2].batch_id == 2  # Lowest priority (1.0)