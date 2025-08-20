"""
Batch processing system for efficient analysis of large codebases.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from datetime import datetime
import logging

from rich.console import Console
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeRemainingColumn,
    TimeElapsedColumn
)
from rich.table import Table

from ..core.issues import Issue, IssueCollection
from ..analyzers.base import AnalysisContext


logger = logging.getLogger(__name__)


@dataclass
class ProcessingStage:
    """Represents a processing stage."""
    name: str
    description: str
    priority: int  # Lower is higher priority
    analyzer_names: List[str]
    estimated_cost: float  # Relative cost estimate
    can_skip: bool = False
    depends_on: List[str] = field(default_factory=list)


@dataclass
class BatchInfo:
    """Information about a batch of files."""
    batch_id: int
    module_name: str
    files: List[str]
    total_size: int
    avg_file_size: float
    priority_score: float = 0.0


@dataclass 
class CheckpointData:
    """Checkpoint data for resuming processing."""
    timestamp: datetime
    stage_name: str
    completed_stages: List[str]
    completed_batches: Dict[str, List[int]]  # stage -> batch_ids
    issues_found: Dict[str, List[Dict]]  # stage -> issues
    processing_stats: Dict[str, Any]
    total_files: int
    processed_files: int


@dataclass
class ProcessingStats:
    """Statistics for batch processing."""
    start_time: float
    end_time: Optional[float] = None
    total_files: int = 0
    processed_files: int = 0
    skipped_files: int = 0
    quarantined_files: int = 0
    total_batches: int = 0
    processed_batches: int = 0
    issues_by_stage: Dict[str, int] = field(default_factory=dict)
    stage_timings: Dict[str, float] = field(default_factory=dict)
    current_stage: Optional[str] = None
    current_batch: Optional[int] = None
    estimated_remaining: Optional[float] = None


class BatchProcessor:
    """Processes files in batches for efficiency."""
    
    # Define processing stages in priority order
    STAGES = [
        ProcessingStage(
            name="quarantine_check",
            description="Check quarantined files",
            priority=1,
            analyzer_names=[],  # Special handling
            estimated_cost=0.1,
            can_skip=False
        ),
        ProcessingStage(
            name="duplicate_detection", 
            description="Detect duplicate functions",
            priority=2,
            analyzer_names=["duplicates", "fast_duplicates"],
            estimated_cost=0.5,
            can_skip=False
        ),
        ProcessingStage(
            name="missing_symbols",
            description="Find missing symbols",
            priority=3,
            analyzer_names=["missing_symbols", "enhanced_missing_symbols"],
            estimated_cost=0.3,
            can_skip=True
        ),
        ProcessingStage(
            name="circular_imports",
            description="Detect circular imports",
            priority=4,
            analyzer_names=["import_graph", "circular_import_resolver"],
            estimated_cost=0.4,
            can_skip=True
        ),
        ProcessingStage(
            name="placeholders",
            description="Find placeholder functions",
            priority=5,
            analyzer_names=["placeholders", "enhanced_placeholders", "phantom_triage"],
            estimated_cost=0.3,
            can_skip=True
        ),
        ProcessingStage(
            name="semantic_analysis",
            description="Semantic duplicate analysis",
            priority=6,
            analyzer_names=["semantic_hv", "enhanced_semantic", "semantic_duplicate"],
            estimated_cost=2.0,
            can_skip=True,
            depends_on=["duplicate_detection"]
        ),
        ProcessingStage(
            name="advanced_patterns",
            description="Advanced pattern detection",
            priority=7,
            analyzer_names=["hallucination_cascade", "context_thrashing", "import_anxiety"],
            estimated_cost=1.5,
            can_skip=True
        )
    ]
    
    def __init__(
        self,
        batch_size: int = 50,
        checkpoint_dir: Optional[Path] = None,
        show_progress: bool = True,
        max_workers: int = 1
    ):
        """
        Initialize batch processor.
        
        Args:
            batch_size: Number of files per batch
            checkpoint_dir: Directory for checkpoint files
            show_progress: Whether to show progress bars
            max_workers: Number of parallel workers (future enhancement)
        """
        self.batch_size = batch_size
        self.checkpoint_dir = checkpoint_dir or Path(".tailchasing_checkpoints")
        self.show_progress = show_progress
        self.max_workers = max_workers
        
        self.console = Console()
        self.stats = ProcessingStats(start_time=time.time())
        self.checkpoints: List[CheckpointData] = []
        self.issue_collection = IssueCollection()
        
        # Create checkpoint directory if needed
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(exist_ok=True)
            
        # Stage management
        self.stages = sorted(self.STAGES, key=lambda s: s.priority)
        self.completed_stages: Set[str] = set()
        self.stage_issues: Dict[str, List[Issue]] = defaultdict(list)
        
    def group_files_by_module(self, files: List[Path]) -> Dict[str, List[Path]]:
        """
        Group files by module/package for cache locality.
        
        Args:
            files: List of file paths
            
        Returns:
            Dictionary mapping module names to file lists
        """
        modules = defaultdict(list)
        
        for file_path in files:
            # Find the module by looking for __init__.py
            parts = file_path.parts
            module_name = "root"
            
            for i in range(len(parts) - 1, 0, -1):
                parent = Path(*parts[:i])
                if (parent / "__init__.py").exists():
                    module_name = parent.name
                    break
                    
            modules[module_name].append(file_path)
            
        return dict(modules)
        
    def create_batches(self, files: List[Path]) -> List[BatchInfo]:
        """
        Create batches from files, grouped by module.
        
        Args:
            files: List of file paths
            
        Returns:
            List of BatchInfo objects
        """
        batches = []
        batch_id = 0
        
        # Group by module first
        modules = self.group_files_by_module(files)
        
        # Sort modules by total size (process larger modules first)
        module_sizes = {}
        for module_name, module_files in modules.items():
            total_size = sum(f.stat().st_size for f in module_files if f.exists())
            module_sizes[module_name] = total_size
            
        sorted_modules = sorted(modules.keys(), key=lambda m: module_sizes[m], reverse=True)
        
        # Create batches for each module
        for module_name in sorted_modules:
            module_files = modules[module_name]
            
            # Sort files within module by size
            module_files.sort(key=lambda f: f.stat().st_size if f.exists() else 0, reverse=True)
            
            # Create batches
            for i in range(0, len(module_files), self.batch_size):
                batch_files = module_files[i:i + self.batch_size]
                
                # Calculate batch statistics
                file_paths = [str(f) for f in batch_files]
                total_size = sum(f.stat().st_size for f in batch_files if f.exists())
                avg_size = total_size / len(batch_files) if batch_files else 0
                
                # Priority based on module importance (heuristic)
                priority_score = self._calculate_priority(module_name, total_size, len(batch_files))
                
                batches.append(BatchInfo(
                    batch_id=batch_id,
                    module_name=module_name,
                    files=file_paths,
                    total_size=total_size,
                    avg_file_size=avg_size,
                    priority_score=priority_score
                ))
                batch_id += 1
                
        # Sort batches by priority
        batches.sort(key=lambda b: b.priority_score, reverse=True)
        
        return batches
        
    def _calculate_priority(self, module_name: str, size: int, file_count: int) -> float:
        """Calculate priority score for a batch."""
        score = 0.0
        
        # Prioritize test files (quick to process, find issues early)
        if "test" in module_name.lower():
            score += 10.0
            
        # Prioritize core/main modules
        if module_name in ["core", "main", "src", "lib"]:
            score += 8.0
            
        # Prioritize smaller batches (quick wins)
        if file_count < 10:
            score += 5.0
            
        # Factor in size (prefer medium-sized batches)
        if 1000 < size < 100000:  # 1KB to 100KB
            score += 3.0
            
        return score
        
    def process_stage(
        self,
        stage: ProcessingStage,
        batches: List[BatchInfo],
        context: AnalysisContext,
        analyzers: Dict[str, Any],
        progress: Optional[Progress] = None
    ) -> List[Issue]:
        """
        Process a single stage across all batches.
        
        Args:
            stage: The processing stage
            batches: List of batches to process
            context: Analysis context
            analyzers: Available analyzers
            progress: Progress bar object
            
        Returns:
            List of issues found in this stage
        """
        stage_issues = []
        stage_start = time.time()
        
        # Filter to only relevant analyzers
        stage_analyzers = [
            a for a in analyzers 
            if hasattr(a, 'name') and a.name in stage.analyzer_names
        ]
        
        if not stage_analyzers and stage.name != "quarantine_check":
            logger.info(f"No analyzers found for stage {stage.name}, skipping")
            return []
            
        # Create progress task if available
        task_id = None
        if progress:
            task_id = progress.add_task(
                f"[cyan]{stage.description}",
                total=len(batches)
            )
            
        # Process each batch
        for batch in batches:
            self.stats.current_batch = batch.batch_id
            
            if stage.name == "quarantine_check":
                # Special handling for quarantine check
                quarantined = self._check_quarantined_files(batch, context)
                self.stats.quarantined_files += quarantined
            else:
                # Run analyzers for this batch
                batch_issues = self._process_batch_with_analyzers(
                    batch,
                    stage_analyzers,
                    context
                )
                stage_issues.extend(batch_issues)
                
            # Update progress
            if progress and task_id is not None:
                progress.update(task_id, advance=1)
                
            # Save checkpoint periodically
            if batch.batch_id % 10 == 0:
                self._save_checkpoint(stage.name, batch.batch_id)
                
        # Record stage timing
        stage_duration = time.time() - stage_start
        self.stats.stage_timings[stage.name] = stage_duration
        self.stats.issues_by_stage[stage.name] = len(stage_issues)
        
        logger.info(
            f"Stage '{stage.name}' completed in {stage_duration:.2f}s, "
            f"found {len(stage_issues)} issues"
        )
        
        return stage_issues
        
    def _check_quarantined_files(self, batch: BatchInfo, context: AnalysisContext) -> int:
        """Check for quarantined files in batch."""
        quarantined_count = 0
        
        for file_path in batch.files:
            # Check if file is in quarantine (parse errors, etc.)
            if hasattr(context, 'parse_results'):
                parse_result = context.parse_results.get(file_path)
                if parse_result and not parse_result.is_valid:
                    quarantined_count += 1
                    logger.debug(f"File {file_path} is quarantined: {parse_result.warnings}")
                    
        return quarantined_count
        
    def _process_batch_with_analyzers(
        self,
        batch: BatchInfo,
        analyzers: List[Any],
        context: AnalysisContext
    ) -> List[Issue]:
        """Process a batch with given analyzers."""
        batch_issues = []
        
        # Create filtered context for this batch
        batch_context = self._create_batch_context(batch, context)
        
        for analyzer in analyzers:
            try:
                # Run analyzer on batch
                for issue in analyzer.run(batch_context):
                    # Only include issues from files in this batch
                    if issue.file in batch.files:
                        batch_issues.append(issue)
            except Exception as e:
                logger.error(f"Analyzer {analyzer.name} failed on batch {batch.batch_id}: {e}")
                
        return batch_issues
        
    def _create_batch_context(self, batch: BatchInfo, context: AnalysisContext) -> AnalysisContext:
        """Create a context filtered to just the batch files."""
        # Filter AST index to batch files
        batch_ast_index = {
            f: ast for f, ast in context.ast_index.items()
            if f in batch.files
        }
        
        # Create new context with filtered data
        from ..analyzers.base import AnalysisContext
        
        batch_context = AnalysisContext(
            config=context.config,
            root_dir=context.root_dir,
            file_paths=[Path(f) for f in batch.files],
            ast_index=batch_ast_index,
            symbol_table=context.symbol_table,
            source_cache=context.source_cache,
            cache=context.cache
        )
        
        # Copy other attributes if present
        if hasattr(context, 'parse_results'):
            batch_context.parse_results = {
                f: r for f, r in context.parse_results.items()
                if f in batch.files
            }
            
        return batch_context
        
    def process(
        self,
        files: List[Path],
        context: AnalysisContext,
        analyzers: List[Any],
        resume_from: Optional[str] = None,
        skip_stages: Optional[List[str]] = None
    ) -> IssueCollection:
        """
        Process files in batches.
        
        Args:
            files: List of files to process
            context: Analysis context
            analyzers: Available analyzers
            resume_from: Stage name to resume from
            skip_stages: List of stage names to skip
            
        Returns:
            Collection of all issues found
        """
        self.stats.total_files = len(files)
        skip_stages = skip_stages or []
        
        # Load checkpoint if resuming
        if resume_from:
            checkpoint = self._load_checkpoint(resume_from)
            if checkpoint:
                self.completed_stages = set(checkpoint.completed_stages)
                self.stats.processed_files = checkpoint.processed_files
                logger.info(f"Resuming from stage '{resume_from}'")
                
        # Create batches
        batches = self.create_batches(files)
        self.stats.total_batches = len(batches)
        
        logger.info(
            f"Processing {len(files)} files in {len(batches)} batches "
            f"(batch size: {self.batch_size})"
        )
        
        # Setup progress display
        if self.show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console,
                refresh_per_second=2
            ) as progress:
                
                # Main progress task
                main_task = progress.add_task(
                    "[green]Overall Progress",
                    total=len(self.stages)
                )
                
                # Process each stage
                for stage in self.stages:
                    # Skip if already completed or in skip list
                    if stage.name in self.completed_stages or stage.name in skip_stages:
                        logger.info(f"Skipping stage '{stage.name}'")
                        progress.update(main_task, advance=1)
                        continue
                        
                    # Check dependencies
                    if not all(dep in self.completed_stages for dep in stage.depends_on):
                        logger.warning(
                            f"Skipping stage '{stage.name}' due to unmet dependencies: "
                            f"{stage.depends_on}"
                        )
                        continue
                        
                    # Process stage
                    self.stats.current_stage = stage.name
                    stage_issues = self.process_stage(
                        stage,
                        batches,
                        context,
                        analyzers,
                        progress
                    )
                    
                    # Store issues
                    self.stage_issues[stage.name] = stage_issues
                    for issue in stage_issues:
                        self.issue_collection.add(issue)
                        
                    # Mark complete
                    self.completed_stages.add(stage.name)
                    progress.update(main_task, advance=1)
                    
                    # Estimate remaining time
                    self._update_time_estimate()
                    
        else:
            # Process without progress bars
            for stage in self.stages:
                if stage.name in self.completed_stages or stage.name in skip_stages:
                    continue
                    
                self.stats.current_stage = stage.name
                stage_issues = self.process_stage(stage, batches, context, analyzers)
                
                self.stage_issues[stage.name] = stage_issues
                for issue in stage_issues:
                    self.issue_collection.add(issue)
                    
                self.completed_stages.add(stage.name)
                
        # Final statistics
        self.stats.end_time = time.time()
        self.stats.processed_files = self.stats.total_files - self.stats.quarantined_files
        
        # Save final checkpoint
        self._save_checkpoint("completed", -1)
        
        # Print summary
        self._print_summary()
        
        return self.issue_collection
        
    def _update_time_estimate(self):
        """Update estimated remaining time based on progress."""
        elapsed = time.time() - self.stats.start_time
        
        # Calculate weighted progress
        total_cost = sum(s.estimated_cost for s in self.stages)
        completed_cost = sum(
            s.estimated_cost for s in self.stages 
            if s.name in self.completed_stages
        )
        
        if completed_cost > 0:
            progress_ratio = completed_cost / total_cost
            estimated_total = elapsed / progress_ratio
            self.stats.estimated_remaining = estimated_total - elapsed
            
    def _save_checkpoint(self, stage_name: str, batch_id: int):
        """Save checkpoint for resuming."""
        if not self.checkpoint_dir:
            return
            
        checkpoint = CheckpointData(
            timestamp=datetime.now(),
            stage_name=stage_name,
            completed_stages=list(self.completed_stages),
            completed_batches={},  # TODO: Track per-stage batch completion
            issues_found={
                stage: [issue.to_dict() for issue in issues]
                for stage, issues in self.stage_issues.items()
            },
            processing_stats=asdict(self.stats),
            total_files=self.stats.total_files,
            processed_files=self.stats.processed_files
        )
        
        # Save to file
        checkpoint_file = self.checkpoint_dir / f"checkpoint_{stage_name}_{int(time.time())}.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(checkpoint), f, indent=2, default=str)
            
        logger.debug(f"Saved checkpoint to {checkpoint_file}")
        
    def _load_checkpoint(self, stage_name: str) -> Optional[CheckpointData]:
        """Load checkpoint for a stage."""
        if not self.checkpoint_dir or not self.checkpoint_dir.exists():
            return None
            
        # Find most recent checkpoint for stage
        checkpoints = list(self.checkpoint_dir.glob(f"checkpoint_{stage_name}_*.json"))
        if not checkpoints:
            return None
            
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        try:
            with open(latest, 'r') as f:
                data = json.load(f)
                return CheckpointData(**data)
        except Exception as e:
            logger.error(f"Failed to load checkpoint {latest}: {e}")
            return None
            
    def _print_summary(self):
        """Print processing summary."""
        if not self.show_progress:
            return
            
        duration = self.stats.end_time - self.stats.start_time if self.stats.end_time else 0
        
        # Create summary table
        table = Table(title="Batch Processing Summary", show_header=True)
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        
        table.add_row("Total Files", str(self.stats.total_files))
        table.add_row("Processed Files", str(self.stats.processed_files))
        table.add_row("Quarantined Files", str(self.stats.quarantined_files))
        table.add_row("Total Batches", str(self.stats.total_batches))
        table.add_row("Total Time", f"{duration:.2f}s")
        
        if self.stats.total_files > 0:
            rate = self.stats.total_files / duration if duration > 0 else 0
            table.add_row("Processing Rate", f"{rate:.1f} files/s")
            
        # Add stage breakdown
        table.add_row("", "")  # Spacer
        table.add_row("[bold]Stage Breakdown[/bold]", "")
        
        for stage_name, timing in self.stats.stage_timings.items():
            issues = self.stats.issues_by_stage.get(stage_name, 0)
            table.add_row(f"  {stage_name}", f"{timing:.2f}s ({issues} issues)")
            
        self.console.print(table)
        
    def get_statistics(self) -> ProcessingStats:
        """Get current processing statistics."""
        return self.stats
        
    def generate_report(self) -> str:
        """Generate detailed processing report."""
        report = []
        
        report.append("=" * 60)
        report.append("BATCH PROCESSING REPORT")
        report.append("=" * 60)
        report.append("")
        
        # Overall statistics
        duration = self.stats.end_time - self.stats.start_time if self.stats.end_time else 0
        report.append(f"Total Duration: {duration:.2f} seconds")
        report.append(f"Files Processed: {self.stats.processed_files}/{self.stats.total_files}")
        report.append(f"Quarantined Files: {self.stats.quarantined_files}")
        report.append(f"Total Batches: {self.stats.total_batches}")
        report.append("")
        
        # Stage breakdown
        report.append("Stage Performance:")
        report.append("-" * 40)
        
        # Show data from stage timings
        for stage_name, timing in self.stats.stage_timings.items():
            issues = self.stats.issues_by_stage.get(stage_name, 0)
            report.append(f"  {stage_name}:")
            report.append(f"    Time: {timing:.2f}s")
            report.append(f"    Issues Found: {issues}")
            if issues > 0 and timing > 0:
                rate = issues / timing
                report.append(f"    Issues/Second: {rate:.2f}")
            report.append("")
                
        # Issue summary
        total_issues = sum(self.stats.issues_by_stage.values())
        report.append(f"Total Issues Found: {total_issues}")
        
        if duration > 0:
            overall_rate = self.stats.processed_files / duration
            report.append(f"Overall Processing Rate: {overall_rate:.1f} files/second")
            
        report.append("=" * 60)
        
        return "\n".join(report)