"""Command-line interface for tail-chasing detector."""

import sys
import argparse
import logging
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from . import __version__
from .config import Config
from .utils.logging_setup import get_logger, log_operation
from .core.loader import collect_files, parse_files
from .core.symbols import SymbolTable
from .core.issues import IssueCollection
from .core.reporting import Reporter
from .core.scoring import RiskScorer
from .analyzers.base import AnalysisContext
from .analyzers.root_cause_clustering import RootCauseClusterer
from .core.issue_provenance import IssueProvenanceTracker
from .plugins import load_analyzers
from .core.watchdog import AnalyzerWatchdog, WatchdogConfig, SemanticAnalysisFallback
from .core.batch_processor import BatchProcessor, ProcessingStats
from .core.resource_monitor import MemoryMonitor, AdaptiveConfig, AdaptiveProcessor
from .cli.output_manager import OutputManager, VerbosityLevel, OutputFormat
from .cli.profiler import PerformanceProfiler


# Setup module logger
logger = get_logger(__name__)

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.WARNING
    logging.basicConfig(
        level=level,
        format="%(levelname)s: %(message)s"
    )


def main():
    """Main CLI entry point."""
    log_operation(logger, "cli_main")
    parser = argparse.ArgumentParser(
        prog="tailchasing",
        description="Detect LLM-assisted tail-chasing anti-patterns in Python code"
    )
    
    parser.add_argument(
        "path",
        nargs="?",
        default=".",
        help="Path to analyze (default: current directory)"
    )
    
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON"
    )
    
    parser.add_argument(
        "--html",
        action="store_true",
        help="Generate HTML report"
    )
    
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output directory for reports"
    )
    
    parser.add_argument(
        "--fail-on",
        type=int,
        metavar="SCORE",
        help="Exit with error if risk score exceeds threshold"
    )
    
    # Verbosity levels
    verbosity_group = parser.add_mutually_exclusive_group()
    verbosity_group.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Quiet mode - only show errors and critical info"
    )
    verbosity_group.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose mode - show detailed output"
    )
    verbosity_group.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode - show all debug information"
    )
    
    # Output format
    parser.add_argument(
        "--output-format",
        choices=["text", "json", "yaml", "html", "sarif"],
        default="text",
        help="Output format (default: text)"
    )
    
    # Watch mode
    parser.add_argument(
        "--watch",
        action="store_true",
        help="Watch mode - show live updates during analysis"
    )
    
    # Performance profiling
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Enable performance profiling and show metrics"
    )
    
    # Dry run
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be analyzed without actually running"
    )
    
    # Color output
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    
    # Stub Guard mode
    parser.add_argument(
        "--guard",
        action="store_true",
        help="Run in guard mode - fail fast on stubs/TODOs in critical paths"
    )
    
    parser.add_argument(
        "--guard-critical-paths",
        nargs="+",
        default=["src", "lib", "core"],
        help="Critical paths for guard mode (default: src lib core)"
    )
    
    parser.add_argument(
        "--guard-format",
        choices=["text", "json", "junit"],
        default="text",
        help="Output format for guard mode (default: text)"
    )
    
    parser.add_argument(
        "--exclude",
        action="append",
        help="Additional paths to exclude (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--cluster-root-causes",
        action="store_true",
        help="Cluster issues by root cause using AST analysis"
    )
    
    parser.add_argument(
        "--cluster-threshold",
        type=float,
        default=0.7,
        metavar="THRESHOLD",
        help="Similarity threshold for clustering (0.0-1.0, default: 0.7)"
    )
    
    parser.add_argument(
        "--generate-canonical-codemod",
        action="store_true",
        help="Generate codemod script for canonical policy violations"
    )
    
    parser.add_argument(
        "--generate-circular-import-fixes",
        action="store_true",
        help="Generate fix script for circular import SCCs"
    )
    
    parser.add_argument(
        "--show-regressions",
        type=int,
        metavar="DAYS",
        help="Show issue regressions from the last N days"
    )
    
    parser.add_argument(
        "--include",
        action="append",
        help="Paths to include (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--disable",
        action="append",
        help="Disable specific analyzers (can be specified multiple times)"
    )
    
    parser.add_argument(
        "--show-suggestions",
        action="store_true",
        help="Show fix suggestions in terminal output"
    )
    
    parser.add_argument(
        "--generate-fixes",
        action="store_true",
        help="Generate an interactive fix script"
    )
    
    parser.add_argument(
        "--phantom-triage-report",
        action="store_true",
        help="Generate phantom stub triage report with P0/P1/P3 classification"
    )
    
    parser.add_argument(
        "--generate-playbooks",
        action="store_true",
        help="Generate fix playbooks for issue clusters with safety checks"
    )
    
    parser.add_argument(
        "--preview-playbooks",
        action="store_true",
        help="Preview playbook changes without executing them"
    )
    
    parser.add_argument(
        "--execute-playbooks",
        action="store_true",
        help="Execute generated playbooks (requires --generate-playbooks)"
    )
    
    # Resource limit arguments
    parser.add_argument(
        "--max-duplicate-pairs",
        type=int,
        metavar="N",
        help="Maximum number of duplicate pairs to analyze (default: 200000)"
    )
    
    parser.add_argument(
        "--analyzer-timeout",
        type=int,
        metavar="SECONDS",
        default=30,
        help="Timeout for each analyzer in seconds (default: 30)"
    )
    
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        metavar="SECONDS",
        default=2.0,
        help="Heartbeat interval for analyzer monitoring in seconds (default: 2.0)"
    )
    
    parser.add_argument(
        "--watchdog-verbose",
        action="store_true",
        help="Enable verbose watchdog reporting"
    )
    
    parser.add_argument(
        "--disable-fallback",
        action="store_true",
        help="Disable fallback to TF-IDF when semantic analysis times out"
    )
    
    parser.add_argument(
        "--max-memory-mb",
        type=int,
        metavar="MB",
        help="Maximum memory usage in MB (default: 8192)"
    )
    
    parser.add_argument(
        "--lsh-bucket-cap",
        type=int,
        metavar="N",
        help="LSH bucket capacity for semantic analysis (default: 2000)"
    )
    
    parser.add_argument(
        "--force-semantic",
        action="store_true",
        help="Force semantic analysis even when file/duplicate counts exceed limits"
    )
    
    parser.add_argument(
        "--auto-fix-trivial-syntax",
        action="store_true",
        help="Attempt to automatically fix trivial syntax errors before parsing"
    )
    
    parser.add_argument(
        "--robust-parsing",
        action="store_true",
        default=True,
        help="Use robust parser with fallback strategies (default: enabled)"
    )
    
    parser.add_argument(
        "--no-robust-parsing",
        action="store_false",
        dest="robust_parsing",
        help="Disable robust parsing and use simple AST parser only"
    )
    
    parser.add_argument(
        "--fast-duplicates",
        action="store_true",
        default=True,
        help="Use fast LSH-based duplicate detection (default: enabled)"
    )
    
    parser.add_argument(
        "--no-fast-duplicates",
        action="store_false",
        dest="fast_duplicates",
        help="Use traditional O(n²) duplicate detection"
    )
    
    parser.add_argument(
        "--lsh-bands",
        type=int,
        metavar="N",
        help="Number of bands for LSH (default: 8)"
    )
    
    parser.add_argument(
        "--lsh-rows",
        type=int,
        metavar="N",
        help="Number of rows per band for LSH (default: 16)"
    )
    
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="Clear all cached analysis data and force full reanalysis"
    )
    
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching for this run"
    )
    
    parser.add_argument(
        "--cache-stats",
        action="store_true",
        help="Show cache statistics after analysis"
    )
    
    parser.add_argument(
        "--generate-missing-stubs",
        action="store_true",
        help="Generate typed skeleton functions for missing symbols"
    )
    
    parser.add_argument(
        "--missing-stubs-output",
        type=Path,
        default=Path("missing_symbols.py"),
        help="Output file for missing symbol stubs (default: missing_symbols.py)"
    )
    
    # Batch processing arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        metavar="N",
        help="Number of files per batch (default: 50)"
    )
    
    parser.add_argument(
        "--use-batching",
        action="store_true",
        help="Enable batch processing for large codebases"
    )
    
    parser.add_argument(
        "--resume-from",
        type=str,
        metavar="STAGE",
        help="Resume processing from a specific stage"
    )
    
    parser.add_argument(
        "--skip-stages",
        nargs="+",
        help="Skip specific processing stages"
    )
    
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(".tailchasing_checkpoints"),
        help="Directory for checkpoint files (default: .tailchasing_checkpoints)"
    )
    
    parser.add_argument(
        "--show-batch-progress",
        action="store_true",
        default=True,
        help="Show progress bars during batch processing (default: enabled)"
    )
    
    parser.add_argument(
        "--no-batch-progress",
        action="store_false",
        dest="show_batch_progress",
        help="Disable batch processing progress bars"
    )
    
    # Fix planner arguments
    parser.add_argument(
        "--generate-fix-plan",
        action="store_true",
        help="Generate a detailed fix plan for detected issues"
    )
    
    parser.add_argument(
        "--fix-plan-output",
        type=Path,
        default=Path("fix_plan.py"),
        help="Output file for fix plan script (default: fix_plan.py)"
    )
    
    parser.add_argument(
        "--interactive-fixes",
        action="store_true",
        help="Interactively review each fix before approving"
    )
    
    parser.add_argument(
        "--fix-confidence-threshold",
        type=float,
        default=0.5,
        metavar="THRESHOLD",
        help="Minimum confidence threshold for auto-approving fixes (0.0-1.0, default: 0.5)"
    )
    
    parser.add_argument(
        "--backup-dir",
        type=Path,
        default=Path(".tailchasing_backups"),
        help="Directory for backups before applying fixes (default: .tailchasing_backups)"
    )
    
    parser.add_argument(
        "--dry-run-fixes",
        action="store_true",
        help="Generate fix plan without executing (dry run mode)"
    )
    
    # Memory monitoring arguments
    parser.add_argument(
        "--mem-ceiling-mb",
        type=int,
        metavar="MB",
        help="Set hard memory ceiling in MB (default: 80% of system memory)"
    )
    
    parser.add_argument(
        "--disable-memory-monitor",
        action="store_true",
        help="Disable memory monitoring and adaptive processing"
    )
    
    parser.add_argument(
        "--memory-gc-threshold",
        type=float,
        default=80.0,
        metavar="PERCENT",
        help="Memory usage percentage to trigger garbage collection (default: 80.0)"
    )
    
    parser.add_argument(
        "--memory-streaming-threshold",
        type=float,
        default=90.0,
        metavar="PERCENT",
        help="Memory usage percentage to trigger streaming mode (default: 90.0)"
    )
    
    args = parser.parse_args()
    
    # Determine verbosity level
    if args.quiet:
        verbosity = VerbosityLevel.QUIET
    elif args.verbose:
        verbosity = VerbosityLevel.VERBOSE
    elif args.debug:
        verbosity = VerbosityLevel.DEBUG
    else:
        verbosity = VerbosityLevel.NORMAL
    
    # Setup logging (for backward compatibility)
    setup_logging(args.debug or args.verbose)
    
    # Determine output format
    if args.json:  # Backward compatibility
        output_format = OutputFormat.JSON
    elif args.html:  # Backward compatibility
        output_format = OutputFormat.HTML
    else:
        output_format = OutputFormat[args.output_format.upper()]
    
    # Initialize output manager
    output_manager = OutputManager(
        verbosity=verbosity,
        output_format=output_format,
        output_file=args.output if args.output else None,
        use_color=not args.no_color,
        watch_mode=args.watch
    )
    
    # Initialize profiler if requested
    profiler = PerformanceProfiler(enabled=args.profile) if args.profile else None
    
    # Initialize memory monitor if requested
    memory_monitor = None
    if not args.disable_memory_monitor:
        memory_config = AdaptiveConfig(
            gc_threshold_percent=args.memory_gc_threshold,
            streaming_threshold_percent=args.memory_streaming_threshold,
            enable_monitoring=True
        )
        memory_monitor = MemoryMonitor(
            memory_limit_mb=args.mem_ceiling_mb,
            config=memory_config,
            verbose=verbosity in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]
        )
        
        if verbosity in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            if memory_monitor.config.enable_monitoring:
                output_manager.log("Initialized memory monitor", VerbosityLevel.VERBOSE)
                stats = memory_monitor.get_stats()
                output_manager.log(f"Memory limit: {stats.limit_mb:.0f}MB", VerbosityLevel.VERBOSE)
            else:
                output_manager.log("Memory monitoring disabled (psutil not available)", VerbosityLevel.VERBOSE)
    
    # Find project root
    root_path = Path(args.path).absolute()
    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        sys.stderr.write(f"Error: Path does not exist: {root_path}\n")
        sys.exit(1)
    
    # Initialize cache manager
    from .core.cache import CacheManager
    cache_enabled = not args.no_cache
    cache_manager = CacheManager(root_path, enabled=cache_enabled)
    
    # Clear cache if requested
    if args.clear_cache:
        logger.info("Clearing cache...")
        cache_manager.clear_cache()
        sys.stdout.write("Cache cleared successfully\n")
    
    # Guard mode - fast stub/TODO checking
    if args.guard:
        from .guards.stub_guard import StubGuard, GuardConfig
        
        config = GuardConfig(
            critical_paths=args.guard_critical_paths,
            output_format=args.guard_format,
            verbose=args.verbose,
            max_critical_violations=0,
            max_high_violations=0,
            fail_on_critical=True
        )
        
        guard = StubGuard(config)
        violations = guard.scan_directory(root_path)
        
        # Output report
        print(guard.generate_report(args.guard_format))
        
        # Check violations
        passed, message = guard.check_violations()
        if not passed:
            print(f"\n{message}", file=sys.stderr)
            sys.exit(1)
        sys.exit(0)
        
    # Load configuration
    if args.config:
        config = Config.from_file(args.config)
    else:
        config = Config.find_and_load(root_path)
        
    # Override config with CLI arguments
    if args.exclude:
        excludes = config.get("paths.exclude", [])
        excludes.extend(args.exclude)
        config.set("paths.exclude", excludes)
        
    if args.include:
        config.set("paths.include", args.include)
        
    if args.disable:
        config.set("disabled_analyzers", args.disable)
        
    if args.json:
        formats = config.get("report.formats", [])
        if "json" not in formats:
            formats.append("json")
        config.set("report.formats", formats)
        
    if args.html:
        formats = config.get("report.formats", [])
        if "html" not in formats:
            formats.append("html")
        config.set("report.formats", formats)
        
    if args.output:
        config.set("report.output_dir", str(args.output))
        
    # Apply resource limit overrides from CLI
    if args.max_duplicate_pairs:
        config.set("resource_limits.max_duplicate_pairs", args.max_duplicate_pairs)
        
    if args.analyzer_timeout:
        config.set("resource_limits.analyzer_timeout_seconds", args.analyzer_timeout)
        
    if args.max_memory_mb:
        config.set("resource_limits.max_memory_mb", args.max_memory_mb)
        
    if args.lsh_bucket_cap:
        config.set("resource_limits.lsh_bucket_cap", args.lsh_bucket_cap)
        
    # Apply duplicate detection settings
    config.set("duplicates.use_fast_detection", args.fast_duplicates)
    if args.lsh_bands:
        config.set("duplicates.lsh_bands", args.lsh_bands)
    if args.lsh_rows:
        config.set("duplicates.lsh_rows", args.lsh_rows)
        
    # Collect files with IgnoreManager support
    logger.info(f"Collecting Python files from {root_path}")
    
    # Create IgnoreManager with exclude patterns from CLI and config
    from .core.ignore import IgnoreManager
    ignore_manager = IgnoreManager(
        root_path=root_path,
        additional_patterns=config.get("paths.exclude", []),
        use_defaults=True
    )
    
    files = collect_files(
        root_path,
        include=config.get("paths.include"),
        exclude=config.get("paths.exclude"),
        ignore_manager=ignore_manager
    )
    
    if not files:
        logger.warning("No Python files found to analyze")
        sys.stderr.write("No Python files found to analyze\n")
        sys.exit(0)
        
    logger.info(f"Found {len(files)} Python files")
    output_manager.log(f"Found {len(files)} Python files to analyze", VerbosityLevel.NORMAL)
    
    # Load analyzers for dry-run info
    analyzers = load_analyzers(config.to_dict())
    analyzer_names = [a.name for a in analyzers if hasattr(a, 'name')]
    
    # Dry-run mode - show what would be analyzed
    if args.dry_run:
        output_manager.show_dry_run_summary(
            files=files,
            analyzers=analyzer_names,
            config=config.to_dict()
        )
        sys.exit(0)
    
    # Parse files with progress tracking
    spinner_id = output_manager.start_spinner("Parsing Python files...")
    
    if profiler:
        profiler_context = profiler.profile("file_parsing")
        profiler_context.__enter__()
    
    # Create robust parser if enabled
    robust_parser = None
    parse_results = {}
    if args.robust_parsing:
        from .core.robust_parser import RobustParser
        robust_parser = RobustParser(auto_fix_trivial=args.auto_fix_trivial_syntax)
        output_manager.log(f"Using robust parser (auto-fix: {args.auto_fix_trivial_syntax})", VerbosityLevel.VERBOSE)
    
    # Parse files with appropriate method
    if robust_parser:
        ast_index, parse_results = parse_files(files, robust_parser, cache_manager)
        
        # Show parsing statistics
        stats = robust_parser.get_statistics()
        if stats['quarantined'] > 0:
            output_manager.warning(f"Quarantined {stats['quarantined']} files due to parsing errors")
            
            # Show details of quarantined files if verbose
            if verbosity == VerbosityLevel.VERBOSE:
                for file_path, result in parse_results.items():
                    if result.is_quarantined:
                        output_manager.log(f"  - {file_path}: {', '.join(result.warnings[:2])}", VerbosityLevel.VERBOSE)
    else:
        ast_index, _ = parse_files(files, cache_manager=cache_manager)
    
    output_manager.stop_spinner(spinner_id)
    
    if profiler:
        profiler_context.__exit__(None, None, None)
    
    if not ast_index:
        logger.error("No valid Python files could be parsed")
        sys.stderr.write("No valid Python files could be parsed\n")
        sys.exit(1)
        
    logger.info(f"Successfully parsed {len(ast_index)} files")
    
    # Check memory usage after parsing and adjust strategy if needed
    if memory_monitor:
        stats = memory_monitor.get_stats()
        output_manager.log(f"Memory after parsing: {stats.current_mb:.1f}MB ({stats.usage_percent:.1f}%)", VerbosityLevel.VERBOSE)
        
        # Force GC if memory usage is high
        if stats.usage_percent > 70:
            memory_monitor.force_gc()
            output_manager.log("Triggered GC after parsing", VerbosityLevel.VERBOSE)
    
    # Build symbol table
    logger.info("Building symbol table")
    symbol_table = SymbolTable()
    source_cache = {}
    
    for file_path, tree in ast_index.items():
        try:
            source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
            source_cache[file_path] = source.splitlines()
            symbol_table.ingest_file(file_path, tree, source)
        except Exception as e:
            logger.warning(f"Failed to process {file_path}: {e}")
            
    # Create analysis context
    ctx = AnalysisContext(
        config=config.to_dict(),
        root_dir=root_path,
        file_paths=files,
        ast_index=ast_index,
        symbol_table=symbol_table,
        source_cache=source_cache,
        cache={},  # Initialize empty cache for analyzers
        parse_results=parse_results,  # Add parse results for quarantine checking
        cache_manager=cache_manager  # Pass cache manager to context
    )
    
    # Load and run analyzers
    logger.info("Running analyzers")
    analyzers = load_analyzers(config.to_dict())
    
    issue_collection = IssueCollection()
    
    # Check if batch processing is enabled for large codebases
    use_batching = args.use_batching or len(files) > 500  # Auto-enable for large codebases
    
    # Also enable batching if memory monitor suggests it
    if memory_monitor and not use_batching:
        should_stream = memory_monitor.should_use_streaming(len(files))
        if should_stream:
            use_batching = True
            output_manager.log("Enabling batch processing due to memory constraints", VerbosityLevel.NORMAL)
    
    if use_batching:
        logger.info(f"Using batch processing for {len(files)} files")
        
        # Create batch processor
        batch_processor = BatchProcessor(
            batch_size=args.batch_size,
            checkpoint_dir=args.checkpoint_dir if args.resume_from else None,
            show_progress=args.show_batch_progress
        )
        
        # Convert file paths to Path objects
        file_paths = [Path(f) for f in files]
        
        # Process in batches
        issue_collection = batch_processor.process(
            files=file_paths,
            context=ctx,
            analyzers=analyzers,
            resume_from=args.resume_from,
            skip_stages=args.skip_stages
        )
        
        # Get statistics for reporting
        batch_stats = batch_processor.get_statistics()
        
        # Skip normal processing
        skip_normal_processing = True
    else:
        skip_normal_processing = False
        
        # Initialize watchdog for normal processing
        watchdog_config = WatchdogConfig(
            analyzer_timeout=args.analyzer_timeout,
            heartbeat_interval=args.heartbeat_interval,
            heartbeat_timeout_multiplier=3.0,
            enable_fallback=not args.disable_fallback,
            verbose=args.watchdog_verbose
        )
        watchdog = AnalyzerWatchdog(watchdog_config)
        watchdog.start()
    
    # Normal processing (non-batch mode)
    if not skip_normal_processing:
        # Check for bail-out conditions for semantic analysis
        file_count = len(files)
        resource_limits = config.get("resource_limits", {})
        semantic_file_limit = resource_limits.get("semantic_analysis_file_limit", 1000)
        semantic_duplicate_limit = resource_limits.get("semantic_analysis_duplicate_limit", 500)
        
        skip_semantic = False
        duplicate_count = 0
    
        if not args.force_semantic:
            # Count duplicates if needed
            for analyzer in analyzers:
                if analyzer.name == 'duplicates':
                    try:
                        temp_issues = list(analyzer.run(ctx))
                        duplicate_count = len(temp_issues)
                        # Add issues to collection
                        for issue in temp_issues:
                            if not ctx.should_ignore_issue(issue.kind):
                                issue_collection.add(issue)
                    except Exception as e:
                        logger.error(f"Failed to count duplicates: {e}")
                    break
            
            # Check bail-out conditions
            if file_count > semantic_file_limit:
                logger.warning(
                    f"Skipping semantic analysis: {file_count} files exceeds limit of {semantic_file_limit}. "
                    f"Use --force-semantic to override."
                )
                skip_semantic = True
            elif duplicate_count > semantic_duplicate_limit:
                logger.warning(
                    f"Skipping semantic analysis: {duplicate_count} duplicates exceeds limit of {semantic_duplicate_limit}. "
                    f"Use --force-semantic to override."
                )
                skip_semantic = True
        
        for analyzer in analyzers:
            # Skip semantic analyzers if bail-out triggered
            if skip_semantic and analyzer.name in ['semantic_hv', 'enhanced_semantic', 'semantic_duplicate']:
                logger.info(f"Skipping {analyzer.name} analyzer due to resource limits")
                continue
                
            # Skip duplicates analyzer if already run
            if analyzer.name == 'duplicates' and duplicate_count > 0:
                continue
                
            logger.info(f"Running {analyzer.name} analyzer")
            
            # Configure memory-aware settings for semantic analyzers
            if memory_monitor and analyzer.name in ['semantic_hv', 'enhanced_semantic', 'semantic_duplicate']:
                # Get optimal hypervector dimensions based on memory
                optimal_dims = memory_monitor.get_hypervector_dimensions()
                
                # Apply to analyzer if it supports dimension configuration
                if hasattr(analyzer, 'set_dimensions'):
                    analyzer.set_dimensions(optimal_dims)
                    output_manager.log(f"Set {analyzer.name} dimensions to {optimal_dims}", VerbosityLevel.VERBOSE)
                elif hasattr(analyzer, 'config') and hasattr(analyzer.config, 'dimensions'):
                    analyzer.config.dimensions = optimal_dims
                    output_manager.log(f"Set {analyzer.name} dimensions to {optimal_dims}", VerbosityLevel.VERBOSE)
            
            # Set up fallback for semantic analyzers
            fallback_func = None
            if analyzer.name in ['semantic_hv', 'enhanced_semantic', 'semantic_duplicate']:
                fallback_func = lambda *args, **kwargs: SemanticAnalysisFallback.tfidf_fallback(*args, **kwargs)
            
            # Wrap analyzer with watchdog
            wrapped_analyzer = watchdog.wrap_analyzer(
                analyzer.name,
                analyzer.run,
                fallback_func
            )
            
            # Execute analyzer with monitoring
            try:
                issues = wrapped_analyzer(ctx)
                for issue in issues:
                    if not ctx.should_ignore_issue(issue.kind):
                        issue_collection.add(issue)
                        
                # Check memory after each analyzer
                if memory_monitor:
                    stats = memory_monitor.get_stats()
                    output_manager.log(f"{analyzer.name} completed - Memory: {stats.current_mb:.1f}MB ({stats.usage_percent:.1f}%)", VerbosityLevel.VERBOSE)
                    
            except Exception as e:
                logger.error(f"Analyzer {analyzer.name} failed: {e}")
            
    # Deduplicate issues
    issue_collection.deduplicate()
    
    # Process issues with provenance tracking
    provenance_tracker = IssueProvenanceTracker(config.to_dict())
    enhanced_issues = provenance_tracker.process_issues(issue_collection.issues, ast_index)
    
    # Update issue collection with enhanced issues
    issue_collection.issues = enhanced_issues
    
    # Perform root cause clustering if requested
    if args.cluster_root_causes and issue_collection.issues:
        logger.info("Performing root cause clustering analysis")
        clusterer = RootCauseClusterer(
            similarity_threshold=args.cluster_threshold,
            min_cluster_size=2
        )
        clusters = clusterer.cluster(issue_collection.issues)
        
        if clusters:
            sys.stdout.write(f"\n{'=' * 60}\n")
            sys.stdout.write("ROOT CAUSE CLUSTERING ANALYSIS\n")
            sys.stdout.write(f"{'=' * 60}\n")
            sys.stdout.write(f"Found {len(clusters)} root cause clusters\n")
            sys.stdout.write(f"Clustered {sum(c.size for c in clusters)} of {len(issue_collection.issues)} issues\n\n")
            
            # Show top clusters
            sorted_clusters = sorted(clusters, key=lambda c: (c.severity, c.size), reverse=True)
            for cluster in sorted_clusters[:5]:
                sys.stdout.write(f"Cluster {cluster.cluster_id}: {cluster.size} issues (severity {cluster.severity})\n")
                sys.stdout.write(f"  Root Cause: {cluster.root_cause_guess}\n")
                sys.stdout.write(f"  Confidence: {cluster.confidence:.1%}\n")
                sys.stdout.write(f"  Fix Playbook: {cluster.fix_playbook_id}\n")
                sys.stdout.write(f"  Locations: {', '.join(f'{f}:{l}' for f, l in cluster.locations[:3])}")
                if len(cluster.locations) > 3:
                    sys.stdout.write(f" ... +{len(cluster.locations) - 3} more")
                sys.stdout.write("\n\n")
            
            if len(clusters) > 5:
                sys.stdout.write(f"... and {len(clusters) - 5} more clusters\n")
            
            # Generate detailed cluster report if output directory specified
            if args.output:
                cluster_report = clusterer.generate_report(clusters)
                cluster_path = args.output / "root_cause_clusters.txt"
                cluster_path.write_text(cluster_report)
                sys.stdout.write(f"\nDetailed cluster report saved to: {cluster_path}\n")
    
    # Generate canonical codemod if requested
    if args.generate_canonical_codemod:
        sys.stdout.write(f"\n{'=' * 60}\n")
        sys.stdout.write("CANONICAL POLICY CODEMOD GENERATION\n")
        sys.stdout.write(f"{'=' * 60}\n")
        
        # Find canonical policy analyzer in loaded analyzers
        canonical_analyzer = None
        for analyzer in analyzers:
            if hasattr(analyzer, 'name') and analyzer.name == 'canonical_policy':
                canonical_analyzer = analyzer
                break
        
        if canonical_analyzer:
            try:
                codemod_path = config.get("canonical_policy", {}).get("codemod_output", "./canonical_codemod.py")
                script = canonical_analyzer.generate_codemod_script(ast_index, codemod_path)
                
                if "No shadow implementations detected" not in script:
                    Path(codemod_path).write_text(script)
                    sys.stdout.write(f"Generated canonical codemod script: {codemod_path}\n")
                    sys.stdout.write("Review the script carefully before executing!\n")
                    sys.stdout.write(f"To apply: python {codemod_path}\n")
                else:
                    sys.stdout.write("No shadow implementations detected - no codemod needed\n")
                    
            except Exception as e:
                sys.stderr.write(f"Error generating codemod: {e}\n")
        else:
            if not config.get("canonical_policy", {}).get("canonical_roots"):
                sys.stdout.write("Canonical policy not configured. Add canonical_roots to config.\n")
            else:
                sys.stdout.write("Canonical policy analyzer not available\n")
    
    # Generate circular import fixes if requested
    if args.generate_circular_import_fixes:
        sys.stdout.write(f"\n{'=' * 60}\n")
        sys.stdout.write("CIRCULAR IMPORT FIX GENERATION\n")
        sys.stdout.write(f"{'=' * 60}\n")
        
        # Find circular import resolver in loaded analyzers
        circular_resolver = None
        for analyzer in analyzers:
            if hasattr(analyzer, 'name') and analyzer.name == 'circular_import_resolver':
                circular_resolver = analyzer
                break
        
        if circular_resolver:
            try:
                # Find circular import issues
                circular_issues = [issue for issue in issue_collection.issues 
                                 if issue.kind in ['circular_import_scc', 'circular_import_fix_orchestration']]
                
                if circular_issues:
                    fix_script_path = config.get("circular_import_resolver", {}).get("fix_script_output", "./circular_import_fixes.py")
                    script = circular_resolver.generate_fix_script(circular_issues, fix_script_path)
                    
                    Path(fix_script_path).write_text(script)
                    sys.stdout.write(f"Generated circular import fix script: {fix_script_path}\n")
                    sys.stdout.write("Review the script carefully before executing!\n")
                    sys.stdout.write(f"To apply: python {fix_script_path}\n")
                else:
                    sys.stdout.write("No circular import SCCs detected - no fixes needed\n")
                    
            except Exception as e:
                sys.stderr.write(f"Error generating circular import fixes: {e}\n")
        else:
            sys.stdout.write("Circular import resolver not available\n")
    
    # Show regressions if requested
    if args.show_regressions:
        sys.stdout.write(f"\n{'=' * 60}\n")
        sys.stdout.write("ISSUE REGRESSION REPORT\n")
        sys.stdout.write(f"{'=' * 60}\n")
        
        regression_report = provenance_tracker.get_regression_report(args.show_regressions)
        sys.stdout.write(regression_report + "\n")
    
    # Generate phantom triage report if requested
    if args.phantom_triage_report:
        sys.stdout.write(f"\n{'=' * 60}\n")
        sys.stdout.write("PHANTOM STUB TRIAGE REPORT\n")
        sys.stdout.write(f"{'=' * 60}\n")
        
        # Find phantom triage analyzer
        phantom_analyzer = None
        for analyzer in analyzers:
            if hasattr(analyzer, 'name') and analyzer.name == 'phantom_triage':
                phantom_analyzer = analyzer
                break
        
        if phantom_analyzer:
            try:
                triage_report = phantom_analyzer.generate_triage_report()
                sys.stdout.write(triage_report + "\n")
                
                # Check for blocked stubs that should fail CI
                stubs = phantom_analyzer.get_detected_stubs()
                blocked_stubs = [s for s in stubs if s.is_blocked]
                
                if blocked_stubs:
                    sys.stdout.write("\n🚨 CRITICAL: Found blocked security stubs that will fail CI:\n")
                    for stub in blocked_stubs:
                        func_name = f"{stub.class_name}.{stub.function_name}" if stub.class_name else stub.function_name
                        sys.stdout.write(f"  - {func_name} ({stub.file_path}:{stub.line_number})\n")
                    
                    if not args.fail_on:
                        sys.exit(2)  # Exit with error for blocked stubs
                        
            except Exception as e:
                sys.stderr.write(f"Error generating phantom triage report: {e}\n")
        else:
            sys.stdout.write("Phantom triage analyzer not available\n")
    
    # Generate reports
    reporter = Reporter(config.to_dict())
    scorer = RiskScorer(config.get("scoring_weights"))
    
    # Calculate scores
    module_scores, global_score = scorer.calculate_scores(issue_collection.issues)
    risk_level = scorer.get_risk_level(global_score, config.get("risk_thresholds", {}))
    
    # Print summary to console
    if not args.json or len(config.get("report.formats", [])) > 1:
        logger.info(f"Analysis complete: {len(issue_collection)} issues, risk score {global_score}")
        sys.stdout.write(f"\nTail-Chasing Analysis Complete\n")
        sys.stdout.write(f"{'=' * 40}\n")
        sys.stdout.write(f"Total Issues: {len(issue_collection)}\n")
        sys.stdout.write(f"Global Risk Score: {global_score} ({risk_level})\n")
        sys.stdout.write(f"Affected Modules: {len(module_scores)}\n")
        
        # Show top issues
        if issue_collection.issues:
            sys.stdout.write(f"\nTop Issues:\n")
            top_issues = sorted(issue_collection.issues, key=lambda i: i.severity, reverse=True)[:5]
            for issue in top_issues:
                location = f"{issue.file}:{issue.line}" if issue.file else "global"
                sys.stdout.write(f"  [{issue.kind}] {location} - {issue.message}\n")
                
                # Show suggestions if requested
                if args.show_suggestions and issue.suggestions:
                    sys.stdout.write("    Suggestions:\n")
                    for suggestion in issue.suggestions[:2]:
                        # Indent multi-line suggestions
                        for line in suggestion.split('\n'):
                            sys.stdout.write(f"      {line}\n")
                    if len(issue.suggestions) > 2:
                        sys.stdout.write(f"      ... and {len(issue.suggestions) - 2} more suggestions\n")
                
    # Generate full reports
    output_dir = Path(config.get("report.output_dir", "."))
    results = reporter.generate_reports(issue_collection.issues, output_dir)
    
    # Handle JSON output to stdout
    if args.json and "json" in results:
        sys.stdout.write(results["json"] + "\n")
        
    # Always show paths to generated reports in terminal (unless pure JSON mode)
    if not (args.json and len(config.get("report.formats", [])) == 1):
        sys.stdout.write("\nGenerated Reports:\n")
        sys.stdout.write("-" * 40 + "\n")
        
        # Show all generated report files
        report_files = []
        for fmt, content_or_path in results.items():
            if fmt.endswith("_file"):
                report_type = fmt.replace("_file", "").upper()
                report_files.append((report_type, content_or_path))
                logger.info(f"Generated {report_type} report: {content_or_path}")
                sys.stdout.write(f"{report_type} report: {content_or_path}\n")
        
        # If no files were saved, show inline report info
        if not report_files:
            if "text" in results:
                sys.stdout.write("Text report: (displayed above)\n")
            if "json" in results and not args.json:
                sys.stdout.write("JSON report: (use --json flag to output)\n")
            if "html" in results:
                sys.stdout.write("HTML report: (use --output to save to file)\n")
            
    # Always check if fix generation would be helpful (unless pure JSON mode)
    if not (args.json and len(config.get("report.formats", [])) == 1) and issue_collection.issues:
        # Calculate how many issues have fixable patterns
        fixable_types = {
            'semantic_duplicate_function', 'duplicate_function', 'phantom_function',
            'missing_symbol', 'circular_import', 'prototype_fragmentation',
            'hallucination_cascade', 'import_anxiety', 'wrapper_abstraction',
            'context_window_thrashing'
        }
        fixable_issues = [i for i in issue_collection.issues if i.kind in fixable_types]
        
        if fixable_issues:
            sys.stdout.write("\nFix Suggestions:\n")
            sys.stdout.write("-" * 40 + "\n")
            
            if args.generate_fixes:
                # Fix generation code (already exists below)
                pass
            else:
                logger.info(f"Found {len(fixable_issues)} fixable issues")
                sys.stdout.write(f"Found {len(fixable_issues)} fixable issues out of {len(issue_collection.issues)} total\n")
                sys.stdout.write("Run with --generate-fixes to create:\n")
                sys.stdout.write("  • Interactive fix script (tailchasing_fixes.py)\n")
                sys.stdout.write("  • Detailed suggestions file (tailchasing_suggestions.md)\n")
                sys.stdout.write("\nExample: tailchasing . --generate-fixes\n")
    
    # Generate fix script if requested
    if args.generate_fixes and issue_collection.issues:
        try:
            from .core.suggestions import InteractiveFixGenerator
            fix_generator = InteractiveFixGenerator()
            
            # Generate fix script
            fix_path = output_dir / "tailchasing_fixes.py"
            fix_script = fix_generator.generate_fix_script(issue_collection.issues, fix_path)
            fix_path.write_text(fix_script)
            fix_path.chmod(0o755)  # Make executable
            
            # Show in the standard location for generated files
            if not args.generate_fixes:
                # This case is handled above in "Fix Suggestions" section
                pass
            else:
                logger.info(f"Generated interactive fix script: {fix_path}")
                sys.stdout.write(f"Interactive fix script: {fix_path}\n")
                sys.stdout.write(f"  Run with: python {fix_path}\n")
            
            # Also generate a detailed suggestions file
            from .core.suggestions import FixSuggestionGenerator
            suggestion_gen = FixSuggestionGenerator()
            
            suggestions_path = output_dir / "tailchasing_suggestions.md"
            with open(suggestions_path, 'w') as f:
                f.write("# Tail-Chasing Fix Suggestions\n\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                # Group by issue type
                from collections import defaultdict
                by_type = defaultdict(list)
                for issue in issue_collection.issues:
                    by_type[issue.kind].append(issue)
                
                for issue_type, issues in sorted(by_type.items()):
                    f.write(f"\n## {issue_type.replace('_', ' ').title()} ({len(issues)} issues)\n\n")
                    
                    # Show first few examples
                    for issue in issues[:3]:
                        location = f"{issue.file}:{issue.line}" if issue.file else "global"
                        f.write(f"### {location}\n")
                        f.write(f"{issue.message}\n\n")
                        
                        # Get enhanced suggestions
                        enhanced_suggestions = suggestion_gen.generate_suggestions(issue)
                        for suggestion in enhanced_suggestions:
                            f.write(f"{suggestion}\n\n")
                        f.write("---\n\n")
                    
                    if len(issues) > 3:
                        f.write(f"*... and {len(issues) - 3} more {issue_type} issues*\n\n")
            
            logger.info(f"Generated detailed suggestions: {suggestions_path}")
            sys.stdout.write(f"Detailed suggestions: {suggestions_path}\n")
            
        except ImportError:
            logger.warning("Fix generation requires the enhanced suggestions module")
            sys.stdout.write("\nNote: Fix generation requires the enhanced suggestions module.\n")
        except Exception as e:
            logger.error(f"Failed to generate fix script: {e}")
            
    # Generate fix plan if requested
    if args.generate_fix_plan and issue_collection.issues:
        sys.stdout.write("\nGenerating Fix Plan:\n")
        sys.stdout.write("-" * 40 + "\n")
        
        try:
            from .core.fix_planner import FixPlanner, InteractiveFixReviewer
            
            # Create fix planner
            fix_planner = FixPlanner(
                root_dir=root_path,
                backup_dir=args.backup_dir,
                interactive=args.interactive_fixes,
                dry_run=args.dry_run_fixes
            )
            
            # Create fix plan
            fix_plan = fix_planner.create_fix_plan(issue_collection.issues)
            
            # Show plan summary
            sys.stdout.write(fix_plan.get_summary())
            sys.stdout.write("\n")
            
            # Interactive review if requested
            if args.interactive_fixes:
                reviewer = InteractiveFixReviewer(fix_plan)
                approved_actions, rejected_actions = reviewer.review()
                
                # Update plan with only approved actions
                fix_plan.actions = approved_actions
                sys.stdout.write(f"\nProceeding with {len(approved_actions)} approved actions\n")
            else:
                # Auto-approve based on confidence threshold
                approved_actions = [
                    action for action in fix_plan.actions
                    if action.confidence >= args.fix_confidence_threshold
                ]
                rejected_actions = [
                    action for action in fix_plan.actions
                    if action.confidence < args.fix_confidence_threshold
                ]
                
                if rejected_actions:
                    sys.stdout.write(f"\nAuto-rejected {len(rejected_actions)} low-confidence actions\n")
                    sys.stdout.write(f"(below threshold {args.fix_confidence_threshold:.1%})\n")
                
                fix_plan.actions = approved_actions
            
            # Generate fix script if we have actions
            if fix_plan.actions:
                # Generate executable script
                fix_script = fix_plan.get_executable_script()
                
                # Write to file
                args.fix_plan_output.write_text(fix_script)
                args.fix_plan_output.chmod(0o755)  # Make executable
                
                sys.stdout.write(f"\nGenerated fix script: {args.fix_plan_output}\n")
                
                if args.dry_run_fixes:
                    sys.stdout.write("DRY RUN MODE - Script will preview changes without applying\n")
                    sys.stdout.write(f"To preview: python {args.fix_plan_output}\n")
                else:
                    sys.stdout.write(f"To review: cat {args.fix_plan_output}\n")
                    sys.stdout.write(f"To apply fixes: python {args.fix_plan_output}\n")
                    sys.stdout.write(f"\n⚠️  WARNING: Review the script carefully before executing!\n")
                    sys.stdout.write(f"Backups will be saved to: {fix_plan.backup_dir}\n")
            else:
                sys.stdout.write("\nNo actions to perform after filtering\n")
                
        except ImportError:
            logger.warning("Fix planner module not available")
            sys.stdout.write("\nNote: Fix planner module not available.\n")
        except Exception as e:
            logger.error(f"Failed to generate fix plan: {e}")
            sys.stderr.write(f"\nError generating fix plan: {e}\n")
    
    # Generate missing symbol stubs if requested
    if args.generate_missing_stubs:
        sys.stdout.write("\nGenerating Missing Symbol Stubs:\n")
        sys.stdout.write("-" * 40 + "\n")
        
        # Find the enhanced missing symbols analyzer
        enhanced_missing_analyzer = None
        for analyzer in analyzers:
            if hasattr(analyzer, 'name') and analyzer.name == 'enhanced_missing_symbols':
                enhanced_missing_analyzer = analyzer
                break
        
        if enhanced_missing_analyzer and hasattr(enhanced_missing_analyzer, 'inferred_signatures'):
            if enhanced_missing_analyzer.inferred_signatures:
                # Generate stub file
                stub_content = enhanced_missing_analyzer.generate_stub_file(args.missing_stubs_output)
                args.missing_stubs_output.write_text(stub_content)
                
                sys.stdout.write(f"Generated {len(enhanced_missing_analyzer.inferred_signatures)} stub functions\n")
                sys.stdout.write(f"Output file: {args.missing_stubs_output}\n")
                sys.stdout.write("\nExample stubs generated:\n")
                
                # Show first few stubs as examples
                for i, (name, sig) in enumerate(list(enhanced_missing_analyzer.inferred_signatures.items())[:3]):
                    params = ", ".join(p.name for p in sig.parameters)
                    return_hint = f" -> {sig.return_type}" if sig.return_type else ""
                    sys.stdout.write(f"  - def {name}({params}){return_hint}\n")
                
                if len(enhanced_missing_analyzer.inferred_signatures) > 3:
                    sys.stdout.write(f"  ... and {len(enhanced_missing_analyzer.inferred_signatures) - 3} more\n")
            else:
                sys.stdout.write("No missing symbols detected that require stubs.\n")
        else:
            sys.stdout.write("Enhanced missing symbol analyzer not available.\n")
            sys.stdout.write("Enable it in configuration to generate stubs.\n")
    
    # Show cache statistics if requested
    if args.cache_stats and cache_enabled:
        stats = cache_manager.get_statistics()
        sys.stdout.write("\nCache Statistics:\n")
        sys.stdout.write("-" * 40 + "\n")
        sys.stdout.write(f"Cache hits: {stats['hits']}\n")
        sys.stdout.write(f"Cache misses: {stats['misses']}\n")
        sys.stdout.write(f"Hit rate: {stats['hit_rate']:.1%}\n")
        sys.stdout.write(f"Cache size: {stats['cache_size_mb']:.2f} MB\n")
        sys.stdout.write(f"Cached files: {stats['cached_files']}\n")
        sys.stdout.write(f"Parse time saved: {stats['bytes_saved_mb']:.2f} MB processed\n")
    
    # Show watchdog report if verbose (for non-batch mode)
    if args.watchdog_verbose and not use_batching:
        watchdog_report = watchdog.get_execution_report()
        sys.stdout.write("\nWatchdog Execution Report:\n")
        sys.stdout.write("-" * 40 + "\n")
        sys.stdout.write(f"Total executions: {watchdog_report['total_executions']}\n")
        sys.stdout.write(f"Total duration: {watchdog_report['total_duration']:.2f}s\n")
        sys.stdout.write(f"Timeouts: {watchdog_report['timeout_count']}\n")
        sys.stdout.write(f"Errors: {watchdog_report['error_count']}\n")
        
        if watchdog_report['slowest_analyzers']:
            sys.stdout.write("\nSlowest Analyzers:\n")
            for slow in watchdog_report['slowest_analyzers']:
                status = " (TIMEOUT)" if slow['timed_out'] else ""
                sys.stdout.write(f"  - {slow['analyzer']}: {slow['duration']:.2f}s{status}\n")
                if slow['file']:
                    sys.stdout.write(f"    File: {slow['file']}\n")
                if slow['function']:
                    sys.stdout.write(f"    Function: {slow['function']}\n")
        
        if watchdog_report['most_problematic']:
            sys.stdout.write("\nMost Problematic Analyzers:\n")
            for prob in watchdog_report['most_problematic']:
                sys.stdout.write(f"  - {prob['analyzer']}: {prob['problem_count']} issues\n")
    
    # Show batch processing report if used
    if use_batching and args.show_batch_progress:
        report = batch_processor.generate_report()
        sys.stdout.write("\n" + report + "\n")
    
    # Stop watchdog (for non-batch mode)
    if not use_batching:
        watchdog.stop()
    
    # Show performance profile if requested
    if profiler:
        profile_report = profiler.get_report()
        output_manager.show_performance_profile(profile_report)
        profiler.stop()
    
    # Show memory statistics if requested
    if memory_monitor:
        if verbosity in [VerbosityLevel.VERBOSE, VerbosityLevel.DEBUG]:
            output_manager.log("Memory Usage Summary:", VerbosityLevel.VERBOSE)
            memory_monitor.log_memory_summary()
        
        # Clean up memory monitor
        memory_monitor.cleanup()
    
    # Flush cache before exiting
    if cache_enabled:
        cache_manager.flush()
    
    # Check fail threshold
    if args.fail_on is not None:
        if global_score >= args.fail_on:
            logger.error(f"Risk score {global_score} exceeds threshold {args.fail_on}")
            sys.stderr.write(f"\nERROR: Risk score {global_score} exceeds threshold {args.fail_on}\n")
            sys.exit(2)
    else:
        # Use config thresholds
        fail_threshold = config.get("risk_thresholds.fail")
        if fail_threshold and global_score >= fail_threshold:
            logger.error(f"Risk score {global_score} exceeds configured fail threshold {fail_threshold}")
            sys.stderr.write(f"\nERROR: Risk score {global_score} exceeds configured fail threshold {fail_threshold}\n")
            sys.exit(2)
            
    sys.exit(0)


if __name__ == "__main__":
    main()
