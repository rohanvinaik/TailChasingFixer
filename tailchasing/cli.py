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
    
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging"
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
        help="Timeout for each analyzer in seconds (default: 120)"
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
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose)
    
    # Find project root
    root_path = Path(args.path).resolve()
    if not root_path.exists():
        logger.error(f"Path does not exist: {root_path}")
        sys.stderr.write(f"Error: Path does not exist: {root_path}\n")
        sys.exit(1)
    
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
    
    # Parse files
    logger.info("Parsing Python files")
    
    # Create robust parser if enabled
    robust_parser = None
    parse_results = {}
    if args.robust_parsing:
        from .core.robust_parser import RobustParser
        robust_parser = RobustParser(auto_fix_trivial=args.auto_fix_trivial_syntax)
        logger.info(f"Using robust parser (auto-fix: {args.auto_fix_trivial_syntax})")
    
    # Parse files with appropriate method
    if robust_parser:
        ast_index, parse_results = parse_files(files, robust_parser)
        
        # Show parsing statistics
        stats = robust_parser.get_statistics()
        if stats['quarantined'] > 0:
            logger.warning(f"Quarantined {stats['quarantined']} files due to parsing errors")
            sys.stderr.write(f"Warning: {stats['quarantined']} files could not be parsed and were quarantined\n")
            
            # Show details of quarantined files if verbose
            if args.verbose:
                for file_path, result in parse_results.items():
                    if result.is_quarantined:
                        sys.stderr.write(f"  - {file_path}: {', '.join(result.warnings[:2])}\n")
    else:
        ast_index, _ = parse_files(files)
    
    if not ast_index:
        logger.error("No valid Python files could be parsed")
        sys.stderr.write("No valid Python files could be parsed\n")
        sys.exit(1)
        
    logger.info(f"Successfully parsed {len(ast_index)} files")
    
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
        parse_results=parse_results  # Add parse results for quarantine checking
    )
    
    # Load and run analyzers
    logger.info("Running analyzers")
    analyzers = load_analyzers(config.to_dict())
    
    issue_collection = IssueCollection()
    
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
        try:
            import time
            import signal
            import threading
            
            # Set up timeout for analyzer
            timeout_seconds = config.get("resource_limits.analyzer_timeout_seconds", 120)
            start_time = time.time()
            
            # Function to run analyzer with timeout
            def run_with_timeout():
                try:
                    for issue in analyzer.run(ctx):
                        if not ctx.should_ignore_issue(issue.kind):
                            issue_collection.add(issue)
                except Exception as e:
                    logger.error(f"Analyzer {analyzer.name} failed: {e}")
            
            # Create and start thread
            thread = threading.Thread(target=run_with_timeout)
            thread.daemon = True
            thread.start()
            thread.join(timeout=timeout_seconds)
            
            if thread.is_alive():
                logger.warning(f"Analyzer {analyzer.name} timed out after {timeout_seconds}s")
                # Thread will continue running but we move on
            else:
                elapsed = time.time() - start_time
                logger.info(f"Analyzer {analyzer.name} completed in {elapsed:.2f}s")
                
                # Warn if approaching timeout
                if elapsed > timeout_seconds * 0.8:
                    logger.warning(f"Analyzer {analyzer.name} took {elapsed:.2f}s, approaching timeout of {timeout_seconds}s")
                    
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
